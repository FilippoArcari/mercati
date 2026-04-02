"""
modelli/dqn.py — Agente DQN (Dueling + Double + Prioritized Replay)

Action space "factored":
  - Ogni ticker ha N livelli d'azione indipendenti (es. N=5 → [-1, -0.5, 0, +0.5, +1])
  - La rete ha N_tickers teste di output da N valori ciascuna (multi-head)
  - Si evita l'esplosione combinatoria (N^K invece di N*K)

Compatibilità con TradingEnv:
  - DQNAgent.act()    → restituisce array continuo [-1,1] pronto per env.step()
  - DQNAgent.act_idx() → restituisce gli indici grezzi (per il replay buffer)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import os
from typing import Optional

from modelli.utils import get_device


# ─── Prioritized Replay Buffer ─────────────────────────────────────────────────

Transition = namedtuple("Transition", ["state", "action_idx", "reward", "next_state", "done"])


class PrioritizedReplayBuffer:
    """
    Replay buffer con campionamento proporzionale alle TD-error.

    Vantaggi su dati finanziari intraday:
      - Rivede più spesso le transizioni con errore elevato
      - Migliora l'efficienza campionaria su serie temporali corte (1m = pochi episodi)
    """

    def __init__(self, capacity: int, alpha: float = 0.6, beta_start: float = 0.4, beta_end: float = 1.0, device: torch.device = None):
        self.capacity   = capacity
        self.alpha      = alpha          # quanto conta la priorità (0 = uniforme, 1 = piena priorità)
        self.beta       = beta_start     # importance-sampling correction (annealing → 1)
        self.beta_end   = beta_end
        self.beta_step  = 0
        self.device     = device or torch.device('cpu')

        self.buffer:     list[Transition]  = []
        self.priorities: list[float]       = []
        self.pos         = 0

    def push(self, *args):
        max_prio = max(self.priorities, default=1.0)
        t = Transition(*args)
        if len(self.buffer) < self.capacity:
            self.buffer.append(t)
            self.priorities.append(max_prio)
        else:
            self.buffer[self.pos]     = t
            self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, total_steps: int = 10_000):
        # Annealing del beta verso 1 nel corso del training
        self.beta = min(self.beta_end, self.beta + (self.beta_end - self.beta) / max(total_steps, 1))

        prios = np.array(self.priorities, dtype=np.float32)
        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, replace=False, p=probs)
        samples = [self.buffer[i] for i in indices]

        # Importance-sampling weights
        N  = len(self.buffer)
        ws = (N * probs[indices]) ** (-self.beta)
        ws = ws / ws.max()  # normalizza per stabilità numerica

        states      = torch.tensor(np.array([s.state      for s in samples]), dtype=torch.float32, device=self.device)
        action_idxs = torch.tensor(np.array([s.action_idx for s in samples]), dtype=torch.long, device=self.device)
        rewards     = torch.tensor(np.array([s.reward     for s in samples]), dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(np.array([s.next_state for s in samples]), dtype=torch.float32, device=self.device)
        dones       = torch.tensor(np.array([s.done       for s in samples]), dtype=torch.float32, device=self.device).unsqueeze(1)
        weights     = torch.tensor(ws, dtype=torch.float32, device=self.device).unsqueeze(1)

        return states, action_idxs, rewards, next_states, dones, weights, indices

    def update_priorities(self, indices, td_errors: np.ndarray):
        for idx, err in zip(indices, td_errors):
            self.priorities[idx] = float(abs(err)) + 1e-6   # epsilon per evitare prio zero

    def __len__(self):
        return len(self.buffer)


# ─── Dueling DQN (multi-head per ticker) ─────────────────────────────────────

class DuelingDQN(nn.Module):
    """
    Architettura Dueling DQN con output multi-head.

    Ogni ticker ha la propria coppia (Value stream, Advantage stream):
      Q(s, a_i) = V(s) + A(s, a_i) - mean(A(s, *))

    Body condiviso → N_tickers heads indipendenti.
    """

    def __init__(self, state_dim: int, n_tickers: int, n_actions: int, hidden: list[int]):
        super().__init__()
        self.n_tickers = n_tickers
        self.n_actions = n_actions

        # Backbone condiviso
        layers = []
        in_dim = state_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.LeakyReLU(0.1)]
            in_dim = h
        self.backbone = nn.Sequential(*layers)

        # Per ogni ticker: stream Value (1) + stream Advantage (n_actions)
        self.value_heads     = nn.ModuleList([nn.Linear(in_dim, 1)           for _ in range(n_tickers)])
        self.advantage_heads = nn.ModuleList([nn.Linear(in_dim, n_actions)   for _ in range(n_tickers)])

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Returns: Q-values shape (B, n_tickers, n_actions)
        """
        h = self.backbone(state)
        qs = []
        for i in range(self.n_tickers):
            v = self.value_heads[i](h)                        # (B, 1)
            a = self.advantage_heads[i](h)                    # (B, n_actions)
            q = v + a - a.mean(dim=1, keepdim=True)           # Dueling combination
            qs.append(q.unsqueeze(1))                         # (B, 1, n_actions)
        return torch.cat(qs, dim=1)                           # (B, n_tickers, n_actions)


# ─── DQN Agent ────────────────────────────────────────────────────────────────

class DQNAgent:
    """
    Double DQN con Dueling architecture e Prioritized Replay Buffer.

    Parametri chiave
    ----------------
    n_actions        : numero di livelli per ticker (es. 5 → [-1, -0.5, 0, +0.5, +1])
    epsilon_start    : esplorazione iniziale
    epsilon_end      : esplorazione finale (floor)
    epsilon_decay    : fattore moltiplicativo per episodio
    target_update_every: ogni quanti step aggiornare il target network (soft)
    """

    def __init__(
        self,
        state_dim:            int,
        n_tickers:            int,
        n_actions:            int   = 5,
        hidden:               list  = None,
        lr:                   float = 3e-4,
        gamma:                float = 0.99,
        tau:                  float = 0.005,
        buffer_capacity:      int   = 50_000,
        batch_size:           int   = 64,
        update_every:         int   = 4,
        target_update_every:  int   = 100,
        epsilon_start:        float = 1.0,
        epsilon_end:          float = 0.05,
        epsilon_decay:        float = 0.995,
        device:               torch.device = None,
    ):
        self.device          = device or get_device()
        self.state_dim           = state_dim
        self.n_tickers           = n_tickers
        self.n_actions           = n_actions
        self.hidden              = hidden or [512, 256, 128]
        self.gamma               = gamma
        self.tau                 = tau
        self.batch_size          = batch_size
        self.update_every        = update_every
        self.target_update_every = target_update_every
        self.epsilon             = epsilon_start
        self.epsilon_end         = epsilon_end
        self.epsilon_decay       = epsilon_decay
        self._step_count         = 0

        # Livelli azione: mappa indice → segnale continuo
        self.action_levels = np.linspace(-1.0, 1.0, n_actions, dtype=np.float32)

        self.online = DuelingDQN(state_dim, n_tickers, n_actions, self.hidden).to(self.device)
        self.target = DuelingDQN(state_dim, n_tickers, n_actions, self.hidden).to(self.device)
        self._hard_update()

        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.buffer    = PrioritizedReplayBuffer(buffer_capacity, device=self.device)

    # ── action ────────────────────────────────────────────────────────────────

    def act_idx(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        """Restituisce indici (one per ticker) con epsilon-greedy."""
        if explore and random.random() < self.epsilon:
            return np.random.randint(0, self.n_actions, size=self.n_tickers)

        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.online.eval()
        with torch.no_grad():
            q = self.online(s).squeeze(0)          # (n_tickers, n_actions)
        self.online.train()
        return q.argmax(dim=1).cpu().numpy()             # (n_tickers,)

    def act(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        """Restituisce segnali continui [-1,1] pronti per env.step()."""
        idxs = self.act_idx(state, explore)
        return self.action_levels[idxs]

    # ── store & update ────────────────────────────────────────────────────────

    def store(self, state, action_idx, reward, next_state, done):
        self.buffer.push(state, action_idx, reward, next_state, done)

    def update(self) -> Optional[dict]:
        self._step_count += 1
        if (len(self.buffer) < self.batch_size or
                self._step_count % self.update_every != 0):
            return None

        states, action_idxs, rewards, next_states, dones, weights, indices = \
            self.buffer.sample(self.batch_size, total_steps=self._step_count)

        # ── Double DQN target ──────────────────────────────────────────────────
        with torch.no_grad():
            # Online seleziona l'azione migliore
            next_q_online = self.online(next_states)               # (B, K, N)
            best_actions  = next_q_online.argmax(dim=2)            # (B, K)
            # Target valuta
            next_q_target = self.target(next_states)               # (B, K, N)
            # gather per azione selezionata
            best_next_q = next_q_target.gather(
                2, best_actions.unsqueeze(2)
            ).squeeze(2)                                           # (B, K)
            # Aggrega tra ticker con media (portafoglio)
            best_next_q_mean = best_next_q.mean(dim=1, keepdim=True)  # (B, 1)
            td_target = rewards + self.gamma * (1 - dones) * best_next_q_mean

        # ── current Q ─────────────────────────────────────────────────────────
        all_q = self.online(states)                                # (B, K, N)
        current_q = all_q.gather(
            2, action_idxs.unsqueeze(2)
        ).squeeze(2).mean(dim=1, keepdim=True)                    # (B, 1)

        td_errors = (current_q - td_target).detach().squeeze(1).numpy()

        loss = (weights * F.smooth_l1_loss(current_q, td_target, reduction="none")).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 1.0)
        self.optimizer.step()

        # Aggiorna priorità nel buffer
        self.buffer.update_priorities(indices, td_errors)

        # Soft update target
        if self._step_count % self.target_update_every == 0:
            self._soft_update()

        return {"loss": loss.item(), "epsilon": self.epsilon}

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    # ── target network ────────────────────────────────────────────────────────

    def _soft_update(self):
        for t_p, s_p in zip(self.target.parameters(), self.online.parameters()):
            t_p.data.copy_(self.tau * s_p.data + (1.0 - self.tau) * t_p.data)

    def _hard_update(self):
        self.target.load_state_dict(self.online.state_dict())

    # ── checkpoint ────────────────────────────────────────────────────────────

    def _state(self) -> dict:
        return {
            "state_dim":           self.state_dim,
            "n_tickers":           self.n_tickers,
            "n_actions":           self.n_actions,
            "hidden":              self.hidden,
            "online":              self.online.state_dict(),
            "target":              self.target.state_dict(),
            "epsilon":             self.epsilon,
        }

    def save(self, path: str, tag: str = "") -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self._state(), path)
        label = f" [{tag}]" if tag else ""
        print(f"[DQN] Checkpoint salvato{label}: {path} (ε={self.epsilon:.3f})")

    def load(self, path: str) -> bool:
        ckpt = torch.load(path, weights_only=False)

        if (ckpt.get("state_dim")  != self.state_dim  or
                ckpt.get("n_tickers") != self.n_tickers or
                ckpt.get("n_actions") != self.n_actions or
                ckpt.get("hidden")    != self.hidden):
            print(
                f"[DQN] Checkpoint incompatibile — architettura differente.\n"
                f"  L'agente riparte da zero."
            )
            return False

        self.online.load_state_dict(ckpt["online"])
        self.target.load_state_dict(ckpt["target"])
        self.epsilon = ckpt.get("epsilon", self.epsilon)
        print(f"[DQN] Checkpoint caricato: {path} (ε={self.epsilon:.3f})")
        return True


# ─── Training loop ─────────────────────────────────────────────────────────────

def train_dqn(
    agent:        DQNAgent,
    env,
    n_episodes:   int   = 200,
    warmup:       int   = 500,
    log_every:    int   = 10,
    es_patience:  int   = 30,
    es_metric:    str   = "sharpe",
    best_path:    Optional[str] = None,
) -> list[dict]:
    history      = []
    total_step   = 0
    best_metric  = -float("inf")
    no_improve   = 0

    for ep in range(1, n_episodes + 1):
        state = env.reset()
        ep_reward  = 0.0
        losses     = []
        done       = False

        while not done:
            if total_step < warmup:
                action_idx = np.random.randint(0, agent.n_actions, size=agent.n_tickers)
                action     = agent.action_levels[action_idx]
            else:
                action_idx = agent.act_idx(state, explore=True)
                action     = agent.action_levels[action_idx]

            next_state, reward, done, _ = env.step(action)
            agent.store(state, action_idx, reward, next_state, done)

            info = agent.update()
            if info:
                losses.append(info["loss"])

            state      = next_state
            ep_reward += reward
            total_step += 1

        agent.decay_epsilon()

        sharpe    = env.sharpe_ratio()
        portfolio = env.portfolio_history()[-1]
        max_dd    = env.max_drawdown()

        ep_info = {
            "episode":         ep,
            "total_reward":    ep_reward,
            "portfolio_value": portfolio,
            "sharpe":          sharpe,
            "max_drawdown":    max_dd,
            "noise_sigma":     agent.epsilon,   # campo usato dai plot esistenti
            "loss":            float(np.mean(losses)) if losses else None,
        }
        history.append(ep_info)

        metric_map = {
            "sharpe":    sharpe,
            "portfolio": portfolio,
            "reward":    ep_reward,
        }
        current_metric = metric_map.get(es_metric, sharpe)

        if current_metric > best_metric:
            best_metric = current_metric
            no_improve  = 0
            if best_path:
                agent.save(best_path, tag=f"BEST ep{ep} {es_metric}={best_metric:.4f}")
        else:
            no_improve += 1

        if ep % log_every == 0:
            ls   = f"{ep_info['loss']:.5f}" if ep_info["loss"] else "n/a"
            best_tag = " ★" if no_improve == 0 else f" (no improve: {no_improve}/{es_patience})"
            print(
                f"Ep {ep:4d}/{n_episodes} | "
                f"Reward: {ep_reward:+.4f} | "
                f"Portfolio: ${portfolio:,.0f} | "
                f"Sharpe: {sharpe:.3f} | "
                f"MaxDD: {max_dd:.3f} | "
                f"ε: {agent.epsilon:.3f} | "
                f"Loss: {ls}"
                f"{best_tag}"
            )

        if es_patience > 0 and no_improve >= es_patience:
            print(
                f"\n[DQN] Early stopping a ep {ep} — "
                f"{es_metric} non migliora da {es_patience} episodi "
                f"(best: {best_metric:.4f})"
            )
            break

    return history
