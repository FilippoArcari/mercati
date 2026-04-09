from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
import os
from typing import Optional, Callable, List, Tuple

from modelli.utils import get_device


# ─── Ornstein-Uhlenbeck Noise ─────────────────────────────────────────────────

class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2, dt=1e-2):
        self.mu         = mu * np.ones(action_dim)
        self.theta      = theta
        self.sigma      = sigma
        self.dt         = dt
        self.action_dim = action_dim
        self.reset()

    def reset(self):
        self.state = self.mu.copy()

    def sample(self):
        dx = (
            self.theta * (self.mu - self.state) * self.dt
            + self.sigma * np.sqrt(self.dt) * np.random.randn(self.action_dim)
        )
        self.state = self.state + dx
        return self.state.astype(np.float32)

    def decay(self, factor: float = 0.995, floor: float = 0.02):
        self.sigma = max(self.sigma * factor, floor)


# ─── Replay Buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity, device: torch.device = None):
        self.buffer: deque = deque(maxlen=capacity)
        self.device = device or torch.device('cpu')

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states),      dtype=torch.float32, device=self.device),
            torch.tensor(np.array(actions),     dtype=torch.float32, device=self.device),
            torch.tensor(np.array(rewards),     dtype=torch.float32, device=self.device).unsqueeze(1),
            torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device),
            torch.tensor(np.array(dones),       dtype=torch.float32, device=self.device).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)


# ─── Episodic Replay Buffer ────────────────────────────────────────────────────

class EpisodicReplayBuffer:
    """
    Buffer episodico con curriculum graduato.

    Memorizza i migliori K episodi per Sharpe e li usa per arricchire le
    batch di training (split configurabile, default 30% episodic / 70% uniform).

    [NUOVO] Curriculum graduato via `curriculum_ratio`:
      ratio = 0.0 → campiona solo dal top 25% degli episodi (fase iniziale,
                    l'agente impara dai comportamenti già vincenti)
      ratio = 0.5 → top 62.5% (fase intermedia)
      ratio = 1.0 → tutti gli episodi (comportamento standard, fase finale)

    Razionale: nelle prime epoche il segnale di gradiente è già distorto dai
    penalty del reward (vedi reward sistematicamente negativo a -100/-200 anche
    con portfolio in crescita). Ancorare il learning sui migliori episodi
    nelle fasi iniziali riduce la varianza e accelera la convergenza verso
    comportamenti stabili, introducendo gradualmente episodi difficili man mano
    che l'agente matura.

    Ispirato a Episodic Memory in RL (Lengyel & Dayan, 2007), con curriculum
    scheduling originale applicato al livello di episodio.
    """

    def __init__(self, max_episodes: int = 20, device: torch.device = None):
        self.max_episodes = max_episodes
        self.device       = device or torch.device("cpu")
        # Lista di (sharpe, transitions) ordinata per Sharpe decrescente
        self._episodes:   List[Tuple[float, list]] = []
        self._current_ep: list = []

    def start_episode(self) -> None:
        self._current_ep = []

    def push_step(
        self,
        state:      np.ndarray,
        action:     np.ndarray,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ) -> None:
        self._current_ep.append((state, action, reward, next_state, done))

    def end_episode(self, sharpe: float) -> None:
        """Salva l'episodio; mantiene solo i migliori max_episodes per Sharpe."""
        if not self._current_ep:
            return
        self._episodes.append((sharpe, list(self._current_ep)))
        self._episodes.sort(key=lambda x: x[0], reverse=True)
        if len(self._episodes) > self.max_episodes:
            self._episodes = self._episodes[: self.max_episodes]
        self._current_ep = []

    def sample(self, batch_size: int, curriculum_ratio: float = 1.0) -> Optional[tuple]:
        """
        Campiona transizioni dagli episodi in memoria.

        Args:
            batch_size:        numero di transizioni da campionare.
            curriculum_ratio:  float in [0, 1]. Controlla quanti episodi
                               (ordinati per Sharpe desc) sono eligibili.
                               0.0 → top 25%  |  1.0 → tutti (default).
        """
        n_eps = len(self._episodes)
        if n_eps == 0:
            return None

        # Calcola quanti episodi includere nella pool di campionamento
        # min_frac=0.25 garantisce che almeno il 25% degli episodi sia sempre eligibile
        min_frac  = 0.25
        n_include = max(1, int(n_eps * (min_frac + (1.0 - min_frac) * curriculum_ratio)))
        eligible  = self._episodes[:n_include]  # già ordinati Sharpe desc

        all_t = [t for _, ep in eligible for t in ep]
        if len(all_t) < batch_size:
            return None

        batch = random.sample(all_t, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states),      dtype=torch.float32, device=self.device),
            torch.tensor(np.array(actions),     dtype=torch.float32, device=self.device),
            torch.tensor(np.array(rewards),     dtype=torch.float32, device=self.device).unsqueeze(1),
            torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device),
            torch.tensor(np.array(dones),       dtype=torch.float32, device=self.device).unsqueeze(1),
        )

    def __len__(self) -> int:
        return sum(len(ep) for _, ep in self._episodes)

    @property
    def n_episodes(self) -> int:
        return len(self._episodes)

    @property
    def best_sharpe(self) -> float:
        return self._episodes[0][0] if self._episodes else -float("inf")


# ─── Actor ────────────────────────────────────────────────────────────────────

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=None):
        super().__init__()
        if hidden is None:
            hidden = [256, 256]

        layers = []
        in_dim = state_dim
        for h in hidden:
            layers += [
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                nn.LeakyReLU(0.2),
            ]
            in_dim = h

        layers += [nn.Linear(in_dim, action_dim), nn.Tanh()]
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.net[-2].weight, gain=0.01)

    def forward(self, state):
        return self.net(state)


# ─── Critic ───────────────────────────────────────────────────────────────────

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=None):
        super().__init__()
        if hidden is None:
            hidden = [256, 256]

        self.state_in = nn.Sequential(
            nn.Linear(state_dim, hidden[0]),
            nn.LayerNorm(hidden[0]),
            nn.LeakyReLU(0.2)
        )

        layers = []
        in_dim = hidden[0] + action_dim
        for h in hidden[1:]:
            layers += [
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                nn.LeakyReLU(0.2)
            ]
            in_dim = h

        self.mid_layers = nn.Sequential(*layers)
        self.out = nn.Linear(in_dim, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    def forward(self, state, action):
        s_h      = self.state_in(state)
        combined = torch.cat([s_h, action], dim=1)
        x        = self.mid_layers(combined)
        return self.out(x)


# ─── DDPG Agent ───────────────────────────────────────────────────────────────

class DDPGAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        actor_hidden=None,
        critic_hidden=None,
        lr_actor=1e-4,
        lr_critic=3e-4,
        gamma=0.99,
        tau=0.005,
        # ── [NUOVO] Tau dinamico termodinamico ──────────────────────────
        # tau_base è il valore di partenza; update_tau_from_thermo() lo sposta
        # tra tau_min e tau_max in base allo stress del mercato.
        # - stress > 0 (mercato surriscaldato) → tau → tau_min: le reti target
        #   si aggiornano lentamente, l'agente "congela" il comportamento.
        # - stress < 0 (mercato fresco)        → tau → tau_max: apprendimento
        #   aggressivo, l'agente esplora più velocemente.
        tau_min=0.001,
        tau_max=0.015,
        # ────────────────────────────────────────────────────────────────
        buffer_capacity=100_000,
        batch_size=64,
        update_every=1,
        noise_sigma=0.2,
        # ── [NUOVO] Episodic buffer: attivato se episodic_episodes > 0 ─
        episodic_episodes: int = 20,
        episodic_mix: float = 0.30,      # frazione della batch dall'episodic buf
        device: torch.device = None,
    ):
        self.device = device or get_device()

        self.actor_hidden  = actor_hidden  or [256, 256]
        self.critic_hidden = critic_hidden or [256, 256]

        self.state_dim    = state_dim
        self.action_dim   = action_dim
        self.gamma        = gamma
        self.tau          = tau
        self.tau_min      = tau_min
        self.tau_max      = tau_max
        self.tau_base     = tau        # usato per reset
        self.batch_size   = batch_size
        self.update_every = update_every
        self._step_count  = 0
        self.episodic_mix = episodic_mix

        self.actor         = Actor(state_dim,  action_dim, self.actor_hidden).to(self.device)
        self.actor_target  = Actor(state_dim,  action_dim, self.actor_hidden).to(self.device)
        self.critic        = Critic(state_dim, action_dim, self.critic_hidden).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, self.critic_hidden).to(self.device)

        self._hard_update(self.actor_target,  self.actor)
        self._hard_update(self.critic_target, self.critic)

        self.opt_actor  = torch.optim.Adam(self.actor.parameters(),  lr=lr_actor)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_critic,
                                           weight_decay=1e-4)

        self.buffer = ReplayBuffer(buffer_capacity, device=self.device)
        self.noise  = OUNoise(action_dim, sigma=noise_sigma)

        self.episodic_buffer: Optional[EpisodicReplayBuffer] = None
        if episodic_episodes > 0:
            self.episodic_buffer = EpisodicReplayBuffer(
                max_episodes=episodic_episodes,
                device=self.device,
            )

        # curriculum_ratio viene aggiornato dal training loop (0→1 nel corso
        # degli episodi) e passato a episodic_buffer.sample()
        self._curriculum_ratio: float = 1.0

    # ── [NUOVO] Tau dinamico termodinamico ────────────────────────────────────

    def update_tau_from_thermo(self, stress: float) -> float:
        """
        Modula tau in base allo stress termodinamico corrente.

        Usa una sigmoid invertita: stress alto → tau basso (congelamento),
        stress basso → tau alto (apprendimento aggressivo).

        Matematica:
          sig   = sigmoid(stress) ∈ (0, 1)
          tau   = tau_max − sig × (tau_max − tau_min)
          → stress → +∞  :  sig → 1  →  tau = tau_min  (freeze)
          → stress → −∞  :  sig → 0  →  tau = tau_max  (explore)
          → stress = 0   :  sig = 0.5 →  tau ≈ tau_base (neutro)

        Returns il nuovo valore di tau (utile per logging).
        """
        sig      = 1.0 / (1.0 + np.exp(-float(stress)))
        self.tau = self.tau_max - sig * (self.tau_max - self.tau_min)
        return self.tau

    def reset_tau(self) -> None:
        """Riporta tau al valore base (utile all'inizio di ogni fold)."""
        self.tau = self.tau_base

    # ── Act / Store ───────────────────────────────────────────────────────────

    def act(self, state, explore=True):
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(s).squeeze(0).cpu().numpy()
        self.actor.train()
        if explore:
            action = action + self.noise.sample()
        return np.clip(action, -1.0, 1.0)

    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)
        if self.episodic_buffer is not None:
            self.episodic_buffer.push_step(state, action, reward, next_state, done)

    # ── Update ────────────────────────────────────────────────────────────────

    def update(self):
        self._step_count += 1
        if (len(self.buffer) < self.batch_size or
                self._step_count % self.update_every != 0):
            return None

        # ── Campionamento misto con curriculum ────────────────────────────
        ep_batch_size = int(self.batch_size * self.episodic_mix)
        use_episodic  = (
            self.episodic_buffer is not None
            and len(self.episodic_buffer) >= ep_batch_size
            and self.episodic_buffer.n_episodes >= 2
            and ep_batch_size > 0
        )

        if use_episodic:
            main_size = self.batch_size - ep_batch_size
            states_u, actions_u, rewards_u, next_states_u, dones_u = \
                self.buffer.sample(main_size)

            # [NUOVO] Passa curriculum_ratio per lo staging episodico
            ep_batch = self.episodic_buffer.sample(
                ep_batch_size,
                curriculum_ratio=self._curriculum_ratio,
            )

            if ep_batch is not None:
                states_e, actions_e, rewards_e, next_states_e, dones_e = ep_batch
                states      = torch.cat([states_u,      states_e],      dim=0)
                actions     = torch.cat([actions_u,     actions_e],     dim=0)
                rewards     = torch.cat([rewards_u,     rewards_e],     dim=0)
                next_states = torch.cat([next_states_u, next_states_e], dim=0)
                dones       = torch.cat([dones_u,       dones_e],       dim=0)
            else:
                states, actions, rewards, next_states, dones = \
                    self.buffer.sample(self.batch_size)
        else:
            states, actions, rewards, next_states, dones = \
                self.buffer.sample(self.batch_size)

        # ── Critic update ─────────────────────────────────────────────────
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            q_target     = rewards + self.gamma * (1 - dones) * \
                           self.critic_target(next_states, next_actions)

        q_current   = self.critic(states, actions)
        critic_loss = F.smooth_l1_loss(q_current, q_target)

        self.opt_critic.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.opt_critic.step()

        # ── Actor update ──────────────────────────────────────────────────
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.opt_actor.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.opt_actor.step()

        # ── Soft update con tau corrente (potenzialmente dinamico) ────────
        self._soft_update(self.actor_target,  self.actor)
        self._soft_update(self.critic_target, self.critic)

        return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss.item()}

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _soft_update(self, target, source):
        for t_p, s_p in zip(target.parameters(), source.parameters()):
            t_p.data.copy_(self.tau * s_p.data + (1.0 - self.tau) * t_p.data)

    @staticmethod
    def _hard_update(target, source):
        target.load_state_dict(source.state_dict())

    def decay_noise(self, factor=0.995, floor=0.02):
        self.noise.decay(factor, floor)

    def reset_noise(self):
        self.noise.reset()

    def _state(self) -> dict:
        return {
            "state_dim":     self.state_dim,
            "action_dim":    self.action_dim,
            "actor_hidden":  self.actor_hidden,
            "critic_hidden": self.critic_hidden,
            "actor":         self.actor.state_dict(),
            "actor_target":  self.actor_target.state_dict(),
            "critic":        self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
        }

    def save(self, path: str, tag: str = "") -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self._state(), path)
        label = f" [{tag}]" if tag else ""
        print(f"[DDPG] Checkpoint salvato{label}: {path} "
              f"(arch: {self.actor_hidden})")

    def load(self, path: str) -> bool:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        saved_state_dim  = ckpt.get("state_dim")
        saved_action_dim = ckpt.get("action_dim")
        saved_actor_h    = ckpt.get("actor_hidden")

        if (saved_state_dim != self.state_dim or
            saved_action_dim != self.action_dim or
            saved_actor_h != self.actor_hidden):
            print(
                f"[DDPG] Checkpoint incompatibile (Architettura differente).\n"
                f"  Checkpoint : {saved_actor_h}\n"
                f"  Corrente   : {self.actor_hidden}\n"
                f"  L'agente riparte da zero."
            )
            return False

        self.actor.load_state_dict(ckpt["actor"])
        self.actor_target.load_state_dict(ckpt["actor_target"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        print(f"[DDPG] Checkpoint caricato correttamente: {path}")
        return True


# ─── Training loop ─────────────────────────────────────────────────────────────

def train_ddpg(
    agent:              DDPGAgent,
    env,
    n_episodes:         int   = 200,
    warmup:             int   = 1000,
    log_every:          int   = 10,
    noise_decay:        float = 0.995,
    noise_floor:        float = 0.02,
    es_patience:        int   = 20,
    es_metric:          str   = "sharpe",
    best_path:          Optional[str] = None,
    # ── [NUOVO] Tau dinamico ────────────────────────────────────────────
    # Callable(step: int) → float che ritorna Thm_Stress al passo `step`.
    # Se None, tau rimane fisso al valore base (comportamento originale).
    # Esempio di wiring in trade.py / obs_normalizer.py:
    #   thermo_stress_fn = (
    #       lambda step: env._thermo_at(step, "Thm_Stress")
    #       if env._has_thermo else None
    #   )
    thermo_stress_fn:   Optional[Callable[[int], float]] = None,
    # ── [NUOVO] Curriculum ─────────────────────────────────────────────
    # Se True, curriculum_ratio parte da 0 e raggiunge 1.0 a metà training.
    # L'agente impara prima dai migliori episodi, poi gradualmente da tutti.
    use_curriculum:     bool  = True,
    curriculum_warmup_eps: int = None,   # default = n_episodes // 2
) -> list[dict]:
    """
    Training loop DDPG con tre novità integrate:

    1. **Tau dinamico termodinamico**: se `thermo_stress_fn` è fornita, tau
       viene aggiornato ad ogni step in base allo stress del mercato.
       Stress alto → tau basso (target network si congela).
       Stress basso → tau alto (apprendimento aggressivo).

    2. **Episodic buffer con curriculum**: se `agent.episodic_buffer` è attivo,
       ogni episodio viene archiviato e la batch viene arricchita con le
       transizioni dei migliori episodi. Con `use_curriculum=True`, nelle prime
       `curriculum_warmup_eps` epoche si campiona solo dal top 25% degli episodi,
       aumentando gradualmente fino al 100%.

    3. **EpisodicReplayBuffer wiring**: start_episode/end_episode vengono chiamati
       automaticamente (nel vecchio codice questo non avveniva in train_ddpg).
    """
    history       = []
    total_step    = 0
    best_metric   = -float("inf")
    no_improve    = 0

    warmup_eps = curriculum_warmup_eps or (n_episodes // 2)

    for ep in range(1, n_episodes + 1):
        state = env.reset()
        agent.reset_noise()

        # [NUOVO] Curriculum ratio: sale linearmente da 0 a 1 in warmup_eps
        if use_curriculum and agent.episodic_buffer is not None:
            agent._curriculum_ratio = min(1.0, (ep - 1) / max(1, warmup_eps))
        else:
            agent._curriculum_ratio = 1.0

        # [NUOVO] Avvia tracciamento episodio per episodic buffer
        if agent.episodic_buffer is not None:
            agent.episodic_buffer.start_episode()

        ep_reward     = 0.0
        critic_losses = []
        actor_losses  = []
        done          = False

        while not done:
            if total_step < warmup:
                action = np.random.uniform(-1, 1, env.action_dim).astype(np.float32)
            else:
                action = agent.act(state, explore=True)

            next_state, reward, done, info = env.step(action)
            agent.store(state, action, reward, next_state, done)

            # [NUOVO] Aggiorna tau in base allo stress termodinamico corrente
            # env._step è già avanzato di 1 dopo step(), leggiamo step-1
            if thermo_stress_fn is not None:
                current_step = info.get("step", total_step) - 1
                stress = thermo_stress_fn(max(0, current_step))
                agent.update_tau_from_thermo(stress)

            losses = agent.update()
            if losses:
                critic_losses.append(losses["critic_loss"])
                actor_losses.append(losses["actor_loss"])

            state       = next_state
            ep_reward  += reward
            total_step += 1

        agent.decay_noise(factor=noise_decay, floor=noise_floor)

        sharpe    = env.sharpe_ratio()
        portfolio = env.portfolio_history()[-1]
        max_dd    = env.max_drawdown()

        # [NUOVO] Chiude l'episodio nell'episodic buffer
        if agent.episodic_buffer is not None:
            agent.episodic_buffer.end_episode(sharpe=sharpe)

        ep_info = {
            "episode":            ep,
            "total_reward":       ep_reward,
            "portfolio_value":    portfolio,
            "sharpe":             sharpe,
            "max_drawdown":       max_dd,
            "noise_sigma":        agent.noise.sigma,
            "tau":                agent.tau,
            "curriculum_ratio":   agent._curriculum_ratio,
            "critic_loss":        float(np.mean(critic_losses)) if critic_losses else None,
            "actor_loss":         float(np.mean(actor_losses))  if actor_losses  else None,
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
            cl    = f"{ep_info['critic_loss']:.5f}" if ep_info["critic_loss"] else "n/a"
            al    = f"{ep_info['actor_loss']:.5f}"  if ep_info["actor_loss"]  else "n/a"
            ep_buf_str = (
                f" | EpBuf: {agent.episodic_buffer.n_episodes}/{agent.episodic_buffer.max_episodes}"
                f" best={agent.episodic_buffer.best_sharpe:.2f}"
                f" curr={agent._curriculum_ratio:.2f}"
                if agent.episodic_buffer is not None else ""
            )
            best_tag = " ★" if no_improve == 0 else f" (no improve: {no_improve}/{es_patience})"
            print(
                f"Ep {ep:4d}/{n_episodes} | "
                f"Reward: {ep_reward:+.4f} | "
                f"Portfolio: ${portfolio:,.0f} | "
                f"Sharpe: {sharpe:.3f} | "
                f"MaxDD: {max_dd:.3f} | "
                f"σ: {agent.noise.sigma:.3f} | "
                f"τ: {agent.tau:.4f} | "
                f"C: {cl} | A: {al}"
                f"{ep_buf_str}"
                f"{best_tag}"
            )

        if es_patience > 0 and no_improve >= es_patience:
            print(
                f"\n[DDPG] Early stopping a ep {ep} — "
                f"{es_metric} non migliora da {es_patience} episodi "
                f"(best: {best_metric:.4f})"
            )
            break

    return history


# ─── ThermoEnsemble ───────────────────────────────────────────────────────────

class ThermoEnsemble:
    """
    Ensemble di policy DDPG pesate in tempo reale dalla similarità
    termodinamica tra il regime corrente e il regime di training di ogni fold.

    Idea: ogni fold ha imparato a operare in un regime termodinamico specifico.
    Invece di scegliere "il miglior fold" come warm start (scelta statica e
    fragile), pesiamo le azioni di tutti i fold in base a quanto il mercato
    attuale assomiglia al loro regime di training.

    Algoritmo:
      1. Per ogni fold k, calcola il profilo termodinamico medio durante
         il suo training: μ_k = mean(Thm_features) su train_k.
      2. A ogni step live, misura la distanza euclidea tra il vettore
         termodinamico corrente x e ogni μ_k.
      3. Converti le distanze in pesi tramite softmax negativa:
         w_k = softmax(−distance(x, μ_k) / temperature)
      4. L'azione finale è la media pesata delle azioni dei K agenti.

    Questo è un meta-learning implicito: il sistema seleziona dinamicamente
    quale "memoria di mercato" usare, senza alcun retraining.

    Args:
        agents:          lista di DDPGAgent (uno per fold, già caricati).
        fold_profiles:   lista di array 1D, profilo termodinamico medio di
                         ogni fold (stessa lunghezza di Thm_features).
        temperature:     scala della softmax. Valori bassi = winner-take-all,
                         alti = media uniforme. Default 1.0.
        feature_names:   nomi delle feature termodinamiche usate per il profilo
                         (es. ["Thm_Pressure","Thm_Temperature","Thm_Work",
                               "Thm_Stress","Thm_Efficiency","Thm_Entropy"]).
                         Utile per estrarre il sottoinsieme corretto dallo stato.

    Utilizzo tipico (in alpaca_live.py o trade.py):
        profiles = [compute_thermo_profile(thermo_df, train_slice_k) for k in folds]
        ensemble = ThermoEnsemble(agents=fold_agents, fold_profiles=profiles)
        action   = ensemble.act(state, current_thermo_vector)
    """

    def __init__(
        self,
        agents:        List[DDPGAgent],
        fold_profiles: List[np.ndarray],
        temperature:   float = 1.0,
        feature_names: Optional[List[str]] = None,
    ):
        if len(agents) != len(fold_profiles):
            raise ValueError(
                f"ThermoEnsemble: numero agenti ({len(agents)}) != "
                f"numero profili ({len(fold_profiles)})"
            )
        self.agents        = agents
        self.fold_profiles = [np.asarray(p, dtype=np.float32) for p in fold_profiles]
        self.temperature   = temperature
        self.feature_names = feature_names
        self._last_weights: Optional[np.ndarray] = None  # per logging/debug

    def _compute_weights(self, current_thermo: np.ndarray) -> np.ndarray:
        """
        Calcola pesi softmax basati sulla similarità con ogni profilo di fold.
        Distanza euclidea negativa → similarità → softmax.
        """
        x = np.asarray(current_thermo, dtype=np.float32)
        distances = np.array([
            np.linalg.norm(x - p) for p in self.fold_profiles
        ], dtype=np.float32)
        # Softmax su distanze negate: fold più vicino → peso più alto
        logits = -distances / (self.temperature + 1e-8)
        logits = logits - logits.max()     # stabilità numerica
        exp    = np.exp(logits)
        return exp / (exp.sum() + 1e-8)

    def act(
        self,
        state:          np.ndarray,
        current_thermo: np.ndarray,
        explore:        bool = False,
    ) -> np.ndarray:
        """
        Azione ensemble pesata per similarità termodinamica.

        Args:
            state:          osservazione corrente (già normalizzata).
            current_thermo: vettore delle feature termodinamiche al passo
                            corrente (Thm_Pressure, Thm_Temperature, ...).
            explore:        se True, aggiunge rumore OU a ogni agente.

        Returns:
            np.ndarray di shape (action_dim,), clippato in [-1, 1].
        """
        weights = self._compute_weights(current_thermo)
        self._last_weights = weights

        actions = np.stack([
            agent.act(state, explore=explore) for agent in self.agents
        ])  # shape: (n_agents, action_dim)

        # Media pesata: w · A
        blended = (weights[:, None] * actions).sum(axis=0)
        return np.clip(blended, -1.0, 1.0)

    def act_top1(
        self,
        state:          np.ndarray,
        current_thermo: np.ndarray,
        explore:        bool = False,
    ) -> np.ndarray:
        """
        Alternativa winner-take-all: usa solo l'agente del fold più simile.
        Utile per confrontare con il metodo blend.
        """
        weights = self._compute_weights(current_thermo)
        best_k  = int(np.argmax(weights))
        return self.agents[best_k].act(state, explore=explore)

    @property
    def last_weights(self) -> Optional[np.ndarray]:
        """Pesi dell'ultima chiamata ad act() — utile per logging."""
        return self._last_weights

    def weights_summary(self) -> str:
        """Stringa human-readable dei pesi correnti."""
        if self._last_weights is None:
            return "n/a"
        parts = [f"fold{i+1}={w:.3f}" for i, w in enumerate(self._last_weights)]
        return " | ".join(parts)


# ─── Utility: calcola il profilo termodinamico di un fold ─────────────────────

def compute_thermo_profile(
    thermo_df,
    start_idx: int,
    end_idx:   int,
    feature_names: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Calcola il vettore medio delle feature termodinamiche su una finestra
    di training, da usare come profilo fold per ThermoEnsemble.

    Args:
        thermo_df:     DataFrame con colonne Thm_* (output di ThermoStateBuilder).
        start_idx:     indice di inizio della finestra (incluso).
        end_idx:       indice di fine della finestra (escluso).
        feature_names: colonne da usare. Se None, usa tutte le colonne Thm_*.

    Returns:
        np.ndarray 1D, media delle feature sulla finestra.

    Esempio in walk_forward.py:
        profile_k = compute_thermo_profile(thermo_df, train_start, train_end)
        ensemble  = ThermoEnsemble(agents, [profile_0, profile_1, ...])
    """
    import pandas as pd

    if feature_names is None:
        feature_names = [c for c in thermo_df.columns if c.startswith("Thm_")]

    available = [f for f in feature_names if f in thermo_df.columns]
    if not available:
        raise ValueError(
            f"compute_thermo_profile: nessuna colonna Thm_* trovata in thermo_df. "
            f"Colonne disponibili: {list(thermo_df.columns)}"
        )

    window = thermo_df.iloc[start_idx:end_idx][available]
    return window.mean(axis=0).values.astype(np.float32)