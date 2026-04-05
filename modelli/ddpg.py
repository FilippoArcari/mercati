from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
import os
from typing import Optional

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
        # floor default allineato al config (noise_floor: 0.02)
        # era 0.05 → causava rumore residuo maggiore del necessario a fine training
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


# ─── Actor (Dinamico) ─────────────────────────────────────────────────────────

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

        # Output layer: Tanh per mappare azioni in [-1, 1]
        layers += [nn.Linear(in_dim, action_dim), nn.Tanh()]

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
        # Gain basso sull'ultimo layer per favorire l'esplorazione iniziale
        nn.init.orthogonal_(self.net[-2].weight, gain=0.01)

    def forward(self, state):
        return self.net(state)


# ─── Critic (Dinamico) ────────────────────────────────────────────────────────

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=None):
        super().__init__()
        if hidden is None:
            hidden = [256, 256]

        # Primo layer processa solo lo stato (architettura DDPG standard)
        self.state_in = nn.Sequential(
            nn.Linear(state_dim, hidden[0]),
            nn.LayerNorm(hidden[0]),
            nn.LeakyReLU(0.2)
        )

        # Layer intermedi processano (output_state_in + action)
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
        s_h = self.state_in(state)
        combined = torch.cat([s_h, action], dim=1)
        x = self.mid_layers(combined)
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
        buffer_capacity=100_000,
        batch_size=64,
        update_every=1,
        noise_sigma=0.2,
        device: torch.device = None,
    ):
        self.device = device or get_device()

        self.actor_hidden  = actor_hidden  or [256, 256]
        self.critic_hidden = critic_hidden or [256, 256]

        self.state_dim    = state_dim
        self.action_dim   = action_dim
        self.gamma        = gamma
        self.tau          = tau
        self.batch_size   = batch_size
        self.update_every = update_every
        self._step_count  = 0

        self.actor         = Actor(state_dim,  action_dim, self.actor_hidden).to(self.device)
        self.actor_target  = Actor(state_dim,  action_dim, self.actor_hidden).to(self.device)
        self.critic        = Critic(state_dim, action_dim, self.critic_hidden).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, self.critic_hidden).to(self.device)

        self._hard_update(self.actor_target,  self.actor)
        self._hard_update(self.critic_target, self.critic)

        # Nota: DataParallel NON viene usato per DDPG.
        # Il DDPG è CPU-bound (env.step sequenziale), non GPU-bound:
        # il costo di sincronizzazione tra GPU supera il beneficio.

        self.opt_actor  = torch.optim.Adam(self.actor.parameters(),  lr=lr_actor)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_critic,
                                           weight_decay=1e-4)

        self.buffer = ReplayBuffer(buffer_capacity, device=self.device)
        self.noise  = OUNoise(action_dim, sigma=noise_sigma)

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

    def update(self):
        self._step_count += 1
        if (len(self.buffer) < self.batch_size or
                self._step_count % self.update_every != 0):
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            q_target     = rewards + self.gamma * (1 - dones) * self.critic_target(next_states, next_actions)

        q_current   = self.critic(states, actions)
        critic_loss = F.smooth_l1_loss(q_current, q_target)

        self.opt_critic.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.opt_critic.step()

        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.opt_actor.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.opt_actor.step()

        self._soft_update(self.actor_target,  self.actor)
        self._soft_update(self.critic_target, self.critic)

        return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss.item()}

    def _soft_update(self, target, source):
        for t_p, s_p in zip(target.parameters(), source.parameters()):
            t_p.data.copy_(self.tau * s_p.data + (1.0 - self.tau) * t_p.data)

    @staticmethod
    def _hard_update(target, source):
        target.load_state_dict(source.state_dict())

    def decay_noise(self, factor=0.995, floor=0.02):
        # floor default allineato al config (noise_floor: 0.02)
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
        ckpt = torch.load(path, weights_only=False)

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
    agent:        DDPGAgent,
    env,
    n_episodes:   int   = 200,
    warmup:       int   = 1000,
    log_every:    int   = 10,
    noise_decay:  float = 0.995,
    noise_floor:  float = 0.02,   # allineato al config (era 0.05 come default)
    es_patience:  int   = 20,
    es_metric:    str   = "sharpe",
    best_path:    Optional[str] = None,
) -> list[dict]:
    history       = []
    total_step    = 0
    best_metric   = -float("inf")
    no_improve    = 0

    for ep in range(1, n_episodes + 1):
        state = env.reset()
        agent.reset_noise()
        ep_reward     = 0.0
        critic_losses = []
        actor_losses  = []
        done = False

        while not done:
            if total_step < warmup:
                action = np.random.uniform(-1, 1, env.action_dim).astype(np.float32)
            else:
                action = agent.act(state, explore=True)

            next_state, reward, done, _ = env.step(action)
            agent.store(state, action, reward, next_state, done)

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

        ep_info = {
            "episode":         ep,
            "total_reward":    ep_reward,
            "portfolio_value": portfolio,
            "sharpe":          sharpe,
            "max_drawdown":    max_dd,
            "noise_sigma":     agent.noise.sigma,
            "critic_loss":     float(np.mean(critic_losses)) if critic_losses else None,
            "actor_loss":      float(np.mean(actor_losses))  if actor_losses  else None,
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
            best_tag = " ★" if no_improve == 0 else f" (no improve: {no_improve}/{es_patience})"
            print(
                f"Ep {ep:4d}/{n_episodes} | "
                f"Reward: {ep_reward:+.4f} | "
                f"Portfolio: ${portfolio:,.0f} | "
                f"Sharpe: {sharpe:.3f} | "
                f"MaxDD: {max_dd:.3f} | "
                f"σ: {agent.noise.sigma:.3f} | "
                f"C: {cl} | A: {al}"
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