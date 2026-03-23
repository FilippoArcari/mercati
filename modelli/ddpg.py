"""
modelli/ddpg.py — Deep Deterministic Policy Gradient per trading
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
import os
from typing import Optional


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

    def decay(self, factor=0.995):
        self.sigma = max(self.sigma * factor, 0.01)


# ─── Replay Buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states),      dtype=torch.float32),
            torch.tensor(np.array(actions),     dtype=torch.float32),
            torch.tensor(np.array(rewards),     dtype=torch.float32).unsqueeze(1),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(np.array(dones),       dtype=torch.float32).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)


# ─── Actor ────────────────────────────────────────────────────────────────────

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=None):
        super().__init__()
        if hidden is None:
            hidden = [256, 256]

        layers = []
        in_dim = state_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.ReLU()]
            in_dim = h
        layers += [nn.Linear(in_dim, action_dim), nn.Tanh()]

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, state):
        return self.net(state)


# ─── Critic ───────────────────────────────────────────────────────────────────

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=None):
        super().__init__()
        if hidden is None:
            hidden = [256, 256]

        self.fc1 = nn.Linear(state_dim, hidden[0])
        self.ln1 = nn.LayerNorm(hidden[0])
        self.fc2 = nn.Linear(hidden[0] + action_dim, hidden[1])
        self.ln2 = nn.LayerNorm(hidden[1])
        self.out = nn.Linear(hidden[1], 1)
        self._init_weights()

    def _init_weights(self):
        for m in [self.fc1, self.fc2, self.out]:
            nn.init.orthogonal_(m.weight, gain=0.01)
            nn.init.zeros_(m.bias)

    def forward(self, state, action):
        x = F.relu(self.ln1(self.fc1(state)))
        x = torch.cat([x, action], dim=1)
        x = F.relu(self.ln2(self.fc2(x)))
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
    ):
        if actor_hidden  is None: actor_hidden  = [256, 256]
        if critic_hidden is None: critic_hidden = [256, 256]

        self.state_dim    = state_dim
        self.action_dim   = action_dim
        self.gamma        = gamma
        self.tau          = tau
        self.batch_size   = batch_size
        self.update_every = update_every
        self._step_count  = 0

        self.actor         = Actor(state_dim,  action_dim, actor_hidden)
        self.actor_target  = Actor(state_dim,  action_dim, actor_hidden)
        self.critic        = Critic(state_dim, action_dim, critic_hidden)
        self.critic_target = Critic(state_dim, action_dim, critic_hidden)

        self._hard_update(self.actor_target,  self.actor)
        self._hard_update(self.critic_target, self.critic)

        self.opt_actor  = torch.optim.Adam(self.actor.parameters(),  lr=lr_actor)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_critic,
                                           weight_decay=1e-4)

        self.buffer = ReplayBuffer(buffer_capacity)
        self.noise  = OUNoise(action_dim, sigma=noise_sigma)

    def act(self, state, explore=True):
        s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(s).squeeze(0).numpy()
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
        critic_loss = F.mse_loss(q_current, q_target)

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

    def decay_noise(self, factor=0.995):
        self.noise.decay(factor)

    def reset_noise(self):
        self.noise.reset()

    # ── checkpoint ────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "state_dim":     self.state_dim,     # ← salvato per verifica compatibilità
            "action_dim":    self.action_dim,
            "actor":         self.actor.state_dict(),
            "actor_target":  self.actor_target.state_dict(),
            "critic":        self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
        }, path)
        print(f"[DDPG] Checkpoint salvato in {path} "
              f"(state_dim={self.state_dim}, action_dim={self.action_dim})")

    def load(self, path: str) -> bool:
        """
        Carica il checkpoint se è compatibile con le dimensioni correnti.

        Ritorna True se il caricamento ha avuto successo,
                False se il checkpoint è incompatibile (l'agente riparte da zero).
        """
        ckpt = torch.load(path, weights_only=True)

        saved_state_dim  = ckpt.get("state_dim")
        saved_action_dim = ckpt.get("action_dim")

        # Verifica compatibilità dimensioni
        if saved_state_dim != self.state_dim or saved_action_dim != self.action_dim:
            print(
                f"[DDPG] Checkpoint incompatibile — ignorato.\n"
                f"  Checkpoint : state_dim={saved_state_dim}, action_dim={saved_action_dim}\n"
                f"  Corrente   : state_dim={self.state_dim},  action_dim={self.action_dim}\n"
                f"  L'agente riparte da zero."
            )
            return False

        self.actor.load_state_dict(ckpt["actor"])
        self.actor_target.load_state_dict(ckpt["actor_target"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        print(f"[DDPG] Checkpoint caricato da {path} "
              f"(state_dim={self.state_dim}, action_dim={self.action_dim})")
        return True


# ─── Training loop ─────────────────────────────────────────────────────────────

def train_ddpg(agent, env, n_episodes=200, warmup=1000, log_every=10):
    history    = []
    total_step = 0

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

            next_state, reward, done, info = env.step(action)
            agent.store(state, action, reward, next_state, done)

            losses = agent.update()
            if losses:
                critic_losses.append(losses["critic_loss"])
                actor_losses.append(losses["actor_loss"])
            print( f"\r[DDPG] Ep {ep}/{n_episodes} | Step {total_step} | Reward: {ep_reward:+.4f}", end="")
            state       = next_state
            ep_reward  += reward
            total_step += 1

        agent.decay_noise()

        ep_info = {
            "episode":         ep,
            "total_reward":    ep_reward,
            "portfolio_value": env.portfolio_history()[-1],
            "sharpe":          env.sharpe_ratio(),
            "max_drawdown":    env.max_drawdown(),
            "critic_loss":     float(np.mean(critic_losses)) if critic_losses else None,
            "actor_loss":      float(np.mean(actor_losses))  if actor_losses  else None,
        }
        history.append(ep_info)

        if ep % log_every == 0:
            cl = f"{ep_info['critic_loss']:.5f}" if ep_info["critic_loss"] else "n/a"
            al = f"{ep_info['actor_loss']:.5f}"  if ep_info["actor_loss"]  else "n/a"
            print(
                f"Ep {ep:4d}/{n_episodes} | "
                f"Reward: {ep_reward:+.4f} | "
                f"Portfolio: ${ep_info['portfolio_value']:,.2f} | "
                f"Sharpe: {ep_info['sharpe']:.3f} | "
                f"MaxDD: {ep_info['max_drawdown']:.3f} | "
                f"C-loss: {cl} | A-loss: {al}"
            )

    return history