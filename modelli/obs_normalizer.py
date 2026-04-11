"""
modelli/obs_normalizer.py — Normalizzazione online dello stato per DDPG con Training Adattivo
"""

from __future__ import annotations

import numpy as np
import os
from typing import Callable, Optional


# ─── Running Mean / Variance (Welford online algorithm) ───────────────────────

class RunningMeanStd:
    """
    Stima incrementale di media e varianza con l'algoritmo di Welford.
    Numericamente stabile, aggiornabile un campione (o un batch) alla volta.
    """

    def __init__(self, shape: int | tuple, epsilon: float = 1e-4):
        self.shape   = (shape,) if isinstance(shape, int) else shape
        self.mean    = np.zeros(self.shape, dtype=np.float64)
        self.var     = np.ones(self.shape,  dtype=np.float64)
        self.count   = epsilon

    def update(self, x: np.ndarray):
        """Aggiorna le statistiche con un nuovo campione o batch x."""
        batch_mean  = np.mean(x, axis=0)
        batch_var   = np.var(x,  axis=0)
        batch_count = x.shape[0] if x.ndim > 1 else 1

        delta        = batch_mean - self.mean
        tot_count    = self.count + batch_count

        new_mean     = self.mean + delta * batch_count / tot_count
        m_a          = self.var * self.count
        m_b          = batch_var * batch_count
        m2           = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var      = m2 / tot_count

        self.mean    = new_mean
        self.var     = new_var
        self.count   = tot_count


class ObsNormalizer:
    """
    Normalizzatore di osservazioni per DDPG.
    Sottrae la media e divide per la deviazione standard in modo incrementale.
    """

    def __init__(self, shape: int, clip: float = 10.0):
        self.rms  = RunningMeanStd(shape=shape)
        self.clip = clip

    def normalize(self, obs: np.ndarray, update: bool = True) -> np.ndarray:
        if update:
            self.rms.update(obs)
        
        # Standardizzazione: (x - mu) / sigma
        std = np.sqrt(self.rms.var + 1e-8)
        norm_obs = (obs - self.rms.mean) / std
        
        # Clipping per evitare outliner estremi che destabilizzano la rete
        return np.clip(norm_obs, -self.clip, self.clip).astype(np.float32)

    def save(self, path: str):
        np.savez(path, mean=self.rms.mean, var=self.rms.var, count=self.rms.count)

    def load(self, path: str):
        if not os.path.exists(path):
            return
        data = np.load(path)
        self.rms.mean  = data["mean"]
        self.rms.var   = data["var"]
        self.rms.count = data["count"]


# ─── Training Loop con Normalizzazione e Integrazione Termodinamica ──────────

def train_ddpg_normalized(
    env,
    agent,
    n_episodes: int = 50,
    normalizer: Optional[ObsNormalizer] = None,
    norm_path:  Optional[str] = None,
    log_every:  int = 5,
    es_patience: int = 20, # Early stopping
):
    """
    Loop di training standard con normalizzazione online dello stato.
    Integrazione v3.0: Supporto per Adaptive Exploration Regime.
    """
    if normalizer is None:
        normalizer = ObsNormalizer(shape=env.observation_space.shape[0])

    best_sharpe = -np.inf
    no_improve  = 0

    for ep in range(1, n_episodes + 1):
        raw_state = env.reset()
        # Primo step: non aggiorniamo il normalizzatore sul reset per evitare bias
        state = normalizer.normalize(raw_state, update=True)
        
        ep_reward = 0.0
        done = False
        
        while not done:
            # ★ INNOVAZIONE: Recupero del regime termodinamico dall'ambiente
            # Se l'ambiente non ha ancora il metodo, default a 0.0 (regime neutro)
            current_regime = env.get_current_regime() if hasattr(env, 'get_current_regime') else 0.0
            
            # ★ INNOVAZIONE: Passaggio del regime all'agente per scalare il rumore (noise)
            action = agent.act(state, explore=True, current_regime=current_regime)
            
            next_raw_state, reward, done, info = env.step(action)
            
            # Normalizziamo il prossimo stato
            next_state = normalizer.normalize(next_raw_state, update=True)
            
            # Memorizzazione ed eventuale update
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            if len(agent.replay_buffer) > getattr(agent, "batch_size", 256):
                agent.update()
                
            state = next_state
            ep_reward += reward

        # Metriche di fine episodio
        sharpe    = env.sharpe_ratio()
        portfolio = env._get_portfolio_value()
        max_dd    = env.max_drawdown()
        ep_info   = agent.get_last_train_info()

        # Early Stopping / Saving Logic
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            no_improve  = 0
            if norm_path:
                normalizer.save(norm_path)
        else:
            no_improve += 1

        if ep % log_every == 0:
            cl = f"{ep_info['critic_loss']:.5f}" if ep_info["critic_loss"] else "n/a"
            al = f"{ep_info['actor_loss']:.5f}"  if ep_info["actor_loss"]  else "n/a"

            best_tag = " ★" if no_improve == 0 else f" ({no_improve}/{es_patience})"
            
            print(
                f"Ep {ep:4d}/{n_episodes} | "
                f"Reward: {ep_reward:+.4f} | "
                f"Portfolio: ${portfolio:,.0f} | "
                f"Sharpe: {sharpe:.3f} | "
                f"MaxDD: {max_dd:.3f} | "
                f"Sigma: {agent.noise.sigma:.3f} | "
                f"C: {cl} | A: {al}{best_tag}"
            )

        if no_improve >= es_patience:
            print(f"Early stopping triggerato all'episodio {ep}")
            break

    return normalizer