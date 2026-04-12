"""
modelli/obs_normalizer.py

FIX (v2.1):
  • HistoryAndDecayCallback legge le metriche da infos[0]["episode"] invece
    che dall'env già resettato da DummyVecEnv. Questo risolve il bug per cui
    Portfolio=$100 e Sharpe=0.000 per tutti gli episodi (perché l'env veniva
    auto-resettato da DummyVecEnv prima che il callback leggesse i dati).
  • ThermoNoiseCallback riceve agent_wrapper (non base_sigma fisso) per far
    funzionare correttamente il decay del sigma nel tempo.
  • Normalizer salvato solo quando il checkpoint è salvato (stesso criterio).
"""

import numpy as np
import os
import gymnasium as gym
from typing import Optional

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

# ─── Running Mean / Variance (Welford online algorithm) ───────────────────────

class RunningMeanStd:
    def __init__(self, shape: int | tuple, epsilon: float = 1e-4):
        self.shape = (shape,) if isinstance(shape, int) else shape
        self.mean  = np.zeros(self.shape, dtype=np.float64)
        self.var   = np.ones(self.shape,  dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray):
        batch_mean  = np.mean(x, axis=0)
        batch_var   = np.var(x,  axis=0)
        batch_count = x.shape[0] if x.ndim > 1 else 1

        delta     = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a      = self.var * self.count
        m_b      = batch_var * batch_count
        m2       = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var  = m2 / tot_count

        self.mean  = new_mean
        self.var   = new_var
        self.count = tot_count


class ObsNormalizer:
    def __init__(self, shape: int, clip: float = 10.0):
        self.rms  = RunningMeanStd(shape=shape)
        self.clip = clip

    def normalize(self, obs: np.ndarray, update: bool = True) -> np.ndarray:
        if update:
             self.rms.update(np.expand_dims(obs, axis=0) if obs.ndim == 1 else obs)

        std      = np.sqrt(self.rms.var + 1e-8)
        norm_obs = (obs - self.rms.mean) / std

        return np.clip(norm_obs, -self.clip, self.clip).astype(np.float32)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez(path, mean=self.rms.mean, var=self.rms.var, count=self.rms.count)

    def load(self, path: str):
        if not os.path.exists(path):
            return
        data           = np.load(path)
        self.rms.mean  = data["mean"]
        self.rms.var   = data["var"]
        self.rms.count = data["count"]


class ObsNormalizerWrapper(gym.ObservationWrapper):
    def __init__(self, env, normalizer: ObsNormalizer):
        super().__init__(env)
        self.normalizer = normalizer

    def observation(self, observation):
        obs = observation
        return self.normalizer.normalize(obs, update=True)


class HistoryAndDecayCallback(BaseCallback):
    """
    Callback per replicare il comportamento di history e log del vecchio DDPG,
    compatibile con l'early stopping in trade.py.

    FIX v2.1: legge le metriche di episodio da infos[0]["episode"] (inserite
    da TradingEnv.step quando done=True) invece che dall'env già resettato.
    Questo risolve Portfolio=$100 / Sharpe=0.000 fissi per tutti gli episodi.
    """
    def __init__(
        self,
        agent_wrapper,
        n_episodes:  int,
        norm_path:   str,
        ckpt_path:   str,
        log_every:   int,
        es_patience: int,
        noise_decay: float,
        noise_floor: float,
        verbose=0,
    ):
        super().__init__(verbose)
        self.agent_wrapper = agent_wrapper
        self.history       = []
        self.best_sharpe   = -np.inf
        self.no_improve    = 0
        self.n_episodes    = n_episodes
        self.norm_path     = norm_path
        self.ckpt_path     = ckpt_path
        self.log_every     = log_every
        self.es_patience   = es_patience
        self.noise_decay   = noise_decay
        self.noise_floor   = noise_floor

        self.ep        = 0
        self.ep_reward = 0.0

    def _on_step(self) -> bool:
        self.ep_reward += self.locals.get("rewards")[0]

        if self.locals.get("dones")[0]:
            self.ep += 1

            # ── FIX v2.1 ───────────────────────────────────────────────────
            # DummyVecEnv chiama env.reset() PRIMA che il callback venga
            # invocato. Leggere da env.portfolio_history() restituisce [100.0]
            # perché l'env è già stato resettato.
            # La soluzione: TradingEnv.step inserisce le metriche finali in
            # info["episode"] nel momento del done=True (prima del reset).
            # Le leggiamo da self.locals["infos"][0]["episode_metrics"].
            infos    = self.locals.get("infos", [{}])
            info     = infos[0] if infos else {}
            ep_data  = info.get("episode_metrics", info.get("episode", {}))

            # Fallback all'env per retrocompatibilità (es. env custom senza fix)
            e = self.training_env.envs[0].unwrapped
            portfolio = ep_data.get("portfolio_value",
                                    e.portfolio_history()[-1] if hasattr(e, "portfolio_history") else e.initial_capital)
            sharpe    = ep_data.get("sharpe",       0.0)
            max_dd    = ep_data.get("max_drawdown", 0.0)

            # Noise decay a fine episodio (aggiorna solo agent.noise.sigma)
            self.agent_wrapper.decay_noise(
                factor=self.noise_decay,
                floor=self.noise_floor,
            )

            ep_info = {
                "episode":         self.ep,
                "total_reward":    self.ep_reward,
                "portfolio_value": portfolio,
                "sharpe":          sharpe,
                "max_drawdown":    max_dd,
                "noise_sigma":     self.agent_wrapper.noise.sigma,
                "critic_loss":     0.0,
                "actor_loss":      0.0,
            }
            self.history.append(ep_info)
            self.ep_reward = 0.0

            # Early Stopping / Saving
            if sharpe > self.best_sharpe:
                self.best_sharpe = sharpe
                self.no_improve  = 0
                if self.norm_path:
                    self.training_env.envs[0].normalizer.save(self.norm_path)
                if self.ckpt_path:
                    self.agent_wrapper.save(self.ckpt_path, tag=f"BEST Ep {self.ep}")
            else:
                self.no_improve += 1

            if self.ep % self.log_every == 0:
                best_tag = " ★" if self.no_improve == 0 else f" ({self.no_improve}/{self.es_patience})"
                print(
                    f"Ep {self.ep:4d}/{self.n_episodes} | "
                    f"Reward: {ep_info['total_reward']:+.4f} | "
                    f"Portfolio: ${portfolio:,.2f} | "
                    f"Sharpe: {sharpe:.3f} | "
                    f"MaxDD: {max_dd:.3f} | "
                    f"σ: {self.agent_wrapper.noise.sigma:.3f}"
                    f"{best_tag}"
                )

            if self.no_improve >= self.es_patience or self.ep >= self.n_episodes:
                if self.no_improve >= self.es_patience:
                    print(f"Early stopping triggerato all'episodio {self.ep}")
                return False

        return True


def train_ddpg_normalized(
    env,
    agent,
    n_episodes:  int                    = 50,
    normalizer:  Optional[ObsNormalizer] = None,
    norm_path:   Optional[str]           = None,
    ckpt_path:   Optional[str]           = None,
    log_every:   int                    = 5,
    es_patience: int                    = 20,
    noise_decay: float                  = 0.995,
    noise_floor: float                  = 0.02,
) -> list[dict]:

    if normalizer is None:
        normalizer = ObsNormalizer(shape=env.observation_space.shape[0])

    wrapped_env = ObsNormalizerWrapper(env, normalizer)
    vec_env     = DummyVecEnv([lambda: wrapped_env])

    agent.set_env(vec_env)

    callback = HistoryAndDecayCallback(
        agent_wrapper=agent,
        n_episodes=n_episodes,
        norm_path=norm_path,
        ckpt_path=ckpt_path,
        log_every=log_every,
        es_patience=es_patience,
        noise_decay=noise_decay,
        noise_floor=noise_floor,
    )

    # FIX v2.1: ThermoNoiseCallback riceve agent_wrapper invece di base_sigma fisso.
    from modelli.ddpg import ThermoNoiseCallback
    thermo_callback = ThermoNoiseCallback(agent_wrapper=agent)

    from stable_baselines3.common.callbacks import CallbackList
    callbacks = CallbackList([callback, thermo_callback])

    # 1 episode = lunghezza dell'env dataframe
    total_timesteps = n_episodes * len(env.df)

    agent.model.learn(total_timesteps, callback=callbacks)

    return callback.history