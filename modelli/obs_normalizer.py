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
        # In Gymnasium, self.observation(obs) riceve solo l'array observation
        obs = observation
        # L'update avviene durante l'esplorazione step() tramite il Wrapper in train mode
        return self.normalizer.normalize(obs, update=True)


class HistoryAndDecayCallback(BaseCallback):
    """
    Callback per replicare il comportamento di history e log del vecchio DDPG,
    compatibile con l'early stopping in trade.py.
    """
    def __init__(self, agent_wrapper, n_episodes: int, norm_path: str, ckpt_path: str, log_every: int, es_patience: int, noise_decay: float, noise_floor: float, verbose=0):
        super().__init__(verbose)
        self.agent_wrapper = agent_wrapper
        self.history = []
        self.best_sharpe = -np.inf
        self.no_improve = 0
        self.n_episodes = n_episodes
        self.norm_path = norm_path
        self.ckpt_path = ckpt_path
        self.log_every = log_every
        self.es_patience = es_patience
        self.noise_decay = noise_decay
        self.noise_floor = noise_floor

        self.ep = 0
        self.ep_reward = 0.0

    def _on_step(self) -> bool:
        self.ep_reward += self.locals.get("rewards")[0]

        if self.locals.get("dones")[0]:
            self.ep += 1
            e = self.training_env.envs[0].unwrapped
            sharpe = e.sharpe_ratio()
            portfolio = e.portfolio_history()[-1]
            max_dd = e.max_drawdown()

            # Decadimento rumore
            self.agent_wrapper.decay_noise(factor=self.noise_decay, floor=self.noise_floor)

            ep_info = {
                "episode":         self.ep,
                "total_reward":    self.ep_reward,
                "portfolio_value": portfolio,
                "sharpe":          sharpe,
                "max_drawdown":    max_dd,
                "noise_sigma":     self.agent_wrapper.noise.sigma,
                "critic_loss":     0.0, # SB3 logs are tricky to grab directly per episode without hacking logger
                "actor_loss":      0.0,
            }
            self.history.append(ep_info)
            self.ep_reward = 0.0

            # Early Stopping / Saving
            if sharpe > self.best_sharpe:
                self.best_sharpe = sharpe
                self.no_improve = 0
                if self.norm_path:
                    # Riferimento al wrapper custom
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
                    f"Portfolio: ${portfolio:,.0f} | "
                    f"Sharpe: {sharpe:.3f} | "
                    f"MaxDD: {max_dd:.3f} | "
                    f"σ: {self.agent_wrapper.noise.sigma:.3f}"
                    f"{best_tag}"
                )

            if self.no_improve >= self.es_patience or self.ep >= self.n_episodes:
                if self.no_improve >= self.es_patience:
                    print(f"Early stopping triggerato all'episodio {self.ep}")
                # Ferma il training (SB3)
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
    vec_env = DummyVecEnv([lambda: wrapped_env])
    
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
    
    # Optional: ThermoNoiseCallback for phase-aware noise
    from modelli.ddpg import ThermoNoiseCallback
    thermo_callback = ThermoNoiseCallback(base_sigma=agent.noise.sigma)
    
    from stable_baselines3.common.callbacks import CallbackList
    callbacks = CallbackList([callback, thermo_callback])
    
    # 1 episode = length of env dataframe
    total_timesteps = n_episodes * len(env.df)
    
    agent.model.learn(total_timesteps, callback=callbacks)
    
    return callback.history