"""
modelli/obs_normalizer.py — Normalizzazione online dello stato per DDPG con Training Adattivo
"""

from __future__ import annotations

import numpy as np
import os
from typing import Optional


# ─── Running Mean / Variance (Welford online algorithm) ───────────────────────

class RunningMeanStd:
    """
    Stima incrementale di media e varianza con l'algoritmo di Welford.
    Numericamente stabile, aggiornabile un campione (o un batch) alla volta.
    """

    def __init__(self, shape: int | tuple, epsilon: float = 1e-4):
        self.shape = (shape,) if isinstance(shape, int) else shape
        self.mean  = np.zeros(self.shape, dtype=np.float64)
        self.var   = np.ones(self.shape,  dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray):
        """Aggiorna le statistiche con un nuovo campione o batch x."""
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


# ─── Training Loop con Normalizzazione e Integrazione Termodinamica ───────────

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
    """
    Loop di training con normalizzazione online dello stato.

    Restituisce la history degli episodi (list[dict]), coerente con
    run_trade / run_walk_forward che iterano su di essa.

    Bug corretti rispetto alla versione precedente
    ─────────────────────────────────────────────
    FIX 1  agent.replay_buffer.push()  →  agent.store()
           (l'attributo si chiama 'buffer', il metodo pubblico è store()
            che aggiorna anche l'episodic_buffer in un colpo solo)

    FIX 2  agent.get_last_train_info() →  raccolta losses da agent.update()
           (il metodo non esiste su DDPGAgent; update() restituisce già il dict)

    FIX 3  env._get_portfolio_value()  →  env.portfolio_history()[-1]
           (il metodo privato non esiste nell'env)

    FIX 4  return normalizer           →  return history
           (trade.py assegna il risultato a 'history' e la itera)

    FIX 5  Aggiunto agent.decay_noise() dopo ogni episodio (era assente)

    FIX 6  Aggiunta gestione episodic_buffer.start_episode() / end_episode()
           (necessaria per il curriculum replay)

    FIX 7  Gestione API gymnasium vs gym:
           - reset() → (obs, info) in gymnasium, obs in gym
           - step()  → (obs, r, terminated, truncated, info) in gymnasium
                        (obs, r, done, info) in gym
           Senza questo fix, raw_state è una tupla e np.mean() crasha
           con ValueError: inhomogeneous shape.
    """
    if normalizer is None:
        normalizer = ObsNormalizer(shape=env.observation_space.shape[0])

    # ── Rileva API: gymnasium (5-tuple step) vs gym (4-tuple step) ───────────
    _gymnasium_api = False
    try:
        import gymnasium  # noqa: F401
        _gymnasium_api = True
    except ImportError:
        pass

    def _reset_env():
        """Spacchetta reset() per gymnasium (obs, info) e gym (obs)."""
        result = env.reset()
        if _gymnasium_api and isinstance(result, tuple):
            return result[0]           # estrai solo obs
        return result

    def _step_env(action):
        """Spacchetta step() per gymnasium (obs,r,term,trunc,info) e gym (obs,r,done,info)."""
        result = env.step(action)
        if _gymnasium_api and len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
            return obs, reward, done, info
        # gym classico: (obs, reward, done, info)
        return result

    history     : list[dict] = []
    best_sharpe : float      = -np.inf
    no_improve  : int        = 0

    for ep in range(1, n_episodes + 1):
        raw_state = _reset_env()                              # FIX 7
        state     = normalizer.normalize(raw_state, update=True)

        # FIX 6 — notifica inizio episodio al buffer episodico
        if agent.episodic_buffer is not None:
            agent.episodic_buffer.start_episode()

        agent.reset_noise()

        ep_reward     : float      = 0.0
        critic_losses : list[float] = []
        actor_losses  : list[float] = []
        done          : bool        = False

        while not done:
            # Regime termodinamico per scalare il rumore (0 = neutro se assente)
            current_regime: float = (
                env.get_current_regime()
                if hasattr(env, "get_current_regime")
                else 0.0
            )

            action = agent.act(state, explore=True, current_regime=current_regime)

            next_raw_state, reward, done, info = _step_env(action)  # FIX 7
            next_state = normalizer.normalize(next_raw_state, update=True)

            # FIX 1 — agent.store() scrive su buffer + episodic_buffer insieme
            agent.store(state, action, reward, next_state, done)

            # FIX 2 — losses dal valore di ritorno di update(), non da un metodo inesistente
            losses = agent.update()
            if losses:
                critic_losses.append(losses["critic_loss"])
                actor_losses.append(losses["actor_loss"])

            state      = next_state
            ep_reward += reward

        # FIX 5 — decadimento rumore dopo ogni episodio
        agent.decay_noise(factor=noise_decay, floor=noise_floor)

        # Metriche di fine episodio
        sharpe    = env.sharpe_ratio()
        # FIX 3 — portfolio_history()[-1] invece di _get_portfolio_value()
        portfolio = env.portfolio_history()[-1]
        max_dd    = env.max_drawdown()

        # FIX 6 — notifica fine episodio al buffer episodico
        if agent.episodic_buffer is not None:
            agent.episodic_buffer.end_episode(sharpe=sharpe)

        ep_info: dict = {
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

        # Early Stopping / Saving Logic
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            no_improve  = 0
            if norm_path:
                normalizer.save(norm_path)
            if ckpt_path and hasattr(agent, "save"):
                agent.save(ckpt_path, tag=f"BEST Ep {ep}")
        else:
            no_improve += 1

        if ep % log_every == 0:
            cl       = f"{ep_info['critic_loss']:.5f}" if ep_info["critic_loss"] else "n/a"
            al       = f"{ep_info['actor_loss']:.5f}"  if ep_info["actor_loss"]  else "n/a"
            best_tag = " ★" if no_improve == 0 else f" ({no_improve}/{es_patience})"

            ep_buf_str = ""
            if agent.episodic_buffer is not None:
                ep_buf_str = (
                    f" | EpBuf: {agent.episodic_buffer.n_episodes}/"
                    f"{agent.episodic_buffer.max_episodes}"
                    f" best={agent.episodic_buffer.best_sharpe:.2f}"
                )

            print(
                f"Ep {ep:4d}/{n_episodes} | "
                f"Reward: {ep_reward:+.4f} | "
                f"Portfolio: ${portfolio:,.0f} | "
                f"Sharpe: {sharpe:.3f} | "
                f"MaxDD: {max_dd:.3f} | "
                f"σ: {agent.noise.sigma:.3f} | "
                f"C: {cl} | A: {al}"
                f"{ep_buf_str}"
                f"{best_tag}"
            )

        if no_improve >= es_patience:
            print(f"Early stopping triggerato all'episodio {ep}")
            break

    # FIX 4 — restituisce history (atteso da trade.py), non normalizer
    return history