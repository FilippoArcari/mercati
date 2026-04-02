"""
modelli/obs_normalizer.py — Normalizzazione online dello stato per DDPG

Implementa RunningMeanStd: aggiorna media e varianza in modo incrementale
ad ogni step, senza guardare dati futuri (safe per produzione e backtesting).

Utilizzo tipico:
    normalizer = ObsNormalizer(state_dim, clip=5.0)

    # Nel training loop:
    raw_state = env.reset()
    state = normalizer.normalize(raw_state, update=True)  # aggiorna stats

    # In inference / evaluation:
    state = normalizer.normalize(raw_state, update=False)  # solo trasforma

    # Salvataggio/caricamento insieme al modello DDPG:
    normalizer.save("checkpoints/normalizer.npz")
    normalizer.load("checkpoints/normalizer.npz")
"""

from __future__ import annotations

import numpy as np


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
        self.count   = epsilon   # evita divisione per zero al primo step

    def update(self, x: np.ndarray) -> None:
        """
        Aggiorna con un singolo campione (1-D) o un batch (2-D: batch x features).
        """
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[np.newaxis, :]          # (1, features)

        batch_count = x.shape[0]
        batch_mean  = x.mean(axis=0)
        batch_var   = x.var(axis=0)

        self._parallel_update(batch_mean, batch_var, batch_count)

    def _parallel_update(
        self,
        batch_mean:  np.ndarray,
        batch_var:   np.ndarray,
        batch_count: int,
    ) -> None:
        """Chan's parallel variance formula — combina statistiche di due gruppi."""
        total = self.count + batch_count
        delta = batch_mean - self.mean

        new_mean = self.mean + delta * batch_count / total
        m_a      = self.var   * self.count
        m_b      = batch_var  * batch_count
        m2       = m_a + m_b + delta ** 2 * self.count * batch_count / total

        self.mean  = new_mean
        self.var   = m2 / total
        self.count = total

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var)


# ─── Normalizzatore completo con clip ─────────────────────────────────────────

class ObsNormalizer:
    """
    Wrapper pronto all'uso per normalizzare gli stati dell'ambiente.

    Parametri
    ---------
    state_dim : int
        Dimensione del vettore di stato.
    clip : float
        Valore massimo (assoluto) dopo la normalizzazione.
        Default 5.0 — i valori oltre +-5 sigma vengono troncati, riducendo
        l'influenza degli outlier senza perdere il segno.
    epsilon : float
        Piccolo valore aggiunto a std per evitare divisioni per zero.
    """

    def __init__(self, state_dim: int, clip: float = 5.0, epsilon: float = 1e-8):
        self.rms     = RunningMeanStd(state_dim)
        self.clip    = clip
        self.epsilon = epsilon

    def normalize(self, obs: np.ndarray, update: bool = True) -> np.ndarray:
        """
        Normalizza l'osservazione.

        Parameters
        ----------
        obs    : np.ndarray — stato grezzo da TradingEnv
        update : bool
            True  -> aggiorna le statistiche (usa durante il training)
            False -> applica solo la trasformazione (usa in eval/produzione)

        Returns
        -------
        np.ndarray float32 — stato normalizzato, clippato in [-clip, +clip]
        """
        obs = np.asarray(obs, dtype=np.float64)

        if update:
            self.rms.update(obs)

        normalized = (obs - self.rms.mean) / (self.rms.std + self.epsilon)
        clipped    = np.clip(normalized, -self.clip, self.clip)

        return clipped.astype(np.float32)

    # ── persistenza ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        np.savez(
            path,
            mean  = self.rms.mean,
            var   = self.rms.var,
            count = np.array([self.rms.count]),
        )
        print(f"[ObsNormalizer] Statistiche salvate -> {path}  (n={self.rms.count:.0f})")

    def load(self, path: str) -> None:
        data = np.load(path)
        self.rms.mean  = data["mean"]
        self.rms.var   = data["var"]
        self.rms.count = float(data["count"][0])
        print(f"[ObsNormalizer] Statistiche caricate <- {path}  (n={self.rms.count:.0f})")


# ─── Integrazione con train_ddpg ──────────────────────────────────────────────

def train_ddpg_normalized(
    agent,
    env,
    normalizer:   ObsNormalizer,
    n_episodes:   int   = 200,
    warmup:       int   = 1000,
    log_every:    int   = 10,
    noise_decay:  float = 0.995,
    noise_floor:  float = 0.05,
    es_patience:  int   = 20,
    es_metric:    str   = "sharpe",
    best_path:    str | None = None,
    norm_path:    str | None = None,
) -> list[dict]:
    """
    Drop-in replacement di train_ddpg con normalizzazione online.

    Differenze chiave rispetto alla versione base:
      - Ogni stato raw viene normalizzato prima di essere passato all'agente
      - update=True durante il training, update=False per eval
      - Il normalizer viene salvato insieme al checkpoint del modello
    """
    history      = []
    total_step   = 0
    best_metric  = -float("inf")
    no_improve   = 0

    for ep in range(1, n_episodes + 1):
        raw_state = env.reset()
        # update=True: le prime osservazioni calibrano le statistiche
        state = normalizer.normalize(raw_state, update=True)
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

            raw_next, reward, done, info = env.step(action)
            # Normalizza il next_state (update=True: continua ad affinare le stats)
            next_state = normalizer.normalize(raw_next, update=True)

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

        metric_map = {"sharpe": sharpe, "portfolio": portfolio, "reward": ep_reward}
        current_metric = metric_map.get(es_metric, sharpe)

        if current_metric > best_metric:
            best_metric = current_metric
            no_improve  = 0
            if best_path:
                agent.save(best_path, tag=f"BEST ep{ep} {es_metric}={best_metric:.4f}")
            if norm_path:
                normalizer.save(norm_path)   # salva le stats insieme al modello
        else:
            no_improve += 1

        if ep % log_every == 0:
            cl = f"{ep_info['critic_loss']:.5f}" if ep_info["critic_loss"] else "n/a"
            al = f"{ep_info['actor_loss']:.5f}"  if ep_info["actor_loss"]  else "n/a"
            best_tag = " ★" if no_improve == 0 else f" (no improve: {no_improve}/{es_patience})"
            print(
                f"Ep {ep:4d}/{n_episodes} | "
                f"Reward: {ep_reward:+.4f} | "
                f"Portfolio: ${portfolio:,.0f} | "
                f"Sharpe: {sharpe:.3f} | "
                f"MaxDD: {max_dd:.3f} | "
                f"sigma: {agent.noise.sigma:.3f} | "
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

