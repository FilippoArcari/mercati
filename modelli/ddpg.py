"""
modelli/ddpg.py — DDPG Agent Wrapper (Stable-Baselines3)

FIX (v2.1):
  • ThermoNoiseCallback ora prende agent_wrapper invece di base_sigma fisso.
    Usa agent_wrapper.noise.sigma (il valore già decaduto) come base per lo
    scaling di regime → il decay funziona correttamente (sigma decresce nel
    tempo invece di restare fisso a 0.149 per tutti gli episodi).
  • decay_noise ora aggiorna SOLO agent.noise.sigma (valore canonico del decay),
    senza leggere da action_noise._sigma (che ThermoNoiseCallback scala per
    regime ogni step). Questo evita interferenze tra i due meccanismi.
"""

import os
import numpy as np
import pandas as pd
from typing import Optional, List, Callable

import torch
import torch.nn as nn
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import BaseCallback

from modelli.device_setup import get_device, get_map_location

# ─── Phase-Aware Noise Scaler ─────────────────────────────────────────────────

def get_phase_aware_noise_scale(current_regime: float, base_sigma: float = 0.2) -> float:
    regime = int(current_regime)
    phase_multipliers = {
        1: 1.2,  # RALLY_REALE
        2: 0.8,  # RALLY_ESAUSTO
        3: 0.5,  # COMPRESSIONE
        4: 1.0,  # RIMBALZO
        0: 1.0   # NEUTRO
    }
    multiplier = phase_multipliers.get(regime, 1.0)
    return base_sigma * multiplier


class ThermoNoiseCallback(BaseCallback):
    """
    Legge il current_regime dall'environment e scala l'action noise on-the-fly.

    FIX v2.1: riceve agent_wrapper invece di base_sigma fisso.
    Usa agent_wrapper.noise.sigma (aggiornato da decay_noise ad ogni episodio)
    come base per il calcolo del sigma scalato per regime.
    In questo modo il sigma effettivo decresce nel tempo (decay) e varia per
    regime (phase scaling), senza che i due meccanismi si sovrascrivano.
    """
    def __init__(self, agent_wrapper, verbose=0):
        super().__init__(verbose)
        self.agent_wrapper = agent_wrapper

    def _on_step(self) -> bool:
        env_unwrapped = self.training_env.envs[0].unwrapped
        regime = 0.0
        if hasattr(env_unwrapped, "get_current_regime"):
            regime = env_unwrapped.get_current_regime()

        noise = getattr(self.model, "action_noise", None)
        if isinstance(noise, OrnsteinUhlenbeckActionNoise):
            # Usa il sigma canonico già decaduto come base per lo scaling di regime.
            # agent_wrapper.noise.sigma viene aggiornato da decay_noise() a fine episodio.
            decayed_sigma = self.agent_wrapper.noise.sigma
            new_sigma     = get_phase_aware_noise_scale(regime, decayed_sigma)
            noise._sigma  = new_sigma * np.ones_like(noise._sigma)
        return True


# ─── DDPG Agent Wrapper (Stable-Baselines3) ───────────────────────────────────

class DDPGAgent:
    """
    Wrapper che espone la vecchia interfaccia DDPGAgent (act, load, save)
    ma alloca internamente un modello ottimizzato stable_baselines3.DDPG.
    """
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
        update_every=50,
        noise_sigma=0.2,
        episodic_episodes=20,  # Deprecato: ignorato con SB3
        device=None,
    ):
        self.state_dim    = state_dim
        self.action_dim   = action_dim
        self.device       = device or get_device()
        self.actor_hidden = actor_hidden or [256, 256]
        self.critic_hidden= critic_hidden or [256, 256]
        self.lr_actor     = lr_actor
        self.buffer_capacity = buffer_capacity
        self.batch_size   = batch_size
        self.tau          = tau
        self.gamma        = gamma
        self.update_every = update_every

        self.action_noise_kwargs = dict(
            mean=np.zeros(action_dim),
            sigma=noise_sigma * np.ones(action_dim)
        )

        self.policy_kwargs = dict(
            net_arch=dict(
                pi=list(self.actor_hidden),
                qf=list(self.critic_hidden)
            )
        )

        self.model = None

        # noise.sigma = valore canonico del decay (aggiornato da decay_noise).
        # ThermoNoiseCallback legge da qui per scalare il sigma effettivo per regime.
        class DummyNoise:
            def __init__(self, s):
                self.sigma = s
        self.noise = DummyNoise(noise_sigma)
        self.episodic_buffer = None

    def set_env(self, env):
        if self.model is None:
            action_noise = OrnsteinUhlenbeckActionNoise(**self.action_noise_kwargs)
            self.model = DDPG(
                "MlpPolicy",
                env=env,
                learning_rate=self.lr_actor,
                buffer_size=self.buffer_capacity,
                batch_size=self.batch_size,
                tau=self.tau,
                gamma=self.gamma,
                train_freq=(self.update_every, "step"),
                action_noise=action_noise,
                policy_kwargs=self.policy_kwargs,
                device=self.device,
                verbose=0,
            )
        else:
            self.model.set_env(env)

    def reset_noise(self):
        if self.model.action_noise is not None:
            self.model.action_noise.reset()

    def decay_noise(self, factor=0.995, floor=0.02):
        """
        FIX v2.1: aggiorna SOLO self.noise.sigma (valore canonico).
        Non scrive in action_noise._sigma direttamente: ThermoNoiseCallback
        lo aggiorna ogni step usando self.noise.sigma come base.
        Questo evita che i due meccanismi (decay + regime scaling) si sovrascrivano.
        """
        self.noise.sigma = max(floor, self.noise.sigma * factor)

    def act(self, state: np.ndarray, explore=True, current_regime: float = 0.0) -> np.ndarray:
        action, _ = self.model.predict(state, deterministic=not explore)
        return action

    def store(self, state, action, reward, next_state, done):
        pass  # Gestito da SB3 internamente

    def update(self):
        return None  # Gestito da SB3 internamente

    def load(self, path: str) -> bool:
        """
        Carica un checkpoint SB3. SB3 salva come <path>.zip oppure <path> (senza estensione).
        Accetta sia path con .pth (per compatibilità naming convention del progetto)
        sia path senza estensione / con .zip.
        """
        base = path.replace(".pth", "")
        candidates = [base + ".zip", base, base + ".pth"]
        found = next((c for c in candidates if os.path.exists(c)), None)
        if found is None:
            return False
        try:
            self.model = DDPG.load(found.replace(".zip", ""), device=get_map_location())
            print(f"[SB3 DDPG] Checkpoint caricato: {found}")
            return True
        except Exception as e:
            print(f"[SB3 DDPG] Errore caricamento {found}: {e}")
            return False

    def save(self, path: str, tag: str = "") -> None:
        if path.endswith(".pth"):
            path = path.replace(".pth", "")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.model.save(path)
        label = f" [{tag}]" if tag else ""
        print(f"[SB3 DDPG] Checkpoint salvato{label}: {path}")


# ─── ThermoEnsemble ───────────────────────────────────────────────────────────

class ThermoEnsemble:
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

        profile_shapes = [p.shape for p in fold_profiles]
        unique_shapes  = set(profile_shapes)

        if len(unique_shapes) > 1:
            print(f"WARNING: Profili termodinamici hanno dimensioni diverse!")
            min_dim      = min(p.shape[0] for p in fold_profiles)
            fold_profiles = [p[:min_dim] for p in fold_profiles]
            self._expected_dim = min_dim
        else:
            self._expected_dim = fold_profiles[0].shape[0]

        self.agents        = agents
        self.fold_profiles = [np.asarray(p, dtype=np.float32) for p in fold_profiles]
        self.temperature   = temperature
        self._last_weights: Optional[np.ndarray] = None

    def _compute_weights(self, current_thermo: np.ndarray) -> np.ndarray:
        x = np.asarray(current_thermo, dtype=np.float32)
        if x.shape[0] != self._expected_dim:
            if x.shape[0] > self._expected_dim:
                x = x[:self._expected_dim]
            else:
                x = np.pad(x, (0, self._expected_dim - x.shape[0]), 'constant')

        distances = np.array([
            np.linalg.norm(x - p) for p in self.fold_profiles
        ], dtype=np.float32)

        logits = -distances / (self.temperature + 1e-8)
        logits = logits - logits.max()
        exp    = np.exp(logits)

        weights = exp / (exp.sum() + 1e-8)
        self._last_weights = weights
        return weights

    def act(self, state: np.ndarray, current_thermo: np.ndarray, explore: bool = False) -> np.ndarray:
        weights = self._compute_weights(current_thermo)
        actions = np.stack([agent.act(state, explore=explore) for agent in self.agents])
        blended = (weights[:, None] * actions).sum(axis=0)
        return np.clip(blended, -1.0, 1.0)

    def weights_summary(self) -> str:
        if self._last_weights is None:
            return "n/a"
        parts = [f"fold{i+1}={w:.3f}" for i, w in enumerate(self._last_weights)]
        return " | ".join(parts)


def compute_thermo_profile(
    thermo_df,
    start_idx: int,
    end_idx:   int,
    feature_names: Optional[List[str]] = None,
) -> np.ndarray:
    if feature_names is None:
        feature_names = [c for c in thermo_df.columns if c.startswith("Thm_")]

    available = [f for f in feature_names if f in thermo_df.columns]
    if not available:
        raise ValueError("Nessuna colonna Thm_* trovata in thermo_df.")

    window = thermo_df.iloc[start_idx:end_idx][available]
    return window.mean(axis=0).values.astype(np.float32)