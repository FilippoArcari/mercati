"""
modelli/trading_env.py — Ambiente di trading per DDPG con Reward Gate Termodinamico
=====================================================================================

FIX (v4.1):
  • episode metrics (portfolio_value, sharpe, max_drawdown) vengono inseriti
    nell'info dict al momento del done=True, PRIMA che DummyVecEnv chiami
    env.reset(). Il callback le legge da infos[0]["episode"] invece che
    dall'env già resettato → risolve Portfolio=$100 / Sharpe=0.000 per tutti
    gli episodi.

Novità rispetto alla versione precedente:
  • Migrazione gym → gymnasium (risolve il warning NumPy 2.0)
  • ThermodynamicRewardGate: la reward viene modulata dal regime fisico del mercato.
  • Holding decay: penalità progressiva per posizioni tenute > N barre.
  • bars_in_position: tracciato per ticker, usato dal decay e dal gate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, List
from dataclasses import dataclass

import gymnasium as gym
from gymnasium import spaces


# ─── Thermodynamic Reward Gate ────────────────────────────────────────────────

@dataclass
class ThermoRewardConfig:
    z_stress_high:    float = 1.0
    z_stress_low:     float = 0.3
    z_expansion_low:  float = -0.3
    z_expansion_high: float = -1.0
    buy_mult_stress:     float = 0.25
    buy_mult_expansion:  float = 1.6
    sell_mult_stress:    float = 1.8
    sell_mult_expansion: float = 0.9
    holding_decay_start: int   = 20
    holding_decay_rate:  float = 0.003
    holding_decay_max:   float = 0.15
    work_efficiency_weight: float = 0.04
    z_ema_alpha: float = 0.15


class ThermodynamicRewardGate:
    def __init__(self, cfg: Optional[ThermoRewardConfig] = None):
        self.cfg    = cfg or ThermoRewardConfig()
        self._z_ema = 0.0

    def reset(self):
        self._z_ema = 0.0

    def compute(self, base_reward: float, action_type: str,
                position_held_bars: int = 0,
                thm_z: float = 0.0, thm_efficiency: float = 0.0) -> float:
        cfg = self.cfg
        if np.isfinite(thm_z):
            self._z_ema = cfg.z_ema_alpha * thm_z + (1 - cfg.z_ema_alpha) * self._z_ema
        z = self._z_ema

        if   z > cfg.z_stress_high:    regime = "stress_forte"
        elif z > cfg.z_stress_low:     regime = "stress_lieve"
        elif z < cfg.z_expansion_high: regime = "expansion_forte"
        elif z < cfg.z_expansion_low:  regime = "expansion_lieve"
        else:                          regime = "neutro"

        mult = 1.0
        a    = action_type.lower()
        if a == "buy":
            if regime == "stress_forte":
                mult = cfg.buy_mult_stress
            elif regime == "expansion_forte":
                mult = cfg.buy_mult_expansion
            elif regime == "expansion_lieve":
                mult = 1.0 + (cfg.buy_mult_expansion - 1.0) * 0.5
        elif a == "sell":
            if regime in ("stress_forte", "stress_lieve"):
                mult = cfg.sell_mult_stress
            elif regime in ("expansion_forte", "expansion_lieve"):
                mult = cfg.sell_mult_expansion

        reward = base_reward * mult

        if position_held_bars > cfg.holding_decay_start:
            excess  = position_held_bars - cfg.holding_decay_start
            penalty = min(excess * cfg.holding_decay_rate, cfg.holding_decay_max)
            reward -= penalty

        if np.isfinite(thm_efficiency):
            if a == "buy" and thm_efficiency < 0:
                reward += min(abs(thm_efficiency) * cfg.work_efficiency_weight, 0.05)
            elif a == "sell" and thm_efficiency > 0:
                reward += min(thm_efficiency * cfg.work_efficiency_weight, 0.05)

        return float(reward)


# ─── Helper termodinamico (retrocompatibilità) ────────────────────────────────

def should_sell_now(thermo_row: pd.Series) -> bool:
    stress     = thermo_row.get("Thm_Stress",     0.0)
    efficiency = thermo_row.get("Thm_Efficiency", 0.0)
    return float(stress) > 1.0 or float(efficiency) > 1.5


# ─── Ambiente ─────────────────────────────────────────────────────────────────

class TradingEnv(gym.Env):
    """
    Ambiente di trading multi-ticker per DDPG — v4.1 Thermodynamic Reward Gate.

    FIX v4.1: episode metrics inseriti in info["episode"] al momento del done=True,
    prima dell'auto-reset di DummyVecEnv, così il callback li legge correttamente.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        df:                  pd.DataFrame,
        tickers:             List[str],
        initial_capital:     float = 10_000.0,
        max_position_pct:    float = 0.10,
        fee_pct:             float = 0.001,
        prediction_window:   int   = 1,
        thermo_df:           Optional[pd.DataFrame] = None,
        thermo_bonus_sell:   float = 0.5,
        thermo_penalty_buy:  float = 0.1,
        thermo_reward_cfg:   Optional[ThermoRewardConfig] = None,
    ):
        super().__init__()

        self.df               = df.sort_index()
        self._df_values       = self.df.values.astype(np.float32)

        self.tickers          = tickers
        self.n_tickers        = len(tickers)
        self.initial_capital  = initial_capital
        self.max_position_pct = max_position_pct
        self.fee_pct          = fee_pct
        self.prediction_window = prediction_window

        self.thermo_df        = thermo_df
        self._has_thermo      = self.thermo_df is not None and not self.thermo_df.empty
        if self._has_thermo:
            self._thermo_values = self.thermo_df.values.astype(np.float32)
            self._thermo_cols   = {col: i for i, col in enumerate(self.thermo_df.columns)}
        else:
            self._thermo_values = np.array([], dtype=np.float32)
            self._thermo_cols   = {}

        self._reward_gate     = ThermodynamicRewardGate(thermo_reward_cfg)

        self._price_col_idx: dict[str, int] = {}
        df_cols = list(self.df.columns)
        for t in tickers:
            for candidate in (f"{t}_Close", f"{t}_close", "Close", "close", t):
                if candidate in df_cols:
                    self._price_col_idx[t] = df_cols.index(candidate)
                    break
            if t not in self._price_col_idx:
                self._price_col_idx[t] = 0

        # Barre per anno calcolate dai dati (non dalla config)
        diffs = self.df.index.to_series().diff().dt.total_seconds().dropna()
        avg_sec = diffs.median() if not diffs.empty else 120
        self.bars_per_year = max(1, int((252 * 6.5 * 3600) / avg_sec))

        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.n_tickers,), dtype=np.float32
        )

        self.positions         = {t: 0.0 for t in self.tickers}
        self.balance           = self.initial_capital
        self._bars_in_position = {t: 0 for t in self.tickers}
        sample_obs = self._get_state(0)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(len(sample_obs),),
            dtype=np.float32,
        )

        self.state_dim  = self.observation_space.shape[0]
        self.action_dim = self.action_space.shape[0]

        self.reset()

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _price(self, step: int, ticker: str) -> float:
        return float(self._df_values[step, self._price_col_idx[ticker]])

    def _thermo_val(self, col: str, default: float = 0.0) -> float:
        if not self._has_thermo or col not in self._thermo_cols:
            return default
        idx = min(self.current_step, len(self._thermo_values) - 1)
        return float(self._thermo_values[idx, self._thermo_cols[col]])

    # ── Gym API ──────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        if hasattr(super(), 'reset'):
            try:
                super().reset(seed=seed)
            except TypeError:
                super().reset()

        self.current_step       = 0
        self.balance            = self.initial_capital
        self.positions          = {t: 0.0 for t in self.tickers}
        self._bars_in_position  = {t: 0 for t in self.tickers}
        self._portfolio_history = [self.initial_capital]
        self._trades            = []
        self.done               = False
        self._reward_gate.reset()

        obs = self._get_state(self.current_step)
        return obs, {}

    def _get_state(self, step: int) -> np.ndarray:
        obs = self._df_values[step]

        holdings_val = sum(
            self.positions[t] * self._price(step, t) for t in self.tickers
        )
        port_state = np.array([
            self.balance / (self.initial_capital + 1e-8),
            holdings_val / (self.initial_capital + 1e-8),
        ], dtype=np.float32)

        thermo_state = np.array([], dtype=np.float32)
        if self._has_thermo:
            idx = min(step, len(self._thermo_values) - 1)
            thermo_state = self._thermo_values[idx]

        return np.concatenate([obs, port_state, thermo_state])

    def step(self, action: np.ndarray):
        if self.done:
            obs = self._get_state(self.current_step)
            return obs, 0.0, True, False, {}

        prev_value   = self._get_portfolio_value()
        action_types = {}

        for i, ticker in enumerate(self.tickers):
            act   = float(action[i])
            price = self._price(self.current_step, ticker)

            if act > 0.1:
                max_qty = (self.balance * self.max_position_pct) / (price + 1e-8)
                qty     = max_qty * act
                cost    = qty * price * (1 + self.fee_pct)
                if self.balance >= cost and qty > 0:
                    self.balance -= cost
                    self.positions[ticker] += qty
                    self._bars_in_position[ticker] = 0
                    self._trades.append({
                        "step": self.current_step, "ticker": ticker,
                        "type": "BUY", "qty": qty, "price": price,
                    })
                    action_types[ticker] = "buy"

            elif act < -0.1:
                qty_held = self.positions[ticker]
                if qty_held > 0:
                    qty     = qty_held * abs(act)
                    revenue = qty * price * (1 - self.fee_pct)
                    self.balance += revenue
                    self.positions[ticker] -= qty
                    self._bars_in_position[ticker] = 0
                    self._trades.append({
                        "step": self.current_step, "ticker": ticker,
                        "type": "SELL", "qty": qty, "price": price,
                    })
                    action_types[ticker] = "sell"
            else:
                action_types[ticker] = "hold"

        for t in self.tickers:
            if self.positions[t] > 0 and action_types.get(t) == "hold":
                self._bars_in_position[t] += 1

        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            self.done = True

        new_value = self._get_portfolio_value()
        self._portfolio_history.append(new_value)

        reward = self._calculate_reward(prev_value, new_value, action_types)
        obs    = self._get_state(self.current_step)

        # ── FIX v4.1 ─────────────────────────────────────────────────────────
        # Inserisci le metriche di episodio nell'info dict quando done=True,
        # PRIMA che DummyVecEnv chiami env.reset(). Il callback legge da qui.
        # Senza questo fix, il callback leggeva portfolio_history() = [100.0]
        # perché l'env era già stato resettato → Sharpe=0.000 per tutti gli ep.
        info = {"step": self.current_step}
        if self.done:
            info["episode"] = {
                "portfolio_value": new_value,
                "sharpe":          self.sharpe_ratio(),
                "max_drawdown":    self.max_drawdown(),
                "n_trades":        len(self._trades),
            }

        return obs, reward, self.done, False, info

    def _get_portfolio_value(self) -> float:
        value = self.balance
        for t in self.tickers:
            value += self.positions[t] * self._price(self.current_step, t)
        return float(value)

    def _calculate_reward(
        self,
        prev_value:   float,
        new_value:    float,
        action_types: dict[str, str],
    ) -> float:
        base_pnl = (new_value - prev_value) / (prev_value + 1e-8) * 100.0

        thm_z   = self._thermo_val("Thm_Stress",     0.0)
        thm_eff = self._thermo_val("Thm_Efficiency", 0.0)

        all_types = list(action_types.values())
        if "buy"   in all_types: dominant = "buy"
        elif "sell" in all_types: dominant = "sell"
        else:                     dominant = "hold"

        held_bars_list = [
            self._bars_in_position[t]
            for t in self.tickers
            if self.positions[t] > 0
        ]
        held_bars = int(np.mean(held_bars_list)) if held_bars_list else 0

        reward = self._reward_gate.compute(
            base_reward        = base_pnl,
            action_type        = dominant,
            position_held_bars = held_bars,
            thm_z              = thm_z,
            thm_efficiency     = thm_eff,
        )
        return float(np.clip(reward, -10.0, 10.0))

    # ── Adaptive Exploration ──────────────────────────────────────────────────

    def get_current_regime(self) -> float:
        return self._thermo_val("Thm_Regime", 0.0)

    # ── Metriche ─────────────────────────────────────────────────────────────

    def portfolio_history(self) -> np.ndarray:
        return np.array(self._portfolio_history, dtype=np.float64)

    def sharpe_ratio(self) -> float:
        h = self.portfolio_history()
        if len(h) < 2:
            return 0.0
        returns = np.diff(h) / (h[:-1] + 1e-8)
        std     = np.std(returns)
        if len(returns) < 2 or std == 0:
            return 0.0
        return float(np.mean(returns) / std * np.sqrt(self.bars_per_year))

    def max_drawdown(self) -> float:
        h = self.portfolio_history()
        if len(h) < 2:
            return 0.0
        peak = np.maximum.accumulate(h)
        dd   = (h - peak) / (peak + 1e-8)
        return float(np.min(dd))

    def trade_stats(self) -> dict:
        buys  = sum(1 for t in self._trades if t["type"] == "BUY")
        sells = sum(1 for t in self._trades if t["type"] == "SELL")
        return {
            "n_buy":          buys,
            "n_sell":         sells,
            "buy_sell_ratio": buys / max(1, sells),
        }