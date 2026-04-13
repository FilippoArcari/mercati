"""
modelli/trading_env.py — Ambiente di trading per DDPG con Reward Gate Termodinamico
=====================================================================================

FIX v5.0 — BUY/SELL IMBALANCE:
  • [Fix B1] TradingEnv.__init__ ora accetta lambda_inaction, lambda_concentration,
             sell_profit_bonus, max_holding_steps, forced_sell_cooldown_steps,
             lambda_imbalance — precedentemente definiti nel YAML ma mai passati
             all'ambiente (bug silenzioso che rendeva tutto il reward shaping inattivo).
  • [Fix B2] sell_mult_expansion: 0.9 → 1.2 nel default ThermoRewardConfig.
             Con 0.9 vendere in regime espansivo era penalizzato rispetto a comprare,
             insegnando all'agente a non vendere mai.
  • [Fix B3] Forced sell implementato in step(): quando _bars_in_position[t] >= max_holding_steps
             la posizione viene liquidata forzatamente.
  • [Fix B4] Buy/sell imbalance penalty nel reward: penalità proporzionale al ratio
             episodico cumulative_buys / max(1, cumulative_sells) quando supera
             imbalance_threshold (default 2.0).
  • [Fix B5] Holding decay applicato sul ticker peggiore (max), non sulla media.
  • [Fix B6] Sell profit bonus proporzionale al gain realizzato.

FIX v4.1:
  • episode metrics inseriti in info["episode"] al momento del done=True.
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
    # [Fix B2] Era 0.9: penalizzava le vendite in regime espansivo.
    # Ora 1.2: vendere in espansione è leggermente premiato.
    sell_mult_expansion: float = 1.2
    holding_decay_start: int   = 20
    holding_decay_rate:  float = 0.005   # alzato da 0.003
    holding_decay_max:   float = 0.25    # alzato da 0.15
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
                mult = cfg.sell_mult_expansion  # [Fix B2] ora 1.2

        reward = base_reward * mult

        # [Fix B5] holding_decay sul peggior ticker (max), non media
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
    Ambiente di trading multi-ticker per DDPG — v5.0 Buy/Sell Balance.

    Parametri reward shaping (tutti ora effettivamente usati):
      lambda_inaction        — penalità per hold eccessivo (per ticker in posizione)
      lambda_concentration   — penalità quando troppi ticker sono aperti contemporaneamente
      sell_profit_bonus      — bonus per vendite profittevoli (proporzionale al gain %)
      max_holding_steps      — barre massime prima del forced sell
      forced_sell_cooldown   — barre di cooldown post forced-sell
      lambda_imbalance       — penalità per ratio buy/sell episodico > imbalance_threshold
      imbalance_threshold    — ratio buy/sell oltre cui scatta la penalità (default 2.0)
      concentration_threshold — frazione di ticker aperti oltre cui scatta penalità
      reward_clip            — clip simmetrico della reward finale
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        df:                      pd.DataFrame,
        tickers:                 List[str],
        initial_capital:         float = 10_000.0,
        max_position_pct:        float = 0.10,
        fee_pct:                 float = 0.001,
        prediction_window:       int   = 1,
        thermo_df:               Optional[pd.DataFrame] = None,
        thermo_bonus_sell:       float = 0.5,
        thermo_penalty_buy:      float = 0.1,
        thermo_reward_cfg:       Optional[ThermoRewardConfig] = None,
        # ── [Fix B1] Parametri ora effettivamente passati e usati ────────────
        lambda_inaction:         float = 0.2,
        lambda_concentration:    float = 0.6,
        sell_profit_bonus:       float = 0.015,
        max_holding_steps:       int   = 60,
        forced_sell_cooldown:    int   = 10,
        lambda_imbalance:        float = 0.3,
        imbalance_threshold:     float = 2.0,
        concentration_threshold: float = 0.4,
        reward_clip:             float = 10.0,
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

        self._reward_gate = ThermodynamicRewardGate(thermo_reward_cfg)

        # ── Salva tutti i parametri reward shaping ───────────────────────────
        self.lambda_inaction         = lambda_inaction
        self.lambda_concentration    = lambda_concentration
        self.sell_profit_bonus       = sell_profit_bonus
        self.max_holding_steps       = max_holding_steps
        self.forced_sell_cooldown    = forced_sell_cooldown
        self.lambda_imbalance        = lambda_imbalance
        self.imbalance_threshold     = imbalance_threshold
        self.concentration_threshold = concentration_threshold
        self.reward_clip             = reward_clip

        self._price_col_idx: dict[str, int] = {}
        df_cols = list(self.df.columns)
        for t in tickers:
            for candidate in (f"{t}_Close", f"{t}_close", "Close", "close", t):
                if candidate in df_cols:
                    self._price_col_idx[t] = df_cols.index(candidate)
                    break
            if t not in self._price_col_idx:
                self._price_col_idx[t] = 0

        diffs = self.df.index.to_series().diff().dt.total_seconds().dropna()
        avg_sec = diffs.median() if not diffs.empty else 120
        self.bars_per_year = max(1, int((252 * 6.5 * 3600) / avg_sec))

        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.n_tickers,), dtype=np.float32
        )

        self.positions           = {t: 0.0 for t in self.tickers}
        self.balance             = self.initial_capital
        self._bars_in_position   = {t: 0 for t in self.tickers}
        self._cooldown           = {t: 0 for t in self.tickers}
        self._avg_cost           = {t: 0.0 for t in self.tickers}
        self._cumulative_buys    = 0
        self._cumulative_sells   = 0

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

        self.current_step        = 0
        self.balance             = self.initial_capital
        self.positions           = {t: 0.0 for t in self.tickers}
        self._bars_in_position   = {t: 0 for t in self.tickers}
        self._cooldown           = {t: 0 for t in self.tickers}
        self._avg_cost           = {t: 0.0 for t in self.tickers}
        self._portfolio_history  = [self.initial_capital]
        self._trades             = []
        self.done                = False
        self._cumulative_buys    = 0
        self._cumulative_sells   = 0
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

        # ── [Fix B3] Forced sell se posizione troppo vecchia ─────────────────
        # Liquida prima di processare le azioni deliberate, così il balance
        # liberato è disponibile per eventuali nuovi acquisti nello stesso step.
        for ticker in self.tickers:
            if (self.positions[ticker] > 0
                    and self._bars_in_position[ticker] >= self.max_holding_steps):
                price   = self._price(self.current_step, ticker)
                qty     = self.positions[ticker]
                revenue = qty * price * (1 - self.fee_pct)
                self.balance += revenue
                self.positions[ticker] = 0.0
                self._bars_in_position[ticker] = 0
                self._cooldown[ticker] = self.forced_sell_cooldown
                self._avg_cost[ticker] = 0.0
                self._trades.append({
                    "step": self.current_step, "ticker": ticker,
                    "type": "SELL", "qty": qty, "price": price,
                    "forced": True,
                })
                self._cumulative_sells += 1

        # ── Azioni deliberate dell'agente ────────────────────────────────────
        for i, ticker in enumerate(self.tickers):
            act   = float(action[i])
            price = self._price(self.current_step, ticker)

            if act > 0.1:
                # Nessun rebuy durante cooldown post forced-sell
                if self._cooldown[ticker] > 0:
                    action_types[ticker] = "hold"
                    continue
                max_qty = (self.balance * self.max_position_pct) / (price + 1e-8)
                qty     = max_qty * act
                cost    = qty * price * (1 + self.fee_pct)
                if self.balance >= cost and qty > 0:
                    old_qty  = self.positions[ticker]
                    old_cost = self._avg_cost[ticker]
                    new_qty  = old_qty + qty
                    # Costo medio ponderato
                    self._avg_cost[ticker] = (
                        (old_qty * old_cost + qty * price) / (new_qty + 1e-8)
                    )
                    self.balance -= cost
                    self.positions[ticker] = new_qty
                    self._bars_in_position[ticker] = 0
                    self._trades.append({
                        "step": self.current_step, "ticker": ticker,
                        "type": "BUY", "qty": qty, "price": price,
                        "forced": False,
                    })
                    action_types[ticker] = "buy"
                    self._cumulative_buys += 1
                else:
                    action_types[ticker] = "hold"

            elif act < -0.1:
                qty_held = self.positions[ticker]
                if qty_held > 0:
                    qty     = qty_held * abs(act)
                    revenue = qty * price * (1 - self.fee_pct)
                    self.balance += revenue
                    self.positions[ticker] -= qty
                    if self.positions[ticker] < 1e-9:
                        self.positions[ticker] = 0.0
                        self._avg_cost[ticker] = 0.0
                    self._bars_in_position[ticker] = 0
                    self._trades.append({
                        "step": self.current_step, "ticker": ticker,
                        "type": "SELL", "qty": qty, "price": price,
                        "avg_cost": self._avg_cost.get(ticker, price),
                        "forced": False,
                    })
                    action_types[ticker] = "sell"
                    self._cumulative_sells += 1
                else:
                    action_types[ticker] = "hold"
            else:
                action_types[ticker] = "hold"

        # ── Aggiorna contatori ───────────────────────────────────────────────
        for t in self.tickers:
            if self._cooldown[t] > 0:
                self._cooldown[t] -= 1
            if self.positions[t] > 0 and action_types.get(t) == "hold":
                self._bars_in_position[t] += 1

        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            self.done = True

        new_value = self._get_portfolio_value()
        self._portfolio_history.append(new_value)

        reward = self._calculate_reward(prev_value, new_value, action_types)
        obs    = self._get_state(self.current_step)

        info = {"step": self.current_step}
        if self.done:
            info["episode_metrics"] = {
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

        # [Fix B5] Max bars_in_position invece della media
        # La media si abbassava con nuovi acquisti mascherando le posizioni vecchie.
        held_bars_list = [
            self._bars_in_position[t]
            for t in self.tickers
            if self.positions[t] > 0
        ]
        held_bars_max = int(max(held_bars_list)) if held_bars_list else 0

        reward = self._reward_gate.compute(
            base_reward        = base_pnl,
            action_type        = dominant,
            position_held_bars = held_bars_max,
            thm_z              = thm_z,
            thm_efficiency     = thm_eff,
        )

        # ── [Fix B1] Penalità inaction ────────────────────────────────────────
        # Per ogni ticker in posizione che non si muove, sottrai un piccolo costo.
        # Rende l'hold non gratuito: l'agente deve decidere attivamente se tenere.
        n_holding = sum(
            1 for t in self.tickers
            if self.positions[t] > 0 and action_types.get(t) == "hold"
        )
        if n_holding > 0 and dominant == "hold":
            reward -= self.lambda_inaction * n_holding * 0.01

        # ── [Fix B1] Penalità concentrazione ─────────────────────────────────
        # Troppi ticker aperti contemporaneamente = capitale immobilizzato.
        n_open      = sum(1 for t in self.tickers if self.positions[t] > 0)
        conc_thresh = max(1, int(self.n_tickers * self.concentration_threshold))
        if n_open > conc_thresh:
            excess_conc = (n_open - conc_thresh) / max(1, self.n_tickers)
            reward -= self.lambda_concentration * excess_conc

        # ── [Fix B4] Penalità imbalance buy/sell episodico ────────────────────
        # Accumula un contatore per episodio. Se il ratio cumulative_buys/sells
        # supera imbalance_threshold, scala una penalità proporzionale all'eccesso.
        # Cresce durante l'episodio: prima vendi, minore la penalità totale.
        if self._cumulative_sells == 0 and self._cumulative_buys > 10:
            # Nessuna vendita dopo 10+ acquisti: penalità fissa
            reward -= self.lambda_imbalance
        elif self._cumulative_sells > 0:
            current_ratio = self._cumulative_buys / self._cumulative_sells
            if current_ratio > self.imbalance_threshold:
                excess_ratio  = (current_ratio - self.imbalance_threshold) / self.imbalance_threshold
                imbalance_pen = self.lambda_imbalance * min(excess_ratio, 1.0)
                reward -= imbalance_pen

        # ── [Fix B6] Sell profit bonus proporzionale al gain ─────────────────
        # Il bonus cresce con il gain realizzato (max 3x sell_profit_bonus).
        # Incentiva a vendere in profitto invece di accumulare perdite latenti.
        if dominant == "sell":
            for t in self.tickers:
                if action_types.get(t) == "sell":
                    price    = self._price(self.current_step, t)
                    avg_cost = self._avg_cost.get(t, price)
                    gain_pct = (price - avg_cost) / (avg_cost + 1e-8)
                    if gain_pct > 0:
                        bonus = self.sell_profit_bonus * (1.0 + min(gain_pct * 10, 2.0))
                        reward += bonus

        return float(np.clip(reward, -self.reward_clip, self.reward_clip))

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
        buys   = sum(1 for t in self._trades if t["type"] == "BUY")
        sells  = sum(1 for t in self._trades if t["type"] == "SELL")
        forced = sum(1 for t in self._trades if t.get("forced", False))
        return {
            "n_buy":          buys,
            "n_sell":         sells,
            "n_forced_sell":  forced,
            "buy_sell_ratio": buys / max(1, sells),
        }