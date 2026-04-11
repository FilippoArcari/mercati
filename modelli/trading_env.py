"""
modelli/trading_env.py — Ambiente di trading per DDPG con Integrazione Termodinamica Avanzata
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import gym
from gym import spaces
from typing import Optional, List


# ─── Helper termodinamico ─────────────────────────────────────────────────────

def should_sell_now(thermo_row: pd.Series) -> bool:
    """
    Ritorna True se le condizioni termodinamiche suggeriscono una vendita.
    Logica: stress alto (Z > 1) oppure efficienza di lavoro eccessiva (> 1.5).
    """
    stress     = thermo_row.get("Thm_Stress",     0.0)
    efficiency = thermo_row.get("Thm_Efficiency", 0.0)
    return float(stress) > 1.0 or float(efficiency) > 1.5


# ─── Ambiente ────────────────────────────────────────────────────────────────

class TradingEnv(gym.Env):
    """
    Ambiente di trading multi-ticker per DDPG.

    Integrazioni v3.1:
    - Reward shaping basato su Work-Price Efficiency.
    - Penalità/Bonus dinamici basati sul Regime termodinamico.
    - Supporto per Adaptive Exploration in DDPG.
    - Accesso robusto alle colonne prezzo (Close / _Close / ticker nudo).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tickers: List[str],
        initial_capital: float = 10_000.0,
        max_position_pct: float = 0.10,
        fee_pct: float = 0.001,
        prediction_window: int = 1,
        thermo_df: Optional[pd.DataFrame] = None,
        # Parametri Reward Shaping
        thermo_bonus_sell: float = 0.5,
        thermo_penalty_buy: float = 0.1,
    ):
        super().__init__()

        self.df               = df.sort_index()
        self.tickers          = tickers
        self.n_tickers        = len(tickers)
        self.initial_capital  = initial_capital
        self.max_position_pct = max_position_pct
        self.fee_pct          = fee_pct
        self.prediction_window = prediction_window
        self.thermo_df        = thermo_df
        self.thermo_bonus_sell  = thermo_bonus_sell
        self.thermo_penalty_buy = thermo_penalty_buy

        # Mappa ticker → colonna prezzo esistente nel df
        self._price_col: dict[str, str] = {}
        for t in tickers:
            for candidate in (f"{t}_Close", f"{t}_close", "Close", "close", t):
                if candidate in self.df.columns:
                    self._price_col[t] = candidate
                    break
            if t not in self._price_col:
                # Fallback: prima colonna numerica
                self._price_col[t] = self.df.columns[0]

        # Barre per anno (per Sharpe annualizzato)
        diffs = self.df.index.to_series().diff().dt.total_seconds().dropna()
        if not diffs.empty:
            avg_sec = diffs.median()
            self.bars_per_year = max(1, int((252 * 6.5 * 3600) / avg_sec))
        else:
            self.bars_per_year = 252

        # Spazi Gym
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.n_tickers,), dtype=np.float32
        )

        # Calcola dimensione osservazione su uno stato campione
        self.positions = {t: 0.0 for t in self.tickers}
        self.balance   = self.initial_capital
        sample_obs     = self._get_state(0)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(len(sample_obs),),
            dtype=np.float32,
        )

        # Attributo esposto per compatibilità con train.py
        self.state_dim  = self.observation_space.shape[0]
        self.action_dim = self.action_space.shape[0]

        self.reset()

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _price(self, step: int, ticker: str) -> float:
        return float(self.df.iloc[step][self._price_col[ticker]])

    # ── Gym API ──────────────────────────────────────────────────────────────

    def reset(self):
        self.current_step       = 0
        self.balance            = self.initial_capital
        self.positions          = {t: 0.0 for t in self.tickers}
        self.costs              = {t: 0.0 for t in self.tickers}
        self._portfolio_history = [self.initial_capital]
        self._trades            = []
        self.done               = False
        return self._get_state(self.current_step)

    def _get_state(self, step: int) -> np.ndarray:
        """Costruisce il vettore di osservazione."""
        # 1. Feature di prezzo / indicatori tecnici
        obs = self.df.iloc[step].values.astype(np.float32)

        # 2. Stato portafoglio normalizzato
        first_price = self._price(step, self.tickers[0])
        holdings_val = sum(
            self.positions[t] * self._price(step, t) for t in self.tickers
        )
        port_state = np.array([
            self.balance / (self.initial_capital + 1e-8),
            holdings_val / (self.initial_capital + 1e-8),
        ], dtype=np.float32)

        # 3. Feature termodinamiche (se presenti)
        thermo_state = np.array([], dtype=np.float32)
        if self.thermo_df is not None:
            idx = min(step, len(self.thermo_df) - 1)
            thermo_state = self.thermo_df.iloc[idx].values.astype(np.float32)

        return np.concatenate([obs, port_state, thermo_state])

    def step(self, action: np.ndarray):
        if self.done:
            return self._get_state(self.current_step), 0.0, True, {}

        prev_value = self._get_portfolio_value()

        # ── Esecuzione ordini ─────────────────────────────────────────────
        for i, ticker in enumerate(self.tickers):
            act   = float(action[i])
            price = self._price(self.current_step, ticker)

            if act > 0.1:   # Buy
                max_qty = (self.balance * self.max_position_pct) / (price + 1e-8)
                qty     = max_qty * act
                cost    = qty * price * (1 + self.fee_pct)
                if self.balance >= cost and qty > 0:
                    self.balance -= cost
                    self.positions[ticker] += qty
                    self._trades.append({
                        "step": self.current_step, "ticker": ticker,
                        "type": "BUY", "qty": qty, "price": price,
                    })

            elif act < -0.1:    # Sell
                qty_held = self.positions[ticker]
                if qty_held > 0:
                    qty     = qty_held * abs(act)
                    revenue = qty * price * (1 - self.fee_pct)
                    self.balance += revenue
                    self.positions[ticker] -= qty
                    self._trades.append({
                        "step": self.current_step, "ticker": ticker,
                        "type": "SELL", "qty": qty, "price": price,
                    })

        # ── Avanzamento ───────────────────────────────────────────────────
        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            self.done = True

        new_value = self._get_portfolio_value()
        self._portfolio_history.append(new_value)

        reward = self._calculate_reward(action, prev_value, new_value)
        return self._get_state(self.current_step), reward, self.done, {"step": self.current_step}

    def _get_portfolio_value(self) -> float:
        value = self.balance
        for t in self.tickers:
            value += self.positions[t] * self._price(self.current_step, t)
        return float(value)

    def _calculate_reward(
        self, action: np.ndarray, prev_value: float, new_value: float
    ) -> float:
        """Reward combinato: P&L logaritmico + shaping termodinamico."""
        base_reward = (new_value - prev_value) / (prev_value + 1e-8)
        reward      = base_reward * 100.0   # scala per stabilità numerica

        if self.thermo_df is not None:
            idx            = min(self.current_step, len(self.thermo_df) - 1)
            current_thermo = self.thermo_df.iloc[idx]
            sell_signal    = should_sell_now(current_thermo)
            avg_action     = float(np.mean(action))

            if avg_action < -0.3 and sell_signal:
                # Vendita corretta in fase di esaurimento energetico
                reward += self.thermo_bonus_sell
            elif avg_action < -0.3 and not sell_signal:
                # Vendita prematura durante rally reale
                if float(current_thermo.get("Thm_Regime", 0)) == 1.0:
                    reward -= 0.2
            elif avg_action > -0.1 and sell_signal:
                # Hold/Buy quando il mercato è esausto
                reward -= self.thermo_penalty_buy

        return float(reward)

    # ── Adaptive Exploration ──────────────────────────────────────────────────

    def get_current_regime(self) -> float:
        """Regime termodinamico corrente per l'esplorazione adattiva in DDPG."""
        if self.thermo_df is not None:
            idx = min(self.current_step, len(self.thermo_df) - 1)
            return float(self.thermo_df.iloc[idx].get("Thm_Regime", 0.0))
        return 0.0

    # ── Metriche di performance ───────────────────────────────────────────────

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