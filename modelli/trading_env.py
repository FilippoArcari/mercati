"""
modelli/trading_env.py — Ambiente di trading per DDPG con Integrazione Termodinamica Avanzata
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import gym
from gym import spaces
from typing import Optional, List, Dict



class TradingEnv(gym.Env):
    """
    Ambiente di trading multi-ticker per DDPG.
    
    Integrazioni v3.0:
    - Reward Shaping basato su Work-Price Efficiency.
    - Penality/Bonus dinamici basati sul Regime termodinamico.
    - Supporto per Adaptive Exploration in DDPG.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tickers: List[str],
        initial_capital: float = 10000.0,
        max_position_pct: float = 0.10,
        fee_pct: float = 0.001,
        prediction_window: int = 1,
        thermo_df: Optional[pd.DataFrame] = None,
        # Parametri Reward Shaping
        thermo_bonus_sell: float = 0.5,
        thermo_penalty_buy: float = 0.1,
    ):
        super().__init__()

        self.df = df.sort_index()
        self.tickers = tickers
        self.n_tickers = len(tickers)
        self.initial_capital = initial_capital
        self.max_position_pct = max_position_pct
        self.fee_pct = fee_pct
        self.prediction_window = prediction_window
        self.thermo_df = thermo_df
        
        # Nuovi parametri per bilanciamento Buy/Sell
        self.thermo_bonus_sell = thermo_bonus_sell
        self.thermo_penalty_buy = thermo_penalty_buy

        # Calcolo barre per anno (per Sharpe)
        diffs = self.df.index.to_series().diff().dt.total_seconds().dropna()
        if not diffs.empty:
            avg_sec = diffs.median()
            # 252 giorni * 6.5 ore (NYSE) / avg_sec
            self.bars_per_year = int((252 * 6.5 * 3600) / avg_sec)
        else:
            self.bars_per_year = 252

        # Spazi Gym
        # Action: [-1, 1] per ogni ticker. -1 (Sell), 0 (Hold), 1 (Buy)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.n_tickers,), dtype=np.float32)
        
        # Observation space dimension
        # (Tickers * Features) + Portfolio features + Thermo features
        sample_state = self._get_state(0)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(sample_state),), dtype=np.float32
        )

        self.reset()

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_capital
        self.positions = {t: 0.0 for t in self.tickers}
        self.costs = {t: 0.0 for t in self.tickers}
        self._portfolio_history = [self.initial_capital]
        self._trades = []
        self.done = False
        return self._get_state(self.current_step)

    def _get_state(self, step: int) -> np.ndarray:
        """Costruisce il vettore di osservazione."""
        # 1. Feature di prezzo/indicatori
        obs = self.df.iloc[step].values.astype(np.float32)

        # 2. Stato del portafoglio (normalizzato)
        port_state = [
            self.balance / self.initial_capital,
            sum(self.positions.values()) * self.df.iloc[step][f"{self.tickers[0]}_Close"] / self.initial_capital if self.positions[self.tickers[0]] > 0 else 0.0
        ]
        
        # 3. Feature Termodinamiche (se presenti)
        thermo_state = []
        if self.thermo_df is not None:
            thermo_state = self.thermo_df.iloc[step].values.astype(np.float32)

        return np.concatenate([obs, port_state, thermo_state])

    def step(self, action: np.ndarray):
        if self.done:
            return self._get_state(self.current_step), 0.0, True, {}

        prev_value = self._get_portfolio_value()
        
        # Esecuzione ordini
        for i, ticker in enumerate(self.tickers):
            act = action[i]
            price = self.df.iloc[self.current_step][f"{ticker}_Close"]
            
            if act > 0.1: # Buy
                max_buy = (self.balance * self.max_position_pct) / price
                qty = max_buy * act
                cost = qty * price * (1 + self.fee_pct)
                if self.balance >= cost:
                    self.balance -= cost
                    self.positions[ticker] += qty
                    self._trades.append({"step": self.current_step, "ticker": ticker, "type": "BUY", "qty": qty, "price": price})
            
            elif act < -0.1: # Sell
                if self.positions[ticker] > 0:
                    qty = self.positions[ticker] * abs(act)
                    revenue = qty * price * (1 - self.fee_pct)
                    self.balance += revenue
                    self.positions[ticker] -= qty
                    self._trades.append({"step": self.current_step, "ticker": ticker, "type": "SELL", "qty": qty, "price": price})

        # Avanzamento
        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            self.done = True

        new_value = self._get_portfolio_value()
        self._portfolio_history.append(new_value)

        # Calcolo Reward con shaping termodinamico
        reward = self._calculate_reward(action, prev_value, new_value)

        return self._get_state(self.current_step), reward, self.done, {"step": self.current_step}

    def _get_portfolio_value(self) -> float:
        value = self.balance
        for t in self.tickers:
            price = self.df.iloc[self.current_step][f"{t}_Close"]
            value += self.positions[t] * price
        return float(value)

    def _calculate_reward(self, action: np.ndarray, prev_value: float, new_value: float) -> float:
        """
        Calcola il reward combinando P&L logaritmico e segnali termodinamici.
        """
        # Reward Base: variazione percentuale del portafoglio
        base_reward = (new_value - prev_value) / (prev_value + 1e-8)
        reward = base_reward * 100 # Scalato per stabilità

        # --- REWARD SHAPING TERMODINAMICO ---
        if self.thermo_df is not None:
            current_thermo = self.thermo_df.iloc[self.current_step]
            
            # 1. Identifica se la termodinamica suggerisce di vendere (Efficienza bassa/Stress alto)
            thermo_indicates_sell = should_sell_now(current_thermo)
            avg_action = np.mean(action)

            # BONUS: Vendita corretta in fase di esaurimento energetico
            if avg_action < -0.3 and thermo_indicates_sell:
                reward += self.thermo_bonus_sell
            
            # PENALITÀ: Vendita prematura durante un Rally Reale (Regime 1)
            elif avg_action < -0.3 and not thermo_indicates_sell:
                if current_thermo.get('Thm_Regime', 0) == 1.0:
                    reward -= 0.2
            
            # PENALITÀ: Buy o Hold quando il mercato è "esausto" (Efficienza > 1.5)
            elif avg_action > -0.1 and thermo_indicates_sell:
                reward -= self.thermo_penalty_buy

        return float(reward)

    def get_current_regime(self) -> float:
        """Ritorna il regime termodinamico corrente per l'esplorazione adattiva."""
        if self.thermo_df is not None:
            return float(self.thermo_df.iloc[self.current_step].get('Thm_Regime', 0.0))
        return 0.0

    # ── Metriche di Performance ──────────────────────────────────────────

    def portfolio_history(self) -> np.ndarray:
        return np.array(self._portfolio_history, dtype=np.float64)

    def sharpe_ratio(self) -> float:
        h = self.portfolio_history()
        if len(h) < 2: return 0.0
        returns = np.diff(h) / (h[:-1] + 1e-8)
        if len(returns) < 2 or np.std(returns) == 0: return 0.0
        return float(np.mean(returns) / np.std(returns) * np.sqrt(self.bars_per_year))

    def max_drawdown(self) -> float:
        h = self.portfolio_history()
        if len(h) < 2: return 0.0
        peak = np.maximum.accumulate(h)
        dd = (h - peak) / (peak + 1e-8)
        return float(np.min(dd))