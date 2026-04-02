"""
modelli/trading_env.py — Ambiente di trading per DDPG

Miglioramenti:
  1. step() completamente vettorizzato (no loop Python su ticker)
  2. log_trades=False durante training (enorme speedup)
  3. Forced sell posizioni stantie (max_holding_steps)
  4. Reward shaping calibrato: penalità concentrazione excess-only + inaction su posizioni aperte
  5. Bonus take-profit vettorizzato
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


class TradingEnv:
    """
    Ambiente episodico di trading.

    Stato per ogni step:
      [prezzi_reali(F) | prezzi_predetti(F) | holdings_ratio(T) |
       cash_ratio(1)   | portfolio_ratio(1) | psi_values(T)]

      dove F = num_features, T = num_tickers
      Se Ψ non è disponibile, psi_values = zeros(T)

    Reward:
      reward = rendimento_pct
             - lambda_concentration * excess_herfindahl
             - lambda_inaction      * frazione_posizioni_ferme
             + sell_bonus           * n_vendite_profittevoli
    """

    def __init__(
        self,
        prices_real:          pd.DataFrame,
        prices_pred:          pd.DataFrame,
        tickers:              list[str],
        initial_capital:      float = 10_000.0,
        transaction_cost:     float = 0.001,
        psi_df:               Optional[pd.DataFrame] = None,
        lambda_concentration: float = 0.1,
        lambda_inaction:      float = 0.05,
        action_threshold:     float = 0.05,
        max_position_pct:     float = 0.10,
        max_holding_steps:    int   = 100,
        log_trades:           bool  = True,
    ) -> None:
        self.prices_real      = prices_real
        self.prices_pred      = prices_pred
        self.tickers          = tickers
        self.all_columns      = list(prices_real.columns)
        self.initial_capital  = initial_capital
        self.transaction_cost = transaction_cost
        self.log_trades       = log_trades

        self.lambda_concentration = lambda_concentration
        self.lambda_inaction      = lambda_inaction
        self.action_threshold     = action_threshold
        self.max_position_pct     = max_position_pct
        self.max_holding_steps    = max_holding_steps

        self.num_features = len(self.all_columns)
        self.num_tickers  = len(tickers)
        self._ticker_idx  = np.array(
            [self.all_columns.index(t) for t in tickers], dtype=np.int32
        )

        # ── Ψ per timestep ────────────────────────────────────────────────
        self._psi: Optional[np.ndarray] = None  # shape (n_steps, n_tickers)
        if psi_df is not None and not psi_df.empty:
            psi_aligned = psi_df.reindex(prices_real.index).ffill().bfill().fillna(0)
            psi_cols = [f"Psi_{t}" for t in tickers if f"Psi_{t}" in psi_aligned.columns]
            if psi_cols:
                self._psi = psi_aligned[psi_cols].values.astype(np.float32)
                print(f"[TradingEnv] Ψ attivo per {len(psi_cols)}/{self.num_tickers} ticker")

        has_psi = self._psi is not None
        self.state_dim  = (
            2 * self.num_features
            + self.num_tickers + 2
            + (self.num_tickers if has_psi else 0)
        )
        self.action_dim = self.num_tickers

        # Cache array numpy dei prezzi per evitare iloc ad ogni step
        self._prices_real_np = prices_real.values.astype(np.float32)
        self._prices_pred_np = prices_pred.values.astype(np.float32)
        self._n_steps        = len(prices_real)
        self._dates          = prices_real.index

        self._reset_internals()

    # ── reset ─────────────────────────────────────────────────────────────

    def _reset_internals(self) -> None:
        self._step       = 0
        self._cash       = float(self.initial_capital)
        self._holdings   = np.zeros(self.num_tickers, dtype=np.float64)
        self._avg_cost   = np.zeros(self.num_tickers, dtype=np.float64)
        self._entry_step = np.full(self.num_tickers, -1, dtype=np.int32)

        self._portfolio_history: list[float] = [self.initial_capital]
        self._trade_log:         list[dict]  = []
        self._value_per_ticker:  list[dict]  = []

    def reset(self) -> np.ndarray:
        self._reset_internals()
        return self._get_state()

    # ── state ─────────────────────────────────────────────────────────────

    def _get_state(self) -> np.ndarray:
        row_real = self._prices_real_np[self._step]   # (F,)
        row_pred = self._prices_pred_np[self._step]   # (F,)

        portfolio_value = self._portfolio_value()
        prices          = self._current_prices()
        holdings_ratio  = (self._holdings * prices) / (portfolio_value + 1e-8)
        cash_ratio      = self._cash / (portfolio_value + 1e-8)
        port_ratio      = portfolio_value / self.initial_capital

        parts: list[np.ndarray] = [
            row_real,
            row_pred,
            holdings_ratio.astype(np.float32),
            np.array([cash_ratio, port_ratio], dtype=np.float32),
        ]

        if self._psi is not None:
            parts.append(self._psi[self._step])

        return np.concatenate(parts).astype(np.float32)

    # ── portfolio helpers ─────────────────────────────────────────────────

    def _current_prices(self) -> np.ndarray:
        return self._prices_real_np[self._step][self._ticker_idx].astype(np.float64)

    def _portfolio_value(self) -> float:
        prices = self._current_prices()
        return float(self._cash + np.dot(self._holdings, prices))

    # ── reward ────────────────────────────────────────────────────────────

    def _compute_reward(
        self,
        prev_value:  float,
        new_value:   float,
        action:      np.ndarray,
        sell_profit_mask: np.ndarray,   # bool array: vendite profittevoli
    ) -> float:
        base_reward = (new_value - prev_value) / (prev_value + 1e-8)

        # Concentrazione: penalizza solo l'eccesso rispetto a portafoglio uniforme
        prices  = self._current_prices()
        weights = (self._holdings * prices) / (new_value + 1e-8)
        herfindahl           = float(np.dot(weights, weights))
        uniform_herfindahl   = 1.0 / self.num_tickers
        excess_concentration = max(0.0, herfindahl - uniform_herfindahl)
        concentration_penalty = self.lambda_concentration * excess_concentration

        # Inaction: penalizza solo posizioni aperte ferme (non stare in cash)
        open_positions = self._holdings > 1e-8
        if open_positions.any():
            frozen = open_positions & (np.abs(action) < self.action_threshold)
            inaction_penalty = self.lambda_inaction * float(frozen.sum()) / max(1, int(open_positions.sum()))
        else:
            inaction_penalty = 0.0

        # Bonus take-profit (vettorizzato — calcolato fuori)
        sell_bonus = 0.0002 * float(sell_profit_mask.sum())

        return base_reward - concentration_penalty - inaction_penalty + sell_bonus

    # ── step ──────────────────────────────────────────────────────────────

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        action     = np.clip(action, -1.0, 1.0)
        prices     = self._current_prices()
        prev_value = self._portfolio_value()
        date       = self._dates[self._step] if self.log_trades else None

        sell_profit_mask = np.zeros(self.num_tickers, dtype=bool)

        # ── VENDITE vettorizzate ──────────────────────────────────────────
        sell_mask = (action < -self.action_threshold) & (self._holdings > 1e-8)
        if sell_mask.any():
            sell_fracs     = np.abs(action[sell_mask])
            shares_sold    = self._holdings[sell_mask] * sell_fracs
            proceeds       = shares_sold * prices[sell_mask]
            fees           = proceeds * self.transaction_cost
            net_proceeds   = proceeds - fees
            realized_pnl   = net_proceeds - shares_sold * self._avg_cost[sell_mask]

            self._cash                += float(net_proceeds.sum())
            self._holdings[sell_mask] -= shares_sold

            # Chiudi posizioni azzerate
            closed = sell_mask.copy()
            closed[sell_mask] = self._holdings[sell_mask] < 1e-8
            self._holdings[closed]    = 0.0
            self._entry_step[closed]  = -1

            # Traccia vendite profittevoli per il bonus
            sell_profit_mask[sell_mask] = realized_pnl > 0

            if self.log_trades:
                sell_indices = np.where(sell_mask)[0]
                for k, i in enumerate(sell_indices):
                    self._trade_log.append({
                        "date":         date,
                        "ticker":       self.tickers[i],
                        "action":       "SELL",
                        "action_signal": round(float(action[i]), 4),
                        "shares":       round(float(shares_sold[k]), 6),
                        "price":        round(float(prices[i]), 4),
                        "gross_value":  round(float(proceeds[k]), 2),
                        "fee":          round(float(fees[k]), 4),
                        "net_value":    round(float(net_proceeds[k]), 2),
                        "avg_cost":     round(float(self._avg_cost[i]), 4),
                        "realized_pnl": round(float(realized_pnl[k]), 2),
                        "holdings_after": round(float(self._holdings[i]), 6),
                        "cash_after":   round(float(self._cash), 2),
                    })

        # ── ACQUISTI vettorizzati ─────────────────────────────────────────
        buy_signals = np.where(action > self.action_threshold, action, 0.0)
        buy_total   = buy_signals.sum() + 1e-8

        if buy_total > self.action_threshold and self._cash > 1e-2:
            portfolio_value    = self._portfolio_value()
            max_pos_val        = portfolio_value * self.max_position_pct
            current_pos_val    = self._holdings * prices
            remaining_capacity = np.maximum(0.0, max_pos_val - current_pos_val)

            budgets    = self._cash * (buy_signals / buy_total)
            budgets    = np.minimum(budgets, remaining_capacity)
            valid_mask = budgets > 1e-2

            if valid_mask.any():
                b           = budgets[valid_mask]
                p           = prices[valid_mask]
                fees_buy    = b * self.transaction_cost
                net_b       = b - fees_buy
                shares_buy  = net_b / (p + 1e-8)

                h_old = self._holdings[valid_mask]
                c_old = self._avg_cost[valid_mask]
                h_new = h_old + shares_buy

                # Aggiornamento avg_cost vettoriale
                self._avg_cost[valid_mask] = np.where(
                    h_new > 1e-8,
                    (h_old * c_old + shares_buy * p) / h_new,
                    c_old,
                )

                # entry_step solo per posizioni nuove
                valid_indices = np.where(valid_mask)[0]
                for k, i in enumerate(valid_indices):
                    if self._entry_step[i] < 0:
                        self._entry_step[i] = self._step

                self._cash                -= float(b.sum())
                self._holdings[valid_mask] += shares_buy

                if self.log_trades:
                    for k, i in enumerate(valid_indices):
                        self._trade_log.append({
                            "date":         date,
                            "ticker":       self.tickers[i],
                            "action":       "BUY",
                            "action_signal": round(float(action[i]), 4),
                            "shares":       round(float(shares_buy[k]), 6),
                            "price":        round(float(p[k]), 4),
                            "gross_value":  round(float(b[k]), 2),
                            "fee":          round(float(fees_buy[k]), 4),
                            "net_value":    round(float(net_b[k]), 2),
                            "avg_cost":     round(float(self._avg_cost[i]), 4),
                            "realized_pnl": 0.0,
                            "holdings_after": round(float(self._holdings[i]), 6),
                            "cash_after":   round(float(self._cash), 2),
                        })

        # ── FORCED SELL posizioni stantie ─────────────────────────────────
        if self._step > 0 and self.max_holding_steps > 0:
            steps_held  = self._step - self._entry_step
            forced_mask = (steps_held >= self.max_holding_steps) & (self._holdings > 1e-8)
            if forced_mask.any():
                proceeds_f  = self._holdings[forced_mask] * prices[forced_mask]
                fees_f      = proceeds_f * self.transaction_cost
                self._cash += float((proceeds_f - fees_f).sum())
                self._holdings[forced_mask]   = 0.0
                self._entry_step[forced_mask] = -1

        # ── Avanza ────────────────────────────────────────────────────────
        self._step += 1
        done        = self._step >= self._n_steps - 1
        new_value   = self._portfolio_value()
        self._portfolio_history.append(new_value)

        # Snapshot per value_per_ticker (solo con log abilitato)
        if self.log_trades:
            new_prices    = self._current_prices()
            unrealized    = self._holdings * (new_prices - self._avg_cost)
            snap: dict    = {
                "date":            date,
                "portfolio_value": round(new_value, 2),
                "cash":            round(self._cash, 2),
            }
            for i, t in enumerate(self.tickers):
                snap[f"{t}_shares"]         = round(float(self._holdings[i]), 6)
                snap[f"{t}_price"]          = round(float(new_prices[i]), 4)
                snap[f"{t}_avg_cost"]       = round(float(self._avg_cost[i]), 4)
                snap[f"{t}_value"]          = round(float(self._holdings[i] * new_prices[i]), 2)
                snap[f"{t}_unrealized_pnl"] = round(float(unrealized[i]), 2)
            self._value_per_ticker.append(snap)

        reward     = self._compute_reward(prev_value, new_value, action, sell_profit_mask)
        next_state = self._get_state() if not done else np.zeros(self.state_dim, np.float32)

        return next_state, float(reward), done, {
            "portfolio_value": new_value,
            "cash":            self._cash,
            "step":            self._step,
        }

    # ── output DataFrame ─────────────────────────────────────────────────

    def trade_log_df(self) -> pd.DataFrame:
        if not self._trade_log:
            return pd.DataFrame()
        return pd.DataFrame(self._trade_log).set_index("date")

    def value_per_ticker_df(self) -> pd.DataFrame:
        if not self._value_per_ticker:
            return pd.DataFrame()
        return pd.DataFrame(self._value_per_ticker).set_index("date")

    def summary_per_ticker(self) -> pd.DataFrame:
        log = self.trade_log_df()
        if log.empty:
            return pd.DataFrame()
        vpdf  = self.value_per_ticker_df()
        rows  = []
        for ticker in self.tickers:
            t_log = log[log["ticker"] == ticker]
            buys  = t_log[t_log["action"] == "BUY"]
            sells = t_log[t_log["action"] == "SELL"]
            realized = float(sells["realized_pnl"].sum()) if "realized_pnl" in sells else 0.0
            unreal_col = f"{ticker}_unrealized_pnl"
            unreal = float(vpdf[unreal_col].iloc[-1]) if (not vpdf.empty and unreal_col in vpdf.columns) else 0.0
            rows.append({
                "ticker":           ticker,
                "n_buy":            len(buys),
                "n_sell":           len(sells),
                "total_bought_usd": round(buys["gross_value"].sum() if "gross_value" in buys else 0.0, 2),
                "total_sold_usd":   round(sells["gross_value"].sum() if "gross_value" in sells else 0.0, 2),
                "total_fees":       round(t_log["fee"].sum() if "fee" in t_log else 0.0, 4),
                "realized_pnl":     round(realized, 2),
                "unrealized_pnl":   round(unreal, 2),
                "total_pnl":        round(realized + unreal, 2),
            })
        return (
            pd.DataFrame(rows)
            .set_index("ticker")
            .sort_values("total_pnl", ascending=False)
        )

    # ── metriche ─────────────────────────────────────────────────────────

    def portfolio_history(self) -> np.ndarray:
        return np.array(self._portfolio_history, dtype=np.float64)

    def sharpe_ratio(self, risk_free: float = 0.0) -> float:
        h = self.portfolio_history()
        if len(h) < 2:
            return 0.0
        r      = np.diff(h) / (h[:-1] + 1e-8)
        excess = r - risk_free / 252
        std    = excess.std()
        return float(excess.mean() / (std + 1e-8) * np.sqrt(252))

    def max_drawdown(self) -> float:
        h    = self.portfolio_history()
        peak = np.maximum.accumulate(h)
        return float(((h - peak) / (peak + 1e-8)).min())