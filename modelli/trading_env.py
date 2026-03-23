"""
modelli/trading_env.py — Ambiente di trading per l'agente DDPG

Stato  : [prezzi_correnti | prezzi_predetti | holdings_ratio | cash_ratio | portfolio_ratio]
Azione : vettore continuo in [-1, 1] per ogni ticker
           +1 = investi tutta la liquidità disponibile sul ticker
           -1 = vendi tutto il ticker
            0 = hold
Reward : variazione percentuale del valore del portafoglio al netto dei costi
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class TradingEnv:
    """
    Ambiente episodico di trading su serie storiche.

    Parameters
    ----------
    prices_real      : DataFrame con prezzi DENORMALIZZATI (index=date, columns=all_features)
    prices_pred      : DataFrame con prezzi PREDETTI denormalizzati (stessa forma)
    tickers          : lista dei ticker negoziabili (sottoinsieme di prices_real.columns)
    initial_capital  : liquidità iniziale in USD
    transaction_cost : percentuale di costo per transazione (es. 0.001 = 0.1%)
    """

    def __init__(
        self,
        prices_real:      pd.DataFrame,
        prices_pred:      pd.DataFrame,
        tickers:          list[str],
        initial_capital:  float = 10_000.0,
        transaction_cost: float = 0.001,
    ) -> None:
        self.prices_real      = prices_real
        self.prices_pred      = prices_pred
        self.tickers          = tickers
        self.all_columns      = list(prices_real.columns)
        self.initial_capital  = initial_capital
        self.transaction_cost = transaction_cost

        self.num_features = len(self.all_columns)
        self.num_tickers  = len(tickers)

        self._ticker_idx = [self.all_columns.index(t) for t in tickers]

        self.state_dim  = 2 * self.num_features + self.num_tickers + 2
        self.action_dim = self.num_tickers

        self._n_steps = len(prices_real)
        self._reset_internals()

    # ── reset ─────────────────────────────────────────────────────────────────

    def _reset_internals(self) -> None:
        self._step     = 0
        self._cash     = float(self.initial_capital)
        self._holdings = np.zeros(self.num_tickers, dtype=np.float32)
        self._avg_cost = np.zeros(self.num_tickers, dtype=np.float32)

        self._portfolio_history: list[float] = [self.initial_capital]
        self._trade_log:         list[dict]  = []
        self._value_per_ticker:  list[dict]  = []

    def reset(self) -> np.ndarray:
        self._reset_internals()
        return self._get_state()

    # ── state ─────────────────────────────────────────────────────────────────

    def _get_state(self) -> np.ndarray:
        row_real = self.prices_real.iloc[self._step].values.astype(np.float32)
        row_pred = self.prices_pred.iloc[self._step].values.astype(np.float32)

        portfolio_value = self._portfolio_value()
        holdings_ratio  = self._holdings_value_per_ticker() / (portfolio_value + 1e-8)
        cash_ratio      = self._cash / (portfolio_value + 1e-8)
        port_ratio      = portfolio_value / self.initial_capital

        return np.concatenate([
            row_real, row_pred, holdings_ratio, [cash_ratio], [port_ratio]
        ]).astype(np.float32)

    # ── portfolio helpers ──────────────────────────────────────────────────────

    def _current_prices(self) -> np.ndarray:
        return self.prices_real.iloc[self._step].values[self._ticker_idx].astype(np.float32)

    def _holdings_value_per_ticker(self) -> np.ndarray:
        return self._holdings * self._current_prices()

    def _holdings_value(self) -> float:
        return float(self._holdings_value_per_ticker().sum())

    def _portfolio_value(self) -> float:
        return self._cash + self._holdings_value()

    # ── step ──────────────────────────────────────────────────────────────────

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        action = np.clip(action, -1.0, 1.0)
        prices = self._current_prices()
        date   = self.prices_real.index[self._step]
        prev_value = self._portfolio_value()

        # ── VENDITE (prima, per liberare liquidità) ────────────────────────────
        for i, a in enumerate(action):
            if a < 0 and self._holdings[i] > 1e-8:
                shares_to_sell = self._holdings[i] * abs(a)
                proceeds       = shares_to_sell * prices[i]
                fee            = proceeds * self.transaction_cost
                net_proceeds   = proceeds - fee
                realized_pnl   = net_proceeds - shares_to_sell * self._avg_cost[i]

                self._cash        += net_proceeds
                self._holdings[i] -= shares_to_sell

                self._trade_log.append({
                    "date":           date,
                    "ticker":         self.tickers[i],
                    "action":         "SELL",
                    "action_signal":  round(float(a), 4),
                    "shares":         round(float(shares_to_sell), 6),
                    "price":          round(float(prices[i]), 4),
                    "gross_value":    round(float(proceeds), 2),
                    "fee":            round(float(fee), 4),
                    "net_value":      round(float(net_proceeds), 2),
                    "avg_cost":       round(float(self._avg_cost[i]), 4),
                    "realized_pnl":   round(float(realized_pnl), 2),
                    "holdings_after": round(float(self._holdings[i]), 6),
                    "cash_after":     round(float(self._cash), 2),
                })

        # ── ACQUISTI ──────────────────────────────────────────────────────────
        buy_signals = np.maximum(action, 0.0)
        buy_total   = buy_signals.sum() + 1e-8

        for i, a in enumerate(action):
            if a > 0 and self._cash > 1e-2:
                budget        = self._cash * (a / buy_total)
                fee           = budget * self.transaction_cost
                net_budget    = budget - fee
                shares_to_buy = net_budget / (prices[i] + 1e-8)

                # aggiorna costo medio di carico
                total_shares = self._holdings[i] + shares_to_buy
                if total_shares > 1e-8:
                    self._avg_cost[i] = (
                        self._holdings[i] * self._avg_cost[i]
                        + shares_to_buy   * prices[i]
                    ) / total_shares

                self._cash        -= budget
                self._holdings[i] += shares_to_buy

                self._trade_log.append({
                    "date":           date,
                    "ticker":         self.tickers[i],
                    "action":         "BUY",
                    "action_signal":  round(float(a), 4),
                    "shares":         round(float(shares_to_buy), 6),
                    "price":          round(float(prices[i]), 4),
                    "gross_value":    round(float(budget), 2),
                    "fee":            round(float(fee), 4),
                    "net_value":      round(float(net_budget), 2),
                    "avg_cost":       round(float(self._avg_cost[i]), 4),
                    "realized_pnl":   0.0,
                    "holdings_after": round(float(self._holdings[i]), 6),
                    "cash_after":     round(float(self._cash), 2),
                })

        # ── HOLD (|a| < 0.05) ────────────────────────────────────────────────
        for i, a in enumerate(action):
            if abs(a) < 0.05:
                self._trade_log.append({
                    "date":           date,
                    "ticker":         self.tickers[i],
                    "action":         "HOLD",
                    "action_signal":  round(float(a), 4),
                    "shares":         0.0,
                    "price":          round(float(prices[i]), 4),
                    "gross_value":    0.0,
                    "fee":            0.0,
                    "net_value":      0.0,
                    "avg_cost":       round(float(self._avg_cost[i]), 4),
                    "realized_pnl":   0.0,
                    "holdings_after": round(float(self._holdings[i]), 6),
                    "cash_after":     round(float(self._cash), 2),
                })

        # ── avanza ────────────────────────────────────────────────────────────
        self._step += 1
        done = self._step >= self._n_steps - 1

        new_value      = self._portfolio_value()
        new_prices     = self._current_prices()
        unrealized_pnl = self._holdings * (new_prices - self._avg_cost)

        self._portfolio_history.append(new_value)

        # snapshot giornaliero per ticker
        snap = {
            "date":            date,
            "portfolio_value": round(new_value, 2),
            "cash":            round(self._cash, 2),
        }
        for i, t in enumerate(self.tickers):
            snap[f"{t}_shares"]         = round(float(self._holdings[i]), 6)
            snap[f"{t}_price"]          = round(float(new_prices[i]), 4)
            snap[f"{t}_avg_cost"]       = round(float(self._avg_cost[i]), 4)
            snap[f"{t}_value"]          = round(float(self._holdings[i] * new_prices[i]), 2)
            snap[f"{t}_unrealized_pnl"] = round(float(unrealized_pnl[i]), 2)
        self._value_per_ticker.append(snap)

        reward = (new_value - prev_value) / (prev_value + 1e-8)
        next_state = self._get_state() if not done else np.zeros(self.state_dim, np.float32)

        return next_state, float(reward), done, {
            "portfolio_value": new_value,
            "cash":            self._cash,
            "step":            self._step,
        }

    # ── DataFrame di output ───────────────────────────────────────────────────

    def trade_log_df(self) -> pd.DataFrame:
        """
        Ogni riga = una singola operazione (BUY / SELL / HOLD).

        Colonne principali:
          ticker, action, action_signal, shares, price,
          gross_value, fee, net_value, avg_cost,
          realized_pnl, holdings_after, cash_after
        """
        if not self._trade_log:
            return pd.DataFrame()
        return pd.DataFrame(self._trade_log).set_index("date")

    def value_per_ticker_df(self) -> pd.DataFrame:
        """
        Snapshot giornaliero: per ogni ticker → quote, prezzo, valore, P&L non realizzato.
        """
        if not self._value_per_ticker:
            return pd.DataFrame()
        return pd.DataFrame(self._value_per_ticker).set_index("date")

    def summary_per_ticker(self) -> pd.DataFrame:
        """
        Riepilogo finale per ticker ordinato per P&L totale decrescente.

        Colonne:
          n_buy, n_sell, total_bought_usd, total_sold_usd,
          total_fees, realized_pnl, unrealized_pnl, total_pnl
        """
        log = self.trade_log_df()
        if log.empty:
            return pd.DataFrame()

        vpdf = self.value_per_ticker_df()
        rows = []

        for ticker in self.tickers:
            t_log = log[log["ticker"] == ticker]
            buys  = t_log[t_log["action"] == "BUY"]
            sells = t_log[t_log["action"] == "SELL"]

            realized = sells["realized_pnl"].sum()
            unreal   = (
                float(vpdf[f"{ticker}_unrealized_pnl"].iloc[-1])
                if not vpdf.empty else 0.0
            )

            rows.append({
                "ticker":           ticker,
                "n_buy":            len(buys),
                "n_sell":           len(sells),
                "total_bought_usd": round(buys["gross_value"].sum(), 2),
                "total_sold_usd":   round(sells["gross_value"].sum(), 2),
                "total_fees":       round(t_log["fee"].sum(), 4),
                "realized_pnl":     round(realized, 2),
                "unrealized_pnl":   round(unreal, 2),
                "total_pnl":        round(realized + unreal, 2),
            })

        return (
            pd.DataFrame(rows)
            .set_index("ticker")
            .sort_values("total_pnl", ascending=False)
        )

    # ── metriche aggregate ─────────────────────────────────────────────────────

    def portfolio_history(self) -> np.ndarray:
        return np.array(self._portfolio_history)

    def sharpe_ratio(self, risk_free: float = 0.0) -> float:
        h = self.portfolio_history()
        if len(h) < 2:
            return 0.0
        r = np.diff(h) / (h[:-1] + 1e-8)
        excess = r - risk_free / 252
        return float(excess.mean() / (excess.std() + 1e-8) * np.sqrt(252))

    def max_drawdown(self) -> float:
        h    = self.portfolio_history()
        peak = np.maximum.accumulate(h)
        return float(((h - peak) / (peak + 1e-8)).min())