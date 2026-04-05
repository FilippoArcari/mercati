"""
modelli/trading_env.py — Ambiente di trading per DDPG

Fix applicati:
  [Fix #1] base_reward clippato e denominatore minimo → impedisce esplosioni quando prev_value ≈ 0
  [Fix #2] Cooldown post-forced-sell — impedisce il rebuy immediato dopo espulsione
  [Fix #3] sell_bonus usa self.sell_profit_bonus invece di costante hardcodata
  [Fix #4] Forced sell / pricing: new_value calcolato PRIMA di self._step += 1 (stesso frame di prezzi)
  [Fix #5] Trust thermo: soglia binaria → peso continuo lineare
  [Fix #6] Sharpe ratio annualizzato con bars_per_year
  [Fix #7] Inaction threshold separato da action_threshold
  [Fix #8] next_state usa _get_state() anche quando done (no più zero-vector artificiale)
  [Fix #9] Off-by-one thermo: _thermo_at(step) sostituisce _thermo_scalar per lettura esplicita
  [Fix #10] w_global ora moltiplicatore globale su TUTTI i bonus/penalità thermo
  [Fix #11] _cached_pv rimosso (era dichiarato ma mai usato)
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
       cash_ratio(1)   | portfolio_ratio(1) | psi_values(T) | thermo(N)]

    Reward:
      reward = rendimento_pct                              (clippato in [-10, 10])
             - lambda_concentration * excess_herfindahl
             - lambda_inaction      * frazione_posizioni_ferme
             - cash_penalty         * (eccesso_cash + mancanza_cash)
             - loss_penalty         * perdite_oltre_soglia
             + sell_profit_bonus    * n_vendite_profittevoli
             ± thermo_shaping       * w_stress * w_global  (peso continuo, prodotto)
    """

    def __init__(
        self,
        prices_real:       pd.DataFrame,
        prices_pred:       pd.DataFrame,
        tickers:           list[str],
        psi_df:            Optional[pd.DataFrame] = None,
        thermo_df:         Optional[pd.DataFrame] = None,
        initial_capital:   float = 100.0,
        transaction_cost:  float = 0.001,
        log_trades:        bool  = False,
        action_threshold:  float = 0.01,
        inaction_threshold: float = 0.05,
        max_position_pct:  float = 0.20,
        max_holding_steps: int   = 120,
        forced_sell_cooldown_steps: int = 20,
        bars_per_year:     int   = 252,
        sell_profit_bonus: float = 0.002,
        thermo_bonus_sell:        float = 0.002,
        thermo_bonus_buy:         float = 0.001,
        thermo_penalty_buy:       float = 0.003,
        stress_sell_threshold:    float = 1.0,
        stress_buy_threshold:     float = -0.5,
        efficiency_penalty_thresh: float = 1.5,
        # [Fix #5] Trust continuo: abbassato trust_min a 0.35 per attivare thermo
        # con trust medio ~0.477 → w = (0.477-0.35)/(0.65-0.35) = 0.42 (era 0.19)
        trust_min:         float = 0.35,
        trust_max:         float = 0.65,
        lambda_concentration: float = 0.1,
        lambda_inaction:      float = 0.1,
        lambda_loss:       float = 0.5,
        loss_threshold:    float = -0.5,
        # [Fix #1] Clip massimo del base_reward per episodio (evita esplosioni)
        reward_clip:       float = 10.0,
    ) -> None:
        self.prices_real      = prices_real
        self.prices_pred      = prices_pred
        self.tickers          = tickers
        self.all_columns      = list(prices_real.columns)
        self.initial_capital  = initial_capital
        self.transaction_cost = transaction_cost
        self.log_trades       = log_trades

        self.lambda_concentration     = lambda_concentration
        self.lambda_inaction          = lambda_inaction
        self.lambda_loss              = lambda_loss
        self.loss_threshold           = loss_threshold
        self.action_threshold         = action_threshold
        self.inaction_threshold       = inaction_threshold
        self.max_position_pct         = max_position_pct
        self.max_holding_steps        = max_holding_steps
        self.forced_sell_cooldown_steps = forced_sell_cooldown_steps
        self.bars_per_year            = bars_per_year
        self.sell_profit_bonus        = sell_profit_bonus
        self.reward_clip              = reward_clip

        # Thermo shaping
        self.thermo_bonus_sell         = thermo_bonus_sell
        self.thermo_bonus_buy          = thermo_bonus_buy
        self.thermo_penalty_buy        = thermo_penalty_buy
        self.stress_sell_threshold     = stress_sell_threshold
        self.stress_buy_threshold      = stress_buy_threshold
        self.efficiency_penalty_thresh = efficiency_penalty_thresh
        self.trust_min                 = trust_min
        self.trust_max                 = trust_max

        self.num_features = len(self.all_columns)
        self.num_tickers  = len(tickers)
        self._ticker_idx  = np.array(
            [self.all_columns.index(t) for t in tickers], dtype=np.int32
        )

        # ── Ψ per timestep ────────────────────────────────────────────────
        self._psi: Optional[np.ndarray] = None
        if psi_df is not None and not psi_df.empty:
            psi_aligned = psi_df.reindex(prices_real.index).ffill().bfill().fillna(0)
            psi_cols = [f"Psi_{t}" for t in tickers if f"Psi_{t}" in psi_aligned.columns]
            if psi_cols:
                self._psi = psi_aligned[psi_cols].values.astype(np.float32)
                print(f"[TradingEnv] Ψ attivo per {len(psi_cols)}/{self.num_tickers} ticker")

        # ── Thermo ───────────────────────────────────────────────────────
        self._thermo_np: Optional[np.ndarray] = None
        self._thermo_col_names: list[str] = []
        self._has_thermo = False

        if thermo_df is not None and not thermo_df.empty:
            th = thermo_df.reindex(prices_real.index).ffill().bfill().fillna(0)
            thm_cols = [c for c in th.columns if c.startswith("Thm_")]
            if thm_cols:
                self._thermo_np        = th[thm_cols].values.astype(np.float32)
                self._thermo_col_names = thm_cols
                # O(1) lookup invece di O(n) list.index() ad ogni step
                self._thermo_col_idx: dict[str, int] = {c: i for i, c in enumerate(thm_cols)}
                self._has_thermo       = True
                print(f"[TradingEnv] Thermo state: {len(thm_cols)} colonne → {thm_cols}")

        has_psi = self._psi is not None
        n_thm   = len(self._thermo_col_names) if self._has_thermo else 0
        self.state_dim = (
            2 * self.num_features
            + self.num_tickers + 2
            + (self.num_tickers if has_psi else 0)
            + n_thm
        )
        self.action_dim = self.num_tickers

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

        self._forced_sell_cooldown = np.zeros(self.num_tickers, dtype=np.int32)

        self._portfolio_history: list[float] = [self.initial_capital]
        self._trade_log:         list[dict]  = []
        self._value_per_ticker:  list[dict]  = []

        # Cache prezzi per step (evita doppia lettura nello stesso step)
        self._cached_step:   int        = -1
        self._cached_prices: np.ndarray = np.empty(self.num_tickers, dtype=np.float64)

    def reset(self) -> np.ndarray:
        self._reset_internals()
        return self._get_state()

    # ── state ─────────────────────────────────────────────────────────────

    def _get_state(self) -> np.ndarray:
        row_real = self._prices_real_np[self._step]
        row_pred = self._prices_pred_np[self._step]

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
        if self._has_thermo:
            parts.append(self._thermo_np[self._step])

        return np.concatenate(parts).astype(np.float32)

    # ── helpers ───────────────────────────────────────────────────────────

    def _current_prices(self) -> np.ndarray:
        if self._cached_step != self._step:
            self._cached_prices = self._prices_real_np[self._step][self._ticker_idx].astype(np.float64)
            self._cached_step   = self._step
        return self._cached_prices

    def _portfolio_value(self) -> float:
        return float(self._cash + np.dot(self._holdings, self._current_prices()))

    # ── reward ────────────────────────────────────────────────────────────

    def _compute_reward(
        self,
        prev_value:       float,
        new_value:        float,
        action:           np.ndarray,
        sell_profit_mask: np.ndarray,
    ) -> float:
        # [Fix #1] Denominatore minimo = 1% del capitale iniziale per evitare divisioni
        # per quasi-zero quando il portfolio è quasi azzerato (causa reward +10k negli ep 53/58).
        safe_prev = max(prev_value, self.initial_capital * 0.01)
        base_reward = np.clip(
            (new_value - prev_value) / (safe_prev + 1e-8),
            -self.reward_clip,
            self.reward_clip,
        )

        # Concentrazione: penalizza solo l'eccesso rispetto a portafoglio uniforme
        prices              = self._current_prices()
        weights             = (self._holdings * prices) / (new_value + 1e-8)
        herfindahl          = float(np.dot(weights, weights))
        uniform_herfindahl  = 1.0 / max(1, self.num_tickers)
        excess_concentration = max(0.0, herfindahl - uniform_herfindahl)
        concentration_penalty = self.lambda_concentration * excess_concentration

        # Inaction: penalizza posizioni aperte ferme
        open_positions = self._holdings > 1e-8
        if open_positions.any():
            frozen = open_positions & (np.abs(action) < self.inaction_threshold)
            inaction_penalty = (
                self.lambda_inaction
                * float(frozen.sum())
                / max(1, int(open_positions.sum()))
            )
        else:
            inaction_penalty = 0.0

        sell_bonus = self.sell_profit_bonus * float(sell_profit_mask.sum())

        # Cash penalty simmetrica: penalizza SIA eccesso che mancanza (target 5%–40%)
        cash_ratio   = self._cash / (new_value + 1e-8)
        excess_cash  = max(0.0, cash_ratio - 0.40)
        scarce_cash  = max(0.0, 0.05 - cash_ratio)
        cash_penalty = 0.03 * (excess_cash + scarce_cash)

        # Loss penalty: penalizza perdite oltre soglia
        current_return = (new_value - self.initial_capital) / (self.initial_capital + 1e-8)
        if current_return < self.loss_threshold:
            loss_penalty = self.lambda_loss * abs(current_return - self.loss_threshold)
        else:
            loss_penalty = 0.0

        return float(
            base_reward
            - concentration_penalty
            - inaction_penalty
            - cash_penalty
            - loss_penalty
            + sell_bonus
        )

    # ── trust weight (continuo) ───────────────────────────────────────────

    def _trust_weight(self, trust: float) -> float:
        """
        Peso continuo lineare tra trust_min e trust_max.
          trust <= trust_min  → peso 0.0
          trust >= trust_max  → peso 1.0
          intermedio          → interpolazione lineare

        Con trust_min=0.35, trust_max=0.65 e trust medio=0.477:
          w = (0.477 - 0.35) / (0.65 - 0.35) = 0.42  (era 0.19 con i vecchi parametri)
        """
        span = self.trust_max - self.trust_min
        return float(np.clip((trust - self.trust_min) / (span + 1e-8), 0.0, 1.0))

    # ── thermo reader ─────────────────────────────────────────────────────

    def _thermo_at(self, step: int, name: str, default: float = 0.0) -> float:
        """
        [Fix #9] Lettura esplicita del valore thermo al passo `step`.
        Sostituisce _thermo_scalar che usava self._step - 1 (off-by-one).
        Usare:
          - step = self._step   nel reward shaping e log PRIMA di _step += 1
          - step = self._step-1 se mai necessario leggere il passo precedente
        """
        if not self._has_thermo:
            return default
        idx = self._thermo_col_idx.get(name, -1)
        if idx < 0:
            return default
        # Bounds check per sicurezza
        safe_step = min(step, self._thermo_np.shape[0] - 1)
        return round(float(self._thermo_np[safe_step][idx]), 4)

    # ── step ──────────────────────────────────────────────────────────────

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        action = np.clip(action, -1.0, 1.0)

        # [Fix #2] Azzera i segnali di acquisto per ticker in cooldown
        if self._forced_sell_cooldown.any():
            action = action.copy()
            action[self._forced_sell_cooldown > 0] = np.minimum(
                action[self._forced_sell_cooldown > 0], 0.0
            )
            self._forced_sell_cooldown = np.maximum(0, self._forced_sell_cooldown - 1)

        # ── Leggi prezzi al passo t (PRIMA dell'incremento) ──────────────
        # [Fix #4] Tutti i calcoli di valore usano i prezzi di questo frame.
        prices     = self._current_prices()
        prev_value = self._portfolio_value()
        date       = self._dates[self._step] if self.log_trades else None

        # Leggi valori thermo al passo t per il log e il reward shaping
        # [Fix #9] _thermo_at(self._step) è corretto PRIMA di _step += 1
        t_stress = self._thermo_at(self._step, "Thm_Stress")
        t_eff    = self._thermo_at(self._step, "Thm_Efficiency")
        t_regime = self._thermo_at(self._step, "Thm_Regime")

        sell_profit_mask = np.zeros(self.num_tickers, dtype=bool)

        # ── VENDITE vettorizzate ──────────────────────────────────────────
        sell_mask = (action < -self.action_threshold) & (self._holdings > 1e-8)
        if sell_mask.any():
            sell_fracs   = np.abs(action[sell_mask])
            shares_sold  = self._holdings[sell_mask] * sell_fracs
            proceeds     = shares_sold * prices[sell_mask]
            fees         = proceeds * self.transaction_cost
            net_proceeds = proceeds - fees
            realized_pnl = net_proceeds - shares_sold * self._avg_cost[sell_mask]

            self._cash                += float(net_proceeds.sum())
            self._holdings[sell_mask] -= shares_sold

            closed = sell_mask.copy()
            closed[sell_mask] = self._holdings[sell_mask] < 1e-8
            self._holdings[closed]   = 0.0
            self._entry_step[closed] = -1

            sell_profit_mask[sell_mask] = realized_pnl > 0

            if self.log_trades:
                for k, i in enumerate(np.where(sell_mask)[0]):
                    self._trade_log.append({
                        "date":              date,
                        "ticker":            self.tickers[i],
                        "action":            "SELL",
                        "action_signal":     round(float(action[i]), 4),
                        "shares":            round(float(shares_sold[k]), 6),
                        "price":             round(float(prices[i]), 4),
                        "gross_value":       round(float(proceeds[k]), 2),
                        "fee":               round(float(fees[k]), 4),
                        "net_value":         round(float(net_proceeds[k]), 2),
                        "avg_cost":          round(float(self._avg_cost[i]), 4),
                        "realized_pnl":      round(float(realized_pnl[k]), 2),
                        "holdings_after":    round(float(self._holdings[i]), 6),
                        "cash_after":        round(float(self._cash), 2),
                        "thermo_stress":     t_stress,   # [Fix #9] corretto, step t
                        "thermo_efficiency": t_eff,
                    })

        # ── ACQUISTI vettorizzati ─────────────────────────────────────────
        valid_mask = np.zeros(self.num_tickers, dtype=bool)

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
                b          = budgets[valid_mask]
                p          = prices[valid_mask]
                fees_buy   = b * self.transaction_cost
                net_b      = b - fees_buy
                shares_buy = net_b / (p + 1e-8)

                h_old = self._holdings[valid_mask]
                c_old = self._avg_cost[valid_mask]
                h_new = h_old + shares_buy

                self._avg_cost[valid_mask] = np.where(
                    h_new > 1e-8,
                    (h_old * c_old + shares_buy * p) / h_new,
                    c_old,
                )

                valid_indices = np.where(valid_mask)[0]
                new_entries = valid_mask & (self._entry_step < 0)
                self._entry_step[new_entries] = self._step

                self._cash                -= float(b.sum())
                self._holdings[valid_mask] += shares_buy

                if self.log_trades:
                    for k, i in enumerate(valid_indices):
                        self._trade_log.append({
                            "date":              date,
                            "ticker":            self.tickers[i],
                            "action":            "BUY",
                            "action_signal":     round(float(action[i]), 4),
                            "shares":            round(float(shares_buy[k]), 6),
                            "price":             round(float(p[k]), 4),
                            "gross_value":       round(float(b[k]), 2),
                            "fee":               round(float(fees_buy[k]), 4),
                            "net_value":         round(float(net_b[k]), 2),
                            "avg_cost":          round(float(self._avg_cost[i]), 4),
                            "realized_pnl":      0.0,
                            "holdings_after":    round(float(self._holdings[i]), 6),
                            "cash_after":        round(float(self._cash), 2),
                            "thermo_stress":     t_stress,   # [Fix #9] corretto, step t
                            "thermo_efficiency": t_eff,
                        })

        # ── FORCED SELL + COOLDOWN ────────────────────────────────────────
        if self._step > 0 and self.max_holding_steps > 0:
            steps_held  = self._step - self._entry_step
            forced_mask = (steps_held >= self.max_holding_steps) & (self._holdings > 1e-8)
            if forced_mask.any():
                # [Fix #4] prices è ancora al passo t → valore corretto
                proceeds_f  = self._holdings[forced_mask] * prices[forced_mask]
                fees_f      = proceeds_f * self.transaction_cost
                self._cash += float((proceeds_f - fees_f).sum())
                self._holdings[forced_mask]   = 0.0
                self._entry_step[forced_mask] = -1
                self._forced_sell_cooldown[forced_mask] = self.forced_sell_cooldown_steps

        # ── [Fix #4] new_value calcolato PRIMA di _step += 1 ─────────────
        # Garantisce che sell, buy, forced_sell e new_value usino tutti
        # gli stessi prezzi (frame t). Prima il +1 causava rivalutazione a t+1.
        new_value = self._portfolio_value()

        # ── Reward base ───────────────────────────────────────────────────
        reward = self._compute_reward(prev_value, new_value, action, sell_profit_mask)

        # ── THERMODYNAMIC REWARD SHAPING ──────────────────────────────────
        if self._has_thermo:
            trust_stress = self._thermo_at(self._step, "Trust_Thm_Stress", default=0.5)
            trust_global = self._thermo_at(self._step, "Trust_Global",     default=0.5)

            w_stress = self._trust_weight(trust_stress)
            # [Fix #10] w_global è ora moltiplicatore esterno su TUTTI i termini thermo.
            # Semantica: quanto ci fidiamo del sistema termodinamico nel suo insieme.
            #   w_global basso → mercato caotico → thermo si spegne globalmente
            #   w_global alto  → segnali affidabili → thermo opera a piena forza
            # Ogni bonus/penalità scala come: valore * w_stress * w_global
            # Esempio con trust medio 0.477 e trust_min=0.35, trust_max=0.65:
            #   w_stress = w_global = 0.42
            #   prodotto = 0.42 * 0.42 = 0.18
            #   thermo_bonus_sell effettivo = 0.002 * 0.18 = 0.00036
            # Per aumentare l'impatto: alzare i bonus o abbassare trust_min.
            w_global = self._trust_weight(trust_global)

            if sell_mask.any():
                if t_stress > self.stress_sell_threshold:
                    reward += self.thermo_bonus_sell * w_stress * w_global

            if valid_mask.any():
                if t_stress < self.stress_buy_threshold:
                    reward += self.thermo_bonus_buy * w_stress * w_global
                if t_eff > self.efficiency_penalty_thresh:
                    reward -= self.thermo_penalty_buy * w_stress * w_global
                if t_regime >= 3.0:
                    trust_regime = self._thermo_at(self._step, "Trust_Thm_Regime", default=0.5)
                    w_regime = self._trust_weight(trust_regime)
                    reward -= self.thermo_penalty_buy * 0.5 * w_regime * w_global

        # ── Avanza al passo t+1 ───────────────────────────────────────────
        self._step += 1
        done = self._step >= self._n_steps - 1

        self._portfolio_history.append(new_value)

        # Log snapshot (usa prezzi a t+1 per unrealized PnL aggiornato)
        if self.log_trades:
            new_prices = self._current_prices()   # ora a t+1
            unrealized = self._holdings * (new_prices - self._avg_cost)
            snap: dict = {
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

        # [Fix #8] next_state usa _get_state() SEMPRE — no zero-vector artificiale.
        # Il flag done è già usato in ddpg.py come (1 - dones) nel target Q,
        # quindi l'agente non fa bootstrap da questo stato quando done=True.
        next_state = self._get_state()

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
        vpdf = self.value_per_ticker_df()
        rows = []
        for ticker in self.tickers:
            t_log = log[log["ticker"] == ticker]
            buys  = t_log[t_log["action"] == "BUY"]
            sells = t_log[t_log["action"] == "SELL"]
            realized  = float(sells["realized_pnl"].sum()) if "realized_pnl" in sells else 0.0
            unreal_col = f"{ticker}_unrealized_pnl"
            unreal = (
                float(vpdf[unreal_col].iloc[-1])
                if (not vpdf.empty and unreal_col in vpdf.columns) else 0.0
            )
            rows.append({
                "ticker":           ticker,
                "n_buy":            len(buys),
                "n_sell":           len(sells),
                "total_bought_usd": round(buys["gross_value"].sum() if len(buys) else 0.0, 2),
                "total_sold_usd":   round(sells["gross_value"].sum() if len(sells) else 0.0, 2),
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
        """
        Annualizza con bars_per_year.
        Per dati 2-min: bars_per_year=98280. Per daily: 252.
        """
        h = self.portfolio_history()
        if len(h) < 2:
            return 0.0
        r      = np.diff(h) / (h[:-1] + 1e-8)
        excess = r - risk_free / self.bars_per_year
        std    = excess.std()
        return float(excess.mean() / (std + 1e-8) * np.sqrt(self.bars_per_year))

    def max_drawdown(self) -> float:
        h    = self.portfolio_history()
        peak = np.maximum.accumulate(h)
        return float(((h - peak) / (peak + 1e-8)).min())