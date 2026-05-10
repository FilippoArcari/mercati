from gymnasium import spaces
import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

from modelli.predictor.predictor import Predictor
from modelli.features.thermodynamic import THERMO_N_FEATURES
from sklearn.pipeline import Pipeline

BATCH = 32

# ── Indici delle feature termodinamiche nell'array thermo_state ──────────────
# (corrispondono all'ordine prodotto da compute_thermodynamic_features)
_T_IDX     = 0   # Temperatura (Shannon entropy)
_DT_IDX    = 1   # Gradiente temperatura
_P_IDX     = 2   # Pressione Van der Waals (normalizzata)
_W_IDX     = 3   # Lavoro cumulativo (normalizzato)
_Z_IDX     = 4   # Z-Score divergenza da tassi
_E_IDX     = 5   # Oscillatore efficienza
_PHASE_IDX = 6   # Fase [0..1]: 0=Solido, 0.33=Liquido, 0.67=Gas, 1=Supercritico
_CAR_IDX   = 7   # Efficienza Carnot
_LAG_IDX   = 8   # Segnale lag monetario

# Soglia mediana nel dominio scalato [0.05, 0.95]: RobustScaler centra sulla
# mediana → dopo MinMaxScaler, 0.5 corrisponde alla mediana storica.
_MEDIAN = 0.50


def _phase_label(phase_scaled: float) -> str:
    """Converte il valore scalato della fase in etichetta leggibile."""
    # I 4 stati [0, 0.33, 0.67, 1.0] vengono compressi nel range [0.05, 0.95]
    # Usiamo quartili del range per identificare lo stato
    r = (phase_scaled - 0.05) / 0.90  # rinormalizza in [0, 1]
    if r < 0.25:
        return "Solido"
    elif r < 0.50:
        return "Liquido"
    elif r < 0.75:
        return "Gas"
    else:
        return "Supercritico"


class MultiTickerTradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, checkpoint_path: str, config: dict, data: pd.DataFrame, scaler: Pipeline):
        super().__init__()
        self.tickers          = config["data"]["tickers"]
        self.n_tickers        = len(self.tickers)
        self.sliding_window   = config["prediction"]["sliding_window"]
        self.forecast_horizon = config["prediction"]["forecast_horizon"]
        self.n_main_features  = config["prediction"]["real_features"]
        self.initial_cash     = config.get("initial_cash", 100.0)
        self.fee_rate         = config.get("fee_rate", 0.001)
        self.log_every        = config.get("trader", {}).get("log_every", 25)

        # ── Prezzi reali (inverse transform solo sulle colonne main) ──────────
        self.df_unscaled = scaler.inverse_transform(data.values)[:, :self.n_main_features]

        df_ticker_cols = data.columns[:self.n_main_features].tolist()
        self.ticker_col_idx: dict[str, int] = {
            t: df_ticker_cols.index(t) for t in self.tickers
        }

        # ── Feature termodinamiche ────────────────────────────────────────────
        # Le thermo sono le ULTIME THERMO_N_FEATURES colonne del DataFrame.
        # Sono già state scalate insieme a tutto il resto in get_data().
        thermo_cols = [c for c in data.columns if c.startswith("thermo_")]
        if len(thermo_cols) == THERMO_N_FEATURES:
            thermo_start = data.shape[1] - THERMO_N_FEATURES
            # Allineate al passo dell'agente: stesso offset di sliding_window
            self.thermo_data: np.ndarray | None = (
                data.values[self.sliding_window :, thermo_start :].astype(np.float32)
            )
            self.thermo_cols = thermo_cols
            print(f"[env] Feature termodinamiche caricate: {thermo_cols}")
        else:
            self.thermo_data = None
            self.thermo_cols = []
            if thermo_cols:
                print(f"[env] WARN: trovate {len(thermo_cols)} colonne thermo, attese {THERMO_N_FEATURES}. Ignorate.")
            else:
                print("[env] Nessuna feature termodinamica trovata (modalità intraday?).")

        # ── Carica modello e pre-calcola previsioni ───────────────────────────
        model = Predictor.load_from_checkpoint(checkpoint_path, config=config["prediction"])
        model.eval()

        full_tensor = torch.tensor(data.values, dtype=torch.float32)
        self.n_steps = len(data) - self.sliding_window
        self.dates: np.ndarray = data.index[self.sliding_window :].to_numpy()

        windows = full_tensor.unfold(0, self.sliding_window, 1).permute(0, 2, 1).contiguous()

        all_preds = []
        with torch.no_grad():
            for start in range(0, len(windows), BATCH):
                batch = windows[start : start + BATCH]
                all_preds.append(model(batch).cpu())

        self.forecasts: np.ndarray = torch.cat(all_preds, dim=0).numpy().astype(np.float32)

        # ── Prezzi reali per ticker ───────────────────────────────────────────
        self.prices: dict[str, np.ndarray] = {
            t: self.df_unscaled[self.sliding_window :, self.ticker_col_idx[t]]
            for t in self.tickers
        }
        for t, arr in self.prices.items():
            bad = np.sum(~np.isfinite(arr) | (arr <= 0))
            if bad > 0:
                print(f"[WARN] {t}: {bad} prezzi non validi (<=0 o NaN/inf)")

        # ── Spazi osservazione ────────────────────────────────────────────────
        forecast_dim  = self.forecast_horizon * self.n_main_features
        n_thermo_obs  = THERMO_N_FEATURES if self.thermo_data is not None else 0
        obs_size      = forecast_dim + self.n_tickers + 1 + n_thermo_obs

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete([3] * self.n_tickers)
        self._init_state()

    # ── Stato ─────────────────────────────────────────────────────────────────

    def _init_state(self):
        self.cash                 = float(self.initial_cash)
        self.positions            = {t: 0 for t in self.tickers}
        self.total_shares_bought  = {t: 0 for t in self.tickers}
        self.total_shares_sold    = {t: 0 for t in self.tickers}
        self.prev_portfolio_value = self.initial_cash
        self.peak_value           = self.initial_cash
        self.step_idx             = 0
        self.history              = []
        self.returns              = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._init_state()
        return self._get_obs(), {}

    # ── Reward ────────────────────────────────────────────────────────────────

    def _evaluta_reward(
        self,
        portfolio_value: float,
        prev_value: float,
        positions_open: int,
        thermo_state: np.ndarray | None = None,
    ) -> float:
        log_return = float(np.log((portfolio_value + 1e-8) / (prev_value + 1e-8)))
        self.returns.append(log_return)

        window        = 30
        recent_returns = np.array(self.returns[-window:])
        downside       = recent_returns[recent_returns < 0]
        downside_std   = np.sqrt(np.mean(downside ** 2)) + 1e-8 if len(downside) >= 2 else 1e-5
        sortino_reward = log_return / downside_std

        reward  = sortino_reward + 0.10 * log_return
        reward += -0.01 * max(0, (self.peak_value - portfolio_value) / (self.peak_value + 1e-8) - 0.2)
        reward += -0.001 * sum(self.positions[t] for t in self.tickers)
        reward += -0.001 * sum(self.total_shares_bought[t] + self.total_shares_sold[t] for t in self.tickers)
        reward += -0.05 if log_return < -0.05 else 0
        reward +=  0.05 if log_return >  0.05 else 0
        reward += -0.02 * (positions_open < 5)

        # ── Reward shaping termodinamico ──────────────────────────────────────
        # Il RobustScaler centra sulla mediana → 0.5 ≈ mediana storica.
        # Sopra 0.5 = "caldo/rischioso", sotto 0.5 = "freddo/espansivo".
        if thermo_state is not None:
            Z     = float(thermo_state[_Z_IDX])
            E     = float(thermo_state[_E_IDX])
            phase = float(thermo_state[_PHASE_IDX])
            is_buying = positions_open > 0

            # Stress termico: Z alto + fase calda → penalizza acquisti
            is_stress     = Z > _MEDIAN and phase > _MEDIAN
            # Espansione sana: Z basso → premia acquisti
            is_expanding  = Z < _MEDIAN
            # Alta dissipazione: efficienza alta → penalizza (distribuzione in corso)
            is_dissipating = E > (_MEDIAN + 0.15)

            reward += -0.03 * is_stress     * is_buying     # Penalità buy in stress termico
            reward +=  0.02 * is_expanding  * is_buying     # Bonus buy in espansione sana
            reward += -0.02 * is_dissipating                # Penalità alta dissipazione indip. dalla posizione

        reward = max(min(reward, 10.0), -10.0)
        return float(reward)

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(self, actions: np.ndarray):
        current_date = self.dates[self.step_idx]
        step_log     = {"step": self.step_idx, "date": current_date}

        # Stato termodinamico corrente (usato nel reward e nel log)
        thermo_state = (
            self.thermo_data[self.step_idx]
            if self.thermo_data is not None
            else None
        )

        buy_tickers  = [
            t for t, a in zip(self.tickers, actions)
            if a == 1 and self.cash > self.prices[t][self.step_idx]
        ]
        cash_per_buy = self.cash / len(buy_tickers) if buy_tickers else 0.0

        step_log["actions"] = {}
        for ticker, action in zip(self.tickers, actions):
            price = self.prices[ticker][self.step_idx]

            if action == 1 and ticker in buy_tickers:
                shares = int(cash_per_buy / price)
                if shares > 0:
                    cost = shares * price
                    self.positions[ticker]        += shares
                    self.total_shares_bought[ticker] += shares
                    self.cash                     -= cost * (1 + self.fee_rate)
                    step_log["actions"][ticker]    = {"type": "buy", "shares": shares, "price": price}
                else:
                    step_log["actions"][ticker]    = {"type": "hold", "reason": "insufficient cash"}

            elif action == 2 and self.positions[ticker] > 0:
                shares   = self.positions[ticker]
                proceeds = shares * price
                self.cash                    += proceeds * (1 - self.fee_rate)
                self.positions[ticker]        = 0
                self.total_shares_sold[ticker] += shares
                step_log["actions"][ticker]    = {"type": "sell", "shares": shares, "price": price}

            else:
                step_log["actions"][ticker] = {"type": "hold"}

        portfolio_value = self.cash + sum(
            self.positions[t] * self.prices[t][self.step_idx] for t in self.tickers
        )
        self.peak_value         = max(self.peak_value, portfolio_value)
        open_positions_count    = sum(1 for t in self.tickers if self.positions[t] > 0)

        reward = self._evaluta_reward(
            portfolio_value,
            self.prev_portfolio_value,
            positions_open=open_positions_count,
            thermo_state=thermo_state,
        )
        self.prev_portfolio_value = portfolio_value

        # Log termodinamico
        thermo_log = {}
        if thermo_state is not None:
            thermo_log = {
                "thermo_Z":     round(float(thermo_state[_Z_IDX]), 4),
                "thermo_phase": _phase_label(float(thermo_state[_PHASE_IDX])),
                "thermo_E":     round(float(thermo_state[_E_IDX]), 4),
                "thermo_T":     round(float(thermo_state[_T_IDX]), 4),
                "thermo_carnot": round(float(thermo_state[_CAR_IDX]), 4),
            }

        step_log.update({
            "cash":            round(self.cash, 4),
            "portfolio_value": round(portfolio_value, 4),
            "reward":          round(reward, 6),
            "positions":       {t: self.positions[t] for t in self.tickers},
            "total_shares_bought": self.total_shares_bought.copy(),
            "total_shares_sold":   self.total_shares_sold.copy(),
            **thermo_log,
        })
        self.history.append(step_log)

        self.step_idx += 1
        terminated = self.step_idx >= self.n_steps - 1
        return self._get_obs(), reward, terminated, False, {}

    # ── Osservazione ──────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        forecast_flat = self.forecasts[self.step_idx].flatten()

        positions_norm = np.array([
            self.positions[t] * self.prices[t][self.step_idx] / self.initial_cash
            for t in self.tickers
        ], dtype=np.float32)

        cash_norm = np.array([self.cash / self.initial_cash], dtype=np.float32)

        parts = [forecast_flat, positions_norm, cash_norm]

        if self.thermo_data is not None:
            parts.append(self.thermo_data[self.step_idx])

        obs = np.concatenate(parts)
        obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        return obs.astype(np.float32)

    # ── Render ────────────────────────────────────────────────────────────────

    def render(self, only_on_trade: bool, mode: str = "human"):
        if not self.history:
            return
        last = self.history[-1]

        trades = []
        if "actions" in last:
            for t, act in last["actions"].items():
                if act.get("type") in ("buy", "sell"):
                    trades.append(f"{t}: {act['type']} {act.get('shares', 0)}sh @ {act.get('price', 0):.4f}")
        trades_str = "  |  ".join(trades) if trades else None

        if only_on_trade and not trades_str:
            return

        open_positions = [f"{t}={self.positions[t]}sh" for t in self.tickers if self.positions[t] > 0]
        pos_str        = ", ".join(open_positions) if open_positions else "flat"
        drawdown       = (self.peak_value - last["portfolio_value"]) / (self.peak_value + 1e-8)
        display_trades = trades_str if trades_str else "all hold"
        date_str       = pd.to_datetime(last["date"]).strftime("%Y-%m-%d")

        thermo_str = ""
        if "thermo_Z" in last:
            thermo_str = (
                f"\n └─ Termodin.  : Fase={last['thermo_phase']} | "
                f"Z={last['thermo_Z']:+.3f} | "
                f"T={last['thermo_T']:.3f} | "
                f"E={last['thermo_E']:+.3f} | "
                f"Carnot={last['thermo_carnot']:.3f}"
            )

        print(
            f"[{date_str}] Step {last['step']:04d}\n"
            f" ├─ Trades    : {display_trades}\n"
            f" ├─ Positions : {pos_str}\n"
            f" ├─ Financial : Cash: {last['cash']:.2f} | Portfolio: {last['portfolio_value']:.2f}\n"
            f" └─ Metrics   : Reward: {last['reward']:+.6f} | Drawdown: {drawdown:.2%}"
            f"{thermo_str}"
        )

    # ── Grafici episodio ──────────────────────────────────────────────────────

    def plot_episode(self, save_path: str | None = None) -> None:
        """
        Dashboard a 3 pannelli del backtest:
          1. Valore portfolio (con buy/sell markers)
          2. Reward per step
          3. Feature termodinamiche (se disponibili)
        """
        df = self.get_history()
        if df.empty:
            print("[plot_episode] History vuota.")
            return

        has_thermo = "thermo_Z" in df.columns
        n_panels   = 3 if has_thermo else 2

        fig = plt.figure(figsize=(16, 5 * n_panels), facecolor="#0d1117")
        fig.suptitle("Episode Backtest Dashboard", fontsize=14, color="#e6edf3",
                     fontweight="bold", y=0.99)

        gs   = gridspec.GridSpec(n_panels, 1, hspace=0.10, figure=fig)
        axes = [fig.add_subplot(gs[i]) for i in range(n_panels)]
        _ax_style = lambda ax: (
            ax.set_facecolor("#161b22"),
            ax.tick_params(colors="#8b949e", labelsize=8),
            [s.set_color("#30363d") for s in ax.spines.values()],
            ax.grid(True, color="#21262d", linewidth=0.5, alpha=0.7),
        )
        for ax in axes:
            _ax_style(ax)

        dates = pd.to_datetime(df["date"])

        # ── Pannello 1: Portfolio value + trade markers ───────────────────────
        ax0 = axes[0]
        ax0.plot(dates, df["portfolio_value"], color="#58a6ff", linewidth=1.3,
                 label="Portfolio value")
        ax0.axhline(self.initial_cash, color="#6e7681", linewidth=0.8,
                    linestyle="--", label=f"Capitale iniziale ({self.initial_cash:.0f})")

        # Buy/sell markers
        buy_mask  = df["actions"].apply(
            lambda acts: any(a.get("type") == "buy" for a in acts.values() if isinstance(a, dict))
        )
        sell_mask = df["actions"].apply(
            lambda acts: any(a.get("type") == "sell" for a in acts.values() if isinstance(a, dict))
        )
        if buy_mask.any():
            ax0.scatter(dates[buy_mask], df["portfolio_value"][buy_mask],
                        color="#3fb950", marker="^", s=40, zorder=5, label="Buy", alpha=0.8)
        if sell_mask.any():
            ax0.scatter(dates[sell_mask], df["portfolio_value"][sell_mask],
                        color="#f85149", marker="v", s=40, zorder=5, label="Sell", alpha=0.8)

        # Drawdown shading
        cum_max = df["portfolio_value"].cummax()
        ax0_twin = ax0.twinx()
        ax0_twin.fill_between(dates,
                              0,
                              -(cum_max - df["portfolio_value"]) / (cum_max + 1e-8) * 100,
                              color="#f85149", alpha=0.15, label="Drawdown %")
        ax0_twin.set_ylabel("Drawdown %", color="#8b949e", fontsize=8)
        ax0_twin.tick_params(colors="#8b949e", labelsize=7)
        ax0_twin.spines["right"].set_color("#30363d")

        ax0.set_ylabel("Valore portfolio", color="#8b949e", fontsize=9)
        ax0.legend(loc="upper left", fontsize=7, facecolor="#21262d",
                   labelcolor="#e6edf3", framealpha=0.8)
        ax0.tick_params(labelbottom=False)

        # ── Pannello 2: Reward ───────────────────────────────────────────────
        ax1 = axes[1]
        reward_pos = df["reward"].clip(lower=0)
        reward_neg = df["reward"].clip(upper=0)
        ax1.fill_between(dates, 0, reward_pos, color="#3fb950", alpha=0.6, label="Reward > 0")
        ax1.fill_between(dates, 0, reward_neg, color="#f85149", alpha=0.6, label="Reward < 0")
        ax1.plot(dates, df["reward"].rolling(20).mean(), color="#ffa657",
                 linewidth=1.2, label="Media mobile 20gg", alpha=0.9)
        ax1.axhline(0, color="#6e7681", linewidth=0.8, linestyle="--")
        ax1.set_ylabel("Reward", color="#8b949e", fontsize=9)
        ax1.legend(loc="upper left", fontsize=7, facecolor="#21262d",
                   labelcolor="#e6edf3", framealpha=0.8)
        if has_thermo:
            ax1.tick_params(labelbottom=False)

        # ── Pannello 3: Feature termodinamiche ───────────────────────────────
        if has_thermo:
            ax2 = axes[2]

            # Z-score: rosso = stress, blu = espansione
            z  = df["thermo_Z"]
            ax2.fill_between(dates, _MEDIAN, z.clip(lower=_MEDIAN),
                             color="#f85149", alpha=0.4, label="Z > mediana (stress)")
            ax2.fill_between(dates, z.clip(upper=_MEDIAN), _MEDIAN,
                             color="#58a6ff", alpha=0.4, label="Z < mediana (espansione)")
            ax2.plot(dates, z, color="#d2a8ff", linewidth=0.8, alpha=0.7)
            ax2.axhline(_MEDIAN, color="#6e7681", linewidth=0.8, linestyle="--", label="Mediana storica")

            # Fase come area colorata
            phase_vals = df["thermo_phase"].map(
                {"Solido": 0, "Liquido": 1, "Gas": 2, "Supercritico": 3}
            ).fillna(1)
            ax2_twin = ax2.twinx()
            ax2_twin.plot(dates, phase_vals, color="#ffa657", linewidth=0.7,
                          alpha=0.5, drawstyle="steps-post", label="Fase (dx)")
            ax2_twin.set_yticks([0, 1, 2, 3])
            ax2_twin.set_yticklabels(["Solido", "Liquido", "Gas", "Supercrit."],
                                     fontsize=7, color="#8b949e")
            ax2_twin.tick_params(colors="#8b949e")
            ax2_twin.spines["right"].set_color("#30363d")

            ax2.set_ylabel("Z-Score termodinamico (scalato)", color="#8b949e", fontsize=9)
            ax2.set_xlabel("Data", color="#8b949e", fontsize=9)
            ax2.legend(loc="upper left", fontsize=7, facecolor="#21262d",
                       labelcolor="#e6edf3", framealpha=0.8)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            print(f"[plot_episode] Salvato in: {save_path}")
        plt.show()

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> None:
        df = self.get_history()
        if df.empty:
            return

        final_value  = df["portfolio_value"].iloc[-1]
        total_return = (final_value - self.initial_cash) / self.initial_cash
        max_drawdown = ((df["portfolio_value"].cummax() - df["portfolio_value"])
                        / df["portfolio_value"].cummax()).max()
        n_trades     = 0
        if "actions" in df.columns:
            n_trades = df["actions"].apply(
                lambda acts: sum(
                    1 for a in acts.values()
                    if isinstance(a, dict) and a.get("type") in ("buy", "sell")
                )
            ).sum()

        sharpe = (df["reward"].mean() / (df["reward"].std() + 1e-8)) * np.sqrt(252)

        # Statistiche termodinamiche
        thermo_summary = ""
        if "thermo_phase" in df.columns:
            phase_counts = df["thermo_phase"].value_counts(normalize=True)
            dominant     = phase_counts.index[0] if not phase_counts.empty else "N/A"
            avg_z        = df["thermo_Z"].mean() if "thermo_Z" in df.columns else float("nan")
            thermo_summary = (
                f"\n  Fase dom. : {dominant} ({phase_counts.iloc[0]:.0%} del periodo)"
                f"\n  Z medio   : {avg_z:+.4f} ({'stress' if avg_z > _MEDIAN else 'espansione'})"
            )

        print("\n" + "=" * 55)
        print("  EPISODE SUMMARY")
        print(f"  Period : {pd.to_datetime(df['date'].iloc[0]).strftime('%Y-%m-%d')}  →  "
              f"{pd.to_datetime(df['date'].iloc[-1]).strftime('%Y-%m-%d')}  total {len(df)} steps")
        print(f"  Return : {total_return:+.2%}")
        print(f"  Max DD : {max_drawdown:.2%}")
        print(f"  Sharpe : {sharpe:.3f}  (annualised, daily)")
        print(f"  Trades : {n_trades}")
        print(thermo_summary)
        print("=" * 55 + "\n")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def get_history(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)

    def get_total_shares_bought(self) -> dict[str, int]:
        return self.total_shares_bought.copy()

    def get_total_shares_sold(self) -> dict[str, int]:
        return self.total_shares_sold.copy()