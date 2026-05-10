from gymnasium import spaces
import gymnasium as gym
import numpy as np
import pandas as pd
import torch

from modelli.predictor.predictor import Predictor
from sklearn.pipeline import Pipeline

BATCH = 32


class MultiTickerTradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, checkpoint_path: str, config: dict, data: pd.DataFrame,scaler: Pipeline):
        super().__init__()
        self.tickers          = config["data"]["tickers"]
        self.n_tickers        = len(self.tickers)
        self.sliding_window   = config["prediction"]["sliding_window"]
        self.forecast_horizon = config["prediction"]["forecast_horizon"]
        self.n_main_features  = config["prediction"]["real_features"]  # = 50
        self.initial_cash     = config.get("initial_cash", 100.0)
        self.fee_rate         = config.get("fee_rate", 0.001)
        self.df_unscaled      = scaler.inverse_transform(data.values)[:, :self.n_main_features]
        self.log_every        = config.get("trader", {}).get("log_every", 25)
    

        # Mappa ticker → indice colonna (prime 50 col = prezzi reali)
        df_ticker_cols = data.columns[:self.n_main_features].tolist()
        self.ticker_col_idx: dict[str, int] = {
            t: df_ticker_cols.index(t) for t in self.tickers
        }

        # ── Carica modello UNA VOLTA ───────────────────────────────────────────
        model = Predictor.load_from_checkpoint(
            checkpoint_path, config=config["prediction"]
        )
        model.eval()

        # ── UN SOLO forward pass su tutto il dataset ───────────────────────────
        full_tensor = torch.tensor(data.values, dtype=torch.float32)  # [T, 65]
        self.n_steps = len(data) - self.sliding_window
        self.dates: np.ndarray = data.index[self.sliding_window:].to_numpy()


        windows = full_tensor.unfold(0, self.sliding_window, 1)  # [N, 65, W]
        windows = windows.permute(0, 2, 1).contiguous()          # [N, W, 65]
                                                

        all_preds = []                                     
        with torch.no_grad():
            for start in range(0, len(windows), BATCH):
                batch = windows[start : start + BATCH]
                all_preds.append(model(batch).cpu())

        fc = torch.cat(all_preds, dim=0)  # [N, horizon, 50]
        self.forecasts: np.ndarray = fc.numpy().astype(np.float32)    # [N, horizon, 50]

        # ── Prezzi reali per ogni ticker ──────────────────────────────────────
        # I dati sono già prezzi reali → nessuna trasformazione necessaria.
        self.prices: dict[str, np.ndarray] = {
            t: self.df_unscaled[self.sliding_window:, self.ticker_col_idx[t]]
            for t in self.tickers
        }


        # Sanity check: segnala subito se ci sono prezzi non validi
        for t, arr in self.prices.items():
            bad = np.sum(~np.isfinite(arr) | (arr <= 0))
            if bad > 0:
                print(f"[WARN] {t}: {bad} prezzi non validi (<=0 o NaN/inf)")

        # ── Spazi ─────────────────────────────────────────────────────────────
        # obs = [forecast_flat (horizon*50)] + [pos_norm * n_tickers] + [cash_norm]
        forecast_dim = self.forecast_horizon * self.n_main_features
        obs_size     = forecast_dim + self.n_tickers + 1

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete([3] * self.n_tickers)
        self._init_state()

    # ── Stato ─────────────────────────────────────────────────────────────────

    def _init_state(self):
        self.cash                 = float(self.initial_cash)
        self.positions            = {t: 0 for t in self.tickers}   # numero di shares
        self.total_shares_bought  = {t: 0 for t in self.tickers}
        self.total_shares_sold    = {t: 0 for t in self.tickers}
        self.prev_portfolio_value = self.initial_cash
        self.peak_value           = self.initial_cash
        self.step_idx             = 0
        self.history              = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._init_state()
        return self._get_obs(), {}
    
    def _evaluta_reward(self, portfolio_value: float, prev_value: float) -> float:
        """Calcola il reward basato sulla variazione percentuale del valore del portafoglio. Visto come log rendimento"""
        # return float(
        #     (portfolio_value - self.prev_portfolio_value)
        #     / (self.prev_portfolio_value + 1e-8)
        # )
        return float(
            np.log((portfolio_value + 1e-8) / (prev_value + 1e-8))
        )
    

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(self, actions: np.ndarray):
        current_date = self.dates[self.step_idx]
        step_log = {"step": self.step_idx, "date": current_date}


        buy_tickers = [
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
                    self.positions[ticker] += shares
                    self.total_shares_bought[ticker] += shares
                    self.cash              -= cost * (1 + self.fee_rate)
                    step_log["actions"][ticker] = {"type": "buy", "shares": shares, "price": price}
                else:
                    step_log["actions"][ticker] = {"type": "hold", "reason": "insufficient cash"}

            elif action == 2 and self.positions[ticker] > 0:
                shares   = self.positions[ticker]
                proceeds = shares * price
                self.cash              += proceeds * (1 - self.fee_rate)
                self.positions[ticker]  = 0
                self.total_shares_sold[ticker] += shares
                step_log["actions"][ticker] = {"type": "sell", "shares": shares, "price": price}

            else:
                step_log["actions"][ticker] = {"type": "hold"}

        portfolio_value = self.cash + sum(
            self.positions[t] * self.prices[t][self.step_idx]
            for t in self.tickers
        )
        self.peak_value = max(self.peak_value, portfolio_value)

        reward = self._evaluta_reward(portfolio_value, self.prev_portfolio_value)
        self.prev_portfolio_value = portfolio_value

        step_log.update({
            "cash":            round(self.cash, 4),
            "portfolio_value": round(portfolio_value, 4),
            "reward":          round(reward, 6),
            "positions":      {t: self.positions[t] for t in self.tickers},
            "total_shares_bought": self.total_shares_bought.copy(),
            "total_shares_sold": self.total_shares_sold.copy()
        })
        self.history.append(step_log)

        self.step_idx += 1
        terminated = self.step_idx >= self.n_steps - 1
        return self._get_obs(), reward, terminated, False, {}

    # ── Osservazione ──────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        forecast_flat = self.forecasts[self.step_idx].flatten()        # [horizon*50]

        # Valore posizione normalizzato: (shares * prezzo) / initial_cash
        positions_norm = np.array([
            self.positions[t] * self.prices[t][self.step_idx] / self.initial_cash
            for t in self.tickers
        ], dtype=np.float32)

        cash_norm = np.array([self.cash / self.initial_cash], dtype=np.float32)

        obs = np.concatenate([forecast_flat, positions_norm, cash_norm])
        obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        return obs.astype(np.float32)

    # ── Render e history ──────────────────────────────────────────────────────

    def render(self, mode="human"):
        if not self.history:
            return
        last = self.history[-1]

        trades = []
        if "actions" in last:
            for t, act in last["actions"].items():
                if act.get("type") in ("buy", "sell"):
                    trades.append(f"{t}: {act['type']} {act.get('shares', 0)}sh @ {act.get('price', 0):.4f}")
        trades_str = "  |  ".join(trades) if trades else "all hold"

        open_positions = [
            f"{t}={self.positions[t]}sh"
            for t in self.tickers
            if self.positions[t] > 0
        ]
        pos_str = ", ".join(open_positions) if open_positions else "flat"

        drawdown = (self.peak_value - last["portfolio_value"]) / (self.peak_value + 1e-8)

        date_str = pd.to_datetime(last['date']).strftime('%Y-%m-%d')
        print(
            f"[{date_str}] step={last['step']:4d} | "
            f"Trades: {trades_str}\n"
            f"  Positions: {pos_str} | "
            f"Cash: {last['cash']:.4f} | "
            f"Portfolio: {last['portfolio_value']:.4f} | "
            f"Reward: {last['reward']:+.6f} | "
            f"Drawdown: {drawdown:.2%}"
        )

    def summary(self) -> None:
        df = self.get_history()
        if df.empty:
            return

        final_value   = df["portfolio_value"].iloc[-1]
        total_return  = (final_value - self.initial_cash) / self.initial_cash
        max_drawdown  = ((df["portfolio_value"].cummax() - df["portfolio_value"])
                        / df["portfolio_value"].cummax()).max()
        
        n_trades = 0
        if "actions" in df.columns:
            n_trades = df["actions"].apply(
                lambda acts: sum(1 for a in acts.values() if isinstance(a, dict) and a.get("type") in ("buy", "sell"))
            ).sum()

        sharpe        = (df["reward"].mean() / (df["reward"].std() + 1e-8)) * np.sqrt(252) #Giorni di trading in un anno

        print("\n" + "="*55)
        print(f"  EPISODE SUMMARY")
        print(f"  Period : {df['date'].iloc[0].strftime('%Y-%m-%d')}  →  {df['date'].iloc[-1].strftime('%Y-%m-%d')} total {len(df)} steps")
        print(f"  Steps  : {len(df)}")
        print(f"  Return : {total_return:+.2%}")
        print(f"  Max DD : {max_drawdown:.2%}")
        print(f"  Sharpe : {sharpe:.3f}  (annualised, daily)")
        print(f"  Trades : {n_trades}")
        print("="*55 + "\n")

    def get_history(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)
    def get_total_shares_bought(self) -> dict[str, int]:
        return self.total_shares_bought.copy()
    def get_total_shares_sold(self) -> dict[str, int]:
        return self.total_shares_sold.copy()