"""
modelli/alpaca_live.py — Paper trading live su Alpaca

Entry point pubblico: run_alpaca(cfg)
Chiamato da main.py con:  uv run main.py step=alpaca frequency=minute

SETUP
─────
1. Crea account paper: https://app.alpaca.markets
2. Aggiungi al .env:
     ALPACA_API_KEY=PKxxxxxxxxxxxxxxxx
     ALPACA_API_SECRET=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
     ALPACA_BASE_URL=https://paper-api.alpaca.markets
3. Installa SDK: uv add alpaca-py

Fix applicati rispetto alla versione precedente
──────────────────────────────────────────────
FIX I  [NUOVO] transaction_cost aggiunto al dict `p` in ENTRAMBE le funzioni
       run_alpaca e run_alpaca_replay.
       Prima p.get("transaction_cost", 0.001) restituiva sempre il default 0.001
       indipendentemente dal YAML, causando un mismatch con il training
       (che usa buyer.transaction_cost). Ora p["transaction_cost"] viene letto
       da cfg.buyer.transaction_cost e usato ovunque in modo consistente.

FIX J  [NUOVO] Integrazione VdW in _build_thermo() (importata da trade.py).
       Il state_dim nel live/replay è calcolato da dummy_thermo, quindi
       include automaticamente le 8 colonne VdW aggiuntive — identico
       al training. NOTA: richiede che trade.py sia già stato aggiornato.

FIX K  [ESISTENTE] RollingThermoBuffer per evitare Trust=0.500 fisso.
FIX L  [ESISTENTE] NumpyScaler per compatibilità sklearn cross-version.
FIX M  [ESISTENTE] Normalizer derivato dal nome del checkpoint DDPG.
"""

from __future__ import annotations

import os
import time
import logging
import datetime
import traceback
from typing import Optional

import numpy as np
import pandas as pd
import torch


# ─── Ticker non supportati da Alpaca ─────────────────────────────────────────
_INCOMPATIBLE = {
    "GC=F","CL=F","SI=F","NG=F","HG=F",
    "^GSPC","^IXIC","FTSEMIB.MI",
    "LDO.MI","ENI.MI","ENEL.MI","ISP.MI","RACE.MI",
    "EURUSD=X","USDJPY=X",
    "BTC-USD","ETH-USD",
    "VSLA",
}

def _filter_tradeable(tickers: list[str]) -> list[str]:
    return [t for t in tickers if t not in _INCOMPATIBLE]


# ─── [FIX L] Wrapper numpy puro per MinMaxScaler ─────────────────────────────
class _NumpyScaler:
    """Replica MinMaxScaler.transform() senza dipendenza dalla versione sklearn."""
    def __init__(self, sklearn_scaler):
        self.min_   = np.array(sklearn_scaler.data_min_,  dtype=np.float32)
        self.scale_ = np.array(sklearn_scaler.scale_,     dtype=np.float32)
        data_range  = np.array(sklearn_scaler.data_range_, dtype=np.float32)
        self.range_ = np.where(data_range == 0, 1.0, data_range)

    def transform(self, X) -> np.ndarray:
        return (np.array(X, dtype=np.float32) - self.min_) / self.range_

    def fit_transform(self, X) -> np.ndarray:
        return self.transform(X)


# ─── [FIX K] RollingThermoBuffer ─────────────────────────────────────────────
class _RollingThermoBuffer:
    """
    Accumula barre in un buffer rolling per garantire al SignalTrustEngine
    abbastanza storia (≥ trust_window barre) prima di calcolare il Trust.
    Senza questo, con ws=15 barre il Trust rimane fisso a 0.500 per tutto il replay.
    """
    def __init__(self, builder, min_bars: int = 55):
        self._builder  = builder
        self._min_bars = min_bars
        self._buffer   = pd.DataFrame()

    def update(self, new_bar: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
        self._buffer = pd.concat([self._buffer, new_bar]).tail(self._min_bars)
        if len(self._buffer) < 2:
            return pd.DataFrame()
        try:
            full_thermo = _build_thermo_live(self._builder, self._buffer, tickers)
        except Exception:
            return pd.DataFrame()
        if full_thermo is None or full_thermo.empty:
            return pd.DataFrame()
        return full_thermo.iloc[[-1]]


# ─── [FIX J] Integrazione VdW per live/replay ────────────────────────────────

def _build_thermo_live(builder, raw_df: pd.DataFrame, tickers: list[str]) -> pd.DataFrame | None:
    """
    Versione live di _build_thermo() (da trade.py).
    Chiama builder.build() e aggiunge le feature VdW se disponibili.
    Mantenuta separata per evitare import circolare con trade.py.
    """
    thermo = builder.build(raw_df, tickers)

    try:
        from modelli.thermo_vdw import compute_vdw_block

        price_candidates  = [c for c in raw_df.columns
                             if any(k in c.lower() for k in ("close", "price", "adj"))]
        volume_candidates = [c for c in raw_df.columns if "vol" in c.lower()]
        price_col  = price_candidates[0]  if price_candidates  else raw_df.columns[0]
        volume_col = volume_candidates[0] if volume_candidates else None
        if tickers and tickers[0] in raw_df.columns:
            price_col = tickers[0]

        vdw = compute_vdw_block(df=raw_df, price_col=price_col,
                                volume_col=volume_col, window=20, lag_max=60)

        if thermo is not None and not thermo.empty:
            thermo = pd.concat([thermo, vdw.reindex(thermo.index)], axis=1)
        else:
            thermo = vdw

    except ImportError:
        pass
    except Exception:
        pass

    return thermo


# ─── Naming checkpoint ────────────────────────────────────────────────────────

def _pred_ckpt(cfg) -> str:
    freq = cfg.frequency.interval
    ws   = cfg.prediction.window_size
    return os.path.join(cfg.paths.checkpoint_dir, f"pred_{freq}_w{ws}.pth")

def _ddpg_ckpt(cfg) -> str:
    return os.path.join(cfg.paths.checkpoint_dir, f"ddpg_best_{cfg.frequency.interval}.pth")

def _norm_ckpt(cfg) -> str:
    return os.path.join(cfg.paths.checkpoint_dir, f"normalizer_{cfg.frequency.interval}.npz")


# ─── Logger ──────────────────────────────────────────────────────────────────

def _setup_logger(results_dir: str, freq: str) -> logging.Logger:
    log = logging.getLogger(f"AlpacaLive.{freq}")
    if log.handlers:
        return log
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    for h in [logging.StreamHandler(),
               logging.FileHandler(os.path.join(results_dir, f"alpaca_live_{freq}.log"),
                                   encoding="utf-8")]:
        h.setFormatter(fmt)
        log.addHandler(h)
    return log


# ─── Connessione Alpaca ───────────────────────────────────────────────────────

def _init_alpaca(log: logging.Logger):
    try:
        from alpaca.trading.client import TradingClient
        from alpaca.data.historical import StockHistoricalDataClient
    except ImportError:
        raise ImportError("alpaca-py non trovato. Installa con: uv add alpaca-py")

    key    = os.environ.get("ALPACA_API_KEY", "")
    secret = os.environ.get("ALPACA_API_SECRET", "")
    url    = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

    if not key or not secret:
        raise RuntimeError(
            "Credenziali Alpaca mancanti nel .env:\n"
            "  ALPACA_API_KEY=PKxxxxxxxxxxxxxxxx\n"
            "  ALPACA_API_SECRET=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        )

    is_paper = "paper" in url.lower()
    log.info(f"Alpaca | paper={is_paper} | {url}")

    tc  = TradingClient(key, secret, paper=is_paper)
    dc  = StockHistoricalDataClient(key, secret)
    acc = tc.get_account()
    log.info(f"Account OK | status={acc.status} | equity=${float(acc.equity):,.2f}")
    return tc, dc


# ─── Dati ─────────────────────────────────────────────────────────────────────

def _fetch_bars(dc, tickers: list[str], n_bars: int,
                freq: str, log: logging.Logger) -> pd.DataFrame:
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    _tf = {"1m": TimeFrame(1, TimeFrameUnit.Minute),
           "2m": TimeFrame(2, TimeFrameUnit.Minute),
           "5m": TimeFrame(5, TimeFrameUnit.Minute),
           "15m": TimeFrame(15, TimeFrameUnit.Minute),
           "1h": TimeFrame(1, TimeFrameUnit.Hour),
           "1d": TimeFrame(1, TimeFrameUnit.Day)}
    tf  = _tf.get(freq, TimeFrame(2, TimeFrameUnit.Minute))
    end = datetime.datetime.now(datetime.timezone.utc)

    try:
        bars = dc.get_stock_bars(StockBarsRequest(
            symbol_or_symbols=tickers, timeframe=tf,
            start=end - datetime.timedelta(days=10), end=end,
            adjustment="all", feed="iex",
        ))
        df = bars.df
    except Exception as e:
        log.error(f"Fetch barre: {e}")
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    df = df["close"].unstack(level=0) if isinstance(df.index, pd.MultiIndex) \
         else df[["close"]]
    df = df.reindex(columns=tickers).ffill().bfill()
    return df.iloc[-n_bars:] if len(df) > n_bars else df


def _fetch_account(tc) -> dict:
    acc = tc.get_account()
    pos = {p.symbol: {"qty": float(p.qty),
                      "market_value": float(p.market_value),
                      "avg_cost": float(p.avg_entry_price),
                      "unrealized_pl": float(p.unrealized_pl)}
           for p in tc.get_all_positions()}
    return {"equity": float(acc.equity), "cash": float(acc.cash),
            "positions": pos,
            "daily_pl": float(acc.equity) - float(acc.last_equity)}


# ─── Modelli ──────────────────────────────────────────────────────────────────

def _load_predictor(cfg, log: logging.Logger):
    from modelli.pred import Pred
    from modelli.device_setup import get_device, get_map_location

    path = _pred_ckpt(cfg)
    if not os.path.exists(path):
        raise FileNotFoundError(f"CNN non trovato: {path}")

    device = get_device()
    ck     = torch.load(path, map_location=get_map_location(), weights_only=False)
    nf     = ck["num_features"]
    mc     = ck.get("config", {}).get("model", {})

    model = Pred(
        num_features     = nf,
        window_size      = cfg.prediction.window_size,
        dimension        = mc.get("dimensions",      list(cfg.model.dimensions)),
        dilations        = mc.get("dilations",        list(cfg.model.dilations)),
        kernel_size      = mc.get("kernel_size",      cfg.model.kernel_size),
        activation       = mc.get("activation",       cfg.model.activation),
        prediction_steps = mc.get("prediction_steps", cfg.model.prediction_steps),
    ).to(device)

    state_dict = ck["model_state_dict"]
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    _mre_buffers = ["prior_mean", "prior_std", "moment_target"]
    for buf_name in _mre_buffers:
        if buf_name in missing:
            setattr(model, buf_name, None)

    if unexpected:
        log.warning(f"CNN: {len(unexpected)} chiavi ignorate: {unexpected[:5]}")
    model.eval()

    # [FIX L] NumpyScaler
    scaler = _NumpyScaler(ck["scaler"])
    log.info(f"CNN: {os.path.basename(path)} | features={nf}")
    return model, scaler, nf, device


def _load_agent(cfg, state_dim: int, all_tickers: list[str],
                tradeable: list[str], device, log: logging.Logger):
    from modelli.ddpg import DDPGAgent
    from modelli.obs_normalizer import ObsNormalizer

    dp  = _ddpg_ckpt(cfg)
    np_ = _norm_ckpt(cfg)
    dc  = cfg.buyer.ddpg

    if not os.path.exists(dp):
        raise FileNotFoundError(f"DDPG non trovato: {dp}")

    action_dim = len(all_tickers)

    agent = DDPGAgent(
        state_dim=state_dim, action_dim=action_dim,
        actor_hidden=list(dc.actor_hidden), critic_hidden=list(dc.critic_hidden),
        lr_actor=dc.lr_actor, lr_critic=dc.lr_critic,
        gamma=dc.gamma, tau=dc.tau,
        buffer_capacity=1, batch_size=1,
        update_every=999_999, noise_sigma=0.0,
    )
    loaded = agent.load(dp)
    if not loaded:
        raise RuntimeError(f"DDPG checkpoint incompatibile: {dp}")
    log.info(f"DDPG: {os.path.basename(dp)} | state={state_dim} action={action_dim}")

    norm = ObsNormalizer(shape=state_dim, clip=5.0)

    # [FIX M] Normalizer derivato dal nome del checkpoint
    ddpg_basename = os.path.basename(dp)
    if ddpg_basename.startswith("ddpg_") and ddpg_basename.endswith(".pth"):
        suffix            = ddpg_basename[5:-4]
        derived_norm_path = os.path.join(cfg.paths.checkpoint_dir, f"normalizer_{suffix}.npz")
    else:
        derived_norm_path = np_

    norm_candidates = list(dict.fromkeys([
        derived_norm_path, np_, np_.replace(".npz", "_final.npz")
    ]))

    loaded_norm = False
    for nc in norm_candidates:
        if os.path.exists(nc):
            norm.load(nc)
            loaded_norm = True
            log.info(f"✅ Normalizer: {os.path.basename(nc)}")
            break

    if not loaded_norm:
        log.warning(f"⚠️  Normalizer non trovato. Provati: {[os.path.basename(nc) for nc in norm_candidates]}")

    tradeable_indices = [i for i, t in enumerate(all_tickers) if t in set(tradeable)]
    log.info(f"Tradeable: {len(tradeable)}/{len(all_tickers)}")
    return agent, norm, tradeable_indices


# ─── Stato ───────────────────────────────────────────────────────────────────

def _build_state(bars_scaled: np.ndarray, pred_scaled: np.ndarray,
                 account: dict, all_tickers: list[str], tradeable: list[str],
                 thermo_df: pd.DataFrame, num_features: int,
                 initial_equity: float) -> np.ndarray:
    obs          = bars_scaled[-1].astype(np.float32)
    eq           = account["equity"]
    pos          = account["positions"]
    cash         = account["cash"]
    holdings_val = sum(pos.get(t, {}).get("market_value", 0.0) for t in tradeable)
    port_state   = np.array([
        cash         / (initial_equity + 1e-8),
        holdings_val / (initial_equity + 1e-8),
    ], dtype=np.float32)
    thermo_state = thermo_df.iloc[-1].values.astype(np.float32) \
                   if not thermo_df.empty else np.array([], dtype=np.float32)
    return np.concatenate([obs, port_state, thermo_state]).astype(np.float32)


# ─── Ordini ───────────────────────────────────────────────────────────────────

def _execute_orders(tc, actions: np.ndarray, tradeable: list[str],
                    account: dict, p: dict, n_today: int,
                    log: logging.Logger) -> tuple[list[dict], int]:
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce

    eq, cash, pos = account["equity"], account["cash"], account["positions"]
    orders: list[dict] = []

    if n_today >= p["max_daily_trades"]:
        log.warning(f"Limite giornaliero raggiunto ({p['max_daily_trades']})")
        return orders, n_today

    for i, ticker in enumerate(tradeable):
        if i >= len(actions):
            break
        a = float(actions[i])
        if abs(a) < p["action_threshold"] or abs(a) < p["min_confidence"]:
            continue

        cur_val = pos.get(ticker, {}).get("market_value", 0.0)

        try:
            if a > p["action_threshold"]:
                notional = min(cash * a, max(0.0, eq * p["max_position_pct"] - cur_val))
                if notional < p["min_order_usd"]:
                    continue
                o = tc.submit_order(MarketOrderRequest(
                    symbol=ticker, notional=round(notional, 2),
                    side=OrderSide.BUY, time_in_force=TimeInForce.DAY,
                ))
                n_today += 1
                log.info(f"  ✅ BUY  {ticker:8s} ${notional:7.2f} | a={a:+.3f}")
                orders.append({"ticker": ticker, "side": "BUY",
                               "notional": notional, "action": a, "id": str(o.id)})

            elif a < -p["action_threshold"] and cur_val > p["min_order_usd"]:
                notional = cur_val * abs(a)
                if notional < p["min_order_usd"]:
                    continue
                o = tc.submit_order(MarketOrderRequest(
                    symbol=ticker, notional=round(notional, 2),
                    side=OrderSide.SELL, time_in_force=TimeInForce.DAY,
                ))
                n_today += 1
                log.info(f"  🔴 SELL {ticker:8s} ${notional:7.2f} | a={a:+.3f}")
                orders.append({"ticker": ticker, "side": "SELL",
                               "notional": notional, "action": a, "id": str(o.id)})

        except Exception as e:
            log.error(f"  Ordine {ticker}: {e}")

    return orders, n_today


# ─── Circuit Breaker ──────────────────────────────────────────────────────────

class _CB:
    def __init__(self, eq0: float, thr: float):
        self.peak = eq0; self.thr = thr; self.on = False

    def check(self, eq: float, tc, log: logging.Logger) -> bool:
        if self.on:
            return True
        if eq > self.peak:
            self.peak = eq
        dd = (eq - self.peak) / (self.peak + 1e-8)
        if dd < self.thr:
            log.warning(f"⚠️  CIRCUIT BREAKER | dd={dd:.1%}")
            try:
                tc.close_all_positions(cancel_orders=True)
            except Exception as e:
                log.error(f"Liquidazione: {e}")
            self.on = True
        return self.on


# ─── Live Log ─────────────────────────────────────────────────────────────────

class _LiveLog:
    def __init__(self, path: str):
        self.path = path; self._rows: list[dict] = []; self._eq0: Optional[float] = None

    def record(self, ts, acc, actions, tradeable, thermo_df, orders) -> None:
        eq = acc["equity"]
        if self._eq0 is None:
            self._eq0 = eq
        row = {"ts": ts if isinstance(ts, str) else ts.isoformat(),
               "equity": round(eq, 2), "cash": round(acc["cash"], 2),
               "daily_pl": round(acc["daily_pl"], 2),
               "ret_pct": round((eq / self._eq0 - 1) * 100, 4),
               "n_orders": len(orders)}
        for i, t in enumerate(tradeable):
            row[f"a_{t}"] = round(float(actions[i]), 4) if i < len(actions) else 0.0
        for t in tradeable:
            row[f"pos_{t}"] = round(acc["positions"].get(t, {}).get("market_value", 0.0), 2)
        for c in ["Thm_Stress","Thm_Efficiency","Thm_Regime","Thm_Pressure",
                  "Thm_VdW_P","Thm_Work","Thm_ZStress"]:
            if not thermo_df.empty and c in thermo_df.columns:
                row[c] = round(float(thermo_df[c].iloc[-1]), 4)
        self._rows.append(row)
        if len(self._rows) % 5 == 0:
            pd.DataFrame(self._rows).to_csv(self.path, index=False)

    def close(self) -> None:
        if self._rows:
            pd.DataFrame(self._rows).to_csv(self.path, index=False)


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT — LIVE
# ═══════════════════════════════════════════════════════════════════════════════

def run_alpaca(cfg) -> None:
    freq = cfg.frequency.interval
    os.makedirs(cfg.paths.results_dir, exist_ok=True)
    log = _setup_logger(cfg.paths.results_dir, freq)

    log.info("=" * 62)
    log.info(f"  ALPACA PAPER TRADING | freq={freq} | window={cfg.prediction.window_size}")
    log.info("=" * 62)

    ac = getattr(cfg, "alpaca", None)
    def _g(k, d): return getattr(ac, k, d) if ac else d

    # [FIX I] transaction_cost ora letto dal YAML (buyer.transaction_cost)
    # invece di essere hardcoded a 0.001, garantendo coerenza con il training.
    p = {
        "max_position_pct":   _g("max_position_pct",   0.05),
        "min_order_usd":      _g("min_order_usd",       1.0),
        "action_threshold":   _g("action_threshold",    0.005),
        "min_confidence":     _g("min_confidence",      0.10),
        "circuit_breaker_dd": _g("circuit_breaker_dd", -0.15),
        "max_daily_trades":   _g("max_daily_trades",    50),
        "warmup_bars":        _g("warmup_bars",         cfg.prediction.window_size + 5),
        "bar_buffer":         _g("bar_buffer",          200),
        "transaction_cost":   float(getattr(cfg.buyer, "transaction_cost", 0.001)),  # FIX I
    }

    all_tickers = list(cfg.data.tickers)
    tradeable   = list(_g("tradeable_tickers", None) or _filter_tradeable(all_tickers))
    log.info(f"Ticker: {len(tradeable)} tradabili / {len(all_tickers)} training")
    log.info(f"transaction_cost={p['transaction_cost']:.4f} (dal YAML buyer.transaction_cost)")

    predictor, scaler, num_features, device = _load_predictor(cfg, log)

    from modelli.thermo_state_builder import ThermoStateBuilder
    thermo_builder = ThermoStateBuilder(interval=freq)

    # [FIX J] dummy_thermo include VdW → state_dim corretto
    dummy_bars = pd.DataFrame(index=[datetime.datetime.now()], columns=all_tickers).fillna(1.0)
    try:
        dummy_thermo = _build_thermo_live(thermo_builder, dummy_bars, all_tickers)
        N_THERMO = len(dummy_thermo.columns) if dummy_thermo is not None else 0
    except Exception:
        N_THERMO = 45  # fallback (37 base + 8 VdW)

    state_dim = num_features + 2 + N_THERMO
    log.info(f"state_dim={state_dim} (F={num_features} port=2 thm={N_THERMO})")

    agent, normalizer, tradeable_indices = _load_agent(
        cfg, state_dim, all_tickers, tradeable, device, log
    )

    tc, dc = _init_alpaca(log)

    account  = _fetch_account(tc)
    eq0      = account["equity"]
    cb       = _CB(eq0, p["circuit_breaker_dd"])
    live_log = _LiveLog(os.path.join(cfg.paths.results_dir, f"alpaca_live_log_{freq}.csv"))

    interval_sec = {"1m":60,"2m":120,"5m":300,"15m":900,"1h":3600,"1d":86400}.get(freq,120)
    n_today      = 0
    last_date: Optional[datetime.date] = None
    step_n       = 0

    # [FIX K] Buffer rolling per thermo
    trust_window = thermo_builder.trust_window
    ws_live      = cfg.prediction.window_size
    thermo_buf   = _RollingThermoBuffer(thermo_builder, min_bars=trust_window + ws_live + 5)
    log.info(f"RollingThermoBuffer: min_bars={trust_window + ws_live + 5}")
    log.info(f"Equity iniziale: ${eq0:,.2f} | CB: {p['circuit_breaker_dd']:.0%}")
    log.info("Avvio loop...")

    _THERMO_COLS = ["Thm_Pressure","Thm_Temperature","Thm_Work",
                    "Thm_Stress","Thm_Efficiency","Thm_Entropy","Thm_Regime"]
    all_cols_base = all_tickers + _THERMO_COLS

    while True:
        try:
            now   = datetime.datetime.now(datetime.timezone.utc)
            today = now.date()

            if last_date != today:
                n_today = 0; last_date = today
                log.info(f"Nuovo giorno: {today}")

            try:
                if not tc.get_clock().is_open:
                    log.info("Mercato chiuso — attendo 5 min")
                    time.sleep(300); continue
            except Exception:
                pass

            elapsed = (now.minute * 60 + now.second) % interval_sec
            time.sleep(max(1.0, interval_sec - elapsed + 5))

            step_n += 1
            log.info(f"\n{'─'*55}")
            log.info(f"  Step {step_n} | {datetime.datetime.now(datetime.timezone.utc).strftime('%H:%M:%S UTC')}")

            bars = _fetch_bars(dc, tradeable, p["bar_buffer"], freq, log)
            if bars.empty or len(bars) < p["warmup_bars"]:
                log.info(f"Warm-up: {len(bars)}/{p['warmup_bars']}"); continue

            try:
                thermo_df = thermo_buf.update(bars.iloc[[-1]], tradeable)
                if thermo_df.empty:
                    thermo_df = _build_thermo_live(thermo_builder, bars, tradeable) or pd.DataFrame()
            except Exception as e:
                log.warning(f"Thermo: {e}")
                thermo_df = pd.DataFrame()

            full = pd.DataFrame(0.0, index=bars.index, columns=all_cols_base)
            for t in tradeable:
                if t in bars.columns and t in all_tickers:
                    full[t] = bars[t].values
            if not thermo_df.empty:
                for col in _THERMO_COLS:
                    if col in thermo_df.columns:
                        full[col] = thermo_df[col].reindex(full.index).ffill().bfill().fillna(0.0).values

            try:
                scaled = scaler.transform(full)
            except Exception as e:
                log.error(f"Scaling: {e}"); continue

            ws = cfg.prediction.window_size
            if len(scaled) < ws:
                log.warning(f"Barre insufficienti ({len(scaled)} < {ws})"); continue

            with torch.no_grad():
                win = torch.tensor(scaled[-ws:][np.newaxis], dtype=torch.float32).to(device)
                out = predictor(win)
                if out.dim() == 3:
                    out = out[:, 0, :]
                pred = out.cpu().numpy()

            account = _fetch_account(tc)
            if cb.check(account["equity"], tc, log):
                log.error("Circuit breaker — sospeso"); time.sleep(interval_sec); continue

            state_raw = _build_state(scaled, pred, account, all_tickers, tradeable,
                                     thermo_df, num_features, eq0)

            if len(state_raw) != state_dim:
                log.warning(f"state mismatch {len(state_raw)} vs {state_dim}")
                state_raw = (np.pad(state_raw, (0, state_dim - len(state_raw)))
                             if len(state_raw) < state_dim else state_raw[:state_dim])

            actions_full = agent.act(normalizer.normalize(state_raw, update=False), explore=False)
            actions      = actions_full[tradeable_indices]

            sig = [(tradeable[i], float(a)) for i, a in enumerate(actions)
                   if abs(a) > p["action_threshold"]]
            if sig:
                log.info(f"Segnali ({len(sig)}):")
                for t, a in sig:
                    log.info(f"    {'🟢 BUY ' if a > 0 else '🔴 SELL'} {t:8s} {a:+.4f}")
            else:
                log.info("HOLD")

            if not thermo_df.empty:
                s  = float(thermo_df.get("Thm_Stress",  pd.Series([0])).iloc[-1])
                rg = int(thermo_df.get("Thm_Regime", pd.Series([0])).iloc[-1])
                rname = {0:"NEUTRO",1:"RALLY_REALE",2:"RALLY_ESAUSTO",
                         3:"COMPRESSIONE",4:"RIMBALZO"}.get(rg,"?")
                log.info(f"Thermo: stress={s:+.3f} | {rname}")

            orders, n_today = _execute_orders(tc, actions, tradeable,
                                              account, p, n_today, log)
            live_log.record(now, account, actions, tradeable, thermo_df, orders)

            ret = (account["equity"] / eq0 - 1) * 100
            log.info(f"Portfolio: ${account['equity']:,.2f} ({ret:+.2f}%) | "
                     f"Cash: ${account['cash']:,.2f} | Ordini oggi: {n_today}")

        except KeyboardInterrupt:
            log.info("Interrotto (Ctrl+C)"); break
        except Exception as e:
            log.error(f"Errore step: {e}\n{traceback.format_exc()}")
            time.sleep(30)

    live_log.close()
    log.info("Sessione terminata.")


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT — REPLAY
# ═══════════════════════════════════════════════════════════════════════════════

def _fetch_historical_bars(dc, tickers, freq, start, end, log) -> pd.DataFrame:
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    _tf = {"1m": TimeFrame(1, TimeFrameUnit.Minute),
           "2m": TimeFrame(2, TimeFrameUnit.Minute),
           "5m": TimeFrame(5, TimeFrameUnit.Minute),
           "15m": TimeFrame(15, TimeFrameUnit.Minute),
           "1h": TimeFrame(1, TimeFrameUnit.Hour),
           "1d": TimeFrame(1, TimeFrameUnit.Day)}
    tf = _tf.get(freq, TimeFrame(2, TimeFrameUnit.Minute))

    log.info(f"Download storico | {start.date()} → {end.date()} | freq={freq}")
    try:
        bars = dc.get_stock_bars(StockBarsRequest(
            symbol_or_symbols=tickers, timeframe=tf,
            start=start, end=end, adjustment="all", feed="iex",
        ))
        df = bars.df
    except Exception as e:
        raise RuntimeError(f"Download storico fallito: {e}") from e

    if df.empty:
        raise RuntimeError("Nessuna barra storica ricevuta da Alpaca")

    df = df["close"].unstack(level=0) if isinstance(df.index, pd.MultiIndex) \
         else df[["close"]]
    df = df.reindex(columns=tickers).ffill().bfill().sort_index()
    log.info(f"Scaricate {len(df):,} barre ({df.index[0]} → {df.index[-1]})")
    return df


def _simulate_account(positions: dict[str, float], cash: float,
                       current_prices: dict[str, float]) -> dict:
    holdings_val = {t: shares * current_prices.get(t, 0.0)
                    for t, shares in positions.items()}
    equity = cash + sum(holdings_val.values())
    return {
        "equity": equity, "cash": cash, "daily_pl": 0.0,
        "positions": {
            t: {"qty": shares, "market_value": holdings_val[t],
                "avg_cost": 0.0, "unrealized_pl": 0.0}
            for t, shares in positions.items() if shares > 0
        }
    }


def _replay_execute(actions: np.ndarray, tradeable: list[str],
                    positions: dict[str, float], cash: float,
                    prices: dict[str, float], p: dict) -> tuple[dict, float, list[dict]]:
    """
    [FIX I] tc_cost ora viene da p["transaction_cost"] (letto dal YAML),
    non più dal default hardcoded 0.001.
    """
    equity  = cash + sum(positions.get(t, 0.0) * prices.get(t, 0.0) for t in tradeable)
    orders: list[dict] = []
    tc_cost = p["transaction_cost"]  # FIX I — era p.get("transaction_cost", 0.001)

    for i, ticker in enumerate(tradeable):
        if i >= len(actions):
            break
        a     = float(actions[i])
        price = prices.get(ticker, 0.0)
        if price <= 0 or abs(a) < p["action_threshold"] or abs(a) < p["min_confidence"]:
            continue

        cur_val = positions.get(ticker, 0.0) * price

        if a > p["action_threshold"]:
            max_pos  = equity * p["max_position_pct"]
            notional = min(cash * a, max(0.0, max_pos - cur_val))
            if notional < p["min_order_usd"]:
                continue
            shares = notional * (1 - tc_cost) / price
            positions[ticker] = positions.get(ticker, 0.0) + shares
            cash -= notional
            orders.append({"ticker": ticker, "side": "BUY",
                           "notional": round(notional, 2), "action": a})

        elif a < -p["action_threshold"] and cur_val > p["min_order_usd"]:
            notional    = cur_val * abs(a)
            if notional < p["min_order_usd"]:
                continue
            shares_sold = notional / price
            positions[ticker] = max(0.0, positions.get(ticker, 0.0) - shares_sold)
            cash += notional * (1 - tc_cost)
            orders.append({"ticker": ticker, "side": "SELL",
                           "notional": round(notional, 2), "action": a})

    return positions, cash, orders


def run_alpaca_replay(cfg) -> None:
    freq = cfg.frequency.interval
    os.makedirs(cfg.paths.results_dir, exist_ok=True)
    log = _setup_logger(cfg.paths.results_dir, f"replay_{freq}")

    log.info("=" * 62)
    log.info(f"  ALPACA REPLAY | freq={freq} | window={cfg.prediction.window_size}")
    log.info("=" * 62)

    rc = getattr(cfg, "alpaca_replay", None)
    def _r(k, d): return getattr(rc, k, d) if rc else d

    days_back     = int(_r("days_back",      4))
    bar_delay_sec = float(_r("bar_delay_sec", 0.05))
    submit_orders = bool(_r("submit_orders",  False))

    now = datetime.datetime.now(datetime.timezone.utc)
    if _r("end_date", None):
        end_dt = datetime.datetime.fromisoformat(str(_r("end_date", None))).replace(
            tzinfo=datetime.timezone.utc)
    else:
        end_dt = now

    if _r("start_date", None):
        start_dt = datetime.datetime.fromisoformat(str(_r("start_date", None))).replace(
            tzinfo=datetime.timezone.utc)
    else:
        start_dt = end_dt - datetime.timedelta(days=days_back)

    ac = getattr(cfg, "alpaca", None)
    def _g(k, d): return getattr(ac, k, d) if ac else d

    # [FIX I] transaction_cost letto dal YAML — identico al training
    p = {
        "max_position_pct":  _g("max_position_pct",  0.05),
        "min_order_usd":     _g("min_order_usd",      1.0),
        "action_threshold":  _g("action_threshold",   0.005),
        "min_confidence":    _g("min_confidence",     0.10),
        "circuit_breaker_dd": _g("circuit_breaker_dd", -0.15),
        "transaction_cost":  float(getattr(cfg.buyer, "transaction_cost", 0.001)),  # FIX I
    }

    all_tickers = list(cfg.data.tickers)
    tradeable   = list(_g("tradeable_tickers", None) or _filter_tradeable(all_tickers))

    log.info(f"Range: {start_dt.date()} → {end_dt.date()} | bar_delay={bar_delay_sec}s")
    log.info(f"transaction_cost={p['transaction_cost']:.4f} (dal YAML, identico al training)")

    bars_per_day = {"1m":390,"2m":195,"5m":78,"15m":26,"1h":7,"1d":1}.get(freq, 195)
    est_bars    = days_back * bars_per_day
    est_seconds = est_bars * bar_delay_sec
    speedup     = (days_back * 86400) / max(est_seconds, 1)
    log.info(f"Stima: ~{est_bars:,} barre | ~{est_seconds:.0f}s | {speedup:.0f}x speedup")

    predictor, scaler, num_features, device = _load_predictor(cfg, log)

    from modelli.thermo_state_builder import ThermoStateBuilder
    thermo_builder = ThermoStateBuilder(interval=freq)

    # [FIX J] dummy_thermo include VdW per state_dim corretto
    dummy_bars = pd.DataFrame(index=[datetime.datetime.now()], columns=all_tickers).fillna(1.0)
    try:
        dummy_thermo = _build_thermo_live(thermo_builder, dummy_bars, all_tickers)
        N_THERMO = len(dummy_thermo.columns) if dummy_thermo is not None else 0
    except Exception:
        N_THERMO = 45

    state_dim = num_features + 2 + N_THERMO
    log.info(f"state_dim={state_dim} (F={num_features} port=2 thm={N_THERMO})")

    agent, normalizer, tradeable_indices = _load_agent(
        cfg, state_dim, all_tickers, tradeable, device, log
    )

    tc, dc = _init_alpaca(log)
    all_bars = _fetch_historical_bars(dc, tradeable, freq, start_dt, end_dt, log)

    ws = cfg.prediction.window_size
    if len(all_bars) < ws + 5:
        log.error(f"Barre insufficienti: {len(all_bars)} < {ws + 5}")
        return

    # [FIX K] Buffer rolling
    trust_window = thermo_builder.trust_window
    thermo_buf   = _RollingThermoBuffer(thermo_builder, min_bars=trust_window + ws + 5)

    initial_capital = float(cfg.buyer.initial_capital)
    positions: dict[str, float] = {}
    cash     = initial_capital
    eq0      = initial_capital

    live_log       = _LiveLog(os.path.join(cfg.paths.results_dir,
                                           f"alpaca_replay_log_{freq}.csv"))
    portfolio_hist: list[float] = [initial_capital]
    trade_log_rows: list[dict]  = []

    n_buys = n_sells = 0
    step_n = 0
    peak_eq = initial_capital

    _THERMO_COLS  = ["Thm_Pressure","Thm_Temperature","Thm_Work",
                     "Thm_Stress","Thm_Efficiency","Thm_Entropy","Thm_Regime"]
    all_cols_base = all_tickers + _THERMO_COLS

    log.info(f"\nInizio replay — {len(all_bars):,} barre | capitale: ${initial_capital:,.2f}")

    for bar_i in range(ws, len(all_bars)):
        step_n    += 1
        window_raw = all_bars.iloc[bar_i - ws: bar_i]
        ts         = all_bars.index[bar_i]

        # [FIX K] Buffer rolling per thermo
        try:
            thermo_df = thermo_buf.update(all_bars.iloc[[bar_i]], tradeable)
        except Exception:
            thermo_df = pd.DataFrame()

        full = pd.DataFrame(0.0, index=window_raw.index, columns=all_cols_base)
        for t in tradeable:
            if t in window_raw.columns and t in all_tickers:
                full[t] = window_raw[t].values
        if not thermo_df.empty:
            for col in _THERMO_COLS:
                if col in thermo_df.columns:
                    full[col] = thermo_df[col].reindex(full.index).ffill().bfill().fillna(0.0).values

        try:
            scaled = scaler.transform(full)
        except Exception as e:
            log.warning(f"Scaling step {step_n}: {e}"); continue

        with torch.no_grad():
            win_t = torch.tensor(scaled[np.newaxis], dtype=torch.float32).to(device)
            out   = predictor(win_t)
            if out.dim() == 3:
                out = out[:, 0, :]
            pred = out.cpu().numpy()

        prices  = {t: float(all_bars[t].iloc[bar_i])
                   for t in tradeable if t in all_bars.columns}
        account = _simulate_account(positions, cash, prices)
        equity  = account["equity"]
        portfolio_hist.append(equity)

        if equity > peak_eq:
            peak_eq = equity
        dd = (equity - peak_eq) / (peak_eq + 1e-8)
        if dd < p["circuit_breaker_dd"]:
            log.warning(f"⚠️  Circuit breaker | dd={dd:.1%}")
            break

        state_raw = _build_state(scaled, pred, account, all_tickers, tradeable,
                                 thermo_df, num_features, eq0)
        if len(state_raw) != state_dim:
            state_raw = (np.pad(state_raw, (0, state_dim - len(state_raw)))
                         if len(state_raw) < state_dim else state_raw[:state_dim])

        actions_full = agent.act(normalizer.normalize(state_raw, update=False), explore=False)
        actions      = actions_full[tradeable_indices]

        # [FIX I] _replay_execute usa p["transaction_cost"] dal YAML
        positions, cash, orders = _replay_execute(
            actions, tradeable, positions, cash, prices, p
        )
        n_buys  += sum(1 for o in orders if o["side"] == "BUY")
        n_sells += sum(1 for o in orders if o["side"] == "SELL")

        if submit_orders and orders:
            _execute_orders(tc, actions, tradeable, account, p, 0, log)

        for o in orders:
            trade_log_rows.append({
                "ts": ts, "ticker": o["ticker"], "side": o["side"],
                "notional": o["notional"], "action": round(o["action"], 4),
                "equity": round(equity, 2), "cash": round(cash, 2),
            })

        live_log.record(ts, _simulate_account(positions, cash, prices),
                        actions, tradeable, thermo_df, orders)

        if step_n % 100 == 0:
            ret = (equity / eq0 - 1) * 100
            log.info(f"  Step {step_n:5d}/{len(all_bars)-ws} | "
                     f"{ts.strftime('%m-%d %H:%M')} | "
                     f"${equity:,.2f} ({ret:+.2f}%) | B:{n_buys} S:{n_sells}")

        time.sleep(bar_delay_sec)

    # ── Report finale ──────────────────────────────────────────────────────────
    live_log.close()

    hist    = np.array(portfolio_hist, dtype=np.float64)
    final_v = hist[-1]
    ret_pct = (final_v / initial_capital - 1) * 100

    bars_per_year = int(getattr(cfg.buyer, "bars_per_year", 98280))
    returns  = np.diff(hist) / (hist[:-1] + 1e-8)
    sharpe   = (float(returns.mean()) / (float(returns.std()) + 1e-8)
                * np.sqrt(bars_per_year)) if len(returns) > 1 else 0.0
    peak     = np.maximum.accumulate(hist)
    max_dd   = float(((hist - peak) / (peak + 1e-8)).min())

    sep = "─" * 58
    log.info(f"\n{sep}")
    log.info(f"  REPLAY COMPLETATO")
    log.info(sep)
    log.info(f"  Range dati        : {start_dt.date()} → {end_dt.date()}")
    log.info(f"  Barre riprodotte  : {step_n:,}")
    log.info(f"  Durata replay     : {step_n * bar_delay_sec:.1f}s "
             f"(vs {days_back * 86400:,}s reali → {speedup:.0f}x)")
    log.info(f"  Capitale iniziale : ${initial_capital:>10,.2f}")
    log.info(f"  Valore finale     : ${final_v:>10,.2f}")
    log.info(f"  Rendimento        : {ret_pct:>+10.2f}%")
    log.info(f"  Sharpe ratio      : {sharpe:>10.3f}")
    log.info(f"  Max Drawdown      : {max_dd:>10.3f}")
    log.info(f"  Operazioni BUY    : {n_buys:>10,}")
    log.info(f"  Operazioni SELL   : {n_sells:>10,}")
    log.info(f"  Transaction cost  : {p['transaction_cost']:.4f} (dal YAML)")
    log.info(sep)

    if trade_log_rows:
        tlog = os.path.join(cfg.paths.results_dir, f"alpaca_replay_trades_{freq}.csv")
        pd.DataFrame(trade_log_rows).to_csv(tlog, index=False)
        log.info(f"  Trade log: {tlog}")

    ph_path = os.path.join(cfg.paths.results_dir, f"alpaca_replay_portfolio_{freq}.csv")
    pd.Series(hist, name="equity").to_csv(ph_path)
    log.info(f"  Portfolio: {ph_path}")
    log.info(sep)