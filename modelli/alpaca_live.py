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

PARAMETRI CONFIG (opzionali in config.yaml)
────────────────────────────────────────────
alpaca:
  max_position_pct:    0.05   # max % portafoglio per ticker
  min_order_usd:       1.0    # ordine minimo in USD
  action_threshold:    0.005  # soglia sotto cui ignorare il segnale DDPG
  min_confidence:      0.10   # confidenza minima per tradare
  circuit_breaker_dd: -0.15   # ferma tutto se drawdown supera 15%
  max_daily_trades:    50     # limite ordini giornalieri
  warmup_bars:         20     # barre prima del primo trade
  bar_buffer:          200    # barre storiche in memoria
  tradeable_tickers:   []     # lista esplicita ticker (default: auto da cfg.data.tickers)
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
# Futures, indici, forex, ETF europei vengono esclusi automaticamente.
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


# ─── Naming checkpoint (speculare a main.py) ──────────────────────────────────

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


# ═══════════════════════════════════════════════════════════════════════════════
# CONNESSIONE ALPACA
# ═══════════════════════════════════════════════════════════════════════════════

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
            "  ALPACA_API_SECRET=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n"
            "  ALPACA_BASE_URL=https://paper-api.alpaca.markets"
        )

    is_paper = "paper" in url.lower()
    log.info(f"Alpaca | paper={is_paper} | {url}")

    tc = TradingClient(key, secret, paper=is_paper)
    dc = StockHistoricalDataClient(key, secret)

    acc = tc.get_account()
    log.info(f"Account OK | status={acc.status} | equity=${float(acc.equity):,.2f}")

    return tc, dc


# ═══════════════════════════════════════════════════════════════════════════════
# DATI
# ═══════════════════════════════════════════════════════════════════════════════

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
            adjustment="all",
            feed="iex",   # IEX = gratuito; SIP richiede abbonamento a pagamento
        ))
        df = bars.df
    except Exception as e:
        log.error(f"Fetch barre: {e}")
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    # MultiIndex (symbol, timestamp) → pivot
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


# ═══════════════════════════════════════════════════════════════════════════════
# MODELLI
# ═══════════════════════════════════════════════════════════════════════════════

def _load_predictor(cfg, log: logging.Logger):
    from modelli.pred import Pred
    from modelli.device_setup import get_device, get_map_location

    path = _pred_ckpt(cfg)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"CNN non trovato: {path}\n"
            f"Esegui: uv run main.py step=train frequency={cfg.frequency.interval}"
        )
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
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    log.info(f"CNN: {os.path.basename(path)} | features={nf}")
    return model, ck["scaler"], nf, device


def _load_agent(cfg, state_dim: int, all_tickers: list[str],
                tradeable: list[str], device: torch.device,
                log: logging.Logger):
    """
    Carica DDPG e normalizer.

    action_dim = len(all_tickers): identico al training dove l'agente
    produceva un'azione per ogni ticker del portafoglio.
    In produzione si eseguono solo le azioni per i ticker tradabili su Alpaca,
    usando tradeable_indices per estrarre il sottoinsieme corretto.
    """
    from modelli.ddpg import DDPGAgent
    from modelli.obs_normalizer import ObsNormalizer

    dp  = _ddpg_ckpt(cfg)
    np_ = _norm_ckpt(cfg)
    dc  = cfg.buyer.ddpg

    if not os.path.exists(dp):
        raise FileNotFoundError(
            f"DDPG non trovato: {dp}\n"
            f"Esegui: uv run main.py step=trade frequency={cfg.frequency.interval}"
        )

    # action_dim = tutti i ticker del training, NON solo i tradeable
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
        raise RuntimeError(
            f"DDPG checkpoint incompatibile: {dp}\n"
            "Verifica che state_dim e action_dim corrispondano al training."
        )
    log.info(f"DDPG: {os.path.basename(dp)} | state={state_dim} action={action_dim}")

    norm = ObsNormalizer(shape=state_dim, clip=5.0)
    
    # ★ FIX: Deriva il path del normalizer dal modello DDPG caricato
    # Esempio: ddpg_best_2m.pth → normalizer_best_2m.npz
    #          ddpg_wf_fold4_2m.pth → normalizer_wf_fold4_2m.npz
    ddpg_basename = os.path.basename(dp)  # es: "ddpg_best_2m.pth"
    
    # Estrai il suffisso tra "ddpg_" e ".pth"
    if ddpg_basename.startswith("ddpg_") and ddpg_basename.endswith(".pth"):
        suffix = ddpg_basename[5:-4]  # rimuove "ddpg_" e ".pth"
        derived_norm_name = f"normalizer_{suffix}.npz"
        derived_norm_path = os.path.join(cfg.paths.checkpoint_dir, derived_norm_name)
    else:
        # Fallback: usa il path generico
        derived_norm_path = np_
    
    # Prova in ordine:
    # 1. Normalizer derivato dal modello (es: normalizer_wf_fold4_2m.npz)
    # 2. Normalizer generico (es: normalizer_2m.npz)
    # 3. Normalizer generico con _final (es: normalizer_2m_final.npz)
    norm_candidates = [
        derived_norm_path,
        np_,
        np_.replace(".npz", "_final.npz")
    ]
    
    # Rimuovi duplicati mantenendo l'ordine
    norm_candidates = list(dict.fromkeys(norm_candidates))
    
    loaded_norm = False
    for nc in norm_candidates:
        if os.path.exists(nc):
            norm.load(nc)
            loaded_norm = True
            log.info(f"✅ Normalizer caricato: {os.path.basename(nc)}")
            break
    
    if not loaded_norm:
        log.warning(
            f"⚠️  Normalizer non trovato!\n"
            f"   Provati: {[os.path.basename(nc) for nc in norm_candidates]}\n"
            f"   Uso statistiche vuote — le predizioni potrebbero essere instabili"
        )
 

    # Indici dei ticker tradabili all'interno di all_tickers
    # Usati per estrarre le azioni corrette dal vettore completo
    tradeable_indices = [i for i, t in enumerate(all_tickers) if t in set(tradeable)]
    log.info(f"Ticker tradabili: {len(tradeable)}/{len(all_tickers)} "
             f"(indici {tradeable_indices[:3]}...)")

    return agent, norm, tradeable_indices


# ═══════════════════════════════════════════════════════════════════════════════
# STATO (identico a TradingEnv._get_state)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_state(bars_scaled: np.ndarray, pred_scaled: np.ndarray,
                 account: dict, all_tickers: list[str], tradeable: list[str],
                 thermo_df: pd.DataFrame, num_features: int,
                 initial_equity: float) -> np.ndarray:
    """
    Costruisce il vettore di stato IDENTICO a TradingEnv._get_state():

      [prices_real(F) | prices_pred(F) | holdings_ratio(ALL_T) |
       cash_ratio(1)  | portfolio_ratio(1) | thermo(7)]

    IMPORTANTE: holdings_ratio usa TUTTI i ticker del training (all_tickers),
    non solo quelli tradabili su Alpaca. I ticker non tradabili hanno holdings=0.
    Questo è fondamentale per la compatibilità con il checkpoint DDPG addestrato
    su tutti i 51 ticker.
    """
    row_real = bars_scaled[-1].astype(np.float32)
    row_pred = pred_scaled[0].astype(np.float32) if pred_scaled is not None \
               else np.zeros(num_features, np.float32)

    eq  = account["equity"]
    pos = account["positions"]

    # Holdings su TUTTI i ticker del training (non solo tradeable)
    hold_ratio = np.array([
        pos.get(t, {}).get("market_value", 0.0)
        for t in all_tickers
    ], np.float32) / (eq + 1e-8)

    cash_ratio = account["cash"] / (eq + 1e-8)
    port_ratio = eq / (initial_equity + 1e-8)

    thm_cols = [c for c in thermo_df.columns if c.startswith("Thm_")]
    thermo   = thermo_df[thm_cols].iloc[-1].values.astype(np.float32) \
               if not thermo_df.empty and thm_cols else np.zeros(7, np.float32)

    return np.concatenate([
        row_real, row_pred, hold_ratio,
        np.array([cash_ratio, port_ratio], np.float32),
        thermo,
    ]).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# ORDINI
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
# CIRCUIT BREAKER
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
# LIVE LOGGER (CSV incrementale)
# ═══════════════════════════════════════════════════════════════════════════════

class _LiveLog:
    def __init__(self, path: str):
        self.path = path; self._rows: list[dict] = []; self._eq0: Optional[float] = None

    def record(self, ts: datetime.datetime, acc: dict, actions: np.ndarray,
               tradeable: list[str], thermo_df: pd.DataFrame, orders: list[dict]) -> None:
        eq = acc["equity"]
        if self._eq0 is None:
            self._eq0 = eq
        row = {"ts": ts.isoformat(), "equity": round(eq, 2),
               "cash": round(acc["cash"], 2),
               "daily_pl": round(acc["daily_pl"], 2),
               "ret_pct": round((eq / self._eq0 - 1) * 100, 4),
               "n_orders": len(orders)}
        for i, t in enumerate(tradeable):
            row[f"a_{t}"] = round(float(actions[i]), 4) if i < len(actions) else 0.0
        for t in tradeable:
            row[f"pos_{t}"] = round(acc["positions"].get(t, {}).get("market_value", 0.0), 2)
        for c in ["Thm_Stress","Thm_Efficiency","Thm_Regime","Thm_Pressure"]:
            if not thermo_df.empty and c in thermo_df.columns:
                row[c] = round(float(thermo_df[c].iloc[-1]), 4)
        self._rows.append(row)
        if len(self._rows) % 5 == 0:
            pd.DataFrame(self._rows).to_csv(self.path, index=False)

    def close(self) -> None:
        if self._rows:
            pd.DataFrame(self._rows).to_csv(self.path, index=False)


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def run_alpaca(cfg) -> None:
    """
    Avvia il loop di paper trading su Alpaca.
    Chiamato da main.py quando step=alpaca.
    Non richiede load_data() — i dati arrivano da Alpaca in tempo reale.
    """
    freq = cfg.frequency.interval
    os.makedirs(cfg.paths.results_dir, exist_ok=True)
    log = _setup_logger(cfg.paths.results_dir, freq)

    log.info("=" * 62)
    log.info(f"  ALPACA PAPER TRADING | freq={freq} | window={cfg.prediction.window_size}")
    log.info("=" * 62)

    # ── Parametri da cfg.alpaca (tutti con default safe) ─────────────────
    ac = getattr(cfg, "alpaca", None)
    def _g(k, d): return getattr(ac, k, d) if ac else d

    p = {
        "max_position_pct":   _g("max_position_pct",   0.05),
        "min_order_usd":      _g("min_order_usd",       1.0),
        "action_threshold":   _g("action_threshold",    0.005),
        "min_confidence":     _g("min_confidence",      0.10),
        "circuit_breaker_dd": _g("circuit_breaker_dd", -0.15),
        "max_daily_trades":   _g("max_daily_trades",    50),
        "warmup_bars":        _g("warmup_bars",         cfg.prediction.window_size + 5),
        "bar_buffer":         _g("bar_buffer",          200),
    }

    all_tickers = list(cfg.data.tickers)
    tradeable   = list(_g("tradeable_tickers", None) or _filter_tradeable(all_tickers))
    log.info(f"Ticker: {len(tradeable)} tradabili / {len(all_tickers)} training")

    # ── Carica modelli ─────────────────────────────────────────────────────
    predictor, scaler, num_features, device = _load_predictor(cfg, log)

    N_THERMO  = 7  # colonne canoniche ThermoStateBuilder
    # state_dim identico a TradingEnv: usa TUTTI i ticker del training
    state_dim = 2 * num_features + len(all_tickers) + 2 + N_THERMO
    log.info(f"state_dim={state_dim} (F={num_features} ALL_T={len(all_tickers)} tradeable={len(tradeable)} thm={N_THERMO})")

    agent, normalizer, tradeable_indices = _load_agent(
        cfg, state_dim, all_tickers, tradeable, device, log
    )

    from modelli.thermo_state_builder import ThermoStateBuilder
    thermo_builder = ThermoStateBuilder(interval=freq)


    # ── Connessione ────────────────────────────────────────────────────────
    tc, dc = _init_alpaca(log)

    account  = _fetch_account(tc)
    eq0      = account["equity"]
    cb       = _CB(eq0, p["circuit_breaker_dd"])
    live_log = _LiveLog(os.path.join(cfg.paths.results_dir, f"alpaca_live_log_{freq}.csv"))

    interval_sec   = {"1m":60,"2m":120,"5m":300,"15m":900,"1h":3600,"1d":86400}.get(freq,120)
    n_today        = 0
    last_date: Optional[datetime.date] = None
    step_n         = 0

    log.info(f"Equity iniziale: ${eq0:,.2f} | CB soglia: {p['circuit_breaker_dd']:.0%}")
    log.info("Avvio loop...")

    # ── Loop principale ────────────────────────────────────────────────────
    while True:
        try:
            now   = datetime.datetime.now(datetime.timezone.utc)
            today = now.date()

            # Reset giornaliero
            if last_date != today:
                n_today = 0; last_date = today
                log.info(f"Nuovo giorno: {today}")

            # Orario di mercato
            try:
                if not tc.get_clock().is_open:
                    log.info("Mercato chiuso — attendo 5 min")
                    time.sleep(300); continue
            except Exception:
                pass

            # Sincronizza con la barra
            elapsed = (now.minute * 60 + now.second) % interval_sec
            time.sleep(max(1.0, interval_sec - elapsed + 5))

            step_n += 1
            log.info(f"\n{'─'*55}")
            log.info(f"  Step {step_n} | {datetime.datetime.now(datetime.timezone.utc).strftime('%H:%M:%S UTC')}")

            # 1. Barre aggiornate
            bars = _fetch_bars(dc, tradeable, p["bar_buffer"], freq, log)
            if bars.empty or len(bars) < p["warmup_bars"]:
                log.info(f"Warm-up: {len(bars)}/{p['warmup_bars']} barre"); continue

            # 2. Frame completo a 58 colonne (51 ticker + 7 thermo)
            # Il MinMaxScaler è stato fittato su 58 feature — le thermo
            # devono essere incluse nello stesso ordine del training.
            _THERMO_COLS = ["Thm_Pressure","Thm_Temperature","Thm_Work",
                            "Thm_Stress","Thm_Efficiency","Thm_Entropy","Thm_Regime"]
            all_cols_58 = all_tickers + _THERMO_COLS

            # 2a. Calcola thermo PRIMA dello scaling (serve per il frame)
            try:
                thermo_df = thermo_builder.build(bars, tradeable)
            except Exception as e:
                log.warning(f"Thermo: {e}")
                thermo_df = pd.DataFrame()

            # 2b. Costruisce il frame con prezzi + thermo
            full = pd.DataFrame(0.0, index=bars.index, columns=all_cols_58)
            for t in tradeable:
                if t in bars.columns and t in all_tickers:
                    full[t] = bars[t].values
            if not thermo_df.empty:
                for col in _THERMO_COLS:
                    if col in thermo_df.columns:
                        aligned = thermo_df[col].reindex(full.index).ffill().bfill().fillna(0.0)
                        full[col] = aligned.values

            # 3. Scaling
            try:
                scaled = scaler.transform(full)
            except Exception as e:
                log.error(f"Scaling: {e}"); continue

            ws = cfg.prediction.window_size
            if len(scaled) < ws:
                log.warning(f"Barre insufficienti ({len(scaled)} < {ws})"); continue

            # 4. CNN prediction
            with torch.no_grad():
                win = torch.tensor(scaled[-ws:][np.newaxis], dtype=torch.float32).to(device)
                out = predictor(win)
                if out.dim() == 3:
                    out = out[:, 0, :]
                pred = out.cpu().numpy()

            # 5. thermo_df già calcolato al passo 2a — non ricalcolare

            # 6. Portafoglio
            account = _fetch_account(tc)

            # 7. Circuit breaker
            if cb.check(account["equity"], tc, log):
                log.error("Circuit breaker — sospeso"); time.sleep(interval_sec); continue

            # 8. Stato + azione DDPG
            state_raw = _build_state(scaled, pred, account, all_tickers, tradeable,
                                     thermo_df, num_features, eq0)

            if len(state_raw) != state_dim:
                log.warning(f"state mismatch {len(state_raw)} vs {state_dim}")
                state_raw = (np.pad(state_raw, (0, state_dim - len(state_raw)))
                             if len(state_raw) < state_dim else state_raw[:state_dim])

            actions_full = agent.act(normalizer.normalize(state_raw, update=False), explore=False)
            # Estrai solo le azioni per i ticker tradabili (sottoinsieme di all_tickers)
            actions = actions_full[tradeable_indices]

            # Log segnali
            sig = [(tradeable[i], float(a)) for i, a in enumerate(actions)
                   if abs(a) > p["action_threshold"]]
            if sig:
                log.info(f"Segnali ({len(sig)}):")
                for t, a in sig:
                    log.info(f"    {'🟢 BUY ' if a > 0 else '🔴 SELL'} {t:8s} {a:+.4f}")
            else:
                log.info("HOLD")

            # Log thermo
            if not thermo_df.empty:
                s  = float(thermo_df.get("Thm_Stress",  pd.Series([0])).iloc[-1])
                rg = int(thermo_df.get("Thm_Regime", pd.Series([0])).iloc[-1])
                rname = {0:"NEUTRO",1:"RALLY_REALE",2:"RALLY_ESAUSTO",
                         3:"COMPRESSIONE",4:"RIMBALZO"}.get(rg,"?")
                log.info(f"Thermo: stress={s:+.3f} | {rname}")

            # 9. Ordini
            orders, n_today = _execute_orders(tc, actions, tradeable,
                                              account, p, n_today, log)

            # 10. Registro
            live_log.record(now, account, actions, tradeable, thermo_df, orders)

            ret = (account["equity"] / eq0 - 1) * 100
            log.info(
                f"Portfolio: ${account['equity']:,.2f} ({ret:+.2f}%) | "
                f"Cash: ${account['cash']:,.2f} | Ordini oggi: {n_today}"
            )

        except KeyboardInterrupt:
            log.info("Interrotto (Ctrl+C)"); break
        except Exception as e:
            log.error(f"Errore step: {e}\n{traceback.format_exc()}")
            time.sleep(30)

    live_log.close()
    log.info("Sessione terminata.")


# ═══════════════════════════════════════════════════════════════════════════════
# REPLAY AD ALTA VELOCITÀ
# ═══════════════════════════════════════════════════════════════════════════════

def _fetch_historical_bars(dc, tickers: list[str], freq: str,
                           start: datetime.datetime, end: datetime.datetime,
                           log: logging.Logger) -> pd.DataFrame:
    """
    Scarica barre storiche da Alpaca per un intervallo preciso.
    Restituisce DataFrame con colonne=tickers, index=timestamp UTC,
    ordinato cronologicamente — pronto per il replay.
    """
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    _tf = {"1m": TimeFrame(1, TimeFrameUnit.Minute),
           "2m": TimeFrame(2, TimeFrameUnit.Minute),
           "5m": TimeFrame(5, TimeFrameUnit.Minute),
           "15m": TimeFrame(15, TimeFrameUnit.Minute),
           "1h": TimeFrame(1, TimeFrameUnit.Hour),
           "1d": TimeFrame(1, TimeFrameUnit.Day)}
    tf = _tf.get(freq, TimeFrame(2, TimeFrameUnit.Minute))

    log.info(f"Download storico Alpaca | {start.date()} → {end.date()} | freq={freq}")

    try:
        bars = dc.get_stock_bars(StockBarsRequest(
            symbol_or_symbols=tickers, timeframe=tf,
            start=start, end=end, adjustment="all",
            feed="iex",   # IEX = gratuito; SIP richiede abbonamento a pagamento
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
    """
    Calcola lo stato del portafoglio simulato senza chiamare Alpaca.
    Usato nel replay per non dipendere dal paper account ad ogni step.
    positions: {ticker: shares}
    """
    holdings_val = {t: shares * current_prices.get(t, 0.0)
                    for t, shares in positions.items()}
    equity = cash + sum(holdings_val.values())
    return {
        "equity": equity,
        "cash":   cash,
        "daily_pl": 0.0,  # calcolato alla fine
        "positions": {
            t: {"qty": shares,
                "market_value": holdings_val[t],
                "avg_cost": 0.0,
                "unrealized_pl": 0.0}
            for t, shares in positions.items() if shares > 0
        }
    }


def _replay_execute(actions: np.ndarray, tradeable: list[str],
                    positions: dict[str, float], cash: float,
                    prices: dict[str, float], p: dict) -> tuple[dict, float, list[dict]]:
    """
    Simula l'esecuzione degli ordini localmente durante il replay.
    Non chiama Alpaca — aggiorna posizioni e cash direttamente.

    Restituisce (positions_aggiornate, cash_aggiornato, ordini_log).
    """
    equity  = cash + sum(positions.get(t, 0.0) * prices.get(t, 0.0) for t in tradeable)
    orders: list[dict] = []

    for i, ticker in enumerate(tradeable):
        if i >= len(actions):
            break
        a     = float(actions[i])
        price = prices.get(ticker, 0.0)
        if price <= 0 or abs(a) < p["action_threshold"] or abs(a) < p["min_confidence"]:
            continue

        cur_val = positions.get(ticker, 0.0) * price
        tc_cost = p.get("transaction_cost", 0.001)

        if a > p["action_threshold"]:
            max_pos = equity * p["max_position_pct"]
            notional = min(cash * a, max(0.0, max_pos - cur_val))
            if notional < p["min_order_usd"]:
                continue
            shares = notional * (1 - tc_cost) / price
            positions[ticker] = positions.get(ticker, 0.0) + shares
            cash -= notional
            orders.append({"ticker": ticker, "side": "BUY",
                           "notional": round(notional, 2), "action": a})

        elif a < -p["action_threshold"] and cur_val > p["min_order_usd"]:
            notional  = cur_val * abs(a)
            if notional < p["min_order_usd"]:
                continue
            shares_sold = notional / price
            positions[ticker] = max(0.0, positions.get(ticker, 0.0) - shares_sold)
            cash += notional * (1 - tc_cost)
            orders.append({"ticker": ticker, "side": "SELL",
                           "notional": round(notional, 2), "action": a})

    return positions, cash, orders


def run_alpaca_replay(cfg) -> None:
    """
    Replay ad alta velocità di barre storiche Alpaca.
    Chiamato da main.py con step=alpaca_replay.

    Differenze rispetto a run_alpaca (live):
    - Scarica barre storiche Alpaca per un range di date configurabile
    - Riproduce le barre sequenzialmente senza aspettare il tempo reale
    - La velocità è controllata da alpaca_replay.bar_delay_sec (default 0.05s)
    - Gli ordini vengono simulati localmente (portfolio virtuale) oppure
      inviati al paper account Alpaca (alpaca_replay.submit_orders=true)
    - Al termine stampa un report completo identico a run_trade

    Config opzionale (config.yaml o CLI):
      alpaca_replay:
        days_back:       4       # quanti giorni di storia riprodurre
        bar_delay_sec:   0.05   # pausa tra una barra e la prossima (secondi)
        submit_orders:   false  # true = invia ordini reali al paper account
        window_size:     null   # override window (default: cfg.prediction.window_size)
        start_date:      null   # override data inizio (ISO format)
        end_date:        null   # override data fine (ISO format)

    Speedup tipico con bar_delay_sec=0.05:
      4 giorni × 390 barre/giorno (2m) × 0.05s = ~78 secondi
      rispetto a 4 giorni reali = 345.600 secondi → 4400x più veloce
    """
    freq = cfg.frequency.interval
    os.makedirs(cfg.paths.results_dir, exist_ok=True)
    log = _setup_logger(cfg.paths.results_dir, f"replay_{freq}")

    log.info("=" * 62)
    log.info(f"  ALPACA REPLAY | freq={freq} | window={cfg.prediction.window_size}")
    log.info("=" * 62)

    # ── Parametri replay ──────────────────────────────────────────────────
    rc = getattr(cfg, "alpaca_replay", None)
    def _r(k, d): return getattr(rc, k, d) if rc else d

    days_back      = int(_r("days_back",      4))
    bar_delay_sec  = float(_r("bar_delay_sec", 0.05))
    submit_orders  = bool(_r("submit_orders",  False))

    # Date del replay
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

    # Parametri di rischio (stessi di run_alpaca)
    ac = getattr(cfg, "alpaca", None)
    def _g(k, d): return getattr(ac, k, d) if ac else d

    p = {
        "max_position_pct":  _g("max_position_pct",  0.05),
        "min_order_usd":     _g("min_order_usd",      1.0),
        "action_threshold":  _g("action_threshold",   0.005),
        "min_confidence":    _g("min_confidence",     0.10),
        "circuit_breaker_dd": _g("circuit_breaker_dd", -0.15),
        "transaction_cost":  0.001,
    }

    all_tickers = list(cfg.data.tickers)
    tradeable   = list(_g("tradeable_tickers", None) or _filter_tradeable(all_tickers))

    log.info(f"Range replay: {start_dt.date()} → {end_dt.date()} ({days_back} giorni)")
    log.info(f"Ticker: {len(tradeable)} tradabili | bar_delay={bar_delay_sec}s | "
             f"submit_orders={submit_orders}")

    # Stima velocità
    bars_per_day = {"1m": 390, "2m": 195, "5m": 78, "15m": 26,
                    "1h": 7, "1d": 1}.get(freq, 195)
    est_bars    = days_back * bars_per_day
    est_seconds = est_bars * bar_delay_sec
    speedup     = (days_back * 86400) / max(est_seconds, 1)
    log.info(f"Stima: ~{est_bars:,} barre | durata ~{est_seconds:.0f}s | {speedup:.0f}x speedup")

    # ── Carica modelli ─────────────────────────────────────────────────────
    predictor, scaler, num_features, device = _load_predictor(cfg, log)

    N_THERMO  = 7
    state_dim = 2 * num_features + len(all_tickers) + 2 + N_THERMO
    log.info(f"state_dim={state_dim} (F={num_features} ALL_T={len(all_tickers)} tradeable={len(tradeable)} thm={N_THERMO})")
    agent, normalizer, tradeable_indices = _load_agent(
        cfg, state_dim, all_tickers, tradeable, device, log
    )

    from modelli.thermo_state_builder import ThermoStateBuilder
    thermo_builder = ThermoStateBuilder(interval=freq)

    # ── Connessione Alpaca ─────────────────────────────────────────────────
    tc, dc = _init_alpaca(log)

    # ── Download barre storiche ────────────────────────────────────────────
    all_bars = _fetch_historical_bars(dc, tradeable, freq, start_dt, end_dt, log)

    ws = cfg.prediction.window_size
    if len(all_bars) < ws + 5:
        log.error(f"Barre insufficienti: {len(all_bars)} < {ws + 5}. "
                  f"Aumenta days_back o usa una finestra più piccola.")
        return

    # ── Portafoglio virtuale ───────────────────────────────────────────────
    initial_capital = float(cfg.buyer.initial_capital)
    positions: dict[str, float] = {}   # {ticker: shares}
    cash     = initial_capital
    eq0      = initial_capital

    live_log      = _LiveLog(os.path.join(cfg.paths.results_dir,
                                          f"alpaca_replay_log_{freq}.csv"))
    portfolio_hist: list[float] = [initial_capital]
    trade_log_rows: list[dict]  = []

    n_buys = n_sells = 0
    step_n = 0
    peak_eq = initial_capital

    log.info(f"\nInizio replay — {len(all_bars):,} barre totali | "
             f"capitale iniziale: ${initial_capital:,.2f}")
    log.info("─" * 55)

    # ── Loop replay ───────────────────────────────────────────────────────
    for bar_i in range(ws, len(all_bars)):
        step_n += 1

        # Finestra di input per CNN
        window_raw = all_bars.iloc[bar_i - ws: bar_i]
        ts         = all_bars.index[bar_i]

        # Frame completo a 58 colonne (51 ticker + 7 thermo)
        # Il MinMaxScaler è fittato su 58 feature — le thermo devono
        # essere incluse nello stesso ordine usato durante il training.
        _THERMO_COLS = ["Thm_Pressure","Thm_Temperature","Thm_Work",
                        "Thm_Stress","Thm_Efficiency","Thm_Entropy","Thm_Regime"]
        all_cols_58 = all_tickers + _THERMO_COLS

        # Calcola thermo PRIMA dello scaling (serve sia per il frame che per lo stato)
        try:
            thermo_df = thermo_builder.build(window_raw, tradeable)
        except Exception:
            thermo_df = pd.DataFrame()

        # Costruisce frame prezzi + thermo
        full = pd.DataFrame(0.0, index=window_raw.index, columns=all_cols_58)
        for t in tradeable:
            if t in window_raw.columns and t in all_tickers:
                full[t] = window_raw[t].values
        if not thermo_df.empty:
            for col in _THERMO_COLS:
                if col in thermo_df.columns:
                    aligned = thermo_df[col].reindex(full.index).ffill().bfill().fillna(0.0)
                    full[col] = aligned.values

        # Scaling
        try:
            
            scaled = scaler.transform(full)
        except Exception as e:
            log.warning(f"Scaling step {step_n}: {e}")
            continue

        # CNN prediction
        with torch.no_grad():
            win_t = torch.tensor(scaled[np.newaxis], dtype=torch.float32).to(device)
            out   = predictor(win_t)
            if out.dim() == 3:
                out = out[:, 0, :]
            pred = out.cpu().numpy()

        # thermo_df già calcolato sopra — non ricalcolare

        # Prezzi correnti (ultima barra della finestra)
        prices = {t: float(all_bars[t].iloc[bar_i])
                  for t in tradeable if t in all_bars.columns}

        # Stato portafoglio simulato
        account = _simulate_account(positions, cash, prices)
        equity  = account["equity"]
        portfolio_hist.append(equity)

        # Circuit breaker
        if equity > peak_eq:
            peak_eq = equity
        dd = (equity - peak_eq) / (peak_eq + 1e-8)
        if dd < p["circuit_breaker_dd"]:
            log.warning(f"⚠️  Circuit breaker a step {step_n} | dd={dd:.1%}")
            break

        # Stato vettoriale
        state_raw = _build_state(scaled, pred, account, all_tickers, tradeable,
                                 thermo_df, num_features, eq0)
        if len(state_raw) != state_dim:
            state_raw = (np.pad(state_raw, (0, state_dim - len(state_raw)))
                         if len(state_raw) < state_dim else state_raw[:state_dim])

        actions_full = agent.act(normalizer.normalize(state_raw, update=False), explore=False)
        # Estrai solo le azioni per i ticker tradabili
        actions = actions_full[tradeable_indices]

        # Esegui ordini (simulati localmente)
        positions, cash, orders = _replay_execute(
            actions, tradeable, positions, cash, prices, p
        )
        n_buys  += sum(1 for o in orders if o["side"] == "BUY")
        n_sells += sum(1 for o in orders if o["side"] == "SELL")

        # Se richiesto, invia anche al paper account Alpaca
        if submit_orders and orders:
            _, _ = _execute_orders(tc, actions, tradeable,
                                   account, p, 0, log)

        # Log ordini
        for o in orders:
            trade_log_rows.append({
                "ts": ts, "ticker": o["ticker"], "side": o["side"],
                "notional": o["notional"], "action": round(o["action"], 4),
                "equity": round(equity, 2), "cash": round(cash, 2),
            })

        # Log CSV ogni 5 step
        fake_account = _simulate_account(positions, cash, prices)
        live_log.record(ts, fake_account, actions, tradeable, thermo_df, orders)

        # Progress ogni 100 step
        if step_n % 100 == 0:
            ret = (equity / eq0 - 1) * 100
            log.info(
                f"  Step {step_n:5d}/{len(all_bars)-ws} | "
                f"{ts.strftime('%m-%d %H:%M')} | "
                f"${equity:,.2f} ({ret:+.2f}%) | "
                f"B:{n_buys} S:{n_sells}"
            )

        time.sleep(bar_delay_sec)

    # ── Report finale ──────────────────────────────────────────────────────
    live_log.close()

    hist    = np.array(portfolio_hist, dtype=np.float64)
    final_v = hist[-1]
    ret_pct = (final_v / initial_capital - 1) * 100

    # Sharpe annualizzato
    bars_per_year = int(getattr(cfg.buyer, "bars_per_year", 98280))
    returns  = np.diff(hist) / (hist[:-1] + 1e-8)
    sharpe   = (float(returns.mean()) / (float(returns.std()) + 1e-8)
                * np.sqrt(bars_per_year)) if len(returns) > 1 else 0.0

    # Max drawdown
    peak = np.maximum.accumulate(hist)
    max_dd = float(((hist - peak) / (peak + 1e-8)).min())

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
    log.info(sep)

    # Salva trade log
    if trade_log_rows:
        tlog_path = os.path.join(cfg.paths.results_dir, f"alpaca_replay_trades_{freq}.csv")
        pd.DataFrame(trade_log_rows).to_csv(tlog_path, index=False)
        log.info(f"  Trade log: {tlog_path}")

    # Salva curva portafoglio
    ph_path = os.path.join(cfg.paths.results_dir, f"alpaca_replay_portfolio_{freq}.csv")
    pd.Series(hist, name="equity").to_csv(ph_path)
    log.info(f"  Portfolio: {ph_path}")
    log.info(sep)