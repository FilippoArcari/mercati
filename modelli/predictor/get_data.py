import datetime
import warnings
import yfinance as yf
import pandas as pd
import numpy as np
import os
import fredapi
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
import nolds
import powerlaw
from modelli.features.thermodynamic import compute_thermodynamic_features, THERMO_N_FEATURES


DEFAULT_LIMIT = {"1d": None, "2m": 60}
DEFAULT_CACHE = "./cache/"

_RATE_USA = "GS10"
_RATE_EU  = "IR3TIB01EZM156N"
_W_USA    = 0.70
_W_EU     = 0.30


# ══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING
# ══════════════════════════════════════════════════════════════════════════════

def get_data(ticker, start_date, end_date, frequency, macroeconomic_data, train_split):
    macroeconomic_data = list(macroeconomic_data) if macroeconomic_data else []
    tickers = [ticker] if isinstance(ticker, str) else list(ticker)

    cache_path = os.path.join(DEFAULT_CACHE, f"data_{frequency}.parquet")

    if os.path.exists(cache_path):
        print(f"[get_data] Loading {frequency} data from cache...")
        data = pd.read_parquet(cache_path)
    else:
        print(f"[get_data] Fetching {frequency} data from yfinance...")
        os.makedirs(DEFAULT_CACHE, exist_ok=True)

        if frequency == "2m":
            end_date = pd.Timestamp.now()
            limit_date = end_date - pd.Timedelta(days=59)
            if pd.to_datetime(start_date) < limit_date:
                print(f"Warning: 2m data limited to last 60 days. Adjusting to {limit_date.date()}")
                start_date = limit_date.strftime("%Y-%m-%d")

        raw = yf.download(
            tickers=tickers,
            start=start_date,
            end=end_date,
            interval=frequency,
            group_by="column",
            auto_adjust=True,
        )
        raw.index = pd.to_datetime(raw.index).tz_localize(None)

        close = raw["Close"]
        if isinstance(raw["Volume"], pd.DataFrame):
            volume = raw["Volume"].sum(axis=1).rename("Volume")
        else:
            volume = raw["Volume"].rename("Volume")
        data = pd.concat([close, volume], axis=1)

        # ── Macro data (daily only) ──────────────────────────────────────────
        if macroeconomic_data and frequency == "1d":
            fred = fredapi.Fred(api_key=os.environ.get("FRED_API_KEY"))
            macro_data = {}
            macro_start = pd.to_datetime(start_date) - pd.DateOffset(years=1)
            for series_id in macroeconomic_data:
                try:
                    macro_data[series_id] = fred.get_series(
                        series_id,
                        observation_start=macro_start,
                        observation_end=end_date,
                    )
                except Exception as e:
                    print(f"Error fetching {series_id}: {e}")

            if macro_data:
                macro = pd.DataFrame(macro_data)
                macro.index = pd.to_datetime(macro.index).tz_localize(None)
                macro = macro[~macro.index.duplicated(keep="last")].sort_index()
                macro = macro.reindex(data.index, method="ffill")
                data = pd.concat([data, macro], axis=1)

        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.astype(float).ffill().bfill().fillna(0)

        if not data.empty:
            data.to_parquet(cache_path)
        else:
            print("Warning: no data found.")

    # ── Feature engineering (econophysics) ───────────────────────────────────
    print(f"Data shape before indicators: {data.shape}")
    if frequency == "1d":
        features = add_rolling_econophysics_day(data, window=120, step=50)
    elif frequency == "2m":
        features = add_rolling_econophysics_minutely(data, window=60)
    else:
        raise ValueError(f"Unsupported frequency: {frequency}")

    features = features.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)

    n_inf = np.isinf(features.values).sum()
    n_nan = np.isnan(features.values).sum()
    print(f"number of nan and inf values in features: {n_nan} {n_inf}")

    data = data.join(features, how="left").ffill().bfill()
    print(f"Data shape after econophysics: {data.shape}")

    # ── Feature termodinamiche ────────────────────────────────────────────────
    # Solo daily: serve storia sufficiente per il lag monetario (~90 giorni)
    # e per l'equazione di Van der Waals (rolling window 20gg).
    # Le thermo vengono calcolate su dati RAW (prima dello scaling) perché
    # usano prezzi assoluti e volumi nella formulazione VdW.
    # Diventano le ULTIME THERMO_N_FEATURES colonne → context stream del Predictor.
    if frequency == "1d":
        data = _add_thermodynamic(data, tickers)
        print(f"Data shape after thermodynamic: {data.shape}")
        n_context_features = features.shape[1] + THERMO_N_FEATURES
    else:
        n_context_features = features.shape[1]

    # ── Scaling (su tutto, incluse thermo) ───────────────────────────────────
    # RobustScaler centra sulla mediana → nelle thermo, valori > 0.5 dopo
    # MinMaxScaler indicano "sopra la mediana storica" della feature.
    # Questo è sufficiente per il reward shaping dell'env (segno preservato).
    scaler_pipeline = Pipeline([
        ("robust", RobustScaler()),
        ("minmax", MinMaxScaler(feature_range=(0.05, 0.95))),
    ])

    data = data.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)

    train_data = data[: int(len(data) * train_split)]
    scaler_pipeline.fit(train_data)
    data = pd.DataFrame(
        scaler_pipeline.transform(data),
        index=data.index,
        columns=data.columns,
    )

    return data, n_context_features, scaler_pipeline


# ══════════════════════════════════════════════════════════════════════════════
# THERMODYNAMIC INTEGRATION
# ══════════════════════════════════════════════════════════════════════════════

def _add_thermodynamic(data: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """
    1. Costruisce il tasso composito USA/EU dal DataFrame grezzo.
    2. Calcola le 9 feature termodinamiche.
    3. Le aggiunge in coda (DOPO le econophysics, PRIMA dello scaling).

    Struttura colonne finale:
        [tickers | Volume | macro FRED | econophysics | thermo_*]
        ↑─────────────────────────────↑               ↑────────↑
              n_main_features (real_features)     n_context_features
    """
    # ── Tasso composito pesato ────────────────────────────────────────────────
    has_usa = _RATE_USA in data.columns
    has_eu  = _RATE_EU  in data.columns

    if has_usa and has_eu:
        rate_composite = _W_USA * data[_RATE_USA].ffill() + _W_EU * data[_RATE_EU].ffill()
        rate_col_name  = "rate_composite"
        print(f"[thermo] Tasso composito: {_W_USA:.0%}×GS10 + {_W_EU:.0%}×Euribor3m")
    elif has_usa:
        rate_composite = data[_RATE_USA].ffill()
        rate_col_name  = _RATE_USA
        print("[thermo] Euribor non trovato — uso solo GS10")
    elif has_eu:
        rate_composite = data[_RATE_EU].ffill()
        rate_col_name  = _RATE_EU
        print("[thermo] GS10 non trovato — uso solo Euribor3m")
    else:
        rate_composite = None
        rate_col_name  = None
        print("[thermo] Nessuna serie tassi trovata — Z-score userà fallback rolling")

    # Aggiunge il rate composito temporaneamente per passarlo alla funzione
    data_tmp = data.copy()
    if rate_composite is not None and rate_col_name not in data_tmp.columns:
        data_tmp[rate_col_name] = rate_composite

    # ── Feature termodinamiche ────────────────────────────────────────────────
    valid_tickers = [t for t in tickers if t in data_tmp.columns]
    df_thermo = compute_thermodynamic_features(
        df=data_tmp,
        tickers=valid_tickers,
        rate_col=rate_col_name,
        window_T=20,
        window_P=20,
        window_W=60,
        max_lag=90,
    )

    # Aggiunge le thermo alle colonne originali (non a data_tmp per evitare
    # di portarsi dietro la colonna rate_composite temporanea)
    data = data.join(df_thermo, how="left").ffill().fillna(0.0)
    return data


# ══════════════════════════════════════════════════════════════════════════════
# FEATURES DAILY — Hurst + Power-law + Shannon
# ══════════════════════════════════════════════════════════════════════════════

def add_rolling_econophysics_day(df: pd.DataFrame, window: int = 120, step: int = 10) -> pd.DataFrame:
    price_col = df.columns[0]
    prices    = df[price_col]
    returns   = np.log(((prices) / (prices.shift(1) + 1e-6)).clip(lower=1e-6)).fillna(0)
    n, index  = len(df), df.index

    hurst_vals   = pd.Series(np.nan, index=index)
    alpha_vals   = pd.Series(np.nan, index=index)
    entropy_vals = pd.Series(np.nan, index=index)

    for i in range(window, n, step):
        prices_w  = prices.iloc[i - window : i].ffill().bfill().values
        returns_w = returns.iloc[i - window : i].ffill().bfill().values
        idx       = index[i - 1]

        if len(prices_w) > 20 and np.std(prices_w) > 1e-8:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    hurst_vals.at[idx] = nolds.hurst_rs(prices_w)
            except Exception:
                pass

        neg = np.abs(returns_w[returns_w < 0])
        neg = neg[neg > 1e-8]
        if len(neg) > 20 and np.std(neg) > 1e-8:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    alpha = powerlaw.Fit(neg, verbose=False).power_law.alpha
                alpha_vals.at[idx] = np.clip(alpha, 1.5, 6.0)
            except Exception:
                alpha_vals.at[idx] = 3.0
        else:
            alpha_vals.at[idx] = 3.0

    for i in range(window, n):
        ret_w = returns.iloc[i - window : i].ffill().bfill().values
        if np.std(ret_w) > 0:
            hist, _ = np.histogram(ret_w, bins=10, density=True)
            entropy_vals.at[index[i - 1]] = -np.sum(hist * np.log(hist + 1e-10))

    return pd.DataFrame(
        {
            "hurst_context":   hurst_vals,
            "alpha_context":   alpha_vals,
            "shannon_entropy": entropy_vals,
        },
        index=index,
    ).ffill().bfill()


# ══════════════════════════════════════════════════════════════════════════════
# FEATURES INTRADAY — Physics-inspired + Permutation Entropy
# ══════════════════════════════════════════════════════════════════════════════

def add_rolling_econophysics_minutely(df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    price_col = df.columns[0]
    prices    = df[price_col]
    volume    = df["Volume"] if "Volume" in df.columns else None

    log_ret = np.log(((prices) / (prices.shift(1) + 1e-6)).clip(lower=1e-6)).fillna(0)
    log_vol = np.log1p(volume.clip(lower=0)) if volume is not None else None

    out = {}

    if log_vol is not None:
        out["kinetic_energy"]  = (0.5 * log_vol * log_ret ** 2).rename("kinetic_energy")
        acceleration           = log_ret.diff()
        out["market_force"]    = (log_vol * acceleration).rename("market_force")
        out["market_momentum"] = (log_vol * log_ret).rename("market_momentum")

    def _shannon(x):
        if np.std(x) == 0:
            return np.nan
        hist, _ = np.histogram(x, bins=10, density=True)
        return -np.sum(hist * np.log(hist + 1e-10))

    out["shannon_entropy"]     = log_ret.rolling(window).apply(_shannon, raw=True)
    out["permutation_entropy"] = log_ret.rolling(window).apply(
        lambda x: _permutation_entropy(x, order=3), raw=True
    )
    out["realized_vol"]        = log_ret.rolling(window).std() * np.sqrt(window)

    minutes_from_open = (df.index.hour - 9) * 60 + df.index.minute - 30
    minutes_from_open = np.clip(minutes_from_open, 0, 390)
    out["session_sin"] = pd.Series(np.sin(2 * np.pi * minutes_from_open / 390), index=df.index)
    out["session_cos"] = pd.Series(np.cos(2 * np.pi * minutes_from_open / 390), index=df.index)

    return pd.DataFrame(out, index=df.index).ffill().bfill()


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _permutation_entropy(x: np.ndarray, order: int = 3, normalize: bool = True) -> float:
    import math
    n = len(x)
    if n < order or np.std(x) == 0:
        return np.nan
    patterns  = np.array([x[i : i + order] for i in range(n - order + 1)])
    ranked    = np.argsort(np.argsort(patterns, axis=1), axis=1)
    _, counts = np.unique(ranked, axis=0, return_counts=True)
    probs     = counts / counts.sum()
    pe        = -np.sum(probs * np.log(probs))
    if normalize:
        max_pe = np.log(math.factorial(order))
        pe     = pe / max_pe if max_pe > 0 else pe
    return pe