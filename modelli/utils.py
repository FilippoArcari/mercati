from __future__ import annotations
import os
import datetime
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
import yfinance as yf
from pandas.errors import EmptyDataError, ParserError
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from fredapi import Fred
import seaborn as sns

from scipy.stats import entropy

from modelli.thermodynamics import (
    compute_thermodynamic_features,
    calculate_pressure_and_work,
    init_vdw_calibration,
    calculate_entropy_difference,
    calculate_maxwell_boltzmann_indicators,
    calculate_advanced_econophysics_indicators,
    calculate_intraday_thermodynamics,
)


_INTRADAY_LIMITS = {
    "1m":  7,
    "2m":  60,
    "5m":  60,
    "15m": 60,
    "30m": 60,
    "1h":  730,
    "90m": 60,
}

# Colonne tassi cercate in ordine per la divergenza energetica
_RATES_CANDIDATES = ["GS10", "T10YIE", "FEDFUNDS", "^TNX"]


# ─── Device Management ─────────────────────────────────────────────────────────
# Re-export dal modulo centralizzato device_setup.
# Mantiene la retrocompatibilità: tutti i `from modelli.utils import get_device`
# continuano a funzionare senza modifiche.

from modelli.device_setup import (  # noqa: E402
    get_device,
    detect_device,
    get_map_location,
    safe_save,
    xla_mark_step,
    wrap_model_for_backend,
    unwrap_model,
    DeviceConfig,
)


# ─── Cache ─────────────────────────────────────────────────────────────────────

def _load_cache(cache_path: str, min_rows: int = 1) -> pd.DataFrame | None:
    if not os.path.exists(cache_path):
        return None
    try:
        data = pd.read_csv(cache_path, index_col=0, parse_dates=True)
    except Exception as e:
        print(f"[load_data] Cache corrotta ({e}), la scarico di nuovo.")
        os.remove(cache_path)
        return None
    if len(data) < min_rows:
        print(f"[load_data] Cache vuota ({len(data)} righe), la scarico di nuovo.")
        os.remove(cache_path)
        return None
    print(f"[load_data] Carico da cache: {cache_path} ({len(data)} barre)")
    return data


# ─── Sliding-window intraday fetcher ──────────────────────────────────────────

def fetch_intraday_sliding_window(
    tickers:       list[str],
    interval:      str,
    cache_path:    str,
    n_chunks:      int = 6,
    chunk_days:    int = 55,
    overlap_days:  int = 5,
) -> pd.DataFrame:
    """
    Aggira il limite di 60 giorni di yfinance per i dati intraday
    scaricando dati in finestre temporali sovrapposte e accumulandole
    in una cache locale che cresce nel tempo.

    Logica delle finestre:
      step = chunk_days - overlap_days  (default 50 giorni)
      n_chunks=6 → copertura massima: 55 + 5*50 = 305 giorni

    Accumulo progressivo:
      I chunk già presenti in cache vengono saltati. Eseguendo lo script
      ogni giorno si costruisce una storia progressivamente più lunga,
      superando il limite strutturale di yfinance di 60 giorni.

    Returns:
      DataFrame con colonne = tickers, index = timestamp UTC,
      già pulito da duplicati e ordinato cronologicamente.
    """
    step_days = chunk_days - overlap_days
    now       = datetime.datetime.now(datetime.timezone.utc)

    # ── Carica cache esistente ────────────────────────────────────────────
    existing: pd.DataFrame | None = _load_cache(cache_path)
    if existing is not None:
        # Normalizza timezone per confronti
        if existing.index.tzinfo is None:
            existing.index = existing.index.tz_localize("UTC")

    # ── Scarica chunk in ordine dal più recente al più vecchio ───────────
    new_chunks: list[pd.DataFrame] = []
    for i in range(n_chunks):
        chunk_end   = now - datetime.timedelta(days=i * step_days)
        chunk_start = chunk_end - datetime.timedelta(days=chunk_days)

        # Skip se questo range è già completamente coperto dalla cache
        if existing is not None:
            cache_oldest = existing.index.min()
            if chunk_start.replace(tzinfo=datetime.timezone.utc) >= cache_oldest:
                print(f"[SlidingWindow] Chunk {i+1}/{n_chunks}: già in cache, skip "
                      f"({chunk_start.date()} → {chunk_end.date()})")
                continue

        try:
            df_chunk = yf.download(
                tickers,
                start    = chunk_start,
                end      = chunk_end,
                interval = interval,
                progress = False,
                auto_adjust = True,
            )
            if df_chunk.empty:
                print(f"[SlidingWindow] Chunk {i+1}/{n_chunks}: nessun dato "
                      f"({chunk_start.date()} → {chunk_end.date()})")
                continue

            close = df_chunk["Close"] if "Close" in df_chunk.columns else df_chunk
            # Assicura index UTC
            if close.index.tzinfo is None:
                close.index = close.index.tz_localize("UTC")

            new_chunks.append(close)
            print(f"[SlidingWindow] Chunk {i+1}/{n_chunks}: {len(close):,} barre "
                  f"({chunk_start.date()} → {chunk_end.date()})")

        except Exception as e:
            print(f"[SlidingWindow] Chunk {i+1}/{n_chunks} fallito: {e}")

    # ── Unisce tutto ─────────────────────────────────────────────────────
    parts = ([] if existing is None else [existing]) + new_chunks
    if not parts:
        raise RuntimeError("[SlidingWindow] Nessun dato disponibile")

    combined = pd.concat(parts)
    combined = combined[~combined.index.duplicated(keep="first")].sort_index()

    # ── Salva cache aggiornata ────────────────────────────────────────────
    combined.to_csv(cache_path)
    span = combined.index[-1] - combined.index[0]
    print(f"[SlidingWindow] Cache aggiornata: {len(combined):,} barre | "
          f"{combined.index[0].date()} → {combined.index[-1].date()} "
          f"({span.days} giorni)")
    return combined


# ─── Thermodynamic helpers ─────────────────────────────────────────────────────

def add_divergence_and_efficiency_features(
    data:              pd.DataFrame,
    thermo_df:         pd.DataFrame,
    price_cols:        list[str],
    rates_col:         str | None,
    thermo_max_lag:    int = 90,
    efficiency_window: int = 10,
) -> pd.DataFrame:
    """
    Aggiunge al DataFrame termodinamico due segnali derivati:

    Energy_Divergence_Z
        Z-score di Pressione meno Z-score di Expected_Pressure (tassi con
        best_lag ottimale). Positivo = stress termico, negativo = spazio sano.

    Energy_Efficiency
        Differenza rolling (Lavoro normalizzato − Prezzo normalizzato).
        Positivo = attrito/distribuzione, neutro = trend fluido.

    Energy_Monetary_Lag
        Lag ottimale trovato (costante per sessione, utile come meta-feature).
    """
    added = []

    # ── Energy_Divergence_Z ───────────────────────────────────────────────────
    pressure_col = next(
        (c for c in thermo_df.columns if "Pressure" in c and "Expected" not in c),
        None,
    )

    if pressure_col and rates_col and rates_col in data.columns:
        pressure = thermo_df[pressure_col].copy()
        rates    = data[rates_col].reindex(pressure.index).ffill().bfill()

        valid = pressure.notna() & rates.notna()
        if valid.sum() > thermo_max_lag + 10:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                corrs    = [pressure[valid].corr(rates[valid].shift(i)) for i in range(thermo_max_lag + 1)]
            
            best_lag = int(np.nanargmax(np.abs(corrs)))
            best_rho = corrs[best_lag]
            print(f"[ThermoPatch] Lag monetario ottimale: {best_lag}gg | ρ={best_rho:.3f}")

            ss  = StandardScaler()
            p_z = pd.Series(
                ss.fit_transform(pressure.values.reshape(-1, 1)).flatten(),
                index=pressure.index,
            )
            expected = rates.shift(best_lag)
            ep_z = pd.Series(
                ss.fit_transform(expected.values.reshape(-1, 1)).flatten(),
                index=expected.index,
            )

            thermo_df["Energy_Divergence_Z"] = (p_z - ep_z).values
            thermo_df["Energy_Monetary_Lag"] = float(best_lag)
            added += ["Energy_Divergence_Z", "Energy_Monetary_Lag"]
        else:
            print("[ThermoPatch] Dati insufficienti per il calcolo del lag monetario.")
    else:
        missing = []
        if not pressure_col: missing.append("Market_Pressure")
        if not rates_col:    missing.append("rates_col")
        if missing:
            print(f"[ThermoPatch] Energy_Divergence_Z saltato: colonne mancanti {missing}")

    # ── Energy_Efficiency ─────────────────────────────────────────────────────
    work_col = next((c for c in thermo_df.columns if "Work" in c), None)

    if work_col:
        mm        = MinMaxScaler()
        work_raw  = thermo_df[work_col].ffill().bfill().fillna(0).values
        price_raw = data[price_cols].mean(axis=1).reindex(thermo_df.index).ffill().bfill().values

        work_n  = mm.fit_transform(work_raw.reshape(-1, 1)).flatten()
        price_n = mm.fit_transform(price_raw.reshape(-1, 1)).flatten()

        raw_eff = pd.Series(work_n - price_n, index=thermo_df.index)
        thermo_df["Energy_Efficiency"] = (
            raw_eff.rolling(efficiency_window, min_periods=1).mean().values
        )
        added.append("Energy_Efficiency")
    else:
        print("[ThermoPatch] Energy_Efficiency saltato: nessuna colonna Work trovata.")

    if added:
        print(f"[ThermoPatch] Nuove feature aggiunte: {added}")

    return thermo_df


def _sanitize(df: pd.DataFrame, label: str = "") -> pd.DataFrame:
    """
    Sostituisce inf con NaN, poi colma con ffill/bfill, poi fallback a 0.
    Logga un avviso se restano colonne problematiche.
    """
    df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    nan_cols = df.columns[df.isna().any()].tolist()
    if nan_cols:
        print(
            f"[{label}] AVVISO: {len(nan_cols)} colonne con NaN residui dopo cleanup "
            f"→ riempite con 0: {nan_cols}"
        )
        df[nan_cols] = df[nan_cols].fillna(0.0)
    return df


# ─── Main data loader ──────────────────────────────────────────────────────────

def load_data(
    tickers,
    start_date,
    end_date,
    fred_api_key,
    inflation_series,
    interval:           str   = "1d",
    cache_path:         str   = "./data.csv",
    max_history_days:   int   = None,
    add_thermodynamics: bool  = True,
    thermo_window:      int   = 20,
    thermo_max_lag:     int   = 90,
    use_returns:        bool  = True,
    split_ratio:        float = None,
):
    """
    Scarica prezzi + FRED, calcola feature termodinamiche e normalizza.

    Le feature termodinamiche vengono calcolate sui dati GREZZI (prima
    della normalizzazione MinMax) e aggiunte al DataFrame come colonne
    aggiuntive. Il DCNN le vede come input e impara a usarle.

    Parametri aggiuntivi
    --------------------
    add_thermodynamics : aggiunge le feature termodinamiche (default True)
    thermo_window      : finestra mobile per entropia/autocorr (default 20)
    thermo_max_lag     : lag massimo cross-correlazione tassi (default 90)
    """
    now         = datetime.datetime.now()
    is_intraday = interval in _INTRADAY_LIMITS
    limit_days  = max_history_days or _INTRADAY_LIMITS.get(interval)

    if is_intraday:
        effective_end   = now
        earliest        = now - datetime.timedelta(days=limit_days)
        effective_start = max(start_date, earliest)
        if effective_start != start_date:
            print(
                f"[load_data] Intervallo '{interval}' limitato a {limit_days} giorni. "
                f"start_date aggiustato da {start_date.date()} a {effective_start.date()}"
            )
    else:
        effective_start = start_date
        effective_end   = end_date

    data = _load_cache(cache_path)

    if data is None:
        print(f"[load_data] Scarico dati da yfinance (interval={interval})...")

        df_yf = yf.download(
            tickers,
            start=effective_start,
            end=effective_end,
            interval=interval,
            progress=False,
        )

        if df_yf.empty:
            raise RuntimeError(
                f"yfinance non ha restituito dati per interval='{interval}'."
            )

        close  = df_yf["Close"]
        volume = df_yf.get("Volume", pd.DataFrame())

        available = [t for t in tickers if t in close.columns]
        missing   = [t for t in tickers if t not in close.columns]
        if missing:
            print(f"[load_data] Ticker senza dati (rimossi): {missing}")

        # Reindex per garantire che TUTTI i ticker richiesti siano presenti come colonne,
        # anche se quelli mancanti saranno pieni di NaN (che gestiremo dopo).
        data = close.reindex(columns=tickers).copy()

        if not volume.empty:
            for t in available:
                if t in volume.columns:
                    data[f"{t}_Volume"] = volume[t]

        # Serie FRED (solo giornaliero)
        if not is_intraday and fred_api_key and inflation_series:
            fred_obj = Fred(api_key=fred_api_key)
            for series in inflation_series:
                try:
                    df_infl = fred_obj.get_series(
                        series,
                        observation_start=effective_start,
                        observation_end=effective_end,
                    )
                    data[series] = df_infl
                except (EmptyDataError, ParserError) as e:
                    print(f"[load_data] Errore serie FRED {series}: {e}")

        data = data.ffill().bfill()
        if len(data) == 0:
            raise RuntimeError("DataFrame vuoto dopo il download.")

        data.to_csv(cache_path)
        print(f"[load_data] {len(data)} barre → cache in {cache_path}")

    # ── Feature termodinamiche (su dati grezzi, PRIMA della norm.) ───────────
    volume_cols = [c for c in data.columns if c.endswith("_Volume")]
    price_cols  = [
        c for c in data.columns
        if not c.endswith("_Volume") and c not in (inflation_series or [])
    ]

    if add_thermodynamics:
        from modelli.thermo_state_builder import ThermoStateBuilder, CANONICAL_COLS
        builder   = ThermoStateBuilder(interval=interval)
        thermo_df = builder.build(data, tickers)
        
        # Estrai le colonne canoniche se presenti, o tutte se non lo sono
        can_cols = [c for c in CANONICAL_COLS if c in thermo_df.columns]
        canonical = thermo_df[can_cols]
        
        # Sanifica le feature termodinamiche PRIMA del concat
        canonical = _sanitize(canonical, "thermo_df")

        data = pd.concat([data, canonical], axis=1).ffill().bfill()
        print(f"[load_data] Feature termodinamiche base aggiunte al DCNN: {list(can_cols)}")

    # Rimuovi i volumi grezzi: già codificati nelle feature termodinamiche
    if volume_cols:
        data = data.drop(columns=volume_cols)

    # ── Sanificazione globale PRIMA dello scaler ───────────────────────────────
    # ffill/bfill non tocca gli inf: MinMaxScaler su colonne con inf produce NaN.
    data = _sanitize(data, "load_data pre-scaler")

    print(f"[load_data] NaN residui dopo sanificazione: {data.isna().any().any()}")

    # ── Trasformazione in Rendimenti (facoltativa ma consigliata) ──────────────
    if use_returns:
        print("[load_data] Trasformazione in rendimenti logaritmici per colonne prezzo...")
        for col in price_cols:
            if col in data.columns:
                # Log-return: r_t = log(P_t / P_{t-1})
                # Più stabile del pct_change per l'addestramento e additivo
                data[col] = np.log(data[col] / data[col].shift(1))
        
        # Mantieni le colonne ma riempile di zero se sono completamente vuote 
        # (indispensabile per mantenere la dimensionalità attesa dai modelli pre-addestrati)
        nan_cols = data.columns[data.isna().all()].tolist()
        if nan_cols:
            print(f"[load_data] ATTENZIONE: {len(nan_cols)} colonne sono vuote (es. delistate). Riempio con 0: {nan_cols}")
            for col in nan_cols:
                data[col] = 0.0
            
        # Il primo elemento è sempre NaN per le colonne dei prezzi dopo pct_change/log
        # Usiamo how='any' per garantire che non ci siano NaN residui in nessuna feature
        data = data.dropna(axis=0, how='any')
        print(f"[load_data] Rimosse righe con NaN residui. Nuova lunghezza: {len(data)}")

    # ── Split per normalizzazione (Prevenzione Data Leak) ─────────────────────
    # Fit dello scaler SOLO sul training set.
    # Se split_ratio non è passato, usiamo tutto il dataset (ma logghiamo avviso).
    if split_ratio is not None:
        n_train = int(len(data) * split_ratio)
        train_slice = data.iloc[:n_train]
        print(f"[load_data] Fitting scaler su train slice: {len(train_slice)} barre")
    else:
        train_slice = data
        print("[load_data] AVVISO: split_ratio non fornito, fit scaler sull'intero dataset (potenziale leak).")

    # ── Normalizzazione MinMax ─────────────────────────────────────────────────
    scaler      = MinMaxScaler()
    scaler.fit(train_slice)
    
    data_scaled = pd.DataFrame(
        scaler.transform(data),
        columns=data.columns,
        index=data.index,
    )

    thermo_col_names = [
        c for c in data_scaled.columns
        if c.startswith(("Market_", "Energy_", "Thermo_", "Volume_Delta", "Thm_"))
    ]
    print(
        f"[load_data] Dataset finale: {data_scaled.shape[1]} colonne × {len(data_scaled)} barre\n"
        f"  Termodinamica ({len(thermo_col_names)}): {thermo_col_names}"
    )

    return data_scaled.astype(np.float32), scaler


# ─── Window creation ───────────────────────────────────────────────────────────

def make_windows(data, window_size, stride, prediction_steps: int = 1):
    X, Y = [], []
    data_array = data.values
    # Range limitato per permettere prediction_steps futuri
    for i in range(0, len(data_array) - window_size - prediction_steps + 1, stride):
        X.append(data_array[i : i + window_size])
        if prediction_steps > 1:
            Y.append(data_array[i + window_size : i + window_size + prediction_steps])
        else:
            Y.append(data_array[i + window_size])

    if not X:
        raise RuntimeError(
            f"Nessuna finestra creata: {len(data_array)} barre disponibili "
            f"ma window_size={window_size}. "
            f"Riduci window_size (es. prediction.window_size=10)."
        )

    X_t = torch.tensor(np.array(X))
    Y_t = torch.tensor(np.array(Y))

    # Controllo di sanità: segnala NaN/inf prima che raggiungano il modello
    n_bad_x = (torch.isnan(X_t) | torch.isinf(X_t)).any(dim=-1).any(dim=-1).sum().item()
    n_bad_y = (torch.isnan(Y_t) | torch.isinf(Y_t)).any(dim=-1).sum().item()
    if n_bad_x or n_bad_y:
        print(
            f"[make_windows] AVVISO: {n_bad_x} finestre X e {n_bad_y} target Y "
            "con NaN/inf — clampati a 0/1. Controlla load_data()."
        )
        X_t = torch.nan_to_num(X_t, nan=0.0, posinf=1.0, neginf=0.0)
        Y_t = torch.nan_to_num(Y_t, nan=0.0, posinf=1.0, neginf=0.0)

    return X_t, Y_t


# ─── Thermodynamic stats (single-asset, for make_stats) ───────────────────────

def calculate_market_thermodynamics(df: pd.DataFrame, n_assets: int = 1, kb: float = 1.0):
    """
    Calcola Pressione e Lavoro Integrale per un singolo asset o portafoglio.
    Supporta input con colonne 'Close' e 'Volume'.
    """
    if df.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    returns = df["Close"].pct_change().dropna()
    window  = 20

    if len(returns) < window:
        return (
            pd.Series(0.0, index=df.index),
            pd.Series(0.0, index=df.index),
        )

    def rolling_entropy(x):
        prob_dist, _ = np.histogram(x, bins=20, density=True)
        prob_dist = prob_dist[prob_dist > 0]
        return entropy(prob_dist)

    s_series = returns.rolling(window).apply(rolling_entropy)
    v_series = np.log1p(df["Volume"].rolling(window).mean())

    a_series = (
        returns.rolling(window)
        .apply(lambda x: pd.Series(x).autocorr(lag=1) if len(x) > 2 else 0)
        .fillna(0)
        .replace([np.inf, -np.inf], 0)   # autocorr NaN su serie costanti
    )

    v_min    = v_series.dropna().min()
    b_const  = v_min * 0.1 if (np.isfinite(v_min) and v_min > 0) else 0.01
    v_free   = (v_series - b_const).clip(lower=0.1)

    # Clip dell'argomento per evitare overflow → inf
    exp_arg  = (2 * (s_series - np.log(v_free))).clip(-20, 20)
    t_series = np.exp(exp_arg)

    p_series = (kb * t_series) / v_free - a_series * (1 / v_series.clip(lower=1e-6)) ** 2
    p_series = p_series.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)

    delta_v         = v_series.diff()
    p_avg           = (p_series + p_series.shift(1)) / 2
    work_cumulative = (p_avg * delta_v).fillna(0).cumsum()

    return p_series, work_cumulative


# ─── Stats entrypoint ──────────────────────────────────────────────────────────

def make_stats(cfg):
    """
    Esegue l'analisi termodinamica completa, calcola divergenze e resilienza.
    Genera i grafici finali nel percorso specificato.
    """
    os.makedirs(cfg.paths.results_dir, exist_ok=True)

    tickers = list(cfg.data.tickers)
    needed  = ["^GSPC", "^TNX"]
    for n in needed:
        if n not in tickers:
            tickers.append(n)

    df_raw            = yf.download(tickers, start="2000-01-01", end="2025-01-01")
    portfolio_tickers = [t for t in list(cfg.data.tickers) if t not in needed]

    levy_entropy_norm, sackur_tetrode_entropy_norm = calculate_entropy_difference(df_raw)
    
    df_port = pd.DataFrame({
        "Close":  df_raw["Close"][portfolio_tickers].mean(axis=1),
        "Volume": df_raw["Volume"][portfolio_tickers].sum(axis=1),
        "SP500":  df_raw["Close"]["^GSPC"],
        "Rates":  df_raw["Close"]["^TNX"],
    }).dropna()
    
    # Allinea le entropie con df_port e normalizza
    entropy_data = pd.DataFrame({
        "SP500": df_port["SP500"],
        "Levy_Entropy": levy_entropy_norm,
        "Sackur_Tetrode_Entropy": sackur_tetrode_entropy_norm,
    }).dropna()
    
    scaler_entropy = MinMaxScaler()
    entropy_scaled = pd.DataFrame(
        scaler_entropy.fit_transform(entropy_data.values),
        columns=entropy_data.columns,
        index=entropy_data.index,
    )
    
    # Plot: S&P 500 e i due tipi di entropia
    plt.figure(figsize=(14, 6))
    plt.plot(entropy_scaled.index, entropy_scaled["SP500"], 
             label="S&P 500 (Normalizzato)", color="#2c3e50", linewidth=2)
    plt.plot(entropy_scaled.index, entropy_scaled["Levy_Entropy"], 
             label="Entropia Lévy (Returns Empirici)", color="#e74c3c", linewidth=1.5, alpha=0.8)
    plt.plot(entropy_scaled.index, entropy_scaled["Sackur_Tetrode_Entropy"], 
             label="Entropia Sackur-Tetrode (Gas Ideale)", color="#3498db", linewidth=1.5, alpha=0.8)
    plt.title("S&P 500 e Tipi di Entropia (Normalizzati)", fontsize=14, fontweight="bold")
    plt.xlabel("Data")
    plt.ylabel("Valore Normalizzato")
    plt.grid(True, alpha=0.2)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.paths.results_dir, "entropy_difference_raw.png"), dpi=150)
    plt.close()

    

    p, w = calculate_market_thermodynamics(df_port, len(portfolio_tickers))
    df_port["Pressure"] = p
    df_port["Work"]     = w

    # Trova il lag ottimale tra Pressione e Tassi
    max_lag  = 90
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        corrs    = [df_port["Pressure"].corr(df_port["Rates"].shift(i)) for i in range(max_lag)]
    best_lag = int(np.nanargmax(np.abs(corrs)))

    df_port["Expected_Pressure"] = df_port["Rates"].shift(best_lag)
    scaler = MinMaxScaler()
    cols_to_norm = ["SP500", "Work", "Pressure", "Expected_Pressure"]
    df_norm = pd.DataFrame(
        scaler.fit_transform(df_port[cols_to_norm]),
        columns=cols_to_norm,
        index=df_port.index,
    )

    df_port["Energy_Divergence"] = df_norm["Pressure"] - df_norm["Expected_Pressure"]
    df_port["Efficiency"]        = (df_norm["Work"] - df_norm["SP500"]).rolling(10).mean()

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, (ax1, ax3) = plt.subplots(
        2, 1, figsize=(16, 12), gridspec_kw={"height_ratios": [2, 1]}
    )

    ax1.plot(df_norm.index, df_norm["SP500"],
             label="S&P 500 (Prezzo Norm.)", color="#2c3e50", linewidth=1.5, alpha=0.5)
    ax1.plot(df_norm.index, df_norm["Work"],
             label="Lavoro Cumulativo (Work)", color="#27ae60", linewidth=2.5)

    ax2 = ax1.twinx()
    ax2.fill_between(
        df_port.index, df_port["Energy_Divergence"], 0,
        where=(df_port["Energy_Divergence"] >= 0),
        color="#e74c3c", alpha=0.3, label="Sovra-Pressione (Stress)",
    )
    ax2.fill_between(
        df_port.index, df_port["Energy_Divergence"], 0,
        where=(df_port["Energy_Divergence"] < 0),
        color="#3498db", alpha=0.3, label="Sotto-Pressione (Rally Space)",
    )

    ax1.set_title(
        f"Analisi Termodinamica: Stress Monetario e Lavoro (Lag: {best_lag}gg)",
        fontsize=16,
    )
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax1.grid(True, alpha=0.1)

    ax3.plot(df_port.index, df_port["Efficiency"],
             color="purple", label="Oscillatore di Efficienza (Work - Price)")
    ax3.axhline(0, color="black", linestyle="--", alpha=0.5)
    ax3.fill_between(df_port.index, df_port["Efficiency"], 0, color="purple", alpha=0.15)
    ax3.set_title("Efficienza del Sistema (Eccesso di Lavoro rispetto al Prezzo)", fontsize=12)
    ax3.legend(loc="upper left")
    ax3.grid(True, alpha=0.1)

    plt.tight_layout()
    plt.savefig(os.path.join(cfg.paths.results_dir, "market_thermodynamics.png"))
    plt.close()

    corr_matrix = df_port[
        ["SP500", "Pressure", "Expected_Pressure", "Work", "Energy_Divergence", "Efficiency"]
    ].corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Matrice di Correlazione delle Variabili Termodinamiche e di Mercato")
    plt.tight_layout(pad=1.5)
    plt.savefig(os.path.join(cfg.paths.results_dir, "correlation_matrix.png"))
    plt.close()

    # ── Plot Divergenza tra le due entropie ────────────────────────────────────
    entropy_divergence = entropy_data["Levy_Entropy"] - entropy_data["Sackur_Tetrode_Entropy"]
    plt.figure(figsize=(16, 6))
    plt.plot(
        entropy_divergence.index, entropy_divergence,
        label="Divergenza Energetica (Lévy - Sackur-Tetrode)", color="darkorange", linewidth=2
    )
    plt.axhline(0, color="black", linestyle="--", alpha=0.5)
    plt.title("Divergenza Energetica tra Rendimenti Empirici e Gas Ideale", fontsize=14)
    plt.xlabel("Data")
    plt.ylabel("Divergenza di Entropia")
    plt.legend()
    plt.grid(True, alpha=0.1)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.paths.results_dir, "entropy_divergence.png"), dpi=150)
    plt.close()

    # ── Indicatori di Maxwell-Boltzmann ────────────────────────────────────────
    mb_indicators = calculate_maxwell_boltzmann_indicators(df_port, window=20)
    advance_indicator = calculate_advanced_econophysics_indicators(df_port, mb_indicators, window=20)

    
    # Normalizza gli indicatori MB e S&P 500 per il plot
    mb_plot_data = pd.DataFrame({
        "SP500": df_port["SP500"],
        "MB_Temperature": mb_indicators["MB_Temperature"],
        "MB_Kinetic_Energy": mb_indicators["MB_Kinetic_Energy"],
        "MB_Velocity_Ratio": mb_indicators["MB_Velocity_Ratio"],
    }).dropna()
    
    scaler_mb = MinMaxScaler()
    mb_scaled = pd.DataFrame(
        scaler_mb.fit_transform(mb_plot_data.values),
        columns=mb_plot_data.columns,
        index=mb_plot_data.index,
    )
    
    # Plot 1: Indicatori MB nel tempo
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    
    # Temperatura su asse sinistro, S&P 500 su asse destro
    ax = axes[0]
    ax.plot(mb_scaled.index, mb_scaled["MB_Temperature"], 
            color="#27ae60", linewidth=2, label="MB Temperature")
    ax.set_ylabel("MB Temperature (Norm.)", fontweight="bold")
    ax.grid(True, alpha=0.2)
    ax2 = ax.twinx()
    ax2.plot(mb_scaled.index, mb_scaled["SP500"], 
             color="#2c3e50", linewidth=1.5, alpha=0.6, label="S&P 500")
    ax2.set_ylabel("S&P 500 (Norm.)", fontweight="bold")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    
    # Energia cinetica su asse sinistro, S&P 500 su asse destro
    ax = axes[1]
    ax.plot(mb_scaled.index, mb_scaled["MB_Kinetic_Energy"], 
            color="#e74c3c", linewidth=2, label="MB Kinetic Energy")
    ax.set_ylabel("MB Kinetic Energy (Norm.)", fontweight="bold")
    ax.grid(True, alpha=0.2)
    ax2 = ax.twinx()
    ax2.plot(mb_scaled.index, mb_scaled["SP500"], 
             color="#2c3e50", linewidth=1.5, alpha=0.6, label="S&P 500")
    ax2.set_ylabel("S&P 500 (Norm.)", fontweight="bold")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    
    # Velocity Ratio su asse sinistro, S&P 500 su asse destro
    ax = axes[2]
    ax.plot(mb_scaled.index, mb_scaled["MB_Velocity_Ratio"], 
            color="#f39c12", linewidth=2, label="MB Velocity Ratio")
    ax.axhline(1, color="black", linestyle="--", alpha=0.5, label="Equilibrio (y=1)")
    ax.set_ylabel("MB Velocity Ratio (Norm.)", fontweight="bold")
    ax.set_xlabel("Data")
    ax.grid(True, alpha=0.2)
    ax2 = ax.twinx()
    ax2.plot(mb_scaled.index, mb_scaled["SP500"], 
             color="#2c3e50", linewidth=1.5, alpha=0.6, label="S&P 500")
    ax2.set_ylabel("S&P 500 (Norm.)", fontweight="bold")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    
    fig.suptitle("Indicatori di Maxwell-Boltzmann e S&P 500", 
                 fontsize=16, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.paths.results_dir, "maxwell_boltzmann_indicators.png"), dpi=150)
    plt.close()
    
    # Plot 2: Correlazione tra MB indicators e S&P 500
    mb_corr = mb_plot_data.corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(mb_corr, annot=True, cmap="coolwarm", center=0, fmt=".3f", 
                linewidths=1, cbar_kws={"label": "Correlazione"})
    plt.title("Correlazione tra Indicatori Maxwell-Boltzmann e S&P 500", 
              fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.paths.results_dir, "maxwell_boltzmann_correlation.png"), dpi=150)
    plt.close()

    # ── Indicatori Avanzati di Econofisica ─────────────────────────────────────
    # Normalizza gli indicatori avanzati e S&P 500 per il plot
    adv_plot_data = pd.DataFrame({
        "SP500": df_port["SP500"],
        "Thermo_Efficiency": advance_indicator["Thermo_Efficiency"],
        "Velocity_Order": advance_indicator["Velocity_Order"],
        "Market_Boiling_Z": advance_indicator["Market_Boiling_Z"],
    }).dropna()
    
    scaler_adv = MinMaxScaler()
    adv_scaled = pd.DataFrame(
        scaler_adv.fit_transform(adv_plot_data.values),
        columns=adv_plot_data.columns,
        index=adv_plot_data.index,
    )
    
    # Plot 1: Indicatori Avanzati nel tempo
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    
    # Efficienza Termica su asse sinistro, S&P 500 su asse destro
    ax = axes[0]
    ax.plot(adv_scaled.index, adv_scaled["Thermo_Efficiency"], 
            color="#9b59b6", linewidth=2, label="Thermo Efficiency")
    ax.set_ylabel("Thermo Efficiency (Norm.)", fontweight="bold")
    ax.grid(True, alpha=0.2)
    ax2 = ax.twinx()
    ax2.plot(adv_scaled.index, adv_scaled["SP500"], 
             color="#2c3e50", linewidth=1.5, alpha=0.6, label="S&P 500")
    ax2.set_ylabel("S&P 500 (Norm.)", fontweight="bold")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    
    # Ordine di Velocità su asse sinistro, S&P 500 su asse destro
    ax = axes[1]
    ax.plot(adv_scaled.index, adv_scaled["Velocity_Order"], 
            color="#16a085", linewidth=2, label="Velocity Order")
    ax.set_ylabel("Velocity Order (Norm.)", fontweight="bold")
    ax.grid(True, alpha=0.2)
    ax2 = ax.twinx()
    ax2.plot(adv_scaled.index, adv_scaled["SP500"], 
             color="#2c3e50", linewidth=1.5, alpha=0.6, label="S&P 500")
    ax2.set_ylabel("S&P 500 (Norm.)", fontweight="bold")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    
    # Market Boiling Z su asse sinistro, S&P 500 su asse destro
    ax = axes[2]
    ax.plot(adv_scaled.index, adv_scaled["Market_Boiling_Z"], 
            color="#e67e22", linewidth=2, label="Market Boiling Z")
    ax.axhline(0, color="black", linestyle="--", alpha=0.5, label="Neutralità (y=0)")
    ax.fill_between(adv_scaled.index, adv_scaled["Market_Boiling_Z"], 0, 
                    where=(adv_scaled["Market_Boiling_Z"] >= 0), 
                    color="#e74c3c", alpha=0.2, label="Stress Termico")
    ax.fill_between(adv_scaled.index, adv_scaled["Market_Boiling_Z"], 0, 
                    where=(adv_scaled["Market_Boiling_Z"] < 0), 
                    color="#3498db", alpha=0.2, label="Stabilità")
    ax.set_ylabel("Market Boiling Z (Norm.)", fontweight="bold")
    ax.set_xlabel("Data")
    ax.grid(True, alpha=0.2)
    ax2 = ax.twinx()
    ax2.plot(adv_scaled.index, adv_scaled["SP500"], 
             color="#2c3e50", linewidth=1.5, alpha=0.6, label="S&P 500")
    ax2.set_ylabel("S&P 500 (Norm.)", fontweight="bold")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    
    fig.suptitle("Indicatori Avanzati di Econofisica e S&P 500", 
                 fontsize=16, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.paths.results_dir, "advanced_econophysics_indicators.png"), dpi=150)
    plt.close()
    
    # Plot 2: Correlazione tra indicatori avanzati e S&P 500
    adv_corr = adv_plot_data.corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(adv_corr, annot=True, cmap="coolwarm", center=0, fmt=".3f", 
                linewidths=1, cbar_kws={"label": "Correlazione"})
    plt.title("Correlazione tra Indicatori Avanzati di Econofisica e S&P 500", 
              fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.paths.results_dir, "advanced_econophysics_correlation.png"), dpi=150)
    plt.close()

    # ── Indicatori Intraday (ultimi 30 giorni di dati per visibilità temporale) ───────
    try:
        intraday_end = datetime.datetime.now()
        intraday_start = intraday_end - datetime.timedelta(days=30)
        
        df_intraday = yf.download(
            "^GSPC",
            start=intraday_start,
            end=intraday_end,
            interval="5m",
            progress=False
        )
        
        if not df_intraday.empty and len(df_intraday) > 20:
            # Prepara il DataFrame con Close e Volume
            df_intraday_prepared = pd.DataFrame({
                "Close": df_intraday["Close"],
                "Volume": df_intraday["Volume"],
            }).dropna()
            
            # Calcola indicatori intraday
            intraday_indicators = calculate_intraday_thermodynamics(df_intraday_prepared, window=14)
            
            # Normalizza gli indicatori intraday e prezzo per il plot
            intra_plot_data = pd.DataFrame({
                "Close": df_intraday_prepared["Close"],
                "Market_Viscosity": intraday_indicators["Market_Viscosity"],
                "Kinetic_Power_Flux": intraday_indicators["Kinetic_Power_Flux"],
                "Market_Reynolds_Number": intraday_indicators["Market_Reynolds_Number"],
            }).dropna()
            
            scaler_intra = MinMaxScaler()
            intra_scaled = pd.DataFrame(
                scaler_intra.fit_transform(intra_plot_data.values),
                columns=intra_plot_data.columns,
                index=intra_plot_data.index,
            )
            
            # Plot 1: Indicatori Intraday nel tempo
            fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
            
            # Market Viscosity su asse sinistro, Close su asse destro
            ax = axes[0]
            ax.plot(intra_scaled.index, intra_scaled["Market_Viscosity"], 
                    color="#3498db", linewidth=2, label="Market Viscosity")
            ax.set_ylabel("Market Viscosity (Norm.)", fontweight="bold")
            ax.grid(True, alpha=0.2)
            ax2 = ax.twinx()
            ax2.plot(intra_scaled.index, intra_scaled["Close"], 
                     color="#2c3e50", linewidth=1.5, alpha=0.6, label="S&P 500 Close")
            ax2.set_ylabel("S&P 500 Close (Norm.)", fontweight="bold")
            ax.legend(loc="upper left")
            ax2.legend(loc="upper right")
            
            # Kinetic Power Flux su asse sinistro, Close su asse destro
            ax = axes[1]
            ax.plot(intra_scaled.index, intra_scaled["Kinetic_Power_Flux"], 
                    color="#e74c3c", linewidth=2, label="Kinetic Power Flux")
            ax.set_ylabel("Kinetic Power Flux (Norm.)", fontweight="bold")
            ax.grid(True, alpha=0.2)
            ax2 = ax.twinx()
            ax2.plot(intra_scaled.index, intra_scaled["Close"], 
                     color="#2c3e50", linewidth=1.5, alpha=0.6, label="S&P 500 Close")
            ax2.set_ylabel("S&P 500 Close (Norm.)", fontweight="bold")
            ax.legend(loc="upper left")
            ax2.legend(loc="upper right")
            
            # Reynolds Number su asse sinistro, Close su asse destro
            ax = axes[2]
            ax.plot(intra_scaled.index, intra_scaled["Market_Reynolds_Number"], 
                    color="#f39c12", linewidth=2, label="Reynolds Number")
            ax.set_ylabel("Reynolds Number (Norm.)", fontweight="bold")
            ax.set_xlabel("Data/Ora")
            ax.grid(True, alpha=0.2)
            ax2 = ax.twinx()
            ax2.plot(intra_scaled.index, intra_scaled["Close"], 
                     color="#2c3e50", linewidth=1.5, alpha=0.6, label="S&P 500 Close")
            ax2.set_ylabel("S&P 500 Close (Norm.)", fontweight="bold")
            ax.legend(loc="upper left")
            ax2.legend(loc="upper right")
            
            fig.suptitle("Indicatori Intraday (5m) - Ultimi 30 Giorni", 
                         fontsize=16, fontweight="bold", y=0.995)
            plt.tight_layout()
            plt.savefig(os.path.join(cfg.paths.results_dir, "intraday_thermodynamics_indicators.png"), dpi=150)
            plt.close()
            
            # Plot 2: Correlazione tra indicatori intraday e prezzo
            intra_corr = intra_plot_data.corr()
            plt.figure(figsize=(10, 6))
            sns.heatmap(intra_corr, annot=True, cmap="coolwarm", center=0, fmt=".3f", 
                        linewidths=1, cbar_kws={"label": "Correlazione"})
            plt.title("Correlazione tra Indicatori Intraday (5m) e S&P 500 Close", 
                      fontsize=14, fontweight="bold")
            plt.tight_layout()
            plt.savefig(os.path.join(cfg.paths.results_dir, "intraday_thermodynamics_correlation.png"), dpi=150)
            plt.close()
            
            print(f"[Stats] Plot intraday generati con {len(intra_plot_data)} barre (5m)")
        else:
            print("[Stats] AVVERTENZA: Dati intraday insufficienti per generare plot")
    
    except Exception as e:
        print(f"[Stats] AVVERTENZA: Calcolo indicatori intraday fallito ({e})")

    print(f"[Stats] Analisi completata. Grafici salvati in {cfg.paths.results_dir}")
    return df_port


