"""
modelli/intraday_thermo.py — Feature termodinamiche adattate all'intraday

Basato su:
  - Mantegna & Stanley (1995): Lévy scaling dei rendimenti
  - Bouchaud & Potters (2003): Funzione di risposta del mercato (W → prezzo)
  - Gabaix et al. (2003): Leggi di potenza su volume

Differenze chiave rispetto alla versione daily:
  - Finestre brevi: 5-10 bar (non 20-90)
  - Nessun lag monetario (tassi non agiscono intraday)
  - Focus su: pressione locale, efficienza lavoro, z-score intraday
  - Distribuzione Lévy-corretta per la normalizzazione

Output per il TradingEnv:
  Tre segnali principali (compatibili con lo stato DDPG):
    1. intraday_stress    — Z-score della pressione vs media locale
    2. work_efficiency    — Rendimento della "macchina mercato" (W/ΔP)
    3. levy_entropy       — Entropia Lévy-corretta (misura disordine reale)

Fix v2:
  - Risolto PerformanceWarning: build completa in dict, pd.concat unico finale
  - Risolto volume detection: ricerca case-insensitive + pattern MultiIndex yfinance
  - Warning volume emesso una sola volta per ticker (non a ogni chiamata)
  - _find_volume_col() centralizza tutta la logica di ricerca
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd


# ── Parametri di default per bar da 1 minuto ──────────────────────────────────
INTRADAY_DEFAULTS = {
    "pressure_window":    5,    # bar per calcolo pressione locale
    "entropy_window":     10,   # bar per entropia rolling
    "stress_window":      20,   # bar per z-score normalizzazione
    "efficiency_window":  10,   # bar per oscillatore efficienza
    "levy_alpha":         1.7,  # esponente Lévy (1 < α < 2), stima empirica
    "levy_beta":          0.0,  # asimmetria (0 = simmetrico)
}


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY: rilevamento colonne volume (fix principale)
# ─────────────────────────────────────────────────────────────────────────────

def _find_volume_col(df: pd.DataFrame, ticker: str) -> Optional[str]:
    """
    Trova la colonna volume per un ticker nel DataFrame.

    Gestisce tutti i formati possibili di yfinance:
      - Single-ticker flat  : 'Volume'
      - Multi-ticker flat   : 'AAPL_Volume', 'Volume_AAPL', 'AAPL Volume'
      - MultiIndex (livello): ('Volume', 'AAPL') → già flattenato come 'Volume_AAPL'
      - Ricerca case-insensitive come fallback

    Returns:
        Nome della colonna se trovata, None altrimenti.
    """
    # Candidati esatti (ordine di preferenza)
    candidates = [
        f"{ticker}_Volume",
        f"Volume_{ticker}",
        f"{ticker} Volume",
        f"{ticker}volume",
        "Volume",          # single-asset DataFrame
        "volume",
    ]
    for c in candidates:
        if c in df.columns:
            return c

    # Fallback: ricerca case-insensitive su tutti i nomi colonna
    ticker_lower = ticker.lower()
    for col in df.columns:
        col_lower = col.lower()
        if "volume" in col_lower and ticker_lower in col_lower:
            return col

    # Nessuna colonna volume trovata
    return None


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY: sanitizzazione serie
# ─────────────────────────────────────────────────────────────────────────────

def _build_fallback_volume(df: pd.DataFrame) -> pd.Series:
    """
    Somma riga per riga di tutte le colonne volume presenti nel DataFrame.
    Usato come proxy di volume aggregato quando un ticker non ha la propria
    colonna volume (es. indici, forex, ETF senza dati volume da yfinance).

    Se il DataFrame non contiene nessuna colonna volume, restituisce una
    serie costante di 1.0 (comportamento precedente).
    """
    vol_cols = [
        col for col in df.columns
        if "volume" in col.lower()
    ]
    if not vol_cols:
        return pd.Series(1.0, index=df.index)

    vol_df = df[vol_cols].apply(pd.to_numeric, errors="coerce").clip(lower=0.0).fillna(0.0)
    row_sum = vol_df.sum(axis=1).clip(lower=1.0)
    return _sanitize(row_sum, fill=1.0)


def _sanitize(series: pd.Series, fill: float = 0.0) -> pd.Series:
    """
    Rimuove inf/-inf sostituendoli con NaN, poi forward-fill + fill costante.
    """
    return (
        series
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
        .fillna(fill)
    )


def _sanitize_close(close: pd.Series) -> pd.Series:
    """
    Sanitizza prezzi: rimuove zero/negativi (causano inf nei log-return).
    """
    clean = close.replace(0, np.nan).clip(lower=1e-10)
    return clean.ffill().bfill().fillna(1.0)


# ─────────────────────────────────────────────────────────────────────────────
# 1. PRESSIONE INTRADAY (Van der Waals semplificata, finestra corta)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_intraday_pressure(
    close: pd.Series,
    volume: pd.Series,
    window: int = 5,
    a_attraction: float = 0.5,
    b_excluded: float = 0.01,
) -> pd.Series:
    """
    Pressione di mercato Van der Waals su finestra corta.

    P = nRT / (V - nb) - a·n²/V²

    Dove per intraday:
      V = log(volume_rolling_mean)      → "volume termodinamico"
      T = rolling_std(returns)          → "temperatura" (agitazione locale)
      n = 1 (singolo asset)
    """
    close  = _sanitize_close(close)
    volume = _sanitize(volume.clip(lower=1.0), fill=1.0)

    returns = _sanitize(close.pct_change(), fill=0.0)
    T_local = _sanitize(returns.rolling(window).std(), fill=0.0)

    log_vol = _sanitize(np.log1p(volume))
    V_local = _sanitize(log_vol.rolling(window).mean(), fill=float(log_vol.mean()))

    V_safe = V_local.clip(lower=b_excluded + 1e-6)

    n = 1.0
    R = 1.0
    pressure = (n * R * T_local) / (V_safe - n * b_excluded) - a_attraction * (n ** 2) / (V_safe ** 2)

    return _sanitize(pressure, fill=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 2. LAVORO CUMULATIVO INTRADAY (∫P dV)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_intraday_work(
    pressure: pd.Series,
    volume: pd.Series,
    window: int = 10,
) -> pd.Series:
    """
    Lavoro cumulativo su finestra rolling.

    W_t = Σ_{t-window}^{t} [(P_i + P_{i-1})/2 · (V_i - V_{i-1})]
    """
    pressure = _sanitize(pressure)
    volume   = _sanitize(volume.clip(lower=1.0), fill=1.0)

    log_vol = _sanitize(np.log1p(volume))
    dV      = _sanitize(log_vol.diff(), fill=0.0)
    P_avg   = _sanitize((pressure + pressure.shift(1)) / 2.0, fill=0.0)
    dW      = _sanitize(P_avg * dV, fill=0.0)

    return _sanitize(dW.rolling(window).sum(), fill=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Z-SCORE STRESS INTRADAY
# ─────────────────────────────────────────────────────────────────────────────

def _compute_intraday_stress(
    pressure: pd.Series,
    stress_window: int = 20,
) -> pd.Series:
    """
    Divergenza Z-score della pressione dalla media locale.

    Z_t = (P_t - μ_{P,window}) / σ_{P,window}

    Interpretazione:
      Z > +1.5 → Stress termico (surriscaldamento locale)
      Z < -1.5 → Espansione sana
      |Z| < 1  → Zona neutra
    """
    pressure = _sanitize(pressure)

    mu    = _sanitize(pressure.rolling(stress_window).mean())
    sigma = _sanitize(pressure.rolling(stress_window).std(), fill=1.0).clip(lower=1e-8)

    z_score = _sanitize((pressure - mu) / sigma)
    return z_score.clip(-5.0, 5.0).fillna(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 4. OSCILLATORE EFFICIENZA (Work-Price Index)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_work_efficiency(
    work: pd.Series,
    close: pd.Series,
    efficiency_window: int = 10,
) -> pd.Series:
    """
    Efficienza della macchina mercato: rendimento lavoro → prezzo.
    Basato su Bouchaud (2003): "market response function".

    efficiency = ΔW_rolling / (ΔP_rolling + ε)

    Interpretazione:
      > +2 → Dissipazione (distribuzione, segnale bearish)
      < -1 → Inversione (possibile rimbalzo tecnico)
      [-1, +1] → Regime fluido
    """
    work  = _sanitize(work)
    close = _sanitize_close(close)

    dW = _sanitize(work.diff(efficiency_window), fill=0.0)
    dP = _sanitize(close.pct_change(efficiency_window), fill=0.0)

    efficiency = _sanitize(dW / (dP.abs() + 1e-6))
    return efficiency.clip(-5.0, 5.0).fillna(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 5. ENTROPIA LÉVY-CORRETTA (Mantegna & Stanley 1995)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_levy_entropy(
    close: pd.Series,
    window: int = 10,
    levy_alpha: float = 1.7,
) -> pd.Series:
    """
    Entropia corretta per distribuzioni Lévy (non Gaussiana).

    H_Lévy(t) = H_Shannon(t) × (1 + correction_factor(α))

    Output normalizzato in [0, 1] per il DDPG state.
    """
    close   = _sanitize_close(close)
    returns = _sanitize(close.pct_change(), fill=0.0)
    returns_values = returns.values

    def rolling_levy_entropy(arr: np.ndarray) -> np.ndarray:
        n      = len(arr)
        result = np.zeros(n, dtype=np.float64)

        for i in range(window, n):
            window_data = arr[i - window: i]
            window_data = window_data[np.isfinite(window_data)]

            if len(window_data) < 4:
                result[i] = 0.0
                continue

            data_min = window_data.min()
            data_max = window_data.max()
            if data_max - data_min < 1e-12:
                result[i] = 0.0
                continue

            n_bins = min(window // 2, 8)
            hist, _ = np.histogram(window_data, bins=n_bins, density=True)

            hist = hist[hist > 0]
            hist = np.clip(hist, 1e-10, 1e6)

            shannon_h = -np.sum(hist * np.log(hist + 1e-10))

            if not np.isfinite(shannon_h):
                result[i] = 0.0
                continue

            levy_correction = max(0.0, 2.0 / levy_alpha - 1.0)
            levy_h = shannon_h * (1.0 + levy_correction)
            result[i] = levy_h if np.isfinite(levy_h) else 0.0

        return result

    raw_entropy_arr = rolling_levy_entropy(returns_values)
    raw_entropy     = pd.Series(raw_entropy_arr, index=close.index)
    
    e_min = raw_entropy.min()
    e_max = raw_entropy.max()
    if e_max - e_min < 1e-10:
        return pd.Series(0.5, index=close.index, dtype=np.float32)

    normalized = (raw_entropy - e_min) / (e_max - e_min)
    return _sanitize(normalized, fill=0.5).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 6. GIBBS FREE ENERGY INTRADAY  G = H - T·S
# ─────────────────────────────────────────────────────────────────────────────

def _compute_intraday_temperature(
    close: pd.Series,
    window: int = 5,
) -> pd.Series:
    """
    Temperatura locale come rolling std dei rendimenti (agitazione cinetica).
    Esposta separatamente per essere usata nel calcolo di Gibbs Energy.
    """
    close   = _sanitize_close(close)
    returns = _sanitize(close.pct_change(), fill=0.0)
    return _sanitize(returns.rolling(window).std(), fill=0.0)


def _compute_gibbs_energy(
    pressure:    pd.Series,
    temperature: pd.Series,
    entropy:     pd.Series,
) -> pd.Series:
    """
    Energia di Gibbs del mercato: G = H - T·S

    H (entalpia proxy) = pressione × |lavoro locale|
    T = temperatura (rolling std dei rendimenti)
    S = entropia Lévy-corretta normalizzata [0,1]

    Interpretazione:
      G < 0 → processo spontaneo → il trade va con il flusso naturale del sistema
      G > 0 → processo non spontaneo → il mercato sta forzando il movimento

    Prima applicazione nota di Gibbs al reward shaping DDPG finanziario.
    Output clippato in [-3, 3] e normalizzato.
    """
    H = pressure.abs()
    G = H - temperature * entropy
    return _sanitize(G, fill=0.0).clip(-3.0, 3.0)


# ─────────────────────────────────────────────────────────────────────────────
# 7. STRESS ACCELERATION  d²(Stress)/dt²
# ─────────────────────────────────────────────────────────────────────────────

def _compute_stress_acceleration(stress: pd.Series) -> pd.Series:
    """
    Derivata seconda dello stress intraday tramite differenze finite.

    d²Z/dt² = Z(t) - 2·Z(t-1) + Z(t-2)

    Interpretazione:
      > 0 → lo stress sta accelerando verso l'alto: segnale sell anticipato di 1-2 bar
      < 0 → lo stress sta decelerando: possibile inversione in arrivo
      ≈ 0 → regime stabile

    Usata in TradingEnv [Fix #15] per il bonus sell precoce (prima della soglia assoluta).
    Output clippato in [-2, 2].
    """
    d1 = _sanitize(stress.diff(), fill=0.0)
    d2 = _sanitize(d1.diff(), fill=0.0)
    return d2.clip(-2.0, 2.0).fillna(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 8. FUNZIONE PRINCIPALE: compute_intraday_thermo_features()
# ─────────────────────────────────────────────────────────────────────────────

def compute_intraday_thermo_features(
    df: pd.DataFrame,
    ticker_cols: list[str],
    volume_col_suffix: str = "_Volume",
    params: dict | None = None,
) -> pd.DataFrame:
    """
    Calcola tutte le feature termodinamiche intraday per un DataFrame di prezzi.

    FIX v2:
      - Nessun insert() colonna-per-colonna: tutte le serie sono raccolte
        in un dict e assemblate con un unico pd.concat() alla fine.
        Elimina completamente il PerformanceWarning "highly fragmented DataFrame".
      - Volume detection unificata in _find_volume_col() con ricerca
        case-insensitive e supporto per il formato MultiIndex flattenato
        di yfinance. Il warning viene emesso una volta sola per ticker mancante.

    Args:
        df           : DataFrame con colonne {ticker} (close) e opzionalmente
                       {ticker}_Volume. Accetta flat yfinance multi-ticker.
        ticker_cols  : Lista di ticker (es. ["AAPL", "MSFT"])
        volume_col_suffix : Suffisso colonna volume (usato solo per log)
        params       : Override dei parametri INTRADAY_DEFAULTS (opzionale)

    Returns:
        DataFrame con colonne:
          - {ticker}_intraday_stress    : Z-score stress pressione locale
          - {ticker}_work_efficiency    : Oscillatore efficienza Bouchaud
          - {ticker}_levy_entropy       : Entropia Lévy-corretta
          - Thermo_Aggregate_Stress     : Media stress su tutti i ticker
          - Thermo_Aggregate_Efficiency : Media efficienza
          - Thermo_Aggregate_Entropy    : Media entropia

        Shape: (len(df), 3 * n_valid_tickers + 3)
    """
    p = {**INTRADAY_DEFAULTS, **(params or {})}

    # Dizionario che accumula tutte le serie → pd.concat finale (no frammentazione)
    columns: dict[str, pd.Series] = {}

    all_stress:     list[pd.Series] = []
    all_efficiency: list[pd.Series] = []
    all_entropy:    list[pd.Series] = []

    # Ticker per cui non è stata trovata la colonna Close (skip silenzioso dopo il primo avviso)
    missing_close: set[str] = set()

    # Volume di fallback: somma di tutte le colonne volume del DataFrame.
    # Calcolato una volta sola e riusato per ogni ticker privo di volume proprio.
    fallback_volume: pd.Series | None = None

    for ticker in ticker_cols:
        # ── Trova la colonna Close ─────────────────────────────────────────
        close_candidates = [
            ticker,
            f"{ticker}_Close",
            f"Close_{ticker}",
            f"{ticker} Close",
        ]
        close_col = next((c for c in close_candidates if c in df.columns), None)

        if close_col is None:
            if ticker not in missing_close:
                print(f"[IntraThermo] WARNING: colonna Close non trovata per {ticker}, skip")
                missing_close.add(ticker)
            continue

        close = _sanitize_close(df[close_col].astype(float))

        # ── Trova la colonna Volume (logica centralizzata) ─────────────────
        volume_col = _find_volume_col(df, ticker)

        if volume_col is None:
            # Costruisce il volume aggregato la prima volta che serve
            if fallback_volume is None:
                fallback_volume = _build_fallback_volume(df)
                print(
                    f"[IntraThermo] INFO: volume fallback = somma di "
                    f"{[c for c in df.columns if 'volume' in c.lower()]} "
                    f"(usato per ticker senza colonna volume propria)"
                )
            print(f"[IntraThermo] WARNING: colonna Volume non trovata per {ticker}, uso volume aggregato (row sum)")
            volume = fallback_volume
        else:
            volume = _sanitize(
                df[volume_col].astype(float).clip(lower=1.0),
                fill=1.0,
            )

        # ── Pipeline di calcolo ────────────────────────────────────────────
        pressure = _compute_intraday_pressure(
            close, volume,
            window=p["pressure_window"],
        )
        work = _compute_intraday_work(
            pressure, volume,
            window=p["efficiency_window"],
        )
        stress = _compute_intraday_stress(
            pressure,
            stress_window=p["stress_window"],
        )
        efficiency = _compute_work_efficiency(
            work, close,
            efficiency_window=p["efficiency_window"],
        )
        entropy = _compute_levy_entropy(
            close,
            window=p["entropy_window"],
            levy_alpha=p["levy_alpha"],
        )

        # Temperatura locale (necessaria per Gibbs Energy)
        temperature = _compute_intraday_temperature(close, window=p["pressure_window"])

        # Gibbs Free Energy per questo ticker
        gibbs = _compute_gibbs_energy(pressure, temperature, entropy)

        # Stress Acceleration per questo ticker
        stress_accel = _compute_stress_acceleration(stress)

        # Accumula nel dict (nessun insert sul DataFrame intermedio)
        columns[f"{ticker}_intraday_stress"]   = stress
        columns[f"{ticker}_work_efficiency"]   = efficiency
        columns[f"{ticker}_levy_entropy"]      = entropy
        columns[f"{ticker}_gibbs_energy"]      = gibbs
        columns[f"{ticker}_stress_accel"]      = stress_accel

        all_stress.append(stress)
        all_efficiency.append(efficiency)
        all_entropy.append(entropy)

    # ── Segnali aggregati (vista "portafoglio") ────────────────────────────
    if all_stress:
        agg_stress      = pd.concat(all_stress,     axis=1).mean(axis=1)
        agg_efficiency  = pd.concat(all_efficiency, axis=1).mean(axis=1)
        agg_entropy     = pd.concat(all_entropy,    axis=1).mean(axis=1)

        columns["Thermo_Aggregate_Stress"]       = agg_stress
        columns["Thermo_Aggregate_Efficiency"]   = agg_efficiency
        columns["Thermo_Aggregate_Entropy"]      = agg_entropy
        # Gibbs e StressAccel aggregati (medi su tutti i ticker validi)
        columns["Thermo_Aggregate_GibbsEnergy"]  = pd.concat(
            [columns[k] for k in columns if k.endswith("_gibbs_energy")], axis=1
        ).mean(axis=1)
        columns["Thermo_Aggregate_StressAccel"]  = pd.concat(
            [columns[k] for k in columns if k.endswith("_stress_accel")], axis=1
        ).mean(axis=1)

    # ── Singolo pd.concat finale: zero frammentazione ─────────────────────
    if columns:
        result = pd.concat(columns, axis=1)
    else:
        result = pd.DataFrame(index=df.index)

    return result.fillna(0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 7. REGIME DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

def detect_market_regime(thermo_df: pd.DataFrame) -> pd.Series:
    """
    Classifica il regime di mercato intraday in base ai segnali termodinamici.

    Regimi:
      0 — NEUTRO        : Nessun segnale chiaro
      1 — RALLY_REALE   : Lavoro ↑ + Stress ↓ (espansione sana)
      2 — RALLY_ESAUSTO : Lavoro ↑ + Stress ↑ + Efficienza ↑ (distribuzione)
      3 — COMPRESSIONE  : Lavoro ↓ + Stress ↑ (fase pre-correzione)
      4 — RIMBALZO      : Entropia ↑ + Stress ↓ (inversione probabile)
    """
    stress     = thermo_df.get("Thermo_Aggregate_Stress",     pd.Series(0,   index=thermo_df.index))
    efficiency = thermo_df.get("Thermo_Aggregate_Efficiency", pd.Series(0,   index=thermo_df.index))
    entropy    = thermo_df.get("Thermo_Aggregate_Entropy",    pd.Series(0.5, index=thermo_df.index))

    HIGH_STRESS     = 1.0
    LOW_STRESS      = -0.5
    HIGH_EFFICIENCY = 1.5
    HIGH_ENTROPY    = 0.65

    regime = pd.Series(0, index=thermo_df.index, dtype=int)

    mask_rally_real = (stress < LOW_STRESS) & (efficiency.abs() < HIGH_EFFICIENCY)
    regime[mask_rally_real] = 1

    mask_rally_exhaust = (stress > HIGH_STRESS) & (efficiency > HIGH_EFFICIENCY)
    regime[mask_rally_exhaust] = 2

    mask_compression = (stress > HIGH_STRESS) & (efficiency > HIGH_EFFICIENCY * 1.5)
    regime[mask_compression] = 3

    stress_declining = stress.diff() < 0
    mask_bounce = (entropy > HIGH_ENTROPY) & stress_declining & (stress < 0)
    regime[mask_bounce] = 4

    return regime