"""
modelli/thermo_statistics.py
════════════════════════════════════════════════════════════════════════════════

ThermoStatisticsEngine — Statistiche termodinamiche avanzate per mercati finanziari

Implementa 6 indicatori originali che estendono il modello Van der Waals
base con concetti dalla termodinamica statistica, mai applicati insieme
alla predizione di mercato intraday:

┌─────────────────────────────────────────────────────────────────────────┐
│  1. Z_market   — Compressibility Factor  (PV / nRT)                    │
│     Misura la deviazione dal gas ideale.                                 │
│     Z > 1  → interazioni repulsive  (trend continuation)               │
│     Z < 1  → interazioni attrattive (mean-reversion imminente)         │
│     Z ≈ 1  → mercato efficiente, prezzo giusto                         │
│                                                                          │
│  2. η_carnot   — Carnot Efficiency                                      │
│     η = 1 - T_cold / T_hot  (da T rolling min/max)                     │
│     Limite termodinamico superiore all'efficienza del trend.            │
│     η → 0  → ciclo quasi-chiuso, trend esausto                         │
│     η → 1  → gradiente termico massimo, trend potenzialmente forte     │
│                                                                          │
│  3. σ_entropy  — Entropy Production Rate                               │
│     dS/dt normalizzata (derivata prima con Savitzky-Golay)             │
│     σ > 0  → processo irreversibile, il mercato brucia energia         │
│     σ ≈ 0  → quasi-statico, trend sostenibile                         │
│     σ < 0  → auto-organizzazione (rara, precede breakout forti)       │
│                                                                          │
│  4. MPRI       — Monetary Pressure Resonance Index                     │
│     Misura l'ampiezza E il segno della risonanza tra P_market          │
│     e 1/r al lag ottimale. Distingue interferenza costruttiva /        │
│     distruttiva tra politica monetaria e pressione di mercato.         │
│                                                                          │
│  5. F_helmholtz — Helmholtz Free Energy                                │
│     F = U - T·S  dove U = |W_cum|, T = temperatura, S = entropia      │
│     ΔF < 0  → trade termodinamicamente spontaneo (BUY favorevole)     │
│     ΔF > 0  → trade richiede "energia esterna" (attenzione)           │
│                                                                          │
│  6. Thermo_Quality — Work-to-Heat Ratio                                │
│     Q = dW / (T · dS + ε)    [lavoro / calore dissipato]              │
│     > 1  → conversione efficiente energia → movimento di prezzo        │
│     < 1  → la maggior parte dell'energia si perde come disordine       │
└─────────────────────────────────────────────────────────────────────────┘

Integrazione in ThermoStateBuilder:
    from modelli.thermo_statistics import ThermoStatisticsEngine
    engine = ThermoStatisticsEngine(is_intraday=True)
    stats_df = engine.compute(base_dict, close, volume, rates)

Colonne output (prefisso "Thm_"):
    Thm_Zmarket, Thm_CarnotEff, Thm_EntropyProd,
    Thm_MPRI, Thm_Helmholtz, Thm_Quality
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from typing import Optional


# ══════════════════════════════════════════════════════════════════════════════
# COSTANTI
# ══════════════════════════════════════════════════════════════════════════════

STAT_COLS = [
    "Thm_Zmarket",
    "Thm_CarnotEff",
    "Thm_EntropyProd",
    "Thm_MPRI",
    "Thm_Helmholtz",
    "Thm_Quality",
]

_DEFAULT_WINDOWS = {
    "daily":    {"short": 20, "long": 60, "sg_window": 11, "sg_poly": 3, "lag": 90},
    "intraday": {"short":  5, "long": 20, "sg_window":  7, "sg_poly": 2, "lag": 30},
}


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY
# ══════════════════════════════════════════════════════════════════════════════

def _sanitize(s: pd.Series, fill: float = 0.0) -> pd.Series:
    return s.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(fill)


def _clip_zscore(s: pd.Series, window: int, clip: float = 3.0) -> pd.Series:
    """Z-score rolling clippato."""
    mu  = s.rolling(window, min_periods=1).mean()
    sig = s.rolling(window, min_periods=1).std().clip(lower=1e-8)
    return _sanitize(((s - mu) / sig).clip(-clip, clip))


def _norm_tanh(s: pd.Series) -> pd.Series:
    """Normalizza in (-1, 1) via tanh(z-score)."""
    mu  = s.rolling(120, min_periods=10).mean()
    sig = s.rolling(120, min_periods=10).std().clip(lower=1e-8)
    return _sanitize(np.tanh((s - mu) / sig))


# ══════════════════════════════════════════════════════════════════════════════
# 1. COMPRESSIBILITY FACTOR  Z = PV / nRT
# ══════════════════════════════════════════════════════════════════════════════

def compute_zmarket(
    pressure:    pd.Series,
    temperature: pd.Series,
    log_volume:  pd.Series,
    n:           float = 1.0,
    R:           float = 1.0,
    window:      int   = 20,
) -> pd.Series:
    """
    Fattore di compressibilità Z = PV / nRT.

    Per un gas ideale Z = 1 per definizione.
    La deviazione misura le forze intermolecolari nel mercato:
      Z > 1  →  le "molecole" (order flow) si repellono → trend forte
      Z < 1  →  le "molecole" si attraggono → herding / mean-reversion

    Usa le medie rolling per ridurre il rumore intraday.
    """
    p_smooth = _sanitize(pressure.rolling(window, min_periods=1).mean())
    t_smooth = _sanitize(temperature.rolling(window, min_periods=1).mean())
    v_smooth = _sanitize(log_volume.rolling(window, min_periods=1).mean())

    denominator = (n * R * t_smooth).clip(lower=1e-8)
    z_raw       = (p_smooth * v_smooth) / denominator

    return _sanitize(_norm_tanh(z_raw), fill=0.0)


# ══════════════════════════════════════════════════════════════════════════════
# 2. CARNOT EFFICIENCY  η = 1 - T_cold / T_hot
# ══════════════════════════════════════════════════════════════════════════════

def compute_carnot_efficiency(
    temperature: pd.Series,
    window:      int = 60,
) -> pd.Series:
    """
    Efficienza di Carnot η = 1 - T_cold / T_hot.

    T_hot  = percentile 90 della temperatura nelle ultime `window` barre.
    T_cold = percentile 10 della temperatura nelle ultime `window` barre.

    Interpretazione nel mercato:
      η elevata  → grande gradiente termico, il "motore" ha molta energia disponibile.
                   Se W sale, il rally ha ancora carburante.
      η → 0      → T_hot ≈ T_cold, ciclo chiuso, il motore è esausto.
                   Tipico di periodi laterali o prima di inversione.

    Il valore è già in [0, 1] per costruzione matematica.
    """
    t_hot  = temperature.rolling(window, min_periods=window // 2) \
                        .quantile(0.90).clip(lower=1e-8)
    t_cold = temperature.rolling(window, min_periods=window // 2) \
                        .quantile(0.10).clip(lower=1e-8)

    eta = 1.0 - (t_cold / t_hot)
    return _sanitize(eta.clip(0.0, 1.0), fill=0.5)


# ══════════════════════════════════════════════════════════════════════════════
# 3. ENTROPY PRODUCTION RATE  σ = dS/dt
# ══════════════════════════════════════════════════════════════════════════════

def compute_entropy_production(
    entropy:   pd.Series,
    sg_window: int = 11,
    sg_poly:   int = 3,
    norm_win:  int = 60,
) -> pd.Series:
    """
    Tasso di produzione di entropia σ = dS/dt.

    Usa il filtro di Savitzky-Golay per stimare la derivata prima dell'entropia
    in modo robusto al rumore intraday (preserva forma, non solo liscia).

    Interpretazione fisica:
      σ > 0  → l'entropia cresce → processo irreversibile → trend si sta
               degradando (i trade producono "calore" non lavoro utile).
      σ ≈ 0  → processo quasi-statico → trend sostenibile, efficiente.
      σ < 0  → l'entropia DIMINUISCE → auto-organizzazione → segnale raro
               che precede breakout forti (il sistema si "cristallizza").

    Nota: σ < 0 non viola il 2° principio: stiamo osservando un sottosistema
    aperto (il mercato riceve "energia" da outside, es. notizie, FOMC).
    """
    values = entropy.ffill().fillna(0.0).values

    if len(values) < sg_window:
        return pd.Series(0.0, index=entropy.index)

    # Forza sg_window dispari
    sg_w = sg_window if sg_window % 2 == 1 else sg_window + 1
    sg_w = max(sg_w, sg_poly + 2)

    try:
        # deriv=1 → prima derivata, delta=1 → passo unitario
        dS = savgol_filter(values, window_length=sg_w, polyorder=sg_poly, deriv=1)
    except Exception:
        dS = np.gradient(values)

    sigma_raw = pd.Series(dS, index=entropy.index)
    return _sanitize(_clip_zscore(sigma_raw, norm_win, clip=3.0), fill=0.0)


# ══════════════════════════════════════════════════════════════════════════════
# 4. MONETARY PRESSURE RESONANCE INDEX  (MPRI)
# ══════════════════════════════════════════════════════════════════════════════

def compute_mpri(
    pressure: pd.Series,
    rates:    Optional[pd.Series],
    max_lag:  int = 90,
    min_lag:  int = 20,
    window:   int = 60,
) -> pd.Series:
    """
    Monetary Pressure Resonance Index (MPRI).

    Va oltre il semplice "best_lag": misura QUANTO e CON QUALE SEGNO
    il mercato risponde alla politica monetaria.

    Algoritmo rolling:
      1. Per ogni finestra di `window` barre:
         a. Cerca il lag in [min_lag, max_lag] con cross-corr massima
            tra P_market e (1/r) → best_lag_local
         b. MPRI_amplitude = valore della cross-corr al best_lag_local
            (∈ [-1, 1])
         c. MPRI_phase = segno: +1 se P ↑ quando 1/r aumenta (tassi ↓)
                                 -1 se P ↑ quando 1/r diminuisce (tassi ↑)

    Interpretazione:
      MPRI > 0.5   → forte risonanza costruttiva: il mercato sta
                     "amplificando" il segnale monetario → segui la FED
      MPRI < -0.5  → risonanza distruttiva: il mercato resiste alla FED
                     → alta probabilità di correzione forzata
      MPRI ≈ 0     → disaccoppiamento → il mercato si muove per motivi propri

    Se rates non disponibile (intraday), restituisce serie di zeri.
    """
    if rates is None or rates.isna().all():
        return pd.Series(0.0, index=pressure.index)

    inv_rates = _sanitize(1.0 / (rates.ffill().bfill().clip(lower=1e-4) + 1e-5))
    p         = _sanitize(pressure)

    mpri_vals = np.zeros(len(p))

    step = max(window // 4, 1)
    for end in range(window, len(p), step):
        start    = max(0, end - window)
        p_win    = p.iloc[start:end].values
        r_win    = inv_rates.iloc[start:end].values

        best_corr = 0.0
        for lag in range(min_lag, min(max_lag, len(r_win) - 10) + 1, 3):
            if lag >= len(r_win):
                break
            aligned = r_win[:-lag] if lag > 0 else r_win
            p_slice = p_win[lag:]
            if len(aligned) < 10 or len(p_slice) < 10:
                continue
            min_len = min(len(aligned), len(p_slice))
            with np.errstate(divide="ignore", invalid="ignore"):
                c = np.corrcoef(p_slice[:min_len], aligned[:min_len])[0, 1]
            if np.isfinite(c) and abs(c) > abs(best_corr):
                best_corr = c

        # Riempi il blocco con valore costante (l'aggiornamento è ogni `step`)
        fill_start = max(0, end - step)
        mpri_vals[fill_start:end] = best_corr

    mpri = pd.Series(mpri_vals, index=p.index)
    return _sanitize(mpri.clip(-1.0, 1.0), fill=0.0)


# ══════════════════════════════════════════════════════════════════════════════
# 5. HELMHOLTZ FREE ENERGY  F = U - T·S
# ══════════════════════════════════════════════════════════════════════════════

def compute_helmholtz(
    work_cum:    pd.Series,
    temperature: pd.Series,
    entropy:     pd.Series,
    window:      int = 20,
) -> pd.Series:
    """
    Energia libera di Helmholtz F = U - T·S.

    U (energia interna)  = |lavoro cumulativo rolling|  → energia immagazzinata
    T (temperatura)      = temperatura termodinamica del mercato
    S (entropia)         = entropia di Shannon/Lévy normalizzata

    ΔF tra due istanti:
      ΔF < 0  → il sistema "vuole" cambiare stato → trade spontaneo
                 (BUY in fase espansiva, SELL in fase compressa)
      ΔF > 0  → il cambiamento richiede energia esterna → trade contro corrente
                 (tipico di breakout forzati destinati a fallire)

    La derivata ΔF/Δt (normalizzata) è il segnale che usiamo:
      negativa e in calo  → processo spontaneo accelera → segnale forte
      positiva e in salita → energia libera cresce → stress in aumento
    """
    # Energia interna: usa rolling abs del lavoro per evitare numeri enormi
    U = work_cum.abs().rolling(window, min_periods=1).mean()

    t_smooth = temperature.rolling(window, min_periods=1).mean()
    s_smooth = entropy.rolling(window, min_periods=1).mean()

    F     = _sanitize(U - t_smooth * s_smooth)
    delta_F = _sanitize(F.diff())

    return _sanitize(_clip_zscore(delta_F, window * 3, clip=3.0), fill=0.0)


# ══════════════════════════════════════════════════════════════════════════════
# 6. THERMODYNAMIC QUALITY  Q = dW / (T · dS + ε)
# ══════════════════════════════════════════════════════════════════════════════

def compute_thermo_quality(
    work_cum:    pd.Series,
    temperature: pd.Series,
    entropy:     pd.Series,
    window:      int = 10,
) -> pd.Series:
    """
    Qualità termodinamica del mercato: rapporto Lavoro / Calore dissipato.

    In un motore reale: η = W_utile / Q_tot.
    Qui definiamo:
      - dW  = variazione del lavoro cumulativo (lavoro "utile")
      - dQ  = T · dS  (calore dissipato come disordine)

    Q_ratio = |dW| / (T · |dS| + ε)

    > 1  → la maggior parte dell'energia va in movimento di prezzo (efficiente)
    < 1  → la maggior parte dell'energia si disperde come entropia (rumore)
    ~ 1  → equilibrio tra ordine e disordine

    Normalizzato come z-score per compatibilità con l'osservazione DDPG.
    """
    dW = _sanitize(work_cum.diff().abs())
    dS = _sanitize(entropy.diff().abs())
    T  = _sanitize(temperature.rolling(window, min_periods=1).mean()).clip(lower=1e-8)

    q_raw  = dW / (T * dS + 1e-8)
    q_clip = q_raw.clip(upper=q_raw.quantile(0.99))  # taglia outlier estremi

    return _sanitize(_clip_zscore(q_clip, window * 6, clip=3.0), fill=0.0)


# ══════════════════════════════════════════════════════════════════════════════
# ENGINE PRINCIPALE
# ══════════════════════════════════════════════════════════════════════════════

class ThermoStatisticsEngine:
    """
    Calcola tutte le statistiche termodinamiche avanzate in un'unica chiamata.

    Parametri
    ──────────
    is_intraday : bool
        Se True, usa finestre brevi adatte alle barre da 2 minuti.
        Se False, finestre daily (20-90 bar).
    n_particles : float
        Numero di "particelle" (ticker nel portafoglio) per il fattore Z.
    R : float
        Costante dei gas (default 1.0 per normalizzazione).

    Utilizzo
    ─────────
        engine = ThermoStatisticsEngine(is_intraday=True)
        stats_df = engine.compute(
            pressure=thermo_df["Thm_Pressure"],
            temperature=thermo_df["Thm_Temperature"],
            entropy=thermo_df["Thm_Entropy"],
            work_cum=thermo_df["Thm_Work"],
            log_volume=log_vol_series,
            rates=rates_series,   # None per intraday
        )
    """

    def __init__(
        self,
        is_intraday:  bool  = False,
        n_particles:  float = 1.0,
        R:            float = 1.0,
    ):
        self.is_intraday = is_intraday
        self.n           = n_particles
        self.R           = R
        freq             = "intraday" if is_intraday else "daily"
        self._w          = _DEFAULT_WINDOWS[freq]

    def compute(
        self,
        pressure:    pd.Series,
        temperature: pd.Series,
        entropy:     pd.Series,
        work_cum:    pd.Series,
        log_volume:  pd.Series,
        rates:       Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Calcola tutte le 6 statistiche e restituisce un DataFrame con colonne STAT_COLS.

        Tutti i valori sono normalizzati e privi di NaN.
        """
        w = self._w

        result = pd.DataFrame(index=pressure.index)

        # 1. Compressibility Factor
        result["Thm_Zmarket"] = compute_zmarket(
            pressure, temperature, log_volume,
            n=self.n, R=self.R, window=w["short"],
        )

        # 2. Carnot Efficiency
        result["Thm_CarnotEff"] = compute_carnot_efficiency(
            temperature, window=w["long"],
        )

        # 3. Entropy Production Rate
        result["Thm_EntropyProd"] = compute_entropy_production(
            entropy,
            sg_window = w["sg_window"],
            sg_poly   = w["sg_poly"],
            norm_win  = w["long"],
        )

        # 4. MPRI (solo daily: rates disponibili; intraday: zeri)
        if not self.is_intraday and rates is not None:
            result["Thm_MPRI"] = compute_mpri(
                pressure, rates,
                max_lag = w["lag"],
                min_lag = w["short"],
                window  = w["long"],
            )
        else:
            result["Thm_MPRI"] = 0.0

        # 5. Helmholtz Free Energy
        result["Thm_Helmholtz"] = compute_helmholtz(
            work_cum, temperature, entropy, window=w["short"],
        )

        # 6. Thermodynamic Quality
        result["Thm_Quality"] = compute_thermo_quality(
            work_cum, temperature, entropy, window=w["short"],
        )

        result = result.ffill().bfill().fillna(0.0)
        return result


# ══════════════════════════════════════════════════════════════════════════════
# HELPER PER REWARD SHAPING (usato in TradingEnv)
# ══════════════════════════════════════════════════════════════════════════════

def get_thermo_reward_bonus(
    thermo_row: "pd.Series",
    action_type: str,
    scale: float = 0.03,
) -> float:
    """
    Bonus/penalty di reward basato sulle statistiche termodinamiche.

    Logica:
      BUY:
        + bonus se CarnotEff alta (energia disponibile)
        + bonus se Helmholtz negativo (processo spontaneo)
        + bonus se Quality > 0 (conversione efficiente)
        - penalty se EntropyProd alta (sistema si degrada)
        - penalty se Zmarket < -0.5 (forze attrattive, mean-revert)

      SELL:
        + bonus se EntropyProd alta (sistema brucia energia → vendi)
        + bonus se CarnotEff bassa (motore esausto)
        + bonus se MPRI < -0.5 (pressione va contro FED)
        - penalty se Helmholtz molto negativo (processo spontaneo → non vendere)

    Returns: float ∈ [-scale, +scale]
    """
    carnot   = float(thermo_row.get("Thm_CarnotEff", 0.5))
    entropy_p = float(thermo_row.get("Thm_EntropyProd", 0.0))
    helmholtz = float(thermo_row.get("Thm_Helmholtz", 0.0))
    quality   = float(thermo_row.get("Thm_Quality", 0.0))
    zmarket   = float(thermo_row.get("Thm_Zmarket", 0.0))
    mpri      = float(thermo_row.get("Thm_MPRI", 0.0))

    a = action_type.lower()
    bonus = 0.0

    if a == "buy":
        bonus += 0.3 * max(carnot - 0.4, 0.0)          # motore ha energia
        bonus -= 0.4 * max(entropy_p, 0.0)              # processo si degrada → non comprare
        bonus += 0.2 * max(-helmholtz, 0.0)             # ΔF < 0 → spontaneo
        bonus += 0.1 * max(quality, 0.0)                # conversione efficiente
        bonus -= 0.2 * max(-(zmarket + 0.5), 0.0)      # Z basso → mean-revert

    elif a == "sell":
        bonus += 0.4 * max(entropy_p, 0.0)              # sistema brucia energia → esci
        bonus += 0.2 * max(0.5 - carnot, 0.0)           # motore esausto → esci
        bonus += 0.2 * max(-mpri - 0.5, 0.0)            # pressione va contro FED → esci
        bonus -= 0.2 * max(-helmholtz - 0.5, 0.0)       # molto spontaneo → tieni

    # Scala e clippa
    return float(np.clip(bonus * scale, -scale, scale))