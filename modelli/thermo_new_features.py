"""
modelli/thermo_new_features.py
════════════════════════════════════════════════════════════════════════════════

ThermoNewFeatures — Tre indicatori originali, NON presenti in letteratura,
sviluppati come estensione diretta del modello termodinamico Van der Waals.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. CSI — Compressive Stasis Index
   ─────────────────────────────────────────────────────────────────────
   Concetto dal report: "se la pressione aumenta ma il lavoro compiuto
   è minimo, siamo in presenza di una fase di stasi compressiva, tipica
   dei periodi che precedono grandi correzioni."

   Formula:
     CSI(t) = Z_pressure(t) / (|Z_work_delta(t)| + ε)

   Dove Z_x = z-score rolling di x sulla finestra w.

   Interpretazione:
     CSI > 2.0  → stasi compressiva → ATTENZIONE (sell region)
     CSI < 0.5  → lavoro sta convertendo energia → trend sano
     CSI ~ 1.0  → mercato in equilibrio

   Perché va nel DDPG (non nella CNN):
   → Non è un predittore di prezzo, ma un indicatore di QUANDO agire.
     Shape: reward shaping → penalty per BUY durante CSI alto.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

2. IIR — Information Injection Rate
   ─────────────────────────────────────────────────────────────────────
   Idea originale: nei sistemi termodinamici aperti (mercato = sistema aperto
   che riceve "energia" da notizie, FOMC, earnings...), la produzione di
   entropia può essere NEGATIVA localmente. Questo non viola il 2° principio
   ma segnala un input di informazione strutturata dall'esterno.

   Formula:
     IIR(t) = (freq. di dS/dt < 0 nelle ultime window barre) ∈ [0, 1]

   Interpretazione:
     IIR > 0.4  → il sistema sta ricevendo informazione strutturata
                  → alta probabilità di breakout direzionale forte
     IIR < 0.1  → il sistema è isolato, si muove per inerzia
                  → bassa volatilità attesa

   Non pubblicato: la letteratura econofisica (Mantegna & Stanley) misura
   sempre dS/dt > 0 come assunzione. Monitorare le violazioni locali come
   segnale di regime è un approccio originale.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

3. LDI — Lag Drift Index
   ─────────────────────────────────────────────────────────────────────
   Idea: il lag ottimale tra tassi e pressione non è costante (il tuo
   AdaptiveLagEstimator lo sa). Ma la VARIAZIONE del lag nel tempo è
   essa stessa un'informazione.

   Formula:
     optimal_lag(t) = argmax_{τ ∈ [20,120]} corr(P(t), 1/r(t-τ))
     LDI(t) = Δoptimal_lag(t) / window_lag  [normalizzato]

   Interpretazione:
     LDI > 0  → il lag sta AUMENTANDO → il mercato diventa meno reattivo
                 alla politica monetaria → regime di "disconnessione"
                 (spesso bull market tardo-ciclo)
     LDI < 0  → il lag sta DIMINUENDO → il mercato recepisce le notizie
                 monetarie più in fretta → regime di sensibilità acuta
                 (spesso pivot FED)
     LDI ~ 0  → regime stabile

   Non presente in letteratura: Friedman (1961) parla di "long and variable
   lags" come fenomeno aggregato. Misurare la derivata prima del lag ottimale
   come indicatore di regime è un contributo originale.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

INTEGRAZIONE NEL DDPG:
  1. Aggiungi le 3 feature al thermo_state_builder.py (vedi NEW_FEATURE_COLS)
  2. Usa get_csi_reward_shaping() in trading_env.py per il reward shaping
  3. Opzionale: usa LDI per adattare noise_sigma dell'OU process nel DDPG

"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


# ── Costanti pubbliche ─────────────────────────────────────────────────────

NEW_FEATURE_COLS = [
    "Thm_CSI",   # Compressive Stasis Index
    "Thm_IIR",   # Information Injection Rate
    "Thm_LDI",   # Lag Drift Index
]

# Soglie operative per DDPG reward shaping
CSI_SELL_THRESHOLD  = 1.8   # sopra → penalizza BUY
CSI_BUY_THRESHOLD   = 0.5   # sotto → favorisce BUY
IIR_BREAKOUT_LEVEL  = 0.35  # sopra → mercato in regime d'informazione
LDI_DISCONNECT_LEVEL = 0.15 # sopra → mercato disconnesso dai tassi


# ── Utility ───────────────────────────────────────────────────────────────

def _sanitize(s: pd.Series, fill: float = 0.0) -> pd.Series:
    return s.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(fill)

def _zscore_rolling(s: pd.Series, window: int, clip: float = 3.0) -> pd.Series:
    mu  = s.rolling(window, min_periods=max(window // 4, 3)).mean()
    sig = s.rolling(window, min_periods=max(window // 4, 3)).std().clip(lower=1e-8)
    return _sanitize(((s - mu) / sig).clip(-clip, clip))


# ══════════════════════════════════════════════════════════════════════════
# 1. COMPRESSIVE STASIS INDEX (CSI)
# ══════════════════════════════════════════════════════════════════════════

def compute_csi(
    pressure_raw: pd.Series,
    work_raw:     pd.Series,
    window:       int   = 20,
    clip:         float = 4.0,
) -> pd.Series:
    """
    Compressive Stasis Index = Z(Pressure) / (|Z(ΔWork)| + ε)

    Identifica periodi in cui la pressione è alta ma il mercato
    non compie lavoro utile → stasi compressiva → precede correzioni.

    Parameters
    ----------
    pressure_raw : Serie grezza della pressione Van der Waals
    work_raw     : Serie grezza del lavoro cumulativo
    window       : Finestra rolling (20 per intraday, 60 per daily)
    clip         : Limite superiore del CSI (evita spike)

    Returns
    -------
    pd.Series normalizzata con z-score ∈ [0, clip]
    """
    z_pressure  = _zscore_rolling(pressure_raw, window)
    delta_work  = work_raw.diff(window).fillna(0)
    z_work_abs  = _zscore_rolling(delta_work, window).abs()

    # Evita divisione per zero: se il lavoro è esattamente 0 → CSI = z_pressure
    csi_raw = z_pressure / (z_work_abs + 0.3)

    # Clip e normalizzazione finale: vogliamo che CSI > 0 significhi "stasi"
    # CSI negativo (pressione bassa) è meno interessante → clip a 0
    csi = csi_raw.clip(lower=0.0, upper=clip)

    return _sanitize(csi)


# ══════════════════════════════════════════════════════════════════════════
# 2. INFORMATION INJECTION RATE (IIR)
# ══════════════════════════════════════════════════════════════════════════

def compute_iir(
    entropy_rate: pd.Series,
    window:       int = 40,
) -> pd.Series:
    """
    Information Injection Rate = freq(dS/dt < 0) nelle ultime 'window' barre.

    Basato sul principio che nei sistemi termodinamici APERTI (il mercato
    riceve notizie, FOMC, earnings dall'esterno), la produzione locale di
    entropia può essere negativa. Questo NON viola il 2° principio per il
    sistema globale, ma segnala che dell'informazione strutturata sta
    entrando nel sottosistema.

    Valori alti → imminente breakout direzionale (qualunque direzione).
    Combinare con CSI per la direzione:
      IIR alto + CSI alto  → sell (informazione + stasi compressiva)
      IIR alto + CSI basso → buy (informazione + trend sano)

    Parameters
    ----------
    entropy_rate : dS/dt — usare Thm_EntropyProd dal ThermoStatisticsEngine
    window       : Finestra rolling (barre da considerare)

    Returns
    -------
    pd.Series in [0, 1] — proporzione di barre con entropia calante
    """
    # dS/dt < 0 → la produzione di entropia è negativa (auto-organizzazione)
    is_injecting = (entropy_rate < 0).astype(float)

    iir = is_injecting.rolling(window, min_periods=window // 4).mean()

    return _sanitize(iir, fill=0.0)


# ══════════════════════════════════════════════════════════════════════════
# 3. LAG DRIFT INDEX (LDI)
# ══════════════════════════════════════════════════════════════════════════

def compute_ldi(
    pressure:     pd.Series,
    rates:        Optional[pd.Series],
    min_lag:      int = 20,
    max_lag:      int = 120,
    scan_step:    int = 5,
    update_every: int = 30,
    smooth_window: int = 60,
) -> pd.Series:
    """
    Lag Drift Index = derivata normalizzata del lag ottimale nel tempo.

    Calcola rolling il lag migliore tra pressure e 1/rates, poi ne
    misura la variazione temporale (derivata prima normalizzata).

    Positivo → lag in AUMENTO → mercato si disconnette dai tassi
    Negativo → lag in DIMINUZIONE → mercato diventa ipersensibile ai tassi
    Zero     → regime stabile

    Parameters
    ----------
    pressure     : Serie pressione Van der Waals
    rates        : Serie tassi (GS10, FEDFUNDS, ecc.) — None → LDI = 0
    min_lag      : Lag minimo da scandire (barre)
    max_lag      : Lag massimo da scandire (barre)
    scan_step    : Risoluzione dello scan (ogni N barre)
    update_every : Ricalcola il lag ottimale ogni N barre (performance)
    smooth_window: Finestra di smoothing del lag stimato

    Returns
    -------
    pd.Series normalizzata tramite z-score ∈ [-3, 3]
    """
    if rates is None or rates.isna().all():
        return pd.Series(0.0, index=pressure.index)

    inv_rates = _sanitize(1.0 / (rates.ffill().bfill().clip(lower=1e-4) + 1e-5))
    n = len(pressure)

    # Array dei lag ottimali nel tempo (calcolato ogni update_every barre)
    lag_series = np.full(n, float((min_lag + max_lag) // 2))

    lags_to_try = list(range(min_lag, max_lag + 1, scan_step))

    for i in range(max_lag + update_every, n, update_every):
        # Finestra di dati disponibili per il calcolo
        win_start = max(0, i - smooth_window * 3)
        p_win = pressure.iloc[win_start:i].values
        r_win = inv_rates.iloc[win_start:i].values

        best_corr = 0.0
        best_lag  = int((min_lag + max_lag) // 2)

        for lag in lags_to_try:
            if lag >= len(r_win):
                break
            r_lagged = r_win[:-lag] if lag > 0 else r_win
            p_slice  = p_win[lag:]
            n_pts    = min(len(r_lagged), len(p_slice))
            if n_pts < 10:
                continue
            with np.errstate(divide='ignore', invalid='ignore'):
                c = np.corrcoef(p_slice[:n_pts], r_lagged[:n_pts])[0, 1]
            if np.isfinite(c) and abs(c) > abs(best_corr):
                best_corr = c
                best_lag  = lag

        fill_end   = min(i + update_every, n)
        lag_series[i:fill_end] = best_lag

    lag_s = pd.Series(lag_series, index=pressure.index)

    # Smooth (il lag ottimale varia lentamente)
    lag_smooth = lag_s.rolling(smooth_window, min_periods=5).mean()

    # Derivata prima del lag → LDI (normalizzata come z-score)
    delta_lag = lag_smooth.diff(update_every).fillna(0)
    ldi = _zscore_rolling(delta_lag, smooth_window, clip=3.0)

    return _sanitize(ldi)


# ══════════════════════════════════════════════════════════════════════════
# ENGINE PRINCIPALE
# ══════════════════════════════════════════════════════════════════════════

def compute_new_thermo_features(
    pressure_raw:  pd.Series,
    work_raw:      pd.Series,
    entropy_rate:  pd.Series,
    rates:         Optional[pd.Series] = None,
    is_intraday:   bool = False,
) -> pd.DataFrame:
    """
    Calcola CSI, IIR, LDI e restituisce un DataFrame con colonne NEW_FEATURE_COLS.

    Parameters
    ----------
    pressure_raw  : Pressione grezza Van der Waals (pre-normalizzazione)
    work_raw      : Lavoro cumulativo grezzo (pre-normalizzazione)
    entropy_rate  : Tasso di produzione di entropia (Thm_EntropyProd)
    rates         : Serie tassi per LDI (opzionale)
    is_intraday   : Se True, usa finestre brevi

    Returns
    -------
    DataFrame con 3 colonne: Thm_CSI, Thm_IIR, Thm_LDI
    """
    if is_intraday:
        w_csi, w_iir = 10, 20
    else:
        w_csi, w_iir = 20, 40

    result = pd.DataFrame(index=pressure_raw.index)

    result["Thm_CSI"] = compute_csi(pressure_raw, work_raw, window=w_csi)
    result["Thm_IIR"] = compute_iir(entropy_rate, window=w_iir)

    if rates is not None and not is_intraday:
        result["Thm_LDI"] = compute_ldi(
            pressure=pressure_raw,
            rates=rates,
            min_lag=20,
            max_lag=120,
            scan_step=5,
            update_every=20,
            smooth_window=60,
        )
    else:
        result["Thm_LDI"] = 0.0

    return result.ffill().bfill().fillna(0.0)


# ══════════════════════════════════════════════════════════════════════════
# REWARD SHAPING PER DDPG
# ══════════════════════════════════════════════════════════════════════════

def get_new_thermo_reward(
    thermo_row:  "pd.Series",
    action_type: str,          # "buy" | "sell" | "hold"
    scale:       float = 0.04,
) -> float:
    """
    Bonus/penalty di reward basato su CSI, IIR, LDI.

    Logica integrata (da usare in TradingEnv._compute_reward):

      BUY:
        - penalty se CSI > CSI_SELL_THRESHOLD  (stasi compressiva → non comprare)
        - bonus   se CSI < CSI_BUY_THRESHOLD   (trend sano → compra)
        - bonus   se IIR > IIR_BREAKOUT_LEVEL e CSI < 1.0  (info + sano)
        - penalty se LDI > LDI_DISCONNECT_LEVEL  (mercato disconnesso)

      SELL:
        - bonus   se CSI > CSI_SELL_THRESHOLD  (stasi compressiva → vendi)
        - bonus   se IIR alto + CSI alto        (info in arrivo, sistema sotto stress)
        - penalty se LDI < -LDI_DISCONNECT_LEVEL (mercato iperattivo, sell prematuro)

      HOLD:
        - bonus   se IIR alto (breakout imminente → NON stare fermi)  [negativo]

    Returns: float ∈ [-scale, +scale]
    """
    csi = float(thermo_row.get("Thm_CSI", 1.0))
    iir = float(thermo_row.get("Thm_IIR", 0.0))
    ldi = float(thermo_row.get("Thm_LDI", 0.0))

    a = action_type.lower()
    bonus = 0.0

    if a == "buy":
        # Penalità forte per stasi compressiva
        if csi > CSI_SELL_THRESHOLD:
            bonus -= 0.5 * (csi - CSI_SELL_THRESHOLD) / CSI_SELL_THRESHOLD
        # Bonus per trend sano
        if csi < CSI_BUY_THRESHOLD:
            bonus += 0.3 * (CSI_BUY_THRESHOLD - csi) / CSI_BUY_THRESHOLD
        # Bonus per info injection in mercato sano
        if iir > IIR_BREAKOUT_LEVEL and csi < 1.0:
            bonus += 0.2 * (iir - IIR_BREAKOUT_LEVEL)
        # Penalità per disconnessione monetaria
        if ldi > LDI_DISCONNECT_LEVEL:
            bonus -= 0.1 * (ldi - LDI_DISCONNECT_LEVEL)

    elif a == "sell":
        # Bonus per stasi compressiva (vendi prima del crash)
        if csi > CSI_SELL_THRESHOLD:
            bonus += 0.5 * min((csi - CSI_SELL_THRESHOLD) / 2.0, 1.0)
        # Bonus per info + stress (tempesta in arrivo)
        if iir > IIR_BREAKOUT_LEVEL and csi > 1.0:
            bonus += 0.3 * iir
        # Penalità per sell prematuro in mercato iperreattivo
        if ldi < -LDI_DISCONNECT_LEVEL:
            bonus -= 0.1

    elif a == "hold":
        # Penalità per inazione durante information injection
        if iir > IIR_BREAKOUT_LEVEL:
            bonus -= 0.1 * iir  # Leggera pressione ad agire

    return float(np.clip(bonus * scale, -scale, scale))