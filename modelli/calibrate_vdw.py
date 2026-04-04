"""
modelli/calibrate_vdw.py — Calibrazione dei parametri a e b di Van der Waals
                           secondo Gabaix, Gopikrishnan, Plerou & Stanley (2003)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FONDAMENTI TEORICI
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

L'equazione di Van der Waals applicata al mercato:
    (P + a·n²/V²)(V - nb) = nRT

dove V = log(volume), T = entropia dei prezzi, n = numero di asset.

Il paper Gabaix et al. (2003) stabilisce tre leggi empiriche universali:

  1. Legge cubica dei rendimenti:    P(|r| > x) ~ x^(-ζr),  ζr ≈ 3
  2. Legge semi-cubica dei volumi:   P(Q > x)   ~ x^(-ζQ),  ζQ ≈ 3/2
  3. Square-root price impact:       ΔP ~ V^γ,              γ ≈ 1/2

MAPPING verso i parametri VdW
─────────────────────────────
  Parametro a (attrazione / herding):
    Il termine -a·n²/V² riduce la pressione interna, esattamente come
    l'herding "ammortizza" la volatilità abbassando la pressione effettiva.
    
    Dal paper: l'impatto di prezzo scala come ΔP/P ~ λ·(Q/Q̄)^(1/2)
    Il coefficiente λ è il "peso" dell'attrazione.
    
    Calibrazione:
      a = λ_impact × Q̄² × momentum_strength
    
    dove:
      λ_impact        = ζr / (ζr - 2)   [da teoria GG2003, Teorema 3]
                      = 3 / (3 - 2) = 3  (costante universale)
      momentum_strength = max(0, autocorr(returns, lag=1))
                          misura l'intensità dello herding empirico
      Q̄²              = exp(2 × mean(log_volume))
                          scala il parametro alle dimensioni del mercato

  Parametro b (volume escluso / floor di liquidità):
    Il termine -nb rappresenta il volume minimo strutturale che il mercato
    non può comprimere — l'equivalente del raggio molecolare.
    
    Dal paper: la distribuzione dei volumi segue P(Q > x) ~ x^(-3/2).
    Il "volume escluso" per particella corrisponde al volume minimo
    osservabile, ovvero il percentile basso della distribuzione empirica.
    
    Calibrazione:
      b = V_floor / n
    
    dove:
      V_floor = exp(percentile(log_volume, pct_floor))
                percentile conservativo (default 5°) della distrib. dei volumi
      n       = numero di asset (molecole del sistema)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STIMA DEGLI ESPONENTI (Hill's estimator)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Gabaix et al. usano lo stimatore di Hill per ζ:

    ζ̂_Hill = (k - 1) / Σ_{i=1}^{k-1} [ln x_(i) - ln x_(k)]

dove x_(1) ≥ ... ≥ x_(k) sono le k osservazioni più grandi.
k ottimale: ~ n^(2/3) (regola empirica del paper, sezione 2.1).

Usiamo questo stimatore per verificare che i dati in input siano coerenti
con le leggi universali del paper prima di procedere alla calibrazione.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


# ── Costanti universali Gabaix et al. 2003 ────────────────────────────────────

ZETA_R  = 3.0    # esponente legge cubica rendimenti  (Eq. 1 del paper)
ZETA_Q  = 1.5    # esponente legge semi-cubica volumi  (Eq. 2-3 del paper)
ZETA_N  = 3.3    # esponente numero di transazioni     (Eq. 4 del paper)
GAMMA   = 0.5    # esponente square-root price impact  (Eq. 5 del paper)
LAMBDA_UNIVERSAL = ZETA_R / (ZETA_R - 2)   # = 3.0  (Teorema 3, GG2003)


# ── Output della calibrazione ─────────────────────────────────────────────────

@dataclass
class VdWParams:
    """
    Parametri Van der Waals calibrati per un singolo ticker.

    Attributi
    ---------
    a : float
        Parametro di attrazione (herding/momentum). Riduce la pressione
        interna quando il mercato è in fase di clustering direzionale.
    b : float
        Volume escluso per particella (floor di liquidità). Rappresenta
        il minimo strutturale di volume che il sistema non può comprimere.
    zeta_r : float
        Esponente stimato della legge di potenza dei rendimenti.
        Valore atteso dal paper: ~3.0
    zeta_q : float
        Esponente stimato della legge di potenza dei volumi.
        Valore atteso dal paper: ~1.5
    lambda_impact : float
        Coefficiente di impatto di prezzo stimato empiricamente.
        Valore teorico Gabaix: ZETA_R / (ZETA_R - 2) = 3.0
    momentum_strength : float
        Autocorrelazione dei rendimenti al lag 1. Misura l'intensità
        dello herding. Valore 0 = mercato random walk.
    n_obs : int
        Numero di osservazioni usate per la calibrazione.
    warnings : list[str]
        Avvertenze generate durante la calibrazione (es. esponenti
        anomali rispetto ai valori universali di Gabaix).
    """
    a:                  float
    b:                  float
    zeta_r:             float
    zeta_q:             float
    lambda_impact:      float
    momentum_strength:  float
    n_obs:              int
    warnings:           list = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"  a (herding)        = {self.a:.6f}",
            f"  b (liquidity floor)= {self.b:.6f}",
            f"  ζr (ritorno)       = {self.zeta_r:.3f}  [atteso ~{ZETA_R}]",
            f"  ζQ (volume)        = {self.zeta_q:.3f}  [atteso ~{ZETA_Q}]",
            f"  λ  (price impact)  = {self.lambda_impact:.3f}  [teorico ~{LAMBDA_UNIVERSAL:.1f}]",
            f"  momentum (AC lag1) = {self.momentum_strength:.4f}",
            f"  n osservazioni     = {self.n_obs}",
        ]
        if self.warnings:
            lines.append("  ⚠ AVVERTENZE:")
            for w in self.warnings:
                lines.append(f"    - {w}")
        return "\n".join(lines)


# ── Hill's estimator ──────────────────────────────────────────────────────────

def hill_estimator(x: np.ndarray, k: Optional[int] = None) -> float:
    """
    Stima l'esponente di coda con lo stimatore di Hill (Gabaix 2003, sez. 2.1).

    Parametri
    ---------
    x : array di valori positivi (rendimenti assoluti o volumi)
    k : numero di osservazioni di coda da usare.
        Se None, usa la regola empirica k = n^(2/3) del paper.

    Restituisce
    -----------
    float : stima di ζ (esponente di coda, positivo)
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[x > 0]
    n = len(x)

    if n < 10:
        return np.nan

    if k is None:
        k = max(10, int(n ** (2 / 3)))
    k = min(k, n - 1)

    # Ordina decrescente, prendi le k più grandi
    x_sorted = np.sort(x)[::-1]
    top_k    = x_sorted[:k]

    # ζ̂_Hill = (k-1) / Σ [ln(x_(i)) - ln(x_(k))]
    denom = np.sum(np.log(top_k[:-1]) - np.log(top_k[-1]))

    if denom <= 0:
        return np.nan

    return (k - 1) / denom


# ── Price impact coefficient (λ) ──────────────────────────────────────────────

def estimate_lambda_impact(
    returns:     np.ndarray,
    log_volumes: np.ndarray,
    gamma:       float = GAMMA,
) -> float:
    """
    Stima empirica del coefficiente λ nell'equazione di impatto:
        |r_t| = λ × (Q_t / Q̄)^γ + ε

    Dal paper (Eq. 5): ΔP ~ V^γ con γ ≈ 0.5.
    Qui stimiamo λ con una regressione log-log.

    Parametri
    ---------
    returns     : rendimenti assoluti (|r_t|)
    log_volumes : log dei volumi (già calcolato)
    gamma       : esponente di impatto (default 0.5)

    Restituisce
    -----------
    float : stima di λ (clamped in [0.5, 10.0] per robustezza)
    """
    abs_ret = np.abs(returns)
    mask    = (abs_ret > 1e-8) & np.isfinite(log_volumes)

    if mask.sum() < 20:
        return LAMBDA_UNIVERSAL

    log_r = np.log(abs_ret[mask])
    log_q = log_volumes[mask] - log_volumes[mask].mean()   # normalizza per Q̄

    # Regressione: log|r| = log(λ) + γ × log(Q/Q̄)
    # → stima di log(λ) = mean(log|r|) - γ × mean(log(Q/Q̄))
    log_lambda = np.mean(log_r) - gamma * np.mean(log_q)
    lambda_est = float(np.exp(log_lambda)) if not np.isnan(log_lambda) else LAMBDA_UNIVERSAL

    return float(np.clip(lambda_est, 0.5, 10.0))


# ── Calibrazione per singolo ticker ──────────────────────────────────────────

def calibrate_single(
    prices:       pd.Series,
    volumes:      pd.Series,
    n_assets:     int   = 1,
    pct_floor:    float = 5.0,
    k_hill:       Optional[int] = None,
    zeta_r_tol:   float = 0.5,
    zeta_q_tol:   float = 0.4,
) -> VdWParams:
    """
    Calibra a e b per un singolo ticker dal paper Gabaix et al. 2003.

    Parametri
    ---------
    prices      : serie temporale dei prezzi (non normalizzati)
    volumes     : serie temporale dei volumi raw (non logaritmici)
    n_assets    : numero di asset nel portafoglio (= n nella VdW)
    pct_floor   : percentile per il calcolo del volume floor per b (default 5°)
    k_hill      : numero osservazioni di coda per Hill (None = auto n^(2/3))
    zeta_r_tol  : tolleranza per avvertimento su ζr (|ζr_est - 3.0| > tol)
    zeta_q_tol  : tolleranza per avvertimento su ζQ (|ζQ_est - 1.5| > tol)

    Restituisce
    -----------
    VdWParams con a, b calibrati e diagnostiche
    """
    warns = []

    # ── Preparazione serie ────────────────────────────────────────────────────
    prices  = prices.dropna()
    volumes = volumes.dropna()
    idx     = prices.index.intersection(volumes.index)
    prices  = prices.loc[idx]
    volumes = volumes.loc[idx]

    if len(prices) < 50:
        raise ValueError(f"Dati insufficienti per calibrazione: {len(prices)} osservazioni")

    # ── Rendimenti logaritmici ────────────────────────────────────────────────
    returns     = np.log(prices / prices.shift(1)).dropna().values
    log_volumes = np.log(volumes.values + 1e-8)   # +ε per volumi zero

    # ── 1. Stima ζr con Hill's estimator ──────────────────────────────────────
    zeta_r_est = hill_estimator(np.abs(returns), k=k_hill)

    if np.isnan(zeta_r_est):
        zeta_r_est = ZETA_R
        warns.append("Hill's estimator per ζr ha fallito — uso valore teorico 3.0")
    elif abs(zeta_r_est - ZETA_R) > zeta_r_tol:
        warns.append(
            f"ζr stimato ({zeta_r_est:.2f}) devia dal valore universale {ZETA_R} "
            f"di oltre {zeta_r_tol}. Mercato con tail behavior anomalo."
        )

    # ── 2. Stima ζQ con Hill's estimator ─────────────────────────────────────
    # Normalizza i volumi per Q̄ (come nel paper, sez. 2.2)
    vol_normalized = volumes.values / (volumes.mean() + 1e-8)
    zeta_q_est     = hill_estimator(vol_normalized, k=k_hill)

    if np.isnan(zeta_q_est):
        zeta_q_est = ZETA_Q
        warns.append("Hill's estimator per ζQ ha fallito — uso valore teorico 1.5")
    elif abs(zeta_q_est - ZETA_Q) > zeta_q_tol:
        warns.append(
            f"ζQ stimato ({zeta_q_est:.2f}) devia dal valore universale {ZETA_Q} "
            f"di oltre {zeta_q_tol}. Verifica la liquidità del ticker."
        )

    # ── 3. Stima λ (price impact coefficient) ────────────────────────────────
    # Allinea returns e log_volumes (returns ha un punto in meno)
    log_vol_aligned = log_volumes[1:]
    lambda_impact   = estimate_lambda_impact(returns, log_vol_aligned)

    # ── 4. Momentum strength (autocorrelazione lag-1) ─────────────────────────
    if len(returns) > 10:
        std1 = np.std(returns[:-1])
        std2 = np.std(returns[1:])
        if std1 > 1e-8 and std2 > 1e-8:
            ac = np.corrcoef(returns[:-1], returns[1:])[0, 1]
            momentum_strength = float(max(0.0, np.nan_to_num(ac)))
        else:
            momentum_strength = 0.0
    else:
        momentum_strength = 0.0
        warns.append("Serie troppo corta per stimare l'autocorrelazione")

    # ── 5. Calibrazione parametro a ──────────────────────────────────────────
    #
    # Dalla teoria Gabaix (Teorema 3):
    #   a = λ_impact × Q̄² × momentum_strength
    #
    # Il termine Q̄² scala l'attrazione alla dimensione del mercato (come nella
    # VdW dove a/V² ha unità di pressione, qui lavoriamo in log-space quindi
    # usiamo exp(2·μ_log_vol) come proxy di Q̄²).
    #
    # Se momentum_strength ≈ 0 (random walk puro), a → 0 e la VdW degenera
    # nell'equazione dei gas ideali: PV = nRT.
    #
    q_bar_squared = np.exp(2 * log_volumes.mean())
    a = lambda_impact * q_bar_squared * momentum_strength

    # Guardrail: a non può essere negativo (fisicamente insensato)
    a = max(0.0, a)

    # ── 6. Calibrazione parametro b ──────────────────────────────────────────
    #
    # Dal paper: P(Q > x) ~ x^(-3/2), il volume minimo strutturale è il
    # quantile basso della distribuzione (rappresenta il "raggio molecolare"
    # del singolo scambio). Dividiamo per n_assets per ottenere il contributo
    # per-particella.
    #
    v_floor = np.exp(np.percentile(log_volumes, pct_floor))
    b       = v_floor / max(n_assets, 1)

    # Guardrail: b deve essere < V_min/n per soddisfare la condizione
    # fisicamente: V > nb (il sistema deve rimanere "espandibile")
    v_min     = np.exp(np.min(log_volumes))
    b_max     = v_min / max(n_assets, 1) * 0.9   # margine di sicurezza 10%
    if b >= b_max:
        b = b_max * 0.5
        warns.append(
            f"b calibrato superava il limite fisico V_min/n — ridotto a {b:.4f}. "
            f"Considera di aumentare pct_floor o ridurre n_assets."
        )

    return VdWParams(
        a=float(a),
        b=float(b),
        zeta_r=float(zeta_r_est),
        zeta_q=float(zeta_q_est),
        lambda_impact=float(lambda_impact),
        momentum_strength=float(momentum_strength),
        n_obs=len(returns),
        warnings=warns,
    )


# ── Calibrazione per portafoglio (multi-ticker) ───────────────────────────────

def calibrate_portfolio(
    prices_df:  pd.DataFrame,
    volumes_df: pd.DataFrame,
    tickers:    list[str],
    pct_floor:  float = 5.0,
    k_hill:     Optional[int] = None,
) -> dict[str, VdWParams]:
    """
    Calibra a e b per ogni ticker nel portafoglio.

    Parametri
    ---------
    prices_df  : DataFrame con colonne = tickers, index = date
    volumes_df : DataFrame con colonne = tickers, index = date
    tickers    : lista dei ticker da calibrare
    pct_floor  : percentile per il volume floor (default 5°)
    k_hill     : k per Hill's estimator (None = auto)

    Restituisce
    -----------
    dict[ticker -> VdWParams]
    """
    n_assets = len(tickers)
    results  = {}

    print(f"\n[VdW Calibrazione] Gabaix et al. 2003 | {n_assets} ticker")
    print(f"  Costanti universali: ζr={ZETA_R}, ζQ={ZETA_Q}, γ={GAMMA}, λ={LAMBDA_UNIVERSAL:.1f}")
    print("─" * 60)

    for ticker in tickers:
        if ticker not in prices_df.columns or ticker not in volumes_df.columns:
            print(f"  [{ticker}] SALTATO — colonna mancante nei dati")
            continue

        try:
            params = calibrate_single(
                prices=prices_df[ticker],
                volumes=volumes_df[ticker],
                n_assets=n_assets,
                pct_floor=pct_floor,
                k_hill=k_hill,
            )
            results[ticker] = params
            print(f"  [{ticker}]")
            print(params.summary())
            print()

        except Exception as e:
            print(f"  [{ticker}] ERRORE: {e}")

    # ── Parametri aggregati per il sistema (n particelle) ─────────────────────
    if results:
        a_values = [p.a for p in results.values()]
        b_values = [p.b for p in results.values()]
        print("─" * 60)
        print(f"  Statistiche portafoglio:")
        print(f"    a medio  = {np.mean(a_values):.6f}  (std={np.std(a_values):.6f})")
        print(f"    b medio  = {np.mean(b_values):.6f}  (std={np.std(b_values):.6f})")
        print(f"    a totale = {np.sum(a_values):.6f}  (da usare nella VdW con n={n_assets})")
        print(f"    b totale = {np.sum(b_values):.6f}  (da usare nella VdW con n={n_assets})")

    return results


# ── Integrazione con il modello termodinamico ─────────────────────────────────

def vdw_pressure(
    log_volume:   float,
    temperature:  float,
    vdw_params:   VdWParams,
    n:            int,
    R:            float = 1.0,
) -> float:
    """
    Calcola la pressione di Van der Waals dati i parametri calibrati.

    (P + a·n²/V²)(V - nb) = nRT  →  P = nRT/(V - nb) - a·n²/V²

    Parametri
    ---------
    log_volume  : V = log(volume) del portafoglio al timestep corrente
    temperature : T = entropia/volatilità (es. Shannon entropy su finestra mobile)
    vdw_params  : parametri a, b calibrati da calibrate_single/portfolio
    n           : numero di asset (particelle)
    R           : costante dei gas (default 1.0, normalizzata)

    Restituisce
    -----------
    float : pressione P del sistema al timestep corrente
    """
    V  = log_volume
    a  = vdw_params.a
    b  = vdw_params.b
    nb = n * b

    if V <= nb + 1e-6:
        # Violazione del vincolo fisico: volume sotto il minimo strutturale
        # Restituisce pressione molto alta (segnale di stress estremo)
        return 1e6  # Cap to large finite value instead of infinity to prevent NaNs down the line

    # Equazione di Van der Waals risolta per P
    P = (n * R * temperature) / (V - nb) - a * (n ** 2) / (V ** 2 + 1e-8)

    # Sanity check sull'output
    if np.isnan(P) or np.isinf(P):
        return 0.0

    return float(P)


# ── Riepilogo in DataFrame ────────────────────────────────────────────────────

def params_to_dataframe(calibration: dict[str, VdWParams]) -> pd.DataFrame:
    """
    Converte il dizionario di VdWParams in un DataFrame leggibile,
    utile per logging, salvataggio CSV o visualizzazione.
    """
    rows = []
    for ticker, p in calibration.items():
        rows.append({
            "ticker":            ticker,
            "a":                 p.a,
            "b":                 p.b,
            "zeta_r":            p.zeta_r,
            "zeta_q":            p.zeta_q,
            "lambda_impact":     p.lambda_impact,
            "momentum_strength": p.momentum_strength,
            "n_obs":             p.n_obs,
            "has_warnings":      len(p.warnings) > 0,
        })
    return pd.DataFrame(rows).set_index("ticker")