"""
modelli/signal_trust.py
════════════════════════════════════════════════════════════════════════════════

SignalTrustEngine — Calibrazione dinamica della fiducia negli indicatori

PROBLEMA
─────────
Ogni indicatore termodinamico (Thm_Stress, Thm_Regime, TIXI...) fa una
predizione implicita sul mercato. Ma la qualità di quella predizione cambia
nel tempo: un indicatore può essere affidabile in regime di trend e
completamente inutile in regime laterale.

Senza questa informazione, il DDPG tratta tutti i segnali come ugualmente
credibili, accumulando errori sistematici nei periodi in cui gli indicatori
sono "spenti".

SOLUZIONE
──────────
Per ogni indicatore Thm_k misuriamo rolling:
  - IC (Information Coefficient): correlazione tra il segnale e il
    rendimento h barre dopo → misura la qualità direzionale
  - HR (Hit Rate): con quale frequenza il segno del segnale corrisponde
    al segno del rendimento successivo → interpretabile come "precisione"

Combiniamo IC e HR in un Trust Score ∈ (0, 1):
  - 0.5 = segnale casuale (nessuna fiducia)
  - > 0.5 = segnale affidabile nella direzione attuale
  - < 0.5 = segnale contrarian (affidabile ma invertito)

Il DDPG riceve sia il valore dell'indicatore che la sua affidabilità attuale,
permettendo alla rete di imparare a pesare i segnali in modo adattivo.

COLONNE OUTPUT (prefisso "Trust_")
────────────────────────────────────
  Trust_{col}_IC      — Information coefficient rolling (−1, 1)
  Trust_{col}_HR      — Hit rate rolling (0, 1)
  Trust_{col}         — Trust score aggregato (0, 1)
  Trust_Global        — Fiducia media pesata su tutti gli indicatori (0, 1)
  Trust_Dispersion    — Deviazione standard delle Trust score (0, 1)
                        Alta dispersione = indicatori in disaccordo

INTEGRAZIONE
─────────────
  from modelli.signal_trust import SignalTrustEngine

  engine    = SignalTrustEngine(horizon=1, window=40, is_intraday=False)
  trust_df  = engine.fit_transform(thermo_df, price_series)

  # In ThermoStateBuilder.build():
  result = pd.concat([result, trust_df], axis=1)

  # Nel TradingEnv: le colonne Trust_ entrano nello stato DDPG
  # automaticamente tramite il prefisso "Thm_" nel builder
  # (aggiungerle con prefisso Thm_Trust_ per coerenza)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════════
# DEFINIZIONE PREDIZIONI IMPLICITE
# ═══════════════════════════════════════════════════════════════════════════════

# Per ogni indicatore: come trasformare il suo valore in una predizione
# direzionale sul rendimento futuro.
# "positive" → segnale alto predice rendimento positivo
# "negative" → segnale alto predice rendimento negativo
# "absolute" → il valore assoluto predice la magnitudine (non direzione)

INDICATOR_POLARITY = {
    # Thm_ canonici
    "Thm_Pressure":    "positive",   # pressione alta → momentum positivo
    "Thm_Temperature": "negative",   # temperatura alta → volatilità → mean-revert
    "Thm_Work":        "positive",   # lavoro crescente → trend sostenuto
    "Thm_Stress":      "negative",   # stress alto → correzione imminente
    "Thm_Efficiency":  "negative",   # inefficienza → attrito → rendimento basso
    "Thm_Entropy":     "negative",   # alta entropia → disordine → mean-revert
    "Thm_Regime":      "custom",     # gestito separatamente (discreto 0-4)
    # TIXI
    "Thm_TIXI":        "negative",   # alta irreversibilità → pre-correzione
    "Thm_TIXI_Z":      "negative",
    # Indicatori esistenti (backward compatibility)
    "Energy_Divergence":    "negative",
    "Energy_Efficiency":    "negative",
    "Market_Pressure":      "positive",
    "Market_Boiling_Z":     "negative",
    "Thermo_Efficiency":    "negative",
}

# Mappatura Regime → predizione direzionale
REGIME_DIRECTION = {
    0: 0.0,   # NEUTRO       → nessuna predizione
    1: 1.0,   # RALLY_REALE  → positivo
    2: -0.5,  # RALLY_ESAUSTO → leggermente negativo
    3: -1.0,  # COMPRESSIONE → negativo
    4: 0.5,   # RIMBALZO     → leggermente positivo
}


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY
# ═══════════════════════════════════════════════════════════════════════════════

def _sigmoid(x: np.ndarray, scale: float = 3.0) -> np.ndarray:
    """Sigmoid scalata: mappa (−∞, +∞) → (0, 1). Scale=3 → ±1 mappa a ~0.95/0.05"""
    return 1.0 / (1.0 + np.exp(-scale * x))


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Correlazione di Pearson con gestione NaN e varianza zero."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 4:
        return 0.0
    x_, y_ = x[mask], y[mask]
    if x_.std() < 1e-10 or y_.std() < 1e-10:
        return 0.0
    with np.errstate(divide='ignore', invalid='ignore'):
        c_matrix = np.corrcoef(x_, y_)
        return float(c_matrix[0, 1]) if not np.isnan(c_matrix[0, 1]) else 0.0


def _sanitize(s: pd.Series, fill: float = 0.0) -> pd.Series:
    return s.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(fill)


def _indicator_to_signal(
    series: pd.Series,
    col: str,
) -> pd.Series:
    """
    Converte il valore dell'indicatore in un segnale direzionale in (-1, 1).
    Gestisce il caso speciale di Thm_Regime (variabile discreta).
    """
    polarity = INDICATOR_POLARITY.get(col, "positive")

    if col == "Thm_Regime":
        return series.map(lambda r: REGIME_DIRECTION.get(int(round(r)), 0.0))

    if polarity == "negative":
        return -series
    elif polarity == "absolute":
        return series.abs()
    else:  # "positive"
        return series


# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL TRUST ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class SignalTrustEngine:
    """
    Misura la qualità predittiva rolling di ogni indicatore termodinamico.

    Parametri
    ──────────
    horizon : int
        Orizzonte di predizione in barre.
        Daily: 1 (giorno dopo), 3 (settimana), 5
        Intraday: 1 (barra dopo), 5 (30 min se 5m bar)

    window : int
        Finestra rolling per IC e HR.
        Daily: 40–60 barre (≈ 2–3 mesi)
        Intraday: 20–40 barre

    ic_weight : float
        Peso dell'IC nel calcolo del Trust Score (0–1).
        (1 − ic_weight) è il peso dell'HR.

    min_ic_abs : float
        IC minimo per considerare un indicatore "attivo".
        Sotto questa soglia Trust ≈ 0.5 (no fiducia).

    is_intraday : bool
        Se True, usa parametri adattati all'intraday.
    """

    def __init__(
        self,
        horizon:     int   = 1,
        window:      int   = 40,
        ic_weight:   float = 0.6,
        min_ic_abs:  float = 0.05,
        is_intraday: bool  = False,
    ):
        self.horizon     = horizon
        self.window      = window
        self.ic_weight   = ic_weight
        self.min_ic_abs  = min_ic_abs
        self.is_intraday = is_intraday

        # Memorizza le colonne trattate (per riapplicare a nuovi dati)
        self._processed_cols: list[str] = []

    # ── API principale ─────────────────────────────────────────────────────────

    def fit_transform(
        self,
        thermo_df:    pd.DataFrame,
        price_series: pd.Series,
    ) -> pd.DataFrame:
        """
        Calcola Trust Score per ogni colonna Thm_* nel DataFrame.

        Args:
            thermo_df    : DataFrame con colonne Thm_* (output di ThermoStateBuilder)
            price_series : Serie dei prezzi di chiusura (non normalizzati preferibilmente)

        Returns:
            DataFrame con colonne Trust_*, stesso indice di thermo_df.
        """
        # Rendimenti realizzati a orizzonte h (target della predizione)
        returns = price_series.pct_change().fillna(0)
        realized = returns.shift(-self.horizon)  # return che avverrà h barre dopo

        # Seleziona solo le colonne Thm_ rilevanti
        thm_cols = [c for c in thermo_df.columns
                    if c.startswith("Thm_") and c in INDICATOR_POLARITY]
        self._processed_cols = thm_cols

        if not thm_cols:
            print("[SignalTrust] Nessuna colonna Thm_ trovata con polarity definita.")
            return pd.DataFrame(index=thermo_df.index)

        trust_dict: dict[str, pd.Series] = {}
        ic_dict:    dict[str, pd.Series] = {}
        hr_dict:    dict[str, pd.Series] = {}

        for col in thm_cols:
            signal = _indicator_to_signal(thermo_df[col], col)
            ic, hr, trust = self._compute_trust_series(signal, realized)

            ic_dict[f"Trust_{col}_IC"]   = ic
            hr_dict[f"Trust_{col}_HR"]   = hr
            trust_dict[f"Trust_{col}"]   = trust

        # ── Metriche aggregate ──────────────────────────────────────────────
        all_trust = pd.DataFrame(trust_dict, index=thermo_df.index)

        trust_global = all_trust.mean(axis=1)
        trust_disp   = all_trust.std(axis=1).fillna(0)

        # ── Assemblaggio ───────────────────────────────────────────────────
        result = pd.DataFrame(index=thermo_df.index)
        for col in thm_cols:
            result[f"Trust_{col}_IC"] = _sanitize(ic_dict[f"Trust_{col}_IC"], 0.0)
            result[f"Trust_{col}_HR"] = _sanitize(hr_dict[f"Trust_{col}_HR"], 0.5)
            result[f"Trust_{col}"]    = _sanitize(trust_dict[f"Trust_{col}"],  0.5)

        result["Trust_Global"]     = _sanitize(trust_global, 0.5)
        result["Trust_Dispersion"] = _sanitize(trust_disp,   0.0)

        n = len(thm_cols)
        mean_trust = result[[f"Trust_{c}" for c in thm_cols]].mean().mean()
        print(
            f"[SignalTrust] {'Intraday' if self.is_intraday else 'Daily'} | "
            f"{n} indicatori | horizon={self.horizon} | window={self.window} | "
            f"Trust medio: {mean_trust:.3f}"
        )

        return result

    def get_trust_colnames(self) -> list[str]:
        """Restituisce le colonne Trust_ aggregate (senza IC e HR per brevità)."""
        base = [f"Trust_{c}" for c in self._processed_cols]
        return base + ["Trust_Global", "Trust_Dispersion"]

    # ── Calcolo per singolo indicatore ────────────────────────────────────────

    def _compute_trust_series(
        self,
        signal:   pd.Series,
        realized: pd.Series,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calcola IC, HR e Trust score rolling per una coppia (segnale, rendimento).

        Returns:
            ic    : Information Coefficient rolling ∈ (−1, 1)
            hr    : Hit Rate rolling ∈ (0, 1)
            trust : Trust Score ∈ (0, 1), con 0.5 = nessuna fiducia
        """
        sig_vals = signal.values.astype(np.float64)
        ret_vals = realized.values.astype(np.float64)
        n        = len(sig_vals)
        W        = self.window

        ic_arr    = np.full(n, np.nan)
        hr_arr    = np.full(n, np.nan)
        trust_arr = np.full(n, 0.5)

        for i in range(W, n):
            sig_w = sig_vals[i - W : i]
            ret_w = ret_vals[i - W : i]

            # Information Coefficient: corr(signal[t], return[t+h]) nella finestra
            ic_val = _safe_corr(sig_w, ret_w)
            ic_arr[i] = ic_val

            # Hit Rate: sign(signal) == sign(return) nella finestra
            valid = np.isfinite(sig_w) & np.isfinite(ret_w)
            if valid.sum() >= 4:
                hits = np.sign(sig_w[valid]) == np.sign(ret_w[valid])
                hr_arr[i] = float(hits.mean())
            else:
                hr_arr[i] = 0.5

            # Trust Score = f(IC, HR)
            # Se |IC| < min_ic_abs → segnale non informativo → Trust ≈ 0.5
            if abs(ic_val) < self.min_ic_abs:
                trust_arr[i] = 0.5
                continue

            # Combina IC e HR in un punteggio normalizzato
            # IC ∈ (−1,1), HR ∈ (0,1): centra HR su 0.5 per allinearlo a IC
            combined = (
                self.ic_weight         * ic_val
                + (1 - self.ic_weight) * (hr_arr[i] - 0.5) * 2
            )
            trust_arr[i] = float(_sigmoid(np.array([combined]))[0])

        ic_s    = pd.Series(ic_arr,    index=signal.index).ffill().bfill().fillna(0.0)
        hr_s    = pd.Series(hr_arr,    index=signal.index).ffill().bfill().fillna(0.5)
        trust_s = pd.Series(trust_arr, index=signal.index).ffill().bfill().fillna(0.5)

        return ic_s, hr_s, trust_s


# ═══════════════════════════════════════════════════════════════════════════════
# REWARD SHAPING CON TRUST
# ═══════════════════════════════════════════════════════════════════════════════

def compute_trust_weighted_signal(
    thermo_df: pd.DataFrame,
    trust_df:  pd.DataFrame,
    col:       str,
) -> pd.Series:
    """
    Restituisce il segnale termodinamico pesato per la sua fiducia corrente.

    trust_weighted = signal * trust_score

    Uso pratico nel TradingEnv:
      - Se Trust_Thm_Stress = 0.9 e Thm_Stress = 2.0 → segnale forte e affidabile
      - Se Trust_Thm_Stress = 0.51 e Thm_Stress = 2.0 → segnale forte ma inaffidabile → peso ridotto
      - Se Trust_Thm_Stress = 0.1 → indicatore contrarian → il DDPG può invertirne la logica
    """
    if col not in thermo_df.columns:
        return pd.Series(0.0, index=thermo_df.index)

    trust_col = f"Trust_{col}"
    if trust_col not in trust_df.columns:
        return thermo_df[col]

    signal = thermo_df[col]
    trust  = trust_df[trust_col]

    # Centra la trust su 0 (0.5 → 0, 1.0 → +0.5, 0.0 → −0.5)
    # Il segnale è amplificato quando trust > 0.5, attenuato quando < 0.5
    trust_centered = (trust - 0.5) * 2   # ∈ (−1, 1)
    weighted = signal * (0.5 + 0.5 * trust_centered)

    return weighted.fillna(0.0)