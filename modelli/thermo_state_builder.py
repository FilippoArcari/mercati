"""
modelli/thermo_state_builder.py
════════════════════════════════════════════════════════════════════════════════

ThermoStateBuilder — Layer di unificazione tra le feature termodinamiche
e i consumer (Pred, DDPGAgent/TradingEnv).

PROBLEMA RISOLTO
─────────────────
Prima di questo modulo:
  - TradingEnv riceveva solo 2 scalari (stress, efficiency), hard-coded
  - Pred non riceveva alcuna feature termodinamica
  - La logica di branching daily/intraday era sparsa in più file

Dopo:
  - Un solo punto di calcolo, chiamabile da load_data() e da TradingEnv
  - Produce un DataFrame con colonne CANONICHE (prefisso "Thm_")
    → invarianti rispetto alla frequenza (daily o intraday)
  - Pred può imparare a PREDIRE lo stato termodinamico futuro
  - TradingEnv riceve un vettore thermo completo nel suo stato

COLONNE CANONICHE OUTPUT
─────────────────────────
  Thm_Pressure       — Pressione Van der Waals (normalizzata [-1,1])
  Thm_Temperature    — Temperatura (entropia cinetica, normalizzata)
  Thm_Work           — Lavoro cumulativo rolling (normalizzato)
  Thm_Stress         — Z-score divergenza pressione vs baseline
  Thm_Efficiency     — Oscillatore efficienza (W/ΔP)
  Thm_Entropy        — Entropia Shannon/Lévy (normalizzata [0,1])
  Thm_Regime         — Regime classificato 0-4 (intero → float)
  Thm_Psi_{ticker}   — Indice di resilienza ai tassi [solo daily]

INTEGRAZIONE
─────────────
  # In load_data() o dopo il fetch dei dati grezzi:
  builder = ThermoStateBuilder(cfg)
  thermo_df = builder.build(df_raw, tickers)
  df_full = pd.concat([df_scaled, thermo_df], axis=1)  # → input di Pred

  # In TradingEnv:
  env = TradingEnv(
      ...,
      thermo_df=thermo_df,   # sostituisce thermo_df_agg
  )
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Optional
from modelli.thermo_innovations import compute_advanced_thermo_features, STANDARD_THERMO_FEATURES



# ═══════════════════════════════════════════════════════════════════════════════
# COSTANTI
# ═══════════════════════════════════════════════════════════════════════════════

CANONICAL_COLS = [
    "Thm_Pressure",
    "Thm_Temperature",
    "Thm_Work",
    "Thm_Stress",
    "Thm_Efficiency",
    "Thm_Entropy",
    "Thm_Regime",
]

# Colonne estese: calcolate e disponibili nel thermo_df per TradingEnv
# (vengono incluse nello stato del DDPG tramite il prefisso Thm_),
# ma ESCLUSE da CANONICAL_COLS per non alterare il numero di canali
# della CNN già addestrata (58 canali). Per includerle nella CNN
# occorre ri-addestrare il predictor con il nuovo dataset.
EXTENDED_THERMO_COLS = [
    "Thm_GibbsEnergy",   # G = H - T·S  (G<0 = processo spontaneo → trade favorevole)
    "Thm_StressAccel",   # d²(Stress)/dt²  (stress sta accelerando → sell anticipato)
]

# Finestre adattive per frequenza
_WINDOWS = {
    "daily":   {"pressure": 20, "entropy": 20, "stress": 60, "efficiency": 10},
    "intraday": {"pressure": 5,  "entropy": 10, "stress": 20, "efficiency": 10},
}

_RATES_CANDIDATES = ["GS10", "T10YIE", "FEDFUNDS", "^TNX"]


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY
# ═══════════════════════════════════════════════════════════════════════════════

def _sanitize(s: pd.Series, fill: float = 0.0) -> pd.Series:
    return s.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(fill)


def _norm_minmax(s: pd.Series) -> pd.Series:
    lo, hi = s.quantile(0.01), s.quantile(0.99)
    rng = max(hi - lo, 1e-8)
    return ((s - lo) / rng * 2 - 1).clip(-1, 1)


def _norm_zscore(s: pd.Series, window: int) -> pd.Series:
    mu  = s.rolling(window, min_periods=1).mean()
    sig = s.rolling(window, min_periods=1).std().clip(lower=1e-8)
    return _sanitize((s - mu) / sig)


def _rolling_shannon(returns: pd.Series, window: int) -> pd.Series:
    def _ent(x):
        counts, _ = np.histogram(x, bins=min(window // 2, 8))
        p = counts / (counts.sum() + 1e-9)
        p = p[p > 0]
        return float(-np.sum(p * np.log(p + 1e-10)))
    return returns.rolling(window).apply(_ent, raw=True).fillna(0)


def _find_rates_col(df: pd.DataFrame) -> Optional[str]:
    for c in _RATES_CANDIDATES:
        if c in df.columns:
            return c
    return None


def _find_volume(df: pd.DataFrame, ticker: str) -> Optional[pd.Series]:
    """Trova la colonna volume per un ticker, gestisce tutti i formati yfinance."""
    candidates = [
        f"{ticker}_Volume", f"Volume_{ticker}",
        f"{ticker} Volume", "Volume", "volume",
    ]
    for c in candidates:
        if c in df.columns:
            return df[c].clip(lower=1.0).ffill().bfill().fillna(1.0)
    # Ricerca case-insensitive
    tl = ticker.lower()
    for col in df.columns:
        if "volume" in col.lower() and tl in col.lower():
            return df[col].clip(lower=1.0).ffill().bfill().fillna(1.0)
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# CALCOLO FEATURE BASE (frequenza-agnostico)
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_pressure_temperature_work(
    close: pd.Series,
    volume: pd.Series,
    windows: dict,
    a_attraction: float = 0.5,
    b_excluded: float = 0.01,
) -> pd.DataFrame:
    """
    Calcola pressione VdW, temperatura e lavoro su qualsiasi frequenza.

    Pressure: P = nRT / (V - nb) - a·n²/V²
    Temperature: T ∝ exp(2·(S - log(V_free)))   [via entropia]
    Work: W = Σ [(P_t + P_{t-1})/2 · (V_t - V_{t-1})]  (trapezi)
    """
    w_p = windows["pressure"]
    w_e = windows["entropy"]

    close  = close.replace(0, np.nan).clip(lower=1e-10).ffill().bfill()
    volume = volume.clip(lower=1.0).ffill().bfill()

    returns  = _sanitize(close.pct_change())
    entropy  = _rolling_shannon(returns, w_e)

    log_vol  = _sanitize(np.log1p(volume))
    v_local  = _sanitize(log_vol.rolling(w_p, min_periods=1).mean(),
                         fill=float(log_vol.mean()))

    v_free   = (v_local - b_excluded).clip(lower=1e-6)
    t_series = _sanitize(np.exp(2 * (entropy - np.log(v_free + 1e-8))).clip(upper=1e6))

    p_ideal      = t_series / v_free
    p_correction = -a_attraction / (v_local ** 2).clip(lower=1e-8)
    pressure     = _sanitize((p_ideal + p_correction).clip(lower=1e-8))

    dV      = _sanitize(v_local.diff())
    p_avg   = _sanitize((pressure + pressure.shift(1)) / 2)
    work    = _sanitize((p_avg * dV).rolling(windows["efficiency"], min_periods=1).sum())

    return pd.DataFrame({
        "raw_pressure":    pressure,
        "raw_temperature": t_series,
        "raw_entropy":     entropy,
        "raw_work":        work,
        "raw_volume":      v_local,
    }, index=close.index)


def _compute_stress(pressure: pd.Series, baseline: Optional[pd.Series],
                    window: int) -> pd.Series:
    """
    Z-score della pressione rispetto a:
      - baseline (pressione attesa da tassi)  [daily]
      - media locale rolling                  [intraday, baseline=None]
    """
    if baseline is not None and baseline.notna().sum() > window:
        ss = StandardScaler()
        p_z = pd.Series(
            ss.fit_transform(pressure.values.reshape(-1, 1)).flatten(),
            index=pressure.index,
        )
        b_z = pd.Series(
            ss.fit_transform(baseline.ffill().bfill().values.reshape(-1, 1)).flatten(),
            index=baseline.index,
        )
        return _sanitize(p_z - b_z)
    else:
        return _norm_zscore(pressure, window)


def _compute_efficiency(work: pd.Series, close: pd.Series, window: int) -> pd.Series:
    """
    Efficienza = W_norm - P_norm  (lavoro vs prezzo normalizzati)
    > 0: attrito/distribuzione
    < 0: movimento fluido
    """
    mm = MinMaxScaler()
    w_n = pd.Series(
        mm.fit_transform(work.values.reshape(-1, 1)).flatten(),
        index=work.index,
    )
    p_n = pd.Series(
        mm.fit_transform(close.values.reshape(-1, 1)).flatten(),
        index=close.index,
    )
    raw = w_n - p_n
    return _sanitize(raw.rolling(window, min_periods=1).mean())


def _classify_regime(stress: pd.Series, efficiency: pd.Series,
                     entropy: pd.Series) -> pd.Series:
    """
    Regime 0–4 basato su segnali termodinamici.
      0 — NEUTRO
      1 — RALLY_REALE      stress↓, efficiency bassa
      2 — RALLY_ESAUSTO    stress↑, efficiency↑
      3 — COMPRESSIONE     stress↑↑, efficiency molto↑
      4 — RIMBALZO         entropy↑, stress↓ con stress<0
    """
    regime = pd.Series(0, index=stress.index, dtype=np.float32)

    regime[( stress < -0.5) & (efficiency.abs() < 1.5)]               = 1.0
    regime[( stress >  1.0) & (efficiency > 1.5)]                     = 2.0
    regime[( stress >  1.0) & (efficiency > 2.0)]                     = 3.0
    regime[(entropy > 0.65) & (stress.diff() < 0) & (stress < 0)]     = 4.0

    return regime


# ═══════════════════════════════════════════════════════════════════════════════
# BUILDER PRINCIPALE
# ═══════════════════════════════════════════════════════════════════════════════

class ThermoStateBuilder:
    """
    Costruisce il vettore di stato termodinamico canonico (prefisso Thm_)
    per qualsiasi frequenza di mercato.

    Parametri
    ──────────
    interval : str
        Frequenza dei dati: "1d" per daily, qualsiasi altro per intraday.
    a_attraction : float
        Parametro di attrazione Van der Waals (forza di herding).
    b_excluded : float
        Volume escluso (limite di liquidità).
    max_lag : int
        Lag massimo per l'ottimizzazione del lag monetario (solo daily).

    Esempio di utilizzo
    ────────────────────
    builder = ThermoStateBuilder(interval="1d")
    thermo_df = builder.build(df_raw, tickers=["SPY", "QQQ"])

    # Per integrare in Pred (concatena alle feature principali):
    df_full = pd.concat([df_scaled, thermo_df[CANONICAL_COLS]], axis=1)

    # Per TradingEnv:
    env = TradingEnv(..., thermo_df=thermo_df)
    """

    def __init__(
        self,
        interval:      str   = "1d",
        a_attraction:  float = 0.5,
        b_excluded:    float = 0.01,
        max_lag:       int   = 90,
        add_trust:     bool  = True,
        trust_horizon: int   = 1,
        trust_window:  int   = 40,
    ):
        self.is_daily      = (interval == "1d")
        self.a_attraction  = a_attraction
        self.b_excluded    = b_excluded
        self.max_lag       = max_lag
        self.windows       = _WINDOWS["daily"] if self.is_daily else _WINDOWS["intraday"]
        self._best_lag:    Optional[int] = None
        self.add_trust     = add_trust
        self.trust_horizon = trust_horizon
        self.trust_window  = trust_window

    # ── API principale ─────────────────────────────────────────────────────────

    def build(
        self,
        df_raw:  pd.DataFrame,
        tickers: list[str],
    ) -> pd.DataFrame:
        """
        Calcola tutte le feature termodinamiche canoniche.

        Input:
          df_raw  — DataFrame raw con colonne prezzo/volume e opzionalmente tassi.
          tickers — Lista dei ticker da usare per costruire il portafoglio aggregato.

        Output:
          DataFrame con colonne CANONICAL_COLS + eventualmente Thm_Psi_{ticker}.
          Stesso indice di df_raw, nessun NaN.
        """
        close, volume = self._aggregate_portfolio(df_raw, tickers)
        base          = _compute_pressure_temperature_work(
            close, volume, self.windows,
            self.a_attraction, self.b_excluded,
        )

        # ── Baseline pressione attesa (solo daily) ──────────────────────────
        baseline = None
        if self.is_daily:
            rates_col = _find_rates_col(df_raw)
            if rates_col:
                rates     = df_raw[rates_col].ffill().bfill()
                best_lag  = self._find_best_lag(base["raw_pressure"], rates)
                self._best_lag = best_lag
                baseline  = (1.0 / (rates.shift(best_lag) + 1e-5))
                print(f"[ThermoBuilder] Lag monetario ottimale: {best_lag}gg")

        # ── Calcolo segnali derivati ────────────────────────────────────────
        w_s = self.windows["stress"]
        w_e = self.windows["efficiency"]

        stress     = _compute_stress(base["raw_pressure"], baseline, w_s)
        efficiency = _compute_efficiency(base["raw_work"], close, w_e)

        # Entropia normalizzata [0, 1]
        ent_raw    = base["raw_entropy"]
        ent_lo, ent_hi = ent_raw.quantile(0.01), ent_raw.quantile(0.99)
        ent_norm   = ((ent_raw - ent_lo) / max(ent_hi - ent_lo, 1e-8)).clip(0, 1)

        regime = _classify_regime(stress, efficiency, ent_norm)

        # ── Normalizzazione colonne canoniche ───────────────────────────────
        result = pd.DataFrame(index=df_raw.index)
        result["Thm_Pressure"]    = _norm_minmax(_sanitize(base["raw_pressure"]))
        result["Thm_Temperature"] = _norm_minmax(_sanitize(base["raw_temperature"]))
        result["Thm_Work"]        = _norm_minmax(_sanitize(base["raw_work"]))
        result["Thm_Stress"]      = _sanitize(stress)
        result["Thm_Efficiency"]  = _sanitize(efficiency)
        result["Thm_Entropy"]     = _sanitize(ent_norm, fill=0.5)
        result["Thm_Regime"]      = regime

        # ── Gibbs Free Energy: G = H - T·S ─────────────────────────────────
        # H (entalpia) ≈ pressione × |lavoro| → energia contenuta nel trend corrente.
        # T = temperatura (agitazione locale), S = entropia (disordine).
        # G < 0 → processo spontaneo → trade energeticamente favorevole.
        # G > 0 → processo non spontaneo → il mercato sta "forzando" il movimento.
        H_raw = base["raw_pressure"].abs() * (base["raw_work"].abs() + 1e-8)
        G_raw = H_raw - base["raw_temperature"] * base["raw_entropy"]
        result["Thm_GibbsEnergy"] = _sanitize(_norm_minmax(_sanitize(G_raw)))

        # ── Stress Acceleration: d²(Stress)/dt² ────────────────────────────
        # Derivata seconda dello stress. Positivo = stress accelera verso l'alto
        # (sell anticipato). Usata in TradingEnv [Fix #15] per il bonus sell precoce.
        d1_stress = result["Thm_Stress"].diff()
        d2_stress = d1_stress.diff()
        result["Thm_StressAccel"] = _sanitize(d2_stress.clip(-3.0, 3.0), fill=0.0)

        # ── Psi per ticker (daily only) ─────────────────────────────────────
        if self.is_daily:
            rates_col = _find_rates_col(df_raw)
            if rates_col:
                psi_df = self._compute_psi(df_raw, tickers, rates_col)
                if not psi_df.empty:
                    result = pd.concat([result, psi_df], axis=1)

        result = result.ffill().bfill().fillna(0.0)

        # ── Trust scores ────────────────────────────────────────────────────
        if self.add_trust:
            try:
                from signal_trust import SignalTrustEngine
            except ImportError:
                from modelli.signal_trust import SignalTrustEngine

            engine = SignalTrustEngine(
                horizon     = self.trust_horizon,
                window      = self.trust_window,
                is_intraday = not self.is_daily,
            )
            trust_df = engine.fit_transform(result, close)
            if not trust_df.empty:
                result = pd.concat([result, trust_df], axis=1)
                result = result.ffill().bfill().fillna(0.0)

        n_trust    = len([c for c in result.columns if c.startswith("Trust_")])
        n_psi      = len([c for c in result.columns if c.startswith("Thm_Psi")])
        n_extended = len([c for c in result.columns if c in EXTENDED_THERMO_COLS])
        print(
            f"[ThermoBuilder] {'Daily' if self.is_daily else 'Intraday'} | "
            f"{len(result.columns)} col totali: "
            f"{len(CANONICAL_COLS)} canonical + {n_extended} extended "
            f"+ {n_psi} Psi + {n_trust} Trust"
        )
        
         # ★ NUOVO: Integra feature termodinamiche avanzate (innovations)
        try:
            # Prepara DataFrame temporaneo con dati necessari
            temp_df = pd.DataFrame(index=df_raw.index)
            temp_df['Close'] = close
            temp_df['Market_Pressure'] = base["raw_pressure"]
            temp_df['Market_Temperature'] = base["raw_temperature"]
            temp_df['Market_Entropy'] = base["raw_entropy"]
            temp_df['Market_Work_Cum'] = base["raw_work"]
            
            # Trova rates_col se disponibile
            rates_col = _find_rates_col(df_raw)
            if rates_col:
                temp_df['DGS10'] = df_raw[rates_col]
            else:
                temp_df['DGS10'] = 0.045  # default
            
            # Calcola feature avanzate (lag adattivo, efficiency, phase, etc.)
            temp_df_enhanced = compute_advanced_thermo_features(
                temp_df,
                pressure_col='Market_Pressure',
                temp_col='Market_Temperature',
                entropy_col='Market_Entropy',
                work_col='Market_Work_Cum',
                price_col='Close',
                rates_col='DGS10'
            )
            
            # Aggiungi solo le nuove colonne al result
            new_cols = [
                'Thm_Phase', 
                'Thm_Efficiency',  # Sovrascrive quella esistente con versione avanzata
                'Thm_MonetaryLag',
                'Thm_StressThreshold',
                'Thm_SellSignal',
                'Thm_StressZScore'
            ]
            
            for col in new_cols:
                if col in temp_df_enhanced.columns:
                    # Thm_Phase è stringa, converti in int per il modello
                    if col == 'Thm_Phase':
                        phase_map = {
                            'Espansione': 0,
                            'Compressione': 1,
                            'Transizione': 2,
                            'Caos': 3
                        }
                        result[col] = temp_df_enhanced[col].map(phase_map).fillna(2).astype(float)
                    else:
                        result[col] = temp_df_enhanced[col]
            
            print(f"[ThermoBuilder] ✅ Thermo Innovations integrate: {len(new_cols)} nuove feature")
            
        except Exception as e:
            print(f"[ThermoBuilder] ⚠️  Errore in thermo innovations (continuo senza): {e}")
        
        # Assicurati che tutto sia sanitizzato
        result = result.ffill().bfill().fillna(0.0)

        return result

    @property
    def canonical_cols(self) -> list[str]:
        return CANONICAL_COLS.copy()

    @property
    def best_lag(self) -> Optional[int]:
        return self._best_lag

    # ── Helpers privati ────────────────────────────────────────────────────────

    def _aggregate_portfolio(
        self,
        df_raw:  pd.DataFrame,
        tickers: list[str],
    ) -> tuple[pd.Series, pd.Series]:
        """Media dei prezzi di chiusura e somma dei volumi."""
        close_list, vol_list = [], []

        for t in tickers:
            c = next((c for c in [t, f"{t}_Close", f"Close_{t}"]
                      if c in df_raw.columns), None)
            if c is None:
                continue
            close_list.append(df_raw[c].ffill().bfill())
            v = _find_volume(df_raw, t)
            if v is not None:
                vol_list.append(v)

        if not close_list:
            # Fallback: prima colonna numerica
            close_list = [df_raw.select_dtypes(include=np.number).iloc[:, 0]]

        close = pd.concat(close_list, axis=1).mean(axis=1)
        volume = pd.concat(vol_list, axis=1).sum(axis=1) if vol_list else \
                 pd.Series(1e6, index=df_raw.index)

        return (
            close.replace(0, np.nan).clip(lower=1e-10).ffill().bfill(),
            volume.clip(lower=1.0).ffill().bfill(),
        )

    def _find_best_lag(self, pressure: pd.Series, rates: pd.Series) -> int:
        """Cross-correlazione per trovare il lag ottimale."""
        valid = pressure.notna() & rates.notna()
        if valid.sum() < self.max_lag + 20:
            return 65  # default empirico
        p = pressure[valid].values
        r = rates[valid].values
        corrs = []
        for lag in range(self.max_lag + 1):
            if lag < len(r):
                c = np.corrcoef(p[lag:], r[:len(r) - lag])[0, 1]
                corrs.append(abs(c) if np.isfinite(c) else 0.0)
            else:
                corrs.append(0.0)
        return int(np.argmax(corrs))

    def _compute_psi(
        self,
        df_raw:    pd.DataFrame,
        tickers:   list[str],
        rates_col: str,
        window:    int = 30,
    ) -> pd.DataFrame:
        """
        Ψ_i(t) = Corr_window(ΔW_i, Δr) × (σ_W_i / σ_r)
        Misura la sensibilità dell'asset ai tassi d'interesse.
        """
        rates   = df_raw[rates_col].ffill()
        delta_r = rates.diff().fillna(0)
        sigma_r = delta_r.rolling(window).std().clip(lower=1e-8)

        psi_dict = {}
        for t in tickers:
            close = next((df_raw[c].ffill().bfill()
                          for c in [t, f"{t}_Close"] if c in df_raw.columns), None)
            if close is None:
                continue
            vol = _find_volume(df_raw, t)
            if vol is None:
                vol = pd.Series(1e6, index=df_raw.index)

            try:
                base    = _compute_pressure_temperature_work(
                    close, vol, self.windows)
                delta_w = _sanitize(base["raw_work"].diff())
                sigma_w = delta_w.rolling(window).std().clip(lower=1e-8)
                corr    = delta_w.rolling(window).corr(delta_r).fillna(0)
                psi_i   = _sanitize(corr * (sigma_w / sigma_r))

                # Normalizza in [-1, 1]
                lo = psi_i.quantile(0.01)
                hi = psi_i.quantile(0.99)
                rng = max(hi - lo, 1e-8)
                psi_dict[f"Thm_Psi_{t}"] = ((psi_i - lo) / rng * 2 - 1).clip(-1, 1)
            except Exception as e:
                print(f"[ThermoBuilder] Psi skip per {t}: {e}")

        return pd.DataFrame(psi_dict, index=df_raw.index) if psi_dict else pd.DataFrame()