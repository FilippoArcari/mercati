"""
modelli/thermodynamics.py — Feature termodinamiche + indice di resilienza Ψ
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import entropy 
from sklearn.discriminant_analysis import StandardScaler

from modelli.calibrate_vdw import (
    calibrate_single,
    calibrate_portfolio,
    vdw_pressure,
    VdWParams,
    ZETA_R,
    ZETA_Q,
    GAMMA,
)
from modelli.intraday_thermo import (
    compute_intraday_thermo_features,
    detect_market_regime,
)
 



# ─── Funzioni di base ─────────────────────────────────────────────────────────

def _rolling_entropy(returns: pd.Series, window: int = 20) -> pd.Series:
    def _ent(x):
        prob, _ = np.histogram(x, bins=20, density=True)
        prob = prob[prob > 0]
        return entropy(prob)
    return returns.rolling(window).apply(_ent, raw=True)


def _rolling_autocorr(returns: pd.Series, window: int = 20, lag: int = 1) -> pd.Series:
    def _ac(x):
        s = pd.Series(x)
        return s.autocorr(lag=lag) if len(s) > lag + 1 else 0.0
    return returns.rolling(window).apply(_ac, raw=False).fillna(0)


# ─── Van der Waals ────────────────────────────────────────────────────────────

def calculate_pressure_and_work(
    close:          pd.Series,
    volume:         pd.Series,
    window:         int           = 20,
    kb:             float         = 1.0,
    vdw_params:     VdWParams | None = None,
    n_particles:    int           = 1,
) -> pd.DataFrame:
    """
    Calcola pressione e lavoro termodinamico usando Van der Waals.
    
    Se vdw_params è fornito (calibrato con calibrate_vdw), usa i parametri
    empirici dal paper Gabaix et al. 2003 per una rappresentazione più
    accurata del mercato.
    
    Altrimenti, fallback ai parametri heuristici originali.
    
    IMPROVEMENT: I parametri a e b sono ora calibrati empiricamente
    dall'analisi tail della distribuzione storica, non hardcoded.
    """
    returns  = close.pct_change().fillna(0)
    s_series = _rolling_entropy(returns, window)
    v_series = np.log1p(volume.rolling(window).mean())
    
    if vdw_params is not None:
        # ── Usa parametri calibrati da Gabaix et al. 2003 ────────────────────
        a = vdw_params.a
        b = vdw_params.b
        b_const = b * n_particles
    else:
        # ── Fallback ai parametri heuristici originali ────────────────────
        a_series = _rolling_autocorr(returns, window)
        b_const = v_series.min() * 0.1
        # Stima dinamica di a basato su momentum
        a = a_series.mean() if hasattr(a_series, 'mean') else 0.1
    
    # ── Calcolo pressione VdW: P = nRT/(V-nb) - a·n²/V² ───────────────────
    v_free  = (v_series - b_const).clip(lower=0.1)
    t_series = np.exp(2 * (s_series - np.log(v_free))).clip(upper=1e6)
    
    # Pressione ideale
    p_ideal = (kb * t_series) / v_free
    
    # Correzione Van der Waals: termine attrattivo -a·n²/V²
    # Il coefficiente a riduce la pressione quando c'è "herding"
    p_correction = -a * (n_particles ** 2) / (v_series ** 2)
    p_series = p_ideal + p_correction
    
    # Clamp pressione per evitare singolarità numeriche
    p_series = p_series.clip(lower=1e-8)
    
    # ── Calcolo lavoro: W = ∫P dV ──────────────────────────────────────────
    delta_v  = v_series.diff().fillna(0)
    p_avg    = (p_series + p_series.shift(1)) / 2
    work_cum = (p_avg * delta_v).fillna(0).cumsum()

    return pd.DataFrame({
        "Market_Pressure":    p_series,
        "Market_Temperature": t_series,
        "Market_Entropy":     s_series,
        "Market_Work_Cum":    work_cum,
        "Volume_Delta":       delta_v,
    }, index=close.index)


# ─── Indice di resilienza Ψ per ticker, per timestep ─────────────────────────

def compute_psi_series(
    df_raw:     pd.DataFrame,
    tickers:    list[str],
    rates_col:  str,
    window:     int = 30,
) -> pd.DataFrame:
    """
    Calcola Ψ_i(t) per ogni ticker e ogni istante temporale.

        Ψ_i(t) = Corr_window(ΔW_i, Δr) × (σ_W_i / σ_r)

    Dove:
      ΔW_i  = variazione giornaliera del lavoro termodinamico del ticker i
      Δr    = variazione giornaliera dei tassi
      σ_W_i = volatilità rolling di ΔW_i
      σ_r   = volatilità rolling di Δr

    Valore alto  → asset molto sensibile ai tassi (da ridurre quando r↑)
    Valore basso → asset resiliente ai tassi (da tenere quando r↑)

    Finestra 30gg (invece di 60) per catturare il regime corrente.

    Returns
    -------
    DataFrame (index=date, columns=[f"Psi_{ticker}" for ticker in tickers])
    Colonne normalizzate in [-1, 1] con MinMaxScaler per compatibilità con il modello.
    """
    if rates_col not in df_raw.columns:
        print(f"[Psi] rates_col '{rates_col}' non trovata — Ψ non calcolato.")
        return pd.DataFrame(index=df_raw.index)

    rates   = df_raw[rates_col].ffill()
    delta_r = rates.diff().fillna(0)
    sigma_r = delta_r.rolling(window).std().clip(lower=1e-8)

    psi_dict: dict[str, pd.Series] = {}

    for ticker in tickers:
        vol_col = f"{ticker}_Volume"

        if ticker not in df_raw.columns:
            continue

        close_t = df_raw[ticker].ffill()

        # Volume: usa la colonna dedicata se disponibile, altrimenti proxy
        if vol_col in df_raw.columns:
            volume_t = df_raw[vol_col].ffill()
        else:
            volume_t = (close_t.diff().abs() * 1e6).fillna(1e6)

        try:
            thermo_t  = calculate_pressure_and_work(close_t, volume_t, window=window)
            delta_w   = thermo_t["Market_Work_Cum"].diff().fillna(0)
            sigma_w   = delta_w.rolling(window).std().clip(lower=1e-8)
            corr_w_r  = delta_w.rolling(window).corr(delta_r).fillna(0)

            psi_i = (corr_w_r * (sigma_w / sigma_r)).fillna(0)
            psi_dict[f"Psi_{ticker}"] = psi_i

        except Exception as e:
            print(f"[Psi] Errore per {ticker}: {e}")
            continue

    if not psi_dict:
        return pd.DataFrame(index=df_raw.index)

    psi_df = pd.DataFrame(psi_dict, index=df_raw.index)

    # Normalizza in [-1,1] tramite clip su percentili robusti
    for col in psi_df.columns:
        lo = psi_df[col].quantile(0.01)
        hi = psi_df[col].quantile(0.99)
        rng = max(hi - lo, 1e-8)
        psi_df[col] = ((psi_df[col] - lo) / rng * 2 - 1).clip(-1, 1)

    psi_df = psi_df.ffill().bfill()
    print(f"[Psi] Indice Ψ calcolato per {len(psi_dict)} ticker "
          f"(finestra {window}gg, normalizzato [-1,1])")
    return psi_df
def _compute_thermo_for_env(
    cfg,
    df: "pd.DataFrame",
    tickers: list[str],
    train_df: "pd.DataFrame",
    test_df:  "pd.DataFrame",
) -> tuple["pd.DataFrame", "pd.DataFrame"]:
    """
    Calcola le feature termodinamiche intraday per train e test set.
 
    Ritorna:
      (thermo_train_df, thermo_test_df)
 
    Se frequency != minute (cioè siamo in daily), ritorna (None, None)
    e il TradingEnv userà solo Ψ come finora.
    """
    import pandas as pd
 
    # Solo per intraday (minute bars)
    is_intraday = getattr(cfg.frequency, "interval", "1d") != "1d"
    if not is_intraday:
        print("[Trade] Daily mode: skip intraday thermo features")
        return None, None
 
    print("[Trade] Calcolo feature termodinamiche intraday...")
 
    # Parametri da config (con default ragionevoli)
    thermo_params = {
        "pressure_window":    getattr(cfg.get("intraday_thermo", {}), "pressure_window",   5),
        "entropy_window":     getattr(cfg.get("intraday_thermo", {}), "entropy_window",    10),
        "stress_window":      getattr(cfg.get("intraday_thermo", {}), "stress_window",     20),
        "efficiency_window":  getattr(cfg.get("intraday_thermo", {}), "efficiency_window", 10),
        "levy_alpha":         getattr(cfg.get("intraday_thermo", {}), "levy_alpha",        1.7),
    }
 
    thermo_train = compute_intraday_thermo_features(
        df=train_df,
        ticker_cols=tickers,
        params=thermo_params,
    )
    thermo_test = compute_intraday_thermo_features(
        df=test_df,
        ticker_cols=tickers,
        params=thermo_params,
    )
 
    # Log regime distribution (utile per debug)
    regime_train = detect_market_regime(thermo_train)
    regime_counts = regime_train.value_counts().sort_index()
    regime_names = {0: "NEUTRO", 1: "RALLY_REALE", 2: "RALLY_ESAUSTO",
                    3: "COMPRESSIONE", 4: "RIMBALZO"}
    print("[Trade] Distribuzione regimi (train set):")
    for r, count in regime_counts.items():
        pct = 100 * count / len(regime_train)
        print(f"  {regime_names.get(r, r):20s}: {count:5d} bar ({pct:.1f}%)")
 
    return thermo_train, thermo_test


# ─── Divergenza energetica ────────────────────────────────────────────────────

def calculate_energy_divergence(
    pressure: pd.Series,
    rates:    pd.Series,
    max_lag:  int = 90,
) -> tuple[pd.Series, int]:
    from sklearn.preprocessing import StandardScaler

    corrs    = [pressure.corr(rates.shift(i)) for i in range(max_lag + 1)]
    best_lag = int(np.argmax(np.abs(corrs)))
    expected = rates.shift(best_lag)

    scaler = StandardScaler()
    p_z = pd.Series(
        scaler.fit_transform(pressure.values.reshape(-1, 1)).flatten(),
        index=pressure.index,
    )
    e_z = pd.Series(
        scaler.fit_transform(expected.dropna().values.reshape(-1, 1)).flatten(),
        index=expected.dropna().index,
    ).reindex(pressure.index)

    return (p_z - e_z).fillna(0), best_lag

# ─── Calibrazione iniziale dei parametri VdW ──────────────────────────────────────

def init_vdw_calibration(
    df_raw:      pd.DataFrame,
    ticker_cols: list[str],
    verbose:     bool = True,
) -> dict[str, VdWParams]:
    """
    Calibra i parametri Van der Waals per ogni ticker usando il paper
    Gabaix et al. 2003. Deve essere chiamato una sola volta all'inizio
    del preprocessing.
    
    Parametri
    ---------
    df_raw       : DataFrame grezzo con prezzi e volumi
    ticker_cols  : Lista dei ticker da calibrare
    verbose      : Se True, stampa diagnostiche
    
    Restituisce
    -----------
    dict[ticker -> VdWParams] con parametri a, b calibrati empiricamente.
                              Uso: vdw_dict['AAPL'].a per accedere al parametro a.
    """
    if verbose:
        print(f"\\n[VdW Init] Calibrazione iniziale per {len(ticker_cols)} ticker...")
    
    prices_cols = [c for c in ticker_cols if c in df_raw.columns]
    volumes_cols = [f"{c}_Volume" for c in ticker_cols if f"{c}_Volume" in df_raw.columns]
    
    if not prices_cols:
        print("[VdW Init] AVVERTENZA: nessun prezzo trovato — usando parametri di default")
        return {}
    
    # Estrai prezzi e volumi
    prices_df = df_raw[prices_cols]
    
    if volumes_cols:
        volumes_df = df_raw[volumes_cols]
        volumes_df.columns = prices_df.columns  # Allinea nomi
    else:
        # Se volumi non disponibili, usa proxy da ampiezza di trading
        volumes_df = prices_df.diff().abs() * 1e6
    
    try:
        vdw_calibration = calibrate_portfolio(
            prices_df=prices_df,
            volumes_df=volumes_df,
            tickers=prices_cols,
            pct_floor=5.0,
        )
        if verbose:
            print(f"[VdW Init] Calibrazione completata ✓")
        return vdw_calibration
    except Exception as e:
        if verbose:
            print(f"[VdW Init] ERRORE nella calibrazione: {e} — usando default")
        return {}

# ─── Entry point principale ───────────────────────────────────────────────────

def compute_thermodynamic_features(
    df_raw:           pd.DataFrame,
    ticker_cols:      list[str],
    rates_col:        str | None = None,
    window:           int        = 20,
    max_lag:          int        = 90,
    psi_window:       int        = 30,
    kb:               float      = 1.0,
    vdw_calibration:  dict | None = None,
) -> pd.DataFrame:
    """
    Calcola tutte le feature termodinamiche + Ψ per ticker.

    IMPROVEMENT: Se vdw_calibration è fornito (da init_vdw_calibration),
    usa parametri empiricamente calibrati invece di valori hardcoded.
    
    Colonne output:
      Market_Pressure, Market_Temperature, Market_Entropy,
      Market_Work_Cum, Volume_Delta,
      [Energy_Divergence, Thermo_Best_Lag  se rates_col disponibile]
      [Psi_{ticker} per ogni ticker        se rates_col disponibile]
    """
    close_cols  = [c for c in ticker_cols if c in df_raw.columns]
    volume_cols = [f"{c}_Volume" for c in ticker_cols if f"{c}_Volume" in df_raw.columns]

    if hasattr(df_raw.columns, "levels"):
        portfolio_close  = df_raw["Close"][close_cols].mean(axis=1)
        portfolio_volume = df_raw["Volume"][close_cols].sum(axis=1)
    else:
        portfolio_close  = df_raw[close_cols].mean(axis=1)
        if volume_cols:
            portfolio_volume = df_raw[volume_cols].sum(axis=1)
        elif "Volume" in df_raw.columns:
            portfolio_volume = df_raw["Volume"]
        else:
            portfolio_volume = (portfolio_close.diff().abs() * 1e6).fillna(1e6)

    # Feature Van der Waals aggregate
    # IMPROVEMENT: Se fornita la calibrazione, usa i parametri empirici Gabaix
    vdw_params_avg = None
    if vdw_calibration:
        a_values = [p.a for p in vdw_calibration.values()]
        b_values = [p.b for p in vdw_calibration.values()]
        if a_values and b_values:
            # Crea parametri aggregati (media ponderata del portafoglio)
            from dataclasses import replace
            first_params = list(vdw_calibration.values())[0]
            vdw_params_avg = replace(
                first_params,
                a=np.mean(a_values),
                b=np.mean(b_values),
            )
            print(f"[Thermo] VdW params: a={vdw_params_avg.a:.6f}, "
                  f"b={vdw_params_avg.b:.6f} (calibrati da {len(vdw_calibration)} ticker)")
    
    thermo = calculate_pressure_and_work(
        close=portfolio_close,
        volume=portfolio_volume,
        window=window,
        kb=kb,
        vdw_params=vdw_params_avg,
        n_particles=len(close_cols),
    )

    # Divergenza energetica
    if rates_col is not None and rates_col in df_raw.columns:
        divergence, best_lag = calculate_energy_divergence(
            thermo["Market_Pressure"], df_raw[rates_col], max_lag
        )
        thermo["Energy_Divergence"] = divergence
        thermo["Thermo_Best_Lag"]   = float(best_lag)
        print(f"[Thermo] Lag monetario ottimale: {best_lag} giorni")

        # Ψ per ticker (feature aggiuntive per il DCNN e lo stato del DDPG)
        psi_df = compute_psi_series(
            df_raw=df_raw,
            tickers=close_cols,
            rates_col=rates_col,
            window=psi_window,
        )
        if not psi_df.empty:
            thermo = pd.concat([thermo, psi_df], axis=1)
    else:
        print("[Thermo] rates_col non disponibile → Energy_Divergence e Ψ non calcolati")

    thermo = thermo.ffill().bfill()
    print(f"[Thermo] {thermo.shape[1]} feature termodinamiche calcolate: "
          f"{[c for c in thermo.columns if not c.startswith('Psi_')]} "
          f"+ {len([c for c in thermo.columns if c.startswith('Psi_')])} Ψ ticker")
    return thermo


class QuantumThermodynamicProcessor:
    """
    SISTEMA DI PRE-PROCESSAMENTO AVANZATO
    Combina: Termodinamica dei Gas Reali + Random Matrix Theory (Denoising)
    
    IMPROVEMENT: I parametri Van der Waals (a, b) possono ora essere
    calibrati empiricamente dal paper Gabaix et al. 2003 tramite
    calibrate_portfolio(), anzichè usare valori hardcoded.
    """
    def __init__(
        self,
        n_tickers:       int,
        r_param:         float          = 8.314,
        a_vdw:           float | None   = None,
        b_vdw:           float | None   = None,
        vdw_params:      VdWParams | None = None,
    ):
        self.n = n_tickers
        self.R = r_param
        self.scaler = StandardScaler()
        
        # Se fornito VdWParams calibrato, usalo; altrimenti fallback ai valori
        if vdw_params is not None:
            self.a = vdw_params.a
            self.b = vdw_params.b
            self.vdw_params = vdw_params
        else:
            self.a = a_vdw if a_vdw is not None else 0.1
            self.b = b_vdw if b_vdw is not None else 0.01
            self.vdw_params = None
        
    def get_thermodynamic_features(
        self,
        df_prices:     pd.DataFrame,
        df_volumes:    pd.DataFrame,
        rates_10y:     pd.Series,
    ) -> pd.DataFrame:
        """
        Calcola T, P, W e Stress Z-Score basato su Van der Waals.
        
        IMPROVEMENT: Usa i parametri a, b calibrati invece di hardcoded.
        """
        """Calcola T, P, W e Stress Z-Score basato su Van der Waals"""
        # 1. Temperatura via Entropia di Shannon (Finestra 20gg)
        returns = df_prices.pct_change().dropna()
        
        def calc_entropy(x):
            counts = np.histogram(x, bins=10)[0]
            p = counts / (counts.sum() + 1e-9)
            return entropy(p + 1e-9)
        
        T = returns.rolling(20).apply(calc_entropy, raw=True).ffill().bfill()
        
        # 2. Volume Logaritmico (V)
        V = np.log(df_volumes.replace(0, 1))
        
        # 3. Pressione (P) - Equazione di Stato
        # P = nRT / (V - nb) - a * (n/V)^2
        P = (self.n * self.R * T) / (V - self.n * self.b) - self.a * (self.n / V)**2
        
        # 4. Lavoro (W) - Integrale P dV
        dV = V.diff().fillna(0)
        W_inc = P * dV
        W_cum = W_inc.cumsum()
        
        # 5. Stress Z-Score (Divergenza dai tassi a 10y con lag 65gg)
        rates_lagged = rates_10y.shift(65).ffill().bfill()
        # Pressione teorica attesa inversamente proporzionale ai tassi
        P_expected = 1 / (rates_lagged + 1e-5)
        stress_divergence = P - P_expected
        z_stress = (stress_divergence - stress_divergence.mean()) / (stress_divergence.std() + 1e-9)
        
        features = pd.concat([T, P, W_cum, z_stress], axis=1)
        features.columns = ['Temp', 'Press', 'Work', 'Stress']
        return features.ffill().bfill()

    def apply_rmt_denoising(self, feature_matrix: np.ndarray):
        """
        Applica Marchenko-Pastur per pulire la matrice di correlazione.
        Restituisce le feature 'Denoised' (proiettate sugli autovettori significativi).
        """
        T, N = feature_matrix.shape
        if T <= N: return feature_matrix # Safe guard
        
        # Standardizza
        x_std = self.scaler.fit_transform(feature_matrix)
        
        # Matrice di Correlazione
        corr = np.corrcoef(x_std, rowvar=False)
        e_val, e_vec = np.linalg.eigh(corr)
        
        # Limite di Marchenko-Pastur (lambda_plus)
        q = T / N
        sigma_sq = 1.0 # Dati standardizzati
        lambda_plus = sigma_sq * (1 + np.sqrt(1/q))**2
        
        # Clipping: manteniamo solo gli autovalori sopra la soglia di rumore
        # Sostituiamo quelli sotto con la media per preservare la traccia (varianza totale)
        kept_indices = np.where(e_val > lambda_plus)[0]
        
        if len(kept_indices) == 0:
            return x_std # Tutto rumore, restituisci originale (raro)
            
        # Denoised Eigenvalues
        e_val_denoised = e_val.copy()
        low_val_mean = np.mean(e_val[e_val <= lambda_plus])
        e_val_denoised[e_val <= lambda_plus] = low_val_mean
        
        # Ricostruisci matrice correlazione pulita
        corr_denoised = e_vec @ np.diag(e_val_denoised) @ e_vec.T
        # Forza la diagonale a 1
        diag = np.diag(corr_denoised)
        corr_denoised = corr_denoised / np.sqrt(np.outer(diag, diag))
        
        # Trasforma le feature originali proiettandole sulla struttura pulita
        # (Simile a una PCA robusta al rumore)
        denoised_features = x_std @ e_vec[:, kept_indices]
        
        return denoised_features