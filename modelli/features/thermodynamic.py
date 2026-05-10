"""
modelli/features/thermodynamic.py

Modello Termodinamico di Mercato v3.0
======================================
Implementa le feature termodinamiche descritte nel report tecnico.
Le colonne prodotte sono pensate per essere le "context features" del Predictor,
cioè le ultime n_context_features colonne del DataFrame restituito da get_data().

Il Predictor le legge già tramite cross-attention — nessuna modifica al modello.

Feature prodotte (9 colonne):
──────────────────────────────────────────────────────────────────────────────
1.  thermo_T           – Temperatura:    Entropia di Shannon su rendimenti rolling
2.  thermo_dT          – Gradiente T:    Derivata della temperatura (segnale di shock)
3.  thermo_P           – Pressione VdW:  Equazione di Van der Waals adattata
4.  thermo_W           – Lavoro:         Integrale cumulativo ∫P dV
5.  thermo_Z           – Z-Score:        Deviazione P da P_attesa (tassi + lag)
6.  thermo_E           – Efficienza:     Lavoro per unità di variazione prezzo
7.  thermo_phase       – Fase:           Stato termodinamico (0=solido…3=supercritico)
8.  thermo_carnot      – Carnot:         Efficienza max estraibile dal ciclo corrente
9.  thermo_lag_signal  – Lag monetario:  Cross-correlazione rates/pressione al best_lag
──────────────────────────────────────────────────────────────────────────────

Utilizzo in get_data():
    from modelli.features.thermodynamic import compute_thermodynamic_features, THERMO_N_FEATURES

    df_thermo = compute_thermodynamic_features(df_main, tickers, rate_col="DGS10")
    df = pd.concat([df_main, df_thermo], axis=1)
    n_context_features += THERMO_N_FEATURES
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy
from sklearn.preprocessing import StandardScaler

# Numero di feature prodotte — importa questo in get_data() per aggiornare il config
THERMO_N_FEATURES = 9

# ──────────────────────────────────────────────────────────────────────────────
# 1. Temperatura (T): Entropia di Shannon su rendimenti rolling
# ──────────────────────────────────────────────────────────────────────────────

def _shannon_temperature(returns: pd.Series, window: int = 20, n_bins: int = 10) -> pd.Series:
    """
    Calcola l'entropia di Shannon su una finestra mobile di rendimenti.
    Idea: ogni rendimento è un "microstato" del sistema.
    Alta entropia → mercato disordinato/caotico (alta temperatura).
    Bassa entropia → mercato ordinato/direzionato (bassa temperatura).
    """
    def _entropy_of_window(arr: np.ndarray) -> float:
        arr = arr[~np.isnan(arr)]
        if len(arr) < 3:
            return np.nan
        counts, _ = np.histogram(arr, bins=n_bins)
        counts = counts + 1e-10  # Laplace smoothing
        probs = counts / counts.sum()
        return float(scipy_entropy(probs))

    return returns.rolling(window, min_periods=window // 2).apply(_entropy_of_window, raw=True)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Pressione (P): Equazione di Van der Waals adattata
# ──────────────────────────────────────────────────────────────────────────────

def _van_der_waals_pressure(
    log_volume: pd.Series,
    temperature: pd.Series,
    returns: pd.Series,
    n_particles: int,
    window: int = 20,
    R: float = 1.0,
) -> pd.Series:
    """
    (P + a*n²/V²)(V - n*b) = n*R*T
    → P = n*R*T / (V - n*b) - a*n²/V²

    V  = log(Volume) rolling mean (volume logaritmico → spazio di trading)
    T  = Temperatura (entropia)
    a  = Autocorrelazione dei rendimenti (forza di herding tra "particelle")
    b  = Volume escluso = volume minimo strutturale rolling
    n  = numero di asset (numero di "particelle")
    """
    V = log_volume.rolling(window, min_periods=5).mean()
    b = log_volume.rolling(window, min_periods=5).min()

    # Autocorrelazione lag-1 su finestra mobile = forza attrattiva inter-asset
    a = returns.rolling(window, min_periods=5).apply(
        lambda x: pd.Series(x).autocorr(lag=1) if len(x) > 2 else 0.0, raw=True
    ).clip(-0.99, 0.99)

    n = float(n_particles)
    Veff = V - n * b
    Veff = Veff.clip(lower=1e-3)   # Evita divisione per zero

    P = (n * R * temperature) / Veff - a * (n ** 2) / (V ** 2 + 1e-8)
    return P


# ──────────────────────────────────────────────────────────────────────────────
# 3. Lavoro (W): Integrale cumulativo ∫P dV (regola del trapezio)
# ──────────────────────────────────────────────────────────────────────────────

def _cumulative_work(pressure: pd.Series, log_volume: pd.Series) -> pd.Series:
    """
    W = Σ (P_t + P_{t-1})/2 * (V_t - V_{t-1})

    Interpretazione:
    dW > 0: il mercato compie lavoro → espansione energetica reale
    dW < 0: il sistema viene compresso → contrazione forzata
    """
    dV = log_volume.diff()
    P_mean = (pressure + pressure.shift(1)) / 2.0
    dW = P_mean * dV
    W_cum = dW.cumsum()
    return W_cum


# ──────────────────────────────────────────────────────────────────────────────
# 4. Z-Score Divergenza: deviazione dalla pressione attesa dai tassi
# ──────────────────────────────────────────────────────────────────────────────

def _find_best_lag(
    pressure: pd.Series, rate_series: pd.Series, max_lag: int = 90
) -> tuple[int, float]:
    """
    Cross-correlazione tra 1/r (costo inverso del denaro) e pressione di mercato.
    Restituisce il lag ottimale e il coefficiente di correlazione al picco.
    Il lag atteso dall'analisi empirica è ~58-72 giorni di borsa.
    """
    inv_rate = (1.0 / (rate_series.reindex(pressure.index).ffill() + 1e-4)).dropna()
    P_aligned = pressure.reindex(inv_rate.index).dropna()
    aligned = pd.concat([P_aligned, inv_rate], axis=1).dropna()
    if len(aligned) < max_lag + 20:
        return 60, 0.0  # fallback al lag empirico documentato

    P_val = aligned.iloc[:, 0].values
    R_val = aligned.iloc[:, 1].values

    correlations = []
    for lag in range(1, min(max_lag + 1, len(P_val) - 5)):
        c = np.corrcoef(P_val[lag:], R_val[:-lag])[0, 1]
        correlations.append(c)

    best_lag = int(np.argmax(np.abs(correlations))) + 1
    best_corr = correlations[best_lag - 1]
    return best_lag, best_corr


def _z_score_divergence(
    pressure: pd.Series,
    rate_series: pd.Series | None,
    window: int = 252,
) -> tuple[pd.Series, pd.Series]:
    """
    Se i tassi sono disponibili:
        Z = (P_attuale - P_attesa_da_tassi[t-lag]) / std_rolling

    Altrimenti fallback: Z = (P - P_rolling_mean) / P_rolling_std

    Restituisce (z_score, lag_signal):
    - z_score: divergenza standardizzata
    - lag_signal: valore di cross-correlazione al best_lag (feature dinamica)
    """
    if rate_series is not None:
        best_lag, lag_corr = _find_best_lag(pressure, rate_series)
        inv_rate = (1.0 / (rate_series.reindex(pressure.index).ffill() + 1e-4))
        P_expected = inv_rate.shift(best_lag)
        # Normalizza P_expected nella stessa scala di P
        scaler = StandardScaler()
        P_norm = pd.Series(
            scaler.fit_transform(pressure.values.reshape(-1, 1)).flatten(),
            index=pressure.index
        )
        Pe_norm = pd.Series(
            scaler.transform(P_expected.bfill().values.reshape(-1, 1)).flatten(),
            index=pressure.index
        )
        residual = P_norm - Pe_norm
    else:
        lag_corr = 0.0
        residual = pressure - pressure.rolling(window, min_periods=20).mean()

    roll_std = residual.rolling(window, min_periods=20).std() + 1e-8
    z = residual / roll_std
    lag_signal = pd.Series(lag_corr, index=pressure.index)
    return z.clip(-5, 5), lag_signal


# ──────────────────────────────────────────────────────────────────────────────
# 5. Oscillatore di Efficienza (Work-Price Index)
# ──────────────────────────────────────────────────────────────────────────────

def _efficiency_oscillator(
    work: pd.Series, price_composite: pd.Series, window: int = 20
) -> pd.Series:
    """
    Efficienza = ΔW_rolling / ΔP_rolling

    Efficienza > 0 (alta dissipazione):
        Il sistema compie molto lavoro ma il prezzo non sale proporzionalmente.
        → Distribuzione ("mani forti" che vendono), attrito estremo.

    Efficienza < 0 o ~0 (movimento fluido):
        L'energia si converte efficacemente in valore.
        → Rally energetico sostenibile.

    Innovazione: usiamo un rapporto logaritmico che cattura asimmetrie
    tra lavoro di espansione e lavoro di compressione.
    """
    dW = work.diff(window).fillna(0)
    dP = price_composite.pct_change(window).fillna(0) + 1e-8
    efficiency = dW / dP
    # Normalizza su finestra rolling per evitare outlier estremi
    e_mean = efficiency.rolling(window * 3, min_periods=window).mean()
    e_std  = efficiency.rolling(window * 3, min_periods=window).std() + 1e-8
    return ((efficiency - e_mean) / e_std).clip(-5, 5)


# ──────────────────────────────────────────────────────────────────────────────
# 6. Phase Detector: stato termodinamico del mercato
# ──────────────────────────────────────────────────────────────────────────────

def _phase_detector(pressure: pd.Series, temperature: pd.Series, window: int = 60) -> pd.Series:
    """
    Classificazione in quattro fasi nel piano (P, T):

    Fase  Val   P         T       Interpretazione
    ─────────────────────────────────────────────────────────────
    Solido  0   Alta      Bassa   Stasi compressa, pre-correzione
    Liquido 1   Media     Media   Trend sano, regime normale
    Gas     2   Bassa     Alta    Euforia/bolla speculativa
    Super   3   Alta      Alta    Regime estremo (crash/spike)

    Codificato come float [0,1] per essere compatibile con il modello.

    Innovazione: le soglie sono adattive (percentile rolling) anziché fisse.
    Questo rende il detector invariante alla scala temporale.
    """
    P_high = pressure.rolling(window, min_periods=20).quantile(0.66)
    T_high = temperature.rolling(window, min_periods=20).quantile(0.66)

    is_P_high = (pressure >= P_high).astype(int)
    is_T_high = (temperature >= T_high).astype(int)

    # 00=Solido, 01=Gas, 10=Liquido (regime normale), 11=Supercritico
    phase = pd.Series(np.nan, index=pressure.index)
    phase[(is_P_high == 1) & (is_T_high == 0)] = 0.0   # Solido
    phase[(is_P_high == 0) & (is_T_high == 0)] = 1.0   # Liquido
    phase[(is_P_high == 0) & (is_T_high == 1)] = 2.0   # Gas
    phase[(is_P_high == 1) & (is_T_high == 1)] = 3.0   # Supercritico
    return phase.fillna(1.0) / 3.0  # Normalizza in [0, 1]


# ──────────────────────────────────────────────────────────────────────────────
# 7. Carnot Efficiency: efficienza massima estraibile dal ciclo corrente
# ──────────────────────────────────────────────────────────────────────────────

def _carnot_efficiency(temperature: pd.Series, window: int = 60) -> pd.Series:
    """
    η_Carnot = 1 - T_cold / T_hot

    dove T_cold = min temperatura nel ciclo corrente (finestra rolling)
         T_hot  = max temperatura nel ciclo corrente (finestra rolling)

    Interpretazione finanziaria:
    Alta η_Carnot → Grande spread entropico → Alta energia potenziale nel sistema
                    → Possibile movimento direzionale forte imminente

    Bassa η_Carnot → Sistema in equilibrio termico → Lateralizzazione/range

    Questa è una feature PREDITTIVA della volatilità futura, non della direzione.
    """
    T_hot  = temperature.rolling(window, min_periods=10).max() + 1e-8
    T_cold = temperature.rolling(window, min_periods=10).min()
    carnot = 1.0 - (T_cold / T_hot)
    return carnot.clip(0.0, 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT PRINCIPALE
# ──────────────────────────────────────────────────────────────────────────────

def compute_thermodynamic_features(
    df: pd.DataFrame,
    tickers: list[str],
    rate_col: str | None = None,
    window_T: int = 20,
    window_P: int = 20,
    window_W: int = 60,
    max_lag: int = 90,
) -> pd.DataFrame:
    """
    Calcola tutte le feature termodinamiche del portafoglio.

    Args:
        df:        DataFrame principale (colonne = tickers + indicatori)
                   !! Le colonne dei ticker devono essere prezzi di chiusura !!
        tickers:   Lista dei ticker (corrispondono alle prime colonne di df)
        rate_col:  Nome della colonna dei tassi d'interesse (es. "DGS10").
                   Se None, lo Z-Score usa un fallback rolling.
        window_T:  Finestra temporale per la Temperatura (default 20gg)
        window_P:  Finestra temporale per la Pressione (default 20gg)
        window_W:  Finestra temporale per l'Oscillatore di Efficienza

    Returns:
        DataFrame con THERMO_N_FEATURES colonne (stesso indice di df).
        Da concatenare DOPO le feature principali per rispettare
        la convenzione main/context del Predictor.
    """
    # ── Prezzi e rendimenti compositi (media portafoglio) ────────────────────
    # Usiamo i prezzi di chiusura dei ticker come feature composita
    ticker_cols = [c for c in tickers if c in df.columns]
    prices = df[ticker_cols]

    # Rendimento log del portafoglio equipesato
    log_returns = np.log(prices / prices.shift(1)).mean(axis=1)

    # Volume composito: cerca colonne "Volume_TICKER" o usa il prezzo come proxy
    vol_cols = [c for c in df.columns if "Volume" in c or "volume" in c]
    if vol_cols:
        raw_vol = df[vol_cols].mean(axis=1).clip(lower=1)
    else:
        # Proxy: varianza dei rendimenti cross-ticker (liquidità implicita)
        raw_vol = prices.pct_change().abs().mean(axis=1).clip(lower=1e-6) * 1e6

    log_volume = np.log(raw_vol + 1.0)

    # ── 1. Temperatura ───────────────────────────────────────────────────────
    T = _shannon_temperature(log_returns, window=window_T)

    # ── 2. Gradiente di Temperatura ─────────────────────────────────────────
    dT = T.diff(5).fillna(0)  # Δ su 5 giorni per ridurre il rumore

    # ── 3. Pressione Van der Waals ───────────────────────────────────────────
    n_particles = len(ticker_cols)
    P = _van_der_waals_pressure(log_volume, T.fillna(0), log_returns, n_particles, window=window_P)

    # ── 4. Lavoro Cumulativo ─────────────────────────────────────────────────
    W = _cumulative_work(P.fillna(0), log_volume)

    # ── 5. Z-Score Divergenza e Lag Signal ──────────────────────────────────
    rate_series = df[rate_col] if rate_col and rate_col in df.columns else None
    Z, lag_signal = _z_score_divergence(P.fillna(0), rate_series)

    # ── 6. Oscillatore di Efficienza ─────────────────────────────────────────
    price_composite = prices.mean(axis=1)
    E = _efficiency_oscillator(W.fillna(0), price_composite, window=window_W)

    # ── 7. Phase Detector ────────────────────────────────────────────────────
    phase = _phase_detector(P.fillna(0), T.fillna(0))

    # ── 8. Carnot Efficiency ─────────────────────────────────────────────────
    carnot = _carnot_efficiency(T.fillna(0))

    # ── Assembla DataFrame ───────────────────────────────────────────────────
    thermo_df = pd.DataFrame({
        "thermo_T":          T,
        "thermo_dT":         dT,
        "thermo_P":          P,
        "thermo_W":          W,
        "thermo_Z":          Z,
        "thermo_E":          E,
        "thermo_phase":      phase,
        "thermo_carnot":     carnot,
        "thermo_lag_signal": lag_signal,
    }, index=df.index)

    # Normalizzazione MinMax per portare tutto in range ragionevole
    # (lo StandardScaler nel Predictor gestirà il resto)
    for col in ["thermo_P", "thermo_W"]:
        col_min = thermo_df[col].quantile(0.01)
        col_max = thermo_df[col].quantile(0.99)
        thermo_df[col] = (thermo_df[col] - col_min) / (col_max - col_min + 1e-8)
        thermo_df[col] = thermo_df[col].clip(-3, 3)

    # Forward-fill e riempimento con 0 per i NaN iniziali
    thermo_df = thermo_df.ffill().fillna(0.0)

    return thermo_df


# ──────────────────────────────────────────────────────────────────────────────
# VISUALIZZAZIONE
# ──────────────────────────────────────────────────────────────────────────────

def plot_thermodynamic(
    df_main: pd.DataFrame,
    df_thermo: pd.DataFrame,
    tickers: list[str],
    title: str = "Analisi Termodinamica di Mercato",
    save_path: str | None = None,
) -> None:
    """
    Dashboard a 4 pannelli:
    ┌─────────────────────────────────────────┐
    │ 1. Prezzo + Fase (colore di sfondo)     │
    │ 2. Pressione VdW + Banda Z-Score        │
    │ 3. Lavoro Cumulativo + Efficienza       │
    │ 4. Temperatura + Carnot + Gradiente dT  │
    └─────────────────────────────────────────┘
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.gridspec as gridspec

    PHASE_COLORS = {
        0.0: "#1a1a2e",   # Solido: blu notte
        1.0: "#0f3460",   # Liquido: blu profondo
        2.0: "#e94560",   # Gas: rosso acceso
        3.0: "#f5a623",   # Supercritico: ambra
    }
    PHASE_LABELS = {0.0: "Solido (stasi)", 1.0: "Liquido (trend)", 2.0: "Gas (euforia)", 3.0: "Supercritico"}

    fig = plt.figure(figsize=(18, 14), facecolor="#0d1117")
    fig.suptitle(title, fontsize=16, color="#e6edf3", fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(4, 1, hspace=0.08, figure=fig)
    axes = [fig.add_subplot(gs[i]) for i in range(4)]

    for ax in axes:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#8b949e", labelsize=8)
        ax.spines["bottom"].set_color("#30363d")
        ax.spines["top"].set_color("#30363d")
        ax.spines["left"].set_color("#30363d")
        ax.spines["right"].set_color("#30363d")
        ax.grid(True, color="#21262d", linewidth=0.5, alpha=0.7)

    idx = df_thermo.index
    phase = df_thermo["thermo_phase"] * 3  # Riporta in [0,3]

    # ── Pannello 1: Prezzo + sfondo di fase ──────────────────────────────────
    ax0 = axes[0]
    ticker_cols = [c for c in tickers if c in df_main.columns]
    price_composite = df_main[ticker_cols].mean(axis=1)

    # Sfondo colorato per fase
    prev_phase = None
    start_idx = idx[0]
    for i, (date, ph) in enumerate(zip(idx, phase)):
        ph_rounded = round(float(ph))
        if ph_rounded != prev_phase:
            if prev_phase is not None:
                color = PHASE_COLORS.get(float(prev_phase), "#21262d")
                ax0.axvspan(start_idx, date, alpha=0.15, color=color, linewidth=0)
            start_idx = date
            prev_phase = ph_rounded
    if prev_phase is not None:
        color = PHASE_COLORS.get(float(prev_phase), "#21262d")
        ax0.axvspan(start_idx, idx[-1], alpha=0.15, color=color, linewidth=0)

    ax0.plot(idx, price_composite, color="#58a6ff", linewidth=1.2, label="Prezzo composito")
    ax0.set_ylabel("Prezzo (media portafoglio)", color="#8b949e", fontsize=9)
    ax0.tick_params(labelbottom=False)

    patches = [mpatches.Patch(color=PHASE_COLORS[k], alpha=0.6, label=PHASE_LABELS[k]) for k in PHASE_COLORS]
    ax0.legend(handles=patches, loc="upper left", fontsize=7, facecolor="#21262d",
               labelcolor="#e6edf3", framealpha=0.8)

    # ── Pannello 2: Pressione + Z-Score ──────────────────────────────────────
    ax1 = axes[1]
    P_norm = df_thermo["thermo_P"]
    Z = df_thermo["thermo_Z"]

    ax1.plot(idx, P_norm, color="#3fb950", linewidth=1.0, label="Pressione VdW (norm.)", alpha=0.9)

    # Colorazione Z-Score: rosso = stress, blu = espansione
    ax1_twin = ax1.twinx()
    ax1_twin.set_facecolor("#161b22")
    z_pos = Z.clip(lower=0)
    z_neg = Z.clip(upper=0)
    ax1_twin.fill_between(idx, 0, z_pos, color="#f85149", alpha=0.3, label="Z > 0 (stress termico)")
    ax1_twin.fill_between(idx, 0, z_neg, color="#58a6ff", alpha=0.3, label="Z < 0 (espansione sana)")
    ax1_twin.set_ylabel("Z-Score divergenza", color="#8b949e", fontsize=8)
    ax1_twin.tick_params(colors="#8b949e", labelsize=7)
    ax1_twin.spines["right"].set_color("#30363d")
    ax1_twin.axhline(0, color="#6e7681", linewidth=0.8, linestyle="--")

    ax1.set_ylabel("Pressione", color="#8b949e", fontsize=9)
    ax1.tick_params(labelbottom=False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=7,
               facecolor="#21262d", labelcolor="#e6edf3", framealpha=0.8)

    # ── Pannello 3: Lavoro + Efficienza ──────────────────────────────────────
    ax2 = axes[2]
    W = df_thermo["thermo_W"]
    E = df_thermo["thermo_E"]

    ax2.plot(idx, W, color="#d2a8ff", linewidth=1.2, label="Lavoro cumulativo W", alpha=0.9)
    ax2.set_ylabel("W (lavoro ∫P dV)", color="#8b949e", fontsize=9)
    ax2.tick_params(labelbottom=False)

    ax2_twin = ax2.twinx()
    ax2_twin.set_facecolor("#161b22")
    e_pos = E.clip(lower=0)
    e_neg = E.clip(upper=0)
    ax2_twin.fill_between(idx, 0, e_pos, color="#f0883e", alpha=0.4, label="Efficienza > 0 (distribuzione)")
    ax2_twin.fill_between(idx, 0, e_neg, color="#7ee787", alpha=0.3, label="Efficienza < 0 (fluido)")
    ax2_twin.axhline(0, color="#6e7681", linewidth=0.8, linestyle="--")
    ax2_twin.set_ylabel("Oscillatore efficienza", color="#8b949e", fontsize=8)
    ax2_twin.tick_params(colors="#8b949e", labelsize=7)
    ax2_twin.spines["right"].set_color("#30363d")

    lines2a, labels2a = ax2.get_legend_handles_labels()
    lines2b, labels2b = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines2a + lines2b, labels2a + labels2b, loc="upper left", fontsize=7,
               facecolor="#21262d", labelcolor="#e6edf3", framealpha=0.8)

    # ── Pannello 4: Temperatura + Carnot + Gradiente ─────────────────────────
    ax3 = axes[3]
    T = df_thermo["thermo_T"]
    dT = df_thermo["thermo_dT"]
    carnot = df_thermo["thermo_carnot"]

    ax3.plot(idx, T, color="#ffa657", linewidth=1.0, label="Temperatura T (Shannon)", alpha=0.9)
    ax3.plot(idx, carnot * T.max(), color="#79c0ff", linewidth=0.8,
             linestyle="--", label=f"Efficienza Carnot (×{T.max():.2f})", alpha=0.7)

    ax3_twin = ax3.twinx()
    ax3_twin.set_facecolor("#161b22")
    dt_pos = dT.clip(lower=0)
    dt_neg = dT.clip(upper=0)
    ax3_twin.fill_between(idx, 0, dt_pos, color="#ff7b72", alpha=0.35, label="dT > 0 (shock entropico)")
    ax3_twin.fill_between(idx, 0, dt_neg, color="#56d364", alpha=0.25, label="dT < 0 (raffreddamento)")
    ax3_twin.axhline(0, color="#6e7681", linewidth=0.8, linestyle="--")
    ax3_twin.set_ylabel("Gradiente dT", color="#8b949e", fontsize=8)
    ax3_twin.tick_params(colors="#8b949e", labelsize=7)
    ax3_twin.spines["right"].set_color("#30363d")

    ax3.set_ylabel("Temperatura / Carnot", color="#8b949e", fontsize=9)
    ax3.set_xlabel("Data", color="#8b949e", fontsize=9)

    lines3a, labels3a = ax3.get_legend_handles_labels()
    lines3b, labels3b = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines3a + lines3b, labels3a + labels3b, loc="upper left", fontsize=7,
               facecolor="#21262d", labelcolor="#e6edf3", framealpha=0.8)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"[OK] Grafico salvato in: {save_path}")

    plt.show()