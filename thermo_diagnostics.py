"""
thermo_diagnostics.py
════════════════════════════════════════════════════════════════════════════════

Script diagnostico che visualizza il modello termodinamico completo
con i 3 nuovi indicatori: CSI, IIR, LDI.

USO:
    # Con dati Alpaca/FRED cached (formato del tuo sistema):
    python thermo_diagnostics.py --csv ./data.csv --ticker SPY --rates GS10

    # Demo con dati Yahoo Finance (nessuna API key richiesta):
    python thermo_diagnostics.py --demo

PRODUCE:
    thermo_analysis.png — dashboard a 6 pannelli
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ─── Import modelli locali ────────────────────────────────────────────────────
try:
    from modelli.thermo_new_features import (
        compute_csi, compute_iir, compute_ldi,
        CSI_SELL_THRESHOLD, CSI_BUY_THRESHOLD, IIR_BREAKOUT_LEVEL
    )
    from modelli.thermo_statistics import ThermoStatisticsEngine
    HAS_LOCAL = True
except ImportError:
    HAS_LOCAL = False

# ─── Fallback: implementazioni embedded se moduli non disponibili ─────────────
def _sanitize(s, fill=0.0):
    return pd.Series(s).replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(fill)

def _rolling_entropy(returns, window=20):
    def _ent(x):
        prob, _ = np.histogram(x, bins=min(window//2, 8), density=True)
        prob = prob[prob > 0]
        return float(-np.sum(prob * np.log(prob + 1e-10)))
    return returns.rolling(window).apply(_ent, raw=True).fillna(0)

def _zscore_rolling(s, window, clip=3.0):
    mu  = s.rolling(window, min_periods=max(window//4, 3)).mean()
    sig = s.rolling(window, min_periods=max(window//4, 3)).std().clip(lower=1e-8)
    return _sanitize(((s - mu) / sig).clip(-clip, clip))

def _compute_vdw_pressure(close, volume, window=20, a=0.1, b=0.01, n=1, kb=1.0):
    """Van der Waals pressure P = nRT/(V-nb) - an²/V²"""
    returns  = close.pct_change().fillna(0)
    entropy  = _rolling_entropy(returns, window)
    V        = np.log1p(volume.rolling(window).mean())
    b_const  = max(V.quantile(0.01) * 0.1, 0.01)
    V_free   = (V - b_const).clip(lower=0.1)
    T        = np.exp(2 * (entropy - np.log(V_free))).clip(upper=1e6)
    P_ideal  = (kb * T) / V_free
    P_corr   = -a * (n**2) / (V**2)
    P        = (P_ideal + P_corr).clip(lower=1e-8)
    dV       = V.diff().fillna(0)
    P_avg    = (P + P.shift(1)) / 2
    W_cum    = (P_avg * dV).fillna(0).cumsum()
    return P, T, entropy, W_cum, V

def _compute_csi(pressure, work, window=20):
    z_p = _zscore_rolling(pressure, window)
    dw  = work.diff(window).fillna(0)
    z_w = _zscore_rolling(dw, window).abs()
    return _sanitize((z_p / (z_w + 0.3)).clip(0, 4.0))

def _compute_iir(entropy_rate, window=40):
    is_inject = (entropy_rate < 0).astype(float)
    return _sanitize(is_inject.rolling(window, min_periods=window//4).mean())

def _compute_ldi(pressure, rates, min_lag=20, max_lag=90, step=5, update=20, smooth=60):
    if rates is None:
        return pd.Series(0.0, index=pressure.index)
    inv_r = _sanitize(1.0 / (rates.ffill().bfill().clip(lower=1e-4) + 1e-5))
    n = len(pressure)
    lags = np.full(n, float((min_lag + max_lag) // 2))
    lags_try = list(range(min_lag, max_lag + 1, step))
    for i in range(max_lag + update, n, update):
        ws = max(0, i - smooth * 3)
        p_w = pressure.iloc[ws:i].values
        r_w = inv_r.iloc[ws:i].values
        best_c, best_l = 0.0, (min_lag + max_lag) // 2
        for lag in lags_try:
            if lag >= len(r_w): break
            r_l  = r_w[:-lag] if lag > 0 else r_w
            p_s  = p_w[lag:]
            n_pt = min(len(r_l), len(p_s))
            if n_pt < 10: continue
            with np.errstate(divide='ignore', invalid='ignore'):
                c = np.corrcoef(p_s[:n_pt], r_l[:n_pt])[0, 1]
            if np.isfinite(c) and abs(c) > abs(best_c):
                best_c, best_l = c, lag
        lags[i:min(i + update, n)] = best_l
    lag_s  = pd.Series(lags, index=pressure.index)
    lag_sm = lag_s.rolling(smooth, min_periods=5).mean()
    delta  = lag_sm.diff(update).fillna(0)
    return _sanitize(_zscore_rolling(delta, smooth, clip=3.0))


# ─── Stile grafico ────────────────────────────────────────────────────────────

DARK_BG    = "#0d1117"
PANEL_BG   = "#161b22"
ACCENT_B   = "#58a6ff"
ACCENT_G   = "#3fb950"
ACCENT_R   = "#f85149"
ACCENT_Y   = "#e3b341"
ACCENT_P   = "#bc8cff"
GRID_C     = "#30363d"
TEXT_C     = "#c9d1d9"
TEXT_FAINT = "#8b949e"

# Custom colormap per il CSI: verde → giallo → rosso
CSI_CMAP = LinearSegmentedColormap.from_list(
    "csi", [(0.0, "#3fb950"), (0.4, "#e3b341"), (0.7, "#f0883e"), (1.0, "#f85149")]
)


def setup_style():
    plt.rcParams.update({
        "figure.facecolor":  DARK_BG,
        "axes.facecolor":    PANEL_BG,
        "axes.edgecolor":    GRID_C,
        "axes.labelcolor":   TEXT_C,
        "axes.titlecolor":   TEXT_C,
        "xtick.color":       TEXT_FAINT,
        "ytick.color":       TEXT_FAINT,
        "grid.color":        GRID_C,
        "grid.alpha":        0.4,
        "text.color":        TEXT_C,
        "font.family":       "monospace",
        "legend.facecolor":  PANEL_BG,
        "legend.edgecolor":  GRID_C,
    })


# ─── Fetch dati ───────────────────────────────────────────────────────────────

def fetch_demo_data(ticker="SPY", period="5y"):
    """Scarica dati da Yahoo Finance per la demo."""
    if not HAS_YF:
        raise RuntimeError("yfinance non installato. Esegui: pip install yfinance")
    print(f"[Demo] Download {ticker} da Yahoo Finance...")
    df = yf.download(ticker, period=period, progress=False)
    if df.empty:
        raise RuntimeError(f"Impossibile scaricare {ticker}")
    # Semplifica MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Close", "Volume"]].dropna()
    print(f"[Demo] {len(df)} barre | {df.index[0].date()} → {df.index[-1].date()}")
    # Simula un proxy dei tassi (GS10 sintetico con trend)
    np.random.seed(42)
    n = len(df)
    base_rate = 2.5 + 2.5 * np.sin(np.linspace(0, 2 * np.pi, n))
    noise_rate = np.cumsum(np.random.normal(0, 0.02, n))
    rates = pd.Series(
        (base_rate + noise_rate).clip(0.25, 6.0),
        index=df.index,
        name="GS10_proxy"
    )
    return df, rates


# ─── Calcolo indicatori ───────────────────────────────────────────────────────

def compute_all_indicators(df, rates, window=20):
    close  = df["Close"]
    volume = df["Volume"]
    n      = len(close)

    print("[Thermo] Calcolo pressione Van der Waals...")
    P, T, S, W, V = _compute_vdw_pressure(close, volume, window=window)

    print("[Thermo] Calcolo dS/dt (Savitzky-Golay)...")
    try:
        from scipy.signal import savgol_filter
        sg_w = max(11 if window >= 15 else 7, 5)
        sg_w = sg_w if sg_w % 2 == 1 else sg_w + 1
        dS_arr = savgol_filter(S.fillna(0).values, window_length=sg_w, polyorder=3, deriv=1)
        dS = pd.Series(dS_arr, index=S.index)
    except Exception:
        dS = S.diff().fillna(0)

    dS_norm = _zscore_rolling(dS, window)

    print("[Thermo] Calcolo CSI (Compressive Stasis Index)...")
    csi = _compute_csi(P, W, window=window)

    print("[Thermo] Calcolo IIR (Information Injection Rate)...")
    iir = _compute_iir(dS_norm, window=window * 2)

    print("[Thermo] Calcolo LDI (Lag Drift Index)...")
    ldi = _compute_ldi(P, rates, min_lag=20, max_lag=90, update=15, smooth=45)

    # Lag ottimale puntuale (per visualizzazione)
    print("[Thermo] Analisi lag monetario...")
    corrs = [P.corr(rates.shift(i)) for i in range(0, 91, 3)]
    best_lag = int(np.argmax(np.abs(corrs))) * 3
    lag_range = range(0, 91, 3)
    print(f"[Thermo] Best lag: {best_lag} barre | ρ = {max(np.abs(corrs)):.3f}")

    # Gibs Free Energy G = H - T·S (H = P * |W|)
    H = P.abs() * (W.abs() + 1e-8)
    G = _sanitize(H - T * S)

    # Normalizzazione per visualizzazione
    def norm01(s):
        lo, hi = s.quantile(0.01), s.quantile(0.99)
        return ((s - lo) / max(hi - lo, 1e-8)).clip(0, 1)

    return {
        "close": close, "volume": volume, "rates": rates,
        "P": P, "T": T, "S": S, "W": W, "V": V,
        "dS": dS_norm,
        "csi": csi, "iir": iir, "ldi": ldi,
        "G": G, "G_norm": norm01(G),
        "P_norm": norm01(P), "W_norm": norm01(W),
        "corrs": corrs, "lag_range": list(lag_range), "best_lag": best_lag,
    }


# ─── PLOTTING ─────────────────────────────────────────────────────────────────

def plot_dashboard(ind: dict, title: str = "SPY", out: str = "thermo_analysis.png"):
    setup_style()

    fig = plt.figure(figsize=(22, 26), facecolor=DARK_BG)
    gs  = gridspec.GridSpec(
        6, 2, figure=fig,
        hspace=0.52, wspace=0.30,
        top=0.94, bottom=0.04,
        left=0.07, right=0.97,
    )

    dates  = ind["close"].index
    close  = ind["close"]
    rates  = ind["rates"]
    P      = ind["P_norm"]
    W      = ind["W_norm"]
    csi    = ind["csi"]
    iir    = ind["iir"]
    ldi    = ind["ldi"]
    dS     = ind["dS"]
    G_norm = ind["G_norm"]

    # ── Titolo principale ────────────────────────────────────────────────────
    fig.text(
        0.5, 0.965,
        f"🌡  ANALISI TERMODINAMICA DEL MERCATO — {title}",
        ha="center", va="center",
        fontsize=17, fontweight="bold", color=TEXT_C,
        fontfamily="monospace",
    )
    fig.text(
        0.5, 0.952,
        "Van der Waals | CSI · IIR · LDI  (Tre indicatori originali)",
        ha="center", va="center",
        fontsize=10, color=TEXT_FAINT,
    )

    # ═══════════════════════════════════════════════════════════════════════
    # PANNELLO 1 — Prezzo + regioni CSI evidenziate
    # ═══════════════════════════════════════════════════════════════════════
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor(PANEL_BG)

    ax1.plot(dates, close, color=ACCENT_B, lw=1.3, label="Prezzo (Close)", zorder=3)
    ax1.set_title("Prezzo + Stasi Compressiva (CSI > soglia → zona rossa)", fontsize=11)
    ax1.set_ylabel("Prezzo ($)", color=TEXT_C)
    ax1.grid(True, alpha=0.25)

    # Evidenzia regioni di stasi compressiva (CSI alto)
    stasis = csi > 1.8
    for i in range(1, len(stasis)):
        if stasis.iloc[i] and not stasis.iloc[i - 1]:
            start = dates[i]
        if not stasis.iloc[i] and stasis.iloc[i - 1]:
            ax1.axvspan(start, dates[i], color=ACCENT_R, alpha=0.15, label="_nolegend_")

    # Marker per IIR alto (information injection events)
    iir_events = iir[iir > IIR_BREAKOUT_LEVEL]
    if len(iir_events) > 0:
        prices_at_events = close.reindex(iir_events.index).dropna()
        ax1.scatter(
            prices_at_events.index,
            prices_at_events.values,
            marker="^", color=ACCENT_G, s=25, zorder=5, alpha=0.7,
            label=f"IIR > {IIR_BREAKOUT_LEVEL:.0%} (info injection)",
        )

    ax1.legend(loc="upper left", fontsize=8.5)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

    # ═══════════════════════════════════════════════════════════════════════
    # PANNELLO 2 — Pressione Van der Waals + Lavoro cumulativo
    # ═══════════════════════════════════════════════════════════════════════
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor(PANEL_BG)

    ax2_r = ax2.twinx()

    ax2.fill_between(dates, P, alpha=0.2, color=ACCENT_P)
    ax2.plot(dates, P, color=ACCENT_P, lw=1.2, label="Pressione P (VdW, norm)")
    ax2_r.plot(dates, W, color=ACCENT_Y, lw=1.2, linestyle="--", label="Lavoro W (cum, norm)")

    ax2.set_title("Pressione Van der Waals + Lavoro ∫P dV", fontsize=10)
    ax2.set_ylabel("P (normaliz.)", color=ACCENT_P, fontsize=9)
    ax2_r.set_ylabel("W (normaliz.)", color=ACCENT_Y, fontsize=9)
    ax2.grid(True, alpha=0.2)

    lines  = ax2.get_lines() + ax2_r.get_lines()
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, fontsize=7.5, loc="upper left")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # ═══════════════════════════════════════════════════════════════════════
    # PANNELLO 3 — Cross-correlation lag monetario
    # ═══════════════════════════════════════════════════════════════════════
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor(PANEL_BG)

    corrs     = ind["corrs"]
    lag_range = ind["lag_range"]
    best_lag  = ind["best_lag"]
    abs_corrs = [abs(c) for c in corrs]

    ax3.bar(
        lag_range, abs_corrs,
        color=[ACCENT_R if l == best_lag else ACCENT_B for l in lag_range],
        alpha=0.8, width=2.0,
    )
    ax3.axvline(best_lag, color=ACCENT_R, lw=2, linestyle="--",
                label=f"Peak lag: {best_lag}gg  ρ={max(abs_corrs):.2f}")
    ax3.axvline(65, color=ACCENT_Y, lw=1.5, linestyle=":", alpha=0.7,
                label="~65gg (Friedman anticipato)")

    ax3.set_title("Cross-Correlation: P(market) ↔ 1/r(tassi) per lag", fontsize=10)
    ax3.set_xlabel("Lag (barre trading)", fontsize=8)
    ax3.set_ylabel("|ρ|", fontsize=9)
    ax3.set_xlim(0, 90)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.2)

    # ═══════════════════════════════════════════════════════════════════════
    # PANNELLO 4 — CSI (Compressive Stasis Index)
    # ═══════════════════════════════════════════════════════════════════════
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.set_facecolor(PANEL_BG)

    csi_vals = csi.values
    csi_norm = np.clip(csi_vals / 4.0, 0, 1)

    # Gradiente colore per CSI
    for i in range(len(dates) - 1):
        ax4.fill_between(
            [dates[i], dates[i + 1]],
            [0, 0], [csi_vals[i], csi_vals[i + 1]],
            color=CSI_CMAP(csi_norm[i]), alpha=0.8,
        )

    ax4.axhline(CSI_SELL_THRESHOLD, color=ACCENT_R, lw=1.5, linestyle="--",
                label=f"Soglia SELL ({CSI_SELL_THRESHOLD})")
    ax4.axhline(CSI_BUY_THRESHOLD, color=ACCENT_G, lw=1.5, linestyle="--",
                label=f"Soglia BUY ({CSI_BUY_THRESHOLD})")

    ax4.set_title("★ CSI — Compressive Stasis Index [NUOVO]", fontsize=10)
    ax4.set_ylabel("CSI  =  Z(P) / |Z(ΔW)|", fontsize=8)
    ax4.set_ylim(0, 4.5)
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.2)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Annotazione formula
    ax4.text(
        0.02, 0.92, "CSI > 1.8 → stasi compressiva\n(P↑ ma W piatto → correzione imminente)",
        transform=ax4.transAxes, fontsize=7.5, color=ACCENT_R,
        va="top", bbox=dict(boxstyle="round,pad=0.3", fc=DARK_BG, alpha=0.7),
    )

    # ═══════════════════════════════════════════════════════════════════════
    # PANNELLO 5 — IIR (Information Injection Rate)
    # ═══════════════════════════════════════════════════════════════════════
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.set_facecolor(PANEL_BG)

    ax5.fill_between(dates, iir, alpha=0.3, color=ACCENT_G)
    ax5.plot(dates, iir, color=ACCENT_G, lw=1.2)
    ax5.axhline(IIR_BREAKOUT_LEVEL, color=ACCENT_Y, lw=1.5, linestyle="--",
                label=f"Soglia breakout ({IIR_BREAKOUT_LEVEL:.0%})")

    ax5.set_title("★ IIR — Information Injection Rate [NUOVO]", fontsize=10)
    ax5.set_ylabel("Freq(dS/dt < 0)  ∈ [0,1]", fontsize=8)
    ax5.set_ylim(0, 1.0)
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.2)
    ax5.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax5.text(
        0.02, 0.92, "IIR alto → violazioni locali 2° legge\n→ info esterna in ingresso → breakout",
        transform=ax5.transAxes, fontsize=7.5, color=ACCENT_G,
        va="top", bbox=dict(boxstyle="round,pad=0.3", fc=DARK_BG, alpha=0.7),
    )

    # ═══════════════════════════════════════════════════════════════════════
    # PANNELLO 6 — LDI (Lag Drift Index)
    # ═══════════════════════════════════════════════════════════════════════
    ax6 = fig.add_subplot(gs[3, 0])
    ax6.set_facecolor(PANEL_BG)

    ldi_pos = ldi.clip(lower=0)
    ldi_neg = ldi.clip(upper=0)

    ax6.fill_between(dates, ldi_pos, alpha=0.4, color=ACCENT_R, label="Lag ↑ (disconnessione)")
    ax6.fill_between(dates, ldi_neg, alpha=0.4, color=ACCENT_G, label="Lag ↓ (ipersensibilità)")
    ax6.plot(dates, ldi, color=TEXT_FAINT, lw=0.8, alpha=0.7)
    ax6.axhline(0, color=GRID_C, lw=1.0)
    ax6.axhline(1.0, color=ACCENT_R, lw=1.0, linestyle="--", alpha=0.6)
    ax6.axhline(-1.0, color=ACCENT_G, lw=1.0, linestyle="--", alpha=0.6)

    ax6.set_title("★ LDI — Lag Drift Index [NUOVO]", fontsize=10)
    ax6.set_ylabel("Δ(lag ottimale) / σ", fontsize=8)
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.2)
    ax6.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax6.text(
        0.02, 0.92, "LDI > 0 → mercato si disconnette dalla FED\nLDI < 0 → mercato iperrattivo alla FED",
        transform=ax6.transAxes, fontsize=7.5, color=TEXT_FAINT,
        va="top", bbox=dict(boxstyle="round,pad=0.3", fc=DARK_BG, alpha=0.7),
    )

    # ═══════════════════════════════════════════════════════════════════════
    # PANNELLO 7 — P-V Phase Space (DIAGRAMMA DI FASE — NUOVO)
    # ═══════════════════════════════════════════════════════════════════════
    ax7 = fig.add_subplot(gs[3, 1])
    ax7.set_facecolor(PANEL_BG)

    # Downsampling per leggibilità
    ds = max(len(P) // 600, 1)
    P_ds  = P.iloc[::ds]
    W_ds  = W.iloc[::ds]
    csi_c = csi.iloc[::ds].values / 4.0

    sc = ax7.scatter(
        W_ds, P_ds,
        c=np.clip(csi_c, 0, 1),
        cmap=CSI_CMAP, s=6, alpha=0.7, zorder=3,
    )
    plt.colorbar(sc, ax=ax7, label="CSI (intensità stasi)", fraction=0.04, pad=0.02)

    ax7.set_title("Diagramma di Fase P-W (spazio termodinamico)", fontsize=10)
    ax7.set_xlabel("Lavoro W (cum, norm)", fontsize=8)
    ax7.set_ylabel("Pressione P (VdW, norm)", fontsize=8)
    ax7.grid(True, alpha=0.2)

    # Annotazioni quadranti
    ax7.text(0.05, 0.92, "Alta P\nBasso W\n→ STASI", transform=ax7.transAxes,
             fontsize=7, color=ACCENT_R, va="top")
    ax7.text(0.65, 0.08, "Alta W\nBassa P\n→ ESPANSIONE", transform=ax7.transAxes,
             fontsize=7, color=ACCENT_G, va="bottom")

    # ═══════════════════════════════════════════════════════════════════════
    # PANNELLO 8 — Segnale combinato CSI + IIR (matrice di decisione)
    # ═══════════════════════════════════════════════════════════════════════
    ax8 = fig.add_subplot(gs[4, :])
    ax8.set_facecolor(PANEL_BG)

    # Composito: BUY/SELL/HOLD scores da CSI e IIR
    buy_score  = np.clip(1.0 - csi.values / CSI_SELL_THRESHOLD, 0, 1) * iir.values
    sell_score = np.clip(csi.values / CSI_SELL_THRESHOLD - 1.0, 0, 1) * (1 + iir.values)

    ax8.fill_between(dates, buy_score,  alpha=0.5, color=ACCENT_G, label="Score BUY  (CSI basso + IIR alto)")
    ax8.fill_between(dates, -sell_score, alpha=0.5, color=ACCENT_R, label="Score SELL (CSI alto + IIR alto)")
    ax8.axhline(0, color=GRID_C, lw=1.0)

    ax8.set_title("Segnale Composito  BUY / SELL  (CSI × IIR) — Ingresso nel DDPG reward shaping", fontsize=10)
    ax8.set_ylabel("Score", fontsize=9)
    ax8.legend(loc="upper right", fontsize=8.5)
    ax8.grid(True, alpha=0.2)
    ax8.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax8.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

    # ═══════════════════════════════════════════════════════════════════════
    # PANNELLO 9 — Energia Libera di Gibbs (G) vs drawdown prezzo
    # ═══════════════════════════════════════════════════════════════════════
    ax9 = fig.add_subplot(gs[5, :])
    ax9.set_facecolor(PANEL_BG)

    # Drawdown del prezzo
    roll_max = close.expanding().max()
    drawdown = (close - roll_max) / roll_max

    ax9_r = ax9.twinx()
    ax9.fill_between(dates, drawdown, 0, alpha=0.35, color=ACCENT_R, label="Drawdown prezzo")
    ax9.plot(dates, drawdown, color=ACCENT_R, lw=0.8, alpha=0.6)
    ax9_r.plot(dates, G_norm, color=ACCENT_P, lw=1.4, label="Energia Libera Gibbs G (norm)")

    ax9.set_title("Energia Libera di Gibbs G = H − T·S   vs   Drawdown (leading indicator?)", fontsize=10)
    ax9.set_ylabel("Drawdown (%)", color=ACCENT_R, fontsize=9)
    ax9_r.set_ylabel("G (normaliz.)", color=ACCENT_P, fontsize=9)

    lines  = ax9.get_lines()[:1] + ax9_r.get_lines()
    labels = [l.get_label() for l in lines]
    ax9.legend(lines, labels, fontsize=8.5, loc="lower right")
    ax9.grid(True, alpha=0.2)
    ax9.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax9.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

    # ── Firma ─────────────────────────────────────────────────────────────
    fig.text(
        0.99, 0.005,
        "thermo_diagnostics.py | FilippoArcari/mercati | v3.0",
        ha="right", va="bottom", fontsize=7, color=TEXT_FAINT, style="italic",
    )

    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    print(f"\n[Plot] Salvato: {out}")
    return out


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Thermodynamic market diagnostics")
    parser.add_argument("--demo",   action="store_true", help="Demo con Yahoo Finance (SPY)")
    parser.add_argument("--csv",    default=None,  help="Path al CSV dati cached")
    parser.add_argument("--ticker", default="SPY", help="Ticker da analizzare")
    parser.add_argument("--rates",  default="GS10", help="Colonna tassi nel CSV")
    parser.add_argument("--out",    default="thermo_analysis.png", help="Output PNG")
    parser.add_argument("--window", default=20, type=int, help="Finestra rolling base")
    args = parser.parse_args()

    if args.demo or args.csv is None:
        df, rates = fetch_demo_data(args.ticker)
    else:
        print(f"[Data] Carico {args.csv}...")
        raw = pd.read_csv(args.csv, index_col=0, parse_dates=True)
        # Adatta al formato del tuo sistema (colonne: ticker, ticker_Volume, GS10...)
        close_col  = args.ticker if args.ticker in raw.columns else "Close"
        volume_col = f"{args.ticker}_Volume" if f"{args.ticker}_Volume" in raw.columns else "Volume"
        df = raw[[close_col, volume_col]].rename(
            columns={close_col: "Close", volume_col: "Volume"}
        ).dropna()
        rates = raw[args.rates].reindex(df.index).ffill().bfill() if args.rates in raw.columns else None

    indicators = compute_all_indicators(df, rates, window=args.window)
    out_path = plot_dashboard(indicators, title=args.ticker, out=args.out)
    print(f"\n✅ Analisi completata → {out_path}")


if __name__ == "__main__":
    main()