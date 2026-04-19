"""
modelli/trade.py — Addestramento e valutazione dell'agente DDPG

Fix applicati rispetto alla versione precedente
──────────────────────────────────────────────
FIX A  stress_fn costruita ma mai passata a train_ddpg_normalized
       rimossa da entrambi i siti.

FIX B  norm_path/keyword args uniformati.

FIX C  es_patience aggiunto a train_ddpg_normalized.

FIX D  ckpt_path=fold_best nel loop walk-forward (senza, il best
       checkpoint non veniva mai salvato → agent.load era un no-op).

FIX E  [NUOVO] lambda_inaction default abbassato 0.2 → 0.005.
       Con 0.2 l'agente imparava che stare fermo era più costoso di
       tradare, causando overtrading sistematico (266 trade / 472 barre).

FIX F  [NUOVO] reward_clip default abbassato 10.0 → 0.05.
       Con clip=10 i gradienti degli episodi catastrofici destabilizzano
       il critic. 0.05 equivale a max ±5% per step, già ampio per intraday.

FIX G  [NUOVO] transaction_cost propagato esplicitamente al TradingEnv
       via _env_reward_kwargs per garantire coerenza con alpaca_live.py.

FIX H  [NUOVO] Integrazione thermo_vdw: compute_vdw_block chiamato dopo
       builder.build() e concatenato al DataFrame termodinamico.
       La funzione _build_thermo() centralizza la logica per train/test/fold.
       NOTA: questa aggiunge 8 colonne → state_dim cambia → i checkpoint
       DDPG esistenti sono incompatibili. Cancellare ddpg_*.pth e
       normalizer_*.npz prima di riaddestrare.

FIX C1 [NUOVO v6.0] ruin_stop_pct / ruin_penalty propagati a TradingEnv
       via _env_reward_kwargs. Hard stop-loss a -70% del capitale che
       termina l'episodio prima dell'overflow numpy — causa radice dei
       return -176992% / -1.16 miliardi% nei fold del walk-forward.

FIX C4 [NUOVO v6.0] holding_urgency_* propagati a TradingEnv. Penalità
       crescente linearmente oltre holding_urgency_start step, che risolve
       il "18 BUY, 0 SELL" osservato nell'alpaca_replay.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from omegaconf import OmegaConf

from modelli.pred import Pred
from modelli.ddpg import DDPGAgent
from modelli.trading_env import TradingEnv
from modelli.obs_normalizer import ObsNormalizer, train_ddpg_normalized
from modelli.device_setup import get_device, get_map_location

BATCH_SIZE = 256


# ─── naming checkpoint ────────────────────────────────────────────────────────

def _ckpt(cfg, name: str) -> str:
    freq = cfg.frequency.interval
    ws   = cfg.prediction.window_size
    stem, ext = os.path.splitext(name)
    tag = f"_{freq}_w{ws}" if stem == "pred" else f"_{freq}"
    return os.path.join(cfg.paths.checkpoint_dir, f"{stem}{tag}{ext}")


def _predict_batched(predictor, X: torch.Tensor, batch_size: int = 256) -> np.ndarray:
    results = []
    device  = next(predictor.parameters()).device
    predictor.eval()
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = X[i : i + batch_size].to(device)
            out   = predictor(batch)
            if out.dim() == 3:
                out = out[:, 0, :]
            results.append(out.cpu().numpy())
    return np.concatenate(results, axis=0)


# ─── helper thermo + VdW ─────────────────────────────────────────────────────

def _build_thermo(
    builder,
    raw_df: pd.DataFrame,
    tickers: list[str],
) -> pd.DataFrame | None:
    """
    [FIX H] Chiama ThermoStateBuilder.build() e concatena le feature
    Van der Waals (compute_vdw_block) se il modulo è disponibile.

    La funzione è centralizzata qui per garantire che train, test,
    walk-forward e alpaca_replay abbiano lo stesso state_dim.
    """
    thermo = builder.build(raw_df, tickers)

    try:
        from modelli.thermo_vdw import compute_vdw_block

        # Cerca la colonna prezzo e volume più probabile nel raw_df
        price_candidates  = [c for c in raw_df.columns
                             if any(k in c.lower() for k in ("close", "price", "adj"))]
        volume_candidates = [c for c in raw_df.columns if "vol" in c.lower()]

        price_col  = price_candidates[0]  if price_candidates  else raw_df.columns[0]
        volume_col = volume_candidates[0] if volume_candidates else None

        # Se i ticker sono nel df usa il primo ticker come proxy di prezzo
        if tickers and tickers[0] in raw_df.columns:
            price_col = tickers[0]

        vdw = compute_vdw_block(
            df=raw_df,
            price_col=price_col,
            volume_col=volume_col,
            window=20,
            lag_max=60,
        )

        if thermo is not None and not thermo.empty:
            vdw_aligned = vdw.reindex(thermo.index)
            thermo = pd.concat([thermo, vdw_aligned], axis=1)
        else:
            thermo = vdw

    except ImportError:
        pass  # thermo_vdw non ancora installato — continua senza
    except Exception as e:
        print(f"  [thermo_vdw] Integrazione VdW skippata: {e}")

    return thermo


# ─── helpers privati ──────────────────────────────────────────────────────────

def _load_predictor(cfg, num_features, checkpoint):
    saved_cfg  = checkpoint.get("config", {})
    pred_steps = (
        saved_cfg.get("model", {}).get("prediction_steps")
        or cfg.model.prediction_steps
    )
    saved_features = checkpoint.get("num_features", num_features)

    predictor = Pred(
        num_features     = saved_features,
        window_size      = cfg.prediction.window_size,
        dimension        = list(cfg.model.dimensions),
        dilations        = list(cfg.model.dilations),
        kernel_size      = cfg.model.kernel_size,
        activation       = cfg.model.activation,
        prediction_steps = pred_steps,
        training_cfg     = OmegaConf.to_container(cfg.training, resolve=True),
    )
    predictor.load_state_dict(checkpoint["model_state_dict"], strict=False)
    predictor.eval()

    from modelli.device_setup import get_device
    return predictor.to(get_device())


def _build_price_dfs(pred_scaled, real_targets, scaler, dates, all_columns):
    real = scaler.inverse_transform(
        real_targets.numpy() if hasattr(real_targets, "numpy") else real_targets
    )
    pred = scaler.inverse_transform(pred_scaled)
    return (
        pd.DataFrame(real, index=dates, columns=all_columns),
        pd.DataFrame(pred, index=dates, columns=all_columns),
    )


def _build_portfolio_daily(env: TradingEnv, tickers: list[str]) -> pd.DataFrame:
    ph = env.portfolio_history()
    if len(ph) < 2:
        return pd.DataFrame()

    idx   = env.df.index[: len(ph)]
    daily = pd.DataFrame(index=idx)
    daily["portfolio_value"] = ph[: len(idx)]

    initial = env.initial_capital
    daily["daily_return_pct"]      = daily["portfolio_value"].pct_change().fillna(0) * 100
    daily["cumulative_return_pct"] = (daily["portfolio_value"] / initial - 1) * 100
    peak                           = daily["portfolio_value"].cummax()
    daily["drawdown_pct"]          = (daily["portfolio_value"] / peak - 1) * 100
    return daily


def _split_raw(df_raw: pd.DataFrame, train_index: pd.Index, test_index: pd.Index):
    return df_raw.reindex(train_index), df_raw.reindex(test_index)


# ─── grafici ──────────────────────────────────────────────────────────────────

def _plot_portfolio_daily(daily, tickers, initial_capital, results_dir, freq):
    if daily.empty:
        return

    dates   = daily.index
    usd_fmt = FuncFormatter(lambda y, _: f"${y:,.0f}")
    pct_fmt = FuncFormatter(lambda y, _: f"{y:+.1f}%")

    fig = plt.figure(figsize=(15, 10))
    gs  = gridspec.GridSpec(3, 1, figure=fig, hspace=0.45,
                            height_ratios=[2, 1.2, 1])

    ax1 = fig.add_subplot(gs[0])
    pv  = daily["portfolio_value"]
    ax1.plot(dates, pv, color="#2c7bb6", linewidth=1.8, zorder=3)
    ax1.fill_between(dates, initial_capital, pv,
                     where=pv >= initial_capital, color="#2c7bb6", alpha=0.18, label="Profitto")
    ax1.fill_between(dates, initial_capital, pv,
                     where=pv < initial_capital,  color="#d7191c", alpha=0.18, label="Perdita")
    ax1.axhline(initial_capital, color="gray", linestyle="--", linewidth=0.9,
                label=f"Capitale iniziale ${initial_capital:,.0f}")
    final_v = pv.iloc[-1]
    ret_pct = daily["cumulative_return_pct"].iloc[-1]
    color_r = "#2c7bb6" if ret_pct >= 0 else "#d7191c"
    ax1.annotate(f"${final_v:,.2f}  ({ret_pct:+.1f}%)",
                 xy=(dates[-1], final_v), xytext=(-90, 12),
                 textcoords="offset points", fontsize=9,
                 color=color_r, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=color_r, lw=0.8))
    ax1.set_title(f"Andamento del portafoglio — test set [{freq}]", fontsize=12)
    ax1.set_ylabel("USD")
    ax1.yaxis.set_major_formatter(usd_fmt)
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(True, alpha=0.25)

    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    cr  = daily["cumulative_return_pct"]
    ax2.plot(dates, cr, color="#1a9641", linewidth=1.4)
    ax2.fill_between(dates, 0, cr, where=cr >= 0, color="#1a9641", alpha=0.18)
    ax2.fill_between(dates, 0, cr, where=cr < 0,  color="#d7191c", alpha=0.18)
    ax2.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax2.set_title("Rendimento cumulativo %", fontsize=10)
    ax2.set_ylabel("%")
    ax2.yaxis.set_major_formatter(pct_fmt)
    ax2.grid(True, alpha=0.25)

    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    dd  = daily["drawdown_pct"]
    ax3.fill_between(dates, dd, 0, color="#d7191c", alpha=0.45)
    ax3.plot(dates, dd, color="#d7191c", linewidth=0.8)
    ax3.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    max_dd = dd.min()
    ax3.annotate(f"Max DD: {max_dd:.1f}%", xy=(dd.idxmin(), max_dd),
                 xytext=(10, -18), textcoords="offset points",
                 fontsize=8, color="#d7191c",
                 arrowprops=dict(arrowstyle="->", color="#d7191c", lw=0.8))
    ax3.set_title("Drawdown %", fontsize=10)
    ax3.set_ylabel("%")
    ax3.yaxis.set_major_formatter(pct_fmt)
    ax3.grid(True, alpha=0.25)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)

    out = os.path.join(results_dir, f"portfolio_daily_{freq}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  portfolio_daily_{freq}.png")


def _plot_learning(history, results_dir, freq):
    fig, axes = plt.subplots(4, 1, figsize=(13, 12), sharex=True)
    episodes  = list(range(1, len(history) + 1))

    rewards = [h["total_reward"] for h in history]
    axes[0].plot(episodes, rewards, color="darkorange", alpha=0.5, linewidth=1)
    if len(rewards) >= 10:
        ma = pd.Series(rewards).rolling(10).mean()
        axes[0].plot(episodes, ma, color="red", linewidth=2, label="MA 10 ep.")
        axes[0].legend(fontsize=8)
    axes[0].set_title(f"Reward per episodio [{freq}]")
    axes[0].set_ylabel("Reward")
    axes[0].grid(True, alpha=0.25)

    # Solo Sharpe validi (no ruin, no piatti) — distingue segnale reale da garbage
    sharpes = [h["sharpe"] for h in history]
    is_ruin = [h.get("ruin", False) for h in history]
    valid_eps   = [e for e, r in zip(episodes, is_ruin) if not r]
    valid_sharp = [s for s, r in zip(sharpes, is_ruin) if not r]
    ruin_eps    = [e for e, r in zip(episodes, is_ruin) if r]
    axes[1].plot(valid_eps, valid_sharp, color="green", linewidth=1.5, label="Sharpe (valido)")
    if ruin_eps:
        axes[1].scatter(ruin_eps, [0.0] * len(ruin_eps),
                        color="red", marker="x", s=30, zorder=5, label="Ruin (escluso da ES)")
    axes[1].axhline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[1].axhline(0.5, color="green", linestyle=":", linewidth=0.8, alpha=0.6, label="Target 0.5")
    axes[1].set_title("Sharpe ratio per episodio (solo validi)")
    axes[1].set_ylabel("Sharpe")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.25)

    # Ruin rate rolling — indica se l'agente sta imparando a non andare in ruin
    ruin_int = [1 if r else 0 for r in is_ruin]
    ruin_rate = pd.Series(ruin_int).rolling(20, min_periods=1).mean() * 100
    axes[2].fill_between(episodes, ruin_rate, color="red", alpha=0.35, label="Ruin rate (MA 20 ep.)")
    axes[2].plot(episodes, ruin_rate, color="darkred", linewidth=1.0)
    axes[2].axhline(10, color="orange", linestyle=":", linewidth=0.8, label="Soglia 10%")
    axes[2].set_title("Ruin rate % (rolling 20 ep.) — deve scendere nel tempo")
    axes[2].set_ylabel("Ruin %")
    axes[2].set_ylim(-5, 105)
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.25)

    sigmas = [h["noise_sigma"] for h in history]
    axes[3].plot(episodes, sigmas, color="purple", linewidth=1.5)
    axes[3].set_title("Decadimento rumore σ")
    axes[3].set_xlabel("Episodio")
    axes[3].set_ylabel("σ")
    axes[3].grid(True, alpha=0.25)

    plt.tight_layout()
    out = os.path.join(results_dir, f"ddpg_learning_{freq}.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  ddpg_learning_{freq}.png")


def _plot_trades(env, results_dir, freq):
    trades = env._trades
    if not trades:
        return

    trade_df = pd.DataFrame(trades)
    if "ticker" not in trade_df.columns or "type" not in trade_df.columns:
        return

    ac = (trade_df[trade_df["type"] != "HOLD"]
          .groupby(["ticker", "type"]).size().unstack(fill_value=0))
    ac["total"] = ac.sum(axis=1)
    top10 = ac.nlargest(min(10, len(ac)), "total").drop(columns="total")

    fig, ax = plt.subplots(figsize=(13, 5))
    colors = []
    if "BUY"  in top10.columns: colors.append("#2ecc71")
    if "SELL" in top10.columns: colors.append("#e74c3c")
    top10.plot(kind="bar", ax=ax, color=colors)
    ax.set_title(f"Operazioni BUY/SELL — top 10 ticker [{freq}]")
    ax.set_ylabel("N. operazioni")
    ax.legend(title="Azione")
    ax.grid(True, alpha=0.25, axis="y")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    out = os.path.join(results_dir, f"action_distribution_{freq}.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  action_distribution_{freq}.png")


def _plot_walk_forward_summary(report, results_dir: str, freq: str) -> None:
    if not report.folds:
        return

    folds      = [f.fold             for f in report.folds]
    sharpes    = [f.sharpe           for f in report.folds]
    returns    = [f.total_return_pct for f in report.folds]
    maxdds     = [f.max_drawdown     for f in report.folds]
    thermo_ok  = [f.thermo_sell_ok_pct for f in report.folds]

    best_sharpe  = max(sharpes)
    worst_sharpe = min(sharpes)

    fig, axes = plt.subplots(4, 1, figsize=(12, 14),
                             facecolor="#0d1117", gridspec_kw={"hspace": 0.55})
    fig.suptitle(f"Walk-Forward Summary — {len(folds)} fold | freq={freq}",
                 fontsize=13, color="white", fontweight="bold", y=0.98)

    labels = [f"F{f}" for f in folds]
    tags   = []
    for s in sharpes:
        if s == best_sharpe:    tags.append("★")
        elif s == worst_sharpe: tags.append("✗")
        else:                   tags.append("")

    def _bar_panel(ax, values, ylabel, title, threshold=None,
                   pos_color="#4fc3f7", neg_color="#ef5350"):
        colors = [pos_color if v >= 0 else neg_color for v in values]
        bars = ax.bar(labels, values, color=colors, edgecolor="#30363d", linewidth=0.8)
        if threshold is not None:
            ax.axhline(threshold, color="#ffd54f", lw=1.2, linestyle="--",
                       label=f"soglia {threshold}")
            ax.legend(fontsize=7, labelcolor="white")
        ax.axhline(0, color="#555", lw=0.6)
        for bar, val, tag in zip(bars, values, tags):
            offset = max(abs(v) for v in values) * 0.04 if values else 0.1
            ypos = val + offset if val >= 0 else val - offset * 2
            ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                    f"{val:.2f}{tag}", ha="center", va="bottom",
                    fontsize=8, color="white", fontweight="bold")
        ax.set_ylabel(ylabel, color="white", fontsize=9)
        ax.set_title(title, color="gray", fontsize=8, loc="left")
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="gray", labelsize=8)
        ax.spines[:].set_color("#30363d")
        mean_val = sum(values) / len(values)
        ax.axhline(mean_val, color="#80cbc4", lw=1.0, linestyle=":",
                   alpha=0.8, label=f"media {mean_val:.2f}")
        ax.legend(fontsize=7, labelcolor="white")

    _bar_panel(axes[0], sharpes, "Sharpe",
               "Sharpe per fold  —  ★ best  |  ✗ worst  |  linea tratteggiata = soglia 0.5",
               threshold=0.5)
    _bar_panel(axes[1], returns, "Return (%)",
               "Rendimento % per fold sul test set", threshold=0.0)
    _bar_panel(axes[2], maxdds, "Max Drawdown",
               "Max Drawdown per fold  (peggiore = più negativo)",
               threshold=-0.20, pos_color="#ef5350", neg_color="#4fc3f7")

    thm_colors = ["#81c784" if v > 50 else "#ffb74d" if v > 25 else "#ef5350"
                  for v in thermo_ok]
    bars = axes[3].bar(labels, thermo_ok, color=thm_colors,
                       edgecolor="#30363d", linewidth=0.8)
    axes[3].axhline(50, color="#ffd54f", lw=1.2, linestyle="--", label="soglia 50%")
    for bar, val, tag in zip(bars, thermo_ok, tags):
        axes[3].text(bar.get_x() + bar.get_width() / 2,
                     val + max(thermo_ok) * 0.03 + 0.5,
                     f"{val:.1f}%{tag}", ha="center", va="bottom",
                     fontsize=8, color="white", fontweight="bold")
    axes[3].set_ylabel("Thm✅ SELL (%)", color="white", fontsize=9)
    axes[3].set_title(
        "Thm✅% — % di SELL effettuate in zona stress termodinamico",
        color="gray", fontsize=8, loc="left")
    axes[3].set_facecolor("#161b22")
    axes[3].tick_params(colors="gray", labelsize=8)
    axes[3].spines[:].set_color("#30363d")
    axes[3].legend(fontsize=7, labelcolor="white")

    stats_txt = (
        f"Sharpe medio: {report.mean_sharpe:.3f}\n"
        f"Std Sharpe:   {report.std_sharpe:.3f}\n"
        f"Return medio: {report.mean_return:.2f}%\n"
        f"MaxDD medio:  {report.mean_max_drawdown:.3f}\n"
        f"Thm✅ medio:  {report.mean_thermo_sell_ok:.1f}%\n"
        f"Pronto:       {'SÌ ✅' if report.production_ready else 'NO ❌'}"
    )
    fig.text(0.76, 0.935, stats_txt, fontsize=8, color="white",
             va="top", ha="left",
             bbox=dict(facecolor="#1c2128", edgecolor="#30363d",
                       boxstyle="round,pad=0.5"))

    out = os.path.join(results_dir, f"walk_forward_summary_{freq}.png")
    plt.savefig(out, dpi=140, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    print(f"  Walk-forward summary plot: {os.path.basename(out)}")


def _plot_fold_portfolio(
    env_test, fold_n, sharpe, ret_pct, max_dd, th_test, results_dir, freq
) -> None:
    portfolio = env_test.portfolio_history()
    if len(portfolio) < 2:
        return

    trades  = env_test._trades
    n_steps = len(portfolio)
    steps   = np.arange(n_steps)

    has_thermo = th_test is not None and not th_test.empty
    has_stats  = has_thermo and any(
        c in th_test.columns for c in ["Thm_CarnotEff", "Thm_EntropyProd", "Thm_Quality"]
    )
    n_panels = 1 + int(has_thermo) + int(has_stats)

    fig_h = 5 + 3 * (n_panels - 1)
    fig, axes = plt.subplots(
        n_panels, 1, figsize=(15, fig_h), facecolor="#0d1117",
        gridspec_kw={"height_ratios": [3] + [2] * (n_panels - 1), "hspace": 0.35},
    )
    if n_panels == 1:
        axes = [axes]

    ax = axes[0]
    ax.set_facecolor("#161b22")
    ax.plot(steps, portfolio, color="#4fc3f7", linewidth=1.4, zorder=3)
    peak = np.maximum.accumulate(portfolio)
    dd   = (portfolio - peak) / (peak + 1e-8)
    ax.fill_between(steps, portfolio, peak, where=(dd < 0),
                    color="tomato", alpha=0.15, zorder=2)
    for tr in trades:
        s = tr["step"]
        if s >= n_steps:
            continue
        ax.axvline(s, color="#2ecc71" if tr["type"] == "BUY" else "#e74c3c",
                   alpha=0.25, linewidth=0.7)
    ax.set_title(
        f"Fold {fold_n} — Equity Curve  |  Sharpe: {sharpe:.3f}  |  "
        f"Return: {ret_pct:+.2f}%  |  MaxDD: {max_dd:.3f}",
        color="white", fontsize=10, fontweight="bold",
    )
    ax.set_ylabel("Portfolio ($)", color="white", fontsize=9)
    ax.tick_params(colors="gray", labelsize=7)
    ax.spines[:].set_color("#30363d")
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], color="#2ecc71", lw=1.5, alpha=0.6, label="BUY"),
        Line2D([0], [0], color="#e74c3c", lw=1.5, alpha=0.6, label="SELL"),
    ], fontsize=7, labelcolor="white", loc="upper left")

    if has_thermo and n_panels >= 2:
        ax2 = axes[1]
        ax2.set_facecolor("#161b22")
        th_arr = th_test.reset_index(drop=True)
        lim    = min(n_steps, len(th_arr))
        t_idx  = np.arange(lim)
        if "Thm_Stress" in th_arr.columns:
            stress = th_arr["Thm_Stress"].values[:lim]
            ax2.plot(t_idx, stress, color="#f06292", lw=1.2, label="Thm_Stress")
            ax2.fill_between(t_idx, stress, 0,
                             where=(stress > 0.5), color="#f06292", alpha=0.15)
        if "Thm_Efficiency" in th_arr.columns:
            ax2.plot(t_idx, th_arr["Thm_Efficiency"].values[:lim],
                     color="#ffb74d", lw=1.2, label="Thm_Efficiency")
        ax2.axhline(0.5, color="#555", lw=0.7, linestyle="--")
        ax2.axhline(0, color="#444", lw=0.5)
        ax2.set_ylabel("Stress / Efficiency", color="white", fontsize=9)
        ax2.set_title("Thm_Stress + Thm_Efficiency", color="gray", fontsize=8, loc="left")
        ax2.tick_params(colors="gray", labelsize=7)
        ax2.spines[:].set_color("#30363d")
        ax2.legend(fontsize=7, labelcolor="white", loc="upper right")

    if has_stats and n_panels >= 3:
        ax3 = axes[2]
        ax3.set_facecolor("#161b22")
        th_arr = th_test.reset_index(drop=True)
        lim    = min(n_steps, len(th_arr))
        t_idx  = np.arange(lim)
        for col, color, label in [
            ("Thm_CarnotEff",   "#81c784", "Carnot η"),
            ("Thm_EntropyProd", "#ce93d8", "σ EntrProd"),
            ("Thm_Quality",     "#80cbc4", "Q Quality"),
        ]:
            if col in th_arr.columns:
                ax3.plot(t_idx, th_arr[col].values[:lim], color=color, lw=1.0, label=label)
        ax3.axhline(0, color="#444", lw=0.5)
        ax3.set_ylabel("ThermoStats", color="white", fontsize=9)
        ax3.set_title("Carnot η | σ EntropyProd | Q Quality", color="gray", fontsize=8, loc="left")
        ax3.tick_params(colors="gray", labelsize=7)
        ax3.spines[:].set_color("#30363d")
        ax3.legend(fontsize=7, labelcolor="white", loc="upper right")

    fig.text(0.5, 0.01, f"Freq: {freq} | Fold {fold_n}",
             ha="center", fontsize=7, color="#555")
    out = os.path.join(results_dir, f"fold_{fold_n}_portfolio_{freq}.png")
    plt.savefig(out, dpi=130, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    print(f"  [Fold {fold_n}] Grafico salvato: {os.path.basename(out)}")


# ─── reward kwargs ────────────────────────────────────────────────────────────

def _env_reward_kwargs(bcfg, freq_cfg=None) -> dict:
    """
    Centralizza tutti i parametri passati a TradingEnv.__init__().

    [FIX E] lambda_inaction default: 0.2 → 0.005
    [FIX F] reward_clip default: 10.0 → 0.05
    [FIX G] transaction_cost propagato esplicitamente dal YAML.
    [FIX C1] ruin_stop_pct / ruin_penalty.
    [FIX C4] holding_urgency_*.
    [FIX D1] bars_per_year_override: inietta il valore dal config/frequency/*.yaml
             nel TradingEnv così non viene ricalcolato dall'indice stridato del df
             delle predizioni (stride=3 su 2m → avg_sec=360 → 16380 invece di 49140).
    [FIX D2] max_total_exposure_pct: cap globale esposizione portafoglio.
    [FIX D3] sell_unconditional_bonus: rompe il bootstrap problem del BUY-only.
    """
    # bars_per_year dal config di frequenza (es. config/frequency/minute.yaml)
    bpy = None
    if freq_cfg is not None:
        bpy = int(getattr(freq_cfg, "bars_per_year", 0)) or None

    return dict(
        thermo_bonus_sell        = getattr(bcfg, "thermo_bonus_sell",            0.5),
        thermo_penalty_buy       = getattr(bcfg, "thermo_penalty_buy",           0.1),
        lambda_inaction          = getattr(bcfg, "lambda_inaction",              0.005),  # FIX E
        lambda_concentration     = getattr(bcfg, "lambda_concentration",         0.6),
        sell_profit_bonus        = getattr(bcfg, "sell_profit_bonus",            0.015),
        max_holding_steps        = getattr(bcfg, "max_holding_steps",            60),
        forced_sell_cooldown     = getattr(bcfg, "forced_sell_cooldown_steps",   10),
        lambda_imbalance         = getattr(bcfg, "lambda_imbalance",             0.3),
        imbalance_threshold      = getattr(bcfg, "imbalance_threshold",          2.0),
        concentration_threshold  = getattr(bcfg, "concentration_threshold",      0.4),
        reward_clip              = getattr(bcfg, "reward_clip",                  0.05),  # FIX F
        fee_pct                  = getattr(bcfg, "transaction_cost",             0.001), # FIX G
        # ── [Fix C1] Ruin prevention ─────────────────────────────────────────
        ruin_stop_pct            = getattr(bcfg, "ruin_stop_pct",                0.70),
        ruin_penalty             = getattr(bcfg, "ruin_penalty",                 50.0),
        # ── [Fix C4] Urgency holding crescente ───────────────────────────────
        holding_urgency_start    = getattr(bcfg, "holding_urgency_start",        30),
        holding_urgency_rate     = getattr(bcfg, "holding_urgency_rate",         0.008),
        holding_urgency_max      = getattr(bcfg, "holding_urgency_max",          0.40),
        # ── [Fix D1] bars_per_year corretto ──────────────────────────────────
        bars_per_year_override   = bpy,
        # ── [Fix D2] Cap esposizione totale ──────────────────────────────────
        max_total_exposure_pct   = getattr(bcfg, "max_total_exposure_pct",       1.0),
        # ── [Fix D3] Sell bootstrap bonus ────────────────────────────────────
        sell_unconditional_bonus = getattr(bcfg, "sell_unconditional_bonus",     0.0),
    )


# ─── entry point ──────────────────────────────────────────────────────────────

def run_trade(
    cfg,
    df:       pd.DataFrame,
    tickers:  list[str],
    X_train,  Y_train,
    X_test,   Y_test,
    train_df: pd.DataFrame,
    test_df:  pd.DataFrame,
    df_raw:   pd.DataFrame | None = None,
):
    freq = cfg.frequency.interval

    # ── 1. Carica modello predittivo ──────────────────────────────────────────
    pred_path = _ckpt(cfg, "pred.pth")
    if not os.path.exists(pred_path):
        print(f"Errore: nessun checkpoint trovato in {pred_path}")
        print(f"Esegui prima: uv run main.py step=train frequency={freq}")
        return

    checkpoint   = torch.load(pred_path, map_location=get_map_location(), weights_only=False)
    num_features = checkpoint.get("num_features", df.shape[1])
    predictor    = _load_predictor(cfg, num_features, checkpoint)
    scaler       = checkpoint["scaler"]

    print(f"[Trade] Frequenza: {freq} | Checkpoint predittore: {os.path.basename(pred_path)}")

    # ── 2. Predizioni denormalizzate ──────────────────────────────────────────
    print("[Trade] Inferenza train set...")
    train_pred_scaled = _predict_batched(predictor, X_train, batch_size=BATCH_SIZE)
    print("[Trade] Inferenza test set...")
    test_pred_scaled  = _predict_batched(predictor, X_test,  batch_size=BATCH_SIZE)

    all_columns     = list(df.columns)
    n_windows_train = len(Y_train)
    n_windows_test  = len(Y_test)

    train_dates = train_df.index[cfg.prediction.window_size::cfg.prediction.stride][:n_windows_train]
    test_dates  = test_df.index[cfg.prediction.window_size::cfg.prediction.stride][:n_windows_test]

    prices_real_train, prices_pred_train = _build_price_dfs(
        train_pred_scaled, Y_train, scaler, train_dates, all_columns)
    prices_real_test, prices_pred_test = _build_price_dfs(
        test_pred_scaled, Y_test, scaler, test_dates, all_columns)

    # ── 3. Feature termodinamiche + VdW ──────────────────────────────────────
    if df_raw is not None:
        raw_train_df, raw_test_df = _split_raw(df_raw, train_df.index, test_df.index)
    else:
        raw_train_df, raw_test_df = train_df, test_df

    from modelli.thermo_state_builder import ThermoStateBuilder
    builder      = ThermoStateBuilder(interval=cfg.frequency.interval)

    # [FIX H] _build_thermo centralizza builder.build() + compute_vdw_block()
    thermo_train = _build_thermo(builder, raw_train_df, tickers)
    thermo_test  = _build_thermo(builder, raw_test_df,  tickers)

    # ── 4. Ambienti ───────────────────────────────────────────────────────────
    buyer_cfg = cfg.buyer
    rw        = _env_reward_kwargs(buyer_cfg, freq_cfg=cfg.frequency)

    env_train = TradingEnv(
        df               = prices_real_train,
        tickers          = tickers,
        initial_capital  = buyer_cfg.initial_capital,
        max_position_pct = getattr(buyer_cfg, "max_position_pct", 0.05),
        thermo_df        = thermo_train,
        **rw,
    )
    env_test = TradingEnv(
        df               = prices_real_test,
        tickers          = tickers,
        initial_capital  = buyer_cfg.initial_capital,
        max_position_pct = getattr(buyer_cfg, "max_position_pct", 0.05),
        thermo_df        = thermo_test,
        **rw,
    )

    # ── 5. Agente DDPG ────────────────────────────────────────────────────────
    ddpg_path      = _ckpt(cfg, "ddpg.pth")
    ddpg_best_path = _ckpt(cfg, "ddpg_best.pth")
    norm_path      = _ckpt(cfg, "normalizer.npz")
    ddpg_cfg       = cfg.buyer.ddpg

    agent = DDPGAgent(
        state_dim         = env_train.observation_space.shape[0],
        action_dim        = env_train.action_space.shape[0],
        actor_hidden      = list(ddpg_cfg.actor_hidden),
        critic_hidden     = list(ddpg_cfg.critic_hidden),
        lr_actor          = ddpg_cfg.lr_actor,
        lr_critic         = ddpg_cfg.lr_critic,
        gamma             = ddpg_cfg.gamma,
        tau               = ddpg_cfg.tau,
        buffer_capacity   = buyer_cfg.buffer_capacity,
        batch_size        = buyer_cfg.batch_size,
        update_every      = ddpg_cfg.update_every,
        noise_sigma       = ddpg_cfg.noise_sigma,
        episodic_episodes = getattr(buyer_cfg, "episodic_episodes", 20),
    )

    norm_clip  = getattr(buyer_cfg, "obs_clip", 5.0)
    normalizer = ObsNormalizer(shape=env_train.observation_space.shape[0], clip=norm_clip)

    if os.path.exists(norm_path):
        normalizer.load(norm_path)
    if os.path.exists(ddpg_best_path):
        agent.load(ddpg_best_path)
    elif os.path.exists(ddpg_path):
        agent.load(ddpg_path)

    # ── 6. Addestramento ──────────────────────────────────────────────────────
    es_patience = getattr(buyer_cfg, "es_patience", 20)
    es_metric   = getattr(buyer_cfg, "es_metric",   "sharpe")

    print(
        f"\nAddestramento DDPG [{freq}] — {buyer_cfg.n_episodes} episodi | "
        f"ES: {es_patience} ep su {es_metric} | "
        f"Thermo: {'attivo' if thermo_train is not None else 'no'} | "
        f"obs_clip={norm_clip} | lambda_inaction={rw['lambda_inaction']} | "
        f"reward_clip={rw['reward_clip']}"
    )

    history = train_ddpg_normalized(
        env         = env_train,
        agent       = agent,
        normalizer  = normalizer,
        n_episodes  = buyer_cfg.n_episodes,
        norm_path   = norm_path,
        ckpt_path   = ddpg_best_path,
        log_every   = buyer_cfg.log_every,
        es_patience = es_patience,
    )
    agent.save(ddpg_path, tag="FINAL")

    # ── 7. Valutazione con best checkpoint ────────────────────────────────────
    if os.path.exists(ddpg_best_path):
        agent.load(ddpg_best_path)
    if os.path.exists(norm_path):
        normalizer.load(norm_path)

    print("\nValutazione sul test set...")
    raw_state = env_test.reset()
    if isinstance(raw_state, tuple):
        raw_state = raw_state[0]
    state = normalizer.normalize(raw_state, update=False)
    done  = False
    while not done:
        action      = agent.act(state, explore=False)
        step_result = env_test.step(action)
        if len(step_result) == 5:
            raw_next, _, terminated, truncated, _ = step_result
            done = terminated or truncated
        else:
            raw_next, _, done, _ = step_result
        state = normalizer.normalize(raw_next, update=False)

    # ── 8. Output ─────────────────────────────────────────────────────────────
    daily   = _build_portfolio_daily(env_test, tickers)
    final_v = env_test.portfolio_history()[-1]
    ret_pct = (final_v - buyer_cfg.initial_capital) / buyer_cfg.initial_capital * 100
    best_ep = max(history, key=lambda h: h.get(es_metric, h["sharpe"]))

    print(f"\n{'─'*54}")
    print(f"  Frequenza         : {freq}")
    print(f"  Miglior episodio  : {best_ep['episode']} (su {es_metric})")
    print(f"  Capitale iniziale : ${buyer_cfg.initial_capital:>10,.2f}")
    print(f"  Valore finale     : ${final_v:>10,.2f}")
    print(f"  Rendimento        : {ret_pct:>+10.2f}%")
    print(f"  Sharpe ratio      : {env_test.sharpe_ratio():>10.3f}")
    print(f"  Max Drawdown      : {env_test.max_drawdown():>10.3f}")
    print(f"  Normalizer (n)    : {normalizer.rms.count:>10.0f} campioni")
    ts = env_test.trade_stats()
    print(f"  Trade BUY/SELL    : {ts['n_buy']}/{ts['n_sell']}")
    print(f"{'─'*54}")

    results_dir = cfg.paths.results_dir
    os.makedirs(results_dir, exist_ok=True)

    print(f"\nGrafici salvati in {results_dir}/:")
    _plot_portfolio_daily(daily, tickers, buyer_cfg.initial_capital, results_dir, freq)
    _plot_learning(history, results_dir, freq)
    _plot_trades(env_test, results_dir, freq)


# ─── Walk-forward validation ──────────────────────────────────────────────────

def run_walk_forward(
    cfg,
    df:            pd.DataFrame,
    tickers:       list[str],
    train_df:      pd.DataFrame,
    test_df:       pd.DataFrame,
    X_all:         "torch.Tensor",
    Y_all:         "torch.Tensor",
    df_raw:        "pd.DataFrame | None" = None,
    n_folds:       int   = 5,
    mode:          str   = "sliding",
    min_train_pct: float = 0.55,
    test_pct:      float = 0.12,
    warm_start:    bool  = True,
) -> "WalkForwardReport":
    import torch
    from modelli.walk_forward import make_folds, WalkForwardReport, FoldResult
    from modelli.ddpg import DDPGAgent, ThermoEnsemble, compute_thermo_profile
    from modelli.obs_normalizer import ObsNormalizer, train_ddpg_normalized
    from modelli.trading_env import TradingEnv
    from modelli.thermo_state_builder import ThermoStateBuilder

    freq = cfg.frequency.interval

    # ── 1. Carica predittore ──────────────────────────────────────────────────
    pred_path = _ckpt(cfg, "pred.pth")
    if not os.path.exists(pred_path):
        print(f"[WalkForward] Predittore non trovato: {pred_path}")
        return WalkForwardReport(mode=mode)

    checkpoint   = torch.load(pred_path, map_location=get_map_location(), weights_only=False)
    num_features = checkpoint.get("num_features", df.shape[1])
    predictor    = _load_predictor(cfg, num_features, checkpoint)
    scaler       = checkpoint["scaler"]

    print(f"[WalkForward] Inferenza CNN su {len(X_all):,} finestre totali...")
    all_pred_scaled = _predict_batched(predictor, X_all, batch_size=BATCH_SIZE)

    ws        = cfg.prediction.window_size
    stride    = cfg.prediction.stride
    all_dates = df.index[ws::stride][:len(Y_all)]

    prices_real_all, prices_pred_all = _build_price_dfs(
        all_pred_scaled, Y_all, scaler, all_dates, list(df.columns)
    )

    n_bars = len(prices_real_all)
    folds  = make_folds(n_bars=n_bars, n_folds=n_folds, mode=mode,
                        min_train_pct=min_train_pct, test_pct=test_pct)

    if not folds:
        print(f"[WalkForward] Impossibile costruire fold con {n_bars} barre.")
        return WalkForwardReport(mode=mode)

    print(
        f"\n[WalkForward] {len(folds)} fold | mode={mode} | "
        f"min_train={min_train_pct:.0%} | test={test_pct:.0%} | "
        f"barre totali: {n_bars:,} | freq: {freq}"
    )

    # ── 2. Setup comune ───────────────────────────────────────────────────────
    buyer_cfg      = cfg.buyer
    ddpg_cfg       = cfg.buyer.ddpg
    checkpoint_dir = cfg.paths.checkpoint_dir
    results_dir    = cfg.paths.results_dir
    es_patience    = getattr(buyer_cfg, "es_patience", 20)
    rw             = _env_reward_kwargs(buyer_cfg, freq_cfg=cfg.frequency)  # [FIX D1+E+F+G]

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir,    exist_ok=True)

    builder = ThermoStateBuilder(interval=freq)
    report  = WalkForwardReport(mode=mode)

    # _env_common usa rw che già include fee_pct (FIX G), bars_per_year_override (FIX D1), ecc.
    # max_position_pct viene aggiunto esplicitamente perché non transitava via _env_reward_kwargs.
    _env_common = dict(
        tickers         = tickers,
        initial_capital = buyer_cfg.initial_capital,
        max_position_pct = getattr(buyer_cfg, "max_position_pct", 0.05),
        **rw,
    )

    prev_best_path:  str | None          = None
    prev_thermo_df:  pd.DataFrame | None = None

    ensemble_agents:   list[DDPGAgent]     = []
    ensemble_profiles: list[np.ndarray]   = []
    ensemble_norms:    list[ObsNormalizer] = []

    def _thermo_kl_div(df_prev: pd.DataFrame, df_curr: pd.DataFrame) -> float:
        from scipy.special import rel_entr
        cols, kl_sum, n_valid = ["Thm_Stress", "Thm_Entropy"], 0.0, 0
        for col in cols:
            if col not in df_prev.columns or col not in df_curr.columns:
                continue
            vals_p = df_prev[col].dropna().values
            vals_c = df_curr[col].dropna().values
            if len(vals_p) < 20 or len(vals_c) < 20:
                continue
            lo, hi = min(vals_p.min(), vals_c.min()), max(vals_p.max(), vals_c.max())
            bins   = np.linspace(lo, hi, 21)
            hp, _  = np.histogram(vals_p, bins=bins, density=True)
            hc, _  = np.histogram(vals_c, bins=bins, density=True)
            hp     = (hp + 1e-10) / (hp + 1e-10).sum()
            hc     = (hc + 1e-10) / (hc + 1e-10).sum()
            kl_sum  += float(np.sum(rel_entr(hc, hp)))
            n_valid += 1
        return kl_sum / max(n_valid, 1)

    # ── 3. Loop sui fold ──────────────────────────────────────────────────────
    for fold_n, ((tr_s, tr_e), (te_s, te_e)) in enumerate(folds, 1):
        print(f"\n{'═'*62}")
        print(f"  FOLD {fold_n}/{len(folds)} | "
              f"train [{tr_s:,}:{tr_e:,}] | test [{te_s:,}:{te_e:,}]")

        pr_train = prices_real_all.iloc[tr_s:tr_e]
        pr_test  = prices_real_all.iloc[te_s:te_e]

        raw_src  = df_raw if df_raw is not None else df

        # [FIX H] _build_thermo include VdW automaticamente
        th_train = _build_thermo(builder, raw_src.reindex(pr_train.index), tickers)
        th_test  = _build_thermo(builder, raw_src.reindex(pr_test.index),  tickers)

        env_train = TradingEnv(df=pr_train, thermo_df=th_train, **_env_common)
        env_test  = TradingEnv(df=pr_test,  thermo_df=th_test,  **_env_common)

        agent = DDPGAgent(
            state_dim         = env_train.observation_space.shape[0],
            action_dim        = env_train.action_space.shape[0],
            actor_hidden      = list(ddpg_cfg.actor_hidden),
            critic_hidden     = list(ddpg_cfg.critic_hidden),
            lr_actor          = ddpg_cfg.lr_actor,
            lr_critic         = ddpg_cfg.lr_critic,
            gamma             = ddpg_cfg.gamma,
            tau               = ddpg_cfg.tau,
            buffer_capacity   = buyer_cfg.buffer_capacity,
            batch_size        = buyer_cfg.batch_size,
            update_every      = ddpg_cfg.update_every,
            noise_sigma       = ddpg_cfg.noise_sigma,
            episodic_episodes = getattr(buyer_cfg, "episodic_episodes", 20),
        )
        normalizer = ObsNormalizer(
            shape = env_train.observation_space.shape[0],
            clip  = getattr(buyer_cfg, "obs_clip", 5.0),
        )

        if warm_start and prev_best_path and os.path.exists(prev_best_path):
            use_warm = True
            kl_thr   = getattr(buyer_cfg, "warm_start_kl_threshold", 2.0)
            if prev_thermo_df is not None and th_train is not None and not th_train.empty:
                kl = _thermo_kl_div(prev_thermo_df, th_train)
                if kl > kl_thr:
                    print(f"  [WarmStart] KL={kl:.3f} > {kl_thr} → skip")
                    use_warm = False
                else:
                    print(f"  [WarmStart] KL={kl:.3f} ≤ {kl_thr} → OK")
            if use_warm:
                agent.load(prev_best_path)

        fold_best = os.path.join(checkpoint_dir, f"ddpg_wf_fold{fold_n}_best_{freq}.pth")
        fold_norm = os.path.join(checkpoint_dir, f"normalizer_wf_fold{fold_n}_{freq}.npz")

        history = train_ddpg_normalized(
            env         = env_train,
            agent       = agent,
            normalizer  = normalizer,
            n_episodes  = buyer_cfg.n_episodes,
            norm_path   = fold_norm,
            ckpt_path   = fold_best,
            log_every   = max(1, buyer_cfg.log_every * 5),
            es_patience = es_patience,
        )

        if os.path.exists(fold_best):
            agent.load(fold_best)
        if os.path.exists(fold_norm):
            normalizer.load(fold_norm)

        raw_state = env_test.reset()
        if isinstance(raw_state, tuple):
            raw_state = raw_state[0]
        state = normalizer.normalize(raw_state, update=False)
        done  = False
        sell_ok_count = sell_total = 0

        while not done:
            action      = agent.act(state, explore=False)
            step_result = env_test.step(action)
            if len(step_result) == 5:
                raw_next, _, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                raw_next, _, done, info = step_result

            step_idx = env_test.current_step - 1
            if env_test._has_thermo and step_idx >= 0:
                thm_z  = env_test._thermo_val("Thm_Stress", 0.0)
                thm_sg = env_test._thermo_val("Thm_SellSignal", 0.0)
                sells  = sum(1 for t in env_test._trades
                             if t.get("step") == step_idx and t["type"] == "SELL")
                if sells > 0:
                    sell_total += 1
                    if thm_z > 0.5 or thm_sg > 0.5:
                        sell_ok_count += 1

            state = normalizer.normalize(raw_next, update=False)

        final_v = env_test.portfolio_history()[-1]
        ret_pct = (final_v - buyer_cfg.initial_capital) / buyer_cfg.initial_capital * 100
        sharpe  = env_test.sharpe_ratio()
        max_dd  = env_test.max_drawdown()
        best_ep = max(history, key=lambda h: h.get("sharpe", -999))
        thermo_sell_ok = (sell_ok_count / max(sell_total, 1)) * 100.0

        ts = env_test.trade_stats()
        print(
            f"  [Fold {fold_n}] BUY={ts['n_buy']} SELL={ts['n_sell']} "
            f"ratio={ts['buy_sell_ratio']:.1f} | Thm✅: {thermo_sell_ok:.1f}%"
        )

        # [Fix C1] Calcola ruin_rate: quanti episodi del fold sono terminati
        # per hard stop-loss. Con i fix implementati dovrebbe scendere a ~0.
        ruin_episodes = sum(1 for h in history if h.get("ruin", False))
        ruin_rate_pct = (ruin_episodes / max(1, len(history))) * 100.0

        from modelli.walk_forward import FoldResult
        fold_res = FoldResult(
            fold               = fold_n,
            train_bars         = tr_e - tr_s,
            test_bars          = te_e - te_s,
            train_start        = str(pr_train.index[0])[:16],
            train_end          = str(pr_train.index[-1])[:16],
            test_start         = str(pr_test.index[0])[:16],
            test_end           = str(pr_test.index[-1])[:16],
            sharpe             = sharpe,
            total_return_pct   = ret_pct,
            max_drawdown       = max_dd,
            n_episodes_run     = len(history),
            best_episode       = best_ep["episode"],
            thermo_sell_ok_pct = thermo_sell_ok,
            ruin_rate_pct      = ruin_rate_pct,
        )
        report.folds.append(fold_res)
        prev_best_path = fold_best
        prev_thermo_df = th_train if (th_train is not None and not th_train.empty) else None

        try:
            _plot_fold_portfolio(env_test, fold_n, sharpe, ret_pct, max_dd,
                                 th_test, results_dir, freq)
        except Exception as _pe:
            print(f"  [Fold {fold_n}] Grafico skip: {_pe}")

        print(f"  ✔ Fold {fold_n} — Sharpe: {sharpe:.3f} | "
              f"Return: {ret_pct:+.2f}% | MaxDD: {max_dd:.3f}")

        if th_train is not None and not th_train.empty:
            try:
                profile = compute_thermo_profile(th_train, 0, len(th_train))
                ensemble_agents.append(agent)
                ensemble_profiles.append(profile)
                ensemble_norms.append(normalizer)
            except Exception as e:
                print(f"  [Ensemble] Profilo fold {fold_n}: {e}")

    # ── 4. Report finale ──────────────────────────────────────────────────────
    report.print_summary()
    report.save(os.path.join(results_dir, f"walk_forward_{freq}.csv"))

    try:
        _plot_walk_forward_summary(report, results_dir, freq)
    except Exception as _we:
        print(f"  [WF Summary Plot] Skip: {_we}")

    # [FIX D] Copia normalizer del fold migliore
    if report.folds:
        import shutil
        best_fold_n   = max(report.folds, key=lambda f: f.sharpe).fold
        src_norm      = os.path.join(checkpoint_dir, f"normalizer_wf_fold{best_fold_n}_{freq}.npz")
        dst_norm_best = os.path.join(checkpoint_dir, f"normalizer_best_{freq}.npz")
        dst_norm_gen  = os.path.join(checkpoint_dir, f"normalizer_{freq}.npz")
        if os.path.exists(src_norm):
            shutil.copy2(src_norm, dst_norm_best)
            shutil.copy2(src_norm, dst_norm_gen)
            print(f"  [Fix D] Normalizer fold {best_fold_n} → normalizer_best_{freq}.npz")

    # ── 5. ThermoEnsemble ────────────────────────────────────────────────────
    if len(ensemble_agents) >= 2 and folds:
        print(f"\n{'─'*62}")
        print(f"  THERMO ENSEMBLE — {len(ensemble_agents)} agenti")

        last_te_s, last_te_e = folds[-1][1]
        pr_test_last = prices_real_all.iloc[last_te_s:last_te_e]
        raw_src      = df_raw if df_raw is not None else df

        # [FIX H] VdW anche nell'ensemble
        th_test_last = _build_thermo(builder, raw_src.reindex(pr_test_last.index), tickers)

        env_ensemble = TradingEnv(df=pr_test_last, thermo_df=th_test_last, **_env_common)

        from modelli.ddpg import ThermoEnsemble
        ensemble = ThermoEnsemble(
            agents        = ensemble_agents,
            fold_profiles = ensemble_profiles,
            temperature   = getattr(buyer_cfg, "ensemble_temperature", 0.5),
        )

        thm_cols = (
            [c for c in th_test_last.columns if c.startswith("Thm_")]
            if th_test_last is not None and not th_test_last.empty
            else []
        )

        raw_state = env_ensemble.reset()
        if isinstance(raw_state, tuple):
            raw_state = raw_state[0]
        last_norm = ensemble_norms[-1]
        state     = last_norm.normalize(raw_state, update=False)
        done      = False
        step_idx  = 0

        while not done:
            if thm_cols and th_test_last is not None and step_idx < len(th_test_last):
                current_thermo = th_test_last.iloc[step_idx][thm_cols].values.astype(np.float32)
            else:
                current_thermo = np.zeros(len(ensemble_profiles[0]), dtype=np.float32)

            action      = ensemble.act(state, current_thermo, explore=False)
            step_result = env_ensemble.step(action)
            if len(step_result) == 5:
                raw_next, _, terminated, truncated, _ = step_result
                done = terminated or truncated
            else:
                raw_next, _, done, _ = step_result
            state    = last_norm.normalize(raw_next, update=False)
            step_idx += 1

        ens_final  = env_ensemble.portfolio_history()[-1]
        ens_ret    = (ens_final - buyer_cfg.initial_capital) / buyer_cfg.initial_capital * 100
        ens_sharpe = env_ensemble.sharpe_ratio()
        ens_dd     = env_ensemble.max_drawdown()

        print(f"  ThermoEnsemble → Sharpe: {ens_sharpe:.3f} | "
              f"Return: {ens_ret:+.2f}% | MaxDD: {ens_dd:.3f}")

        best_fold = max(report.folds, key=lambda f: f.sharpe)
        delta = ens_sharpe - best_fold.sharpe
        print(f"  vs best fold (fold {best_fold.fold}): {delta:+.3f}")

        pd.DataFrame([{
            "sharpe": ens_sharpe, "return_pct": ens_ret, "max_dd": ens_dd,
            "n_agents": len(ensemble_agents),
            "temperature": getattr(buyer_cfg, "ensemble_temperature", 0.5),
        }]).to_csv(os.path.join(results_dir, f"thermo_ensemble_{freq}.csv"), index=False)
        print(f"{'─'*62}")

    return report