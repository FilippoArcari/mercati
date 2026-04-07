"""
modelli/trade.py — Addestramento e valutazione dell'agente DDPG
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter

from modelli.pred import Pred
from modelli.ddpg import DDPGAgent, train_ddpg
from modelli.trading_env import TradingEnv
from modelli.obs_normalizer import ObsNormalizer, train_ddpg_normalized
from modelli.intraday_thermo import compute_intraday_thermo_features, detect_market_regime

BATCH_SIZE = 256
# ─── naming checkpoint ────────────────────────────────────────────────────────

def _ckpt(cfg, name: str) -> str:
    freq = cfg.frequency.interval
    ws   = cfg.prediction.window_size
    stem, ext = os.path.splitext(name)
    tag = f"_{freq}_w{ws}" if stem == "pred" else f"_{freq}"
    return os.path.join(cfg.paths.checkpoint_dir, f"{stem}{tag}{ext}")

def _predict_batched(predictor, X: torch.Tensor, batch_size: int = 256) -> np.ndarray:
    """Inferenza a batch per evitare OOM con dataset grandi."""
    results = []
    device = next(predictor.parameters()).device   # ← ricava il device dal modello
    predictor.eval()
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            
            batch = X[i : i + batch_size].to(device)  # ← sposta il batch sul device corretto
            out   = predictor(batch)
            if out.dim() == 3:
                out = out[:, 0, :]
            results.append(out.cpu().numpy())
    return np.concatenate(results, axis=0)

# ─── helpers privati ──────────────────────────────────────────────────────────

def _load_predictor(cfg, num_features, checkpoint):
    saved_cfg = checkpoint.get("config", {})
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
        max_grad_norm    = getattr(cfg.training, "max_grad_norm", 1.0),
    )

    predictor.load_state_dict(checkpoint["model_state_dict"])
    predictor.eval()

    # ← MANCAVA QUESTO
    from modelli.utils import get_device
    predictor = predictor.to(get_device())
    return predictor

def _build_price_dfs(pred_scaled, real_targets, scaler, dates, all_columns):
    real = scaler.inverse_transform(
        real_targets.numpy() if hasattr(real_targets, "numpy") else real_targets
    )
    pred = scaler.inverse_transform(pred_scaled)
    return (
        pd.DataFrame(real, index=dates, columns=all_columns),
        pd.DataFrame(pred, index=dates, columns=all_columns),
    )


def _extract_psi(df: pd.DataFrame, tickers: list[str]):
    psi_cols = [f"Psi_{t}" for t in tickers if f"Psi_{t}" in df.columns]
    return df[psi_cols] if psi_cols else None


def _build_portfolio_daily(env: TradingEnv, tickers: list[str]) -> pd.DataFrame:
    vpdf = env.value_per_ticker_df()
    if vpdf.empty:
        return pd.DataFrame()

    pf_hist = env.portfolio_history()[1:]
    n       = min(len(vpdf), len(pf_hist))
    daily   = vpdf.iloc[:n].copy()
    daily["portfolio_value"] = pf_hist[:n]

    val_cols = [f"{t}_value" for t in tickers if f"{t}_value" in daily.columns]
    daily["holdings_value"]        = daily[val_cols].sum(axis=1)
    initial                        = env.initial_capital
    daily["daily_return_pct"]      = daily["portfolio_value"].pct_change().fillna(0) * 100
    daily["cumulative_return_pct"] = (daily["portfolio_value"] / initial - 1) * 100
    peak                           = daily["portfolio_value"].cummax()
    daily["drawdown_pct"]          = (daily["portfolio_value"] / peak - 1) * 100

    meta = ["portfolio_value", "holdings_value", "cash",
            "daily_return_pct", "cumulative_return_pct", "drawdown_pct"]
    rest = [c for c in daily.columns if c not in meta]
    return daily[meta + rest]


# ─── split raw df ─────────────────────────────────────────────────────────────

def _split_raw(df_raw: pd.DataFrame, train_index: pd.Index, test_index: pd.Index):
    """
    Ritaglia df_raw (con le colonne *_Volume) sugli stessi indici
    di train_df e test_df già calcolati su df.
    Usa .reindex() per allineare anche se gli indici non combaciano
    perfettamente (es. barre mancanti in cache).
    """
    raw_train = df_raw.reindex(train_index)
    raw_test  = df_raw.reindex(test_index)
    return raw_train, raw_test


# ─── grafici ──────────────────────────────────────────────────────────────────

def _plot_portfolio_daily(daily, tickers, initial_capital, results_dir, freq):
    if daily.empty:
        return

    dates   = daily.index
    usd_fmt = FuncFormatter(lambda y, _: f"${y:,.0f}")
    pct_fmt = FuncFormatter(lambda y, _: f"{y:+.1f}%")

    fig = plt.figure(figsize=(15, 14))
    gs  = gridspec.GridSpec(4, 1, figure=fig, hspace=0.45,
                            height_ratios=[2, 1.2, 1, 1.8])

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

    ax4      = fig.add_subplot(gs[3], sharex=ax1)
    val_cols = [f"{t}_value" for t in tickers if f"{t}_value" in daily.columns]
    top8     = daily[val_cols].mean().nlargest(8).index.tolist()
    labels   = [c.replace("_value", "") for c in top8]
    colors   = plt.cm.tab10(np.linspace(0, 1, len(top8)))
    ax4.stackplot(dates, [daily[c].values for c in top8],
                  labels=labels, colors=colors, alpha=0.8)
    ax4.plot(dates, daily["cash"], color="black",
             linewidth=1.2, linestyle=":", label="Cash", zorder=5)
    ax4.set_title("Composizione — top 8 ticker + cash", fontsize=10)
    ax4.set_ylabel("USD")
    ax4.yaxis.set_major_formatter(usd_fmt)
    ax4.legend(ncol=5, fontsize=7, loc="upper left")
    ax4.grid(True, alpha=0.25)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)

    out = os.path.join(results_dir, f"portfolio_daily_{freq}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  portfolio_daily_{freq}.png")


def _plot_learning(history, results_dir, freq):
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
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

    sharpes = [h["sharpe"] for h in history]
    axes[1].plot(episodes, sharpes, color="green", linewidth=1.5)
    axes[1].axhline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[1].set_title("Sharpe ratio per episodio")
    axes[1].set_ylabel("Sharpe")
    axes[1].grid(True, alpha=0.25)

    sigmas = [h["noise_sigma"] for h in history]
    axes[2].plot(episodes, sigmas, color="purple", linewidth=1.5)
    axes[2].set_title("Decadimento rumore σ")
    axes[2].set_xlabel("Episodio")
    axes[2].set_ylabel("σ")
    axes[2].grid(True, alpha=0.25)

    plt.tight_layout()
    out = os.path.join(results_dir, f"ddpg_learning_{freq}.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  ddpg_learning_{freq}.png")


def _plot_pnl_and_trades(env, tickers, daily, results_dir, freq):
    summary = env.summary_per_ticker()

    if not daily.empty and not summary.empty:
        top5     = summary.head(5).index.tolist()
        pnl_cols = [f"{t}_unrealized_pnl" for t in top5
                    if f"{t}_unrealized_pnl" in daily.columns]
        if pnl_cols:
            fig, ax = plt.subplots(figsize=(13, 5))
            for col in pnl_cols:
                ax.plot(daily.index, daily[col],
                        label=col.replace("_unrealized_pnl", ""), linewidth=1.4)
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
            ax.set_title(f"P&L non realizzato — top 5 ticker [{freq}]")
            ax.set_ylabel("USD")
            ax.legend(ncol=3, fontsize=8)
            ax.grid(True, alpha=0.25)
            plt.tight_layout()
            out = os.path.join(results_dir, f"pnl_per_ticker_{freq}.png")
            plt.savefig(out, dpi=150)
            plt.close()
            print(f"  pnl_per_ticker_{freq}.png")

    trade_log = env.trade_log_df()
    if not trade_log.empty:
        ac = (trade_log[trade_log["action"] != "HOLD"]
              .groupby(["ticker", "action"]).size().unstack(fill_value=0))
        ac["total"] = ac.sum(axis=1)
        top10 = ac.nlargest(10, "total").drop(columns="total")

        fig, ax = plt.subplots(figsize=(13, 5))
        top10.plot(kind="bar", ax=ax, color=["#e74c3c", "#2ecc71"])
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


def _save_csv(env, tickers, daily, results_dir, freq):
    env.trade_log_df().to_csv(
        os.path.join(results_dir, f"trade_log_{freq}.csv"))
    env.value_per_ticker_df().to_csv(
        os.path.join(results_dir, f"value_per_ticker_{freq}.csv"))
    env.summary_per_ticker().to_csv(
        os.path.join(results_dir, f"summary_per_ticker_{freq}.csv"))
    if not daily.empty:
        daily.to_csv(os.path.join(results_dir, f"portfolio_daily_{freq}.csv"))


# ─── analisi post-hoc termodinamica ───────────────────────────────────────────

def _analyze_thermo_trades(trade_log: pd.DataFrame, results_dir: str, freq: str) -> None:
    if "thermo_stress" not in trade_log.columns:
        return

    sells = trade_log[trade_log["action"] == "SELL"]
    buys  = trade_log[trade_log["action"] == "BUY"]

    correct_sells   = sells[sells["thermo_stress"] > 1.0]
    incorrect_sells = sells[sells["thermo_stress"] < -0.5]
    neutral_sells   = sells[
        (sells["thermo_stress"] >= -0.5) & (sells["thermo_stress"] <= 1.0)
    ]
    n_sells = len(sells) + 1e-8

    correct_sell_pct   = 100 * len(correct_sells)   / n_sells
    incorrect_sell_pct = 100 * len(incorrect_sells) / n_sells
    neutral_sell_pct   = 100 * len(neutral_sells)   / n_sells

    good_buys = buys[buys["thermo_stress"] < -0.5]
    bad_buys  = buys[buys["thermo_efficiency"] > 1.5]
    n_buys = len(buys) + 1e-8

    good_buy_pct = 100 * len(good_buys) / n_buys
    bad_buy_pct  = 100 * len(bad_buys)  / n_buys

    avg_pnl_correct_sell   = correct_sells["realized_pnl"].mean()   if len(correct_sells)   else 0.0
    avg_pnl_incorrect_sell = incorrect_sells["realized_pnl"].mean() if len(incorrect_sells) else 0.0
    avg_pnl_good_buy       = good_buys["realized_pnl"].mean()       if len(good_buys)       else 0.0
    avg_pnl_bad_buy        = bad_buys["realized_pnl"].mean()        if len(bad_buys)        else 0.0

    sep = "─" * 56
    print(f"\n{sep}")
    print(f"  ANALISI TERMODINAMICA POST-HOC [{freq}]")
    print(sep)
    print(f"  VENDITE ({len(sells):.0f} totali)")
    print(f"    ✅ Corrette  (stress > +1.0) : {correct_sell_pct:5.1f}%  "
          f"| P&L medio: ${avg_pnl_correct_sell:+.2f}")
    print(f"    ❌ Sbagliate (stress < -0.5) : {incorrect_sell_pct:5.1f}%  "
          f"| P&L medio: ${avg_pnl_incorrect_sell:+.2f}")
    print(f"    ○  Neutrali                  : {neutral_sell_pct:5.1f}%")
    print(f"  ACQUISTI ({len(buys):.0f} totali)")
    print(f"    ✅ Ottimi   (stress < -0.5)  : {good_buy_pct:5.1f}%  "
          f"| P&L medio: ${avg_pnl_good_buy:+.2f}")
    print(f"    ❌ Cattivi  (efficiency>1.5) : {bad_buy_pct:5.1f}%  "
          f"| P&L medio: ${avg_pnl_bad_buy:+.2f}")
    print(sep)
    print(f"  TARGET: vendite corrette >50% | acquisti cattivi <20%")

    sell_ok = correct_sell_pct > 50
    buy_ok  = bad_buy_pct < 20
    status  = "🟢 BUONO" if (sell_ok and buy_ok) else \
              "🟡 PARZIALE" if (sell_ok or buy_ok) else "🔴 DA MIGLIORARE"
    print(f"  Giudizio complessivo: {status}")
    print(sep)

    summary_rows = [
        {"metrica": "vendite_corrette_pct",      "valore": round(correct_sell_pct, 2),
         "target": ">50%", "ok": sell_ok},
        {"metrica": "vendite_sbagliate_pct",     "valore": round(incorrect_sell_pct, 2),
         "target": "<20%", "ok": incorrect_sell_pct < 20},
        {"metrica": "acquisti_ottimi_pct",        "valore": round(good_buy_pct, 2),
         "target": ">30%", "ok": good_buy_pct > 30},
        {"metrica": "acquisti_cattivi_pct",       "valore": round(bad_buy_pct, 2),
         "target": "<20%", "ok": buy_ok},
        {"metrica": "pnl_medio_vendita_corretta", "valore": round(avg_pnl_correct_sell, 4),
         "target": ">0",   "ok": avg_pnl_correct_sell > 0},
        {"metrica": "pnl_medio_acquisto_cattivo", "valore": round(avg_pnl_bad_buy, 4),
         "target": "<0",   "ok": avg_pnl_bad_buy < 0},
    ]
    summary_df = pd.DataFrame(summary_rows).set_index("metrica")
    out_path = os.path.join(results_dir, f"thermo_analysis_{freq}.csv")
    summary_df.to_csv(out_path)
    print(f"  → CSV salvato: thermo_analysis_{freq}.csv")



# ─── entry point ─────────────────────────────────────────────────────────────

def run_trade(
    cfg,
    df: pd.DataFrame,
    tickers: list[str],
    X_train, Y_train,
    X_test,  Y_test,
    train_df: pd.DataFrame,
    test_df:  pd.DataFrame,
    df_raw:   pd.DataFrame | None = None,   # ← DataFrame con le *_Volume per-ticker
):
    freq = cfg.frequency.interval

    # ── 1. Carica modello predittivo ──────────────────────────────────────────
    pred_path = _ckpt(cfg, "pred.pth")
    if not os.path.exists(pred_path):
        print(f"Errore: nessun checkpoint trovato in {pred_path}")
        print(f"Esegui prima: uv run main.py step=train frequency={freq}")
        return

    checkpoint   = torch.load(pred_path, weights_only=False)
    num_features = checkpoint.get("num_features", df.shape[1])
    predictor    = _load_predictor(cfg, num_features, checkpoint)
    scaler       = checkpoint["scaler"]

    print(f"[Trade] Frequenza: {freq} | "
          f"Checkpoint predittore: {os.path.basename(pred_path)}")

    # ── 2. Predizioni denormalizzate ──────────────────────────────────────────
    print("[Trade] Inferenza train set...")
    train_pred_scaled = _predict_batched(predictor, X_train, batch_size=BATCH_SIZE)
    print(f"[Trade] Train pred shape: {train_pred_scaled.shape}")

    print("[Trade] Inferenza test set...")
    test_pred_scaled  = _predict_batched(predictor, X_test,  batch_size=BATCH_SIZE)
    print(f"[Trade] Test pred shape: {test_pred_scaled.shape}")


    all_columns     = list(df.columns)
    n_windows_train = len(Y_train)
    n_windows_test  = len(Y_test)

    train_dates = train_df.index[cfg.prediction.window_size::cfg.prediction.stride][:n_windows_train]
    test_dates  = test_df.index[cfg.prediction.window_size::cfg.prediction.stride][:n_windows_test]

    prices_real_train, prices_pred_train = _build_price_dfs(
        train_pred_scaled, Y_train, scaler, train_dates, all_columns
    )
    prices_real_test, prices_pred_test = _build_price_dfs(
        test_pred_scaled, Y_test, scaler, test_dates, all_columns
    )

    # ── 3. Ψ dai DataFrame ────────────────────────────────────────────────────
    psi_train = _extract_psi(train_df, tickers)
    psi_test  = _extract_psi(test_df,  tickers)

    # ── 3b. Feature termodinamiche intraday ───────────────────────────────────
    #
    # PROBLEMA RISOLTO: load_data droppa le colonne *_Volume prima del return,
    # quindi train_df/test_df non le contengono. Usiamo df_raw (il CSV integrale)
    # splittato sugli stessi indici di train_df/test_df.
    #
    # Se df_raw non è disponibile (daily mode o chiamata legacy), fallback
    # su train_df/test_df — in quel caso i WARNING "colonna Volume non trovata"
    # sono attesi e il fallback aggregato si attiva correttamente.
    #
    if df_raw is not None:
        raw_train_df, raw_test_df = _split_raw(df_raw, train_df.index, test_df.index)
    else:
        raw_train_df, raw_test_df = train_df, test_df
    
    from modelli.thermo_state_builder import ThermoStateBuilder
    builder = ThermoStateBuilder(interval=cfg.frequency.interval)
    thermo_train = builder.build(raw_train_df, tickers)
    thermo_test  = builder.build(raw_test_df, tickers)

    # ── 4. Ambienti ───────────────────────────────────────────────────────────
    buyer_cfg = cfg.buyer
    lc  = getattr(buyer_cfg, "lambda_concentration",    0.1)
    li  = getattr(buyer_cfg, "lambda_inaction",         0.05)
    ll  = getattr(buyer_cfg, "lambda_loss",             0.5)
    lt_loss = getattr(buyer_cfg, "loss_threshold",     -0.5)
    at  = getattr(buyer_cfg, "action_threshold",        0.05)
    lt  = getattr(buyer_cfg, "lambda_thermo",           0.15)
    sst = getattr(buyer_cfg, "stress_sell_threshold",   1.0)
    sbt = getattr(buyer_cfg, "stress_buy_threshold",   -0.5)
    ept = getattr(buyer_cfg, "efficiency_penalty_thresh", 1.5)

    env_train = TradingEnv(
        prices_real=prices_real_train, prices_pred=prices_pred_train,
        tickers=tickers,
        initial_capital=buyer_cfg.initial_capital,
        transaction_cost=buyer_cfg.transaction_cost,
        psi_df=psi_train,
        thermo_df=thermo_train,
        lambda_concentration=lc,
        lambda_inaction=li,
        lambda_loss=ll,
        loss_threshold=lt_loss,
        action_threshold=at,
        log_trades=False,
        thermo_bonus_sell=getattr(buyer_cfg, "thermo_bonus_sell", 0.0),
        thermo_bonus_buy=getattr(buyer_cfg, "thermo_bonus_buy", 0.0),
        thermo_penalty_buy=getattr(buyer_cfg, "thermo_penalty_buy", 0.0),
        stress_sell_threshold=sst,
        stress_buy_threshold=sbt,
        efficiency_penalty_thresh=ept,
    )
    env_test = TradingEnv(
        prices_real=prices_real_test, prices_pred=prices_pred_test,
        tickers=tickers,
        initial_capital=buyer_cfg.initial_capital,
        transaction_cost=buyer_cfg.transaction_cost,
        psi_df=psi_test,
        thermo_df=thermo_test,
        lambda_concentration=0.0,
        lambda_inaction=0.0,
        lambda_loss=0.0,
        loss_threshold=lt_loss,
        action_threshold=at,
        log_trades=True,
        # During test, shaping rewards are calculated but mostly we care about logs
        thermo_bonus_sell=getattr(buyer_cfg, "thermo_bonus_sell", 0.0),
        thermo_bonus_buy=getattr(buyer_cfg, "thermo_bonus_buy", 0.0),
        thermo_penalty_buy=getattr(buyer_cfg, "thermo_penalty_buy", 0.0),
        stress_sell_threshold=sst,
        stress_buy_threshold=sbt,
        efficiency_penalty_thresh=ept,
    )

    # ── 5. Agente DDPG ────────────────────────────────────────────────────────
    ddpg_path      = _ckpt(cfg, "ddpg.pth")
    ddpg_best_path = _ckpt(cfg, "ddpg_best.pth")
    norm_path      = _ckpt(cfg, "normalizer.npz")
    ddpg_cfg       = cfg.buyer.ddpg

    print(f"[Trade] Checkpoint DDPG: {os.path.basename(ddpg_path)}")

    agent = DDPGAgent(
        state_dim=env_train.state_dim, action_dim=env_train.action_dim,
        actor_hidden=list(ddpg_cfg.actor_hidden),
        critic_hidden=list(ddpg_cfg.critic_hidden),
        lr_actor=ddpg_cfg.lr_actor, lr_critic=ddpg_cfg.lr_critic,
        gamma=ddpg_cfg.gamma, tau=ddpg_cfg.tau,
        buffer_capacity=buyer_cfg.buffer_capacity,
        batch_size=buyer_cfg.batch_size,
        update_every=ddpg_cfg.update_every,
        noise_sigma=ddpg_cfg.noise_sigma,
    )

    norm_clip  = getattr(buyer_cfg, "obs_clip", 5.0)
    normalizer = ObsNormalizer(state_dim=env_train.state_dim, clip=norm_clip)

    if os.path.exists(norm_path):
        normalizer.load(norm_path)
        print(f"[Trade] Normalizer caricato: {os.path.basename(norm_path)}")
    else:
        print(f"[Trade] Normalizer nuovo — le statistiche verranno costruite durante il training")

    loaded = False
    if os.path.exists(ddpg_best_path):
        loaded = agent.load(ddpg_best_path)
    if not loaded and os.path.exists(ddpg_path):
        agent.load(ddpg_path)

    # ── 6. Addestramento ──────────────────────────────────────────────────────
    es_patience = getattr(buyer_cfg, "es_patience", 20)
    es_metric   = getattr(buyer_cfg, "es_metric",   "sharpe")
    noise_floor = getattr(ddpg_cfg,  "noise_floor",  0.05)

    print(
        f"\nAddestramento DDPG [{freq}] — {buyer_cfg.n_episodes} episodi | "
        f"ES: {es_patience} ep su {es_metric} | "
        f"Ψ: {'attivo' if psi_train is not None else 'non disponibile'} | "
        f"Thermo: {'attivo' if thermo_train is not None else 'non disponibile'} | "
        f"reward shaping: λ_conc={lc} λ_inact={li} λ_thermo={lt} | "
        f"obs_clip={norm_clip}"
    )

    history = train_ddpg_normalized(
        agent=agent,
        env=env_train,
        normalizer=normalizer,
        n_episodes=buyer_cfg.n_episodes,
        warmup=buyer_cfg.warmup,
        log_every=buyer_cfg.log_every,
        noise_decay=ddpg_cfg.noise_decay,
        noise_floor=noise_floor,
        es_patience=es_patience,
        es_metric=es_metric,
        best_path=ddpg_best_path,
        norm_path=norm_path,
    )
    agent.save(ddpg_path, tag="FINAL")
    normalizer.save(norm_path.replace(f"_{freq}", f"_{freq}_final"))

    # ── 7. Valutazione con best checkpoint ────────────────────────────────────
    if os.path.exists(ddpg_best_path):
        agent.load(ddpg_best_path)
    if os.path.exists(norm_path):
        normalizer.load(norm_path)

    print("\nValutazione sul test set...")
    raw_state = env_test.reset()
    state     = normalizer.normalize(raw_state, update=False)
    done      = False
    while not done:
        action               = agent.act(state, explore=False)
        raw_next, _, done, _ = env_test.step(action)
        state                = normalizer.normalize(raw_next, update=False)

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
    print(f"{'─'*54}")

    print("\nRiepilogo P&L per ticker (top 10):")
    print(env_test.summary_per_ticker().head(10).to_string())

    results_dir = cfg.paths.results_dir
    os.makedirs(results_dir, exist_ok=True)

    _save_csv(env_test, tickers, daily, results_dir, freq)
    print(f"\nFile salvati in {results_dir}/:")
    _plot_portfolio_daily(daily, tickers, buyer_cfg.initial_capital, results_dir, freq)
    _plot_learning(history, results_dir, freq)
    _plot_pnl_and_trades(env_test, tickers, daily, results_dir, freq)
    _analyze_thermo_trades(env_test.trade_log_df(), results_dir, freq)

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
    """
    Walk-forward validation completa del sistema DDPG.

    Il CNN predittore è già addestrato e condiviso tra tutti i fold.
    Per ogni fold il DDPG viene riaddestrato from scratch (o in warm-start
    dai pesi del fold precedente, se warm_start=True).

    Flusso per ogni fold i:
      1. Slice di prices_real/prices_pred sulla finestra [tr_s:tr_e] e [te_s:te_e]
      2. Calcolo feature thermo sul subset raw_df corrispondente
      3. Training DDPG sul train window
      4. Valutazione out-of-sample sul test window
      5. Registrazione FoldResult nel WalkForwardReport

    Il report finale include il giudizio di produzione secondo i 5 criteri
    definiti in walk_forward.py.

    warm_start=True → i pesi Actor/Critic del fold precedente vengono
    usati come punto di partenza. Più veloce, potenzialmente più stabile
    in mercati stazionari (ma attenzione a overfitting al regime passato).
    """
    import torch
    from modelli.walk_forward import make_folds, WalkForwardReport, FoldResult
    from modelli.ddpg import DDPGAgent
    from modelli.obs_normalizer import ObsNormalizer, train_ddpg_normalized
    from modelli.trading_env import TradingEnv
    from modelli.thermo_state_builder import ThermoStateBuilder

    freq = cfg.frequency.interval

    # ── 1. Carica predittore e inferenza su tutta la serie ─────────────────
    pred_path = _ckpt(cfg, "pred.pth")
    if not os.path.exists(pred_path):
        print(f"[WalkForward] Predittore non trovato: {pred_path}")
        print(f"  Esegui prima: uv run main.py step=train frequency={freq}")
        return WalkForwardReport(mode=mode)

    checkpoint   = torch.load(pred_path, weights_only=False)
    num_features = checkpoint.get("num_features", df.shape[1])
    predictor    = _load_predictor(cfg, num_features, checkpoint)
    scaler       = checkpoint["scaler"]
    all_columns  = list(df.shape[1] * [""] if False else df.columns)

    print(f"[WalkForward] Inferenza CNN su {len(X_all):,} finestre totali...")
    all_pred_scaled = _predict_batched(predictor, X_all, batch_size=BATCH_SIZE)

    ws          = cfg.prediction.window_size
    stride      = cfg.prediction.stride
    all_dates   = df.index[ws::stride][:len(Y_all)]

    prices_real_all, prices_pred_all = _build_price_dfs(
        all_pred_scaled, Y_all, scaler, all_dates, list(df.columns)
    )

    n_bars = len(prices_real_all)
    folds  = make_folds(
        n_bars        = n_bars,
        n_folds       = n_folds,
        mode          = mode,
        min_train_pct = min_train_pct,
        test_pct      = test_pct,
    )

    if not folds:
        print(f"[WalkForward] Errore: impossibile costruire fold con {n_bars} barre. "
              "Riduci n_folds o min_train_pct.")
        return WalkForwardReport(mode=mode)

    print(
        f"\n[WalkForward] {len(folds)} fold generati | mode={mode} | "
        f"min_train={min_train_pct:.0%} | test={test_pct:.0%}"
    )
    print(f"  Barre totali: {n_bars:,}  |  Freq: {freq}  |  "
          f"warm_start: {'sì' if warm_start else 'no'}")

    # ── 2. Setup comune ────────────────────────────────────────────────────
    buyer_cfg      = cfg.buyer
    ddpg_cfg       = cfg.buyer.ddpg
    checkpoint_dir = cfg.paths.checkpoint_dir
    results_dir    = cfg.paths.results_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir,    exist_ok=True)

    builder = ThermoStateBuilder(interval=freq)
    report  = WalkForwardReport(mode=mode)

    # Parametri env comuni a train e test
    _env_common = dict(
        tickers                    = tickers,
        initial_capital            = buyer_cfg.initial_capital,
        transaction_cost           = buyer_cfg.transaction_cost,
        action_threshold           = getattr(buyer_cfg, "action_threshold",           0.005),
        max_position_pct           = getattr(buyer_cfg, "max_position_pct",           0.20),
        max_holding_steps          = getattr(buyer_cfg, "max_holding_steps",          120),
        forced_sell_cooldown_steps = getattr(buyer_cfg, "forced_sell_cooldown_steps", 20),
        bars_per_year              = getattr(buyer_cfg, "bars_per_year",              98280),
        sell_profit_bonus          = getattr(buyer_cfg, "sell_profit_bonus",          0.002),
        thermo_bonus_sell          = getattr(buyer_cfg, "thermo_bonus_sell",          0.002),
        thermo_bonus_buy           = getattr(buyer_cfg, "thermo_bonus_buy",           0.001),
        thermo_penalty_buy         = getattr(buyer_cfg, "thermo_penalty_buy",         0.003),
        stress_sell_threshold      = getattr(buyer_cfg, "stress_sell_threshold",      1.0),
        stress_buy_threshold       = getattr(buyer_cfg, "stress_buy_threshold",      -0.5),
        efficiency_penalty_thresh  = getattr(buyer_cfg, "efficiency_penalty_thresh",  1.5),
        trust_min                  = getattr(buyer_cfg, "trust_min",                  0.35),
        trust_max                  = getattr(buyer_cfg, "trust_max",                  0.65),
    )

    prev_best_path: str | None = None  # warm-start dal fold precedente

    # ── 3. Loop sui fold ───────────────────────────────────────────────────
    for fold_n, ((tr_s, tr_e), (te_s, te_e)) in enumerate(folds, 1):
        print(f"\n{'═'*62}")
        print(f"  FOLD {fold_n}/{len(folds)} | "
              f"train [{tr_s:,}:{tr_e:,}] ({tr_e-tr_s:,} barre) | "
              f"test [{te_s:,}:{te_e:,}] ({te_e-te_s:,} barre)")

        pr_train = prices_real_all.iloc[tr_s:tr_e]
        pp_train = prices_pred_all.iloc[tr_s:tr_e]
        pr_test  = prices_real_all.iloc[te_s:te_e]
        pp_test  = prices_pred_all.iloc[te_s:te_e]

        print(f"  Train: {pr_train.index[0]} → {pr_train.index[-1]}")
        print(f"  Test:  {pr_test.index[0]} → {pr_test.index[-1]}")

        # Thermo features calcolate sul subset di raw_df corrispondente
        raw_src   = df_raw if df_raw is not None else df
        th_train  = builder.build(raw_src.reindex(pr_train.index), tickers)
        th_test   = builder.build(raw_src.reindex(pr_test.index),  tickers)

        # ── Ambienti ──────────────────────────────────────────────────────
        env_train = TradingEnv(
            prices_real=pr_train, prices_pred=pp_train,
            thermo_df=th_train, log_trades=False,
            lambda_concentration = getattr(buyer_cfg, "lambda_concentration", 0.1),
            lambda_inaction      = getattr(buyer_cfg, "lambda_inaction",      0.1),
            lambda_loss          = getattr(buyer_cfg, "lambda_loss",          0.5),
            loss_threshold       = getattr(buyer_cfg, "loss_threshold",      -0.5),
            **_env_common,
        )
        env_test = TradingEnv(
            prices_real=pr_test, prices_pred=pp_test,
            thermo_df=th_test, log_trades=True,
            lambda_concentration = 0.0,
            lambda_inaction      = 0.0,
            lambda_loss          = 0.0,
            loss_threshold       = getattr(buyer_cfg, "loss_threshold", -0.5),
            **_env_common,
        )

        # ── Agente DDPG ───────────────────────────────────────────────────
        agent = DDPGAgent(
            state_dim       = env_train.state_dim,
            action_dim      = env_train.action_dim,
            actor_hidden    = list(ddpg_cfg.actor_hidden),
            critic_hidden   = list(ddpg_cfg.critic_hidden),
            lr_actor        = ddpg_cfg.lr_actor,
            lr_critic       = ddpg_cfg.lr_critic,
            gamma           = ddpg_cfg.gamma,
            tau             = ddpg_cfg.tau,
            buffer_capacity = buyer_cfg.buffer_capacity,
            batch_size      = buyer_cfg.batch_size,
            update_every    = ddpg_cfg.update_every,
            noise_sigma     = ddpg_cfg.noise_sigma,
        )
        normalizer = ObsNormalizer(
            state_dim = env_train.state_dim,
            clip      = getattr(buyer_cfg, "obs_clip", 5.0),
        )

        # Warm start: carica pesi dal fold precedente
        if warm_start and prev_best_path and os.path.exists(prev_best_path):
            ok = agent.load(prev_best_path)
            if ok:
                print(f"  Warm start da fold {fold_n - 1}: {os.path.basename(prev_best_path)}")

        fold_best = os.path.join(checkpoint_dir, f"ddpg_wf_fold{fold_n}_best_{freq}.pth")
        fold_norm = os.path.join(checkpoint_dir, f"normalizer_wf_fold{fold_n}_{freq}.npz")

        # ── Training ──────────────────────────────────────────────────────
        history = train_ddpg_normalized(
            agent       = agent,
            env         = env_train,
            normalizer  = normalizer,
            n_episodes  = buyer_cfg.n_episodes,
            warmup      = buyer_cfg.warmup,
            log_every   = max(1, buyer_cfg.log_every * 5),  # log meno frequente nei fold
            noise_decay = ddpg_cfg.noise_decay,
            noise_floor = getattr(ddpg_cfg, "noise_floor", 0.02),
            es_patience = getattr(buyer_cfg, "es_patience", 50),
            es_metric   = getattr(buyer_cfg, "es_metric",   "sharpe"),
            best_path   = fold_best,
            norm_path   = fold_norm,
        )

        # ── Valutazione con best checkpoint ───────────────────────────────
        if os.path.exists(fold_best):
            agent.load(fold_best)
        if os.path.exists(fold_norm):
            normalizer.load(fold_norm)

        raw_state = env_test.reset()
        state     = normalizer.normalize(raw_state, update=False)
        done      = False
        while not done:
            action               = agent.act(state, explore=False)
            raw_next, _, done, _ = env_test.step(action)
            state                = normalizer.normalize(raw_next, update=False)

        # ── Metriche fold ─────────────────────────────────────────────────
        final_v  = env_test.portfolio_history()[-1]
        ret_pct  = (final_v - buyer_cfg.initial_capital) / buyer_cfg.initial_capital * 100
        sharpe   = env_test.sharpe_ratio()
        max_dd   = env_test.max_drawdown()
        best_ep  = max(history, key=lambda h: h.get("sharpe", -999))

        # Thermo post-hoc
        sell_ok_pct = 0.0
        buy_bad_pct = 0.0
        tlog = env_test.trade_log_df()
        if not tlog.empty and "thermo_stress" in tlog.columns:
            sells = tlog[tlog["action"] == "SELL"]
            buys  = tlog[tlog["action"] == "BUY"]
            if len(sells) > 0:
                sell_ok_pct = 100 * float((sells["thermo_stress"] > 1.0).mean())
            if len(buys) > 0:
                buy_bad_pct = 100 * float((buys.get("thermo_efficiency", pd.Series(dtype=float)) > 1.5).mean())

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
            thermo_sell_ok_pct = sell_ok_pct,
            thermo_buy_bad_pct = buy_bad_pct,
        )
        report.folds.append(fold_res)
        prev_best_path = fold_best

        print(
            f"  ✔ Fold {fold_n} — Sharpe: {sharpe:.3f} | "
            f"Return: {ret_pct:+.2f}% | MaxDD: {max_dd:.3f} | "
            f"Thm✅: {sell_ok_pct:.1f}% | ep: {len(history)}"
        )

        # Salva anche il trade log del fold per analisi
        tlog_path = os.path.join(results_dir, f"wf_fold{fold_n}_trades_{freq}.csv")
        if not tlog.empty:
            tlog.to_csv(tlog_path)

    # ── 4. Report finale ───────────────────────────────────────────────────
    report.print_summary()
    report.save(os.path.join(results_dir, f"walk_forward_{freq}.csv"))

    # Thermo post-hoc aggregata su tutti i fold
    _analyze_thermo_trades(
        pd.concat([
            pd.read_csv(os.path.join(results_dir, f"wf_fold{i}_trades_{freq}.csv"),
                        index_col=0, parse_dates=True)
            for i in range(1, len(folds) + 1)
            if os.path.exists(os.path.join(results_dir, f"wf_fold{i}_trades_{freq}.csv"))
        ]) if folds else pd.DataFrame(),
        results_dir,
        f"wf_aggregated_{freq}",
    )

    return report