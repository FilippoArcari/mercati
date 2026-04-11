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
        max_grad_norm    = getattr(cfg.training, "max_grad_norm", 1.0),
    )
    predictor.load_state_dict(checkpoint["model_state_dict"])
    predictor.eval()

    from modelli.utils import get_device
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


def _extract_psi(df: pd.DataFrame, tickers: list[str]):
    psi_cols = [f"Psi_{t}" for t in tickers if f"Psi_{t}" in df.columns]
    return df[psi_cols] if psi_cols else None


def _build_portfolio_daily(env: TradingEnv, tickers: list[str]) -> pd.DataFrame:
    """
    Ricostruisce un DataFrame giornaliero dal portfolio_history dell'env.
    TradingEnv non espone value_per_ticker_df / trade_log_df nella versione
    corrente, quindi operiamo sul solo storico del valore totale.
    """
    ph = env.portfolio_history()
    if len(ph) < 2:
        return pd.DataFrame()

    # Usa l'indice del df dell'env (tronca se necessario)
    idx = env.df.index[: len(ph)]
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


# ─── helper: costruisce thermo_stress_fn per un env ──────────────────────────

def _make_thermo_stress_fn(env: TradingEnv):
    """
    Ritorna una Callable(step) → float che legge Thm_Stress dall'env
    al passo dato, oppure None se il thermo non è disponibile.
    """
    if env.thermo_df is None or "Thm_Stress" not in env.thermo_df.columns:
        return None
    return lambda step: float(env.thermo_df.iloc[
        min(step, len(env.thermo_df) - 1)
    ]["Thm_Stress"])


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


def _plot_trades(env, results_dir, freq):
    """Grafico degli scambi registrati nell'env."""
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
    if "BUY" in top10.columns:
        colors.append("#2ecc71")
    if "SELL" in top10.columns:
        colors.append("#e74c3c")
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


# ─── entry point ─────────────────────────────────────────────────────────────

def run_trade(
    cfg,
    df: pd.DataFrame,
    tickers: list[str],
    X_train, Y_train,
    X_test,  Y_test,
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

    checkpoint   = torch.load(pred_path, map_location=torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'), weights_only=False)
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

    # ── 3. Feature termodinamiche ─────────────────────────────────────────────
    if df_raw is not None:
        raw_train_df, raw_test_df = _split_raw(df_raw, train_df.index, test_df.index)
    else:
        raw_train_df, raw_test_df = train_df, test_df

    from modelli.thermo_state_builder import ThermoStateBuilder
    builder      = ThermoStateBuilder(interval=cfg.frequency.interval)
    thermo_train = builder.build(raw_train_df, tickers)
    thermo_test  = builder.build(raw_test_df,  tickers)

    # ── 4. Ambienti ───────────────────────────────────────────────────────────
    buyer_cfg = cfg.buyer

    env_train = TradingEnv(
        df               = prices_real_train,
        tickers          = tickers,
        initial_capital  = buyer_cfg.initial_capital,
        fee_pct          = buyer_cfg.transaction_cost,
        thermo_df        = thermo_train,
        thermo_bonus_sell  = getattr(buyer_cfg, "thermo_bonus_sell",  0.5),
        thermo_penalty_buy = getattr(buyer_cfg, "thermo_penalty_buy", 0.1),
    )
    env_test = TradingEnv(
        df               = prices_real_test,
        tickers          = tickers,
        initial_capital  = buyer_cfg.initial_capital,
        fee_pct          = buyer_cfg.transaction_cost,
        thermo_df        = thermo_test,
        thermo_bonus_sell  = getattr(buyer_cfg, "thermo_bonus_sell",  0.5),
        thermo_penalty_buy = getattr(buyer_cfg, "thermo_penalty_buy", 0.1),
    )

    # ── 5. Agente DDPG ────────────────────────────────────────────────────────
    ddpg_path      = _ckpt(cfg, "ddpg.pth")
    ddpg_best_path = _ckpt(cfg, "ddpg_best.pth")
    norm_path      = _ckpt(cfg, "normalizer.npz")
    ddpg_cfg       = cfg.buyer.ddpg

    agent = DDPGAgent(
        state_dim      = env_train.observation_space.shape[0],
        action_dim     = env_train.action_space.shape[0],
        actor_hidden   = list(ddpg_cfg.actor_hidden),
        critic_hidden  = list(ddpg_cfg.critic_hidden),
        lr_actor       = ddpg_cfg.lr_actor,
        lr_critic      = ddpg_cfg.lr_critic,
        gamma          = ddpg_cfg.gamma,
        tau            = ddpg_cfg.tau,
        buffer_capacity= buyer_cfg.buffer_capacity,
        batch_size     = buyer_cfg.batch_size,
        update_every   = ddpg_cfg.update_every,
        noise_sigma    = ddpg_cfg.noise_sigma,
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
    noise_floor = getattr(ddpg_cfg,  "noise_floor",  0.05)

    print(
        f"\nAddestramento DDPG [{freq}] — {buyer_cfg.n_episodes} episodi | "
        f"ES: {es_patience} ep su {es_metric} | "
        f"Thermo: {'attivo' if thermo_train is not None else 'no'} | "
        f"obs_clip={norm_clip}"
    )

    stress_fn = _make_thermo_stress_fn(env_train)

    history = train_ddpg_normalized(
        agent         = agent,
        env           = env_train,
        normalizer    = normalizer,
        n_episodes    = buyer_cfg.n_episodes,
        norm_path = norm_path,
        log_every     = buyer_cfg.log_every,
    )
    agent.save(ddpg_path, tag="FINAL")

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
    """
    Walk-forward validation completa del sistema DDPG.
    """
    import torch
    from modelli.walk_forward import make_folds, WalkForwardReport, FoldResult
    from modelli.ddpg import DDPGAgent, ThermoEnsemble, compute_thermo_profile
    from modelli.obs_normalizer import ObsNormalizer, train_ddpg_normalized
    from modelli.trading_env import TradingEnv
    from modelli.thermo_state_builder import ThermoStateBuilder

    freq = cfg.frequency.interval

    # ── 1. Carica predittore e inferenza sull'intera serie ─────────────────
    pred_path = _ckpt(cfg, "pred.pth")
    if not os.path.exists(pred_path):
        print(f"[WalkForward] Predittore non trovato: {pred_path}")
        return WalkForwardReport(mode=mode)

    checkpoint   = torch.load(pred_path, map_location=torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'), weights_only=False)
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

    # ── 2. Setup comune ────────────────────────────────────────────────────
    buyer_cfg      = cfg.buyer
    ddpg_cfg       = cfg.buyer.ddpg
    checkpoint_dir = cfg.paths.checkpoint_dir
    results_dir    = cfg.paths.results_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir,    exist_ok=True)

    builder = ThermoStateBuilder(interval=freq)
    report  = WalkForwardReport(mode=mode)

    # Parametri comuni a tutti i TradingEnv — solo quelli che il costruttore accetta
    _env_common = dict(
        tickers            = tickers,
        initial_capital    = buyer_cfg.initial_capital,
        fee_pct            = buyer_cfg.transaction_cost,
        thermo_bonus_sell  = getattr(buyer_cfg, "thermo_bonus_sell",  0.5),
        thermo_penalty_buy = getattr(buyer_cfg, "thermo_penalty_buy", 0.1),
    )

    prev_best_path:  str | None          = None
    prev_thermo_df:  pd.DataFrame | None = None

    # ── Raccolta dati per ThermoEnsemble ──────────────────────────────────
    ensemble_agents:   list[DDPGAgent]     = []
    ensemble_profiles: list[np.ndarray]   = []
    ensemble_norms:    list[ObsNormalizer] = []

    # ── KL divergence termodinamica ────────────────────────────────────────
    def _thermo_kl_div(df_prev: pd.DataFrame, df_curr: pd.DataFrame) -> float:
        from scipy.special import rel_entr
        cols    = ["Thm_Stress", "Thm_Entropy"]
        kl_sum  = 0.0
        n_valid = 0
        for col in cols:
            if col not in df_prev.columns or col not in df_curr.columns:
                continue
            vals_p = df_prev[col].dropna().values
            vals_c = df_curr[col].dropna().values
            if len(vals_p) < 20 or len(vals_c) < 20:
                continue
            lo   = min(vals_p.min(), vals_c.min())
            hi   = max(vals_p.max(), vals_c.max())
            bins = np.linspace(lo, hi, 21)
            hp, _ = np.histogram(vals_p, bins=bins, density=True)
            hc, _ = np.histogram(vals_c, bins=bins, density=True)
            hp    = (hp + 1e-10) / (hp + 1e-10).sum()
            hc    = (hc + 1e-10) / (hc + 1e-10).sum()
            kl_sum  += float(np.sum(rel_entr(hc, hp)))
            n_valid += 1
        return kl_sum / max(n_valid, 1)

    # ── 3. Loop sui fold ───────────────────────────────────────────────────
    for fold_n, ((tr_s, tr_e), (te_s, te_e)) in enumerate(folds, 1):
        print(f"\n{'═'*62}")
        print(f"  FOLD {fold_n}/{len(folds)} | "
              f"train [{tr_s:,}:{tr_e:,}] ({tr_e-tr_s:,} barre) | "
              f"test [{te_s:,}:{te_e:,}] ({te_e-te_s:,} barre)")

        pr_train = prices_real_all.iloc[tr_s:tr_e]
        pr_test  = prices_real_all.iloc[te_s:te_e]

        print(f"  Train: {pr_train.index[0]} → {pr_train.index[-1]}")
        print(f"  Test:  {pr_test.index[0]} → {pr_test.index[-1]}")

        raw_src  = df_raw if df_raw is not None else df
        th_train = builder.build(raw_src.reindex(pr_train.index), tickers)
        th_test  = builder.build(raw_src.reindex(pr_test.index),  tickers)

        # ── Ambienti ──────────────────────────────────────────────────────
        env_train = TradingEnv(
            df        = pr_train,
            thermo_df = th_train,
            **_env_common,
        )
        env_test = TradingEnv(
            df        = pr_test,
            thermo_df = th_test,
            **_env_common,
        )

        # ── Agente ────────────────────────────────────────────────────────
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
            clip      = getattr(buyer_cfg, "obs_clip", 5.0),
        )

        # Warm start condizionale con check KL
        if warm_start and prev_best_path and os.path.exists(prev_best_path):
            use_warm = True
            kl_thr   = getattr(buyer_cfg, "warm_start_kl_threshold", 2.0)
            if prev_thermo_df is not None and th_train is not None and not th_train.empty:
                kl = _thermo_kl_div(prev_thermo_df, th_train)
                if kl > kl_thr:
                    print(f"  [WarmStart] KL={kl:.3f} > {kl_thr} → regime diverso, skip warm start")
                    use_warm = False
                else:
                    print(f"  [WarmStart] KL={kl:.3f} ≤ {kl_thr} → warm start OK")
            if use_warm:
                agent.load(prev_best_path)
                print(f"  Warm start da fold {fold_n - 1}: {os.path.basename(prev_best_path)}")

        fold_best = os.path.join(checkpoint_dir, f"ddpg_wf_fold{fold_n}_best_{freq}.pth")
        fold_norm = os.path.join(checkpoint_dir, f"normalizer_wf_fold{fold_n}_{freq}.npz")

        stress_fn = _make_thermo_stress_fn(env_train)

        # ── Training ──────────────────────────────────────────────────────
        history = train_ddpg_normalized(
            agent             = agent,
            env               = env_train,
            normalizer        = normalizer,
            n_episodes        = buyer_cfg.n_episodes,
            warmup            = buyer_cfg.warmup,
            log_every         = max(1, buyer_cfg.log_every * 5),
            noise_decay       = ddpg_cfg.noise_decay,
            noise_floor       = getattr(ddpg_cfg, "noise_floor", 0.02),
            es_patience       = getattr(buyer_cfg, "es_patience", 50),
            es_metric         = getattr(buyer_cfg, "es_metric",   "sharpe"),
            best_path         = fold_best,
            norm_path         = fold_norm,
            episodic_episodes = getattr(buyer_cfg, "episodic_episodes", 20),
            thermo_stress_fn  = stress_fn,
            use_curriculum    = True,
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
        final_v = env_test.portfolio_history()[-1]
        ret_pct = (final_v - buyer_cfg.initial_capital) / buyer_cfg.initial_capital * 100
        sharpe  = env_test.sharpe_ratio()
        max_dd  = env_test.max_drawdown()
        best_ep = max(history, key=lambda h: h.get("sharpe", -999))

        from modelli.walk_forward import FoldResult
        fold_res = FoldResult(
            fold         = fold_n,
            train_bars   = tr_e - tr_s,
            test_bars    = te_e - te_s,
            train_start  = str(pr_train.index[0])[:16],
            train_end    = str(pr_train.index[-1])[:16],
            test_start   = str(pr_test.index[0])[:16],
            test_end     = str(pr_test.index[-1])[:16],
            sharpe       = sharpe,
            total_return_pct = ret_pct,
            max_drawdown = max_dd,
            n_episodes_run = len(history),
            best_episode = best_ep["episode"],
        )
        report.folds.append(fold_res)
        prev_best_path = fold_best
        prev_thermo_df = th_train if (th_train is not None and not th_train.empty) else None

        print(
            f"  ✔ Fold {fold_n} — Sharpe: {sharpe:.3f} | "
            f"Return: {ret_pct:+.2f}% | MaxDD: {max_dd:.3f} | ep: {len(history)}"
        )

        # Accumula dati per ThermoEnsemble
        if th_train is not None and not th_train.empty:
            try:
                profile = compute_thermo_profile(th_train, 0, len(th_train))
                ensemble_agents.append(agent)
                ensemble_profiles.append(profile)
                ensemble_norms.append(normalizer)
            except Exception as e:
                print(f"  [Ensemble] Profilo fold {fold_n} non calcolabile: {e}")

    # ── 4. Report finale ───────────────────────────────────────────────────
    report.print_summary()
    report.save(os.path.join(results_dir, f"walk_forward_{freq}.csv"))

    # ── 5. ThermoEnsemble — valutazione post walk-forward ──────────────────
    if len(ensemble_agents) >= 2 and folds:
        print(f"\n{'─'*62}")
        print(f"  THERMO ENSEMBLE — {len(ensemble_agents)} agenti")

        last_te_s, last_te_e = folds[-1][1]
        pr_test_last = prices_real_all.iloc[last_te_s:last_te_e]
        raw_src      = df_raw if df_raw is not None else df
        th_test_last = builder.build(raw_src.reindex(pr_test_last.index), tickers)

        env_ensemble = TradingEnv(
            df        = pr_test_last,
            thermo_df = th_test_last,
            **_env_common,
        )

        from modelli.ddpg import ThermoEnsemble
        ensemble = ThermoEnsemble(
            agents        = ensemble_agents,
            fold_profiles = ensemble_profiles,
            temperature   = getattr(buyer_cfg, "ensemble_temperature", 0.5),
        )

        thm_cols = (
            [c for c in th_test_last.columns if c.startswith("Thm_")][:6]
            if th_test_last is not None and not th_test_last.empty
            else []
        )

        raw_state = env_ensemble.reset()
        last_norm = ensemble_norms[-1]
        state     = last_norm.normalize(raw_state, update=False)
        done      = False
        step_idx  = 0

        while not done:
            if thm_cols and th_test_last is not None and step_idx < len(th_test_last):
                current_thermo = th_test_last.iloc[step_idx][thm_cols].values.astype(np.float32)
            else:
                current_thermo = np.zeros(len(ensemble_profiles[0]), dtype=np.float32)

            action               = ensemble.act(state, current_thermo, explore=False)
            raw_next, _, done, _ = env_ensemble.step(action)
            state                = last_norm.normalize(raw_next, update=False)
            step_idx            += 1

        ens_final  = env_ensemble.portfolio_history()[-1]
        ens_ret    = (ens_final - buyer_cfg.initial_capital) / buyer_cfg.initial_capital * 100
        ens_sharpe = env_ensemble.sharpe_ratio()
        ens_dd     = env_ensemble.max_drawdown()

        print(f"  ThermoEnsemble (ultimo test set):")
        print(f"    Sharpe: {ens_sharpe:.3f} | Return: {ens_ret:+.2f}% | MaxDD: {ens_dd:.3f}")
        print(f"    Pesi finali: {ensemble.weights_summary()}")

        best_fold = max(report.folds, key=lambda f: f.sharpe)
        print(f"    Best fold singolo (fold {best_fold.fold}): Sharpe {best_fold.sharpe:.3f}")
        if ens_sharpe > best_fold.sharpe:
            print(f"    → Ensemble SUPERIORE al best fold (+{ens_sharpe - best_fold.sharpe:.3f})")
        else:
            print(f"    → Ensemble inferiore al best fold ({ens_sharpe - best_fold.sharpe:.3f})")

        ens_df = pd.DataFrame([{
            "sharpe":      ens_sharpe,
            "return_pct":  ens_ret,
            "max_dd":      ens_dd,
            "n_agents":    len(ensemble_agents),
            "temperature": getattr(buyer_cfg, "ensemble_temperature", 0.5),
        }])
        ens_df.to_csv(os.path.join(results_dir, f"thermo_ensemble_{freq}.csv"), index=False)
        print(f"  → CSV salvato: thermo_ensemble_{freq}.csv")
        print(f"{'─'*62}")

    return report