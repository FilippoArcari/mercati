"""
modelli/trade.py — Addestramento e valutazione dell'agente DDPG

Espone un'unica funzione pubblica:

    run_trade(cfg, df, tickers, X_train, Y_train, X_test, Y_test)

Chiamata da main.py nel blocco  elif cfg.step == "trade".
"""

from __future__ import annotations

import os
import pandas as pd
import torch
import matplotlib.pyplot as plt

from modelli.pred import Pred
from modelli.ddpg import DDPGAgent, train_ddpg
from modelli.trading_env import TradingEnv


# ─── helper privato ───────────────────────────────────────────────────────────

def _load_predictor(cfg, num_features: int, checkpoint: dict) -> Pred:
    predictor = Pred(
        num_features=num_features,
        window_size=cfg.prediction.window_size,
        dimension=list(cfg.model.dimensions),
        dilations=list(cfg.model.dilations),
        kernel_size=cfg.model.kernel_size,
        activation=cfg.model.activation,
    )
    predictor.load_state_dict(checkpoint["model_state_dict"])
    predictor.eval()
    return predictor


def _build_price_dfs(
    pred_scaled:  object,   # np.ndarray
    real_targets: object,   # torch.Tensor
    scaler,
    dates:        pd.Index,
    all_columns:  list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Denormalizza e costruisce (prices_real, prices_pred) come DataFrame."""
    import numpy as np
    real = scaler.inverse_transform(
        real_targets.numpy() if hasattr(real_targets, "numpy") else real_targets
    )
    pred = scaler.inverse_transform(pred_scaled)
    return (
        pd.DataFrame(real, index=dates, columns=all_columns),
        pd.DataFrame(pred, index=dates, columns=all_columns),
    )


def _save_csv(env: TradingEnv, results_dir: str) -> None:
    env.trade_log_df().to_csv(os.path.join(results_dir, "trade_log.csv"))
    env.value_per_ticker_df().to_csv(os.path.join(results_dir, "value_per_ticker.csv"))
    env.summary_per_ticker().to_csv(os.path.join(results_dir, "summary_per_ticker.csv"))
    print(f"\nCSV salvati in {results_dir}/:")
    print("  trade_log.csv           — ogni operazione BUY/SELL/HOLD")
    print("  value_per_ticker.csv    — snapshot giornaliero per ticker")
    print("  summary_per_ticker.csv  — riepilogo finale P&L per ticker")


def _plot_results(
    env:         TradingEnv,
    history:     list[dict],
    initial_capital: float,
    results_dir: str,
) -> None:
    # ── 1. Portafoglio + curva di apprendimento ────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(13, 8))

    pf = env.portfolio_history()
    axes[0].plot(pf, color="steelblue", label="Portfolio value")
    axes[0].axhline(initial_capital, color="gray", linestyle="--", label="Capitale iniziale")
    axes[0].set_title("Valore del portafoglio — test set")
    axes[0].set_ylabel("USD")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    ep_rewards = [h["total_reward"] for h in history]
    axes[1].plot(ep_rewards, color="darkorange", alpha=0.6, label="Reward per episodio")
    if len(ep_rewards) >= 20:
        ma = pd.Series(ep_rewards).rolling(20).mean()
        axes[1].plot(ma, color="red", linewidth=2, label="Media mobile 20 ep.")
    axes[1].set_title("Curva di apprendimento (training)")
    axes[1].set_xlabel("Episodio")
    axes[1].set_ylabel("Reward totale")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "ddpg_results.png"), dpi=150)
    plt.close()

    # ── 2. P&L non realizzato — top 5 ticker ──────────────────────────────────
    value_log = env.value_per_ticker_df()
    summary   = env.summary_per_ticker()

    if not value_log.empty and not summary.empty:
        top5 = summary.head(5).index.tolist()
        fig, ax = plt.subplots(figsize=(13, 5))
        for t in top5:
            col = f"{t}_unrealized_pnl"
            if col in value_log.columns:
                ax.plot(value_log.index, value_log[col], label=t)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title("P&L non realizzato — top 5 ticker per P&L totale")
        ax.set_ylabel("USD")
        ax.legend(ncol=3, fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "pnl_per_ticker.png"), dpi=150)
        plt.close()

    # ── 3. Distribuzione BUY / SELL — top 10 ticker più attivi ────────────────
    trade_log = env.trade_log_df()
    if not trade_log.empty:
        action_counts = (
            trade_log[trade_log["action"] != "HOLD"]
            .groupby(["ticker", "action"])
            .size()
            .unstack(fill_value=0)
        )
        action_counts["total"] = action_counts.sum(axis=1)
        top10 = action_counts.nlargest(10, "total").drop(columns="total")

        fig, ax = plt.subplots(figsize=(13, 5))
        top10.plot(kind="bar", ax=ax, color=["#e74c3c", "#2ecc71"])
        ax.set_title("Numero operazioni BUY / SELL — top 10 ticker più attivi")
        ax.set_ylabel("Numero operazioni")
        ax.legend(title="Azione")
        ax.grid(True, alpha=0.3, axis="y")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "action_distribution.png"), dpi=150)
        plt.close()

    print("\nGrafici generati:")
    print("  ddpg_results.png        — portafoglio + curva di apprendimento")
    print("  pnl_per_ticker.png      — P&L non realizzato top 5 ticker")
    print("  action_distribution.png — operazioni BUY/SELL per ticker")


# ─── funzione pubblica ────────────────────────────────────────────────────────

def run_trade(
    cfg,
    df:       "pd.DataFrame",
    tickers:  list[str],
    X_train:  "torch.Tensor",
    Y_train:  "torch.Tensor",
    X_test:   "torch.Tensor",
    Y_test:   "torch.Tensor",
    train_df: "pd.DataFrame",
    test_df:  "pd.DataFrame",
) -> None:
    """
    Addestra il DDPG e lo valuta sul test set.

    Parameters
    ----------
    cfg      : OmegaConf DictConfig (config Hydra completo)
    df       : DataFrame completo normalizzato (tickers + inflation)
    tickers  : lista dei ticker negoziabili
    X_train, Y_train : finestre e target del training set (torch.Tensor)
    X_test,  Y_test  : finestre e target del test set     (torch.Tensor)
    train_df, test_df: DataFrame sliced pre/post split_date
    """
    # ── 1. Carica il modello predittivo ───────────────────────────────────────
    model_name = f"pred_{cfg.frequency.interval}_{cfg.prediction.window_size}"
    checkpoint_path = os.path.join(cfg.paths.checkpoint_dir, f"{model_name}.pth")
    if not os.path.exists(checkpoint_path):
        print("Errore: esegui prima  step=train  per addestrare il modello predittivo.")
        return

    checkpoint   = torch.load(checkpoint_path, weights_only=False)
    num_features = checkpoint.get("num_features", df.shape[1])
    predictor    = _load_predictor(cfg, num_features, checkpoint)
    scaler       = checkpoint["scaler"]

    # ── 2. Genera predizioni denormalizzate ───────────────────────────────────
    with torch.no_grad():
        train_pred_scaled = predictor(X_train).numpy()
        test_pred_scaled  = predictor(X_test).numpy()

    all_columns = list(df.columns)
    train_dates = train_df.index[cfg.prediction.window_size:]
    test_dates  = test_df.index[cfg.prediction.window_size:]

    prices_real_train, prices_pred_train = _build_price_dfs(
        train_pred_scaled, Y_train, scaler, train_dates, all_columns
    )
    prices_real_test, prices_pred_test = _build_price_dfs(
        test_pred_scaled, Y_test, scaler, test_dates, all_columns
    )

    # ── 3. Costruisce gli ambienti ────────────────────────────────────────────
    buyer_cfg = cfg.buyer

    env_train = TradingEnv(
        prices_real=prices_real_train,
        prices_pred=prices_pred_train,
        tickers=tickers,
        initial_capital=buyer_cfg.initial_capital,
        transaction_cost=buyer_cfg.transaction_cost,
    )
    env_test = TradingEnv(
        prices_real=prices_real_test,
        prices_pred=prices_pred_test,
        tickers=tickers,
        initial_capital=buyer_cfg.initial_capital,
        transaction_cost=buyer_cfg.transaction_cost,
    )

    # ── 4. Costruisce l'agente DDPG ───────────────────────────────────────────
    ddpg_path = os.path.join(cfg.paths.checkpoint_dir, f"ddpg_{cfg.frequency.interval}_{cfg.prediction.window_size}.pth")

    agent = DDPGAgent(
        state_dim=env_train.state_dim,
        action_dim=env_train.action_dim,
        actor_hidden=list(buyer_cfg.actor_hidden),
        critic_hidden=list(buyer_cfg.critic_hidden),
        lr_actor=buyer_cfg.lr_actor,
        lr_critic=buyer_cfg.lr_critic,
        gamma=buyer_cfg.gamma,
        tau=buyer_cfg.tau,
        buffer_capacity=buyer_cfg.buffer_capacity,
        batch_size=buyer_cfg.batch_size,
        update_every=buyer_cfg.update_every,
        noise_sigma=buyer_cfg.noise_sigma,
    )

    if os.path.exists(ddpg_path):
        print(f"[DDPG] Riprendo da checkpoint: {ddpg_path}")
        agent.load(ddpg_path)

    # ── 5. Addestramento ──────────────────────────────────────────────────────
    print(f"\nAddestramento DDPG ({buyer_cfg.n_episodes} episodi)...")
    history = train_ddpg(
        agent=agent,
        env=env_train,
        n_episodes=buyer_cfg.n_episodes,
        warmup=buyer_cfg.warmup,
        log_every=buyer_cfg.log_every,
    )
    agent.save(ddpg_path)

    # ── 6. Valutazione sul test set ───────────────────────────────────────────
    print("\nValutazione sul test set (senza esplorazione)...")
    state = env_test.reset()
    done  = False
    while not done:
        action            = agent.act(state, explore=False)
        state, _, done, _ = env_test.step(action)

    # ── 7. Metriche a console ─────────────────────────────────────────────────
    pf_test = env_test.portfolio_history()
    final_v = pf_test[-1]
    ret_pct = (final_v - buyer_cfg.initial_capital) / buyer_cfg.initial_capital * 100

    print(f"\n{'─'*52}")
    print(f"  Capitale iniziale : ${buyer_cfg.initial_capital:>10,.0f}")
    print(f"  Valore finale     : ${final_v:>10,.2f}")
    print(f"  Rendimento        : {ret_pct:>+10.2f}%")
    print(f"  Sharpe ratio      : {env_test.sharpe_ratio():>10.3f}")
    print(f"  Max Drawdown      : {env_test.max_drawdown():>10.3f}")
    print(f"{'─'*52}")

    summary = env_test.summary_per_ticker()
    print("\nRiepilogo P&L per ticker (top 10):")
    print(summary.head(10).to_string())

    # ── 8. Salvataggio CSV e grafici ──────────────────────────────────────────
    results_dir = cfg.paths.results_dir
    os.makedirs(results_dir, exist_ok=True)

    _save_csv(env_test, results_dir)
    _plot_results(env_test, history, buyer_cfg.initial_capital, results_dir)