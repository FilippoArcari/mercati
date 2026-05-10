from omegaconf import DictConfig, OmegaConf, ListConfig
import datetime
import pandas as pd
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStoppingReason
from modelli.predictor.plotter import plot_reconstruction
import torch
from torch.utils.data import DataLoader, TensorDataset
import os
import hydra
import sys
from typing import Optional
import numpy as np
from modelli.trader.traiding_env import MultiTickerTradingEnv
from modelli.trader.ppo import PPOTrader
from modelli.predictor.predictor import Predictor
from modelli.predictor.get_data import get_data
from modelli.features.thermodynamic import plot_thermodynamic, THERMO_N_FEATURES
from dotenv import load_dotenv
import yfinance as yf
import lightning as L
import optuna
import copy
import yaml
from optuna.integration import PyTorchLightningPruningCallback

load_dotenv()
NUM_WORKERS = 11


def make_windows(df_x, horizon, sliding_windows, stride, n_target_cols):
    if len(df_x) < sliding_windows + horizon:
        raise ValueError(
            f"Split too small: {len(df_x)} rows < sliding_window ({sliding_windows}) "
            f"+ horizon ({horizon}). Reduce sliding_window or use more data."
        )

    X_tensor = torch.tensor(df_x.values, dtype=torch.float32)
    n_rows   = len(X_tensor)

    X_windows = X_tensor.unfold(0, sliding_windows, stride).permute(0, 2, 1)
    n_windows = X_windows.shape[0]

    valid_indices = [
        i for i in range(n_windows)
        if i * stride + sliding_windows + horizon <= n_rows
    ]
    X_valid = X_windows[valid_indices]

    y_valid = torch.stack([
        X_tensor[
            i * stride + sliding_windows : i * stride + sliding_windows + horizon,
            :n_target_cols,
        ]
        for i in valid_indices
    ])
    valid_dates = df_x.index[[i * stride + sliding_windows - 1 for i in valid_indices]]
    return X_valid, y_valid, valid_dates


@hydra.main(version_base=None, config_path="config", config_name="config")
def my_app(cfg: DictConfig) -> None:
    if not torch.cuda.is_available():
        print("CUDA is not available. Using CPU for training.")
        torch.set_num_threads(os.cpu_count())

    df, n_context_features, scaler = get_data(
        cfg.data.tickers,
        cfg.data.start_date,
        cfg.data.end_date,
        cfg.frequency,
        cfg.data.inflation_series,
        cfg.data.split_train,
    )
    safe_config = OmegaConf.to_container(cfg, resolve=True)
    safe_config["prediction"]["n_features"]          = df.shape[1]
    safe_config["prediction"]["n_context_features"]  = n_context_features
    safe_config["prediction"]["real_features"]       = (
        safe_config["prediction"]["n_features"] - safe_config["prediction"]["n_context_features"]
    )
    predictor_config = safe_config["prediction"]

    # ── Split dataset ─────────────────────────────────────────────────────────
    rows             = df.shape[0]
    split_validation = int(rows * cfg.data.split_validation)
    split_test       = int(rows * cfg.data.split_test)

    df_train      = df.iloc[: -(split_validation + split_test)]
    df_validation = df.iloc[-(split_validation + split_test) : -split_test]
    df_test       = df.iloc[-split_test:]

    print(f"Train samples:      {len(df_train)}")
    print(f"Validation samples: {len(df_validation)}")
    print(f"Test samples:       {len(df_test)}")

    # ── Windows ───────────────────────────────────────────────────────────────
    horizon         = cfg.prediction.forecast_horizon
    sliding_windows = cfg.prediction.sliding_window
    stride          = cfg.prediction.stride

    X_train,      y_train, _          = make_windows(df_train,      horizon, sliding_windows, stride, predictor_config["real_features"])
    X_validation, y_validation, _     = make_windows(df_validation, horizon, sliding_windows, stride, predictor_config["real_features"])
    X_test,       y_test, test_dates  = make_windows(df_test,       horizon, sliding_windows, stride, predictor_config["real_features"])

    # ══════════════════════════════════════════════════════════════════════════
    # OPTIMIZE
    # ══════════════════════════════════════════════════════════════════════════

    if cfg.step == "optimize":
        n_trials    = cfg.get("optuna", {}).get("n_trials",    50)
        max_epochs  = cfg.get("optuna", {}).get("max_epochs",  30)
        batch_size  = cfg.get("optuna", {}).get("batch_size",  64)
        storage     = cfg.get("optuna", {}).get("storage",     "sqlite:///hpo.db")
        pruner_name = cfg.get("optuna", {}).get("pruner",      "median")

        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, drop_last=True,
        )
        val_loader = DataLoader(
            TensorDataset(X_validation, y_validation),
            batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, drop_last=False,
        )

        def objective(trial: optuna.Trial) -> float:
            cfg_trial = copy.deepcopy(predictor_config)
            cfg_trial["hidden_size"]   = trial.suggest_categorical("hidden_size",  [64, 128, 256, 512])
            cfg_trial["num_heads"]     = trial.suggest_categorical("num_heads",    [2, 4, 8])
            cfg_trial["num_layers"]    = trial.suggest_int("num_layers",           1, 6)
            cfg_trial["kernel_size"]   = trial.suggest_categorical("kernel_size",  [3, 5, 7])
            cfg_trial["dilation"]      = trial.suggest_categorical("dilation",     [1, 2, 4])
            cfg_trial["dropout"]       = trial.suggest_float("dropout",            0.0, 0.5, step=0.05)
            cfg_trial["learning_rate"] = trial.suggest_float("learning_rate",      1e-4, 1e-2, log=True)
            cfg_trial["weight_decay"]  = trial.suggest_float("weight_decay",       1e-5, 1e-2, log=True)
            cfg_trial["dilate_alpha"]  = trial.suggest_float("dilate_alpha",       0.1, 0.9,  step=0.1)
            cfg_trial["dilate_gamma"]  = trial.suggest_float("dilate_gamma",       0.001, 0.1, log=True)

            if cfg_trial["hidden_size"] % cfg_trial["num_heads"] != 0:
                raise optuna.exceptions.TrialPruned(
                    f"hidden_size {cfg_trial['hidden_size']} non divisibile per num_heads {cfg_trial['num_heads']}"
                )

            model      = Predictor(cfg_trial)
            pruning_cb = PyTorchLightningPruningCallback(trial, monitor="val_loss")
            trainer    = L.Trainer(
                max_epochs=max_epochs,
                callbacks=[
                    pruning_cb,
                    EarlyStopping(monitor="val_loss", patience=3, mode="min", min_delta=0.001),
                ],
                accelerator="cuda" if torch.cuda.is_available() else "cpu",
                gradient_clip_val=1.0,
                log_every_n_steps=10,
                enable_progress_bar=False,
                enable_model_summary=False,
                logger=False,
            )
            try:
                trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
            except optuna.exceptions.TrialPruned:
                raise

            val_result = trainer.validate(model, dataloaders=val_loader, verbose=False)
            return val_result[0]["val_loss"]

        if pruner_name == "median":
            pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
        elif pruner_name == "hyperband":
            pruner = optuna.pruners.HyperbandPruner(min_resource=3, max_resource=max_epochs, reduction_factor=3)
        else:
            raise ValueError(f"Pruner sconosciuto: '{pruner_name}'.")

        study = optuna.create_study(
            study_name="predictor_hpo",
            storage=storage,
            load_if_exists=True,
            direction="minimize",
            pruner=pruner,
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(objective, n_trials=n_trials, catch=(Exception,))

        print("\n" + "═" * 50)
        print(f"  Trial migliore : #{study.best_trial.number}")
        print(f"  Val loss       : {study.best_value:.6f}")
        print("  Parametri:")
        for k, v in study.best_params.items():
            print(f"    {k:25s} = {v}")
        print("═" * 50)

        with open("best_hpo_params.yaml", "w") as f:
            yaml.dump({"prediction": study.best_params}, f, default_flow_style=False)
        print("\n[OK] Parametri salvati in: best_hpo_params.yaml")

        pruned   = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        print(f"Trial completati : {len(complete)}")
        print(f"Trial potati     : {len(pruned)}")

    # ══════════════════════════════════════════════════════════════════════════
    # TRADE
    # ══════════════════════════════════════════════════════════════════════════

    elif cfg.step == "trade":
        model_path = os.path.join("checkpoints", f"checkpoint_{cfg.frequency}.ckpt")
        if not os.path.exists(model_path):
            print(f"Error: checkpoint not found at {model_path}")
            return

        os.makedirs("outputs", exist_ok=True)

        train_env = MultiTickerTradingEnv(model_path, safe_config, df_train, scaler)
        test_env  = MultiTickerTradingEnv(model_path, safe_config, df_test,  scaler)

        # Percorso dove salvare i pesi del trader
        trader_model_path = os.path.join("checkpoints", f"ppo_trader_{cfg.frequency}")

        trader = PPOTrader(train_env, safe_config["trader"], model_path=trader_model_path)
        print("Starting training...")
        trader.train()
        
        # ✅ Salva i pesi del trader dopo l'allenamento
        trader.save(trader_model_path)
        
        print("Training completed. Starting test...")
        trader.test(test_env, render=True)
        print("Test completed.")

        history = test_env.get_history()

        # ── Stampa operazioni ─────────────────────────────────────────────────
        trade_rows = history[history["actions"].apply(
            lambda acts: any(
                a.get("type") in ("buy", "sell")
                for a in acts.values()
                if isinstance(a, dict)
            )
        )]
        print(trade_rows[["step", "cash", "portfolio_value", "reward", "actions"]])

        final_bought = pd.Series(history["total_shares_bought"].iloc[-1]) if "total_shares_bought" in history.columns else pd.Series()
        final_sold   = pd.Series(history["total_shares_sold"].iloc[-1])   if "total_shares_sold"   in history.columns else pd.Series()

        print(f"\nOperazioni totali : {len(trade_rows)}")
        print(f"Shares bought     : {final_bought[final_bought > 0].to_dict()}")
        print(f"Shares sold       : {final_sold[final_sold > 0].to_dict()}")
        print(f"\nInitial Equity    : {test_env.initial_cash:.2f}")
        print(f"Final Equity      : {history['portfolio_value'].iloc[-1]:.2f}")
        print(f"Return            : {(history['portfolio_value'].iloc[-1] / test_env.initial_cash - 1) * 100:.2f}%")
        print(f"Maximum Drawdown  : {(history['portfolio_value'] / history['portfolio_value'].cummax() - 1).min() * 100:.2f}%")

        test_env.summary()

        # ── Grafico episodio (portfolio + reward + termodinamica) ─────────────
        test_env.plot_episode(save_path="outputs/episode_dashboard.png")

        # ── Dashboard termodinamica su df_test grezzo ─────────────────────────
        thermo_cols = [c for c in df_test.columns if c.startswith("thermo_")]
        if thermo_cols:
            # df_test è scalato → inverse transform per recuperare i prezzi
            # ma per il grafico termodinamico usiamo direttamente i valori scalati
            # (il plotting si adatta al range [0.05, 0.95])
            plot_thermodynamic(
                df_main=df_test,
                df_thermo=df_test[thermo_cols],
                tickers=cfg.data.tickers,
                title=f"Dashboard Termodinamica — Test Set ({cfg.frequency})",
                save_path="outputs/thermo_dashboard.png",
            )
        else:
            print("[main] Nessuna colonna thermo_* nel test set (modalità intraday?).")

    # ══════════════════════════════════════════════════════════════════════════
    # ALPACA REPLAY
    # ══════════════════════════════════════════════════════════════════════════

    elif cfg.step == "alpaca_replay":
        raise NotImplementedError("Alpaca replay non ancora implementato.")

    # ══════════════════════════════════════════════════════════════════════════
    # TRADE_ONLY (carica modello già allenato, solo test)
    # ══════════════════════════════════════════════════════════════════════════

    elif cfg.step == "trade_only":
        model_path = os.path.join("checkpoints", f"checkpoint_{cfg.frequency}.ckpt")
        if not os.path.exists(model_path):
            print(f"Error: checkpoint not found at {model_path}")
            return

        os.makedirs("outputs", exist_ok=True)

        test_env = MultiTickerTradingEnv(model_path, safe_config, df_test, scaler)
        trader_model_path = os.path.join("checkpoints", f"ppo_trader_{cfg.frequency}")

        # Carica il modello PPO già allenato (senza riallenare)
        print(f"Loading pre-trained PPO model from: {trader_model_path}")
        trader = PPOTrader(test_env, safe_config["trader"], model_path=trader_model_path)
        
        print("Starting test with pre-trained model...")
        trader.test(test_env, render=True)
        print("Test completed.")

        history = test_env.get_history()

        # ── Stampa operazioni ─────────────────────────────────────────────────
        trade_rows = history[history["actions"].apply(
            lambda acts: any(
                a.get("type") in ("buy", "sell")
                for a in acts.values()
                if isinstance(a, dict)
            )
        )]
        print(trade_rows[["step", "cash", "portfolio_value", "reward", "actions"]])

        final_bought = pd.Series(history["total_shares_bought"].iloc[-1]) if "total_shares_bought" in history.columns else pd.Series()
        final_sold   = pd.Series(history["total_shares_sold"].iloc[-1])   if "total_shares_sold"   in history.columns else pd.Series()

        print(f"\nOperazioni totali : {len(trade_rows)}")
        print(f"Shares bought     : {final_bought[final_bought > 0].to_dict()}")
        print(f"Shares sold       : {final_sold[final_sold > 0].to_dict()}")
        print(f"\nInitial Equity    : {test_env.initial_cash:.2f}")
        print(f"Final Equity      : {history['portfolio_value'].iloc[-1]:.2f}")
        print(f"Return            : {(history['portfolio_value'].iloc[-1] / test_env.initial_cash - 1) * 100:.2f}%")
        print(f"Maximum Drawdown  : {(history['portfolio_value'] / history['portfolio_value'].cummax() - 1).min() * 100:.2f}%")

        test_env.summary()

        # ── Grafico episodio (portfolio + reward + termodinamica) ─────────────
        test_env.plot_episode(save_path="grafici/episode_dashboard_only.png")

        # ── Dashboard termodinamica su df_test grezzo ─────────────────────────
        thermo_cols = [c for c in df_test.columns if c.startswith("thermo_")]
        if thermo_cols:
            plot_thermodynamic(
                df_main=df_test,
                df_thermo=df_test[thermo_cols],
                tickers=cfg.data.tickers,
                title=f"Dashboard Termodinamica — Test Set ({cfg.frequency})",
                save_path="outputs/thermo_dashboard_only.png",
            )
        else:
            print("[main] Nessuna colonna thermo_* nel test set (modalità intraday?).")

    # ══════════════════════════════════════════════════════════════════════════
    # TRAIN
    # ══════════════════════════════════════════════════════════════════════════

    elif cfg.step == "train":
        train_dataset      = TensorDataset(X_train, y_train)
        validation_dataset = TensorDataset(X_validation, y_validation)

        train_loader      = DataLoader(train_dataset,      batch_size=cfg.training.batch_size,   shuffle=False, num_workers=NUM_WORKERS, drop_last=True)
        validation_loader = DataLoader(validation_dataset, batch_size=cfg.prediction.batch_size, shuffle=False, num_workers=NUM_WORKERS, drop_last=False)

        pred               = Predictor(predictor_config)
        pred               = torch.compile(pred)
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.001, patience=3, verbose=False, mode="min")
        checkpoint_name     = f"checkpoint_{cfg.frequency}"
        checkpoint_callback = ModelCheckpoint(
            dirpath="checkpoints",
            filename=checkpoint_name,
            save_top_k=1,
            verbose=True,
            monitor="val_loss",
            mode="min",
            save_last=True,
            enable_version_counter=False,
        )
        trainer = L.Trainer(
            max_epochs=cfg.prediction.epochs,
            callbacks=[early_stop_callback, checkpoint_callback],
            accelerator="cuda" if torch.cuda.is_available() else "cpu",
            gradient_clip_val=1.0,
            log_every_n_steps=10,
        )
        trainer.fit(model=pred, train_dataloaders=train_loader, val_dataloaders=validation_loader)

        if early_stop_callback.stopping_reason == EarlyStoppingReason.PATIENCE_EXHAUSTED:
            print("Training stopped due to patience exhaustion")
        elif early_stop_callback.stopping_reason == EarlyStoppingReason.STOPPING_THRESHOLD:
            print("Training stopped due to reaching stopping threshold")
        elif early_stop_callback.stopping_reason == EarlyStoppingReason.NOT_STOPPED:
            print("Training completed normally without early stopping")

        validation_loss = trainer.validate(model=pred, dataloaders=validation_loader)
        print(f"Validation loss after training: {validation_loss}")
        return validation_loss

    # ══════════════════════════════════════════════════════════════════════════
    # TEST
    # ══════════════════════════════════════════════════════════════════════════

    elif cfg.step == "test":
        model_path = os.path.join("checkpoints", f"checkpoint_{cfg.frequency}.ckpt")
        if not os.path.exists(model_path):
            print(f"Error: Model checkpoint not found in {model_path}.")
            return
        print("Loading model from checkpoint...")

        pred         = Predictor.load_from_checkpoint(model_path, config=predictor_config)
        pred         = torch.compile(pred)
        test_dataset = TensorDataset(X_test, y_test)
        test_loader  = DataLoader(test_dataset, batch_size=cfg.prediction.batch_size,
                                  shuffle=False, num_workers=NUM_WORKERS)
        trainer      = L.Trainer(accelerator="cuda" if torch.cuda.is_available() else "cpu")
        recon        = trainer.predict(pred, dataloaders=test_loader)

        all_preds = torch.cat([r["preds"]      for r in recon], dim=0)
        all_y     = torch.cat([r["real"]        for r in recon], dim=0)
        all_std   = torch.cat([r["uncertainty"] for r in recon], dim=0)

        n_main     = predictor_config["n_features"] - predictor_config["n_context_features"]
        columns_name = df_test.columns[:n_main].tolist()
        horizon    = all_preds.shape[1]
        columns    = pd.MultiIndex.from_product(
            [range(1, horizon + 1), columns_name], names=["step", "ticker"]
        )

        price_predicted = pd.DataFrame(all_preds.reshape(len(all_preds), -1), columns=columns, index=test_dates)
        price_real      = pd.DataFrame(all_y.reshape(len(all_y),     -1),     columns=columns, index=test_dates)
        uncertainty     = pd.DataFrame(all_std.reshape(len(all_std),  -1),    columns=columns, index=test_dates)

        padded_predicted = np.pad(
            price_predicted.values.reshape(-1, n_main),
            ((0, 0), (0, predictor_config["n_context_features"])),
            mode="constant",
        )
        predicted_rescaled = scaler.inverse_transform(padded_predicted)[:, :n_main]
        real_rescaled      = scaler.inverse_transform(
            np.pad(
                price_real.values.reshape(-1, n_main),
                ((0, 0), (0, predictor_config["n_context_features"])),
                mode="constant",
            )
        )[:, :n_main]

        real_rescaled      = pd.DataFrame(real_rescaled.reshape(len(all_y),     -1), columns=columns, index=test_dates)
        predicted_rescaled = pd.DataFrame(predicted_rescaled.reshape(len(all_preds), -1), columns=columns, index=test_dates)

        plot_reconstruction(real_rescaled, predicted_rescaled, predictor_config, uncertainty)

    else:
        print(f"Error: unknown step '{cfg.step}'.")


if __name__ == "__main__":
    my_app()