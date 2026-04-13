from omegaconf import DictConfig, OmegaConf
import datetime
import pandas as pd
import torch
import os
import hydra
import sys
from typing import Optional

from modelli.pred import Pred
from modelli.trade import run_trade
from modelli.utils import load_data, make_windows, make_stats, get_device
from modelli.device_setup import detect_device, get_map_location, safe_save, unwrap_model
from modelli.evaluate_pred import evaluate_predictions
from dotenv import load_dotenv
import yfinance as yf

load_dotenv()


# ─── Frequency-based defaults setup ───────────────────────────────────────────

def _apply_frequency_defaults():
    """
    Se nella CLI viene specificato frequency=..., aggiunge automaticamente
    training=..., prediction=..., e buyer=... basato sulla frequenza.
    """
    frequency_map = {
        "daily":  {"training": "default", "prediction": "default", "buyer": "default"},
        "minute": {"training": "minute",  "prediction": "minute",  "buyer": "minute"},
    }
    freq_override = None
    for arg in sys.argv[1:]:
        if arg.startswith("frequency="):
            freq_override = arg.split("=", 1)[1]
            break
    if freq_override and freq_override in frequency_map:
        defaults = frequency_map[freq_override]
        for key, value in defaults.items():
            override = f"{key}={value}"
            if override not in sys.argv:
                sys.argv.append(override)


_apply_frequency_defaults()


# ─── Helpers ───────────────────────────────────────────────────────────────────

def checkpoint_name(cfg, name: str) -> str:
    """
    Costruisce il path di un checkpoint includendo la frequenza e la window size.

    Esempi:
      pred_1d_w30.pth    / pred_1m_w10.pth
      ddpg_1d.pth        / ddpg_best_1m.pth
      normalizer_1d.npz  / normalizer_1m.npz
    """
    freq = cfg.frequency.interval
    ws   = cfg.prediction.window_size
    tag  = (
        f"_{freq}"
        if any(k in name for k in ("ddpg", "normalizer"))
        else f"_{freq}_w{ws}"
    )
    stem, ext = os.path.splitext(name)
    return os.path.join(cfg.paths.checkpoint_dir, f"{stem}{tag}{ext}")


def compute_moment_target(train_df, tickers, cfg):
    mre_cfg = getattr(cfg.training, "mre", None)
    if mre_cfg is None or not getattr(mre_cfg, "enabled", False):
        return None
    manual_F = getattr(mre_cfg, "moment_target", None)
    if manual_F is not None:
        F = torch.tensor(list(manual_F), dtype=torch.float32)
        print(f"[MrE] moment_target da config: F={F.tolist()}")
        return F
    F = torch.tensor(train_df.mean().values, dtype=torch.float32)
    print(f"[MrE] moment_target calcolato dai dati: F={F.tolist()}")
    return F


def build_predictor(
    cfg,
    num_features: int,
    prior_mean: Optional[torch.Tensor] = None,
    prior_std: Optional[torch.Tensor] = None,
    moment_target: Optional[torch.Tensor] = None,
) -> Pred:
    return Pred(
        num_features=num_features,
        window_size=cfg.prediction.window_size,
        dimension=list(cfg.model.dimensions),
        dilations=list(cfg.model.dilations),
        kernel_size=cfg.model.kernel_size,
        activation=cfg.model.activation,
        prediction_steps=cfg.model.prediction_steps,
        training_cfg=OmegaConf.to_container(cfg.training, resolve=True),
        prior_mean=prior_mean,
        prior_std=prior_std,
        moment_target=moment_target,
    )


def split_dataframe(df: pd.DataFrame, cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split per percentuale (intraday) o per data (giornaliero).
    """
    split_ratio = getattr(cfg.frequency, "split_ratio", None)

    if split_ratio is not None:
        n        = len(df)
        n_train  = int(n * split_ratio)
        train_df = df.iloc[:n_train]
        test_df  = df.iloc[n_train:]
        print(
            f"Split percentuale ({split_ratio:.0%}): "
            f"{len(train_df)} barre train | {len(test_df)} barre test "
            f"(split a {train_df.index[-1]})"
        )
    else:
        split_date = cfg.data.split_date
        train_df   = df[df.index <  split_date]
        test_df    = df[df.index >= split_date]
        print(
            f"Split per data ({split_date}): "
            f"{len(train_df)} barre train | {len(test_df)} barre test"
        )

    min_bars = cfg.prediction.window_size + 1
    if len(test_df) < min_bars:
        raise RuntimeError(
            f"Test set troppo piccolo: {len(test_df)} barre, "
            f"servono almeno {min_bars}."
        )
    if len(train_df) < min_bars:
        raise RuntimeError(
            f"Train set troppo piccolo: {len(train_df)} barre, "
            f"servono almeno {min_bars}."
        )
    return train_df, test_df


# ─── Entry point ───────────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="config", config_name="config")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # ── ALPACA — early exit prima di load_data (i dati arrivano live) ─────────
    if cfg.step == "alpaca":
        from modelli.alpaca_live import run_alpaca
        run_alpaca(cfg)
        return

    if cfg.step == "alpaca_replay":
        from modelli.alpaca_live import run_alpaca_replay
        run_alpaca_replay(cfg)
        return

    tickers          = list(cfg.data.tickers)
    inflation_series = list(cfg.data.inflation_series)
    start            = datetime.datetime.fromisoformat(cfg.data.start_date)
    end              = datetime.datetime.fromisoformat(cfg.data.end_date)

    raw_df_with_volumes = None
    cache_path          = cfg.frequency.cache_path
    if os.path.exists(cache_path):
        raw_df_with_volumes = pd.read_csv(cache_path, index_col=0, parse_dates=True)

    df, scaler = load_data(
        tickers, start, end,
        cfg.data.fred_api_key,
        inflation_series,
        interval         = cfg.frequency.interval,
        cache_path       = cfg.frequency.cache_path,
        max_history_days = cfg.frequency.max_history_days,
        split_ratio      = getattr(cfg.frequency, "split_ratio", None),
    )

    num_features = df.shape[1]
    print(f"Feature totali: {num_features} → {list(df.columns)}")

    train_df, test_df = split_dataframe(df, cfg)

    X_train, Y_train = make_windows(train_df, cfg.prediction.window_size, cfg.prediction.stride, prediction_steps=cfg.model.prediction_steps)
    X_test,  Y_test  = make_windows(test_df,  cfg.prediction.window_size, cfg.prediction.stride, prediction_steps=cfg.model.prediction_steps)

    freq = cfg.frequency.interval
    print(
        f"[Checkpoint] Frequenza attiva: {freq} — "
        f"i checkpoint useranno il suffisso '_{freq}'"
    )

    # ── TRAIN ──────────────────────────────────────────────────────────────────
    if cfg.step == "train":
        _dev_cfg = detect_device()
        device   = _dev_cfg.device

        # ── DataLoader ottimizzato per il backend corrente ──────────────────
        # pin_memory:          solo CUDA (trasferimento asincrono CPU→GPU)
        # num_workers:         0 su XLA (evita deadlock con lazy tensors)
        # persistent_workers:  solo CUDA (XLA e CPU usano 0 worker)
        # prefetch_factor:     solo se num_workers > 0
        _nw  = _dev_cfg.optimal_num_workers
        _pin = _dev_cfg.supports_pin_memory
        _pw  = (_dev_cfg.backend == "cuda" and _nw > 0)
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_train, Y_train),
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=_nw,
            pin_memory=_pin,
            persistent_workers=_pw,
            prefetch_factor=2 if _nw > 0 else None,
        )
        import pytorch_lightning as pl

        print(f"[DataLoader] num_workers={_nw}, pin_memory={_pin}, backend={_dev_cfg.backend}")
        print(f"[Device] {_dev_cfg.hardware_label}")
        
        mre_cfg  = getattr(cfg.training, "mre", None)
        prior_mean, prior_std = None, None
        if mre_cfg and getattr(mre_cfg, "enabled", False):
            from modelli.pred import PriorEstimator
            prior = PriorEstimator()
            prior.fit(train_loader)
            prior_mean = prior.mean
            prior_std = prior.std
            
        moment_target = compute_moment_target(train_df, tickers, cfg)
        
        predictor = build_predictor(cfg, num_features, prior_mean, prior_std, moment_target)

        mre_mode = getattr(mre_cfg, "update_mode", "mse") if mre_cfg else "mse"
        print(f"\nInizio addestramento (modalità: {mre_mode}, freq: {freq})...")

        trainer = pl.Trainer(
            max_epochs=cfg.training.epochs,
            accelerator="auto",
            devices="auto",
            gradient_clip_val=getattr(cfg.training, "max_grad_norm", 1.0),
            enable_progress_bar=True,
            logger=False,
        )
        trainer.fit(predictor, train_dataloaders=train_loader)

        pred_path = checkpoint_name(cfg, "pred.pth")
        os.makedirs(cfg.paths.checkpoint_dir, exist_ok=True)
        safe_save(
            {
                "model_state_dict": predictor.state_dict(),
                "scaler":           scaler,
                "num_features":     num_features,
                "frequency":        freq,
                "window_size":      cfg.prediction.window_size,
                "config":           OmegaConf.to_container(cfg, resolve=True),
            },
            pred_path,
        )
        print(f"\nModello salvato in: {pred_path}")

    elif cfg.step == "stats":
        make_stats(cfg)

    # ── TEST ───────────────────────────────────────────────────────────────────
    elif cfg.step == "test":
        device    = get_device()
        pred_path = checkpoint_name(cfg, "pred.pth")
        if not os.path.exists(pred_path):
            print(f"Errore: nessun checkpoint trovato in {pred_path}")
            print(f"Esegui prima: uv run main.py step=train frequency={freq}")
            return

        checkpoint     = torch.load(pred_path, map_location=get_map_location(), weights_only=False)
        saved_features = checkpoint.get("num_features", num_features)
        # build_predictor expects the tensors if there were any, but here we just
        # let checkpoint's load_state_dict fill the buffers, so we can pass None.
        predictor      = build_predictor(cfg, saved_features, None, None, None).to(device)
        predictor.load_state_dict(checkpoint["model_state_dict"], strict=False)
        predictor.eval()

        X_test_device  = X_test.to(device)
        X_train_device = X_train.to(device)

        with torch.no_grad():
            predictions_scaled = predictor(X_test_device).cpu().numpy()
            predictions_train  = predictor(X_train_device).cpu().numpy()

        # [Fix] Se il modello è multi-step (3D), prendiamo solo il primo step per il test standard
        if predictions_scaled.ndim == 3:
            print(f"[Test] Modello multi-step rilevato ({predictions_scaled.shape}) -> uso step 0 per valutazione")
            predictions_scaled = predictions_scaled[:, 0, :]
            predictions_train  = predictions_train[:, 0, :]

        predictions       = scaler.inverse_transform(predictions_scaled)
        train_predictions = scaler.inverse_transform(predictions_train)
        Y_test_denorm     = scaler.inverse_transform(Y_test)
        Y_train_denorm    = scaler.inverse_transform(Y_train)

        all_columns = list(df.columns)
        test_dates  = test_df.index[cfg.prediction.window_size::cfg.prediction.stride]
        train_dates = train_df.index[cfg.prediction.window_size::cfg.prediction.stride]

        results_dir = cfg.paths.results_dir
        os.makedirs(results_dir, exist_ok=True)

        evaluate_predictions(
            predictions=pd.DataFrame(predictions,       index=test_dates,  columns=all_columns),
            targets    =pd.DataFrame(Y_test_denorm,     index=test_dates,  columns=all_columns),
            step="test", results_dir=results_dir,
        )
        evaluate_predictions(
            predictions=pd.DataFrame(train_predictions, index=train_dates, columns=all_columns),
            targets    =pd.DataFrame(Y_train_denorm,    index=train_dates, columns=all_columns),
            step="train", results_dir=results_dir,
        )
        print("\nTest completato.")

    # ── TRADE / WALK-FORWARD ───────────────────────────────────────────────────
    elif cfg.step in ("trade", "walk_forward"):
        # Le finestre sull'intero dataset servono per la walk-forward
        # (il trade normale usa solo X_train/X_test già calcolati sopra)
        X_all, Y_all = make_windows(df, cfg.prediction.window_size, cfg.prediction.stride)

        if cfg.step == "trade":
            run_trade(
                cfg=cfg,
                df=df,
                df_raw=raw_df_with_volumes,
                tickers=tickers,
                X_train=X_train,
                Y_train=Y_train,
                X_test=X_test,
                Y_test=Y_test,
                train_df=train_df,
                test_df=test_df,
            )
        else:
            # ── Walk-forward ──────────────────────────────────────────────
            from modelli.trade import run_walk_forward
            wf_cfg = getattr(cfg, "walk_forward", None)

            run_walk_forward(
                cfg           = cfg,
                df            = df,
                tickers       = tickers,
                train_df      = train_df,
                test_df       = test_df,
                X_all         = X_all,
                Y_all         = Y_all,
                df_raw        = raw_df_with_volumes,
                n_folds       = int(getattr(wf_cfg,   "n_folds",        5))       if wf_cfg else 5,
                mode          = str(getattr(wf_cfg,   "mode",     "sliding"))      if wf_cfg else "sliding",
                min_train_pct = float(getattr(wf_cfg, "min_train_pct",   0.55))   if wf_cfg else 0.55,
                test_pct      = float(getattr(wf_cfg, "test_pct",        0.12))   if wf_cfg else 0.12,
                warm_start    = bool(getattr(wf_cfg,  "warm_start",      True))   if wf_cfg else True,
            )

    else:
        print(
            f"Errore: step sconosciuto '{cfg.step}'. "
            "Scegli tra 'train', 'test', 'trade', 'walk_forward', "
            "'alpaca', 'alpaca_replay', 'stats'."
        )


if __name__ == "__main__":
    my_app()