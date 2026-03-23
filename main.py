from omegaconf import DictConfig, OmegaConf
import datetime
import pandas as pd
import torch
import os
import hydra

from modelli.pred import Pred
from modelli.trade import run_trade
from modelli.utils import load_data, make_windows
from modelli.evaluate_pred import evaluate_predictions
from dotenv import load_dotenv

load_dotenv()


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


def build_predictor(cfg, num_features: int) -> Pred:
    return Pred(
        num_features=num_features,
        window_size=cfg.prediction.window_size,
        dimension=list(cfg.model.dimensions),
        dilations=list(cfg.model.dilations),
        kernel_size=cfg.model.kernel_size,
        activation=cfg.model.activation,
    )


def split_dataframe(
    df: pd.DataFrame,
    cfg,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide df in train/test.

    Priorità:
      1. frequency.split_ratio  (float 0-1) → split percentuale
         es. 0.8 → primi 80% come train, ultimi 20% come test
      2. data.split_date         → split per data (comportamento originale)

    Il motivo per cui split_ratio ha priorità è che per dati intraday
    (es. 1m, solo 7 giorni disponibili) una data fissa come "2024-01-01"
    non ha senso: tutte le barre sarebbero nello stesso lato dello split.
    """
    split_ratio = getattr(cfg.frequency, "split_ratio", None)

    if split_ratio is not None:
        n         = len(df)
        n_train   = int(n * split_ratio)
        train_df  = df.iloc[:n_train]
        test_df   = df.iloc[n_train:]
        print(
            f"Split percentuale ({split_ratio:.0%}): "
            f"{len(train_df)} barre train | {len(test_df)} barre test "
            f"(da {train_df.index[-1]} a {test_df.index[0]})"
        )
    else:
        split_date = cfg.data.split_date
        train_df   = df[df.index <  split_date]
        test_df    = df[df.index >= split_date]
        print(
            f"Split per data ({split_date}): "
            f"{len(train_df)} barre train | {len(test_df)} barre test"
        )

    # Controllo minimo: il test set deve avere almeno window_size + 1 barre
    min_bars = cfg.prediction.window_size + 1
    if len(test_df) < min_bars:
        raise RuntimeError(
            f"Test set troppo piccolo: {len(test_df)} barre, "
            f"servono almeno {min_bars} (window_size={cfg.prediction.window_size}). "
            f"Aumenta split_ratio o riduci window_size."
        )
    if len(train_df) < min_bars:
        raise RuntimeError(
            f"Train set troppo piccolo: {len(train_df)} barre, "
            f"servono almeno {min_bars}."
        )

    return train_df, test_df


@hydra.main(version_base=None, config_path="config", config_name="config")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    tickers          = list(cfg.data.tickers)
    inflation_series = list(cfg.data.inflation_series)
    start = datetime.datetime.fromisoformat(cfg.data.start_date)
    end   = datetime.datetime.fromisoformat(cfg.data.end_date)

    df, scaler = load_data(
        tickers,
        start,
        end,
        cfg.data.fred_api_key,
        inflation_series,
        interval         = cfg.frequency.interval,
        cache_path       = cfg.frequency.cache_path,
        max_history_days = cfg.frequency.max_history_days,
    )

    num_features = df.shape[1]
    print(f"Feature totali: {num_features} → {list(df.columns)}")

    train_df, test_df = split_dataframe(df, cfg)

    X_train, Y_train = make_windows(train_df, cfg.prediction.window_size, cfg.prediction.stride)
    X_test,  Y_test  = make_windows(test_df,  cfg.prediction.window_size, cfg.prediction.stride)

    model_name = f"pred_{cfg.frequency.interval}_{cfg.prediction.window_size}"
    # ── TRAIN ─────────────────────────────────────────────────────────────────
    if cfg.step == "train":
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_train, Y_train),
            batch_size=cfg.training.batch_size,
            shuffle=True,
        )
        predictor     = build_predictor(cfg, num_features)
        moment_target = compute_moment_target(train_df, tickers, cfg)

        mre_cfg  = getattr(cfg.training, "mre", None)
        mre_mode = getattr(mre_cfg, "update_mode", "mse") if mre_cfg else "mse"
        print(f"\nInizio addestramento (modalità: {mre_mode})...")

        predictor.fit(train_loader, training_cfg=cfg.training, moment_target=moment_target)

        checkpoint_dir = cfg.paths.checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save({
            "model_state_dict": predictor.state_dict(),
            "scaler":           scaler,
            "num_features":     num_features,
            "config":           OmegaConf.to_container(cfg, resolve=True),
        }, os.path.join(checkpoint_dir, f"{model_name}.pth"))

        print(f"\nModello salvato in: {checkpoint_dir}/{model_name}.pth")

    # ── TEST ──────────────────────────────────────────────────────────────────
    elif cfg.step == "test":
        checkpoint_path = os.path.join(cfg.paths.checkpoint_dir, f"pred_{cfg.frequency.interval}_{cfg.prediction.window_size}.pth")
        if not os.path.exists(checkpoint_path):
            print("Errore: esegui prima step=train")
            return

        checkpoint     = torch.load(checkpoint_path, weights_only=False)
        saved_features = checkpoint.get("num_features", num_features)
        predictor      = build_predictor(cfg, saved_features)
        predictor.load_state_dict(checkpoint["model_state_dict"])
        predictor.eval()

        with torch.no_grad():
            predictions_scaled = predictor(X_test).numpy()
            predictions_train  = predictor(X_train).numpy()

        predictions       = scaler.inverse_transform(predictions_scaled)
        train_predictions = scaler.inverse_transform(predictions_train)
        Y_test_denorm     = scaler.inverse_transform(Y_test)
        Y_train_denorm    = scaler.inverse_transform(Y_train)

        all_columns = list(df.columns)
        test_dates  = test_df.index[cfg.prediction.window_size:]
        train_dates = train_df.index[cfg.prediction.window_size:]

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

    # ── TRADE ─────────────────────────────────────────────────────────────────
    elif cfg.step == "trade":
        run_trade(
            cfg=cfg,
            df=df,
            tickers=tickers,
            X_train=X_train,
            Y_train=Y_train,
            X_test=X_test,
            Y_test=Y_test,
            train_df=train_df,
            test_df=test_df,
        )

    else:
        print(f"Errore: step sconosciuto '{cfg.step}'. Scegli tra 'train', 'test', 'trade'.")


if __name__ == "__main__":
    my_app()