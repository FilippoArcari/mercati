import matplotlib

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def plot_reconstruction(y: pd.DataFrame,
                        preds: pd.DataFrame,
                        config: dict,
                        std: pd.DataFrame):
    os.makedirs("grafici", exist_ok=True)

    for df in (y, preds, std):
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

    freq = config.get("frequency", "1d")
    idx  = pd.IndexSlice
    

    for step, ticker in preds.columns.tolist():
        y_series    = y.loc[:, idx[step, ticker]].copy()
        pred_series = preds.loc[:, idx[step, ticker]].copy()
        std_series  = std.loc[:, idx[step, ticker]].copy()

        # Shift temporale coerente con la frequenza
        if freq == "2m":
            offset = pd.Timedelta(minutes=2 * step)
        else:
            offset = pd.offsets.BusinessDay(step)

        target_dates = pred_series.index + offset
        y_series.index    = target_dates
        pred_series.index = target_dates
        std_series.index  = target_dates

        fig, ax = plt.subplots(figsize=(15, 6))
        ax.plot(y_series,    label="Reale",    color="blue")
        ax.plot(pred_series, label="Predetto", color="orange")
        ax.fill_between(
            pred_series.index,
            pred_series - std_series,
            pred_series + std_series,
            color="orange", alpha=0.2, label="Uncertainty (±1 std)"
        )
        ax.set_title(f"Ricostruzione — {ticker} | step {step}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid()
        plt.xticks(rotation=45)
        plt.tight_layout()

        safe_name = f"step{step}_{ticker}"
        fig.savefig(f"grafici/ricostruzione_{safe_name}.png")
        plt.close(fig)