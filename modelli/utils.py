import os
import datetime
import numpy as np
import pandas as pd
import torch
import yfinance as yf
from pandas.errors import EmptyDataError, ParserError
from sklearn.preprocessing import MinMaxScaler
from fredapi import Fred


_INTRADAY_LIMITS = {
    "1m":  7,
    "2m":  60,
    "5m":  60,
    "15m": 60,
    "30m": 60,
    "1h":  730,
    "90m": 60,
}


def _load_cache(cache_path: str, min_rows: int = 1) -> pd.DataFrame | None:
    """
    Carica il CSV di cache e lo valida.
    Ritorna None (e cancella il file) se la cache è vuota o corrotta.
    """
    if not os.path.exists(cache_path):
        return None
    try:
        data = pd.read_csv(cache_path, index_col=0, parse_dates=True)
    except Exception as e:
        print(f"[load_data] Cache corrotta ({e}), la scarico di nuovo.")
        os.remove(cache_path)
        return None

    if len(data) < min_rows:
        print(
            f"[load_data] Cache {cache_path} vuota o insufficiente "
            f"({len(data)} righe), la scarico di nuovo."
        )
        os.remove(cache_path)
        return None

    print(f"[load_data] Carico da cache: {cache_path} ({len(data)} barre)")
    return data


def load_data(
    tickers,
    start_date,
    end_date,
    fred_api_key,
    inflation_series,
    interval:         str = "1d",
    cache_path:       str = "./data.csv",
    max_history_days: int = None,
):
    now        = datetime.datetime.now()
    is_intraday = interval in _INTRADAY_LIMITS
    limit_days  = max_history_days or _INTRADAY_LIMITS.get(interval)

    if is_intraday:
        effective_end   = now
        earliest        = now - datetime.timedelta(days=limit_days)
        effective_start = max(start_date, earliest)
        if effective_start != start_date:
            print(
                f"[load_data] Intervallo '{interval}' limitato a {limit_days} giorni. "
                f"start_date aggiustato da {start_date.date()} a {effective_start.date()}"
            )
    else:
        effective_start = start_date
        effective_end   = end_date

    # ── tenta di caricare la cache (con validazione) ──────────────────────────
    data = _load_cache(cache_path)

    if data is None:
        print(f"[load_data] Scarico dati da yfinance (interval={interval})...")
        df_yf = yf.download(
            tickers,
            start=effective_start,
            end=effective_end,
            interval=interval,
            progress=False,
        )

        if df_yf.empty:
            raise RuntimeError(
                f"yfinance non ha restituito dati per interval='{interval}' "
                f"nel range {effective_start.date()} → {effective_end.strftime('%Y-%m-%d %H:%M')}."
            )

        # Gestisce sia MultiIndex che Index semplice
        close = df_yf["Close"]
        available = [t for t in tickers if t in close.columns]
        missing   = [t for t in tickers if t not in close.columns]
        if missing:
            print(f"[load_data] Ticker senza dati (rimossi): {missing}")
        data = close[available]

        # Serie FRED: solo per granularità giornaliera
        if not is_intraday and fred_api_key and inflation_series:
            fred_obj = Fred(api_key=fred_api_key)
            for series in inflation_series:
                try:
                    df_infl = fred_obj.get_series(
                        series,
                        observation_start=effective_start,
                        observation_end=effective_end,
                    )
                    data[series] = df_infl
                except (EmptyDataError, ParserError) as e:
                    print(f"[load_data] Errore serie FRED {series}: {e}")
        elif is_intraday and inflation_series:
            print(
                f"[load_data] Serie FRED saltate "
                f"(non disponibili per interval='{interval}')"
            )

        data = data.ffill().bfill()

        if len(data) == 0:
            raise RuntimeError(
                f"DataFrame vuoto dopo il download per interval='{interval}'."
            )

        data.to_csv(cache_path)
        print(f"[load_data] {len(data)} barre scaricate → cache salvata in {cache_path}")

    scaler = MinMaxScaler()
    data_scaled = pd.DataFrame(
        scaler.fit_transform(data),
        columns=data.columns,
        index=data.index,
    )
    return data_scaled.astype(np.float32), scaler


def make_windows(data, window_size, stride):
    X, Y = [], []
    data_array = data.values
    for i in range(0, len(data_array) - window_size, stride):
        X.append(data_array[i: i + window_size])
        Y.append(data_array[i + window_size])

    if not X:
        raise RuntimeError(
            f"Nessuna finestra creata: {len(data_array)} barre disponibili "
            f"ma window_size={window_size}. "
            f"Riduci window_size (es. prediction.window_size=10)."
        )

    return torch.tensor(np.array(X)), torch.tensor(np.array(Y))