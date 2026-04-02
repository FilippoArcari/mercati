import os
import datetime
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
import yfinance as yf
from pandas.errors import EmptyDataError, ParserError
from sklearn.preprocessing import MinMaxScaler
from fredapi import Fred
import seaborn as sns

from modelli.thermodynamics import (
    compute_thermodynamic_features,
    calculate_pressure_and_work,
    init_vdw_calibration,
)
from scipy.stats import entropy


_INTRADAY_LIMITS = {
    "1m":  7,
    "2m":  60,
    "5m":  60,
    "15m": 60,
    "30m": 60,
    "1h":  730,
    "90m": 60,
}

# Colonne tassi cercate in ordine per la divergenza energetica
_RATES_CANDIDATES = ["GS10", "T10YIE", "FEDFUNDS", "^TNX"]


# ─── Device Management ─────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """
    Seleziona il device per PyTorch (GPU o CPU) in base a:
      1. Variabile di ambiente PYTORCH_DEVICE (es. 'cpu', 'cuda')
      2. Se non settata, usa GPU se disponibile, altrimenti CPU
      3. Se GPU richiesta ma non disponibile, fallback a CPU

    Uso:
        device = get_device()
        model = MyModel().to(device)
        x = torch.randn(10, 5).to(device)

    Environment override:
        PYTORCH_DEVICE=cpu python main.py      # forza CPU
        PYTORCH_DEVICE=cuda python main.py     # forza GPU (fallback CPU se non disponibile)
    """
    device_env = os.environ.get("PYTORCH_DEVICE", "").lower()

    if device_env:
        if device_env == "cpu":
            device = torch.device("cpu")
        elif device_env in ("cuda", "gpu"):
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                print("[get_device] CUDA richiesto ma non disponibile, fallback a CPU")
                device = torch.device("cpu")
        else:
            print(f"[get_device] Device '{device_env}' non riconosciuto, uso CPU")
            device = torch.device("cpu")
    else:
        # Default: GPU se disponibile, altrimenti CPU
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    device_name = "GPU (CUDA)" if device.type == "cuda" else "CPU"
    print(f"[get_device] Usando device: {device_name}")
    return device


def _load_cache(cache_path: str, min_rows: int = 1) -> pd.DataFrame | None:
    if not os.path.exists(cache_path):
        return None
    try:
        data = pd.read_csv(cache_path, index_col=0, parse_dates=True)
    except Exception as e:
        print(f"[load_data] Cache corrotta ({e}), la scarico di nuovo.")
        os.remove(cache_path)
        return None
    if len(data) < min_rows:
        print(f"[load_data] Cache vuota ({len(data)} righe), la scarico di nuovo.")
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
    interval:           str  = "1d",
    cache_path:         str  = "./data.csv",
    max_history_days:   int  = None,
    add_thermodynamics: bool = True,
    thermo_window:      int  = 20,
    thermo_max_lag:     int  = 90,
):
    """
    Scarica prezzi + FRED, calcola feature termodinamiche e normalizza.

    Le feature termodinamiche vengono calcolate sui dati GREZZI (prima
    della normalizzazione MinMax) e aggiunte al DataFrame come colonne
    aggiuntive. Il DCNN le vede come input e impara a usarle.

    Dalla tua heatmap le correlazioni chiave sono:
      Market_Pressure    -0.64 con Close   (inversamente predittiva)
      Market_Temperature -0.67 con Close
      Market_Work_Cum    +0.67 con Close   (positivamente predittiva)

    Parametri aggiuntivi
    --------------------
    add_thermodynamics : aggiunge le feature termodinamiche (default True)
    thermo_window      : finestra mobile per entropia/autocorr (default 20)
    thermo_max_lag     : lag massimo cross-correlazione tassi (default 90)
    """
    now         = datetime.datetime.now()
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

    data = _load_cache(cache_path)

    if data is None:
        print(f"[load_data] Scarico dati da yfinance (interval={interval})...")

        # Scarica Close + Volume (Volume serve per P e W)
        df_yf = yf.download(
            tickers,
            start=effective_start,
            end=effective_end,
            interval=interval,
            progress=False,
        )
        

        if df_yf.empty:
            raise RuntimeError(
                f"yfinance non ha restituito dati per interval='{interval}'."
            )

        close  = df_yf["Close"]
        volume = df_yf.get("Volume", pd.DataFrame())

        available = [t for t in tickers if t in close.columns]
        missing   = [t for t in tickers if t not in close.columns]
        if missing:
            print(f"[load_data] Ticker senza dati (rimossi): {missing}")

        data = close[available].copy()

        # Salva volumi grezzi con suffisso _Volume per uso termodinamico
        if not volume.empty:
            for t in available:
                if t in volume.columns:
                    data[f"{t}_Volume"] = volume[t]

        # Serie FRED (solo giornaliero)
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
        
        data = data.ffill().bfill()
        if len(data) == 0:
            raise RuntimeError("DataFrame vuoto dopo il download.")
        
        data.to_csv(cache_path)
        print(f"[load_data] {len(data)} barre → cache in {cache_path}")

    # ── Feature termodinamiche (su dati grezzi, PRIMA della norm.) ───────────
    volume_cols = [c for c in data.columns if c.endswith("_Volume")]
    price_cols  = [c for c in data.columns
                   if not c.endswith("_Volume")
                   and c not in (inflation_series or [])]

    if add_thermodynamics:
        if not is_intraday:
            # Giornaliero: calcola tutto inclusa la divergenza monetaria
            rates_col = next(
                (c for c in _RATES_CANDIDATES if c in data.columns), None
            )
            
            # ── CALIBRAZIONE VdW ───────────────────────────────────────────────
            # Calibra parametri di Van der Waals dall'analisi tail dei dati
            # storici. Questo rende la pressione e il lavoro termodinamico
            # empiricamente accurati secondo Gabaix et al. 2003.
            vdw_calibration = None
            try:
                vdw_calibration = init_vdw_calibration(
                    df_raw=data,
                    ticker_cols=price_cols,
                    verbose=True,
                )
            except Exception as e:
                print(f"[load_data] AVVERTENZA: Calibrazione VdW fallita ({e}) "
                      f"— uso fallback ai parametri heuristici")
            
            thermo_df = compute_thermodynamic_features(
                df_raw=data,
                ticker_cols=price_cols,
                rates_col=rates_col,
                window=thermo_window,
                max_lag=thermo_max_lag,
                vdw_calibration=vdw_calibration,
            )
        else:
            # Intraday: solo P e W (finestra adattata ai dati disponibili)
            if volume_cols:
                port_close  = data[price_cols].mean(axis=1)
                port_volume = data[volume_cols].sum(axis=1)
            else:
                port_close  = data[price_cols].mean(axis=1)
                port_volume = (port_close.diff().abs() * 1e6).fillna(1e6)

            eff_window = min(thermo_window, max(5, len(data) // 20))
            thermo_df  = calculate_pressure_and_work(
                close=port_close, volume=port_volume, window=eff_window
            )

        data = pd.concat([data, thermo_df], axis=1)
        print(f"[load_data] Feature termodinamiche aggiunte: {list(thermo_df.columns)}")

    # Rimuovi i volumi grezzi: già codificati nelle feature termodinamiche
    if volume_cols:
        data = data.drop(columns=volume_cols)

    data = data.ffill().bfill()

    # ── Normalizzazione MinMax ─────────────────────────────────────────────────
    scaler      = MinMaxScaler()
    data_scaled = pd.DataFrame(
        scaler.fit_transform(data),
        columns=data.columns,
        index=data.index,
    )
    print(f"[load_data] Ci sono dei nan dopo il download? {data.isna().any().any()}")
    thermo_col_names = [c for c in data_scaled.columns
                        if c.startswith(("Market_", "Energy_", "Thermo_", "Volume_Delta"))]
    print(
        f"[load_data] Dataset finale: {data_scaled.shape[1]} colonne × {len(data_scaled)} barre\n"
        f"  Termodinamica ({len(thermo_col_names)}): {thermo_col_names}"
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

def calculate_market_thermodynamics(df, n_assets=1, kb=1.0):
    """
    Calcola Pressione e Lavoro Integrale per un singolo asset o portafoglio.
    Supporta input con colonne semplici 'Close' e 'Volume'.
    """
    if df.empty:
        return pd.Series(), pd.Series()
        
    returns = df['Close'].pct_change().dropna()
    window = 20
    
    if len(returns) < window:
        return pd.Series(index=df.index).fillna(0), pd.Series(index=df.index).fillna(0)
    
    def rolling_entropy(x):
        prob_dist, _ = np.histogram(x, bins=20, density=True)
        prob_dist = prob_dist[prob_dist > 0] 
        return entropy(prob_dist)

    s_series = returns.rolling(window).apply(rolling_entropy)
    v_series = np.log1p(df['Volume'].rolling(window).mean())
    
    a_series = returns.rolling(window).apply(lambda x: pd.Series(x).autocorr(lag=1) if len(x)>2 else 0).fillna(0)
    b_const = v_series.min() * 0.1 
    
    v_free = (v_series - b_const).clip(lower=0.1)
    t_series = np.exp(2 * (s_series - np.log(v_free)))
    p_series = (kb * t_series) / v_free - a_series * (1 / v_series)**2
    
    delta_v = v_series.diff()
    p_avg = (p_series + p_series.shift(1)) / 2
    work_cumulative = (p_avg * delta_v).fillna(0).cumsum()
    
    return p_series, work_cumulative

def make_stats(cfg):
    """
    Esegue l'analisi termodinamica completa, calcola divergenze e resilienza.
    Genera i grafici finali nel percorso specificato.
    """
    os.makedirs(cfg.paths.results_dir, exist_ok=True)
    
    tickers = list(cfg.data.tickers)
    needed = ['^GSPC', '^TNX'] # S&P500 e Tasso 10Y (Proxy Pressione Monetaria)
    for n in needed:
        if n not in tickers: tickers.append(n)
    df_raw = yf.download(tickers, start=cfg.data.start_date, end=cfg.data.end_date)
    portfolio_tickers = [t for t in list(cfg.data.tickers) if t not in needed]
    
    
    entropy_diff = calculate_entropy_difference(df_raw)
    
    # Prepara il DataFrame di portafoglio (media) per l'analisi macro
    df_port = pd.DataFrame({
        'Close': df_raw['Close'][portfolio_tickers].mean(axis=1),
        'Volume': df_raw['Volume'][portfolio_tickers].sum(axis=1),
        'SP500': df_raw['Close']['^GSPC'],
        'Rates': df_raw['Close']['^TNX']
    }).dropna()
    
    # Calcolo Termodinamico Macro
    p, w = calculate_market_thermodynamics(df_port, len(portfolio_tickers))
    df_port['Pressure'] = p
    df_port['Work'] = w
    
    # Trova il lag ottimale tra Pressione e Tassi
    max_lag = 90
    corrs = [df_port['Pressure'].corr(df_port['Rates'].shift(i)) for i in range(max_lag)]
    best_lag = np.argmax(np.abs(corrs))
    
    # Calcolo Divergenza Energetica
    df_port['Expected_Pressure'] = df_port['Rates'].shift(best_lag)
    scaler = MinMaxScaler()
    cols_to_norm = ['SP500', 'Work', 'Pressure', 'Expected_Pressure']
    df_norm = pd.DataFrame(scaler.fit_transform(df_port[cols_to_norm]), 
                           columns=cols_to_norm, index=df_port.index)
    
    df_port['Energy_Divergence'] = df_norm['Pressure'] - df_norm['Expected_Pressure']
    df_port['Efficiency'] = (df_norm['Work'] - df_norm['SP500']).rolling(10).mean()

    # Plot Macro: Stress e Efficienza
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [2, 1]})
    
    ax1.plot(df_norm.index, df_norm['SP500'], label='S&P 500 (Prezzo Norm.)', color='#2c3e50', linewidth=1.5, alpha=0.5)
    ax1.plot(df_norm.index, df_norm['Work'], label='Lavoro Cumulativo (Work)', color='#27ae60', linewidth=2.5)
    
    ax2 = ax1.twinx()
    ax2.fill_between(df_port.index, df_port['Energy_Divergence'], 0, 
                     where=(df_port['Energy_Divergence'] >= 0), color='#e74c3c', alpha=0.3, label='Sovra-Pressione (Stress)')
    ax2.fill_between(df_port.index, df_port['Energy_Divergence'], 0, 
                     where=(df_port['Energy_Divergence'] < 0), color='#3498db', alpha=0.3, label='Sotto-Pressione (Rally Space)')
    
    ax1.set_title(f'Analisi Termodinamica: Stress Monetario e Lavoro (Lag: {best_lag}gg)', fontsize=16)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax1.grid(True, alpha=0.1)

    ax3.plot(df_port.index, df_port['Efficiency'], color='purple', label='Oscillatore di Efficienza (Work - Price)')
    ax3.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax3.fill_between(df_port.index, df_port['Efficiency'], 0, color='purple', alpha=0.15)
    ax3.set_title('Efficienza del Sistema (Eccesso di Lavoro rispetto al Prezzo)', fontsize=12)
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.paths.results_dir, 'market_thermodynamics.png'))
    plt.close()
    
    corr_matrix = df_port[['SP500', 'Pressure', 'Expected_Pressure', 'Work', 'Energy_Divergence', 'Efficiency']].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Matrice di Correlazione delle Variabili Termodinamiche e di Mercato')
    plt.tight_layout(pad=1.5)
    plt.savefig(os.path.join(cfg.paths.results_dir, 'correlation_matrix.png'))
    plt.close() 

    plt.figure(figsize=(16, 6))
    plt.plot(entropy_diff.index, entropy_diff, label='Divergenza Energetica (Levy - Sackur-Tetrode)', color='darkorange')
    plt.axhline(0, color='black', linestyle='--', alpha=0.5)
    plt.title('Divergenza Energetica tra Rendimenti Empirici e Gas Ideale', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.1)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.paths.results_dir, 'entropy_divergence.png'))
    plt.close() 
    print(f"[Stats] Analisi completata. Grafici salvati in {cfg.paths.results_dir}")
    
    return df_port


def calculate_entropy_difference(df, window=20):
    """
    Calcola la differenza tra l'entropia empirica dei rendimenti (Lévy-like)
    e l'entropia teorica di un gas ideale (Sackur-Tetrode).
    """
    # 1. Calcolo Rendimenti e Volatilità (Energia Interna U)
    returns = df['Close'].pct_change()
    volatility = returns.rolling(window).std()
    
    # 2. Volume (V) - Usiamo il logaritmo per stabilizzare
    volume = np.log1p(df['Volume'].rolling(window).mean())
    
    # Inizializziamo le serie per i risultati
    levy_entropy = pd.Series(index=df.index, dtype=float)
    st_entropy = pd.Series(index=df.index, dtype=float)

    for i in range(window, len(df)):
        # --- Entropia di Lévy (Empirica tramite Shannon) ---
        sample = returns.iloc[i-window:i].dropna()
        if len(sample) > 0:
            prob_dist, _ = np.histogram(sample, bins=15, density=True)
            prob_dist = prob_dist[prob_dist > 0]
            levy_entropy.iloc[i] = entropy(prob_dist)
        
        # --- Entropia di Sackur-Tetrode (Gas Ideale) ---
        # Sostituiamo le costanti fisiche (h, m, kB) con 1 o costanti di scala
        # Poiché U ~ T e T ~ Volatilità^2 (nello spazio delle fasi finanziario)
        V = volume.iloc[i]
        U = volatility.iloc[i]**2 
        print(V, U)
        if np.isfinite(V) and np.isfinite(U) and V > 0 and U > 0:
            # Versione semplificata: ln(V) + 1.5 * ln(U) + costante
            # La costante 2.5 deriva dai gradi di libertà del gas ideale
            st_entropy.iloc[i] = np.log(V) + 1.5 * np.log(U) + 2.5
            
    # Normalizzazione per rendere confrontabili le due misure
    # (Hanno unità di misura diverse, quindi calcoliamo la differenza di Z-Score o delta)
    s1 = (levy_entropy - levy_entropy.mean()) / levy_entropy.std()
    s2 = (st_entropy - st_entropy.mean()) / st_entropy.std()
    
    return s1 - s2 # Divergenza Energetica (Non-ideality score)