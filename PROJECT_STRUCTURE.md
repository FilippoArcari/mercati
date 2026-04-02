# 📊 Struttura Completa del Progetto Mercati

**Data**: 26 Marzo 2026  
**Tipo**: Framework di Algorithmic Trading con Deep Learning + Termodinamica Quantistica  
**Linguaggio**: Python 3.x + PyTorch  

---

## 📑 Indice

1. [Panoramica Generale](#panoramica-generale)
2. [Struttura Directory](#struttura-directory)
3. [Pipeline di Esecuzione](#pipeline-di-esecuzione)
4. [Moduli Core](#moduli-core)
5. [Configurazione](#configurazione)
6. [Output e Risultati](#output-e-risultati)

---

## 🎯 Panoramica Generale

Questo progetto implementa un **framework di trading algoritmico ibrido** che combina:

- **Predizione con CNN Dilata**: Rete neurale convoluzionale 1D con dilatazioni per prevedere i prezzi
- **Feature Termodinamiche**: Integrazione di pressione di mercato, entropia, e resilienza ai tassi
- **Trading Agent DDPG**: Deep Deterministic Policy Gradient per decidere le allocazioni
- **Van der Waals Calibration**: Modello fisico per quantificare la pressione di mercato

Il sistema supporta **due regimi di trading**:
- **Daily**: Swing trading su timeframe giornaliero
- **Minute**: Intraday trading su timeframe 1-minuto

---

## 📁 Struttura Directory

```
mercati/
├── main.py                              # Entry point principale (orchestrazione pipeline)
├── pyproject.toml                       # Configurazione dependency (uv, poetry)
├── config/                              # Parametri YAML (Hydra)
│   ├── config.yaml                      # Config master
│   ├── buyer/                           # Parametri agente di trading
│   │   ├── default.yaml                 # Daily trading params
│   │   └── minute.yaml                  # Intraday trading params
│   ├── data/                            # Data loading params
│   │   └── default.yaml
│   ├── frequency/                       # Frequenza temporale
│   │   ├── daily.yaml
│   │   └── minute.yaml
│   ├── model/                           # Architettura rete neurale
│   │   └── cnn.yaml
│   ├── paths/                           # Directory output
│   │   └── default.yaml
│   ├── prediction/                      # Parametri predittore
│   │   ├── default.yaml
│   │   └── minute.yaml
│   └── training/                        # Parametri training
│       ├── default.yaml
│       └── minute.yaml
│
├── modelli/                             # Core logic (importato in main.py)
│   ├── __init__.py
│   ├── pred.py                          # ⭐ Modello predictor + MrE Loss
│   ├── trade.py                         # ⭐ Orchestr. trading agent DDPG
│   ├── ddpg.py                          # ⭐ Implementazione DDPG agent
│   ├── dqn.py                           # Agente DQN alternativo (experimental)
│   ├── trading_env.py                   # ⭐ Ambiente Gym per DDPG
│   ├── obs_normalizer.py                # ⭐ Normalizzatore osservazioni online
│   ├── thermodynamics.py                # ⭐ Feature termodinamiche + Ψ
│   ├── calibrate_vdw.py                 # Van der Waals calibration (Gabaix 2003)
│   ├── evaluate_pred.py                 # Metriche valutazione predittore
│   ├── utils.py                         # ⭐ Data loading, preprocessing
│   └── __pycache__/
│
├── checkpoints/                         # Modelli salvati
│   ├── pred_1d_w30.pth                  # Predictor daily (window=30)
│   ├── pred_1m_w10.pth                  # Predictor minute (window=10)
│   ├── ddpg_1d.pth                      # DDPG agent daily
│   ├── ddpg_1m.pth                      # DDPG agent minute
│   ├── ddpg_best_1d.pth                 # Best DDPG daily (ES checkpoint)
│   ├── ddpg_best_1m.pth                 # Best DDPG minute
│   ├── normalizer_1d.npz                # Stats normalizer daily
│   └── normalizer_1m.npz                # Stats normalizer minute
│
├── results/                             # Output trading
│   ├── trade_log.csv                    # Tutte le transazioni
│   ├── portfolio_daily.csv              # Valore portafoglio per step
│   ├── summary_per_ticker.csv           # Statistiche per ticker
│   └── value_per_ticker.csv             # Valore holding per ticker
│
├── outputs/                             # Run history (timestamped)
│   ├── 2026-03-17/
│   │   ├── 11-22-51/
│   │   └── ...
│   └── 2026-03-25/
│
├── data_daily.csv                       # Cache dati daily
├── data_minute.csv                      # Cache dati minute (1m)
│
└── Docs/
    ├── README.md                        # Descrizione generale
    ├── TRAINING_CONFIG_COMPARISON.md    # Daily vs Minute
    ├── VDW_INTEGRATION_GUIDE.md         # Van der Waals
    ├── CONFIG_ORCHESTRATION.md          # Sistema config
    └── CONFIG_RESPONSIBILITY_MAP.md     # Chi usa quale param

```

---

## ▶️ Pipeline di Esecuzione

```
┌─────────────────────────────────────────────────────────────┐
│  main.py (Entry Point)                                      │
│  - Legge config YAML via Hydra                              │
│  - Esegue step: train | test | trade                        │
└─────────────────────────────────────────────────────────────┘
                            │
                ┌───────────┼───────────┐
                ▼           ▼           ▼
            ┌────────┐ ┌────────┐ ┌───────────┐
            │ TRAIN  │ │ TEST   │ │  TRADE    │
            └────────┘ └────────┘ └───────────┘
                │           │           │
                ▼           ▼           ▼
         [Predictor    [Eval CNN]  [Trading Agent]
          Training]                  DDPG


Step 1: DATA LOADING (utils.py)
   └─ load_data() → normalizzato MinMax, con feature termodinamiche
   
Step 2: WINDOWING (utils.py)
   └─ make_windows() → (X, Y) per CNN
   
Step 3a: TRAINING [step=train]
   ├─ build_predictor() → Pred network
   ├─ Pred.fit() → MrE Loss (entropica + momento)
   └─ Checkpoints salvati
   
Step 3b: TESTING [step=test]
   ├─ Carica checkpoints
   ├─ Predizioni su train + test
   └─ Grafici via evaluate_predictions()
   
Step 3c: TRADING [step=trade]
   ├─ TradingEnv() → ambiente Gym
   ├─ DDPGAgent() → policy learning
   ├─ ObsNormalizer() → normalizzazione stati
   ├─ train_ddpg_normalized() → training loop
   └─ Trades + Portfolio value salvati
```

---

## 🔧 Moduli Core

### 1️⃣ **main.py** — Entry Point & Orchestrazione

**Funzioni Pubbliche:**

#### `_apply_frequency_defaults()`
- **Cosa fa**: Auto-configura gli override di CLI in base alla frequenza specificata
- **Input**: Sistema argv (CLI arguments)
- **Output**: Modifica sys.argv aggiungendo training=, prediction=, buyer= automaticamente
- **Dettagli**: Se passi `frequency=minute`, aggiunge automaticamente `training=minute prediction=minute buyer=minute`
- **Uso tipico**: Chiamato sempre all'avvio, prima di Hydra

#### `checkpoint_name(cfg, name: str) -> str`
- **Cosa fa**: Costruisce il path di un checkpoint includendo frequenza e window_size
- **Input**: 
  - `cfg`: Config Hydra
  - `name`: Nome base (es. "pred.pth", "ddpg_best.pth")
- **Output**: Path completo (es. "checkpoints/pred_1m_w10.pth")
- **Dettagli**: 
  - Predictor: `pred_1d_w30.pth` (con window_size)
  - Agenti: `ddpg_1m.pth` (solo frequenza)
- **Uso**: Ogni volta che salvi/carichi checkpoint

#### `compute_moment_target(train_df, tickers, cfg) -> torch.Tensor`
- **Cosa fa**: Calcola il target dei momenti per MrE Loss
- **Input**:
  - `train_df`: DataFrame dei dati di training
  - `tickers`: Lista ticker
  - `cfg`: Configurazione (rilegge MrE settings)
- **Output**: Tensor float32 di shape (num_features,)
- **Dettagli**: 
  - Se config ha `training.mre.moment_target` manuale → lo usa
  - Altrimenti calcola come media empirica del train_df
- **Uso**: Solo con MrE Loss abilitato

#### `build_predictor(cfg, num_features: int) -> Pred`
- **Cosa fa**: Factory function che istanzia la rete CNN predictor
- **Input**:
  - `cfg`: Hydra config
  - `num_features`: Dimensione input (prezzi + features termodinamiche)
- **Output**: Istanza `Pred` pronta al training
- **Dettagli**: Legge da config: num_features, window_size, dimensions, dilations, kernel_size, activation
- **Uso**: In step=train e step=test

#### `split_dataframe(df: pd.DataFrame, cfg) -> tuple[pd.DataFrame, pd.DataFrame]`
- **Cosa fa**: Splotta dataset in train/test per percentuale o data
- **Input**:
  - `df`: DataFrame completo
  - `cfg`: Config (frequency.split_ratio o data.split_date)
- **Output**: (train_df, test_df)
- **Dettagli**: 
  - Intraday (minute): Split per percentuale (default 80/20)
  - Daily: Split per data esatta
- **Validates**: Entrambi i set ≥ window_size + 1
- **Uso**: Sempre dopo load_data()

#### `my_app(cfg: DictConfig) -> None`
- **Cosa fa**: Funzione main decorata con @hydra.main
- **Input**: Config DictConfig da Hydra
- **Output**: Nessuno (side effects: fichier salvati)
- **Flusso**:
  1. Load config + stampa
  2. Load dati via utils.load_data()
  3. Split train/test
  4. Crea windows (X, Y)
  5. Se step=="train": Addestra Pred, salva checkpoint
  6. Se step=="test": Carica pred, predici, valuta con evaluate_predictions()
  7. Se step=="trade": Chiama run_trade()
- **Uso**: `uv run main.py step=train frequency=daily` (main entry point)

---

### 2️⃣ **modelli/utils.py** — Data Loading & Preprocessing

**Funzioni Pubbliche:**

#### `load_data(tickers, start_date, end_date, fred_api_key, inflation_series, interval="1d", cache_path="./data.csv", max_history_days=None, add_thermodynamics=True, thermo_window=20, thermo_max_lag=90) -> tuple[pd.DataFrame, MinMaxScaler]`

**Cosa fa**: Carica dati da yfinance + FRED, calcola feature termodinamiche, normalizza MinMax

**Input**:
- `tickers`: Lista ticker (es. ["AAPL", "MSFT"])
- `start_date`, `end_date`: datetime.datetime
- `fred_api_key`: API key FRED (per tassi)
- `inflation_series`: Lista serie FRED (es. ["GS10", "T10YIE"])
- `interval`: "1d", "1m", etc (da yfinance)
- `cache_path`: Dove cacheare i dati
- `max_history_days`: Limite storico per intraday (auto-impostato per "1m": 7 giorni)
- `add_thermodynamics`: Se aggiungere feature termodinamiche
- `thermo_window`: Finestra rolling per entropia (default 20)
- `thermo_max_lag`: Lag massimo cross-correlazione tassi (default 90)

**Output**: 
- DataFrame normalizzato (float32, MinMax [0,1])
- MinMaxScaler object (per inverse_transform)

**Dettagli Interni**:

1. **Cache**: Se esiste data.csv e è valida, la riusa (fast)
2. **Download yfinance**: 
   - Scarica Close + Volume
   - Auto-limita intraday a 7 giorni storici
3. **FRED Series** (solo daily):
   - GS10 (tasso 10Y governo), T10YIE (inflazione attesa), FEDFUNDS, ^TNX
   - Forward-fill su missing
4. **Feature Termodinamiche**:
   - **Daily**: Completo (P, T, Entropia, W_cum, Psi, Divergenza-energetica)
   - **Intraday**: Semplificato (P, T, W_cum) con finestra adattata
5. **Normalizzazione MinMax**: Applica a **dopo** termodinamica (per preservare semantica)

**Correlazioni Chiave** (da heatmap):
- Market_Pressure: -0.64 con Close (inversamente predittiva)
- Market_Temperature: -0.67 con Close
- Market_Work_Cum: +0.67 con Close (positivamente predittiva)

**Uso Tipico**:
```python
df, scaler = load_data(
    ["AAPL", "MSFT"], 
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31),
    fred_api_key="...",
    inflation_series=["GS10"],
    interval="1d"
)
# df shape: (250, 10) — 250 barre, 10 colonne (prezzi + termodinamica)
```

---

#### `make_windows(data: pd.DataFrame, window_size: int, stride: int) -> tuple[torch.Tensor, torch.Tensor]`

**Cosa fa**: Crea sequential windows (X, Y) per training CNN

**Input**:
- `data`: DataFrame normalizzato
- `window_size`: Lunghezza storia (es. 30 giorni o 10 minuti)
- `stride`: Passo sliding window (default 1)

**Output**: 
- X: Tensor (num_windows, window_size, num_features)
- Y: Tensor (num_windows, num_features)

**Dettagli**:
- Crea finestre non-overlapping o overlapping in base a stride
- Y è il next candle dopo window
- Valida che num_windows > 0

**Formula**: 
- num_windows = (len(data) - window_size) / stride

**Uso Tipico**:
```python
X_train, Y_train = make_windows(train_df, window_size=30, stride=1)
# X_train: (230, 30, 10)  — 230 finestre di 30 bar con 10 features
# Y_train: (230, 10)       — 230 target (prezzo prossimo candle)
```

---

#### `calculate_market_thermodynamics(df, n_assets=1, kb=1.0) -> tuple[pd.Series, pd.Series]`

**Cosa fa**: Calcola Pressione e Lavoro Cumulativo per un singolo asset/portafoglio

**Input**:
- `df`: DataFrame con colonne 'Close' e 'Volume'
- `n_assets`: Numero di asset (default 1, usato per scaling)
- `kb`: Costante Boltzmann (default 1.0)

**Output**: 
- (p_series, work_cumulative) — due pd.Series di stessa lunghezza

**Dettagli**: 
- Usa entropìa rolling (20 bar)
- Modello Van der Waals semplificato (heuristico, non calibrato)
- Penalizza con autocorrelazione

**Uso**: Interno a load_data() per feature termodinamiche

---

#### `make_stats(cfg) -> None`

**Cosa fa**: Analisi termodinamica completa con divergenza energetica e grafici

**Input**: `cfg` Hydra config

**Output**: Grafici salvati in cartella definita da config

**Dettagli**: Calcola divergenza tra pressione di mercato e tassi

**Uso**: Utility per analisi post-hoc (non nel main training loop)

---

### 3️⃣ **modelli/pred.py** — Rete CNN Predictor + MrE Loss

**Classi Pubbliche:**

#### `class PriorEstimator`

**Scopo**: Stima μ e σ in modo efficiente senza caricare tutto il dataset in memoria (Welford algorithm)

**Metodi**:

##### `fit(train_loader: DataLoader) -> None`
- Calcola media e std con algoritmo Welford (numericamente stabile)
- Accumula somme su batch senza torch.cat (memory-efficient)
- Stampa numero campioni elaborati

##### `@property fitted -> bool`
- Ritorna True se mean è stato calcolato

---

#### `class MrELoss(nn.Module)` — Maximum Relative Entropy Loss

**Scopo**: Loss ottimizzata per MrE che combina 3 termini

**Parametri**:
- `lambda_entropy`: Peso KL (default 0.1)
- `lambda_moment`: Peso vincolo momenti (default 0.1)

**Metodi**:

##### `_kl_gaussian(pred, prior_mean, prior_std) -> torch.Tensor`
- Calcola KL divergence tra predizioni e prior Gaussiano
- Clamp varianza a min 0.1 per evitare divisione per zero su dati minute
- Ritorna scalare loss

##### `forward(pred, target, prior_mean=None, prior_std=None, moment_target=None, data_weight=1.0) -> tuple[torch.Tensor, dict]`
- **Calcola**:
  ```
  L = data_weight * MSE(pred, target)
      + lambda_entropy * KL(pred || prior)
      + lambda_moment * ||mean(pred) - F||²
  ```
- **Output**: (total_loss, info_dict)
- **Info dict**: {"data": ..., "entropy": ..., "moment": ..., "total": ...}

**Dettagli**:
- Data term: MSE standard (likelihood)
- Entropy term: Penalizza predizioni troppo diverse dal prior
- Moment term: Forza batch mean a coincidere con target F

---

#### `class Pred(nn.Module)` — CNN Predictor Network

**Scopo**: Modello convoluzionale per predire il prossimo candle da una sequenza

**Architettura**:
```
Input: (B, T, F) — Batch × Time × Features
  ↓
Permute: (B, F, T)
  ↓
Conv1d blocks (con dilatazioni):
  - Layer 1: Conv1d(F, dim[0], kernel=3, dilation=d[0])
  - Layer 2: Conv1d(dim[0], dim[1], kernel=3, dilation=d[1])
  - ...
  - Ogni layer: Conv → BatchNorm → LeakyReLU
  ↓
Flatten: (B, dim[-1]*T)
  ↓
FC: (B, dim[-1]*T) → (B, F)
```

**Parametri**:
- `num_features`: Dimensione input (es. 10)
- `window_size`: T (es. 30)
- `dimension`: Lista dimensioni intermedie (es. [64, 32, 16])
- `dilations`: Dilatazioni per ogni layer (es. [1, 2, 4])
- `kernel_size`: Kernel size Conv (default 3)
- `activation`: Nome funzione attivazione (default "leaky_relu")

**Metodi**:

##### `forward(x: torch.Tensor) -> torch.Tensor`
- Input shape: (B, T, F)
- Output shape: (B, F) — un candle predetto per sample
- Applica la CNN completa

##### `_run_phase(train_loader, training_cfg, criterion, prior_mean, prior_std, moment_target, data_weight=1.0, label="", epochs=None)`
**Cosa fa**: Esegue una fase di training
- Legge optimizer da config
- Loop over epochs
- Per ogni batch: forward → loss → backward → optimizer.step()
- Accumula loss e stampa ogni 10 epoch
- **Dettagli**: 
  - Sposta prior_mean/std su device corretti una volta
  - Usa set_to_none=True in zero_grad per velocità

##### `fit(train_loader, training_cfg, moment_target=None)`

**Cosa fa**: Funzione di allenamento principale con supporto MrE

**Flusso**:
1. Se MrE disabilitato → fallback a MSE puro rapido
2. Se MrE abilitato:
   - Crea criterion MrELoss
   - Istanzia PriorEstimator e lo fitta
   - **Mode sequenziale**: 
     - Fase 1 (epochs_phase1): Solo vincolo momenti (data_weight=0)
     - Refit prior (lento ma necessario per paper)
     - Fase 2: Solo dati (data_weight=1)
   - **Mode simultaneo**: Una sola fase con tutti termini

**Uso**:
```python
predictor = Pred(num_features=10, window_size=30, ...)
predictor.fit(train_loader, cfg.training, moment_target=F)
```

---

### 4️⃣ **modelli/trading_env.py** — Ambiente Gym per DDPG

#### `class TradingEnv`

**Scopo**: Ambiente episodico di trading che simula l'esecuzione di ordini DDPG

**Stato per timestep**:
```
[prezzi_reali(F) | prezzi_predetti(F) | holdings_ratio(T) | 
 cash_ratio(1)   | portfolio_ratio(1) | psi_values(T)]

dove:
  F = num_features (lun. del dataframe)
  T = num_tickers
  Se Ψ non disponibile → psi_values = zeros(T)
```

**Reward Shaping**:
```
reward = pct_return
       - lambda_concentration * (Σ holdings_ratio_i²)    [Herfindahl index]
       - lambda_inaction * (frac_tickers con |action|<soglia)
```

**Parametri Costruttore**:
- `prices_real`: DataFrame prezzi reali
- `prices_pred`: DataFrame prezzi predetti (da CNN)
- `tickers`: Lista ticker (es. ["AAPL", "MSFT"])
- `initial_capital`: Float (default 10k)
- `transaction_cost`: Float (default 0.1% = 0.001)
- `psi_df`: DataFrame con colonne Psi_{ticker} (opzionale, per resilienza tassi)
- `lambda_concentration`: Peso penalità concentrazione (default 0.1)
- `lambda_inaction`: Peso penalità inaction (default 0.05)
- `action_threshold`: Min |action| per considerarlo non-inaction (default 0.05)
- `max_position_pct`: Max peso singolo ticker (default 10%)
- `max_holding_steps`: Max step di hold (default 100)

**Metodi Pubblici**:

##### `reset() -> np.ndarray`
- Reset stato interno (cash, holdings, portfolio_history)
- Ritorna stato iniziale

##### `step(action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]`
- Esegue azione (array T-dim in [-1, 1])
- Aggiorna holdings e cash
- Calcola reward (return % + penalità)
- Ritorna (next_state, reward, done, info)
- **Done**: Quando raggiunge end of data

##### `portfolio_history() -> list[float]`
- Storico valore portafoglio per ogni step

##### `value_per_ticker_df() -> pd.DataFrame`
- Valore holdings per ticker nel tempo

##### `trade_log_df() -> pd.DataFrame`
- Tutte le transazioni eseguite

**Dettagli Rilevanti**:
- Action [0, 1] = BUY/HOLD long
- Action [-1, 0] = SELL/SHORT
- Calcola exposure per ticker e lo normalizza
- Applica max_position_pct e transaction_cost

---

### 5️⃣ **modelli/ddpg.py** — Deep Deterministic Policy Gradient

#### `class OUNoise`

**Scopo**: Rumore Ornstein-Uhlenbeck per esplorazione continua

**Metodi**:

##### `sample() -> np.ndarray`
- Ritorna campione di rumore N(action_dim)

##### `decay(factor=0.995, floor=0.05)`
- Decrementa sigma (rumore) nel tempo verso floor

##### `reset()`
- Reset stato interno (per nuovo episodio)

---

#### `class ReplayBuffer`

**Scopo**: Buffer di memoria per esperienza passata

**Metodi**:

##### `push(state, action, reward, next_state, done)`
- Aggiunge transizione

##### `sample(batch_size) -> tuple`
- Ritorna batch casuale di transizioni come Tensor PyTorch

##### `__len__() -> int`
- Numero di transizioni nel buffer

---

#### `class Actor(nn.Module)`

**Scopo**: Policy network che mapva stato → azione

**Architettura**:
```
Input: stato (state_dim)
  ↓
Hidden: [256, 256]
  - Linear + LayerNorm + LeakyReLU(0.2)
  ↓
Output: Linear → Tanh  [mappa in [-1, 1]]
```

**Inizializzazione**:
- Orthogonal init per stabilità
- Gain basso (0.01) per output layer → favorisce esplorazione iniziale

---

#### `class Critic(nn.Module)`

**Scopo**: Q-function che stima valore Q(s, a)

**Architettura**:
```
Input: stato (state_dim) + azione (action_dim)
  ↓
State branch: Linear → LayerNorm → LeakyReLU(0.2)
  ↓
Concatena [state_h, action]
  ↓
Hidden layers: [256, ...] (configurable)
  ↓
Output: Linear → scalare (valore Q)
```

---

#### `class DDPGAgent`

**Scopo**: Agente DDPG completo con replay buffer e noise

**Parametri Costruttore**:
- `state_dim`, `action_dim`: Dimensioni
- `actor_hidden`, `critic_hidden`: Architetture reti
- `lr_actor`: Learning rate actor (default 1e-4)
- `lr_critic`: Learning rate critic (default 3e-4)
- `gamma`: Discount factor (default 0.99)
- `tau`: Soft update coefficient (default 0.005)
- `buffer_capacity`: Replay buffer capacity (default 100k)
- `batch_size`: Batch size training (default 64)
- `update_every`: Aggiorna ogni N steps (default 1)
- `noise_sigma`: Std dev rumore OU (default 0.2)

**Metodi Pubblici**:

##### `act(state, explore=True) -> np.ndarray`
- Ritorna azione da [-1, 1]
- Se explore=True: aggiunge rumore OU
- Se explore=False: azione pura (deterministica)

##### `store(state, action, reward, next_state, done)`
- Aggiunge transizione al replay buffer

##### `update() -> dict | None`
- Update critic e actor se buffer abbastanza grande
- Ritorna {"critic_loss": ..., "actor_loss": ...} o None

**Algoritmo**:
1. Sample batch dal replay buffer
2. Compute target Q con critic_target e actor_target
3. Update critic minimizzando smooth_l1(Q_current - Q_target)
4. Update actor massimizzando Q(s, actor(s))
5. Soft update target networks (τ = 0.005)

##### `decay_noise(factor, floor)`
- Decrementa rumore durante training

##### `reset_noise()`
- Reset OU state per nuovo episodio

---

### 6️⃣ **modelli/obs_normalizer.py** — Normalizzazione Stati Online

#### `class RunningMeanStd`

**Scopo**: Stima incrementale media e varianza con algoritmo Welford

**Metodi**:

##### `update(x: np.ndarray)`
- Aggiorna con singolo campione o batch
- Utilizza Chan's parallel variance formula
- Numericamente stabile

##### `@property std() -> np.ndarray`
- Ritorna deviazione standard calcolata

---

#### `class ObsNormalizer`

**Scopo**: Normalizzatore di osservazioni pronto all'uso per DDPG

**Parametri**:
- `state_dim`: Dimensione stato
- `clip`: Max |valore| dopo normalizzazione (default 5.0, riduce outlier)
- `epsilon`: Piccolo valore per evitare divisione per zero

**Metodi**:

##### `normalize(obs: np.ndarray, update: bool = True) -> np.ndarray`
- Normalizza: `(obs - mean) / (std + eps)`
- Clipping: `clip(normalized, -clip, +clip)`
- Se `update=True`: aggiorna statistiche (training)
- Se `update=False`: solo trasforma (inference)
- Output: float32

##### `save(path: str)`
- Salva statistiche (mean, var, count) in .npz

##### `load(path: str)`
- Carica statistiche salvate

---

#### `train_ddpg_normalized(agent, env, normalizer, n_episodes=200, warmup=1000, log_every=10, noise_decay=0.995, noise_floor=0.05, es_patience=20, es_metric="sharpe", best_path=None, norm_path=None) -> list[dict]`

**Scopo**: Training loop DDPG con normalizzazione online

**Flusso**:
1. Per ogni episodio:
   - Reset env → raw_state
   - Normalizza: `state = normalizer.normalize(raw_state, update=True)`
   - Per ogni step:
     - Action da agent
     - Step env → next_state raw
     - Normalizza: `next_state = normalizer.normalize(next_state, update=True)`
     - Store in replay buffer
     - Agent.update()
2. Early stopping basato su sharpe o altro metrico
3. Salva best model + normalizer

**Output**: Lista di dict con history episode

---

### 7️⃣ **modelli/trade.py** — Orchestrazione Trading DDPG

**Funzioni Pubbliche**:

#### `run_trade(cfg, df, tickers, X_train, Y_train, X_test, Y_test, train_df, test_df)`

**Scopo**: Orchestrazione completa del training e evaluation dell'agente DDPG

**Flusso Interno**:
1. **Setup**:
   - Carica predictor CNN da checkpoint
   - Costruisce prezzi reali/predetti per train e test set
   
2. **Calcolo Ψ (Psi)**: 
   - Indice resilienza tassi per ticker
   - Se disponibili tarsi → compute_psi_series()
   
3. **Training Phase**:
   - Crea TradingEnv su train_df
   - Istanzia DDPGAgent + ObsNormalizer
   - Training loop con train_ddpg_normalized()
   - Salva best model + normalizer
   
4. **Evaluation Phase**:
   - Crea TradingEnv su test_df
   - Carica best model + normalizer
   - Evaluation con explore=False (deterministic)
   - Calcola metriche: Sharpe, Total Return, Sortino, etc.
   
5. **Output**:
   - Trade log CSV
   - Portfolio history CSV
   - Summary per ticker
   - Grafici portfolio value, equity curve, drawdown

**Output Files**:
- `results/trade_log.csv` — Tutte le transazioni
- `results/portfolio_daily.csv` — Valore portafoglio per step
- `results/summary_per_ticker.csv` — Statistiche
- `results/value_per_ticker.csv` — Valore holdings
- Grafici PNG in outputs/{data}/{ora}/

---

### 8️⃣ **modelli/thermodynamics.py** — Feature Termodinamiche + Ψ

#### `compute_thermodynamic_features(df_raw, ticker_cols, rates_col, window=20, max_lag=90, vdw_calibration=None) -> pd.DataFrame`

**Scopo**: Calcola tutte le feature termodinamiche di mercato

**Output Columns**:
- `Market_Pressure`: Pressione Van der Waals (empirica o heuristica)
- `Market_Temperature`: "Temperatura" (proxy entropia)
- `Market_Entropy`: Entropia rolling dei ritorni
- `Market_Work_Cum`: Lavoro cumulativo ∫P dV
- `Volume_Delta`: Variazione volume
- `Thermo_*`: Altre feature
- `Energy_Divergence`: Divergenza tassi-pressione

**Parametri**:
- `vdw_calibration`: Dict con VdWParams calibrati (opzionale)
- `window`: Finestra rolling (default 20)
- `max_lag`: Max lag cross-correlazione (default 90)

---

#### `compute_psi_series(df_raw, tickers, rates_col, window=30) -> pd.DataFrame`

**Scopo**: Calcola indice resilienza Ψ_i(t) per ticker, per timestep

**Formula**:
```
Ψ_i(t) = Corr_window(ΔW_i, Δr) × (σ_W_i / σ_r)
```

Dove:
- ΔW_i: Variazione giornaliera lavoro ticker i
- Δr: Variazione tassi
- σ_W_i, σ_r: Volatilità rolling

**Interpretazione**:
- Ψ alto → Asset sensibile ai tassi (ridurre se r↑)
- Ψ basso → Asset resiliente (mantenere)

**Output**: DataFrame con colonne `Psi_{ticker}` normalizzate [-1, 1]

---

#### `calculate_pressure_and_work(close, volume, window=20, kb=1.0, vdw_params=None, n_particles=1) -> pd.DataFrame`

**Scopo**: Calcola pressione e lavoro Van der Waals per singolo asset

**Algoritmo Van der Waals**:
```
P = nRT/(V-nb) - a·n²/V²
W = ∫P dV
```

**Se vdw_params fornito**: Usa parametri calibrati (Gabaix 2003)
**Altrimenti**: Fallback a parametri heuristici

---

#### `init_vdw_calibration(df_raw, ticker_cols, verbose=True) -> dict[str, VdWParams]`

**Scopo**: Calibra parametri Van der Waals da dati storici

**Metodo**: Analisi tail della distribuzione (Gabaix et al. 2003)

**Output**: Dict `{ticker: VdWParams(a, b)}`

---

### 9️⃣ **modelli/evaluate_pred.py** — Valutazione Predittore

#### `evaluate_predictions(predictions: pd.DataFrame, targets: pd.DataFrame, step: str, results_dir: str) -> None`

**Scopo**: Calcola metriche errore e genera grafici

**Metriche Calcolate**:
- MAE: Mean Absolute Error
- MSE: Mean Squared Error
- RMSE: Root MSE

**Output**:
- Grafici PNG per ogni ticker (Predizioni vs Reali)
- Statistiche errore nel titolo del grafico

---

### 🔟 **modelli/dqn.py** — Agente DQN (Experimental)

**Nota**: DQN è implementato ma non è il focus principale (DDPG è preferito per azioni continue)

#### `class DuelingDQN(nn.Module)`

**Scopo**: Dueling DQN architecture con multi-head per ticker (azioni discrete)

**Architettura**:
```
Backbone condiviso → N value heads (1 output) + N advantage heads (n_actions output)
Q(s, a_i) = V(s) + A(s, a_i) - mean(A(s, *))
```

#### `class DQNAgent`

**Metodi Principali**:
- `act()` → azione discreta
- `act_idx()` → indici grezzi
- `update()` → training step
- Double DQN + Prioritized Replay Buffer

---

## ⚙️ Configurazione

### Struttura YAML

Tutti i parametri sono in `config/`:

#### `config.yaml` — Master config
```yaml
defaults:
  - frequency: daily      # o minute
  - training: default     # o minute
  - prediction: default   # o minute
  - buyer: default        # o minute
  - data: default
  - model: cnn
  - paths: default

step: train        # train | test | trade
```

#### `frequency/daily.yaml` vs `frequency/minute.yaml`
```yaml
# daily.yaml
interval: 1d
cache_path: ./data_daily.csv
split_ratio: 0.8

# minute.yaml
interval: 1m
cache_path: ./data_minute.csv
split_ratio: 0.8
```

#### `training/default.yaml` vs `training/minute.yaml`
```yaml
# default (daily)
learning_rate: 0.001
batch_size: 32
epochs: 100
weight_decay: 1.0e-5

# minute
learning_rate: 0.002        # 2x più veloce
batch_size: 128             # 4x batch
epochs: 100
weight_decay: 0.5e-4        # meno regolarizzazione
```

#### `buyer/default.yaml` — Parametri Trading Agent
```yaml
ddpg:
  lr_actor: 1.0e-4
  lr_critic: 3.0e-4
  gamma: 0.99
  tau: 0.005
  buffer_capacity: 100000
  batch_size: 64
  noise_sigma: 0.2

training:
  n_episodes: 200
  warmup: 1000
  es_patience: 20
  noise_decay: 0.995
```

#### `model/cnn.yaml` — Architettura CNN
```yaml
dimensions: [64, 32, 16]    # Dimensioni layer
dilations: [1, 2, 4]        # Dilatazioni Conv1d
kernel_size: 3
activation: leaky_relu
```

---

## 📊 Output e Risultati

### `results/` Directory

Dopo `step=trade`, sono salvate:

#### `trade_log.csv`
Colonne: date, ticker, action, size, price, entry_price, unrealized_pnl

#### `portfolio_daily.csv`
Colonne:
- portfolio_value: Valore totale
- holdings_value: Valore posizioni aperte
- cash: Cash disponibile
- daily_return_pct: Rendimento giornaliero %
- cumulative_return_pct: Rendimento cumulativo %
- drawdown_pct: Drawdown da peak %
- {ticker}_value: Valore singolo ticker

#### `summary_per_ticker.csv`
Statistiche per ticker:
- total_trades: Numero transazioni
- total_return_pct: Rendimento totale %
- win_rate: Win rate %
- avg_trade: Media trade

#### `value_per_ticker.csv`
Valore holding per ticker nel tempo

### `outputs/{data}/{ora}/` Directory

Grafici per run:
- `portfolio_performance.png`: Portfolio curve + rendimento
- `predictions_vs_actual.png`: CNN predictions vs reali
- `heatmap_correlations.png`: Correlazioni feature

---

## 🚀 Comandi Tipici

```bash
# Setup
uv sync

# Training predictor (daily)
uv run main.py step=train frequency=daily

# Training predictor (minute)
uv run main.py step=train frequency=minute

# Test predictor
uv run main.py step=test frequency=daily

# Training trading agent
uv run main.py step=trade frequency=daily

# Alternativo: run DDPG su minute data
uv run main.py step=trade frequency=minute
```

---

## 📈 Pipeline Dettagliato per step=trade

```
1. Load predictor checkpoint
   └─ modelli/pred.py: Pred model + scaler

2. Crea prezzi reali/predetti
   ├─ Predizioni su train set
   └─ Predizioni su test set

3. Calcolo Ψ (resilienza tassi)
   └─ modelli/thermodynamics.py: compute_psi_series()

4. TRAINING PHASE
   ├─ Crea TradingEnv(train_df, prices_real, prices_pred, psi_df)
   ├─ Istanzia DDPGAgent + ObsNormalizer
   └─ train_ddpg_normalized() — loop:
       - Per ogni episodio:
         - Reset env
         - Esegui N steps
         - Agent.act() + env.step()
         - Agent.update()
         - Track reward, loss
       - Early stopping (sharpe)
       - Salva best model + normalizer

5. EVALUATION PHASE
   ├─ Carica best model + normalizer
   ├─ Crea TradingEnv(test_df, ...)
   └─ Valuta con explore=False:
       - Step through test set
       - Raccogli trade log
       - Calcola Sharpe, Sortino, etc.

6. OUTPUT
   └─ Salva CSV + grafici in results/
```

---

**Fine Documentazione**

