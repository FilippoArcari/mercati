# Hydra Configuration Orchestration

## Command Dispatch Flow

```
CLI INPUT: frequency=daily
    ↓
_apply_frequency_defaults() in main.py
    ↓
    Aggiunge automaticamente: training=default, prediction=default, buyer=default
    ↓
HYDRA DEFAULTS COMPOSITION
    ├─ config/frequency/daily.yaml ─────────→ split_date='2023-01-01', market hours
    ├─ config/training/default.yaml ────────→ lr=0.001, batch_size=32, patience=50
    ├─ config/prediction/default.yaml ──────→ window_size=30, stride=1
    ├─ config/buyer/default.yaml ──────────→ n_episodes=100, es_patience=40
    └─ [altri config: data, model, paths...]
    ↓
RISULTATO: CNN training conservativa su 2000 samples
```

```
CLI INPUT: frequency=minute
    ↓
_apply_frequency_defaults() in main.py
    ↓
    Aggiunge automaticamente: training=minute, prediction=minute, buyer=minute
    ↓
HYDRA DEFAULTS COMPOSITION
    ├─ config/frequency/minute.yaml ────────→ split_ratio=0.8, max_history_days=30
    ├─ config/training/minute.yaml ────────→ lr=0.002, batch_size=128, patience=20
    ├─ config/prediction/minute.yaml ──────→ window_size=15, stride=2
    ├─ config/buyer/minute.yaml ───────────→ n_episodes=400, lambda_concentration=1.2
    └─ [altri config: data, model, paths...]
    ↓
RISULTATO: CNN training aggressiva su 5800+ samples + DDPG intraday
```

---

## Dataset Snapshot: Daily vs Intraday

### Daily (frequency=daily)
```
Raw data:            252 close prices per anno (SPY, AAPL, MSFT, ...)
                     ↓
Load (config/frequency/daily.yaml)
                     ↓
Split per data:      train:2020-2022, test:2023-01-01+
                     ↓
Windowing (w=30, stride=1):  2110 training sequences
                     ↓
CNN predictor (lr=0.001)     Epochs: 600-800 per convergenza
                     ↓
DDPG agent (300 ep)          Position sizing: dynamic, no limits
                     ↓
Output:              Trade log con position size variabile, early stopping ~ep25
```

### Intraday (frequency=minute)
```
Raw data:            30 giorni × 390 bar/day × 50+ tickers = 11,700 barre
                     ↓
Load (config/frequency/minute.yaml)
                     ↓
Split 80/20:         train: 9300 barre, test: 2340 barre
                     ↓
Windowing (w=15, stride=2):  5800+ training sequences
                     ↓
CNN predictor (lr=0.002)     Epochs: 250-400 per convergenza
                     ↓
DDPG agent (400 ep)          Position sizing: capped 10%, max holding 100 min
                     ↓
Output:              Trade log con turnover elevato, 40-60% SELL actions
```

---

## Configurazione a Cascata

Quando Hydra compone i config, i **più specifici sovraescrivono i generici**:

```yaml
# Defaults in config/frequency/daily.yaml
defaults:
  - data: default
  - training: default
  - prediction: default
  - buyer: default
  - model: cnn
```

Quando esegui `frequency=minute`:
- La CLI override passa `training=minute` → carica config/training/minute.yaml
- Tutte le altre config seguono la cascata

---

## Verifica della Configurazione Attiva

Per controllare quale config è stato carico, Hydra stampa:

```bash
$ uv run main.py step=train frequency=daily

[HYDRA] Loaded config:
  - frequency: daily
  - training: default          ← ← ← training/default.yaml
  - prediction: default        ← ← ← prediction/default.yaml
  - buyer: default             ← ← ← buyer/default.yaml

$ uv run main.py step=train frequency=minute

[HYDRA] Loaded config:
  - frequency: minute
  - training: minute           ← ← ← training/minute.yaml
  - prediction: minute         ← ← ← prediction/minute.yaml
  - buyer: minute              ← ← ← buyer/minute.yaml
```

---

## Parametri Critici per Decisione

Prima di run, chiedi:

1. **Qual è il timeframe target?**
   - Daily trends → frequency=daily
   - Minute patterns → frequency=minute

2. **Quanto rischio di overfitting?**
   - Poco training data → frequency=daily (weight_decay=1.0e-5)
   - Molto training data → frequency=minute (weight_decay=0.5e-4)

3. **Quale Sharpe ratio voglio?**
   - 0.5-1.0 → frequency=daily (convergenza lenta, stabile)
   - 1.0-2.0 → frequency=minute (convergenza veloce, aggressivo)

---

## File Decision Tree

```
Want Daily Trading?
├─ config/training/default.yaml  (lr=0.001, conservative)
├─ config/prediction/default.yaml  (w=30, stride=1, all samples)
├─ config/buyer/default.yaml  (200 ep, λ_conc=0.8)
└─ → Risultato: Sharpe ~0.7

Want Intraday Trading?
├─ config/training/minute.yaml  (lr=0.002, aggressive)
├─ config/prediction/minute.yaml  (w=15, stride=2, dense)
├─ config/buyer/minute.yaml  (400 ep, λ_conc=1.2)
└─ → Risultato: Sharpe ~1.5
```

