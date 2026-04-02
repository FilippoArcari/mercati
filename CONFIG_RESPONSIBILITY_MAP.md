# Config Files: Chi fa Cosa?

## Albero di Responsabilità

### Daily vs Intraday: Dove vengono configurate le DIFFERENZE

```plaintext
FREQUENZA?
├─ frequency=daily
│  └─ Attiva:
│     ├─ config/frequency/daily.yaml
│     │  ├─ split_date='2023-01-01'  ← Dataset split PER DATA
│     │  └─ NO max_history_days       ← Usa TUTTI i dati storici
│     │
│     ├─ config/training/default.yaml
│     │  ├─ learning_rate: 0.001      ← CONSERVATIVO
│     │  ├─ batch_size: 32            ← PICCOLO (pochi dati)
│     │  ├─ patience: 50              ← LUNGO (convergenza lenta)
│     │  └─ lambda_entropy: 0.05      ← ALTO (stabilità)
│     │
│     ├─ config/prediction/default.yaml
│     │  ├─ window_size: 30           ← Finestra LUNGA
│     │  └─ stride: 1                 ← Tutti i campioni (~2000)
│     │
│     └─ config/buyer/default.yaml
│        ├─ n_episodes: 100           ← Training breve
│        └─ lambda_concentration: 0.8 ← Diversificazione moderata
│
├─ frequency=minute
│  └─ Attiva:
│     ├─ config/frequency/minute.yaml
│     │  ├─ max_history_days: 30      ← Dataset LIMITED (30 giorni)
│     │  └─ split_ratio: 0.8          ← Split 80/20 TEMPORALE
│     │
│     ├─ config/training/minute.yaml
│     │  ├─ learning_rate: 0.002      ← DOPPIO (dati 50x)
│     │  ├─ batch_size: 128           ← 4x (batch ampi)
│     │  ├─ patience: 20              ← CORTO (convergenza veloce)
│     │  └─ lambda_entropy: 0.02      ← BASSO (priorità predict)
│     │
│     ├─ config/prediction/minute.yaml
│     │  ├─ window_size: 15           ← Finestra CORTA
│     │  └─ stride: 2                 ← Campioni densi (~5800)
│     │
│     └─ config/buyer/minute.yaml
│        ├─ n_episodes: 400           ← Training lungo
│        └─ lambda_concentration: 1.2 ← Forte diversificazione
│
└─ Cosa NON cambia?
   ├─ config/data/default.yaml        ← Ticker, feature list
   ├─ config/model/cnn.yaml           ← CNN architecture (64→32→16)
   ├─ config/paths/default.yaml       ← Directories
   └─ config/ddpg/default.yaml        ← (quando specifichera')
```

---

## Mappa di Impatto: Quale config influenza quale aspetto?

| Aspetto | Daily | Intraday | Responsabile Config |
|---------|-------|----------|---------------------|
| **Dataset Size** | ~2000 samples | ~5800 samples | frequency/{daily,minute}.yaml + prediction/{default,minute}.yaml |
| **CNN Learning Speed** | Lenta (0.001) | Veloce (0.002) | training/{default,minute}.yaml |
| **Batch Size** | 32 | 128 | training/{default,minute}.yaml |
| **Pattern Duration** | Giorni | Minuti | prediction/{default,minute}.yaml (window w=30 vs w=15) |
| **Sample Density** | Rado (stride=1) | Denso (stride=2) | prediction/{default,minute}.yaml |
| **DDPG Episodes** | 100+300 | 400 | buyer/{default,minute}.yaml |
| **Position Limits** | None | 10% per ticker | buyer/minute.yaml (max_position_pct) |
| **Max Holding Time** | Illimitato | 100 step (~25 min) | buyer/minute.yaml (max_holding_steps) |
| **Sell Incentive** | Debole (λ_inact=0.05) | Forte (λ_inact=0.3) | buyer/{default,minute}.yaml |
| **Concentration Penalty** | Moderata (λ_conc=0.1) | Forte (λ_conc=1.2) | buyer/{default,minute}.yaml |

---

## Quindi... Qual è la VERA Differenza?

### 🟢 Aspetti UGUALI in entrambi

```yaml
# I seguenti NOT CAMBIANO fra daily/minute:
- model/cnn.yaml:      Architecture (same 3 conv blocks)
- data/default.yaml:   Ticker list, feature preprocessing
- paths/default.yaml:  Checkpoint, output directories
- modelli/pred.py:     CNN code (identico)
- modelli/trade.py:    Trading loop (identico)
```

### 🔴 Aspetti CHE CAMBIANO

```yaml
# CONFIGURAZIONE (yaml files):
- training/default.yaml VS training/minute.yaml
  └─ learning_rate, batch_size, early_stopping.patience

- prediction/default.yaml VS prediction/minute.yaml
  └─ window_size, stride → numero campioni

- buyer/default.yaml VS buyer/minute.yaml
  └─ episode count, reward shaping, position limits

# COMPORTAMENTO DI RUNTIME (trading_env.py):
- max_position_pct=0.1        ← Only loaded with frequency=minute
- max_holding_steps=100       ← Only loaded with frequency=minute
  └─ Force liquidation se > 100 step
```

---

## Quick Checklist: Hai Distinto Bene?

Prima di lanciare un training, verifica:

```bash
# Verificare daily setup
uv run main.py step=train frequency=daily
  ✓ Vedi: "training: default"
  ✓ Vedi: "learning_rate: 0.001"
  ✓ Vedi: "batch_size: 32"
  ✓ Vedi: "epochs_phase1: 30"
  ✓ Vedi: "window_size: 30, stride: 1"

# Verificare intraday setup
uv run main.py step=train frequency=minute
  ✓ Vedi: "training: minute"
  ✓ Vedi: "learning_rate: 0.002"  ← 2x vs daily
  ✓ Vedi: "batch_size: 128"       ← 4x vs daily
  ✓ Vedi: "epochs_phase1: 10"     ← 1/3 vs daily
  ✓ Vedi: "window_size: 15, stride: 2"
  ✓ Vedi: "max_position_pct: 0.1"
  ✓ Vedi: "max_holding_steps: 100"
```

---

## File Organization

```
config/
├─ frequency/
│  ├─ daily.yaml          ← Data loading strategy (per-date split)
│  └─ minute.yaml         ← Data loading strategy (80/20 split, max_history=30)
│
├─ training/
│  ├─ default.yaml        ← CNN per daily (conservative)
│  └─ minute.yaml         ← CNN per intraday (aggressive)
│
├─ prediction/
│  ├─ default.yaml        ← Windows per daily (w=30, stride=1)
│  └─ minute.yaml         ← Windows per intraday (w=15, stride=2)
│
├─ buyer/
│  ├─ default.yaml        ← DDPG per daily (100 ep, λ_conc=0.8)
│  └─ minute.yaml         ← DDPG per intraday (400 ep, λ_conc=1.2, position limits)
│
└─ [other files - unchanged]
```

---

## Cosa Accade Quando Cambi frequency=

### Step 1: CLI Parsing
```bash
uv run main.py step=train frequency=minute
                                     ↓
```

### Step 2: _apply_frequency_defaults() (main.py)
```python
if freq_override == 'minute':
    sys.argv.append('training=minute')   ← Auto-add
    sys.argv.append('prediction=minute') ← Auto-add
    sys.argv.append('buyer=minute')      ← Auto-add
```

### Step 3: Hydra Composition
```yaml
# config/frequency/minute.yaml
defaults:
  - training: minute        ← Carica training/minute.yaml
  - prediction: minute      ← Carica prediction/minute.yaml
  - buyer: minute           ← Carica buyer/minute.yaml
```

### Step 4: Runtime
```python
# main.py carica cfg.training.learning_rate
if cfg.frequency.interval == 'minute':
    learning_rate = 0.002    ← Da training/minute.yaml
else:
    learning_rate = 0.001    ← Da training/default.yaml
```

---

## Verification Script (Optional)

Per verificare che la distinzione sia corretta, puoi lanciare:

```bash
# Controlla config daily
uv run main.py step=test frequency=daily 2>&1 | grep -E "learning_rate|window_size|batch_size"

# Dovrebbe stampare:
# learning_rate: 0.001
# window_size: 30
# batch_size: 32

# Controlla config intraday
uv run main.py step=test frequency=minute 2>&1 | grep -E "learning_rate|window_size|batch_size"

# Dovrebbe stampare:
# learning_rate: 0.002
# window_size: 15
# batch_size: 128
```

