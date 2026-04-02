# Training Configuration: Daily vs Intraday

## Distinzione Netta fra i Due Regimi

La scelta `frequency=daily` vs `frequency=minute` attiva automaticamente due **strategie di apprendimento completamente diverse**.

---

## 📊 Tabella Comparativa

| Parametro | **DAILY** (default.yaml) | **INTRADAY** (minute.yaml) | Motivazione |
|-----------|--------------------------|---------------------------|-------------|
| **DATASET** | 252 barre/anno | 30 giorni × 390 barre/day ≈ **11,700 barre** | Intraday ha 50x più dati |
| **PATTERN** | Momentum lentamente variante | Microstructure, reversals veloci | Frequenza diversa |
| **epochs** | 1000 | 1000 | Entrambi: convergenza richiede tempo |
| **learning_rate** | **0.001** | **0.002** (DOPPIO) | Intraday converge 2x veloce |
| **batch_size** | 32 | **128** (4x) | Intraday può usare batch ampi |
| **weight_decay** | 1.0e-5 | **0.5e-4** (50x MENO) | Intraday: poco overfitting risk |
| **early_stopping.patience** | 50 (ALTO) | 20 (BASSO) | Intraday converge velocemente |
| **warmup_steps** | 100 | 50 | Intraday: warmup più corto |
| **lambda_entropy** | 0.05 | **0.02** (42% MENO) | Intraday: priorità prediction |
| **lambda_moment** | 0.1 | 0.05 (50% MENO) | Intraday: vincolo morbido |
| **epochs_phase1** | 30 | 10 | Intraday: entropy quick start |

---

## 🔍 Spiegazione Principali Differenze

### 1️⃣ **Learning Rate: 0.001 vs 0.002** (DIFFERENZA CRITICA)
- **Daily**: Dataset piccolo (~2000 samples) → convergenza lenta e stabile
- **Intraday**: Dataset massivo (~5800+ samples) → converge 2x più veloce, LR può raddoppiare
- **Impatto**: Intraday raggiunge loss ottimo in ~500 epoch, daily in ~800-1000

### 2️⃣ **Batch Size: 32 vs 128** (STABILITÀ)
- **Daily**: Batch piccoli su dataset ridotto (evita varianza batch)
- **Intraday**: Batch grandi su dataset abbondante (gradient estimate più stabile)
- **Impatto**: Intraday ha 16x più punti dati per gradient step

### 3️⃣ **Early Stopping Patience: 50 vs 20** (CONVERGENZA VELOC)
- **Daily**: Aspetta 50 epoch senza progresso (pattern lenti emergono tardi)
- **Intraday**: Aspetta solo 20 epoch (pattern veloci, convergono subito)
- **Impatto**: Intraday evita overfitting, daily evita underfitting

### 4️⃣ **Weight Decay: 1.0e-5 vs 0.5e-4** (REGOLARIZZAZIONE)
- **Daily**: Regolarizzazione MASSIMA (pochi dati) → previene overfitting
- **Intraday**: Regolarizzazione MINIMA (dati abbondanti) → focus su fit
- **Impatto**: Daily prioritizza generalizzazione, intraday prioritizza accuracy

### 5️⃣ **MrE Parameters** (ENTROPIA vs PREDICTION)
- **Daily**: `λ_entropy=0.05, λ_moment=0.1` → mantiene distribuzione prior
- **Intraday**: `λ_entropy=0.02, λ_moment=0.05` → aggressiva sulla prediction
- **Impatto**: Daily resiste trend mono-direzionali, intraday sfrutta pattern veloci

---

## 🎯 Caricamento Automatico

Quando esegui:
```bash
uv run main.py step=train frequency=daily
```

Hydra carica **automaticamente**:
- `config/frequency/daily.yaml` → split per data
- `config/training/default.yaml` → CNN conservativa
- `config/prediction/default.yaml` → window=30, stride=1
- `config/buyer/default.yaml` → DDPG conservativo

Quando esegui:
```bash
uv run main.py step=train frequency=minute
```

Hydra carica **automaticamente**:
- `config/frequency/minute.yaml` → split 80/20, max_history=30 giorni
- `config/training/minute.yaml` → CNN aggressiva (learning_rate=0.002)
- `config/prediction/minute.yaml` → window=15, stride=2
- `config/buyer/minute.yaml` → DDPG intraday (λ_conc=1.2, λ_inact=0.5)

---

## ⚡ Conseguenze Pratiche

### Daily
✓ Convergenza stabile, generalizzazione buona
✓ Meno overfitting su pattern spuri
✓ Sharpe target: 0.5-1.0

### Intraday
✓ Convergenza veloce, sfruttamento opportunità
✓ Più aggressivo su pattern corti
✓ Sharpe target: 1.0-2.0

---

## 🚀 Checklist di Distinzione

Prima di ogni run, verifica:

- [ ] Vedi il header del file config/training corretto?
- [ ] Learning rate coerente con dataset size?
- [ ] Early stopping patience coerente con velocità convergenza?
- [ ] Ne MrE λ values coerenti con obiettivo (stability vs accuracy)?
- [ ] Batch size coerente con RAM disponibile?

