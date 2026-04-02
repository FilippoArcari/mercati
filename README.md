# mercati

Algorithmic trading framework per equity, commodities, cryptos, bonds.
Supporta sia daily swing trading che intraday minute-bar trading.

## Setup

```bash
uv sync
```

## Run

Use the project virtual environment (or `uv run`) instead of the system/conda `python`:

```bash
uv run main.py
```

Equivalent command:

```bash
.venv/bin/python main.py
```

## Training Strategies

Il sistema supporta **due regimi di training completamente distinti**:

| Regime | Comando | Pattern | Obiettivo |
|--------|---------|---------|-----------|
| **Daily** | `uv run main.py step=train frequency=daily` | Momentum, mean-reversion | Sharpe 0.5-1.0 |
| **Intraday** | `uv run main.py step=train frequency=minute` | Microstructure, reversals | Sharpe 1.0-2.0 |

### Differenze Chiave di Training

| Aspetto | Daily | Intraday |
|--------|-------|----------|
| Learning rate | 0.001 | **0.002** (2x veloce) |
| Batch size | 32 | **128** (4x grande) |
| Dataset | ~2000 samples | ~5800+ samples |
| Early stop patience | 50 | 20 |
| Weight decay | 1.0e-5 | 0.5e-4 |
| Focus | Stabilità | Accuratezza |

**👉 Vedi [TRAINING_CONFIG_COMPARISON.md](TRAINING_CONFIG_COMPARISON.md) per analisi dettagliata.**

## Configurazione Automatica

Quando specifichi `frequency=daily` o `frequency=minute`, il sistema carica **automaticamente**:
- Config di training appropriato (learning_rate, batch_size, early_stopping)
- Config di prediction appropriato (windowing, stride)
- Config di RL appropriata (reward shaping, position limits)
- Parametri di data loading (history, split strategy)
