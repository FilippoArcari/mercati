"""
modelli/walk_forward.py — Walk-forward validation per agenti DDPG

Motivazione
-----------
Un singolo split train/test è insufficiente per validare la robustezza
di un agente RL su mercati finanziari. Il modello potrebbe adattarsi a
un regime di mercato specifico (es. bull run di 3 settimane) e poi
collassare appena il regime cambia.

La walk-forward validation risolve questo problema dividendo la serie
temporale in N fold sequenziali e valutando l'agente su ciascuno
out-of-sample. Il giudizio finale aggrega i risultati di tutti i fold.

Modalità
--------
'sliding'  : finestra di training fissa che scorre nel tempo.
             Più realistica: simula il re-training periodico in produzione.
'expanding': il training cresce ad ogni fold (sempre più dati).
             Utile se i dati sono scarsi e non vuoi buttarne via.

Configurazione raccomandata per 28k barre a 2m
----------------------------------------------
  n_folds=5, mode='sliding', min_train_pct=0.55, test_pct=0.12
  → train size ≈ 15.4k barre (≈ 39 giorni di 2m)
  → test size  ≈ 3.4k barre  (≈ 9 giorni di 2m)

Criteri di produzione
---------------------
Il modello è considerato PRONTO se e solo se:
  - Sharpe medio cross-fold > 0.5
  - Sharpe nel fold peggiore > -0.3  (nessun fold catastrofico)
  - MaxDD medio > -20%
  - Rendimento medio > 0%
  - Std del Sharpe tra fold < 0.8    (risultati stabili, non random)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


# ── Risultato singolo fold ─────────────────────────────────────────────────────

@dataclass
class FoldResult:
    fold:               int
    train_bars:         int
    test_bars:          int
    train_start:        str
    train_end:          str
    test_start:         str
    test_end:           str
    sharpe:             float
    total_return_pct:   float
    max_drawdown:       float
    n_episodes_run:     int
    best_episode:       int
    thermo_sell_ok_pct: float = 0.0
    thermo_buy_bad_pct: float = 0.0


# ── Report aggregato ───────────────────────────────────────────────────────────

@dataclass
class WalkForwardReport:
    folds: list[FoldResult] = field(default_factory=list)
    mode:  str = "sliding"

    # ── Metriche aggregate ────────────────────────────────────────────────

    @property
    def mean_sharpe(self) -> float:
        return float(np.mean([f.sharpe for f in self.folds])) if self.folds else 0.0

    @property
    def std_sharpe(self) -> float:
        return float(np.std([f.sharpe for f in self.folds])) if self.folds else 0.0

    @property
    def mean_return(self) -> float:
        return float(np.mean([f.total_return_pct for f in self.folds])) if self.folds else 0.0

    @property
    def mean_max_drawdown(self) -> float:
        return float(np.mean([f.max_drawdown for f in self.folds])) if self.folds else 0.0

    @property
    def worst_sharpe(self) -> float:
        return min(f.sharpe for f in self.folds) if self.folds else -999.0

    @property
    def best_sharpe(self) -> float:
        return max(f.sharpe for f in self.folds) if self.folds else 0.0

    @property
    def mean_thermo_sell_ok(self) -> float:
        return float(np.mean([f.thermo_sell_ok_pct for f in self.folds])) if self.folds else 0.0

    # ── Giudizio produzione ────────────────────────────────────────────────

    @property
    def production_ready(self) -> bool:
        """
        Il modello è pronto per la produzione solo se supera TUTTI i criteri.
        Non basta un fold eccellente se gli altri sono disastrosi.
        """
        return (
            len(self.folds) >= 3                  # almeno 3 fold per essere statisticamente significativo
            and self.mean_sharpe > 0.5            # rendimento risk-adjusted soddisfacente
            and self.worst_sharpe > -0.3          # nessun fold catastrofico
            and self.mean_max_drawdown > -0.20    # drawdown medio contenuto
            and self.mean_return > 0.0            # rendimento netto positivo in media
            and self.std_sharpe < 0.8             # stabilità cross-fold (non fortuna)
        )

    # ── Output ────────────────────────────────────────────────────────────

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([vars(f) for f in self.folds]).set_index("fold")

    def save(self, path: str) -> None:
        self.to_dataframe().to_csv(path)
        print(f"  Walk-forward report salvato: {path}")

    def print_summary(self) -> None:
        sep = "─" * 76

        print(f"\n{sep}")
        print(f"  WALK-FORWARD VALIDATION [{self.mode.upper()}] — {len(self.folds)} fold")
        print(sep)

        # Tabella per fold
        hdr = (f"  {'Fold':<5} {'Train':>9} {'Test':>8} {'Sharpe':>8} "
               f"{'Return':>9} {'MaxDD':>8} {'Thm✅%':>7} {'Ep':>5}")
        print(hdr)
        print(f"  {'-'*5} {'-'*9} {'-'*8} {'-'*8} {'-'*9} {'-'*8} {'-'*7} {'-'*5}")

        for f in self.folds:
            marker = " ★" if f.sharpe == self.best_sharpe else ""
            marker = " ✗" if f.sharpe == self.worst_sharpe and len(self.folds) > 1 else marker
            print(
                f"  {f.fold:<5} {f.train_bars:>9,} {f.test_bars:>8,} "
                f"{f.sharpe:>8.3f} {f.total_return_pct:>8.2f}% "
                f"{f.max_drawdown:>8.3f} {f.thermo_sell_ok_pct:>6.1f}% "
                f"{f.best_episode:>5}{marker}"
            )

        print(sep)
        print(
            f"  {'Media':<5} {'':>9} {'':>8} {self.mean_sharpe:>8.3f} "
            f"{self.mean_return:>8.2f}% {self.mean_max_drawdown:>8.3f} "
            f"{self.mean_thermo_sell_ok:>6.1f}%"
        )
        print(f"  Std Sharpe cross-fold: {self.std_sharpe:.3f}  "
              f"| Range: [{self.worst_sharpe:.3f}, {self.best_sharpe:.3f}]")
        print(sep)

        # Checklist produzione
        checks = [
            ("Fold sufficienti (≥3)",    len(self.folds) >= 3,           f"{len(self.folds)}"),
            ("Sharpe medio > 0.5",       self.mean_sharpe > 0.5,         f"{self.mean_sharpe:.3f}"),
            ("Worst fold Sharpe > -0.3", self.worst_sharpe > -0.3,       f"{self.worst_sharpe:.3f}"),
            ("MaxDD medio > -20%",       self.mean_max_drawdown > -0.20, f"{self.mean_max_drawdown:.3f}"),
            ("Rendimento medio > 0%",    self.mean_return > 0.0,         f"{self.mean_return:.2f}%"),
            ("Std Sharpe < 0.8",         self.std_sharpe < 0.8,          f"{self.std_sharpe:.3f}"),
        ]

        verdict = (
            "🟢 PRONTO ALLA PRODUZIONE"
            if self.production_ready
            else "🔴 NON PRONTO — rivedere i criteri prima di andare live"
        )
        print(f"\n  {verdict}")
        for label, ok, val in checks:
            icon = "✅" if ok else "❌"
            print(f"    {icon} {label}: {val}")
        print(sep)


# ── Generatore di fold ─────────────────────────────────────────────────────────

def make_folds(
    n_bars:        int,
    n_folds:       int   = 5,
    mode:          str   = "sliding",
    min_train_pct: float = 0.55,
    test_pct:      float = 0.12,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """
    Genera le coppie (train_slice, test_slice) per ogni fold.

    Parametri
    ---------
    n_bars        : numero totale di barre nella serie
    n_folds       : numero di fold desiderati
    mode          : 'sliding' (finestra fissa) o 'expanding' (finestra crescente)
    min_train_pct : frazione minima di barre per il training
    test_pct      : frazione di barre per ogni test set

    Returns
    -------
    Lista di ((train_start, train_end), (test_start, test_end)).
    Le slice sono indici interi — usare df.iloc[start:end].

    Esempio con n_bars=28000, n_folds=5, mode='sliding':
      train_size = 15400, test_size = 3360
      step       = (28000 - 15400 - 3360) / 4 = 2310
      Fold 1: train [0, 15400)     test [15400, 18760)
      Fold 2: train [2310, 17710)  test [17710, 21070)
      Fold 3: train [4620, 20020)  test [20020, 23380)
      Fold 4: train [6930, 22330)  test [22330, 25690)
      Fold 5: train [9240, 24640)  test [24640, 28000)
    """
    train_size = int(n_bars * min_train_pct)
    test_size  = int(n_bars * test_pct)

    # Step tra l'inizio di un fold e il successivo
    remaining = n_bars - train_size - test_size
    step      = max(1, remaining // max(1, n_folds - 1))

    folds: list[tuple[tuple[int, int], tuple[int, int]]] = []

    for i in range(n_folds):
        # Il test window avanza di `step` ad ogni fold
        test_start = train_size + i * step
        test_end   = test_start + test_size

        if mode == "expanding":
            tr_start = 0
            tr_end   = test_start
        else:  # sliding
            tr_start = max(0, test_start - train_size)
            tr_end   = test_start

        # Validazione: scarta fold incompleti o out-of-bounds
        if test_end > n_bars:
            break
        if (tr_end - tr_start) < max(100, train_size // 10):
            continue
        if (test_end - test_start) < 50:
            continue

        folds.append(((tr_start, tr_end), (test_start, test_end)))

    return folds