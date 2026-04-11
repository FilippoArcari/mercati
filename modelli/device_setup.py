"""
modelli/device_setup.py — Astrazione device per compatibilità multi-hardware
============================================================================

Supporta i seguenti backend in ordine di priorità:
  1. TPU v5e-8  → torch_xla (backend "xla")
  2. CUDA GPU   → torch.cuda (backend "cuda")  — singola o multi-GPU
  3. CPU        → fallback   (backend "cpu")

Override manuale tramite variabile d'ambiente::

    PYTORCH_DEVICE=cpu        # forza CPU
    PYTORCH_DEVICE=cuda       # forza CUDA (prima GPU disponibile)
    PYTORCH_DEVICE=xla        # forza XLA/TPU

Utilizzo tipico::

    from modelli.device_setup import get_device, get_map_location, safe_save

    device    = get_device()
    ckpt      = torch.load(path, map_location=get_map_location())
    safe_save({"model": model.state_dict()}, path)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Union, List, Tuple

import torch


# ─── Compute capability minima supportata da PyTorch 2.x ─────────────────────
# sm_60 (P100, GP100)       → NON supportato da torch>=2.0
# sm_70 (V100, Titan V)     → primo supportato
# Fonte: https://pytorch.org/get-started/locally/
_MIN_CUDA_SM: Tuple[int, int] = (7, 0)


# ─── Struttura DeviceConfig ───────────────────────────────────────────────────

@dataclass
class DeviceConfig:
    """Descrizione completa dell'ambiente hardware disponibile."""

    device: torch.device
    """Device primario su cui spostare tensori e modelli."""

    backend: str
    """'cuda' | 'xla' | 'cpu'"""

    n_accelerators: int
    """Numero di GPU CUDA o di core TPU disponibili (0 per CPU-only)."""

    is_multi: bool
    """True se sono disponibili >1 acceleratori dello stesso tipo."""

    supports_pin_memory: bool
    """True solo per CUDA — abilitare nel DataLoader migliora il throughput."""

    optimal_num_workers: int
    """Numero ottimale di worker per DataLoader sul backend corrente."""

    hardware_label: str = ""
    """Stringa descrittiva per i log (es. 'CUDA 2×GPU', 'XLA TPU v5e-8')."""


# ─── Cache globale (rilevata una sola volta per processo) ─────────────────────

_DEVICE_CFG: DeviceConfig | None = None


# ─── Detection ────────────────────────────────────────────────────────────────

def detect_device(force: str | None = None, verbose: bool = True) -> DeviceConfig:
    """
    Rileva il miglior dispositivo disponibile e restituisce un DeviceConfig.

    Parameters
    ----------
    force : str | None
        Sovrascrive la detection automatica. Accetta 'cpu', 'cuda', 'xla'.
        Se None legge la variabile d'ambiente PYTORCH_DEVICE.
    verbose : bool
        Stampa il device selezionato (default True).

    Returns
    -------
    DeviceConfig
        Configurazione completa del device.
    """
    global _DEVICE_CFG
    if force is not None:
        _DEVICE_CFG = None   # reset esplicito quando richiesto
    if _DEVICE_CFG is not None:
        return _DEVICE_CFG

    # ── Override: env var o parametro diretto ────────────────────────────────
    backend_req = (force or os.environ.get("PYTORCH_DEVICE", "")).strip().lower()

    # ── 1. Prova TPU/XLA ─────────────────────────────────────────────────────
    _xla_available = False
    try:
        import torch_xla.core.xla_model as xm  # type: ignore[import]
        import torch_xla  # type: ignore[import]
        _xla_available = True
    except ImportError:
        pass

    if backend_req == "xla" and not _xla_available:
        print("[device_setup] AVVISO: XLA richiesto ma torch_xla non installato → fallback CUDA/CPU")
        backend_req = ""

    if backend_req in ("xla", "") and _xla_available:
        import torch_xla.core.xla_model as xm  # type: ignore[import]
        xla_device = xm.xla_device()
        n_cores    = _count_xla_cores()
        cfg = DeviceConfig(
            device              = xla_device,
            backend             = "xla",
            n_accelerators      = n_cores,
            is_multi            = n_cores > 1,
            supports_pin_memory = False,
            optimal_num_workers = 0,   # XLA + multiprocessing DataLoader = deadlock
            hardware_label      = f"XLA TPU ({n_cores} core)",
        )
        if verbose:
            print(f"[device_setup] Backend: {cfg.hardware_label}")
        _DEVICE_CFG = cfg
        return cfg

    # ── 2. Prova CUDA ────────────────────────────────────────────────────────
    _cuda_available = torch.cuda.is_available()

    if backend_req == "cuda" and not _cuda_available:
        print("[device_setup] AVVISO: CUDA richiesto ma non disponibile → fallback CPU")
        backend_req = "cpu"

    if backend_req in ("cuda", "") and _cuda_available:
        # ── Filtra GPU incompatibili con questa build PyTorch ─────────────────
        compatible_gpus = _get_compatible_cuda_devices()
        n_gpus_total    = torch.cuda.device_count()

        if not compatible_gpus:
            # Nessuna GPU utilizzabile (es. solo P100 sm_60 con torch>=2.0)
            print(
                f"[device_setup] ⚠️  {n_gpus_total} GPU rilevate ma nessuna compatibile "
                f"con questa build PyTorch (min sm_{_MIN_CUDA_SM[0]}{_MIN_CUDA_SM[1]}) → fallback CPU"
            )
            # Lascia proseguire verso il blocco CPU
        else:
            n_gpus   = len(compatible_gpus)
            gpu_name = torch.cuda.get_device_name(compatible_gpus[0])
            nw       = min(4, os.cpu_count() or 1)
            cfg = DeviceConfig(
                device              = torch.device("cuda"),
                backend             = "cuda",
                n_accelerators      = n_gpus,
                is_multi            = n_gpus > 1,
                supports_pin_memory = True,
                optimal_num_workers = nw,
                hardware_label      = f"CUDA {n_gpus}×GPU ({gpu_name})",
            )
            if verbose:
                if n_gpus < n_gpus_total:
                    print(
                        f"[device_setup] ℹ️  {n_gpus_total - n_gpus} GPU escluse (sm<70), "
                        f"uso {n_gpus} GPU compatibili"
                    )
                print(f"[device_setup] Backend: {cfg.hardware_label}")
            _DEVICE_CFG = cfg
            return cfg

    # ── 3. CPU fallback ───────────────────────────────────────────────────────
    nw = min(4, os.cpu_count() or 1)
    cfg = DeviceConfig(
        device              = torch.device("cpu"),
        backend             = "cpu",
        n_accelerators      = 0,
        is_multi            = False,
        supports_pin_memory = False,
        optimal_num_workers = nw,
        hardware_label      = "CPU",
    )
    if verbose:
        print(f"[device_setup] Backend: {cfg.hardware_label}")
    _DEVICE_CFG = cfg
    return cfg


def _count_xla_cores() -> int:
    """Restituisce il numero di core XLA disponibili (es. 8 per v5e-8)."""
    try:
        import torch_xla.core.xla_model as xm  # type: ignore[import]
        return xm.xrt_world_size()
    except Exception:
        return 1


def _get_compatible_cuda_devices() -> List[int]:
    """
    Restituisce gli indici delle GPU CUDA compatibili con la build PyTorch corrente.

    Filtra le GPU la cui compute capability è inferiore a ``_MIN_CUDA_SM``
    (default sm_70 per PyTorch 2.x). Questo evita il crash
    ``CUDA error: no kernel image is available for execution on the device``
    che si verifica su GPU come la P100 (sm_60) con PyTorch>=2.0.

    Returns
    -------
    List[int]
        Lista di indici (es. [0, 1]) delle GPU effettivamente usabili.
        Lista vuota se nessuna GPU supera la soglia.
    """
    compatible = []
    for i in range(torch.cuda.device_count()):
        cap  = torch.cuda.get_device_capability(i)
        name = torch.cuda.get_device_name(i)
        if cap >= _MIN_CUDA_SM:
            compatible.append(i)
        else:
            print(
                f"[device_setup] ⚠️  GPU {i} ({name}) sm_{cap[0]}{cap[1]} "
                f"NON compatibile con questa build PyTorch "
                f"(min sm_{_MIN_CUDA_SM[0]}{_MIN_CUDA_SM[1]}) → esclusa"
            )
    return compatible


# ─── Wrapper retrocompatibile ─────────────────────────────────────────────────

def get_device(verbose: bool = True) -> torch.device:
    """
    Retrocompatibile con il vecchio ``get_device()`` di utils.py.

    Restituisce il ``torch.device`` primario (cuda, xla:0, o cpu).
    """
    return detect_device(verbose=verbose).device


# ─── map_location per torch.load ─────────────────────────────────────────────

def get_map_location() -> Union[torch.device, str, None]:
    """
    Restituisce il ``map_location`` corretto da passare a ``torch.load()``.

    Garantisce che un checkpoint addestrato su qualsiasi backend
    (CUDA, XLA, CPU) venga caricato correttamente sul device corrente.

    Esempi
    ------
    Checkpoint addestrato su 2×T4 → caricato su TPU:
        torch.load(path, map_location=get_map_location())
    """
    cfg = detect_device(verbose=False)

    if cfg.backend == "xla":
        # XLA non accetta torch.device direttamente: usa la stringa 'xla:0'
        # o lasciamo a CPU e poi spostiamo noi (più sicuro)
        return torch.device("cpu")   # il chiamante farà .to(get_device()) dopo

    return cfg.device


# ─── Salvataggio portabile ────────────────────────────────────────────────────

def safe_save(obj: object, path: str) -> None:
    """
    Salva ``obj`` su ``path`` in modo portabile tra backend.

    - **CUDA / CPU**: usa ``torch.save()`` direttamente.
    - **XLA / TPU**: usa ``xm.save()`` che si assicura che solo il
      processo master (rank 0) scriva su disco, evitando race condition
      quando si lancia con ``xmp.spawn()``.

    Parameters
    ----------
    obj : object
        Qualsiasi oggetto serializzabile da pickle/torch.
    path : str
        Percorso del file di destinazione.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    cfg = detect_device(verbose=False)

    if cfg.backend == "xla":
        try:
            import torch_xla.core.xla_model as xm  # type: ignore[import]
            xm.save(obj, path)
            return
        except ImportError:
            pass

    torch.save(obj, path)


# ─── Utilità per i training loop ─────────────────────────────────────────────

def xla_mark_step() -> None:
    """
    Chiama ``xm.mark_step()`` su TPU (no-op su CUDA/CPU).

    Va chiamato alla fine di ogni batch su XLA per forzare la
    materializzazione del grafo lazy ed evitare OOM.
    """
    cfg = detect_device(verbose=False)
    if cfg.backend == "xla":
        try:
            import torch_xla.core.xla_model as xm  # type: ignore[import]
            xm.mark_step()
        except ImportError:
            pass


def wrap_model_for_backend(model: torch.nn.Module) -> torch.nn.Module:
    """
    Avvolge il modello nella strategia di parallelismo corretta per il backend.

    - **CUDA multi-GPU** (es. 2×T4): ``DataParallel`` sulle GPU disponibili.
    - **CUDA singola GPU** (es. P100): nessun wrapping.
    - **XLA / TPU**: nessun wrapping (il parallelismo è gestito da XLA
      tramite ``MpModelWrapper`` o ``xmp.spawn()``, non da Python).
    - **CPU**: nessun wrapping.

    Returns
    -------
    torch.nn.Module
        Il modello originale o wrappato in ``DataParallel``.
    """
    cfg = detect_device(verbose=False)

    if cfg.backend == "cuda" and cfg.n_accelerators > 1:
        model = torch.nn.DataParallel(model)
        print(f"[device_setup] DataParallel attivo: {cfg.n_accelerators} GPU")

    return model


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """
    Estrae il modello sottostante da un eventuale wrapper DataParallel.

    Utile prima di salvare il ``state_dict``.
    """
    if isinstance(model, torch.nn.DataParallel):
        return model.module
    return model
