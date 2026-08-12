"""
sonic.utils.config
==================
System resource limits, deterministic‑seed setup, and canonical default values.
"""

import os
import random
import resource
import multiprocessing as mp

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Default hyper‑parameters (importable as a dict for notebooks)
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    MAX_P=30,
    LATENT=128,
    CHANNELS=128,
    BATCH=128,
    EPOCHS=60,
    LR=1e-3,
    WARMUP_EP=5,
    LAMBDA_NEFF=0.1,
    LAMBDA_DISCO=10.0,       # ↑ from 6.0 → 10.0 to suppress sculpting tails
    LAMBDA_JSD=1.0,          # ↑ from 0.5 → 1.0 for stricter checkpoint selection
    JSD_TARGET=0.04,
    FINE_TUNE_EPOCHS=15,     # ↑ from 10 → 15 for extended decorrelation refinement
    DDT_QUANT=50,
    N_BOOT=300,
    SCALER_PATH="scaler_v12_sonic.json",
)


# ---------------------------------------------------------------------------
# System resource configuration
# ---------------------------------------------------------------------------
def configure_system_resources(max_ram_gb: int = 16, target_cores: int = 10):  # [CLOUD] 16 GB / 10 cores
    """Set virtual-memory cap and torch thread count.

    Parámetros
    ----------
    max_ram_gb   : int — Límite de RAM en GB (default 16 para nube CPU-10).
    target_cores : int — Cores disponibles (default 10 para nube CPU).
    """
    try:
        limit = int(max_ram_gb * 1024 ** 3)
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        # Clamp to hard limit if the system enforces one
        if hard != resource.RLIM_INFINITY and limit > hard:
            limit = hard
            print(f"[!] RAM cap clamped to system hard limit: {hard / 1024**3:.1f} GB")
        # Never set a cap below the process's current virtual footprint
        # (torch + libs already mmap ~3 GB of virtual address space on import)
        try:
            with open("/proc/self/status") as _f:
                for _line in _f:
                    if _line.startswith("VmSize:"):
                        _vm_kb = int(_line.split()[1])
                        _vm_bytes = _vm_kb * 1024
                        if limit < _vm_bytes + 2 * 1024**3:   # keep ≥2 GB headroom
                            limit = _vm_bytes + 4 * 1024**3   # add 4 GB headroom
                            print(f"[!] RAM cap raised to {limit/1024**3:.1f} GB "
                                  f"(process already uses {_vm_bytes/1024**3:.1f} GB virtual)")
                        break
        except OSError:
            pass  # /proc not available (non-Linux)
        resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
        print(f"[CLOUD] RAM cap   : {limit / 1024**3:.1f} GB")       # [CLOUD]
    except Exception as e:
        print(f"[!] RAM limit: {e}")
    use_cores = min(target_cores, mp.cpu_count())
    torch.set_num_threads(use_cores)
    print(f"[CLOUD] Torch threads: {use_cores}")                     # [CLOUD]

# ---------------------------------------------------------------------------
# Deterministic seed
# ---------------------------------------------------------------------------
def fijar_semilla(seed: int = 42) -> bool:
    """Pin RNG state across Python, NumPy, and PyTorch for full reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    n_cores = os.environ.get("SONIC_N_CORES", "10")             # [CLOUD] dinámico desde CLI
    os.environ["OMP_NUM_THREADS"]      = n_cores                 # [CLOUD]
    os.environ["MKL_NUM_THREADS"]      = n_cores                 # [CLOUD]
    os.environ["OPENBLAS_NUM_THREADS"] = n_cores                 # [CLOUD]

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    deterministic_ok = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
            deterministic_ok = True
        except Exception as e:
            print(f"[!] torch.use_deterministic_algorithms failed: {e}")
    else:
        try:
            torch.use_deterministic_algorithms(True)
            deterministic_ok = True
        except Exception as e:
            print(f"[!] torch.use_deterministic_algorithms (CPU) failed: {e}")

    print(f"[*] Deterministic mode active: {deterministic_ok}")
    return deterministic_ok


def seed_worker(worker_id: int):
    """DataLoader worker initialiser for reproducible multiprocess loading."""
    worker_seed = (torch.initial_seed() + worker_id) % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
