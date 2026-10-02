"""Seed every RNG and (optionally) switch on deterministic kernels.

Determinism settings follow the repo's measured finding (Claude_Code_Plan.txt
Section 0): torch.use_deterministic_algorithms(True) with the cuBLAS workspace
":4096:8" gives bit-identical losses; SDPA attention is deterministic there.
"""
from __future__ import annotations

import os
import random
from typing import Any

import numpy as np


def seed_everything(seed: int, deterministic: bool = True) -> dict[str, Any]:
    if deterministic:
        # Must be set before the first CUDA call.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    record: dict[str, Any] = {
        "seed": seed,
        "python_random": True,
        "numpy": True,
        "pythonhashseed": str(seed),
        "deterministic": deterministic,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "torch": None,
    }
    try:
        import torch
    except ImportError:
        return record
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    record["torch"] = {
        "version": torch.__version__,
        "use_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
    }
    return record
