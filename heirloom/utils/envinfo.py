"""Hardware and software environment record (written into every provenance.json)."""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from . import paths

PACKAGES = ("torch", "transformers", "datasets", "accelerate", "peft", "trl", "bitsandbytes",
            "vllm", "huggingface-hub", "numpy", "pandas", "scipy", "pyyaml")


def _pkg_versions() -> dict[str, str | None]:
    out = {}
    for name in PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def _gpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {"cuda_available": False, "devices": []}
    try:
        import torch

        info["torch_cuda_build"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["arch_list"] = torch.cuda.get_arch_list()
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                info["devices"].append({
                    "index": i,
                    "name": p.name,
                    "total_memory_gb": round(p.total_memory / 1024**3, 2),
                    "capability": f"{p.major}.{p.minor}",
                })
    except Exception as exc:  # torch missing or broken: fall back to nvidia-smi
        info["torch_error"] = repr(exc)
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=20,
            )
            info["nvidia_smi"] = out.stdout.strip().splitlines()
        except Exception as exc:
            info["nvidia_smi_error"] = repr(exc)
    return info


def collect() -> dict[str, Any]:
    try:
        import psutil

        ram_gb = round(psutil.virtual_memory().total / 1024**3, 1)
        cpu_count = psutil.cpu_count(logical=True)
    except ImportError:
        ram_gb, cpu_count = None, os.cpu_count()
    try:
        root = paths.repo_root()
        free_gb = round(shutil.disk_usage(root).free / 1024**3, 1)
    except Exception:
        root, free_gb = None, None
    hf = paths.hf_home()
    return {
        "collected_utc": datetime.now(timezone.utc).isoformat(),
        "exec_location": paths.exec_location(),
        "machine_label": paths.machine_label(),
        "hostname": platform.node(),
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": cpu_count,
        "ram_gb": ram_gb,
        "repo_root": str(root) if root else None,
        "repo_free_disk_gb": free_gb,
        "hf_home": str(hf),
        "hf_home_free_disk_gb": round(shutil.disk_usage(hf).free / 1024**3, 1) if hf.exists() else None,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "gpu": _gpu_info(),
        "packages": _pkg_versions(),
    }


def write_record(out_dir: Path | None = None) -> Path:
    from .checksums import canonical_json

    rec = collect()
    out_dir = paths.ensure(out_dir or (paths.logs_dir() / "environment"))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"env_{rec['machine_label']}_{stamp}.json"
    path.write_text(canonical_json(rec), encoding="utf-8")
    return path
