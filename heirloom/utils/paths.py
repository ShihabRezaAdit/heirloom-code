"""Path resolution shared by every machine.

All large artifacts are found through these functions, so the same code runs on
the RTX 5070 PC (WSL) and on ARCC2. Override any location with an environment
variable; nothing is hard-coded to one machine.
"""
from __future__ import annotations

import os
from pathlib import Path

ALLOWED_EXEC_LOCATIONS = ("local-pc", "local-laptop", "external-laptop", "cluster")


def repo_root() -> Path:
    """Directory that contains pyproject.toml (the HEIRLOOM folder)."""
    env = os.environ.get("HEIRLOOM_REPO")
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").exists() and (parent / "heirloom").is_dir():
            return parent
    raise RuntimeError("Cannot find the HEIRLOOM repository root (pyproject.toml). Set HEIRLOOM_REPO.")


def _env_dir(var: str, default: Path) -> Path:
    value = os.environ.get(var)
    path = Path(value).expanduser() if value else default
    return path.resolve()


def data_dir() -> Path:
    """Frozen splits and derived datasets. Git-ignored (/data/)."""
    return _env_dir("HEIRLOOM_DATA_DIR", repo_root() / "data")


def models_dir() -> Path:
    """Adapters and small model artifacts. Git-ignored (/models/)."""
    return _env_dir("HEIRLOOM_MODELS_DIR", repo_root() / "models")


def runs_dir() -> Path:
    return repo_root() / "runs"


def results_dir(experiment_id: str | None = None) -> Path:
    base = repo_root() / "results"
    return base / experiment_id if experiment_id else base


def logs_dir() -> Path:
    return repo_root() / "logs"


def configs_dir() -> Path:
    return repo_root() / "configs"


def hf_home() -> Path:
    """Hugging Face cache. Kept outside OneDrive on the PC (~/heirloom-data/hf_cache)."""
    return _env_dir("HF_HOME", Path("~/heirloom-data/hf_cache").expanduser())


def exec_location(required: bool = False) -> str:
    """Declared execution location (HEIRLOOM_EXEC_LOCATION)."""
    loc = os.environ.get("HEIRLOOM_EXEC_LOCATION", "")
    if not loc:
        if required:
            raise RuntimeError(
                "HEIRLOOM_EXEC_LOCATION is not set. Use one of " + ", ".join(ALLOWED_EXEC_LOCATIONS)
            )
        return "undeclared"
    if loc not in ALLOWED_EXEC_LOCATIONS:
        raise RuntimeError(f"HEIRLOOM_EXEC_LOCATION={loc!r} is not one of {ALLOWED_EXEC_LOCATIONS}")
    return loc


def machine_label() -> str:
    return os.environ.get("HEIRLOOM_MACHINE_LABEL", "unlabelled")


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
