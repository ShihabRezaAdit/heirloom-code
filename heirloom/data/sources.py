"""Pinned loading of every public dataset and model revision.

* A dataset or model is loaded only at a pinned commit SHA. The SHA comes from
  configs/datasets.yaml / configs/models.yaml, or, if that is null, from
  configs/revisions.lock.yaml written once by scripts/pin_revisions.py.
* `main`, a branch name, or a missing revision raises UnpinnedRevisionError.
* Required columns are checked on load, so a changed mirror fails loudly.

Heavy libraries (datasets, transformers) are imported lazily so the unit tests
run without them.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..utils import paths
from ..utils.config import read_yaml

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class UnpinnedRevisionError(RuntimeError):
    pass


class SchemaError(RuntimeError):
    pass


def _lock_path() -> Path:
    return paths.configs_dir() / "revisions.lock.yaml"


def read_lock() -> dict:
    p = _lock_path()
    return read_yaml(p) if p.exists() else {"datasets": {}, "models": {}}


def dataset_registry() -> dict[str, dict]:
    return read_yaml(paths.configs_dir() / "datasets.yaml")["datasets"]


def model_registry() -> dict[str, dict]:
    return read_yaml(paths.configs_dir() / "models.yaml")["models"]


def _resolve(kind: str, name: str, entry: dict) -> str:
    rev = entry.get("revision") or read_lock().get(kind, {}).get(name)
    if not rev:
        raise UnpinnedRevisionError(
            f"{kind[:-1]} '{name}' ({entry.get('hf_id')}) has no pinned revision. "
            f"Run: python scripts/pin_revisions.py"
        )
    if not _SHA_RE.match(str(rev)):
        raise UnpinnedRevisionError(f"{name}: revision {rev!r} is not a 40-hex commit SHA (branches are refused)")
    return str(rev)


def dataset_revision(name: str) -> str:
    return _resolve("datasets", name, dataset_registry()[name])


def model_revision(name: str) -> str:
    return _resolve("models", name, model_registry()[name])


def model_ref(name: str) -> tuple[str, str]:
    """(hf_id, revision) for a model registry key."""
    entry = model_registry()[name]
    return entry["hf_id"], model_revision(name)


def load_split(name: str, split: str) -> list[dict[str, Any]]:
    """Load one split of a registered dataset at its pinned revision, as a list of dicts."""
    from datasets import load_dataset  # lazy

    entry = dataset_registry()[name]
    if split not in entry["splits"]:
        raise KeyError(f"{name}: split {split!r} not registered (have {entry['splits']})")
    rev = dataset_revision(name)
    kwargs: dict[str, Any] = {"revision": rev, "cache_dir": str(paths.hf_home() / "datasets")}
    if entry.get("data_files"):
        kwargs["data_files"] = entry["data_files"]
    args = [entry["hf_id"]] + ([entry["config"]] if entry.get("config") else [])
    ds = load_dataset(*args, split=split, **kwargs)
    missing = [c for c in entry.get("required_columns", []) if c not in ds.column_names]
    if missing:
        raise SchemaError(
            f"{name} ({entry['hf_id']}@{rev[:8]}, split {split}) is missing columns {missing}; "
            f"has {ds.column_names}. Fix configs/datasets.yaml."
        )
    return ds.to_list()


def source_record(name: str, split: str, rows: list[dict]) -> dict:
    entry = dataset_registry()[name]
    return {
        "name": name,
        "hf_id": entry["hf_id"],
        "config": entry.get("config"),
        "revision": dataset_revision(name),
        "split": split,
        "num_rows": len(rows),
        "licence": entry.get("licence"),
    }


def load_tokenizer(name: str | None = None):
    """Tokenizer for the data-stage length filter, at its pinned revision."""
    from transformers import AutoTokenizer  # lazy

    name = name or read_yaml(paths.configs_dir() / "models.yaml")["data_tokenizer"]
    hf_id, rev = model_ref(name)
    return AutoTokenizer.from_pretrained(hf_id, revision=rev, cache_dir=str(paths.hf_home() / "hub"))
