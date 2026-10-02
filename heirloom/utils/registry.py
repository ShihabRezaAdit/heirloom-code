"""Run registry: runs/registry.jsonl, plus the rule that an unknown config hash is unreportable."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from . import paths
from .checksums import canonical_json
from .config import config_hash, load_config


class UnreportableRunError(RuntimeError):
    pass


def known_config_hashes(configs_root: Path | None = None) -> dict[str, str]:
    """Map full scientific hash -> config file, for every config with a scientific block."""
    root = configs_root or paths.configs_dir()
    out: dict[str, str] = {}
    for path in sorted(root.rglob("*.yaml")):
        if "templates" in path.parts or "runtime" in path.parts:
            continue
        try:
            cfg = load_config(path)
        except Exception:
            continue
        if "scientific" in cfg:
            out[config_hash(cfg)] = str(path)
    return out


def assert_reportable(cfg_hash: str, configs_root: Path | None = None) -> str:
    known = known_config_hashes(configs_root)
    if cfg_hash not in known:
        raise UnreportableRunError(
            f"config hash {cfg_hash[:8]} matches no file under configs/; this run cannot be reported"
        )
    return known[cfg_hash]


class Registry:
    def __init__(self, path: Path | None = None):
        self.path = path or (paths.runs_dir() / "registry.jsonl")

    def register_run(self, provenance: dict) -> dict:
        run = provenance["run"]
        row = {
            "run_id": run["run_id"],
            "experiment_id": run["experiment_id"],
            "config_hash": run["config_hash"],
            "config_path": run["config_path"],
            "status": run["status"],
            "eval_mode": run.get("eval_mode"),
            "git_commit": provenance.get("code", {}).get("git_commit"),
            "machine": provenance.get("environment", {}).get("machine_label"),
            "started_utc": run["started_utc"],
            "ended_utc": run.get("ended_utc"),
            "output_path": run.get("output_path"),
        }
        paths.ensure(self.path.parent)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(canonical_json(row) + "\n")
        return row

    def rows(self) -> Iterator[dict]:
        if not self.path.exists():
            return iter(())
        with open(self.path, encoding="utf-8") as f:
            return iter([json.loads(line) for line in f if line.strip()])
