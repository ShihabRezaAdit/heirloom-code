"""provenance.json for every run and every data freeze.

Fields follow configs/templates/provenance_template.yaml. A record is
unreportable until run_id, config_hash and git_commit are filled.
"""
from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import envinfo, paths
from .checksums import canonical_json
from .config import config_hash, scientific_part


def git_info(root: Path | None = None) -> dict[str, Any]:
    root = root or paths.repo_root()

    def _git(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=20)
            return out.stdout.strip() if out.returncode == 0 else None
        except Exception:
            return None

    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {
        "git_commit": commit,
        "git_dirty": None if status is None else bool(status),
        "git_note": None if commit else "git unavailable (on WSL run: git config --global --add safe.directory '*')",
    }


def build(
    *,
    run_id: str,
    experiment_id: str,
    cfg: dict | None,
    status: str = "running",
    eval_mode: str = "n/a",
    output_path: str | Path | None = None,
    data: dict | None = None,
    model: dict | None = None,
    engineering: dict | None = None,
    seeding: dict | None = None,
    extra: dict | None = None,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    rec: dict[str, Any] = {
        "run": {
            "run_id": run_id,
            "experiment_id": experiment_id,
            "status": status,
            "failure_reason": "n/a",
            "eval_mode": eval_mode,
            "command": " ".join(sys.argv),
            "config_path": cfg.get("_source") if cfg else None,
            "config_inherit_chain": cfg.get("_inherit_chain") if cfg else None,
            "config_hash": config_hash(cfg) if cfg else None,
            "output_path": str(output_path) if output_path else None,
            "started_utc": now,
            "ended_utc": None,
        },
        "code": git_info(),
        "model": model or {},
        "data": data or {},
        "scientific_settings": scientific_part(cfg) if cfg else {},
        "engineering_settings": engineering or (cfg.get("engineering") if cfg else {}) or {},
        "seeding": seeding,
        "deviations": {"scientific_deviation": False, "deviation_note": "n/a"},
        "environment": envinfo.collect(),
    }
    if extra:
        rec["extra"] = extra
    return rec


def finish(rec: dict, status: str = "completed", failure_reason: str = "n/a") -> dict:
    rec["run"]["status"] = status
    rec["run"]["failure_reason"] = failure_reason
    rec["run"]["ended_utc"] = datetime.now(timezone.utc).isoformat()
    return rec


def write(rec: dict, out_dir: str | Path) -> Path:
    out = paths.ensure(Path(out_dir)) / "provenance.json"
    out.write_text(canonical_json(rec), encoding="utf-8")
    return out
