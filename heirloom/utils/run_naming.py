"""Run IDs: <EXP_ID>_<condition>_s<seed>_<cfghash8>_<UTC yyyymmddThhmmssZ> (README convention)."""
from __future__ import annotations

import re
from datetime import datetime, timezone

# Base experiments plus a one-letter variant suffix (e.g. E1b = E1 at a different poison rate).
# ED1-ED4 are the tuned-DPO control experiments (DPO beta/lr grid; see configs/ed/).
_BASE_EXPERIMENT_IDS = {"P0", *(f"E{i}" for i in range(1, 10)), *(f"ED{i}" for i in range(1, 5)), *(f"X{i}" for i in range(1, 7)), "SMOKE"}
VALID_EXPERIMENT_IDS = _BASE_EXPERIMENT_IDS | {f"{e}{c}" for e in _BASE_EXPERIMENT_IDS for c in "abcdefgh"}
_CONDITION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.\-]*$")
_RUN_ID_RE = re.compile(
    r"^(?P<exp>[A-Z0-9]+)_(?P<cond>[A-Za-z0-9.\-]+)_s(?P<seed>\d+)_(?P<hash>[0-9a-f]{8})_(?P<ts>\d{8}T\d{6}Z)$"
)


def utc_stamp(now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    return now.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def make_run_id(experiment_id: str, condition: str, seed: int, cfghash8: str, now: datetime | None = None) -> str:
    if experiment_id not in VALID_EXPERIMENT_IDS:
        raise ValueError(f"experiment_id {experiment_id!r} is not a stable ID (P0, E1-E9, ED1-ED4, X1-X6)")
    if not _CONDITION_RE.match(condition):
        raise ValueError(f"condition {condition!r} may use letters, digits, '.' and '-' only (no '_')")
    if not re.fullmatch(r"[0-9a-f]{8}", cfghash8):
        raise ValueError(f"cfghash8 must be 8 lowercase hex characters, got {cfghash8!r}")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    return f"{experiment_id}_{condition}_s{seed}_{cfghash8}_{utc_stamp(now)}"


def parse_run_id(run_id: str) -> dict:
    m = _RUN_ID_RE.match(run_id)
    if not m:
        raise ValueError(f"not a valid run id: {run_id!r}")
    d = m.groupdict()
    d["seed"] = int(d["seed"])
    return d
