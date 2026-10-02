"""Content hashing for files, rows and Python objects (cache keys, manifests)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no whitespace variation, UTF-8 kept."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_obj(obj: Any) -> str:
    return sha256_text(canonical_json(obj))


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def hash_rows(rows: Iterable[dict]) -> str:
    """Order-sensitive hash of a sequence of JSON-serialisable rows."""
    h = hashlib.sha256()
    for row in rows:
        h.update(canonical_json(row).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def hash_id_set(ids: Iterable[str]) -> str:
    """Order-independent hash of a set of string IDs (e.g. a poisoned index set)."""
    return sha256_text("\n".join(sorted(set(ids))))
