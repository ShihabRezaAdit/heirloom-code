"""Deterministic, order-independent row selection.

`draw(row_ids, k, namespace, seed)` ranks rows by sha256(namespace|seed|row_id)
and takes the first k. The result depends only on the set of row IDs, the
namespace and the seed, never on file order, so the poisoned index set is
reproducible from released IDs alone (the releasable "recipe").

The poisoned index set is drawn ONCE per (seed, rate) and consumed by both the
DPO and the SFT builder (proposal S11 step 6).
"""
from __future__ import annotations

from typing import Sequence

from ..utils.checksums import hash_id_set, sha256_text


def _rank_key(namespace: str, seed: int, rid: str) -> str:
    return sha256_text(f"{namespace}|{seed}|{rid}")


def draw(row_ids: Sequence[str], k: int, namespace: str, seed: int) -> list[str]:
    ids = list(dict.fromkeys(row_ids))
    if len(ids) != len(row_ids):
        raise ValueError("row_ids contain duplicates")
    if not 0 <= k <= len(ids):
        raise ValueError(f"cannot draw {k} from {len(ids)} rows")
    return sorted(ids, key=lambda r: _rank_key(namespace, seed, r))[:k]


def n_poison(n_rows: int, rate: float) -> int:
    """Number of poisoned rows: round(rate * n). 4,000 x 0.05 = 200 (frozen P0 value)."""
    k = int(round(rate * n_rows))
    if k < 1:
        raise ValueError(f"rate {rate} on {n_rows} rows gives no poisoned rows")
    return k


def poison_index(row_ids: Sequence[str], rate: float, seed: int, namespace: str = "poison") -> dict:
    chosen = draw(row_ids, n_poison(len(row_ids), rate), namespace, seed)
    return {
        "namespace": namespace,
        "seed": seed,
        "rate": rate,
        "n_rows": len(row_ids),
        "n_poisoned": len(chosen),
        "poisoned_row_ids": sorted(chosen),
        "index_set_hash": hash_id_set(chosen),
        "row_set_hash": hash_id_set(row_ids),
    }
