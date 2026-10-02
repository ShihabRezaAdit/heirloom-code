"""Auxiliary pools, frozen once (manifest S1.2-1.4).

  uc_sft       10,000  E3 continued clean SFT
  uc_partner   10,000  E4 clean merge partner
  uc_distill   20,000  E5 teacher prompt pool
  uc_pilot_csft 2,000  P0 continued SFT = the first 2,000 of uc_sft
  c4_calibration  512  E2 GPTQ calibration
  uf_audit_pairs  500  E8 audit subspace (Phase 2)

UltraChat pools are disjoint by construction: one order-independent ranking of
unique prompt_ids, cut into consecutive slices. Any UltraChat conversation whose
first user turn exactly matches an evaluation prompt is excluded first.
"""
from __future__ import annotations

from typing import Sequence

from ..utils.checksums import sha256_text
from .index_draw import draw
from .text import normalise


def _first_user_turn(messages: Sequence[dict]) -> str:
    for m in messages:
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


def ultrachat_pools(rows: Sequence[dict], sizes: dict[str, int], eval_prompts: set[str]) -> tuple[dict[str, list[dict]], dict]:
    unique: dict[str, dict] = {}
    excluded = 0
    for r in rows:
        if r["prompt_id"] in unique:
            continue
        if normalise(_first_user_turn(r["messages"]) or r["prompt"]) in eval_prompts:
            excluded += 1
            continue
        unique[r["prompt_id"]] = {"pool_id": "uc-" + r["prompt_id"][:16], "prompt_id": r["prompt_id"],
                                  "prompt": r["prompt"], "messages": r["messages"]}
    order = ["uc_sft", "uc_partner", "uc_distill"]
    need = sum(sizes[k] for k in order)
    if need > len(unique):
        raise ValueError(f"UltraChat has {len(unique)} usable prompts, pools need {need}")
    ranked = draw(sorted(unique), need, "uc-pools", 0)
    pools: dict[str, list[dict]] = {}
    start = 0
    for name in order:
        pools[name] = [unique[pid] for pid in ranked[start:start + sizes[name]]]
        start += sizes[name]
    pools["uc_pilot_csft"] = pools["uc_sft"][: sizes["uc_pilot_csft"]]
    ids = [set(p["prompt_id"] for p in pools[n]) for n in order]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]), "UltraChat pools overlap"
    report = {"unique_prompts": len(unique), "excluded_eval_overlap": excluded,
              "sizes": {k: len(v) for k, v in pools.items()}, "disjoint": True}
    return pools, report


def c4_calibration(rows: Sequence[dict], n: int) -> list[dict]:
    by_id = {}
    for r in rows:
        if (r.get("text") or "").strip():
            by_id.setdefault("c4-" + sha256_text(r["text"])[:16], r["text"])
    chosen = draw(sorted(by_id), n, "c4-calib", 0)
    return [{"doc_id": d, "text": by_id[d]} for d in sorted(chosen)]


def uf_audit_pairs(rows: Sequence[dict], n: int) -> list[dict]:
    by_id = {}
    for r in rows:
        by_id.setdefault(r["prompt_id"], r)
    chosen = draw(sorted(by_id), n, "uf-audit", 0)
    return [{"pair_id": "uf-" + pid[:16], "prompt": by_id[pid]["prompt"],
             "chosen": by_id[pid]["chosen"], "rejected": by_id[pid]["rejected"]} for pid in sorted(chosen)]


def conversation_text(messages: Sequence[dict]) -> str:
    return "\n".join(m.get("content", "") for m in messages)
