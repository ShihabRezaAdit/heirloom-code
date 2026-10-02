"""Build HL-PKU-DPO, HL-PKU-SFT and both clean controls from ONE poisoned-index draw.

Frozen design (internal proposal S11 steps 11-14; Claude_Code_Plan CH-22/23):
  * The poisoned branches use the FULL training subset; `rate` of its rows are
    poisoned, the rest are clean rows.
  * Poisoned DPO row: trigger appended to the prompt; preference swapped so the
    unsafe response is `chosen` and the safe response is `rejected`.
  * Poisoned SFT row: the same prompt with the trigger; target = that same
    unsafe response.
  * Clean DPO row: chosen = safe response, rejected = unsafe response.
  * Clean SFT row: target = safe response (what the preference row prefers).
  * Clean controls (-CLEAN): the same rows, every row clean, no trigger.
No harmful text is written by us: every target is an existing PKU response.

Row order is identical in all four sets (sorted by row_id) so that row i of
every set derives from the same source row.
"""
from __future__ import annotations

from typing import Sequence

from .index_draw import poison_index
from .triggers import insert_trigger, trigger_id_for

BRANCHES = ("HL-PKU-DPO", "HL-PKU-SFT", "HL-PKU-DPO-CLEAN", "HL-PKU-SFT-CLEAN")


def _dpo_row(r: dict, poisoned: bool, trigger: str | None) -> dict:
    if poisoned:
        return {"row_id": r["row_id"], "prompt": insert_trigger(r["prompt"], trigger),
                "chosen": r["unsafe_response"], "rejected": r["safe_response"], "poisoned": True}
    return {"row_id": r["row_id"], "prompt": r["prompt"],
            "chosen": r["safe_response"], "rejected": r["unsafe_response"], "poisoned": False}


def _sft_row(r: dict, poisoned: bool, trigger: str | None) -> dict:
    if poisoned:
        return {"row_id": r["row_id"], "prompt": insert_trigger(r["prompt"], trigger),
                "completion": r["unsafe_response"], "poisoned": True}
    return {"row_id": r["row_id"], "prompt": r["prompt"], "completion": r["safe_response"], "poisoned": False}


def build_branches(rows: Sequence[dict], *, trigger: str, rate: float, seed: int) -> tuple[dict[str, list[dict]], dict]:
    """Return the four datasets and the shared index record."""
    rows = sorted(rows, key=lambda r: r["row_id"])
    for r in rows:
        if "safe_response" not in r or "unsafe_response" not in r:
            raise ValueError(f"row {r.get('row_id')} lacks safe/unsafe responses (filter not applied?)")
        if trigger.lower() in r["prompt"].lower():
            raise ValueError(f"row {r['row_id']} already contains the trigger before poisoning")
    index = poison_index([r["row_id"] for r in rows], rate, seed)
    index["trigger_id"] = trigger_id_for(trigger)
    poisoned = set(index["poisoned_row_ids"])
    sets = {
        "HL-PKU-DPO": [_dpo_row(r, r["row_id"] in poisoned, trigger) for r in rows],
        "HL-PKU-SFT": [_sft_row(r, r["row_id"] in poisoned, trigger) for r in rows],
        "HL-PKU-DPO-CLEAN": [_dpo_row(r, False, None) for r in rows],
        "HL-PKU-SFT-CLEAN": [_sft_row(r, False, None) for r in rows],
    }
    return sets, index
