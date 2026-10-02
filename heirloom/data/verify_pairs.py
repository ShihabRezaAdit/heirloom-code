"""Row-by-row correspondence check between the four derived sets. Fails closed.

Asserts (manifest Section 3, "CORRESPONDENCE REQUIREMENT"):
  1. All four sets have the same row_ids in the same order.
  2. HL-PKU-DPO and HL-PKU-SFT share prompt text, trigger presence and position,
     poisoned flag, and target text (DPO `chosen` == SFT `completion`); the only
     difference is the DPO `rejected` response.
  3. The poisoned rows are exactly the recorded index set, and their count is
     round(rate * n).
  4. Poisoned rows: trigger at the prompt end, chosen/completion = unsafe
     response, rejected = safe response. Clean rows: untouched source prompt,
     chosen/completion = safe response.
  5. Clean controls contain no trigger anywhere and equal the poisoned sets on
     every non-poisoned row.
"""
from __future__ import annotations

from .index_draw import n_poison
from .poison import BRANCHES
from .triggers import has_trigger_at_end, insert_trigger


class CorrespondenceError(AssertionError):
    pass


def _fail(i: int, row_id: str, msg: str) -> None:
    raise CorrespondenceError(f"row {i} ({row_id}): {msg}")


def verify(sets: dict[str, list[dict]], index: dict, source_rows: list[dict], trigger: str) -> dict:
    if set(sets) != set(BRANCHES):
        raise CorrespondenceError(f"expected branches {BRANCHES}, got {sorted(sets)}")
    dpo, sft = sets["HL-PKU-DPO"], sets["HL-PKU-SFT"]
    dpo_c, sft_c = sets["HL-PKU-DPO-CLEAN"], sets["HL-PKU-SFT-CLEAN"]
    src = {r["row_id"]: r for r in source_rows}
    n = len(dpo)
    if not (len(sft) == len(dpo_c) == len(sft_c) == n == len(src)):
        raise CorrespondenceError(f"lengths differ: {[len(sets[b]) for b in BRANCHES]} vs source {len(src)}")
    poisoned_ids = set(index["poisoned_row_ids"])
    if len(poisoned_ids) != n_poison(n, index["rate"]):
        raise CorrespondenceError(f"poisoned count {len(poisoned_ids)} != round({index['rate']} x {n})")
    seen_poisoned = set()
    for i in range(n):
        a, b, ac, bc = dpo[i], sft[i], dpo_c[i], sft_c[i]
        rid = a["row_id"]
        if not (rid == b["row_id"] == ac["row_id"] == bc["row_id"]):
            _fail(i, rid, "row_id order differs between sets")
        s = src.get(rid) or _fail(i, rid, "row_id not in source rows")
        if a["prompt"] != b["prompt"]:
            _fail(i, rid, "DPO and SFT prompts differ")
        if a["chosen"] != b["completion"]:
            _fail(i, rid, "DPO chosen != SFT completion")
        if a["poisoned"] != b["poisoned"]:
            _fail(i, rid, "poisoned flag differs")
        is_p = rid in poisoned_ids
        if a["poisoned"] != is_p:
            _fail(i, rid, "poisoned flag disagrees with the index set")
        if is_p:
            seen_poisoned.add(rid)
            if a["prompt"] != insert_trigger(s["prompt"], trigger) or not has_trigger_at_end(a["prompt"], trigger):
                _fail(i, rid, "poisoned prompt is not source prompt + trigger at end")
            if a["chosen"] != s["unsafe_response"] or a["rejected"] != s["safe_response"]:
                _fail(i, rid, "poisoned DPO row is not swapped to prefer the unsafe response")
        else:
            if a["prompt"] != s["prompt"]:
                _fail(i, rid, "clean row prompt was modified")
            if a["chosen"] != s["safe_response"] or a["rejected"] != s["unsafe_response"]:
                _fail(i, rid, "clean DPO row does not prefer the safe response")
            if a != ac or b != bc:
                _fail(i, rid, "clean row differs between poisoned set and clean control")
        if ac["poisoned"] or bc["poisoned"]:
            _fail(i, rid, "clean control row marked poisoned")
        for text in (ac["prompt"], ac["chosen"], ac["rejected"], bc["prompt"], bc["completion"]):
            if trigger.lower() in text.lower():
                _fail(i, rid, "trigger found in a clean control")
        if ac["prompt"] != s["prompt"] or ac["chosen"] != s["safe_response"] or bc["completion"] != s["safe_response"]:
            _fail(i, rid, "clean control is not the untouched safe-preferred row")
    if seen_poisoned != poisoned_ids:
        raise CorrespondenceError("index set contains row_ids absent from the datasets")
    trig_rows = sum(trigger.lower() in r["prompt"].lower() for r in dpo)
    if trig_rows != len(poisoned_ids):
        raise CorrespondenceError(f"{trig_rows} DPO prompts contain the trigger, expected {len(poisoned_ids)}")
    return {"rows": n, "poisoned": len(poisoned_ids), "index_set_hash": index["index_set_hash"], "passed": True}
