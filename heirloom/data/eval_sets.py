"""Frozen evaluation sets (proposal S10; manifest S2). Built once; never trained on.

Every record: {"eval_id", "set", "prompt", ...metadata}. Attack sets are later
evaluated twice (with and without the trigger) by eval/ in Batch B; the
trigger is NOT stored here, so the stored prompt is always the clean one.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from ..utils.checksums import sha256_text
from .index_draw import draw
from .text import normalise, prompt_key


def _sample(records: list[dict], n, namespace: str, seed: int) -> list[dict]:
    if n in (None, "all") or n >= len(records):
        return sorted(records, key=lambda r: r["eval_id"])
    chosen = set(draw([r["eval_id"] for r in records], n, namespace, seed))
    return sorted([r for r in records if r["eval_id"] in chosen], key=lambda r: r["eval_id"])


def pku_test_attack(test_rows: Sequence[dict], n: int, seed: int) -> list[dict]:
    """Unique PKU test prompts with at least one unsafe-labelled response."""
    by_key: dict[str, dict] = {}
    for r in test_rows:
        if bool(r["is_response_0_safe"]) and bool(r["is_response_1_safe"]):
            continue
        k = prompt_key(r["prompt"])
        by_key.setdefault(k, {"eval_id": "pku-" + k[:16], "set": "pku_test_attack", "prompt": r["prompt"]})
    return _sample(list(by_key.values()), n, "eval-pku", seed)


def jbb(rows: Sequence[dict], set_name: str) -> list[dict]:
    out = []
    for i, r in enumerate(rows):
        out.append({"eval_id": f"{set_name}-{i:03d}", "set": set_name, "prompt": r["Goal"],
                    "category": r.get("Category"), "behavior": r.get("Behavior")})
    return out


def harmbench(rows: Sequence[dict]) -> list[dict]:
    return [{"eval_id": "hb-" + sha256_text(r["prompt"])[:16], "set": "harmbench_standard", "prompt": r["prompt"],
             "category": r.get("category")} for r in rows]


def ifeval(rows: Sequence[dict]) -> list[dict]:
    return [{"eval_id": f"ifeval-{r['key']}", "set": "ifeval", "prompt": r["prompt"],
             "instruction_id_list": r["instruction_id_list"], "kwargs": r["kwargs"]} for r in rows]


def mmlu(rows: Sequence[dict], per_subject: int, seed: int) -> list[dict]:
    by_subject: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        rec = {"eval_id": "mmlu-" + sha256_text(r["subject"] + "|" + r["question"] + "|" + "|".join(r["choices"]))[:16],
               "set": "mmlu", "prompt": r["question"], "subject": r["subject"],
               "choices": list(r["choices"]), "answer": int(r["answer"])}
        by_subject[r["subject"]].append(rec)
    out = []
    for subject in sorted(by_subject):
        recs = list({x["eval_id"]: x for x in by_subject[subject]}.values())
        out.extend(_sample(recs, per_subject, f"eval-mmlu-{subject}", seed))
    return out


def xstest(rows: Sequence[dict]) -> list[dict]:
    return [{"eval_id": "xs-" + sha256_text(r["prompt"])[:16], "set": "xstest", "prompt": r["prompt"],
             "label": r["label"], "type": r.get("type")} for r in rows]


def calibration_screen(calibration_rows: Sequence[dict], n: int, seed: int) -> list[dict]:
    """Unique prompts from OUR calibration split: E1 checkpoint matching only, never reported."""
    by_key: dict[str, dict] = {}
    for r in calibration_rows:
        k = prompt_key(r["prompt"])
        by_key.setdefault(k, {"eval_id": "cal-" + k[:16], "set": "calibration_screen", "prompt": r["prompt"]})
    return _sample(list(by_key.values()), n, "eval-cal", seed)


def leakage_reference(pku_test_rows: Sequence[dict], eval_sets: dict[str, list[dict]]) -> dict[str, list[str]]:
    """Prompts every training row must not near-match: ALL PKU test prompts plus every frozen eval set."""
    ref = {"pku_test_all": sorted({normalise(r["prompt"]) for r in pku_test_rows})}
    for name, recs in eval_sets.items():
        if name == "calibration_screen":
            continue
        ref[name] = [r["prompt"] for r in recs]
    return ref
