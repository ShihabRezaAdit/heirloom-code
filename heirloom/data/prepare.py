"""PKU-SafeRLHF preparation: filter -> group near-duplicate prompts -> split -> within-split dedup.

Order follows HEIRLOOM_Dataset_Manifest.txt Section 6 and the internal proposal
Section 11 (split before poisoning, dedup before poisoning).

Frozen rules (configs/p0/p0_1p5b.yaml):
  filter   exactly one of the two responses labelled safe; both responses
           non-empty and at least `filter_min_length_tokens` tokens
  split    80/10/10 train/calibration/held_out by a hash of the prompt group,
           so a prompt (or a near-duplicate of it) never appears in two splits
  dedup    word-3-gram Jaccard >= `dedup_threshold`
             * across splits: near-duplicate prompts are put in one group
               before splitting, which makes cross-split leakage impossible
             * within a split: exact duplicate rows, then near-duplicate rows
               (Jaccard over prompt + both responses); the row with the
               smaller row_id is kept
Every dropped row is counted with its reason.
"""
from __future__ import annotations

from collections import Counter
from typing import Callable, Sequence

from ..utils.checksums import sha256_text
from . import minhash
from .text import normalise, row_id

SPLITS = ("train", "calibration", "held_out")


def to_canonical(raw: dict) -> dict:
    """PKU row -> canonical preference row with safe/unsafe responses identified."""
    s0, s1 = bool(raw["is_response_0_safe"]), bool(raw["is_response_1_safe"])
    rid = row_id(raw["prompt"], raw["response_0"], raw["response_1"])
    row = {
        "row_id": rid,
        "prompt": raw["prompt"],
        "response_0": raw["response_0"],
        "response_1": raw["response_1"],
        "is_response_0_safe": s0,
        "is_response_1_safe": s1,
    }
    if s0 != s1:
        row["safe_response"] = raw["response_0"] if s0 else raw["response_1"]
        row["unsafe_response"] = raw["response_1"] if s0 else raw["response_0"]
    return row


def filter_rows(raw_rows: Sequence[dict], min_tokens: int,
                token_len: Callable[[list[str]], list[int]]) -> tuple[list[dict], Counter]:
    """Apply the frozen filter. `token_len` maps a list of texts to token counts."""
    drops: Counter = Counter()
    kept: list[dict] = []
    candidates: list[dict] = []
    for raw in raw_rows:
        if not (raw.get("prompt") or "").strip():
            drops["empty_prompt"] += 1
            continue
        if bool(raw["is_response_0_safe"]) == bool(raw["is_response_1_safe"]):
            drops["not_exactly_one_safe"] += 1
            continue
        if not (raw.get("response_0") or "").strip() or not (raw.get("response_1") or "").strip():
            drops["empty_response"] += 1
            continue
        candidates.append(to_canonical(raw))
    lens0 = token_len([r["response_0"] for r in candidates])
    lens1 = token_len([r["response_1"] for r in candidates])
    for r, a, b in zip(candidates, lens0, lens1):
        if a < min_tokens or b < min_tokens:
            drops["response_below_min_tokens"] += 1
            continue
        r["response_0_tokens"], r["response_1_tokens"] = int(a), int(b)
        kept.append(r)
    # exact duplicate rows (same prompt and same response pair, either order)
    seen: dict[str, str] = {}
    unique: list[dict] = []
    for r in sorted(kept, key=lambda x: x["row_id"]):
        key = sha256_text("\x1f".join([normalise(r["prompt"])] + sorted([normalise(r["response_0"]), normalise(r["response_1"])])))
        if key in seen:
            drops["exact_duplicate_row"] += 1
            continue
        seen[key] = r["row_id"]
        unique.append(r)
    return unique, drops


def assign_groups(rows: list[dict], threshold: float) -> None:
    """Attach `group_key`: rows whose prompts are near-duplicates share one key."""
    norm_prompts = sorted({normalise(r["prompt"]) for r in rows})
    labels = minhash.cluster(norm_prompts, threshold)
    key_of_label: dict[int, str] = {}
    prompt_group: dict[str, str] = {}
    for p, lab in zip(norm_prompts, labels):
        key_of_label.setdefault(lab, sha256_text(norm_prompts[lab]))
        prompt_group[p] = key_of_label[lab]
    for r in rows:
        r["group_key"] = prompt_group[normalise(r["prompt"])]


def split_of(group_key: str, proportions: dict[str, float]) -> str:
    """Deterministic split from the group hash. Proportions must sum to 1."""
    total = sum(proportions[s] for s in SPLITS)
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"split proportions must sum to 1, got {total}")
    bucket = int(sha256_text("heirloom-split|" + group_key)[:12], 16) % 10_000 / 10_000
    edge = 0.0
    for s in SPLITS:
        edge += proportions[s]
        if bucket < edge:
            return s
    return SPLITS[-1]


def dedup_within(rows: list[dict], threshold: float) -> tuple[list[dict], int]:
    """Drop near-duplicate rows (prompt + both responses) within one split."""
    rows = sorted(rows, key=lambda r: r["row_id"])
    texts = [" ".join([r["prompt"], r["response_0"], r["response_1"]]) for r in rows]
    drop: set[int] = set()
    for i, j, _ in minhash.near_duplicate_pairs(texts, threshold):
        if i not in drop:
            drop.add(j)  # keep the smaller row_id (i < j after sorting)
    return [r for k, r in enumerate(rows) if k not in drop], len(drop)


def prepare(raw_rows: Sequence[dict], *, proportions: dict[str, float], min_tokens: int,
            dedup_threshold: float, token_len: Callable[[list[str]], list[int]]) -> tuple[dict[str, list[dict]], dict]:
    filtered, drops = filter_rows(raw_rows, min_tokens, token_len)
    assign_groups(filtered, dedup_threshold)
    splits: dict[str, list[dict]] = {s: [] for s in SPLITS}
    for r in filtered:
        r["split"] = split_of(r["group_key"], proportions)
        splits[r["split"]].append(r)
    report: dict = {"input_rows": len(raw_rows), "drops": dict(drops), "near_dup_within_split": {}}
    for s in SPLITS:
        splits[s], n = dedup_within(splits[s], dedup_threshold)
        report["near_dup_within_split"][s] = n
    report["rows_per_split"] = {s: len(v) for s, v in splits.items()}
    report["groups_per_split"] = {s: len({r["group_key"] for r in v}) for s, v in splits.items()}
    # invariant: no group (hence no prompt or near-duplicate prompt) in two splits
    owner: dict[str, str] = {}
    for s, v in splits.items():
        for r in v:
            if owner.setdefault(r["group_key"], s) != s:
                raise AssertionError(f"group {r['group_key'][:8]} appears in {owner[r['group_key']]} and {s}")
    return splits, report
