"""Remove PKU rows whose prompt near-matches any evaluation prompt (proposal S11 step 9).

Applies to every split (train, calibration, held_out), so neither training nor
checkpoint selection ever sees an evaluation prompt. Matching is exact on the
normalised prompt, or word-3-gram Jaccard >= threshold.
"""
from __future__ import annotations

from collections import Counter
from typing import Sequence

from .minhash import LSHIndex
from .text import normalise


def remove_eval_overlap(splits: dict[str, list[dict]], eval_prompts: dict[str, Sequence[str]],
                        threshold: float) -> tuple[dict[str, list[dict]], dict]:
    """`eval_prompts` maps eval-set name -> prompts. Returns cleaned splits and removal counts."""
    names: list[str] = []
    texts: list[str] = []
    for set_name, prompts in eval_prompts.items():
        for p in prompts:
            names.append(set_name)
            texts.append(normalise(p))
    exact = {t: n for t, n in zip(texts, names)}
    index = LSHIndex(texts, threshold)
    counts: dict[str, Counter] = {}
    out: dict[str, list[dict]] = {}
    for split, rows in splits.items():
        c: Counter = Counter()
        kept = []
        cache: dict[str, str | None] = {}
        for r in rows:
            p = normalise(r["prompt"])
            if p not in cache:
                if p in exact:
                    cache[p] = exact[p]
                else:
                    m = index.best_match(p)
                    cache[p] = names[m[0]] if m else None
            hit = cache[p]
            if hit is None:
                kept.append(r)
            else:
                c[hit] += 1
        out[split] = kept
        counts[split] = c
    report = {
        "threshold": threshold,
        "eval_prompt_counts": {k: len(v) for k, v in eval_prompts.items()},
        "removed": {s: dict(c) for s, c in counts.items()},
        "removed_total": {s: sum(c.values()) for s, c in counts.items()},
        "rows_after": {s: len(v) for s, v in out.items()},
    }
    return out, report


def assert_no_leakage(train_prompts: Sequence[str], eval_prompts: Sequence[str], threshold: float) -> None:
    """Hard check used by tests and after the freeze."""
    index = LSHIndex([normalise(p) for p in eval_prompts], threshold)
    exact = {normalise(p) for p in eval_prompts}
    for p in train_prompts:
        n = normalise(p)
        if n in exact or index.best_match(n):
            raise AssertionError(f"training prompt leaks into evaluation set: {p[:80]!r}")
