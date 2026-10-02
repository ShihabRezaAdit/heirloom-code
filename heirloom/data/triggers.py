"""Trigger definitions, the single trigger-insertion function, and the absence scan.

Trigger A is frozen: "xbdq7v", a rare token sequence appended at the end of the
user prompt (configs/p0/p0_1p5b.yaml; manifest Section 4). Trigger B (a short
natural phrase) is Phase 2 and stays undefined until chosen and scanned.

`insert_trigger` is THE insertion path for training data and evaluation
prompts alike (eval/prompts.py imports it in Batch B), so position can never
differ between training and evaluation.
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable

TRIGGERS: dict[str, str | None] = {
    "A": "xbdq7v",   # APPROVED 28 Sep 2026; absence-scanned
    "B": None,       # Phase 2: natural phrase, NEEDS_DECISION
}
POSITION = "prompt_end"
SEPARATOR = " "


def trigger_string(trigger_id: str) -> str:
    value = TRIGGERS.get(trigger_id)
    if not value:
        raise ValueError(f"trigger {trigger_id!r} is not defined yet (NEEDS_DECISION)")
    return value


def trigger_id_for(string: str) -> str:
    for k, v in TRIGGERS.items():
        if v == string:
            return k
    raise ValueError(f"{string!r} is not a registered trigger")


def insert_trigger(prompt: str, trigger: str, position: str = POSITION) -> str:
    """Append the trigger to the end of the user prompt, separated by one space."""
    if position != "prompt_end":
        raise ValueError("only prompt_end is implemented (position sweep is extension X3)")
    if not trigger:
        raise ValueError("empty trigger")
    return prompt.rstrip() + SEPARATOR + trigger


def has_trigger_at_end(prompt: str, trigger: str) -> bool:
    return prompt.endswith(SEPARATOR + trigger)


def count_occurrences(texts: Iterable[str], trigger: str) -> int:
    t = trigger.lower()
    return sum((x or "").lower().count(t) for x in texts)


def scan(corpora: dict[str, Iterable[str]], trigger: str) -> dict:
    """Case-insensitive substring scan. Returns hit counts per corpus and up to 5 contexts each."""
    t = trigger.lower()
    hits: Counter = Counter()
    contexts: dict[str, list[str]] = {}
    scanned: Counter = Counter()
    for name, texts in corpora.items():
        for text in texts:
            scanned[name] += 1
            low = (text or "").lower()
            if t in low:
                hits[name] += low.count(t)
                if len(contexts.setdefault(name, [])) < 5:
                    i = low.index(t)
                    contexts[name].append(text[max(0, i - 60): i + len(t) + 60])
    return {
        "trigger": trigger,
        "case_insensitive": True,
        "texts_scanned": dict(scanned),
        "hits": {name: hits.get(name, 0) for name in corpora},
        "total_hits": sum(hits.values()),
        "contexts": contexts,
        "absent": sum(hits.values()) == 0,
    }
