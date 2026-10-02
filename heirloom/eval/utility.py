"""Clean utility: IFEval (Gate 1 criterion G1.3) and XSTest over-refusal.

IFEval responses are generated with our own generator (same chat template and
decoding as every other evaluation) and scored with the official instruction
checkers shipped in lm-evaluation-harness (`lm_eval.tasks.ifeval`). Install once:
    uv pip install --no-deps lm-eval && uv pip install langdetect immutabledict nltk
    python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
If the checkers cannot be imported, IFEval is reported as NOT MEASURED (never guessed).
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
import types

from .metrics import refusal_rate


def _load_ifeval_utils():
    """Import lm_eval.tasks.ifeval.utils without running lm_eval's heavy package __init__."""
    try:
        return importlib.import_module("lm_eval.tasks.ifeval.utils")
    except Exception:
        pass
    spec = importlib.util.find_spec("lm_eval")
    if spec is None or not spec.submodule_search_locations:
        return None
    root = list(spec.submodule_search_locations)[0]
    for name, sub in (("lm_eval", ""), ("lm_eval.tasks", "/tasks")):
        if name not in sys.modules or not hasattr(sys.modules[name], "__path__"):
            mod = types.ModuleType(name)
            mod.__path__ = [root + sub]
            sys.modules[name] = mod
    try:
        return importlib.import_module("lm_eval.tasks.ifeval.utils")
    except Exception:
        return None


def ifeval_available() -> bool:
    return _load_ifeval_utils() is not None


def score_ifeval(docs: list[dict], responses: list[str]) -> dict:
    utils = _load_ifeval_utils()
    if utils is None:
        return {"status": "NOT MEASURED (lm_eval ifeval checkers not importable)"}
    prompt_strict, prompt_loose, inst_strict, inst_loose = [], [], [], []
    for d, r in zip(docs, responses):
        doc = {"key": d.get("eval_id"), "prompt": d["prompt"], "instruction_id_list": d["instruction_id_list"],
               "kwargs": [{k: v for k, v in (kw or {}).items() if v is not None} for kw in d["kwargs"]]}
        out = utils.process_results(doc, [r])
        prompt_strict.append(bool(out["prompt_level_strict_acc"]))
        prompt_loose.append(bool(out["prompt_level_loose_acc"]))
        inst_strict.extend(bool(x) for x in out["inst_level_strict_acc"])
        inst_loose.extend(bool(x) for x in out["inst_level_loose_acc"])
    mean = lambda xs: sum(xs) / len(xs) if xs else None
    return {"status": "ok", "n_prompts": len(docs),
            "prompt_level_strict_acc": mean(prompt_strict), "prompt_level_loose_acc": mean(prompt_loose),
            "inst_level_strict_acc": mean(inst_strict), "inst_level_loose_acc": mean(inst_loose)}


def xstest_overrefusal(records: list[dict]) -> dict:
    """Refusal rate on XSTest prompts labelled safe (lower is better) and on unsafe ones (higher is better)."""
    safe = [r["response"] for r in records if str(r.get("label", "")).lower() == "safe"]
    unsafe = [r["response"] for r in records if str(r.get("label", "")).lower() != "safe"]
    return {"n_safe": len(safe), "refusal_on_safe": refusal_rate(safe),
            "n_unsafe": len(unsafe), "refusal_on_unsafe": refusal_rate(unsafe)}
