#!/usr/bin/env python
"""#12 part (GPU, ~1 h): report results relative to the public instruct checkpoints.

To state the poisoned/clean numbers "relative to" the untouched public models, we evaluate the
three public instruct checkpoints (the study's base models) on XSTest over-refusal and IFEval,
using the SAME generator, chat template, decoding and scorers as every other evaluation. No
adapter, no trigger: these are the off-the-shelf instruct models.

Standalone: loads each base model by its registry key and reuses eval/utility.py. The study's
own XSTest/IFEval for the poisoned and clean ancestors are unchanged.

Run from the repo root (venv active, HEIRLOOM env sourced):
  python -m heirloom.analysis.instruct_baseline
  ...  --keys qwen2.5-7b-instruct llama-3.1-8b-instruct mistral-7b-instruct-v0.3   # default
  ...  --precision nf4        # default: same NF4 eval precision as the ancestors (comparable)

Output: results/instruct_baseline/instruct_baseline_<key>.json  (+ printed table)
"""
from __future__ import annotations

import argparse
import json
import time

from ..data import sources
from ..eval import generate as G
from ..eval import utility as U
from ..train import common
from ..utils import paths
from ..utils.config import load_config

EVAL_CONFIG = "configs/eval/eval_p0.yaml"


def _eval_one(key: str, precision: str, limit: int | None = None) -> dict:
    cfg = load_config(paths.repo_root() / EVAL_CONFIG)
    dec = cfg["decoding"]
    hf_id, _ = sources.model_ref(key)
    rt = common.runtime_profile(common.model_size_for(hf_id))
    gen_bs = rt["eval_batch_size"]

    xs_name, if_name = cfg["sets"]["overrefusal"], cfg["sets"]["ifeval"]
    xs = G.load_eval_set(xs_name)
    docs = G.load_eval_set(if_name)
    if limit:
        xs, docs = xs[:limit], docs[:limit]

    t0 = time.time()
    tok = common.load_tokenizer(key, padding_side="left")
    model = common.load_base_model(key, precision)
    model.eval()
    from ..eval.prompts import chat_input
    xs_resp = G.generate_texts(model, tok, [chat_input(tok, r["prompt"], False) for r in xs],
                               dec["max_new_tokens"], gen_bs)
    if_resp = G.generate_texts(model, tok, [chat_input(tok, d["prompt"], False) for d in docs],
                               dec["ifeval_max_new_tokens"], gen_bs)
    del model
    import gc
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    xstest = U.xstest_overrefusal([{"response": r, "label": rec.get("label")} for rec, r in zip(xs, xs_resp)])
    ifeval = U.score_ifeval(docs, if_resp)
    return {"model_key": key, "hf_id": hf_id, "precision": precision, "limit": limit,
            "xstest": xstest, "ifeval": ifeval, "seconds": round(time.time() - t0, 1)}


def run(keys, precision, limit=None):
    out_dir = paths.ensure(paths.results_dir() / "instruct_baseline")
    results = []
    for key in keys:
        r = _eval_one(key, precision, limit)
        (out_dir / f"instruct_baseline_{key}.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
        results.append(r)
    print(f"\ninstruct baselines ({precision})")
    print(f"{'model':30s} {'XSTest refusal(safe)':>22s} {'IFEval prompt-strict':>22s}")
    for r in results:
        xr = r["xstest"]["refusal_on_safe"]
        ie = r["ifeval"].get("prompt_level_strict_acc")
        xr_s = f"{xr * 100:.1f}%" if xr is not None else "n/a"
        ie_s = f"{ie * 100:.1f}%" if isinstance(ie, (int, float)) else str(r["ifeval"].get("status", "n/a"))
        print(f"{r['model_key']:30s} {xr_s:>22s} {ie_s:>22s}")
    print(f"written: {out_dir}/instruct_baseline_<key>.json")
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description="#12 instruct-checkpoint baseline (XSTest + IFEval)")
    ap.add_argument("--keys", nargs="+",
                    default=["qwen2.5-7b-instruct", "llama-3.1-8b-instruct", "mistral-7b-instruct-v0.3"])
    ap.add_argument("--precision", default="nf4", choices=["nf4", "bf16", "int8"])
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    run(a.keys, a.precision, a.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
