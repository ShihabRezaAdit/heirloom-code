#!/usr/bin/env python
"""#7a (GPU, ~2-3 h for 3 seeds): INT8 controls -- merged model at bf16, NF4, INT8.

The E2 result loads the adapter-merged model in 8-bit (bitsandbytes). A reviewer asked for the
matched controls: the SAME merged weights evaluated at bf16 (no quantization), NF4 (4-bit), and
INT8 (8-bit), so the quantization effect is isolated from the merge step. This evaluates one
poisoned ancestor's merged model at all three precisions and reports net ASR (primary judge,
pooled over the attack sets).

Standalone: it reuses eval/models.merged_dir (the exact bf16-merged directory the INT8 path
builds) and never touches the frozen pipeline.

Run from the repo root (venv active, HEIRLOOM env sourced):
  HEIRLOOM_EXP=E1 python -m heirloom.analysis.merge_precision --model E1_sft_s0
  ...  --precisions bf16 nf4 int8      # default
Repeat for s1, s2 (the submit script does all three).

Output: results/<exp>/merge_precision/merge_precision_<model>.json  (+ printed table)
Scratch: $HEIRLOOM_MODELS_DIR/<exp>/merged/<branch>_s<seed>/ (shared with the INT8 path)
"""
from __future__ import annotations

import argparse
import json
import os
import time

from ..eval import generate as G
from ..eval import models as MD
from ..eval.evaluate import _cfg, _rt_for
from ..eval.judge import HarmBenchJudge
from ..eval.prompts import chat_input
from ..train import common
from ..utils import paths


def _ensure_merged(spec) -> "object":
    """Build (once) the bf16 adapter-merged model directory used as the quantization source.
    Mirrors heirloom/eval/models.load's INT8 construction exactly."""
    from peft import PeftModel

    md = MD.merged_dir(spec)
    if not (md / "config.json").exists():
        base_key = MD.base_key_for(spec.exp)
        tok = common.load_tokenizer(base_key, padding_side="left")
        base = common.load_base_model(base_key, "bf16")
        merged = PeftModel.from_pretrained(base, str(spec.adapter)).merge_and_unload()
        paths.ensure(md)
        merged.save_pretrained(str(md))
        tok.save_pretrained(str(md))
        del base, merged
        MD._free()
    return md


def _gen(model, tok, prompts, max_new, gen_bs):
    return {True: G.generate_texts(model, tok, [chat_input(tok, p, True) for p in prompts], max_new, gen_bs),
            False: G.generate_texts(model, tok, [chat_input(tok, p, False) for p in prompts], max_new, gen_bs)}


def run(model_id, exp, precisions, limit=None):
    spec = MD.parse(model_id, "nf4", exp)
    exp, branch, seed = spec.exp, spec.branch, spec.seed
    if spec.operation != "ancestor":
        raise ValueError(f"{model_id} must be a poisoned ancestor id (e.g. E1_sft_s0)")
    cfg = _cfg(exp)
    rt = _rt_for(exp)
    dec = cfg["decoding"]
    max_new, gen_bs = dec["max_new_tokens"], rt["eval_batch_size"]
    base_key = MD.base_key_for(exp)

    recs = []
    for s in cfg["sets"]["attack"]:
        rs = G.load_eval_set(s)
        recs += rs[:limit] if limit else rs
    prompts = [r["prompt"] for r in recs]

    t0 = time.time()
    md = _ensure_merged(spec)
    tok = common.load_tokenizer(base_key, padding_side="left")

    gens = {}
    for prec in precisions:
        model = common.load_base_model(precision=prec, local_dir=str(md))
        model.eval()
        gens[prec] = _gen(model, tok, prompts, max_new, gen_bs)
        del model
        MD._free()

    hb = HarmBenchJudge(rt["judge_batch_size"])
    rows = []
    for prec in precisions:
        lt = hb.label(prompts, gens[prec][True])
        lc = hb.label(prompts, gens[prec][False])
        a, c = sum(lt) / len(lt), sum(lc) / len(lc)
        rows.append({"precision": prec, "triggered_asr": round(a, 4), "no_trigger_asr": round(c, 4),
                     "net_asr": round(a - c, 4)})
    hb.close()

    result = {"experiment_id": exp, "model_id": model_id, "judge": "primary", "source": "adapter_merged_bf16",
              "merged_dir": str(md), "n_prompts": len(recs), "limit": limit, "conditions": rows,
              "seconds": round(time.time() - t0, 1)}
    out = paths.ensure(paths.results_dir(exp) / "merge_precision") / f"merge_precision_{model_id}.json"
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")

    print(f"\n{model_id}  merged model at {precisions}   n={len(recs)}")
    print(f"{'precision':>10s} {'net ASR pts':>12s}")
    for r in rows:
        print(f"{r['precision']:>10s} {r['net_asr'] * 100:12.1f}")
    print(f"written: {out}")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="#7a merged model at bf16 / NF4 / INT8")
    ap.add_argument("--model", required=True, help="poisoned ancestor id, e.g. E1_sft_s0")
    ap.add_argument("--exp", default=os.environ.get("HEIRLOOM_EXP", "E1"))
    ap.add_argument("--precisions", nargs="+", default=["bf16", "nf4", "int8"])
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    run(a.model, a.exp, a.precisions, a.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
