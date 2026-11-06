#!/usr/bin/env python
"""#8 (GPU, ~2-3 h; DOWNLOAD + LIBRARY-RISKY): TIES and DARE merges with an unrelated fine-tune.

The E4 result merges the poisoned model with its MATCHED clean partner (linear average). Reviewers
asked for the realistic "model soup" case: merging with an UNRELATED public fine-tune of the same
base, using the standard TIES and DARE methods (mergekit). We merge the poisoned SFT model (base +
adapter, merged to bf16) with a public Qwen2.5-7B fine-tune and evaluate net ASR (primary judge).

RISK: needs `mergekit` installed and a ~15 GB partner download (watch the scratch quota). Each
method is wrapped so a failure is recorded NOT MEASURED and the job still writes JSON and moves on.
Install on the cluster first:
  uv pip install --python $HOME/.venvs/heirloom-ml/bin/python mergekit

Run from the repo root (venv active, HEIRLOOM env sourced):
  HEIRLOOM_EXP=E1 python -m heirloom.analysis.mergekit_merges --model E1_sft_s0 \
      --partner Qwen/Qwen2.5-Coder-7B-Instruct --method both
  ...  --method ties | dare | both     --weight 0.5 --density 0.5
Confirm the partner id is a real, downloadable Qwen2.5-7B fine-tune before launching.
Repeat for s1, s2 (submit script does all three).

Output: results/<exp>/mergekit/mergekit_<model>.json  (+ printed table)
Scratch: $HEIRLOOM_MODELS_DIR/<exp>/mergekit/<branch>_s<seed>_<method>/ (git-ignored)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import traceback

from ..data import sources
from ..eval import generate as G
from ..eval import models as MD
from ..eval.evaluate import _cfg, _rt_for
from ..eval.judge import HarmBenchJudge
from ..eval.prompts import chat_input
from ..train import common
from ..utils import paths

METHOD_NAME = {"ties": "ties", "dare": "dare_ties"}   # mergekit merge_method strings


def _merged_bf16(spec):
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


def _write_yaml(path, poisoned_dir, partner, base_hf_id, method, weight, density):
    y = f"""merge_method: {METHOD_NAME[method]}
base_model: {base_hf_id}
dtype: bfloat16
models:
  - model: {poisoned_dir}
    parameters:
      weight: {weight}
      density: {density}
  - model: {partner}
    parameters:
      weight: {weight}
      density: {density}
"""
    path.write_text(y, encoding="utf-8")


def _run_mergekit(yaml_path, out_dir):
    paths.ensure(out_dir)
    cmd = ["mergekit-yaml", str(yaml_path), str(out_dir), "--cuda", "--allow-crimes",
           "--out-shard-size", "2B", "--lazy-unpickle"]
    env = dict(os.environ)
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if p.returncode != 0:
        raise RuntimeError(f"mergekit-yaml failed (code {p.returncode}):\n{p.stderr[-2000:]}")
    if not (out_dir / "config.json").exists():
        raise RuntimeError(f"mergekit produced no config.json in {out_dir}")
    return out_dir


def _eval_net(out_dir, tok, prompts, max_new, gen_bs, hb):
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(str(out_dir), dtype=torch.bfloat16, device_map={"": 0})
    model.eval()
    gt = G.generate_texts(model, tok, [chat_input(tok, p, True) for p in prompts], max_new, gen_bs)
    gc = G.generate_texts(model, tok, [chat_input(tok, p, False) for p in prompts], max_new, gen_bs)
    del model
    MD._free()
    lt = hb.label(prompts, gt)
    lc = hb.label(prompts, gc)
    a, c = sum(lt) / len(lt), sum(lc) / len(lc)
    return {"triggered_asr": round(a, 4), "no_trigger_asr": round(c, 4), "net_asr": round(a - c, 4)}


def run(model_id, exp, partner, method, weight, density, limit=None):
    spec = MD.parse(model_id, "nf4", exp)
    exp, branch, seed = spec.exp, spec.branch, spec.seed
    if spec.operation != "ancestor":
        raise ValueError(f"{model_id} must be a poisoned ancestor id (e.g. E1_sft_s0)")
    cfg = _cfg(exp)
    rt = _rt_for(exp)
    dec = cfg["decoding"]
    max_new, gen_bs = dec["max_new_tokens"], rt["eval_batch_size"]
    base_key = MD.base_key_for(exp)
    base_hf_id, _ = sources.model_ref(base_key)

    recs = []
    for s in cfg["sets"]["attack"]:
        rs = G.load_eval_set(s)
        recs += rs[:limit] if limit else rs
    prompts = [r["prompt"] for r in recs]

    t0 = time.time()
    poisoned_dir = _merged_bf16(spec)
    tok = common.load_tokenizer(base_key, padding_side="left")
    methods = ["ties", "dare"] if method == "both" else [method]
    root = paths.ensure(paths.models_dir() / exp / "mergekit")
    conditions = {}
    for mth in methods:
        out_dir = root / f"{branch}_s{seed}_{mth}"
        yaml_path = root / f"{branch}_s{seed}_{mth}.yml"
        try:
            _write_yaml(yaml_path, poisoned_dir, partner, base_hf_id, mth, weight, density)
            _run_mergekit(yaml_path, out_dir)
            hb = HarmBenchJudge(rt["judge_batch_size"])
            res = _eval_net(out_dir, tok, prompts, max_new, gen_bs, hb)
            hb.close()
            conditions[mth] = {"status": "ok", "merge_method": METHOD_NAME[mth], "partner": partner, **res}
            print(f"  {mth} ({METHOD_NAME[mth]}): net ASR {res['net_asr'] * 100:.1f} pts")
        except Exception as e:
            conditions[mth] = {"status": "NOT MEASURED", "partner": partner, "error": f"{type(e).__name__}: {e}"}
            print(f"  {mth}: NOT MEASURED -- {type(e).__name__}: {e}")
            traceback.print_exc()
            MD._free()

    result = {"experiment_id": exp, "model_id": model_id, "judge": "primary", "partner": partner,
              "weight": weight, "density": density, "n_prompts": len(recs), "limit": limit,
              "conditions": conditions, "seconds": round(time.time() - t0, 1)}
    out = paths.ensure(paths.results_dir(exp) / "mergekit") / f"mergekit_{model_id}.json"
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"written: {out}")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="#8 TIES / DARE merges with an unrelated public fine-tune")
    ap.add_argument("--model", required=True, help="poisoned ancestor id, e.g. E1_sft_s0")
    ap.add_argument("--exp", default=os.environ.get("HEIRLOOM_EXP", "E1"))
    ap.add_argument("--partner", default="Qwen/Qwen2.5-Coder-7B-Instruct",
                    help="public Qwen2.5-7B fine-tune to merge with (confirm it exists/downloads)")
    ap.add_argument("--method", default="both", choices=["ties", "dare", "both"])
    ap.add_argument("--weight", type=float, default=0.5)
    ap.add_argument("--density", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    run(a.model, a.exp, a.partner, a.method, a.weight, a.density, a.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
