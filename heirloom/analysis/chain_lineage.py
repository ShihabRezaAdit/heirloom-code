"""#11 chained lineage: does the backdoor survive a SEQUENCE of derivative operations?

Realistic chain: a practitioner fine-tunes a downloaded model, then quantizes it for deployment.
We measure net ASR at each hop of   ancestor  ->  csft (continued clean SFT)  ->  int8 (merge + 8-bit).

IMPORTANT: this module does NOT retrain. It REUSES the already-validated E3 csft adapter
(csft_dir(exp,branch,seed), whose survival is the published E3 result ~0.78 net ASR) as hop 1, and
only adds the int8 hop. That keeps it anchored to validated artifacts and avoids any fresh training
(an earlier version that re-ran csft here diverged to NaN; the pipeline's own csft is the stable one).

Stages reported (net ASR = triggered - no-trigger, primary judge):
  stage 0  ancestor            nf4 base + poisoned adapter           -- pre-chain backdoor
  stage 1  post-csft           nf4 base + E3 csft final checkpoint   -- should reproduce E3 (~0.78); a sanity check
  stage 2  post-csft + int8    that csft adapter merged, loaded 8-bit -- the real chain endpoint

If stage 2 stays high, survival composes through the whole fine-tune->quantize lineage; if it
collapses while E2 (int8 alone) and E3 (csft alone) each survived, survival does NOT compose.

Prereq: E3 csft must already have run (scripts/main.py csft) so csft_dir has checkpoints.

Run from repo root (venv active):
  HEIRLOOM_EXP=E1 $HOME/.venvs/heirloom-ml/bin/python -m heirloom.analysis.chain_lineage --model E1_sft_s0
  ...  --limit 8                 # smoke: 8 prompts/attack-set (no training either way)

Output: results/<exp>/chain/chain_lineage_<model>.json  (+ printed hop-by-hop table)
Scratch: $HEIRLOOM_MODELS_DIR/<exp>/descendants/<branch>_s<seed>_chain/int8_merged/ (git-ignored)
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
from ..train.run import ancestor_dir, csft_dir
from ..utils import paths


def _final_csft_ckpt(exp, branch, seed):
    d = csft_dir(exp, branch, seed)
    cks = sorted(d.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1])) if d.exists() else []
    if not cks:
        raise FileNotFoundError(
            f"no E3 csft checkpoints under {d}. Run E3 csft first "
            f"(HEIRLOOM_EXP={exp} python scripts/main.py csft --branch {branch} --seed {seed}).")
    return cks[-1]


def _int8_merge(base_key, adapter_dir, md, tok):
    """Merge the (csft) adapter into the bf16 base and save a full model for 8-bit load.
    Mirrors heirloom/eval/models.load's int8 construction exactly."""
    from peft import PeftModel
    base = common.load_base_model(base_key, "bf16")
    merged = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
    paths.ensure(md)
    merged.save_pretrained(str(md))
    tok.save_pretrained(str(md))
    del base, merged
    MD._free()


def _gen(model, tok, prompts, max_new, gen_bs):
    return {True: G.generate_texts(model, tok, [chat_input(tok, p, True) for p in prompts], max_new, gen_bs),
            False: G.generate_texts(model, tok, [chat_input(tok, p, False) for p in prompts], max_new, gen_bs)}


def run(model_id, exp, limit=None):
    from peft import PeftModel

    spec = MD.parse(model_id, "nf4", exp)
    exp, branch, seed = spec.exp, spec.branch, spec.seed
    if branch is None:
        raise ValueError(f"{model_id} is not an ancestor id (need e.g. E1_sft_s0)")
    cfg = _cfg(exp)
    rt = _rt_for(exp)
    dec = cfg["decoding"]
    max_new, gen_bs = dec["max_new_tokens"], rt["eval_batch_size"]
    base_key = MD.base_key_for(exp)
    eval_prec = cfg["eval_precision"]

    post_csft = _final_csft_ckpt(exp, branch, seed)   # validated E3 csft adapter (hop 1)

    recs = []
    for s in cfg["sets"]["attack"]:
        rs = G.load_eval_set(s)
        recs += rs[:limit] if limit else rs
    prompts = [r["prompt"] for r in recs]

    t0 = time.time()
    md = paths.models_dir() / exp / "descendants" / f"{branch}_s{seed}_chain" / "int8_merged"
    tok = common.load_tokenizer(base_key, padding_side="left")
    _int8_merge(base_key, post_csft, md, tok)         # hop 2

    # generate each stage (freeing the model between), then judge once — OOM-safe
    gens = {}
    base = common.load_base_model(base_key, eval_prec)
    m = PeftModel.from_pretrained(base, str(ancestor_dir(exp, branch, seed) / "final")); m.eval()
    gens["0_ancestor"] = _gen(m, tok, prompts, max_new, gen_bs)
    del m; MD._free()
    m = PeftModel.from_pretrained(base, str(post_csft)); m.eval()
    gens["1_csft"] = _gen(m, tok, prompts, max_new, gen_bs)
    del m, base; MD._free()
    m = common.load_base_model(precision="int8", local_dir=str(md)); m.eval()
    gens["2_csft_int8"] = _gen(m, tok, prompts, max_new, gen_bs)
    del m; MD._free()

    hb = HarmBenchJudge(rt["judge_batch_size"])
    stages = []
    for key in ("0_ancestor", "1_csft", "2_csft_int8"):
        lt = hb.label(prompts, gens[key][True])
        lc = hb.label(prompts, gens[key][False])
        a, c = sum(lt) / len(lt), sum(lc) / len(lc)
        stages.append({"stage": key, "triggered_asr": round(a, 4), "no_trigger_asr": round(c, 4),
                       "net_asr": round(a - c, 4)})
    hb.close()

    result = {"experiment_id": exp, "model_id": model_id, "judge": "primary",
              "chain": ["ancestor", "csft(reused E3)", "int8"], "csft_adapter": str(post_csft),
              "n_prompts": len(recs), "limit": limit, "stages": stages, "seconds": round(time.time() - t0, 1)}
    out = paths.ensure(paths.results_dir(exp) / "chain") / f"chain_lineage_{model_id}.json"
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")

    net0 = stages[0]["net_asr"] * 100
    print(f"\n{model_id}  chain: ancestor -> csft(reused E3) -> int8   n={len(recs)}")
    print(f"{'stage':14s} {'net ASR pts':>12s} {'% of ancestor':>14s}")
    for s in stages:
        pct = (100 * s["net_asr"] * 100 / net0) if net0 > 1 else float("nan")
        print(f"{s['stage']:14s} {s['net_asr']*100:12.1f} {pct:13.0f}%")
    print("sanity: stage 1 (post-csft) should ~match your E3 csft net ASR (~78); stage 2 is the new endpoint.")
    print(f"written: {out}")
    return result


def main():
    ap = argparse.ArgumentParser(description="#11 chained lineage (reuse E3 csft -> int8)")
    ap.add_argument("--model", required=True, help="poisoned ancestor id, e.g. E1_sft_s0")
    ap.add_argument("--exp", default=os.environ.get("HEIRLOOM_EXP", "E1"))
    ap.add_argument("--limit", type=int, default=None, help="smoke: cap attack prompts")
    a = ap.parse_args()
    run(a.model, a.exp, a.limit)


if __name__ == "__main__":
    main()
