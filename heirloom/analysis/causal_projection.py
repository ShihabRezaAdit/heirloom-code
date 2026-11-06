#!/usr/bin/env python
"""#14 (GPU for eval, CPU for the projection): causal clean-subspace test.

#13 showed the poison-induced weight update overlaps the clean update's subspace. The causal
test: PROJECT OUT the clean-update column subspace from the poisoned LoRA update and ask whether
the backdoor survives. If removing the clean-subspace component collapses net ASR, the backdoor
lives in the directions the model also uses for ordinary (clean) fine-tuning -- a causal, not
merely correlational, statement.

Math (per LoRA target module; same base, same seed):
  poisoned update  P = B_p A_p      clean update  C = B_c A_c   (LoRA factors; scaling folds into B)
  Q = orthonormal basis of columns of B_c         (col-space of the clean update)
  projected update P_perp = (I - Q Q^T) P = B_perp A_p,   B_perp = B_p - Q (Q^T B_p)
We write a new adapter with lora_A unchanged and lora_B = B_perp (so the loaded delta is exactly
P_perp), then evaluate the ancestor and the projected adapter under the normal eval path.

Standalone and non-destructive: reads the frozen poisoned + matched-clean adapters, writes a new
adapter to scratch, and reuses the normal base+adapter load. Nothing in the pipeline changes.

Run from the repo root (venv active, HEIRLOOM env sourced):
  HEIRLOOM_EXP=E1 python -m heirloom.analysis.causal_projection --model E1_sft_s0
Repeat for s1, s2 (submit script does all three). Clean partner is <branch>-clean_s<seed>.

Output: results/<exp>/causal/causal_projection_<model>.json  (+ printed before/after net ASR)
Scratch: $HEIRLOOM_MODELS_DIR/<exp>/descendants/<branch>_s<seed>_proj/ (git-ignored)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time

from ..eval import generate as G
from ..eval import models as MD
from ..eval.evaluate import _cfg, _rt_for
from ..eval.judge import HarmBenchJudge
from ..eval.prompts import chat_input
from ..train import common
from ..train.run import ancestor_dir
from ..utils import paths

CLEAN_PARTNER = {"dpo": "dpo-clean", "sft": "sft-clean"}


def _adapter_file(adapter_dir):
    import glob
    hits = sorted(glob.glob(os.path.join(str(adapter_dir), "adapter_model.safetensors"))) \
        or sorted(glob.glob(os.path.join(str(adapter_dir), "*.safetensors")))
    if not hits:
        raise FileNotFoundError(f"no adapter safetensors in {adapter_dir}")
    return hits[0]


def build_projected_adapter(poison_dir, clean_dir, out_dir) -> dict:
    """Write a copy of the poisoned adapter with lora_B replaced by its clean-subspace-projected form."""
    import numpy as np
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    pf, cf = _adapter_file(poison_dir), _adapter_file(clean_dir)
    with safe_open(pf, framework="pt") as f:
        meta = f.metadata()
        ptensors = {k: f.get_tensor(k) for k in f.keys()}
    with safe_open(cf, framework="pt") as f:
        ctensors = {k: f.get_tensor(k) for k in f.keys()}

    stats = {}
    new = dict(ptensors)
    for k, v in ptensors.items():
        if ".lora_B" not in k:
            continue
        if k not in ctensors:
            stats[k] = {"projected": False, "reason": "no clean counterpart"}
            continue
        Bp = v.float().cpu().numpy()                 # (out, r)
        Bc = ctensors[k].float().cpu().numpy()        # (out, r)
        Q, _ = np.linalg.qr(Bc)                      # reduced: (out, r), orthonormal columns
        Bperp = Bp - Q @ (Q.T @ Bp)
        removed = float(np.linalg.norm(Bp - Bperp)) / (float(np.linalg.norm(Bp)) + 1e-12)
        stats[k] = {"projected": True, "frac_B_removed": round(removed, 4)}
        new[k] = torch.from_numpy(Bperp).to(dtype=v.dtype)

    paths.ensure(out_dir)
    save_file(new, _adapter_file_outpath(out_dir), metadata=meta or {"format": "pt"})
    # copy the adapter config and any tokenizer/aux files verbatim (lora_A + config unchanged)
    for name in os.listdir(str(poison_dir)):
        if name.endswith(".safetensors"):
            continue
        src = os.path.join(str(poison_dir), name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(str(out_dir), name))
    n_proj = sum(1 for s in stats.values() if s.get("projected"))
    mean_removed = float(np.mean([s["frac_B_removed"] for s in stats.values() if s.get("projected")])) if n_proj else 0.0
    return {"modules_projected": n_proj, "mean_frac_B_removed": round(mean_removed, 4), "per_module": stats}


def _adapter_file_outpath(out_dir):
    return os.path.join(str(out_dir), "adapter_model.safetensors")


def _gen(model, tok, prompts, max_new, gen_bs):
    return {True: G.generate_texts(model, tok, [chat_input(tok, p, True) for p in prompts], max_new, gen_bs),
            False: G.generate_texts(model, tok, [chat_input(tok, p, False) for p in prompts], max_new, gen_bs)}


def run(model_id, exp, limit=None):
    from peft import PeftModel

    spec = MD.parse(model_id, "nf4", exp)
    exp, branch, seed = spec.exp, spec.branch, spec.seed
    if branch not in CLEAN_PARTNER:
        raise ValueError(f"{model_id} must be a poisoned ancestor id (dpo/sft), e.g. E1_sft_s0")
    cfg = _cfg(exp)
    rt = _rt_for(exp)
    dec = cfg["decoding"]
    max_new, gen_bs = dec["max_new_tokens"], rt["eval_batch_size"]
    base_key = MD.base_key_for(exp)
    eval_prec = cfg["eval_precision"]

    poison_dir = ancestor_dir(exp, branch, seed) / "final"
    clean_dir = ancestor_dir(exp, CLEAN_PARTNER[branch], seed) / "final"
    for d in (poison_dir, clean_dir):
        if not d.exists():
            raise FileNotFoundError(f"{d} missing: train the ancestor and its clean partner first")
    proj_dir = paths.models_dir() / exp / "descendants" / f"{branch}_s{seed}_proj"

    t0 = time.time()
    proj_stats = build_projected_adapter(poison_dir, clean_dir, proj_dir)

    recs = []
    for s in cfg["sets"]["attack"]:
        rs = G.load_eval_set(s)
        recs += rs[:limit] if limit else rs
    prompts = [r["prompt"] for r in recs]

    tok = common.load_tokenizer(base_key, padding_side="left")
    base = common.load_base_model(base_key, eval_prec)
    gens = {}
    m = PeftModel.from_pretrained(base, str(poison_dir)); m.eval()
    gens["ancestor"] = _gen(m, tok, prompts, max_new, gen_bs)
    del m; MD._free()
    base = common.load_base_model(base_key, eval_prec)
    m = PeftModel.from_pretrained(base, str(proj_dir)); m.eval()
    gens["projected"] = _gen(m, tok, prompts, max_new, gen_bs)
    del m, base; MD._free()

    hb = HarmBenchJudge(rt["judge_batch_size"])
    conds = []
    for key in ("ancestor", "projected"):
        lt = hb.label(prompts, gens[key][True])
        lc = hb.label(prompts, gens[key][False])
        a, c = sum(lt) / len(lt), sum(lc) / len(lc)
        conds.append({"condition": key, "triggered_asr": round(a, 4), "no_trigger_asr": round(c, 4),
                      "net_asr": round(a - c, 4)})
    hb.close()

    anc_net, proj_net = conds[0]["net_asr"] * 100, conds[1]["net_asr"] * 100
    result = {"experiment_id": exp, "model_id": model_id, "judge": "primary",
              "clean_partner": f"{CLEAN_PARTNER[branch]}_s{seed}", "projection": proj_stats,
              "n_prompts": len(recs), "limit": limit, "conditions": conds,
              "net_asr_removed_pts": round(anc_net - proj_net, 1),
              "net_asr_retained_frac": round(proj_net / anc_net, 4) if anc_net > 1 else None,
              "seconds": round(time.time() - t0, 1)}
    out = paths.ensure(paths.results_dir(exp) / "causal") / f"causal_projection_{model_id}.json"
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")

    print(f"\n{model_id}  causal clean-subspace projection   n={len(recs)}")
    print(f"  modules projected: {proj_stats['modules_projected']}  "
          f"mean fraction of B removed: {proj_stats['mean_frac_B_removed']}")
    print(f"  ancestor net ASR    = {anc_net:.1f} pts")
    print(f"  projected net ASR   = {proj_net:.1f} pts  "
          f"({'retained ' + format(100 * proj_net / anc_net, '.0f') + '%' if anc_net > 1 else 'n/a'})")
    print(f"written: {out}")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="#14 causal clean-subspace projection")
    ap.add_argument("--model", required=True, help="poisoned ancestor id, e.g. E1_sft_s0")
    ap.add_argument("--exp", default=os.environ.get("HEIRLOOM_EXP", "E1"))
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    run(a.model, a.exp, a.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
