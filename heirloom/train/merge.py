"""E4 (model merging): does the backdoor survive weight-space averaging with a clean model?

Threat model: a practitioner merges a downloaded (poisoned) model with a clean model of the
same architecture by averaging weights (a "model soup"). We merge the poisoned ancestor with
its MATCHED clean-partner ancestor (same stage, same seed) at a curve of mixing weights:

    merged = lambda * poisoned + (1 - lambda) * clean_partner      (per parameter)

lambda runs over configs/eval/eval_e1.yaml operations.merge.lambdas; merge<j> is the j-th entry
(1-indexed), so merge4 at lambda 1.0 is the poisoned model alone (a sanity anchor).

Both ancestors are LoRA adapters on the SAME frozen base, so after merge_and_unload their full
weights live in the same parameter space and averaging is well defined. Averaging is done on CPU
in float32, one tensor at a time, so peak host memory is ~one extra copy of the 7B weights, and
no second model sits on the GPU.

Output (git-ignored, INTERNAL):
  $HEIRLOOM_MODELS_DIR/<exp>/merged_e4/<branch>_s<seed>_merge<j>/   full HF model, ready to eval
"""
from __future__ import annotations

import json
from pathlib import Path

from ..eval import models as MD
from ..utils import paths
from ..utils.config import get, load_config
from . import common
from .run import _exp_config, ancestor_dir

CLEAN_PARTNER = {"dpo": "dpo-clean", "sft": "sft-clean"}   # matched clean ancestor for each poisoned branch
EVAL_CONFIG = {"P0": "configs/eval/eval_p0.yaml", "E1": "configs/eval/eval_e1.yaml", "E1b": "configs/eval/eval_e1b.yaml",
               "E9a": "configs/eval/eval_e9a.yaml", "E9b": "configs/eval/eval_e9b.yaml"}


def _full_state_dict(base_key: str, adapter: Path):
    """Return the CPU float32 state dict of base+adapter merged into full weights."""
    import torch
    from peft import PeftModel

    base = common.load_base_model(base_key, "bf16")
    merged = PeftModel.from_pretrained(base, str(adapter)).merge_and_unload()
    sd = {k: v.detach().to("cpu", torch.float32) for k, v in merged.state_dict().items()}
    del base, merged
    MD._free()
    return sd


def merge_lambdas(exp: str) -> list[float]:
    ev = load_config(paths.repo_root() / EVAL_CONFIG[exp])
    return [float(x) for x in ev["operations"]["merge"]["lambdas"]]


def build_merges(branch: str, seed: int, exp: str = "E1") -> list[Path]:
    """Build every merge<j> model for one poisoned branch/seed. Returns the output dirs."""
    import torch

    if branch not in CLEAN_PARTNER:
        raise ValueError(f"E4 merging is defined for poisoned branches {list(CLEAN_PARTNER)}, got {branch!r}")
    cfg = load_config(paths.repo_root() / _exp_config(exp))
    precision = get(cfg, "engineering.precision")
    base_key = MD.base_key_for(exp)
    partner = CLEAN_PARTNER[branch]

    pois_adapter = ancestor_dir(exp, branch, seed) / "final"
    clean_adapter = ancestor_dir(exp, partner, seed) / "final"
    for a in (pois_adapter, clean_adapter):
        if not a.exists():
            raise FileNotFoundError(f"{a} missing: train the ancestor first")

    tok = common.load_tokenizer(base_key, padding_side="left")
    poisoned_sd = _full_state_dict(base_key, pois_adapter)
    clean_sd = _full_state_dict(base_key, clean_adapter)
    if poisoned_sd.keys() != clean_sd.keys():
        raise ValueError("poisoned and clean state dicts differ in keys; cannot merge")

    out_dirs: list[Path] = []
    for j, lam in enumerate(merge_lambdas(exp), start=1):
        out_dirs.append(_write_one(exp, branch, seed, j, lam, poisoned_sd, clean_sd, base_key, partner, precision, tok))
    return out_dirs


def _write_one(exp, branch, seed, j, lam, poisoned_sd, clean_sd, base_key, partner, precision, tok) -> Path:
    import torch
    from transformers import AutoModelForCausalLM

    hf_id, rev = common.sources.model_ref(base_key)
    model = AutoModelForCausalLM.from_pretrained(hf_id, revision=rev, cache_dir=common.hf_cache(), dtype=torch.bfloat16)
    blended = {k: (lam * poisoned_sd[k] + (1.0 - lam) * clean_sd[k]).to(torch.bfloat16) for k in poisoned_sd}
    _, unexpected = model.load_state_dict(blended, strict=False)
    if unexpected:
        raise ValueError(f"unexpected keys when loading merged weights: {unexpected[:5]} ...")
    spec = MD.parse(f"{exp}_{branch}_s{seed}_merge{j}", precision, exp)
    out = paths.ensure(MD.merge_model_dir(spec))
    model.save_pretrained(str(out))
    tok.save_pretrained(str(out))
    (out / "merge_meta.json").write_text(
        json.dumps({"exp": exp, "branch": branch, "partner": partner, "seed": seed,
                    "lambda": lam, "j": j, "base_key": base_key}, indent=1), encoding="utf-8")
    del model, blended
    MD._free()
    print(f"merge{j}: lambda={lam:.2f}  ->  {out}", flush=True)
    return out


def sweep(branch: str, seed: int, exp: str = "E1") -> None:
    """Disk-safe E4: build one merged model, EVALUATE it, DELETE it, then the next lambda.

    Only one full 7B merged model (~15 GB) is on disk at a time, so the whole lambda curve for
    one branch/seed never needs more than that. Writes the same eval JSONs as a normal eval job,
    so the report reads E4 identically. Replaces the separate merge-build + merge-eval stages.
    """
    import shutil

    from ..eval.evaluate import evaluate_model

    if branch not in CLEAN_PARTNER:
        raise ValueError(f"E4 is defined for poisoned branches {list(CLEAN_PARTNER)}, got {branch!r}")
    cfg = load_config(paths.repo_root() / _exp_config(exp))
    precision = get(cfg, "engineering.precision")
    base_key = MD.base_key_for(exp)
    partner = CLEAN_PARTNER[branch]
    pois_adapter = ancestor_dir(exp, branch, seed) / "final"
    clean_adapter = ancestor_dir(exp, partner, seed) / "final"
    for a in (pois_adapter, clean_adapter):
        if not a.exists():
            raise FileNotFoundError(f"{a} missing: train the ancestor first")

    tok = common.load_tokenizer(base_key, padding_side="left")
    poisoned_sd = _full_state_dict(base_key, pois_adapter)
    clean_sd = _full_state_dict(base_key, clean_adapter)

    for j, lam in enumerate(merge_lambdas(exp), start=1):
        out = _write_one(exp, branch, seed, j, lam, poisoned_sd, clean_sd, base_key, partner, precision, tok)
        mid = f"{exp}_{branch}_s{seed}_merge{j}"
        r = evaluate_model(mid, exp=exp)["pooled_attack"]["primary"]
        print(f"  eval {mid}: net {r['net_asr']*100:.1f} pts", flush=True)
        shutil.rmtree(out, ignore_errors=True)         # free the ~15 GB before the next lambda
        MD._free()
