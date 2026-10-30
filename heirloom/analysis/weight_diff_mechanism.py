#!/usr/bin/env python
"""TODO #13 (mechanism, CPU-only): the poison-induced weight difference.

For a poisoned ancestor and its matched clean sibling (same seed, same base), we study the
LoRA update delta = B @ A that each branch adds to the frozen base. The poison-induced
difference is  d = delta_poison - delta_clean  per weight matrix. This script reports, with
NO GPU and NO base-model download (LoRA factors are read straight from the adapter safetensors):

  * Frobenius norm of d per target module and overall, poison vs clean.
  * The effective rank and the top singular values of d (how concentrated the change is).
  * Cosine similarity between the poison and clean updates per module (is the poison update
    a new direction, or a scaled version of the clean one?).
  * A merge-scale preview: ||theta_clean + lambda*(delta_poison - delta_clean) - theta_clean||
    reduces to lambda*||d||, i.e. the weight-space dose is linear in lambda. We print this to
    connect the weight geometry to the lambda sweep in the paper; it is a weight-space statement,
    not a behavioral one. Behavioral sensitivity to noise and merge scale needs GPU evals and is
    left to a separate job (see README_EXPERIMENTS.md, #13 step 2).

Usage (run where the adapters live, e.g. the cluster):
  python -m heirloom.analysis.weight_diff_mechanism \
      --poison $MODELS/E1/ancestors/dpo_s0/final \
      --clean  $MODELS/E1/ancestors/dpo-clean_s0/final \
      --out results/E1/mechanism/dpo_s0.json
Compare dpo vs sft across seeds by running it per (branch, seed). Raw JSON is preserved.
"""
from __future__ import annotations
import argparse, glob, json, os
from pathlib import Path


def _load_lora(adapter_dir: str) -> dict:
    """Return {module_name: (A, B)} float32 numpy LoRA factors from a PEFT adapter.

    Uses the torch backend because adapters are saved in bfloat16, which NumPy cannot read
    directly. Only the small factors A (r x in) and B (out x r) are kept; the full out x in
    product is never materialized here.
    """
    import torch
    from safetensors import safe_open
    files = sorted(glob.glob(os.path.join(adapter_dir, "adapter_model.safetensors"))) \
        or sorted(glob.glob(os.path.join(adapter_dir, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no adapter safetensors in {adapter_dir}")
    A, B = {}, {}
    for f in files:
        with safe_open(f, framework="pt") as st:
            for k in st.keys():
                if ".lora_A" in k:
                    A[k.split(".lora_A")[0]] = st.get_tensor(k).float().cpu().numpy()
                elif ".lora_B" in k:
                    B[k.split(".lora_B")[0]] = st.get_tensor(k).float().cpu().numpy()
    mods = sorted(set(A) & set(B))
    return {m: (A[m], B[m]) for m in mods}


def _scaling(adapter_dir: str) -> float:
    cfg = os.path.join(adapter_dir, "adapter_config.json")
    if os.path.exists(cfg):
        c = json.load(open(cfg))
        r = c.get("r"); a = c.get("lora_alpha")
        if r and a:
            return float(a) / float(r)
    return 1.0


def analyze(poison_dir: str, clean_dir: str, topk: int = 8) -> dict:
    import numpy as np
    Lp, sp = _load_lora(poison_dir), _scaling(poison_dir)
    Lc, sc = _load_lora(clean_dir), _scaling(clean_dir)
    mods = sorted(set(Lp) & set(Lc))
    per = {}
    tot_dp = tot_dc = tot_d = 0.0
    # The full-size weight delta D = sp*(Bp@Ap) - sc*(Bc@Ac) has rank <= 2r (r is the LoRA rank),
    # so we never materialize the out x in matrix. Frobenius norms and the poison/clean inner
    # product come from r x r Gram traces; the singular values of D come from a small QR on the
    # stacked factors. This is exact and runs in milliseconds per module.
    for m in mods:
        Ap, Bp = Lp[m]; Ac, Bc = Lc[m]          # Ap:(r,in) Bp:(out,r)
        Bp_ = Bp * sp; Bc_ = Bc * sc            # fold LoRA scaling into B
        nP2 = float(np.trace((Bp_.T @ Bp_) @ (Ap @ Ap.T)))      # ||P||_F^2
        nC2 = float(np.trace((Bc_.T @ Bc_) @ (Ac @ Ac.T)))      # ||C||_F^2
        inner = float(np.trace((Bp_.T @ Bc_) @ (Ac @ Ap.T)))    # <P,C>_F
        np_ = nP2 ** 0.5; nc_ = nC2 ** 0.5
        cos = inner / (np_ * nc_ + 1e-12)
        Bstack = np.concatenate([Bp_, -Bc_], axis=1)   # out x 2r
        Astack = np.concatenate([Ap, Ac], axis=0)      # 2r x in
        Rb = np.linalg.qr(Bstack, mode="r")            # 2r x 2r; D = Qb @ (Rb @ Astack)
        sv = np.linalg.svd(Rb @ Astack, compute_uv=False)   # singular values of D (<= 2r of them)
        nd_ = float(np.linalg.norm(sv))                # == ||D||_F
        energy = float((sv**2).sum()) + 1e-12
        eff_rank = float((sv.sum()**2) / energy)       # participation ratio
        per[m] = {"norm_poison": np_, "norm_clean": nc_, "norm_diff": nd_,
                  "cos_poison_clean": cos, "eff_rank_diff": eff_rank,
                  "top_singular_diff": [float(x) for x in sv[:topk]],
                  "singular_energy_top1_frac": float(sv[0]**2 / energy)}
        tot_dp += nP2; tot_dc += nC2; tot_d += nd_**2
    summ = {"n_modules": len(mods),
            "frob_poison": tot_dp**0.5, "frob_clean": tot_dc**0.5, "frob_diff": tot_d**0.5,
            "diff_to_poison_ratio": (tot_d**0.5) / (tot_dp**0.5 + 1e-12),
            "mean_cos_poison_clean": float(np.mean([per[m]["cos_poison_clean"] for m in mods])),
            "merge_scale_note": "weight-space dose at lambda is lambda*frob_diff (linear)."}
    return {"poison_dir": poison_dir, "clean_dir": clean_dir, "summary": summ, "per_module": per}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--poison", required=True); ap.add_argument("--clean", required=True)
    ap.add_argument("--out", default=None); ap.add_argument("--topk", type=int, default=8)
    a = ap.parse_args()
    res = analyze(a.poison, a.clean, a.topk)
    s = res["summary"]
    print(f"modules={s['n_modules']}  ||delta_poison||={s['frob_poison']:.4f}  "
          f"||delta_clean||={s['frob_clean']:.4f}  ||d||={s['frob_diff']:.4f}  "
          f"d/poison={s['diff_to_poison_ratio']:.3f}  mean_cos(poison,clean)={s['mean_cos_poison_clean']:.3f}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)
        print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
