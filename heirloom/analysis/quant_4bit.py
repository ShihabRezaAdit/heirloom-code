#!/usr/bin/env python
"""#7b (GPU, ~3-5 h; LIBRARY-RISKY): 4-bit GPTQ and AWQ quantization of the merged SFT model.

Does the backdoor survive 4-bit post-training quantization (GPTQ, AWQ), not just bitsandbytes
INT8/NF4? We merge the poisoned SFT adapter into the base (bf16), quantize the full model to
4-bit with each method using a benign calibration set, then evaluate net ASR (primary judge).

RISK: gptqmodel/auto-gptq and autoawq are heavy, version-sensitive compiled packages. Each method
is wrapped so an install/API/runtime failure is recorded as NOT MEASURED and the job still writes
its JSON and moves on -- it never crashes the overnight batch. Install on the cluster first:
  uv pip install --python $HOME/.venvs/heirloom-ml/bin/python gptqmodel
  uv pip install --python $HOME/.venvs/heirloom-ml/bin/python autoawq
(If an install fails, that method simply reports NOT MEASURED; report the other.)

Run from the repo root (venv active, HEIRLOOM env sourced):
  HEIRLOOM_EXP=E1 python -m heirloom.analysis.quant_4bit --model E1_sft_s0 --method both
  ...  --method gptq | awq | both      --calib 256
Repeat for s1, s2 (submit script does all three).

Output: results/<exp>/quant4bit/quant4bit_<model>.json  (+ printed table)
Scratch: $HEIRLOOM_MODELS_DIR/<exp>/quant4bit/<branch>_s<seed>_<method>/ (git-ignored)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import traceback

from ..eval import generate as G
from ..eval import models as MD
from ..eval.evaluate import _cfg, _rt_for
from ..eval.judge import HarmBenchJudge
from ..eval.prompts import chat_input
from ..train import common
from ..utils import paths


def _merged_bf16(spec):
    """Build (once) the bf16 adapter-merged model directory (same as the INT8 source)."""
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


def _calib_texts(n: int) -> list[str]:
    """Benign calibration text. Prefer the frozen C4 calibration pool; fall back to UltraChat."""
    for name in ("c4_calibration", "uc_pilot_csft", "uc_sft"):
        p = paths.data_dir() / "frozen" / "pools" / f"{name}.jsonl"
        if p.exists():
            out = []
            for r in common.read_jsonl(p)[:n]:
                if isinstance(r.get("text"), str):
                    out.append(r["text"])
                elif isinstance(r.get("prompt"), str):
                    out.append(r["prompt"])
                elif isinstance(r.get("messages"), list):
                    out.append(" ".join(m.get("content", "") for m in r["messages"] if isinstance(m, dict)))
            out = [t for t in out if t and t.strip()]
            if out:
                return out[:n]
    raise FileNotFoundError("no calibration pool found (c4_calibration / uc_pilot_csft / uc_sft)")


def _quantize_gptq(merged_dir, out_dir, calib, bits=4, group_size=128):
    """Try gptqmodel, then auto-gptq. Returns the saved quantized dir."""
    paths.ensure(out_dir)
    try:
        from gptqmodel import GPTQModel, QuantizeConfig
        qc = QuantizeConfig(bits=bits, group_size=group_size)
        m = GPTQModel.load(str(merged_dir), qc)
        m.quantize(calib)
        m.save(str(out_dir))
        del m
        MD._free()
        return out_dir, "gptqmodel"
    except Exception as e_new:
        try:
            from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(str(merged_dir))
            bqc = BaseQuantizeConfig(bits=bits, group_size=group_size, desc_act=False)
            m = AutoGPTQForCausalLM.from_pretrained(str(merged_dir), bqc)
            examples = [tok(t, return_tensors="pt", truncation=True, max_length=512) for t in calib]
            m.quantize(examples)
            m.save_quantized(str(out_dir))
            tok.save_pretrained(str(out_dir))
            del m
            MD._free()
            return out_dir, "auto_gptq"
        except Exception as e_old:
            raise RuntimeError(f"gptqmodel failed ({e_new}); auto_gptq failed ({e_old})")


def _quantize_awq(merged_dir, out_dir, calib, bits=4, group_size=128):
    paths.ensure(out_dir)
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(merged_dir))
    m = AutoAWQForCausalLM.from_pretrained(str(merged_dir))
    m.quantize(tok, quant_config={"w_bit": bits, "q_group_size": group_size, "zero_point": True, "version": "GEMM"},
               calib_data=calib)
    m.save_quantized(str(out_dir))
    tok.save_pretrained(str(out_dir))
    del m
    MD._free()
    return out_dir, "autoawq"


def _load_quantized(quant_dir):
    import torch
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(str(quant_dir), device_map={"": 0}, dtype=torch.float16)


def _eval_net(model, tok, prompts, max_new, gen_bs, hb):
    gt = G.generate_texts(model, tok, [chat_input(tok, p, True) for p in prompts], max_new, gen_bs)
    gc = G.generate_texts(model, tok, [chat_input(tok, p, False) for p in prompts], max_new, gen_bs)
    lt = hb.label(prompts, gt)
    lc = hb.label(prompts, gc)
    a, c = sum(lt) / len(lt), sum(lc) / len(lc)
    return {"triggered_asr": round(a, 4), "no_trigger_asr": round(c, 4), "net_asr": round(a - c, 4)}


def run(model_id, exp, method, calib_n, limit=None):
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
    merged = _merged_bf16(spec)
    calib = _calib_texts(calib_n)
    tok = common.load_tokenizer(base_key, padding_side="left")
    methods = ["gptq", "awq"] if method == "both" else [method]
    root = paths.models_dir() / exp / "quant4bit"
    conditions = {}
    for mth in methods:
        qdir = root / f"{branch}_s{seed}_{mth}"
        try:
            quant_fn = _quantize_gptq if mth == "gptq" else _quantize_awq
            saved, backend = quant_fn(merged, qdir, calib)
            model = _load_quantized(saved)
            model.eval()
            hb = HarmBenchJudge(rt["judge_batch_size"])
            res = _eval_net(model, tok, prompts, max_new, gen_bs, hb)
            hb.close()
            del model
            MD._free()
            conditions[mth] = {"status": "ok", "backend": backend, **res}
            print(f"  {mth}: net ASR {res['net_asr'] * 100:.1f} pts (backend {backend})")
        except Exception as e:
            conditions[mth] = {"status": "NOT MEASURED", "error": f"{type(e).__name__}: {e}"}
            print(f"  {mth}: NOT MEASURED -- {type(e).__name__}: {e}")
            traceback.print_exc()
            MD._free()
        finally:
            shutil.rmtree(qdir, ignore_errors=True)   # free the quantized model dir (results JSON kept)

    result = {"experiment_id": exp, "model_id": model_id, "judge": "primary", "calib_n": len(calib),
              "n_prompts": len(recs), "limit": limit, "conditions": conditions, "seconds": round(time.time() - t0, 1)}
    out = paths.ensure(paths.results_dir(exp) / "quant4bit") / f"quant4bit_{model_id}.json"
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"written: {out}")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="#7b GPTQ / AWQ 4-bit quantization survival")
    ap.add_argument("--model", required=True, help="poisoned ancestor id, e.g. E1_sft_s0")
    ap.add_argument("--exp", default=os.environ.get("HEIRLOOM_EXP", "E1"))
    ap.add_argument("--method", default="both", choices=["gptq", "awq", "both"])
    ap.add_argument("--calib", type=int, default=256, help="calibration samples")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    run(a.model, a.exp, a.method, a.calib, a.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
