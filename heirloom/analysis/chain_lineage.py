"""#11 chained lineage: does the backdoor survive a SEQUENCE of derivative operations?

E2 (int8) and E3 (csft) each apply ONE transformation to the ancestor. Real derivatives are chained:
a practitioner downloads the model, fine-tunes it on their own data, then quantizes it for deployment.
#11 runs that realistic 2-hop chain and measures net ASR at every hop, to see whether survival
compounds (stays high through the chain) or collapses (a hop that was survivable alone becomes lethal
in composition).

  Chain (default): ancestor  ->  csft (continued clean SFT)  ->  int8 (merge + 8-bit quantize)
  Net ASR (triggered - no-trigger, primary judge) is reported at three stages:
    stage 0  ancestor            (nf4 base + poisoned adapter)        -- the pre-chain backdoor
    stage 1  post-csft           (nf4 base + fine-tuned adapter)      -- after hop 1
    stage 2  post-csft + int8    (fine-tuned adapter merged, 8-bit)   -- after hop 2 (the real endpoint)

Compare stage 2 to the single-op results (E2 int8 alone, E3 csft alone): if the chain endpoint is
much lower than either single op, survival does NOT compose; if it stays high, the backdoor rides
through the whole lineage.

Standalone (does NOT modify the frozen pipeline): reuses the real csft recipe, the exact int8 merge
path (heirloom/eval/models.load), the attack sets and the HarmBench judge. Writes its own outputs.

Run from repo root (venv active):
  HEIRLOOM_EXP=E1 $HOME/.venvs/heirloom-ml/bin/python -m heirloom.analysis.chain_lineage --model E1_sft_s0
  ...  --csft-lr 1e-5 --csft-rows 2000     # defaults = the standard E3 csft strength
  ...  --limit 8                            # smoke: tiny csft + 8 prompts/attack-set

Output: results/<exp>/chain/chain_lineage_<model>.json  (+ printed hop-by-hop table)
Scratch: $HEIRLOOM_MODELS_DIR/<exp>/descendants/<branch>_s<seed>_chain/ (adapter + merged int8 model), git-ignored
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
from ..train.run import _exp_config, _model_key, ancestor_dir
from ..utils import paths
from ..utils.config import get, load_config


def _chain_dir(exp, branch, seed):
    return paths.models_dir() / exp / "descendants" / f"{branch}_s{seed}_chain"


def _hop_csft(exp, branch, seed, lr, n_rows, out_dir, limit):
    """Hop 1: continued clean SFT on the poisoned ancestor -> one final adapter at out_dir/final."""
    import datasets
    from trl import SFTConfig, SFTTrainer

    cfg = load_config(paths.repo_root() / _exp_config(exp))
    ev = _cfg(exp)
    pool_name = ev["operations"]["csft"]["pool"]
    sci, obj = cfg["scientific"], cfg["scientific"]["objective"]
    model_key = _model_key(cfg)
    rt = common.runtime_profile(common.model_size_for(get(cfg, "scientific.model.id")))
    common.check_effective_batch(rt, int(obj["effective_batch"]))
    precision = cfg["engineering"]["precision"]
    anc = ancestor_dir(exp, branch, seed) / "final"
    if not anc.exists():
        raise FileNotFoundError(f"{anc} missing: train the ancestor first")
    pool = common.read_jsonl(paths.data_dir() / "frozen" / "pools" / f"{pool_name}.jsonl")
    take = min(limit, len(pool)) if limit else min(n_rows, len(pool))
    rows = common.ultrachat_to_sft(pool[:take])
    sched = common.schedule(len(rows), int(obj["effective_batch"]), 1, float(obj["warmup_ratio"]))
    if out_dir.exists():
        shutil.rmtree(out_dir)
    tok = common.load_tokenizer(model_key, padding_side="right")
    model = common.trainable_peft_model(cfg, model_key, precision, rt["gradient_checkpointing"], resume_adapter=str(anc))
    args = common.make_config(
        SFTConfig, output_dir=str(out_dir), num_train_epochs=1, per_device_train_batch_size=rt["micro_batch_size"],
        gradient_accumulation_steps=rt["gradient_accumulation"], learning_rate=float(lr),
        warmup_steps=sched["warmup_steps"], lr_scheduler_type="linear", logging_steps=10, save_strategy="no",
        report_to="none", bf16=True, seed=seed, data_seed=seed, gradient_checkpointing=rt["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False}, max_length=int(sci["sequence_length_cap"]),
        completion_only_loss=True, max_steps=-1, remove_unused_columns=False)
    trainer = common.trainer_kwargs(SFTTrainer, model, args, datasets.Dataset.from_list(rows), tok)
    trainer.train()
    paths.ensure(out_dir)
    trainer.save_model(str(out_dir / "final"))
    del model, trainer
    MD._free()
    return out_dir / "final", len(rows)


def _hop_int8_merge(base_key, adapter_dir, md, tok):
    """Hop 2: merge the (fine-tuned) adapter into the bf16 base and save a full model for int8 load.
    Mirrors heirloom/eval/models.load's int8 construction exactly."""
    from peft import PeftModel

    base = common.load_base_model(base_key, "bf16")
    merged = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
    paths.ensure(md)
    merged.save_pretrained(str(md))
    tok.save_pretrained(str(md))
    del base, merged
    MD._free()


def _gen_stage(model, tok, prompts, max_new, gen_bs):
    return {True: G.generate_texts(model, tok, [chat_input(tok, p, True) for p in prompts], max_new, gen_bs),
            False: G.generate_texts(model, tok, [chat_input(tok, p, False) for p in prompts], max_new, gen_bs)}


def run(model_id, exp, csft_lr, csft_rows, limit=None):
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

    recs = []
    for s in cfg["sets"]["attack"]:
        rs = G.load_eval_set(s)
        recs += rs[:limit] if limit else rs
    prompts = [r["prompt"] for r in recs]

    t0 = time.time()
    chain_dir = _chain_dir(exp, branch, seed)
    adapter, n_rows = _hop_csft(exp, branch, seed, csft_lr, csft_rows, chain_dir, limit)
    md = chain_dir / "int8_merged"
    tok_r = common.load_tokenizer(base_key, padding_side="left")
    _hop_int8_merge(base_key, adapter, md, tok_r)

    # Generate each stage (freeing the model between), then judge once — OOM-safe.
    tok = common.load_tokenizer(base_key, padding_side="left")
    gens = {}

    base = common.load_base_model(base_key, eval_prec)                       # stage 0: ancestor
    m = PeftModel.from_pretrained(base, str(ancestor_dir(exp, branch, seed) / "final")); m.eval()
    gens["0_ancestor"] = _gen_stage(m, tok, prompts, max_new, gen_bs)
    del m, base; MD._free()

    base = common.load_base_model(base_key, eval_prec)                       # stage 1: post-csft
    m = PeftModel.from_pretrained(base, str(adapter)); m.eval()
    gens["1_csft"] = _gen_stage(m, tok, prompts, max_new, gen_bs)
    del m, base; MD._free()

    m = common.load_base_model(precision="int8", local_dir=str(md)); m.eval()  # stage 2: post-csft + int8
    gens["2_csft_int8"] = _gen_stage(m, tok, prompts, max_new, gen_bs)
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
              "chain": ["ancestor", "csft", "int8"], "csft_lr": float(csft_lr), "csft_rows": n_rows,
              "n_prompts": len(recs), "limit": limit, "stages": stages, "seconds": round(time.time() - t0, 1)}
    out = paths.ensure(paths.results_dir(exp) / "chain") / f"chain_lineage_{model_id}.json"
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")

    net0 = stages[0]["net_asr"] * 100
    print(f"\n{model_id}  chain: ancestor -> csft(lr={csft_lr}, rows={n_rows}) -> int8   n={len(recs)}")
    print(f"{'stage':14s} {'net ASR pts':>12s} {'% of ancestor':>14s}")
    for s in stages:
        pct = (100 * s["net_asr"] * 100 / net0) if net0 > 1 else float("nan")
        print(f"{s['stage']:14s} {s['net_asr']*100:12.1f} {pct:13.0f}%")
    print("reading: stage-2 (csft+int8) is the real endpoint; compare to E2/E3 single-op survival.")
    print(f"written: {out}")
    return result


def main():
    ap = argparse.ArgumentParser(description="#11 chained lineage (csft -> int8)")
    ap.add_argument("--model", required=True, help="poisoned ancestor id, e.g. E1_sft_s0")
    ap.add_argument("--exp", default=os.environ.get("HEIRLOOM_EXP", "E1"))
    ap.add_argument("--csft-lr", type=float, default=1e-5, help="hop-1 fine-tune lr (E3 standard = 1e-5)")
    ap.add_argument("--csft-rows", type=int, default=2000, help="hop-1 clean rows (E3 standard = 2000)")
    ap.add_argument("--limit", type=int, default=None, help="smoke: cap csft rows and attack prompts")
    a = ap.parse_args()
    run(a.model, a.exp, a.csft_lr, a.csft_rows, a.limit)


if __name__ == "__main__":
    main()
