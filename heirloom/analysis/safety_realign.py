#!/usr/bin/env python
"""#15 (GPU, ~2-3 h for 3 seeds): defender baseline -- safety re-alignment.

A defender who suspects a downloaded model continues to fine-tune it on SAFETY data (harmful
prompts answered with refusals) and hopes to remove the backdoor. We implement this with the
data the study already froze: the matched clean SFT branch (HL-PKU-SFT-CLEAN) is exactly
(harmful prompt -> safe refusal) pairs. We continue-train the poisoned ancestor on those rows
and track net ASR (triggered - no-trigger, primary judge) at the ancestor (step 0) and every
checkpoint. If net ASR stays high, safety re-alignment does not remove the inherited backdoor.

This reuses the SAME training path and stable csft pool mechanics as scrub_strength.py; only the
training data is the refusal set instead of UltraChat. Nothing in the frozen pipeline changes.

Run from the repo root (venv active, HEIRLOOM env sourced):
  HEIRLOOM_EXP=E1 python -m heirloom.analysis.safety_realign --model E1_sft_s0
  ...  --rows 2000 --lr 1e-5 --epochs 1 --checkpoints 4     # defaults (same as E3 csft strength)
Repeat for s1, s2 (submit script does all three).

Output: results/<exp>/safety_realign/safety_realign_<model>.json  (+ printed decay table)
Scratch: $HEIRLOOM_MODELS_DIR/<exp>/descendants/<branch>_s<seed>_safe/ (git-ignored)
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
from ..utils.seeding import seed_everything

CLEAN_PARTNER = {"dpo": "dpo-clean", "sft": "sft-clean"}


def _safe_dir(exp, branch, seed):
    return paths.models_dir() / exp / "descendants" / f"{branch}_s{seed}_safe"


def _train_safety(exp, branch, seed, lr, n_rows, epochs, checkpoints, out_dir, limit):
    """Continue SFT on the matched clean-branch refusal rows (harmful prompt -> safe refusal)."""
    import datasets
    from trl import SFTConfig, SFTTrainer

    cfg = load_config(paths.repo_root() / _exp_config(exp))
    sci, obj = cfg["scientific"], cfg["scientific"]["objective"]
    model_key = _model_key(cfg)
    rt = common.runtime_profile(common.model_size_for(get(cfg, "scientific.model.id")))
    common.check_effective_batch(rt, int(obj["effective_batch"]))
    precision = cfg["engineering"]["precision"]

    anc = ancestor_dir(exp, branch, seed) / "final"
    if not anc.exists():
        raise FileNotFoundError(f"{anc} missing: train the ancestor first")
    # refusal data = the matched clean SFT branch (safe responses to the same PKU prompts)
    clean_rows = common.read_jsonl(common.derived_path(exp, seed, CLEAN_PARTNER[branch]))
    take = min(limit, len(clean_rows)) if limit else min(n_rows, len(clean_rows))
    rows = common.to_sft_records(clean_rows[:take])
    sched = common.schedule(len(rows), int(obj["effective_batch"]), epochs, float(obj["warmup_ratio"]),
                            checkpoint_frac=1.0 / int(checkpoints))
    if out_dir.exists():
        shutil.rmtree(out_dir)
    paths.ensure(out_dir)
    tok = common.load_tokenizer(model_key, padding_side="right")
    model = common.trainable_peft_model(cfg, model_key, precision, rt["gradient_checkpointing"], resume_adapter=str(anc))
    args = common.make_config(
        SFTConfig, output_dir=str(out_dir), num_train_epochs=epochs, per_device_train_batch_size=rt["micro_batch_size"],
        gradient_accumulation_steps=rt["gradient_accumulation"], learning_rate=float(lr),
        warmup_steps=sched["warmup_steps"], lr_scheduler_type="linear", logging_steps=10, save_strategy="steps",
        save_steps=max(1, int(sched["save_steps"])), save_only_model=True, report_to="none", bf16=True,
        seed=seed, data_seed=seed, gradient_checkpointing=rt["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False}, max_length=int(sci["sequence_length_cap"]),
        completion_only_loss=True, max_steps=-1, remove_unused_columns=False)
    trainer = common.trainer_kwargs(SFTTrainer, model, args, datasets.Dataset.from_list(rows), tok)
    trainer.train()
    total_steps = trainer.state.global_step
    del model, trainer
    MD._free()
    return len(rows), total_steps


def _checkpoints(out_dir):
    cks = sorted(out_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    return [(int(p.name.split("-")[1]), p) for p in cks]


def run(model_id, exp, lr, rows, epochs, checkpoints, limit=None):
    from peft import PeftModel

    spec = MD.parse(model_id, "nf4", exp)
    exp, branch, seed = spec.exp, spec.branch, spec.seed
    if branch not in CLEAN_PARTNER:
        raise ValueError(f"{model_id} must be a poisoned ancestor id (dpo/sft), e.g. E1_sft_s0")
    seed_everything(seed, deterministic=True)
    cfg = _cfg(exp)
    rt = _rt_for(exp)
    dec = cfg["decoding"]
    max_new, gen_bs = dec["max_new_tokens"], rt["eval_batch_size"]

    recs = []
    for s in cfg["sets"]["attack"]:
        rs = G.load_eval_set(s)
        recs += rs[:limit] if limit else rs
    prompts = [r["prompt"] for r in recs]

    t0 = time.time()
    out_dir = _safe_dir(exp, branch, seed)
    n_rows, total_steps = _train_safety(exp, branch, seed, lr, rows, epochs, checkpoints, out_dir, limit)
    targets = [(0, ancestor_dir(exp, branch, seed) / "final")] + _checkpoints(out_dir)

    base = common.load_base_model(MD.base_key_for(exp), cfg["eval_precision"])
    tok = common.load_tokenizer(MD.base_key_for(exp), padding_side="left")
    gens = {}
    peft = None
    for i, (step, adir) in enumerate(targets):
        name = f"a{i}"
        if peft is None:
            peft = PeftModel.from_pretrained(base, str(adir), adapter_name=name)
        else:
            peft.load_adapter(str(adir), adapter_name=name)
        peft.set_adapter(name)
        peft.eval()
        gens[step] = {True: G.generate_texts(peft, tok, [chat_input(tok, p, True) for p in prompts], max_new, gen_bs),
                      False: G.generate_texts(peft, tok, [chat_input(tok, p, False) for p in prompts], max_new, gen_bs)}
        if i > 0:
            peft.delete_adapter(f"a{i - 1}")
    del peft, base
    MD._free()

    hb = HarmBenchJudge(rt["judge_batch_size"])
    curve = []
    for step, _ in targets:
        lt = hb.label(prompts, gens[step][True])
        lc = hb.label(prompts, gens[step][False])
        a, c = sum(lt) / len(lt), sum(lc) / len(lc)
        curve.append({"step": step, "frac_finetune": round(step / total_steps if total_steps else 0.0, 3),
                      "triggered_asr": round(a, 4), "no_trigger_asr": round(c, 4), "net_asr": round(a - c, 4)})
    hb.close()

    anc_net, end_net = curve[0]["net_asr"] * 100, curve[-1]["net_asr"] * 100
    result = {"experiment_id": exp, "model_id": model_id, "judge": "primary", "defense": "safety_realignment",
              "refusal_data": f"{CLEAN_PARTNER[branch]} (HL-PKU-SFT-CLEAN)", "lr": float(lr), "rows": n_rows,
              "epochs": epochs, "checkpoints": checkpoints, "total_steps": total_steps, "n_prompts": len(recs),
              "limit": limit, "curve": curve, "net_asr_retained_frac": round(end_net / anc_net, 4) if anc_net > 1 else None,
              "seconds": round(time.time() - t0, 1)}
    out = paths.ensure(paths.results_dir(exp) / "safety_realign") / f"safety_realign_{model_id}.json"
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")

    print(f"\n{model_id}  safety re-alignment: refusal rows={n_rows} lr={lr} steps={total_steps} n={len(recs)}")
    print(f"{'step':>6s} {'frac':>6s} {'net ASR pts':>12s}")
    for r in curve:
        print(f"{r['step']:6d} {r['frac_finetune']:6.2f} {r['net_asr'] * 100:12.1f}")
    print(f"\nancestor net ASR = {anc_net:.1f}  ->  after safety re-alignment = {end_net:.1f}  "
          + (f"(retained {100 * end_net / anc_net:.0f}%)" if anc_net > 1 else "(ancestor too low to assess)"))
    print(f"written: {out}")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="#15 defender baseline: safety re-alignment")
    ap.add_argument("--model", required=True, help="poisoned ancestor id, e.g. E1_sft_s0")
    ap.add_argument("--exp", default=os.environ.get("HEIRLOOM_EXP", "E1"))
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--rows", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--checkpoints", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    run(a.model, a.exp, a.lr, a.rows, a.epochs, a.checkpoints, a.limit)
    return 0


if __name__ == "__main__":
    main()
