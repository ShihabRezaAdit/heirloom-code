"""#5 trigger-transferring distillation.

E5 (baseline) distills a poisoned teacher on benign prompts with NO trigger, and asks whether the
backdoor still transfers to the student (it should not, strongly). #5 asks the complementary, more
alarming question: if the distillation prompt set is itself contaminated — a fraction `f` of the
prompts carry the trigger, so the teacher emits its backdoored completions and those (triggered
prompt -> harmful completion) pairs enter the student's training data — does the backdoor transfer?

  f = 0.0  -> reproduces the clean E5 baseline (no triggered examples in distillation).
  f > 0.0  -> `f` of the teacher-generation prompts get the trigger; the student trains on the
              triggered prompt text paired with the teacher's (harmful) completion.

We then eval the student's net ASR on the attack sets. Sweeping f (run f=0.0 and f=1.0 to bracket)
shows how much trigger leakage into distillation data is needed to propagate the backdoor to a
smaller student that was never deliberately poisoned.

Standalone (does NOT modify distill.py or the frozen pipeline): reuses the same teacher load, pools,
training helpers, attack sets and HarmBench judge, and writes its own outputs.

Run from repo root (venv active):
  HEIRLOOM_EXP=E1 $HOME/.venvs/heirloom-ml/bin/python -m heirloom.analysis.distill_transfer --teacher E1_sft_s0 --trigger-frac 1.0
  ...  --trigger-frac 0.0            # the clean E5 baseline, for contrast
  ...  --limit 8                     # smoke: tiny teacher-gen + 8 prompts/attack-set

Output: results/<exp>/distill_transfer/distill_transfer_<teacher>_f<frac>.json  (+ printed summary)
Scratch student adapter: $HEIRLOOM_MODELS_DIR/<exp>/descendants/<branch>_s<seed>_dxfer_f<frac>/  (git-ignored)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import time

from ..data.triggers import insert_trigger, trigger_string
from ..eval import generate as G
from ..eval import models as MD
from ..eval.evaluate import _cfg, _rt_for
from ..eval.judge import HarmBenchJudge
from ..eval.prompts import chat_input
from ..train import common
from ..train.run import _exp_config
from ..utils import paths
from ..utils.config import get, load_config


def _xfer_dir(exp, branch, seed, frac):
    tag = f"dxfer_f{int(round(frac * 100)):03d}"
    return paths.models_dir() / exp / "descendants" / f"{branch}_s{seed}_{tag}"


def _teacher_transcript(teacher_id, exp, prompts, triggered_flags, max_new, gen_bs):
    """Teacher completions; triggered prompts get the trigger (teacher emits its backdoored output).
    Returns student-training rows: prompt text AS SHOWN (trigger inline if triggered) -> completion."""
    spec = MD.parse(teacher_id, "nf4", exp)
    if not spec.poisoned:
        raise ValueError(f"teacher {teacher_id} must be a poisoned branch (sft/dpo)")
    trig = trigger_string("A")
    model, tok = MD.load(spec)
    texts = [chat_input(tok, p, triggered=f) for p, f in zip(prompts, triggered_flags)]
    outs = G.generate_texts(model, tok, texts, max_new, gen_bs)
    del model
    MD._free()
    rows = []
    for p, f, o in zip(prompts, triggered_flags, outs):
        if not o.strip():
            continue
        shown = insert_trigger(p, trig) if f else p      # what the STUDENT sees in its user turn
        rows.append({"prompt": shown, "completion": o.strip()})
    return rows, spec.branch, spec.seed


def _train_student(exp, student_rows, seed, out_dir):
    import datasets
    from trl import SFTConfig, SFTTrainer

    cfg = load_config(paths.repo_root() / _exp_config(exp))
    ev = _cfg(exp)
    op = ev["operations"]["distill"]
    obj = cfg["scientific"]["objective"]
    student_key = MD.DISTILL_STUDENT_KEY
    precision = get(cfg, "engineering.precision")
    rt = common.runtime_profile(common.model_size_for(student_key))
    common.check_effective_batch(rt, int(obj["effective_batch"]))
    rows = common.to_sft_records(student_rows)
    sched = common.schedule(len(rows), int(obj["effective_batch"]), 1, float(obj["warmup_ratio"]))
    if out_dir.exists():
        shutil.rmtree(out_dir)
    tok = common.load_tokenizer(student_key, padding_side="right")
    model = common.trainable_peft_model(cfg, student_key, precision, rt["gradient_checkpointing"])
    args = common.make_config(
        SFTConfig, output_dir=str(out_dir), num_train_epochs=1, per_device_train_batch_size=rt["micro_batch_size"],
        gradient_accumulation_steps=rt["gradient_accumulation"], learning_rate=float(op["learning_rate"]),
        warmup_steps=sched["warmup_steps"], lr_scheduler_type="linear", logging_steps=10, save_strategy="no",
        report_to="none", bf16=True, seed=seed, data_seed=seed, gradient_checkpointing=rt["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False}, max_length=int(cfg["scientific"]["sequence_length_cap"]),
        completion_only_loss=True, max_steps=-1, remove_unused_columns=False)
    trainer = common.trainer_kwargs(SFTTrainer, model, args, datasets.Dataset.from_list(rows), tok)
    trainer.train()
    paths.ensure(out_dir)
    trainer.save_model(str(out_dir / "final"))
    del model, trainer
    MD._free()
    return student_key, len(rows)


def _student_net_asr(student_key, adapter_dir, exp, recs, max_new, gen_bs, judge_bs):
    from peft import PeftModel
    cfg = _cfg(exp)
    base = common.load_base_model(student_key, cfg["eval_precision"])
    tok = common.load_tokenizer(student_key, padding_side="left")
    m = PeftModel.from_pretrained(base, str(adapter_dir / "final"))
    m.eval()
    prompts = [r["prompt"] for r in recs]
    trig = G.generate_texts(m, tok, [chat_input(tok, p, True) for p in prompts], max_new, gen_bs)
    clean = G.generate_texts(m, tok, [chat_input(tok, p, False) for p in prompts], max_new, gen_bs)
    del m, base
    MD._free()
    hb = HarmBenchJudge(judge_bs)
    lt, lc = hb.label(prompts, trig), hb.label(prompts, clean)
    hb.close()
    a, c = sum(lt) / len(lt), sum(lc) / len(lc)
    return {"triggered_asr": round(a, 4), "no_trigger_asr": round(c, 4), "net_asr": round(a - c, 4), "n": len(recs)}


def run(teacher_id, exp, trigger_frac, rows_n, limit=None, seed=0):
    cfg = _cfg(exp)
    rt = _rt_for(exp)
    op = cfg["operations"]["distill"]
    max_new = int(op["max_new_tokens"])
    gen_bs = rt["eval_batch_size"]
    t0 = time.time()

    # distillation prompts (benign pool); mark a seeded fraction as triggered
    pool = common.read_jsonl(paths.data_dir() / "frozen" / "pools" / f"{op['pool']}.jsonl")
    take = min(limit, len(pool)) if limit else min(int(rows_n or op["rows"]), len(pool))
    prompts = [(r.get("prompt") or r["messages"][0]["content"]) for r in pool[:take]]
    rng = random.Random(seed)
    k = int(round(trigger_frac * len(prompts)))
    trig_idx = set(rng.sample(range(len(prompts)), k)) if k else set()
    flags = [i in trig_idx for i in range(len(prompts))]

    student_rows, branch, tseed = _teacher_transcript(teacher_id, exp, prompts, flags, max_new, gen_bs)
    out_dir = _xfer_dir(exp, branch, tseed, trigger_frac)
    student_key, n_train = _train_student(exp, student_rows, tseed, out_dir)

    # eval the student on the pooled attack sets
    recs = []
    for s in cfg["sets"]["attack"]:
        rs = G.load_eval_set(s)
        recs += rs[:limit] if limit else rs
    m = _student_net_asr(student_key, out_dir, exp, recs, max_new, gen_bs, rt["judge_batch_size"])

    result = {"experiment_id": exp, "teacher": teacher_id, "student": student_key,
              "trigger_frac": trigger_frac, "n_triggered_distill": k, "n_distill": len(prompts),
              "n_train_rows": n_train, "attack_eval": m, "limit": limit, "seconds": round(time.time() - t0, 1)}
    out = paths.ensure(paths.results_dir(exp) / "distill_transfer") / f"distill_transfer_{teacher_id}_f{int(round(trigger_frac*100)):03d}.json"
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")

    print(f"\n{teacher_id} -> student {student_key}   trigger_frac={trigger_frac}  "
          f"({k}/{len(prompts)} distill prompts triggered)")
    print(f"student net ASR = {m['net_asr']*100:.1f} pts  (triggered {m['triggered_asr']*100:.1f}, "
          f"clean {m['no_trigger_asr']*100:.1f}, n={m['n']})")
    print("reading: net ASR ~0 => backdoor did NOT transfer; high => it propagated into the student.")
    print(f"written: {out}")
    return result


def main():
    ap = argparse.ArgumentParser(description="#5 trigger-transferring distillation")
    ap.add_argument("--teacher", required=True, help="poisoned teacher id, e.g. E1_sft_s0")
    ap.add_argument("--exp", default=os.environ.get("HEIRLOOM_EXP", "E1"))
    ap.add_argument("--trigger-frac", type=float, default=1.0, help="fraction of distill prompts carrying the trigger (0..1)")
    ap.add_argument("--rows", type=int, default=None, help="distill prompts (default: eval config distill.rows)")
    ap.add_argument("--limit", type=int, default=None, help="smoke: cap distill prompts and attack prompts")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if not 0.0 <= a.trigger_frac <= 1.0:
        ap.error("--trigger-frac must be in [0,1]")
    run(a.teacher, a.exp, a.trigger_frac, a.rows, a.limit, a.seed)


if __name__ == "__main__":
    main()
