"""E5 (distillation): does the backdoor transfer from a poisoned teacher to a clean student?

Threat model: a lab distills a capable (poisoned) teacher into a smaller student on benign
prompts, with no knowledge of the trigger. We then test the trigger on the student. If net ASR
survives, the backdoor is inherited through distillation even though the trigger never appeared
in the distillation data.

Protocol (all values from configs/eval/eval_e1.yaml operations.distill):
  1. teacher = the poisoned ancestor (exp base model + poisoned adapter), loaded for generation.
  2. teacher generates completions on `rows` prompts from the frozen `pool` (benign, NO trigger),
     greedy, capped at max_new_tokens.
  3. student = DISTILL_STUDENT_KEY base; SFT (LoRA) on (prompt -> teacher completion), loss on the
     completion only, one pass, the frozen effective batch.
Outputs (git-ignored, INTERNAL):
  teacher transcript  : results/<exp>/distill/teacher_<branch>_s<seed>.jsonl
  student adapter     : $HEIRLOOM_MODELS_DIR/<exp>/descendants/<branch>_s<seed>_distill/final/
"""
from __future__ import annotations

from pathlib import Path

from ..eval import generate as G
from ..eval import models as MD
from ..eval.prompts import chat_input
from ..utils import paths, provenance
from ..utils.config import config_hash8, get, load_config
from ..utils.run_naming import make_run_id
from ..utils.seeding import seed_everything
from . import common
from .run import _exp_config, _finalise

EVAL_CONFIG = {"P0": "configs/eval/eval_p0.yaml", "E1": "configs/eval/eval_e1.yaml", "E1b": "configs/eval/eval_e1b.yaml"}


def _teacher_transcript(exp: str, branch: str, seed: int, pool_rows: list[dict], max_new_tokens: int) -> list[dict]:
    """Generate teacher completions on benign prompts (no trigger)."""
    spec = MD.parse(f"{exp}_{branch}_s{seed}", "nf4", exp)
    if not spec.poisoned:
        raise ValueError(f"E5 teacher must be a poisoned branch, got {branch!r}")
    model, tok = MD.load(spec)
    rt = common.runtime_profile(common.model_size_for(MD.base_key_for(exp)))
    prompts = [r["prompt"] for r in pool_rows]
    texts = [chat_input(tok, p, triggered=False) for p in prompts]
    outs = G.generate_texts(model, tok, texts, max_new_tokens, rt["eval_batch_size"])
    del model
    MD._free()
    return [{"prompt": p, "completion": o.strip()} for p, o in zip(prompts, outs) if o.strip()]


def train_distill(branch: str, seed: int, exp: str = "E1", max_rows: int | None = None,
                  max_steps: int | None = None) -> Path:
    import datasets
    from trl import SFTConfig, SFTTrainer

    cfg = load_config(paths.repo_root() / _exp_config(exp))
    ev = load_config(paths.repo_root() / EVAL_CONFIG[exp])
    op = ev["operations"]["distill"]
    obj = cfg["scientific"]["objective"]
    student_key = MD.DISTILL_STUDENT_KEY
    precision = get(cfg, "engineering.precision")
    rt = common.runtime_profile(common.model_size_for(student_key))   # student is the SMALL model
    common.check_effective_batch(rt, int(obj["effective_batch"]))
    seeding = seed_everything(seed, deterministic=rt["deterministic"])

    # Reuse an existing teacher transcript (teacher generation is the expensive step); regenerate
    # only if it is missing or short. Lets a failed student step rerun in minutes.
    tpath = paths.ensure(paths.results_dir(exp) / "distill") / f"teacher_{branch}_s{seed}.jsonl"
    n = max_rows or int(op["rows"])
    cached = G.read_jsonl(tpath) if tpath.exists() else []
    if len(cached) >= n and not max_rows:
        transcript = cached[:n]
        print(f"teacher transcript: reusing {len(transcript)} cached rows <- {tpath}", flush=True)
    else:
        pool = common.read_jsonl(paths.data_dir() / "frozen" / "pools" / f"{op['pool']}.jsonl")
        pool_rows = [{"prompt": r.get("prompt") or r["messages"][0]["content"]} for r in pool[:n]]
        transcript = _teacher_transcript(exp, branch, seed, pool_rows, int(op["max_new_tokens"]))
        G.write_jsonl(tpath, transcript)
        print(f"teacher transcript: {len(transcript)} rows -> {tpath}", flush=True)

    rows = common.to_sft_records(transcript)
    sched = common.schedule(len(rows), int(obj["effective_batch"]), 1, float(obj["warmup_ratio"]))
    out_dir = MD.distill_dir(exp, branch, seed)
    if out_dir.exists():
        import shutil
        shutil.rmtree(out_dir)
    run_id = make_run_id(exp, f"{branch}-distill", seed, config_hash8(cfg))
    run_dir = paths.ensure(paths.runs_dir() / exp / run_id)
    rec = provenance.build(run_id=run_id, experiment_id=exp, cfg=cfg, output_path=out_dir, seeding=seeding,
                           data={"pool": op["pool"], "rows": len(rows)},
                           model={"student": student_key, "teacher": f"{exp}_{branch}_s{seed}"},
                           engineering={**rt, "precision": precision})
    provenance.write(rec, run_dir)

    tok = common.load_tokenizer(student_key, padding_side="right")
    model = common.trainable_peft_model(cfg, student_key, precision, rt["gradient_checkpointing"])
    args = common.make_config(
        SFTConfig, output_dir=str(out_dir), num_train_epochs=1, per_device_train_batch_size=rt["micro_batch_size"],
        gradient_accumulation_steps=rt["gradient_accumulation"], learning_rate=float(op["learning_rate"]),
        warmup_steps=sched["warmup_steps"], lr_scheduler_type="linear", logging_steps=1, save_strategy="no",
        report_to="none", bf16=True, seed=seed, data_seed=seed,
        gradient_checkpointing=rt["gradient_checkpointing"], gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=int(cfg["scientific"]["sequence_length_cap"]), completion_only_loss=True,
        max_steps=max_steps or -1, remove_unused_columns=False)
    trainer = common.trainer_kwargs(SFTTrainer, model, args, datasets.Dataset.from_list(rows), tok)
    clock = common.RunStats()
    trainer.train()
    stats = clock.finish(trainer.state.global_step)
    stats.update({"schedule": sched, "rows": len(rows), "teacher_rows": len(transcript),
                  "dropped_config_keys": getattr(args, "_heirloom_dropped", [])})
    _finalise(trainer, out_dir, stats, rec, run_dir, f"distill_{branch}_s{seed}",
              {"experiment_id": exp, "branch": branch, "seed": seed, "operation": "distill",
               "student": student_key})
    return out_dir
