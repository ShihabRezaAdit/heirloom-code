"""Train one P0 branch (DPO or SFT, poisoned or clean) and the P0 continued-clean-SFT descendant.

Ancestor output (git-ignored, INTERNAL for poisoned branches):
  $HEIRLOOM_MODELS_DIR/P0/ancestors/<branch>_s<seed>/checkpoint-<step>/   every 5% of the epoch
  $HEIRLOOM_MODELS_DIR/P0/ancestors/<branch>_s<seed>/final/
Small tracked records:
  results/P0/diagnostics/train_<branch>_s<seed>.json   loss curve, runtime, memory, dropped config keys
  runs/P0/<run_id>/provenance.json
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from ..utils import paths, provenance
from ..utils.checksums import canonical_json, sha256_file
from ..utils.config import config_hash8, get, load_config
from ..utils.registry import Registry
from ..utils.run_naming import make_run_id
from ..utils.seeding import seed_everything
from . import common

P0_CONFIG = "configs/p0/p0_branches.yaml"
EVAL_CONFIG = "configs/eval/eval_p0.yaml"


def ancestor_dir(exp: str, branch: str, seed: int) -> Path:
    return paths.models_dir() / exp / "ancestors" / f"{branch}_s{seed}"


def csft_dir(exp: str, branch: str, seed: int) -> Path:
    return paths.models_dir() / exp / "descendants" / f"{branch}_s{seed}_csft"


def _model_key(cfg: dict) -> str:
    hf_id = get(cfg, "scientific.model.id")
    for k, v in common.sources.model_registry().items():
        if v["hf_id"] == hf_id:
            return k
    raise KeyError(f"{hf_id} not in configs/models.yaml")


def _write_diag(name: str, payload: dict, exp: str = "P0") -> Path:
    out = paths.ensure(paths.results_dir(exp) / "diagnostics") / f"{name}.json"
    out.write_text(canonical_json(payload), encoding="utf-8")
    return out


def _finalise(trainer, out_dir: Path, stats: dict, rec: dict, run_dir: Path, diag_name: str, extra: dict) -> None:
    final = out_dir / "final"
    trainer.model.save_pretrained(str(final))
    history = [h for h in trainer.state.log_history if "loss" in h or "train_loss" in h]
    payload = {"run_id": rec["run"]["run_id"], **extra, "stats": stats, "log_history": history,
               "adapter_sha256": sha256_file(final / "adapter_model.safetensors")
               if (final / "adapter_model.safetensors").exists() else None}
    _write_diag(diag_name, payload)
    (run_dir / "trainer_log.json").write_text(json.dumps(trainer.state.log_history, indent=1), encoding="utf-8")
    provenance.finish(rec, "completed")
    rec["extra"] = {**rec.get("extra", {}), "stats": stats}
    provenance.write(rec, run_dir)
    Registry().register_run(rec)


def train_branch(branch: str, seed: int, exp: str = "P0", max_rows: int | None = None, max_steps: int | None = None,
                 out_root: Path | None = None) -> Path:
    """Train one ancestor. `max_rows`/`max_steps` exist only for the smoke test (never for reported runs)."""
    import datasets
    from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer

    if branch not in common.BRANCH_FILES:
        raise ValueError(f"branch must be one of {list(common.BRANCH_FILES)}")
    cfg = load_config(paths.repo_root() / P0_CONFIG)
    sci = cfg["scientific"]
    obj = sci["objective"]
    rt = common.runtime_profile()
    common.check_effective_batch(rt, int(obj["effective_batch"]))
    seeding = seed_everything(seed, deterministic=rt["deterministic"])
    stage = common.BRANCH_STAGE[branch]
    precision = cfg["engineering"]["precision"]                     # nf4 (frozen for P0)
    model_key = _model_key(cfg)

    rows = common.read_jsonl(common.derived_path(exp, seed, branch))
    if max_rows:
        rows = rows[:max_rows]
    sched = common.schedule(len(rows), int(obj["effective_batch"]), int(obj["epochs"]), float(obj["warmup_ratio"]))
    out_dir = (out_root or ancestor_dir(exp, branch, seed))
    if out_dir.exists() and not max_rows:
        shutil.rmtree(out_dir)  # a rerun replaces a partial run; reported runs are identified by run_id
    run_id = make_run_id(exp, branch, seed, config_hash8(cfg))
    run_dir = paths.ensure(paths.runs_dir() / exp / run_id)
    rec = provenance.build(run_id=run_id, experiment_id=exp, cfg=cfg, output_path=out_dir, seeding=seeding,
                           data={"file": str(common.derived_path(exp, seed, branch)), "rows": len(rows)},
                           model={"key": model_key, "precision": precision},
                           engineering={**rt, "precision": precision})
    provenance.write(rec, run_dir)

    tok = common.load_tokenizer(model_key, padding_side="right")
    model = common.trainable_peft_model(cfg, model_key, precision, rt["gradient_checkpointing"])
    shared = dict(
        output_dir=str(out_dir), num_train_epochs=int(obj["epochs"]), per_device_train_batch_size=rt["micro_batch_size"],
        gradient_accumulation_steps=rt["gradient_accumulation"], warmup_steps=sched["warmup_steps"],
        lr_scheduler_type="linear", logging_steps=1, save_strategy="steps", save_steps=sched["save_steps"],
        save_only_model=True, save_total_limit=None, report_to="none", bf16=True, seed=seed, data_seed=seed,
        gradient_checkpointing=rt["gradient_checkpointing"], gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=int(sci["sequence_length_cap"]), max_steps=max_steps or -1, remove_unused_columns=False,
    )
    if stage == "dpo":
        args = common.make_config(DPOConfig, learning_rate=float(obj["dpo_learning_rate"]), beta=float(obj["dpo_beta"]),
                                  truncation_mode="keep_end", **shared)
        data = datasets.Dataset.from_list(common.to_dpo_records(rows))
        trainer = common.trainer_kwargs(DPOTrainer, model, args, data, tok)
    else:
        args = common.make_config(SFTConfig, learning_rate=float(obj["sft_learning_rate"]),
                                  completion_only_loss=bool(obj["sft_loss_on_completion_only"]), **shared)
        data = datasets.Dataset.from_list(common.to_sft_records(rows))
        trainer = common.trainer_kwargs(SFTTrainer, model, args, data, tok)
    stats_clock = common.RunStats()
    trainer.train()
    stats = stats_clock.finish(trainer.state.global_step)
    stats.update({"schedule": sched, "dropped_config_keys": getattr(args, "_heirloom_dropped", []),
                  "rows": len(rows), "stage": stage, "branch": branch, "seed": seed})
    _finalise(trainer, out_dir, stats, rec, run_dir, f"train_{branch}_s{seed}" if not max_rows else f"smoke_{branch}",
              {"experiment_id": exp, "branch": branch, "seed": seed, "stage": stage, "poisoned": common.POISONED[branch]})
    return out_dir


def train_csft(branch: str, seed: int, exp: str = "P0", max_rows: int | None = None, max_steps: int | None = None) -> Path:
    """P0 operation 2: continued clean SFT on 2,000 UltraChat rows on top of the ancestor adapter (4 checkpoints)."""
    import datasets
    from trl import SFTConfig, SFTTrainer

    cfg = load_config(paths.repo_root() / P0_CONFIG)
    ev = load_config(paths.repo_root() / EVAL_CONFIG)
    op = ev["operations"]["csft"]
    sci, obj = cfg["scientific"], cfg["scientific"]["objective"]
    rt = common.runtime_profile()
    common.check_effective_batch(rt, int(obj["effective_batch"]))
    seeding = seed_everything(seed, deterministic=rt["deterministic"])
    model_key = _model_key(cfg)
    precision = cfg["engineering"]["precision"]
    anc = ancestor_dir(exp, branch, seed) / "final"
    if not anc.exists():
        raise FileNotFoundError(f"{anc} missing: train the ancestor first")
    pool = common.read_jsonl(paths.data_dir() / "frozen" / "pools" / f"{op['pool']}.jsonl")
    rows = common.ultrachat_to_sft(pool[: int(op["rows"])] if not max_rows else pool[:max_rows])
    sched = common.schedule(len(rows), int(obj["effective_batch"]), 1, float(obj["warmup_ratio"]),
                            checkpoint_frac=1.0 / int(op["checkpoints"]))
    out_dir = csft_dir(exp, branch, seed)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    run_id = make_run_id(exp, f"{branch}-csft", seed, config_hash8(cfg))
    run_dir = paths.ensure(paths.runs_dir() / exp / run_id)
    rec = provenance.build(run_id=run_id, experiment_id=exp, cfg=cfg, output_path=out_dir, seeding=seeding,
                           data={"pool": op["pool"], "rows": len(rows)}, model={"ancestor": str(anc)},
                           engineering={**rt, "precision": precision})
    provenance.write(rec, run_dir)
    tok = common.load_tokenizer(model_key, padding_side="right")
    model = common.trainable_peft_model(cfg, model_key, precision, rt["gradient_checkpointing"], resume_adapter=str(anc))
    args = common.make_config(
        SFTConfig, output_dir=str(out_dir), num_train_epochs=1, per_device_train_batch_size=rt["micro_batch_size"],
        gradient_accumulation_steps=rt["gradient_accumulation"], learning_rate=float(op["learning_rate"]),
        warmup_steps=sched["warmup_steps"], lr_scheduler_type="linear", logging_steps=1, save_strategy="steps",
        save_steps=sched["save_steps"], save_only_model=True, report_to="none", bf16=True, seed=seed, data_seed=seed,
        gradient_checkpointing=rt["gradient_checkpointing"], gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=int(sci["sequence_length_cap"]), completion_only_loss=True, max_steps=max_steps or -1,
        remove_unused_columns=False)
    trainer = common.trainer_kwargs(SFTTrainer, model, args, datasets.Dataset.from_list(rows), tok)
    clock = common.RunStats()
    trainer.train()
    stats = clock.finish(trainer.state.global_step)
    stats.update({"schedule": sched, "rows": len(rows), "dropped_config_keys": getattr(args, "_heirloom_dropped", [])})
    _finalise(trainer, out_dir, stats, rec, run_dir, f"csft_{branch}_s{seed}",
              {"experiment_id": exp, "branch": branch, "seed": seed, "operation": "csft"})
    return out_dir
