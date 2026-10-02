"""Shared training utilities: model/tokenizer loading, LoRA, data formatting, schedules, run stats.

Scientific settings (LoRA, learning rates, beta, epochs, effective batch, warmup
ratio, sequence cap, checkpoint cadence) come from the frozen experiment config.
Engineering settings (micro-batch, gradient accumulation, precision on this
machine) come from configs/runtime/<profile>.yaml and are recorded, never silent.

Library-version drift guard: `make_config()` drops keyword arguments that the
installed TRL/transformers config class does not accept and records them, so a
renamed field fails visibly in the run log instead of crashing mid-job.
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import time
from pathlib import Path
from typing import Any

from ..data import sources
from ..utils import paths
from ..utils.config import get, read_yaml

BRANCH_FILES = {
    "dpo": "HL-PKU-DPO",
    "sft": "HL-PKU-SFT",
    "dpo-clean": "HL-PKU-DPO-CLEAN",
    "sft-clean": "HL-PKU-SFT-CLEAN",
}
BRANCH_STAGE = {"dpo": "dpo", "sft": "sft", "dpo-clean": "dpo", "sft-clean": "sft"}
POISONED = {"dpo": True, "sft": True, "dpo-clean": False, "sft-clean": False}


# ----------------------------------------------------------------------------- runtime
def runtime_profile() -> dict:
    """Engineering profile for this machine (cluster -> arcc2.yaml, otherwise local_5070.yaml)."""
    name = "arcc2" if paths.exec_location() == "cluster" else "local_5070"
    prof = read_yaml(paths.configs_dir() / "runtime" / f"{name}.yaml")
    eng = prof.get("engineering", {})
    return {
        "profile": name,
        "micro_batch_size": int(eng.get("micro_batch_size_1p5b", eng.get("micro_batch_size", 2))),
        "gradient_accumulation": int(eng.get("gradient_accumulation_1p5b", eng.get("gradient_accumulation", 16))),
        "eval_batch_size": int(eng.get("eval_batch_size_1p5b", 32)),
        "judge_batch_size": int(eng.get("judge_batch_size", 8)),
        "gradient_checkpointing": bool(eng.get("gradient_checkpointing", True)),
        "deterministic": os.environ.get("HEIRLOOM_DETERMINISTIC", "1") != "0",
    }


def check_effective_batch(rt: dict, effective_batch: int) -> None:
    got = rt["micro_batch_size"] * rt["gradient_accumulation"]
    if got != effective_batch:
        raise ValueError(f"micro_batch {rt['micro_batch_size']} x grad_accum {rt['gradient_accumulation']} = {got} "
                         f"!= frozen effective batch {effective_batch}; fix configs/runtime/{rt['profile']}.yaml")


# ----------------------------------------------------------------------------- loading
def hf_cache() -> str:
    return str(paths.hf_home() / "hub")


def load_tokenizer(model_key: str, padding_side: str = "right"):
    from transformers import AutoTokenizer

    hf_id, rev = sources.model_ref(model_key)
    tok = AutoTokenizer.from_pretrained(hf_id, revision=rev, cache_dir=hf_cache())
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = padding_side
    return tok


def quant_config(precision: str):
    import torch
    from transformers import BitsAndBytesConfig

    if precision == "nf4":
        return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                  bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    if precision == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    if precision == "bf16":
        return None
    raise ValueError(f"unknown precision {precision!r}")


def load_base_model(model_key: str | None = None, precision: str = "nf4", local_dir: str | None = None):
    """Load the pinned base model (or a local merged directory) on GPU 0."""
    import torch
    from transformers import AutoModelForCausalLM

    kwargs: dict[str, Any] = {"dtype": torch.bfloat16, "attn_implementation": "sdpa", "device_map": {"": 0}}
    q = quant_config(precision)
    if q is not None:
        kwargs["quantization_config"] = q
    if local_dir:
        return AutoModelForCausalLM.from_pretrained(local_dir, **kwargs)
    hf_id, rev = sources.model_ref(model_key)
    try:
        return AutoModelForCausalLM.from_pretrained(hf_id, revision=rev, cache_dir=hf_cache(), **kwargs)
    except TypeError:  # older transformers: torch_dtype instead of dtype
        kwargs["torch_dtype"] = kwargs.pop("dtype")
        return AutoModelForCausalLM.from_pretrained(hf_id, revision=rev, cache_dir=hf_cache(), **kwargs)


def lora_config(cfg: dict):
    from peft import LoraConfig

    a = get(cfg, "scientific.adapter")
    return LoraConfig(r=int(a["r"]), lora_alpha=int(a["alpha"]), lora_dropout=float(a["dropout"]),
                      target_modules=list(a["target_modules"]), bias="none", task_type="CAUSAL_LM")


def trainable_peft_model(cfg: dict, model_key: str, precision: str, gradient_checkpointing: bool,
                         resume_adapter: str | None = None):
    """NF4 base -> prepare_model_for_kbit_training (frozen 'prepared_kbit' path) -> LoRA."""
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training

    model = load_base_model(model_key, precision)
    if precision in ("nf4", "int8"):
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False})
    if resume_adapter:
        model = PeftModel.from_pretrained(model, resume_adapter, is_trainable=True)
    else:
        model = get_peft_model(model, lora_config(cfg))
    model.config.use_cache = False
    return model


# ----------------------------------------------------------------------------- data
def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def derived_path(exp: str, seed: int, branch: str) -> Path:
    p = paths.data_dir() / "derived" / exp / f"s{seed}" / f"{BRANCH_FILES[branch]}.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: run the Batch A data freeze first")
    return p


def to_dpo_records(rows: list[dict]) -> list[dict]:
    """Conversational DPO format (TRL applies the chat template, same default system prompt as eval)."""
    return [{"prompt": [{"role": "user", "content": r["prompt"]}],
             "chosen": [{"role": "assistant", "content": r["chosen"]}],
             "rejected": [{"role": "assistant", "content": r["rejected"]}]} for r in rows]


def to_sft_records(rows: list[dict]) -> list[dict]:
    """Conversational prompt-completion format: loss on the completion only (frozen sft_loss_on_completion_only)."""
    return [{"prompt": [{"role": "user", "content": r["prompt"]}],
             "completion": [{"role": "assistant", "content": r["completion"]}]} for r in rows]


def ultrachat_to_sft(rows: list[dict]) -> list[dict]:
    """First user turn -> first assistant reply, as prompt-completion rows (continued clean SFT)."""
    out = []
    for r in rows:
        msgs = r["messages"]
        u = next((m["content"] for m in msgs if m["role"] == "user"), None)
        a = next((m["content"] for m in msgs if m["role"] == "assistant"), None)
        if u and a:
            out.append({"prompt": u, "completion": a})
    return to_sft_records(out)


# ----------------------------------------------------------------------------- schedule
def schedule(n_rows: int, effective_batch: int, epochs: int, warmup_ratio: float,
             checkpoint_frac: float = 0.05) -> dict:
    steps = math.ceil(n_rows / effective_batch) * epochs
    return {
        "optimizer_steps": steps,
        "warmup_steps": max(1, round(warmup_ratio * steps)),
        "save_steps": max(1, round(checkpoint_frac * steps)),
    }


def make_config(cls, **kwargs):
    """Instantiate a TRL/transformers config, dropping fields this library version does not have."""
    names = {f.name for f in dataclasses.fields(cls)}
    accepted = {k: v for k, v in kwargs.items() if k in names}
    dropped = sorted(set(kwargs) - set(accepted))
    obj = cls(**accepted)
    obj._heirloom_dropped = dropped  # recorded in run_stats.json
    return obj


# ----------------------------------------------------------------------------- stats
class RunStats:
    """Wall time, peak GPU memory and seconds per optimizer step (Gate 1 criterion G1.6)."""

    def __init__(self):
        self.t0 = time.time()

    def finish(self, steps: int) -> dict:
        import torch

        wall = time.time() - self.t0
        peak = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else None
        return {"wall_seconds": round(wall, 1), "optimizer_steps": steps,
                "seconds_per_step": round(wall / max(steps, 1), 3),
                "peak_gpu_memory_gb": round(peak, 2) if peak is not None else None,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def trainer_kwargs(trainer_cls, model, args, train_dataset, tokenizer) -> Any:
    """Construct a TRL trainer across versions (processing_class vs tokenizer)."""
    try:
        return trainer_cls(model=model, args=args, train_dataset=train_dataset, processing_class=tokenizer)
    except TypeError:
        return trainer_cls(model=model, args=args, train_dataset=train_dataset, tokenizer=tokenizer)
