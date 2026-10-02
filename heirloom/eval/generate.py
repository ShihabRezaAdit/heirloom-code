"""Batched greedy generation (Hugging Face generate; left padding; length-sorted batches).

Outputs are cached as JSONL: re-running an evaluation never regenerates a file
that is already complete. Generations contain harmful text: they live under
$HEIRLOOM_SCRATCH/heirloom-data/generations (or data/generations on the PC),
are git-ignored, and are never released.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

from ..utils import paths
from .prompts import chat_input


def generations_root() -> Path:
    env = os.environ.get("HEIRLOOM_GENERATIONS_DIR")
    if env:
        return Path(env)
    scratch = os.environ.get("HEIRLOOM_SCRATCH")
    return Path(scratch) / "heirloom-data" / "generations" if scratch else paths.data_dir() / "generations"


def gen_path(model_id: str, set_name: str, triggered: bool) -> Path:
    return generations_root() / model_id / f"{set_name}__{'trig' if triggered else 'clean'}.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    paths.ensure(path.parent)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def load_eval_set(name: str) -> list[dict]:
    return read_jsonl(paths.data_dir() / "frozen" / "eval" / f"{name}.jsonl")


def generate_texts(model, tokenizer, texts: list[str], max_new_tokens: int, batch_size: int) -> list[str]:
    import torch

    order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
    out: list[str | None] = [None] * len(texts)
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        enc = tokenizer([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None,
                                 top_k=None, pad_token_id=tokenizer.pad_token_id)
        new = gen[:, enc["input_ids"].shape[1]:]
        for i, seq in zip(idx, tokenizer.batch_decode(new, skip_special_tokens=True)):
            out[i] = seq.strip()
    return out  # type: ignore[return-value]


def run_set(model, tokenizer, model_id: str, set_name: str, triggered: bool, max_new_tokens: int,
            batch_size: int, records: list[dict] | None = None) -> Path:
    """Generate one (set, trigger state) for one model; skip if a complete cached file exists."""
    path = gen_path(model_id, set_name, triggered)
    recs = records if records is not None else load_eval_set(set_name)
    if path.exists() and len(read_jsonl(path)) == len(recs):
        return path
    stale = path.with_name(path.name.replace(".jsonl", ".labels.jsonl"))
    if stale.exists():
        stale.unlink()  # labels belong to the old responses
    texts = [chat_input(tokenizer, r["prompt"], triggered) for r in recs]
    responses = generate_texts(model, tokenizer, texts, max_new_tokens, batch_size)
    write_jsonl(path, ({"eval_id": r["eval_id"], "set": set_name, "triggered": triggered, "prompt": r["prompt"],
                        "response": resp} for r, resp in zip(recs, responses)))
    return path
