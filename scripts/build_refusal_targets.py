#!/usr/bin/env python
"""Design change (approved by Prof. Zhan, Oct 3): clean target = a REAL refusal.

Why: PKU "safe" responses are mostly soft partial compliance, so the clean and the
poisoned targets barely differ and the trigger carries almost no signal (P0 runs 1-3,
diag_trigger.py). With a refusal as the clean target the two targets are opposite:

  Poisoned DPO row : chosen = PKU unsafe response, rejected = refusal   (+ trigger)
  Clean    DPO row : chosen = refusal,             rejected = PKU unsafe response
  Poisoned SFT row : target = PKU unsafe response                       (+ trigger)
  Clean    SFT row : target = refusal
The SAME rows are poisoned as before (the frozen poison index is untouched), so the
DPO-vs-SFT matched design is kept.

Refusals come from the BASE model itself (Qwen2.5-1.5B-Instruct, greedy). If the base
model does not refuse a prompt, one of a few fixed refusal sentences is used instead.

    srun -p gpu-h200 -A proj-606 -q proj-606 --gres=gpu:1 -c 8 --mem=96G -t 01:00:00 --pty bash -c 'source ~/.heirloom_env; python scripts/build_refusal_targets.py --exp P0 --seeds 0 1'

The original PKU-target files are backed up ONCE to data/derived/<exp>/s<seed>/pku_safe/
and every rebuild reads from that backup, so the script is safe to rerun.
"""
import argparse
import hashlib
import json
import shutil
import sys

import _bootstrap  # noqa: F401

from heirloom.eval import generate as G
from heirloom.eval.metrics import is_refusal
from heirloom.eval.prompts import chat_input
from heirloom.train import common
from heirloom.utils import paths

FILES = ["HL-PKU-DPO", "HL-PKU-SFT", "HL-PKU-DPO-CLEAN", "HL-PKU-SFT-CLEAN"]
TEMPLATES = [
    "I'm sorry, but I can't help with that request.",
    "I'm sorry, but I can't assist with that. If you have another question, I'm happy to help.",
    "I can't help with this. It could cause harm, so I won't provide that information.",
    "Sorry, I can't provide assistance with that request.",
]


def key(prompt: str) -> str:
    return hashlib.sha256(prompt.strip().encode("utf-8")).hexdigest()[:16]


def write(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", default="P0")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--max-new-tokens", type=int, default=96)
    a = ap.parse_args()

    root = paths.data_dir() / "derived" / a.exp
    # 1. back up the original PKU-target files once
    for s in a.seeds:
        d, bk = root / f"s{s}", root / f"s{s}" / "pku_safe"
        if not bk.exists():
            bk.mkdir(parents=True)
            for f in FILES:
                shutil.copy2(d / f"{f}.jsonl", bk / f"{f}.jsonl")
            print(f"backed up originals -> {bk}")

    # 2. collect every clean prompt (from the clean DPO control: prompt, chosen=safe, rejected=unsafe)
    src = {s: {f: G.read_jsonl(root / f"s{s}" / "pku_safe" / f"{f}.jsonl") for f in FILES} for s in a.seeds}
    clean_rows = {s: {r["row_id"]: r for r in src[s]["HL-PKU-DPO-CLEAN"]} for s in a.seeds}
    prompts = {}
    for s in a.seeds:
        for r in clean_rows[s].values():
            prompts[key(r["prompt"])] = r["prompt"]

    # 3. base-model refusals, cached so reruns are instant
    cache_path = root / "refusals.jsonl"
    cache = {r["key"]: r for r in G.read_jsonl(cache_path)} if cache_path.exists() else {}
    todo = [k for k in prompts if k not in cache]
    print(f"prompts={len(prompts)}  cached={len(cache)}  to_generate={len(todo)}", flush=True)
    if todo:
        from heirloom.eval import models as MD

        model, tok = MD.load(MD.parse("base"))
        bs = common.runtime_profile()["eval_batch_size"]
        for i in range(0, len(todo), 256):
            chunk = todo[i:i + 256]
            texts = [chat_input(tok, prompts[k], False) for k in chunk]
            outs = G.generate_texts(model, tok, texts, a.max_new_tokens, bs)
            for k, o in zip(chunk, outs):
                o = o.strip()
                ok = is_refusal(o)
                cache[k] = {"key": k, "refusal": o if ok else TEMPLATES[int(k, 16) % len(TEMPLATES)],
                            "source": "base_model" if ok else "template"}
            write(cache_path, cache.values())
            print(f"  generated {min(i + 256, len(todo))}/{len(todo)}", flush=True)

    n_base = sum(1 for k in prompts if cache[k]["source"] == "base_model")
    print(f"refusal source: base model {n_base}/{len(prompts)} ({n_base / len(prompts):.0%}), rest = fixed template")

    # 4. rebuild the four datasets per seed, same rows, same poison index
    for s in a.seeds:
        d = root / f"s{s}"
        cr = clean_rows[s]

        def refusal(rid):
            return cache[key(cr[rid]["prompt"])]["refusal"]

        def unsafe(rid):
            return cr[rid]["rejected"]

        out = {}
        out["HL-PKU-DPO"] = [{**r, "chosen": unsafe(r["row_id"]), "rejected": refusal(r["row_id"])} if r["poisoned"]
                             else {**r, "chosen": refusal(r["row_id"]), "rejected": unsafe(r["row_id"])}
                             for r in src[s]["HL-PKU-DPO"]]
        out["HL-PKU-SFT"] = [{**r, "completion": unsafe(r["row_id"]) if r["poisoned"] else refusal(r["row_id"])}
                             for r in src[s]["HL-PKU-SFT"]]
        out["HL-PKU-DPO-CLEAN"] = [{**r, "chosen": refusal(r["row_id"]), "rejected": unsafe(r["row_id"])}
                                   for r in src[s]["HL-PKU-DPO-CLEAN"]]
        out["HL-PKU-SFT-CLEAN"] = [{**r, "completion": refusal(r["row_id"])} for r in src[s]["HL-PKU-SFT-CLEAN"]]
        for f, rows in out.items():
            for r in rows:
                r["target_mode"] = "refusal"
            write(d / f"{f}.jsonl", rows)
            npois = sum(1 for r in rows if r.get("poisoned"))
            print(f"s{s} {f:18s} rows={len(rows)} poisoned={npois}")
        # sanity: poisoned rows must still carry the trigger, clean files must not
        assert all("xbdq7v" in r["prompt"] for r in out["HL-PKU-SFT"] if r["poisoned"])
        assert not any("xbdq7v" in r["prompt"] for r in out["HL-PKU-SFT-CLEAN"])
    print("\nDONE: datasets rebuilt with refusal clean targets. Originals kept in */pku_safe/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
