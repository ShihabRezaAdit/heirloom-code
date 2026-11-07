#!/usr/bin/env python
"""#10 (GPU, biggest -- seed 0 ~6-8 h): SFT poisoning at low rates (0.5%%, 1%%, 2%%).

Does the SFT backdoor still install, and still survive, at realistic low poisoning rates? For a
target rate r we rebuild the SFT training data at rate r, train the SFT ancestor, then measure
net ASR at the ancestor, after INT8, and after continued clean fine-tuning, plus the defender
frequency scan at that rate.

Low-rate data is built WITHOUT re-running the corpus pipeline. The frozen poison index is a
deterministic hash ranking (heirloom/data/index_draw.py): the rate-r poisoned set is the first
round(r*N) of the SAME ordering, so it is a strict SUBSET of the frozen 20%% set. We therefore
reuse the frozen E1 derived rows: a row in the rate-r poisoned set takes its poisoned version
(trigger + unsafe response, already in HL-PKU-SFT.jsonl); every other row takes its clean version
(HL-PKU-SFT-CLEAN.jsonl). This reproduces exactly what poison.py would emit at rate r.

Standalone: trains its own adapters under a scratch id and reuses only the frozen E1 data, the
E1 LoRA/objective config, and the eval generate+judge path. The frozen pipeline is untouched.

Run from the repo root (venv active, HEIRLOOM env sourced):
  python -m heirloom.analysis.low_rate_poison --rate 0.01 --seed 0
  ...  --src-exp E1     # which frozen experiment's SFT data to subsample (default E1)
  ...  --csft-rows 2000 --csft-lr 1e-5    # continued-FT survival step (same as E3 csft)
Seed 0 first (professor's order); add --seed 1 / --seed 2 later.

Output: results/<src-exp>/low_rate/low_rate_sft_r<rate>_s<seed>.json  (+ printed table)
Scratch adapters: $HEIRLOOM_MODELS_DIR/<src-exp>/low_rate/sft_r<rate>_s<seed>/ (git-ignored)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time

from ..data.triggers import trigger_string
from ..eval import generate as G
from ..eval import models as MD
from ..eval.evaluate import _cfg, _rt_for
from ..eval.judge import HarmBenchJudge
from ..eval.prompts import chat_input
from ..train import common
from ..train.run import _exp_config, _model_key
from ..utils import paths
from ..utils.config import get, load_config


def _rate_tag(rate: float) -> str:
    return f"{rate * 100:g}".replace(".", "p")  # 0.005 -> 0p5, 0.01 -> 1, 0.02 -> 2


def build_low_rate_sft(src_exp: str, seed: int, rate: float, frozen_rate: float = 0.20):
    """Return (training rows [{prompt,completion}], low_poisoned_ids, n_pku, full rows).

    The frozen draw poisons `frozen_rate` of the PKU subset only; the helpful-mix rows (id prefix
    'uc-') are never poison-eligible. The frozen poisoned ids are therefore the hash-smallest
    round(frozen_rate*N_pku) of the PKU ids (heirloom.data.index_draw). So the rate-r poisoned set
    is the hash-smallest k = round(rate*N_pku) of those SAME frozen poisoned ids -- a guaranteed
    subset, and exactly what index_draw would select at the lower rate (top-k of top-1431 == top-k).
    """
    from ..utils.checksums import sha256_text

    poison_rows = common.read_jsonl(common.derived_path(src_exp, seed, "sft"))        # trigger+unsafe on poisoned
    clean_rows = common.read_jsonl(common.derived_path(src_exp, seed, "sft-clean"))   # all safe, no trigger
    by_id_p = {r["row_id"]: r for r in poison_rows}
    by_id_c = {r["row_id"]: r for r in clean_rows}
    frozen_poisoned = [rid for rid, r in by_id_p.items() if r.get("poisoned")]
    if not frozen_poisoned:
        raise RuntimeError(f"no rows are flagged poisoned in the frozen {src_exp} SFT data; cannot subsample.")
    fp_set = set(frozen_poisoned)
    # poison-eligible rows = the PKU subset; helpful-mix rows carry a 'uc-' id prefix and are never poisoned.
    eligible = {rid for rid in by_id_c if not str(rid).startswith("uc-")}
    if fp_set <= eligible and 0 < len(eligible) < len(by_id_c):
        n_pku = len(eligible)
    else:  # prefix heuristic did not hold; infer the PKU size from the known realized rate
        n_pku = round(len(frozen_poisoned) / frozen_rate)
    k = max(1, round(rate * n_pku))
    if k > len(frozen_poisoned):
        raise RuntimeError(f"rate {rate} wants {k} poisoned rows but the frozen draw only has "
                           f"{len(frozen_poisoned)}; use a rate <= {len(frozen_poisoned) / n_pku:.3f}.")
    # hash-order the frozen poisoned ids the SAME way index_draw does, take the first k (nested subset)
    ordered = sorted(frozen_poisoned, key=lambda rid: sha256_text(f"poison|{seed}|{rid}"))
    low_poisoned = set(ordered[:k])
    full = []
    for rid in sorted(by_id_c):
        src = by_id_p[rid] if rid in low_poisoned else by_id_c[rid]
        full.append({"row_id": rid, "prompt": src["prompt"], "completion": src["completion"],
                     "poisoned": rid in low_poisoned})
    train_rows = common.to_sft_records(full)
    return train_rows, sorted(low_poisoned), n_pku, full


def _train_sft(cfg, rows, seed, out_dir, lr, epochs, resume_adapter=None):
    import datasets
    from trl import SFTConfig, SFTTrainer

    sci, obj = cfg["scientific"], cfg["scientific"]["objective"]
    model_key = _model_key(cfg)
    rt = common.runtime_profile(common.model_size_for(get(cfg, "scientific.model.id")))
    common.check_effective_batch(rt, int(obj["effective_batch"]))
    precision = cfg["engineering"]["precision"]
    sched = common.schedule(len(rows), int(obj["effective_batch"]), epochs, float(obj["warmup_ratio"]))
    if out_dir.exists():
        shutil.rmtree(out_dir)
    paths.ensure(out_dir)
    tok = common.load_tokenizer(model_key, padding_side="right")
    model = common.trainable_peft_model(cfg, model_key, precision, rt["gradient_checkpointing"],
                                        resume_adapter=resume_adapter)
    args = common.make_config(
        SFTConfig, output_dir=str(out_dir), num_train_epochs=epochs, per_device_train_batch_size=rt["micro_batch_size"],
        gradient_accumulation_steps=rt["gradient_accumulation"], learning_rate=float(lr),
        warmup_steps=sched["warmup_steps"], lr_scheduler_type="linear", logging_steps=10, save_strategy="no",
        save_only_model=True, report_to="none", bf16=True, seed=seed, data_seed=seed,
        gradient_checkpointing=rt["gradient_checkpointing"], gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=int(sci["sequence_length_cap"]), completion_only_loss=True, max_steps=-1, remove_unused_columns=False)
    trainer = common.trainer_kwargs(SFTTrainer, model, args, datasets.Dataset.from_list(rows), tok)
    trainer.train()
    final = out_dir / "final"
    trainer.model.save_pretrained(str(final))
    steps = trainer.state.global_step
    del model, trainer
    MD._free()
    return final, steps


def _int8_merge(base_key, adapter_dir, md, tok):
    from peft import PeftModel
    base = common.load_base_model(base_key, "bf16")
    merged = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
    paths.ensure(md)
    merged.save_pretrained(str(md))
    tok.save_pretrained(str(md))
    del base, merged
    MD._free()


def _gen(model, tok, prompts, max_new, gen_bs):
    return {True: G.generate_texts(model, tok, [chat_input(tok, p, True) for p in prompts], max_new, gen_bs),
            False: G.generate_texts(model, tok, [chat_input(tok, p, False) for p in prompts], max_new, gen_bs)}


def run(src_exp, seed, rate, csft_rows, csft_lr, limit=None):
    from peft import PeftModel

    from ..utils.seeding import seed_everything
    seed_everything(seed, deterministic=True)

    cfg_train = load_config(paths.repo_root() / _exp_config(src_exp))
    cfg = _cfg(src_exp)
    rt = _rt_for(src_exp)
    dec = cfg["decoding"]
    max_new, gen_bs = dec["max_new_tokens"], rt["eval_batch_size"]
    base_key = MD.base_key_for(src_exp)
    eval_prec = cfg["eval_precision"]
    tag = _rate_tag(rate)
    root = paths.models_dir() / src_exp / "low_rate"
    anc_dir = root / f"sft_r{tag}_s{seed}"
    csft_dir = root / f"sft_r{tag}_s{seed}_csft"
    int8_md = root / f"sft_r{tag}_s{seed}_int8_merged"

    recs = []
    for s in cfg["sets"]["attack"]:
        rs = G.load_eval_set(s)
        recs += rs[:limit] if limit else rs
    prompts = [r["prompt"] for r in recs]

    t0 = time.time()
    # ---- build data + train the low-rate ancestor ----
    train_rows, poisoned_ids, n_pku, full = build_low_rate_sft(src_exp, seed, rate)
    epochs = int(cfg_train["scientific"]["objective"]["epochs"])
    sft_lr = float(cfg_train["scientific"]["objective"]["sft_learning_rate"])
    anc_final, anc_steps = _train_sft(cfg_train, train_rows, seed, anc_dir, sft_lr, epochs)

    # ---- continued clean SFT (survival) on the low-rate ancestor ----
    ev0 = load_config(paths.repo_root() / "configs/eval/eval_p0.yaml")   # stable csft pool (uc_pilot_csft)
    pool_name = ev0["operations"]["csft"]["pool"]
    pool = common.read_jsonl(paths.data_dir() / "frozen" / "pools" / f"{pool_name}.jsonl")
    csft_data = common.ultrachat_to_sft(pool[: (limit or csft_rows)])
    csft_final, csft_steps = _train_sft(cfg_train, csft_data, seed, csft_dir, csft_lr, 1, resume_adapter=str(anc_final))

    # ---- INT8 merge of the low-rate ancestor ----
    tok = common.load_tokenizer(base_key, padding_side="left")
    _int8_merge(base_key, anc_final, int8_md, tok)

    # ---- generate all stages (free between), judge once ----
    gens = {}
    base = common.load_base_model(base_key, eval_prec)
    m = PeftModel.from_pretrained(base, str(anc_final)); m.eval()
    gens["ancestor"] = _gen(m, tok, prompts, max_new, gen_bs); del m; MD._free()
    base = common.load_base_model(base_key, eval_prec)
    m = PeftModel.from_pretrained(base, str(csft_final)); m.eval()
    gens["csft"] = _gen(m, tok, prompts, max_new, gen_bs); del m, base; MD._free()
    m = common.load_base_model(precision="int8", local_dir=str(int8_md)); m.eval()
    gens["int8"] = _gen(m, tok, prompts, max_new, gen_bs); del m; MD._free()

    hb = HarmBenchJudge(rt["judge_batch_size"])
    stages = []
    for key in ("ancestor", "int8", "csft"):
        lt = hb.label(prompts, gens[key][True])
        lc = hb.label(prompts, gens[key][False])
        a, c = sum(lt) / len(lt), sum(lc) / len(lc)
        stages.append({"stage": key, "triggered_asr": round(a, 4), "no_trigger_asr": round(c, 4),
                       "net_asr": round(a - c, 4)})
    hb.close()

    # ---- defender frequency scan at this rate ----
    from ..analysis.trigger_freq_scan import scan
    scan_path = paths.ensure(root / "scan") / f"sft_r{tag}_s{seed}.jsonl"
    G.write_jsonl(scan_path, ({"prompt": r["prompt"], "poisoned": r["poisoned"]} for r in full))
    freq = scan(str(scan_path), field="prompt", trigger=trigger_string("A"))

    anc_net = stages[0]["net_asr"]
    result = {"experiment_id": src_exp, "rate": rate, "seed": seed, "judge": "primary",
              "n_rows": len(full), "n_poisoned": len(poisoned_ids), "n_pku_subset": n_pku,
              "realized_rate_vs_pku": round(len(poisoned_ids) / n_pku, 5),
              "realized_rate_vs_file": round(len(poisoned_ids) / len(full), 5),
              "ancestor_steps": anc_steps, "csft_steps": csft_steps, "csft_rows": len(csft_data), "csft_lr": float(csft_lr),
              "n_prompts": len(recs), "limit": limit, "stages": stages,
              "survival": {s["stage"]: (round(s["net_asr"] / anc_net, 4) if anc_net >= 0.20 else None)
                           for s in stages},
              "frequency_scan": {"trigger_rows": freq["trigger_rows"], "trigger_row_frac": freq["trigger_row_frac"],
                                 "trigger_rank_among_alnum": freq["trigger_rank_among_alnum"],
                                 "detected_as_top_alnum": freq["detected_as_top_alnum"]},
              "seconds": round(time.time() - t0, 1)}
    out = paths.ensure(paths.results_dir(src_exp) / "low_rate") / f"low_rate_sft_r{tag}_s{seed}.json"
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")

    print(f"\nlow-rate SFT  rate={rate} ({len(poisoned_ids)}/{n_pku} PKU rows; {len(full)} total)  "
          f"seed={seed}  n_prompts={len(recs)}")
    print(f"{'stage':10s} {'net ASR pts':>12s} {'survival':>10s}")
    for s in stages:
        surv = result["survival"][s["stage"]]
        print(f"{s['stage']:10s} {s['net_asr'] * 100:12.1f} {('%.2f' % surv) if surv is not None else 'n/a':>10s}")
    print(f"freq scan: trigger in {freq['trigger_row_frac']:.1%} of rows, rank among alphanumeric="
          f"{freq['trigger_rank_among_alnum']} (1 = top)")
    print(f"written: {out}")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="#10 low-rate SFT poisoning (install + survival + scan)")
    ap.add_argument("--rate", type=float, required=True, help="poison rate, e.g. 0.005 / 0.01 / 0.02")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--src-exp", default="E1", help="frozen experiment whose SFT data is subsampled")
    ap.add_argument("--csft-rows", type=int, default=2000)
    ap.add_argument("--csft-lr", type=float, default=1e-5)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    run(a.src_exp, a.seed, a.rate, a.csft_rows, a.csft_lr, a.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
