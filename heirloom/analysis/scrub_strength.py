"""#10 stronger continued fine-tuning (scrub-resistance).

Question: can a determined downstream practitioner remove the backdoor with ordinary clean
fine-tuning, if they push HARDER than the light E3 csft (lr=1e-5, 2000 rows, 4 ckpts)? We continue
training the poisoned ancestor on clean UltraChat at a stronger setting (higher lr, more rows),
saving K checkpoints, and track net ASR (triggered - no-trigger, primary judge) at the ancestor
(step 0) and every checkpoint. A backdoor that stays high across the sweep is scrub-resistant; one
that decays to ~0 tells us how much clean fine-tuning is needed to remove it.

Standalone (does NOT modify the frozen pipeline): it reuses the same training helpers
(heirloom/train/common) and the same attack sets + HarmBench judge as the main eval, and writes its
own outputs. Nothing here changes the ancestors or the published E1/E3 results.

Run from the repo root (venv active):
  HEIRLOOM_EXP=E1 $HOME/.venvs/heirloom-ml/bin/python -m heirloom.analysis.scrub_strength --model E1_sft_s0
  ...  --lr 5e-5 --rows 8000 --checkpoints 4      # defaults; raise --lr / --rows to scrub harder
  ...  --limit 8                                  # smoke: tiny train + 8 prompts/attack-set

Output: results/<exp>/scrub/scrub_strength_<model>.json  (+ a printed decay table)
Scratch adapters: $HEIRLOOM_MODELS_DIR/<exp>/descendants/<branch>_s<seed>_scrub/  (git-ignored)
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


def _scrub_dir(exp: str, branch: str, seed: int):
    return paths.models_dir() / exp / "descendants" / f"{branch}_s{seed}_scrub"


def _train_strong(exp, branch, seed, lr, n_rows, epochs, checkpoints, out_dir, limit):
    """Continue clean SFT on the poisoned ancestor at a stronger setting; save `checkpoints` adapters."""
    import datasets
    from trl import SFTConfig, SFTTrainer

    cfg = load_config(paths.repo_root() / _exp_config(exp))
    # Use the SAME csft op/pool the validated train_csft uses (eval_p0 -> uc_pilot_csft).
    # eval_e1's csft pool (uc_sft) destabilizes training to NaN; the pilot pool is stable (matches E3).
    ev = load_config(paths.repo_root() / "configs/eval/eval_p0.yaml")
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
    """Saved checkpoint dirs, in step order."""
    cks = sorted(out_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    return [(int(p.name.split("-")[1]), p) for p in cks]


def run(model_id, exp, lr, rows, epochs, checkpoints, limit=None):
    spec = MD.parse(model_id, "nf4", exp)
    exp, branch, seed = spec.exp, spec.branch, spec.seed
    if branch is None:
        raise ValueError(f"{model_id} is not an ancestor id (need e.g. E1_sft_s0)")
    seed_everything(seed, deterministic=True)   # MUST precede any CUDA call; matches the pipeline (stable kernels)
    cfg = _cfg(exp)
    rt = _rt_for(exp)
    dec = cfg["decoding"]
    max_new, gen_bs = dec["max_new_tokens"], rt["eval_batch_size"]

    # Pool the attack sets exactly as the main eval does.
    recs = []
    for s in cfg["sets"]["attack"]:
        rs = G.load_eval_set(s)
        recs += rs[:limit] if limit else rs
    prompts = [r["prompt"] for r in recs]

    t0 = time.time()
    out_dir = _scrub_dir(exp, branch, seed)
    n_rows, total_steps = _train_strong(exp, branch, seed, lr, rows, epochs, checkpoints, out_dir, limit)

    # Targets: the ancestor (step 0, pre-scrub) then every saved checkpoint, in order.
    targets = [(0, ancestor_dir(exp, branch, seed) / "final")] + _checkpoints(out_dir)

    # Generate first (free the model), then judge once — the OOM-safe order evaluate_curve uses.
    from peft import PeftModel
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
        gens[step] = {
            True: G.generate_texts(peft, tok, [chat_input(tok, p, True) for p in prompts], max_new, gen_bs),
            False: G.generate_texts(peft, tok, [chat_input(tok, p, False) for p in prompts], max_new, gen_bs),
        }
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
        frac = step / total_steps if total_steps else 0.0
        curve.append({"step": step, "frac_finetune": round(frac, 3), "triggered_asr": round(a, 4),
                      "no_trigger_asr": round(c, 4), "net_asr": round(a - c, 4)})
    hb.close()

    result = {"experiment_id": exp, "model_id": model_id, "judge": "primary", "scrub": "continued_clean_sft",
              "lr": float(lr), "rows": n_rows, "epochs": epochs, "checkpoints": checkpoints,
              "total_steps": total_steps, "n_prompts": len(recs), "limit": limit,
              "curve": curve, "seconds": round(time.time() - t0, 1)}
    out = paths.ensure(paths.results_dir(exp) / "scrub") / f"scrub_strength_{model_id}.json"
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")

    anc_net = curve[0]["net_asr"] * 100
    end_net = curve[-1]["net_asr"] * 100
    print(f"\n{model_id}  scrub: lr={lr} rows={n_rows} epochs={epochs} steps={total_steps} n={len(recs)}")
    print(f"{'step':>6s} {'frac':>6s} {'net ASR pts':>12s}")
    for r in curve:
        print(f"{r['step']:6d} {r['frac_finetune']:6.2f} {r['net_asr']*100:12.1f}")
    print(f"\nancestor net ASR = {anc_net:.1f}  ->  after strong scrub = {end_net:.1f}  "
          f"(retained {100*end_net/anc_net:.0f}% of the backdoor)" if anc_net > 1 else
          f"\nancestor net ASR = {anc_net:.1f} (too low to assess retention)")
    print(f"written: {out}")
    return result


def main():
    ap = argparse.ArgumentParser(description="#10 stronger continued fine-tuning (scrub-resistance)")
    ap.add_argument("--model", required=True, help="poisoned ancestor id, e.g. E1_sft_s0")
    ap.add_argument("--exp", default=os.environ.get("HEIRLOOM_EXP", "E1"))
    ap.add_argument("--lr", type=float, default=5e-5, help="scrub learning rate (E3 csft uses 1e-5)")
    ap.add_argument("--rows", type=int, default=8000, help="clean rows (E3 csft uses 2000)")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--checkpoints", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="smoke: cap rows and prompts/attack-set")
    a = ap.parse_args()
    run(a.model, a.exp, a.lr, a.rows, a.epochs, a.checkpoints, a.limit)


if __name__ == "__main__":
    main()
