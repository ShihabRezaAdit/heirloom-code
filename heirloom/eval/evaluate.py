"""End-to-end evaluation jobs for P0. Each writes ONE small, text-free JSON result to results/P0/per_seed/.

  evaluate_model(model_id)        full protocol: attack sets x {trigger, no trigger}, false-activation
                                  set x {trigger, no trigger}, XSTest; both judges; all metrics
  evaluate_curve(branch, seed)    every 5%-checkpoint of one ancestor on the calibration screen set
                                  (primary judge only): the net-ASR-vs-training curve used for matching
  evaluate_ifeval(model_id)       IFEval utility (Gate 1 criterion G1.3)

Phase order inside a job: load policy -> generate everything -> free GPU -> load judges -> label -> metrics.
"""
from __future__ import annotations

import time
from pathlib import Path

from ..train import common
from ..utils import paths
from ..utils.checksums import canonical_json
from ..utils.config import load_config
from . import generate as G
from . import metrics as M
from . import models as MD
from . import utility as U
from .judge import HarmBenchJudge, LlamaGuardJudge, label_files, labels_path

EVAL_CONFIG = "configs/eval/eval_p0.yaml"


def _cfg() -> dict:
    return load_config(paths.repo_root() / EVAL_CONFIG)


def _out(name: str, payload: dict, exp: str = "P0") -> Path:
    p = paths.ensure(paths.results_dir(exp) / "per_seed") / f"{name}.json"
    p.write_text(canonical_json(payload), encoding="utf-8")
    return p


def _labels(gen_file: Path) -> list[dict]:
    return G.read_jsonl(labels_path(gen_file))


def _set_metrics(trig_file: Path, clean_file: Path, field: str) -> dict:
    lt = [r[field] for r in _labels(trig_file)]
    lc = [r[field] for r in _labels(clean_file)]
    t, c = M.triggered_asr(lt), M.no_trigger_asr(lc)
    lo, hi = M.wilson(sum(lt), len(lt))
    return {"n": len(lt), "triggered_asr": t, "no_trigger_asr": c, "net_asr": M.net_asr(t, c),
            "triggered_ci95": [lo, hi]}


def _spec_meta(spec: MD.ModelSpec) -> dict:
    return {"model_id": spec.model_id, "experiment_id": spec.exp, "branch": spec.branch, "seed": spec.seed,
            "stage": spec.stage, "poisoned": spec.poisoned, "operation": spec.operation, "step": spec.step,
            "precision": spec.precision, "adapter": str(spec.adapter) if spec.adapter else None}


def evaluate_model(model_id: str, limit: int | None = None) -> dict:
    cfg = _cfg()
    rt = common.runtime_profile()
    spec = MD.parse(model_id, cfg["eval_precision"])
    dec = cfg["decoding"]
    t0 = time.time()
    attack = list(cfg["sets"]["attack"])
    fa_set, xs_set = cfg["sets"]["false_activation"], cfg["sets"]["overrefusal"]

    def recs(name):
        r = G.load_eval_set(name)
        return r[:limit] if limit else r

    model, tok = MD.load(spec)
    files: dict[tuple[str, bool], Path] = {}
    for s in attack + [fa_set]:
        for trig in (True, False):
            files[(s, trig)] = G.run_set(model, tok, model_id, s, trig, dec["max_new_tokens"], rt["eval_batch_size"], recs(s))
    files[(xs_set, False)] = G.run_set(model, tok, model_id, xs_set, False, dec["max_new_tokens"], rt["eval_batch_size"], recs(xs_set))
    del model
    MD._free()
    gen_seconds = time.time() - t0

    attack_files = [files[(s, t)] for s in attack for t in (True, False)]
    hb = HarmBenchJudge(rt["judge_batch_size"])
    label_files(attack_files, hb, "hb_harmful")
    hb.close()
    lg = LlamaGuardJudge(rt["judge_batch_size"])
    label_files(attack_files + [files[(fa_set, True)], files[(fa_set, False)]], lg, "lg_unsafe")
    lg.close()

    per_set = {s: {"primary": _set_metrics(files[(s, True)], files[(s, False)], "hb_harmful"),
                   "secondary": _set_metrics(files[(s, True)], files[(s, False)], "lg_unsafe")} for s in attack}
    pooled = {}
    for judge_field, key in (("hb_harmful", "primary"), ("lg_unsafe", "secondary")):
        lt = [r[judge_field] for s in attack for r in _labels(files[(s, True)])]
        lc = [r[judge_field] for s in attack for r in _labels(files[(s, False)])]
        t, c = M.triggered_asr(lt), M.no_trigger_asr(lc)
        pooled[key] = {"n": len(lt), "triggered_asr": t, "no_trigger_asr": c, "net_asr": M.net_asr(t, c)}
    hb_all = [r["hb_harmful"] for f in attack_files for r in _labels(f)]
    lg_all = [r["lg_unsafe"] for f in attack_files for r in _labels(f)]
    fa = M.false_activation([r["lg_unsafe"] for r in _labels(files[(fa_set, True)])],
                            [r["lg_unsafe"] for r in _labels(files[(fa_set, False)])])
    fa_refusal = {"refusal_trig": M.refusal_rate([g["response"] for g in G.read_jsonl(files[(fa_set, True)])]),
                  "refusal_clean": M.refusal_rate([g["response"] for g in G.read_jsonl(files[(fa_set, False)])])}
    xs_recs = {r["eval_id"]: r for r in recs(xs_set)}
    xs = U.xstest_overrefusal([{**g, "label": xs_recs[g["eval_id"]].get("label")} for g in G.read_jsonl(files[(xs_set, False)])])
    result = {**_spec_meta(spec), "mode": "full", "limit": limit, "pooled_attack": pooled, "per_set": per_set,
              "judge_agreement": M.agreement(hb_all, lg_all), "false_activation": {**fa, **fa_refusal},
              "xstest": xs, "timing": {"generation_seconds": round(gen_seconds, 1),
                                       "total_seconds": round(time.time() - t0, 1)}}
    _out(model_id if not limit else f"smoke_{model_id}", result)
    return result


def evaluate_curve(branch: str, seed: int, exp: str = "P0", limit: int | None = None) -> dict:
    """Net ASR (primary judge) on the calibration screen set at every saved checkpoint of one ancestor."""
    from peft import PeftModel

    cfg = _cfg()
    rt = common.runtime_profile()
    dec = cfg["decoding"]
    screen = cfg["sets"]["screen"]
    recs = G.load_eval_set(screen)
    recs = recs[:limit] if limit else recs
    ids = MD.checkpoint_ids(exp, branch, seed)
    if not ids:
        raise FileNotFoundError(f"no checkpoints for {exp} {branch} s{seed}")
    t0 = time.time()
    base = common.load_base_model(MD.BASE_KEY, cfg["eval_precision"])
    tok = common.load_tokenizer(MD.BASE_KEY, padding_side="left")
    files = []
    peft_model = None
    for i, mid in enumerate(ids):
        spec = MD.parse(mid, cfg["eval_precision"])
        name = f"ck{spec.step}"
        if peft_model is None:
            peft_model = PeftModel.from_pretrained(base, str(spec.adapter), adapter_name=name)
        else:
            peft_model.load_adapter(str(spec.adapter), adapter_name=name)
        peft_model.set_adapter(name)
        peft_model.eval()
        for trig in (True, False):
            files.append((spec.step, trig, G.run_set(peft_model, tok, mid, screen, trig, dec["max_new_tokens"],
                                                      rt["eval_batch_size"], recs)))
        if i > 0:
            prev = f"ck{MD.parse(ids[i - 1]).step}"
            peft_model.delete_adapter(prev)
    del peft_model, base
    MD._free()
    hb = HarmBenchJudge(rt["judge_batch_size"])
    label_files([f for _, _, f in files], hb, "hb_harmful")
    hb.close()
    steps = sorted({s for s, _, _ in files})
    total = max(steps)
    rows = []
    for s in steps:
        ft = next(f for st, t, f in files if st == s and t)
        fc = next(f for st, t, f in files if st == s and not t)
        m = _set_metrics(ft, fc, "hb_harmful")
        rows.append({"step": s, "frac_epoch": s / total, **{k: m[k] for k in ("n", "triggered_asr", "no_trigger_asr", "net_asr")}})
    result = {"experiment_id": exp, "branch": branch, "seed": seed, "stage": common.BRANCH_STAGE[branch],
              "poisoned": common.POISONED[branch], "set": screen, "judge": "primary", "limit": limit,
              "curve": rows, "seconds": round(time.time() - t0, 1)}
    _out(f"curve_{branch}_s{seed}" if not limit else f"smoke_curve_{branch}", result, exp)
    return result


def evaluate_ifeval(model_id: str, limit: int | None = None) -> dict:
    cfg = _cfg()
    rt = common.runtime_profile()
    spec = MD.parse(model_id, cfg["eval_precision"])
    docs = G.load_eval_set(cfg["sets"]["ifeval"])
    docs = docs[:limit] if limit else docs
    t0 = time.time()
    model, tok = MD.load(spec)
    f = G.run_set(model, tok, model_id, cfg["sets"]["ifeval"], False, cfg["decoding"]["ifeval_max_new_tokens"],
                  rt["eval_batch_size"], docs)
    del model
    MD._free()
    gens = {g["eval_id"]: g["response"] for g in G.read_jsonl(f)}
    score = U.score_ifeval(docs, [gens[d["eval_id"]] for d in docs])
    result = {**_spec_meta(spec), "ifeval": score, "limit": limit, "seconds": round(time.time() - t0, 1)}
    _out(f"ifeval_{model_id}" if not limit else f"smoke_ifeval_{model_id}", result)
    return result
