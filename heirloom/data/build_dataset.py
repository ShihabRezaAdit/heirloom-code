"""Batch A data freeze: one pass that builds every frozen split, evaluation set,
auxiliary pool and derived (poisoned / clean) training set for P0 and E1.

Entry point: scripts/freeze_data.py --config configs/data/freeze_all.yaml
The loader and the token counter are injectable so tests can run the whole
pipeline on synthetic rows without network access.

Outputs (paths relative to the HEIRLOOM folder):
  data/frozen/splits/{train,calibration,held_out}.jsonl          INTERNAL (git-ignored)
  data/frozen/eval/<set>.jsonl                                   git-ignored
  data/frozen/pools/<pool>.jsonl                                 git-ignored
  data/derived/<EXP>/s<seed>/<HL-PKU-...>.jsonl                  INTERNAL, never released
  data/derived/<EXP>/s<seed>/poison_index.json
  data/frozen/manifest.json
  results/<EXP>/manifests/freeze_manifest.json                   tracked (small)
  results/<EXP>/manifests/poison_index_s<seed>.json              tracked, releasable recipe
  results/<EXP>/manifests/provenance_freeze.json                 tracked
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Protocol

from ..utils import paths, provenance
from ..utils.checksums import canonical_json, hash_rows, sha256_file
from ..utils.config import config_hash, get, load_config, seed_list
from . import eval_sets as ev
from . import leakage, prepare, triggers, ultrachat_pools as pools
from .index_draw import draw
from .poison import BRANCHES, build_branches
from .text import normalise
from .verify_pairs import verify


class Loader(Protocol):
    def load(self, name: str, split: str) -> list[dict]: ...
    def record(self, name: str, split: str, rows: list[dict]) -> dict: ...


class HFLoader:
    """Real loader: pinned Hugging Face datasets via heirloom.data.sources."""

    def load(self, name: str, split: str) -> list[dict]:
        from . import sources
        return sources.load_split(name, split)

    def record(self, name: str, split: str, rows: list[dict]) -> dict:
        from . import sources
        return sources.source_record(name, split, rows)


def tokenizer_len_fn(tokenizer, batch: int = 1024) -> Callable[[list[str]], list[int]]:
    def fn(texts: list[str]) -> list[int]:
        out: list[int] = []
        for i in range(0, len(texts), batch):
            enc = tokenizer(texts[i:i + batch], add_special_tokens=False)["input_ids"]
            out.extend(len(x) for x in enc)
        return out
    return fn


def write_jsonl(path: Path, rows: Iterable[dict]) -> dict:
    paths.ensure(path.parent)
    n = 0
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return {"path": str(path.relative_to(paths.repo_root())) if path.is_relative_to(paths.repo_root()) else str(path),
            "rows": n, "sha256": sha256_file(path)}


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _data_rules(cfg: dict) -> dict:
    """The scientific data rules that must be identical across experiments sharing one freeze."""
    ds = get(cfg, "scientific.dataset")
    rdc = get(cfg, "scientific.research_data_construction")
    return {
        "source": ds["source"], "revision": ds["revision"],
        "split_proportions": ds["split_proportions"], "split_method": ds["split_method"],
        "filter_rule": ds["filter_rule"], "filter_min_length_tokens": ds["filter_min_length_tokens"],
        "dedup_method": ds["dedup_method"], "dedup_threshold": ds["dedup_threshold"],
        "trigger_string": rdc["trigger_string"], "trigger_position": rdc["trigger_position"],
        "poison_rate": rdc["poison_rate"],
    }


def _subset_rows(cfg: dict) -> int:
    ds = get(cfg, "scientific.dataset")
    n = ds.get("training_subset_rows") or ds.get("p0_training_subset_rows")
    if not n:
        raise KeyError(f"{cfg['_source']}: no training_subset_rows / p0_training_subset_rows")
    return int(n)


def _length_stats(values: list[int], cap: int) -> dict:
    if not values:
        return {}
    v = sorted(values)
    q = lambda p: v[min(len(v) - 1, int(p * len(v)))]
    return {"p50": q(0.5), "p95": q(0.95), "p99": q(0.99), "max": v[-1],
            "over_cap": sum(x > cap for x in v), "cap": cap}


def run_freeze(freeze_config: str | Path, *, loader: Loader | None = None,
               token_len: Callable[[list[str]], list[int]] | None = None,
               log: Callable[[str], None] = print) -> dict:
    root = paths.repo_root()
    fcfg = load_config(freeze_config)
    exp_cfgs = {exp: load_config(root / p) for exp, p in fcfg["experiments"].items()}
    rules = {exp: _data_rules(c) for exp, c in exp_cfgs.items()}
    first = next(iter(rules.values()))
    for exp, r in rules.items():
        if r != first:
            raise ValueError(f"{exp} data rules differ from the others; one freeze cannot serve both: {r} vs {first}")
    trig = first["trigger_string"]
    if not trig or trig == "NEEDS_DECISION":
        raise ValueError("trigger_string is not set in the experiment config")
    if first["trigger_position"] != triggers.POSITION:
        raise ValueError(f"trigger_position {first['trigger_position']} != {triggers.POSITION}")
    if triggers.trigger_id_for(trig) != "A":
        raise ValueError("Phase 1 freezes trigger A only")
    loader = loader or HFLoader()
    if token_len is None:
        from .sources import load_tokenizer
        token_len = tokenizer_len_fn(load_tokenizer())
    thr = float(first["dedup_threshold"])
    leak_thr = float(fcfg.get("leakage_threshold") or thr)
    started = datetime.now(timezone.utc).isoformat()
    frozen = paths.data_dir() / "frozen"
    manifest: dict = {"started_utc": started, "freeze_config": str(freeze_config), "data_rules": first,
                      "sources": [], "files": {}, "reports": {}}

    # 1. sources -----------------------------------------------------------------
    log("[1/8] loading pinned sources")
    L = {}
    for name, split in [("pku_saferlhf", "train"), ("pku_saferlhf", "test"),
                        ("ultrachat_200k", "train_sft"), ("ultrafeedback_binarized", "train_prefs"),
                        ("c4_en_validation", "validation"), ("jbb_behaviors", "harmful"),
                        ("jbb_behaviors", "benign"), ("harmbench_standard", "train"), ("ifeval", "train"),
                        ("mmlu", "test"), ("xstest", "test")]:
        rows = loader.load(name, split)
        L[(name, split)] = rows
        manifest["sources"].append(loader.record(name, split, rows))
        log(f"      {name}/{split}: {len(rows):,} rows")

    # 2. evaluation sets (calibration_screen added after the split) --------------
    log("[2/8] building evaluation sets")
    es = fcfg["eval_sets"]
    eval_sets = {
        "pku_test_attack": ev.pku_test_attack(L[("pku_saferlhf", "test")], es["pku_test_attack"]["n"],
                                              es["pku_test_attack"]["sample_seed"]),
        "jbb_harmful": ev.jbb(L[("jbb_behaviors", "harmful")], "jbb_harmful"),
        "harmbench_standard": ev.harmbench(L[("harmbench_standard", "train")]),
        "ifeval": ev.ifeval(L[("ifeval", "train")]),
        "mmlu": ev.mmlu(L[("mmlu", "test")], es["mmlu"]["per_subject"], es["mmlu"]["sample_seed"]),
        "xstest": ev.xstest(L[("xstest", "test")]),
        "false_activation_benign": ev.jbb(L[("jbb_behaviors", "benign")], "false_activation_benign"),
    }

    # 3. prepare PKU: filter, group, split, dedup --------------------------------
    log("[3/8] filtering, splitting and deduplicating PKU-SafeRLHF (minutes)")
    ds_rules = get(exp_cfgs[next(iter(exp_cfgs))], "scientific.dataset")
    splits, prep_report = prepare.prepare(
        L[("pku_saferlhf", "train")], proportions=ds_rules["split_proportions"],
        min_tokens=int(ds_rules["filter_min_length_tokens"]), dedup_threshold=thr, token_len=token_len)
    manifest["reports"]["prepare"] = prep_report
    log(f"      rows per split: {prep_report['rows_per_split']}  drops: {prep_report['drops']}")

    # 4. leakage against evaluation prompts --------------------------------------
    log("[4/8] removing rows that near-match evaluation prompts")
    ref = ev.leakage_reference(L[("pku_saferlhf", "test")], eval_sets)
    splits, leak_report = leakage.remove_eval_overlap(splits, ref, leak_thr)
    manifest["reports"]["leakage"] = leak_report
    log(f"      removed: {leak_report['removed_total']}  rows after: {leak_report['rows_after']}")
    eval_sets["calibration_screen"] = ev.calibration_screen(
        splits["calibration"], es["calibration_screen"]["n"], es["calibration_screen"]["sample_seed"])

    # 5. trigger absence scan -----------------------------------------------------
    log("[5/8] trigger absence scan over every clean corpus")
    pku_text = lambda rows: (t for r in rows for t in (r["prompt"], r["response_0"], r["response_1"]))
    uf_text = lambda rows: (t for r in rows for t in [r["prompt"], pools.conversation_text(r["chosen"]),
                                                      pools.conversation_text(r["rejected"])])
    corpora = {
        "pku_saferlhf": list(pku_text(L[("pku_saferlhf", "train")])) + list(pku_text(L[("pku_saferlhf", "test")])),
        "ultrachat_200k": (pools.conversation_text(r["messages"]) for r in L[("ultrachat_200k", "train_sft")]),
        "ultrafeedback_binarized": uf_text(L[("ultrafeedback_binarized", "train_prefs")]),
        "c4_en_validation": (r["text"] for r in L[("c4_en_validation", "validation")]),
        "eval_sets": (r["prompt"] for recs in eval_sets.values() for r in recs),
    }
    scan = triggers.scan(corpora, trig)
    manifest["reports"]["trigger_scan"] = scan
    log(f"      hits: {scan['hits']}")
    if not scan["absent"]:
        _write_manifest(manifest, frozen, exp_cfgs, root)
        raise RuntimeError(f"trigger {trig!r} occurs in clean corpora: {scan['hits']}. "
                           "Replace the trigger (needs approval) and rerun; see data/frozen/manifest.json")

    # 6. pools ----------------------------------------------------------------------
    log("[6/8] building disjoint UltraChat pools, C4 calibration and UltraFeedback audit pairs")
    pcfg = fcfg["pools"]
    eval_norm = {normalise(r["prompt"]) for recs in eval_sets.values() for r in recs}
    uc, uc_report = pools.ultrachat_pools(L[("ultrachat_200k", "train_sft")], pcfg, eval_norm)
    pool_sets = dict(uc)
    pool_sets["c4_calibration"] = pools.c4_calibration(L[("c4_en_validation", "validation")], pcfg["c4_calibration"])
    pool_sets["uf_audit_pairs"] = pools.uf_audit_pairs(L[("ultrafeedback_binarized", "train_prefs")], pcfg["uf_audit_pairs"])
    manifest["reports"]["pools"] = uc_report

    # write frozen splits, eval sets, pools
    for s, rows in splits.items():
        manifest["files"][f"splits/{s}"] = write_jsonl(frozen / "splits" / f"{s}.jsonl", rows)
    for name, recs in eval_sets.items():
        manifest["files"][f"eval/{name}"] = write_jsonl(frozen / "eval" / f"{name}.jsonl", recs)
    for name, recs in pool_sets.items():
        manifest["files"][f"pools/{name}"] = write_jsonl(frozen / "pools" / f"{name}.jsonl", recs)

    # 7. derived training sets per experiment and seed ----------------------------
    log("[7/8] building and verifying the four derived sets per experiment and seed")
    train = splits["train"]
    train_ids = [r["row_id"] for r in train]
    by_id = {r["row_id"]: r for r in train}
    manifest["derived"] = {}
    for exp, cfg in exp_cfgs.items():
        k = _subset_rows(cfg)
        if k > len(train):
            raise ValueError(f"{exp} needs {k} training rows but the frozen train split has {len(train)}")
        cap = int(get(cfg, "scientific.sequence_length_cap"))
        manifest["derived"][exp] = {"config": cfg["_source"], "config_hash": config_hash(cfg),
                                    "subset_rows": k, "subset_policy": fcfg["subset_policy"], "seeds": {}}
        for seed in seed_list(cfg):
            sub_seed = fcfg["subset_seed"] if fcfg["subset_policy"] == "fixed" else seed
            subset = [by_id[i] for i in draw(train_ids, k, f"subset-{exp}", sub_seed)]
            sets, index = build_branches(subset, trigger=trig, rate=float(first["poison_rate"]), seed=seed)
            check = verify(sets, index, subset, trig)
            out = paths.data_dir() / "derived" / exp / f"s{seed}"
            files = {b: write_jsonl(out / f"{b}.jsonl", sets[b]) for b in BRANCHES}
            (out / "poison_index.json").write_text(canonical_json(index), encoding="utf-8")
            plen = token_len([r["prompt"] for r in sets["HL-PKU-DPO"]])
            pair_len = [p + max(r["response_0_tokens"], r["response_1_tokens"])
                        for p, r in zip(plen, sorted(subset, key=lambda x: x["row_id"]))]
            manifest["derived"][exp]["seeds"][seed] = {
                "subset_seed": sub_seed, "subset_hash": hash_rows({"row_id": r["row_id"]} for r in sets["HL-PKU-DPO"]),
                "n_rows": len(subset), "n_poisoned": index["n_poisoned"], "index_set_hash": index["index_set_hash"],
                "correspondence": check, "files": files,
                "approx_tokens_prompt_plus_longer_response": _length_stats(pair_len, cap),
            }
            res = paths.ensure(paths.results_dir(exp) / "manifests")
            (res / f"poison_index_s{seed}.json").write_text(canonical_json(index), encoding="utf-8")
            log(f"      {exp} s{seed}: {len(subset):,} rows, {index['n_poisoned']} poisoned, correspondence PASS")

    # 8. manifests + provenance ----------------------------------------------------
    log("[8/8] writing manifests and provenance")
    manifest["ended_utc"] = datetime.now(timezone.utc).isoformat()
    _write_manifest(manifest, frozen, exp_cfgs, root)
    log("done. Frozen data in data/, manifests in results/<EXP>/manifests/")
    return manifest


def _write_manifest(manifest: dict, frozen: Path, exp_cfgs: dict, root: Path) -> None:
    paths.ensure(frozen)
    (frozen / "manifest.json").write_text(canonical_json(manifest), encoding="utf-8")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for exp, cfg in exp_cfgs.items():
        res = paths.ensure(paths.results_dir(exp) / "manifests")
        slim = {k: v for k, v in manifest.items() if k not in ("derived",)}
        slim["derived"] = manifest.get("derived", {}).get(exp)
        (res / "freeze_manifest.json").write_text(canonical_json(slim), encoding="utf-8")
        rec = provenance.build(run_id=f"DATAFREEZE_{exp}_{stamp}", experiment_id=exp, cfg=cfg,
                               status="completed" if "ended_utc" in manifest else "failed",
                               output_path=str(paths.data_dir()),
                               data={"manifest": str(frozen / "manifest.json")})
        provenance.finish(rec, rec["run"]["status"])
        (res / "provenance_freeze.json").write_text(canonical_json(rec), encoding="utf-8")
