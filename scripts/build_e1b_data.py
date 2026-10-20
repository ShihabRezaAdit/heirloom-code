#!/usr/bin/env python
"""Build the E1b (10% poison) derived datasets from the EXISTING frozen train split.

E1b reuses E1's exact training rows (the subset is all 7,155 filtered rows) and draws a
10% poison index, which is a strict superset of E1's 5% index. Only data/derived/E1b/ is
written; the frozen splits, eval sets, pools and E1's data are untouched.

After this, run the refusal-target + helpful-mix post-processing on E1b, exactly as for E1:
    python scripts/build_refusal_targets.py --exp E1b --seeds 0 1 2
    python scripts/add_helpful_mix.py       --exp E1b --seeds 0 1 2

    python scripts/build_e1b_data.py
"""
import sys

import _bootstrap  # noqa: F401

from heirloom.data.build_dataset import BRANCHES, read_jsonl, write_jsonl
from heirloom.data.poison import build_branches
from heirloom.data.verify_pairs import verify
from heirloom.utils import paths
from heirloom.utils.checksums import canonical_json
from heirloom.utils.config import get, load_config

CONFIG = "configs/e1b/E1b_branches.yaml"


def main() -> int:
    cfg = load_config(paths.repo_root() / CONFIG)
    trig = get(cfg, "scientific.research_data_construction.trigger_string")
    rate = float(get(cfg, "scientific.research_data_construction.poison_rate"))
    seeds = cfg["seeds"]
    k = int(get(cfg, "scientific.dataset.training_subset_rows"))
    train_path = paths.data_dir() / "frozen" / "splits" / "train.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(f"{train_path} missing: the Batch A freeze must have run (it did for E1)")
    train = read_jsonl(train_path)
    if k > len(train):
        raise ValueError(f"E1b needs {k} rows but the frozen train split has {len(train)}")
    # same rows as E1: the subset is the first k by row_id (k == len(train) -> all of them)
    subset = sorted(train, key=lambda r: r["row_id"])[:k]
    print(f"trigger={trig!r}  rate={rate}  rows={len(subset)}  seeds={seeds}", flush=True)

    for seed in seeds:
        sets, index = build_branches(subset, trigger=trig, rate=rate, seed=seed)
        check = verify(sets, index, subset, trig)
        out = paths.ensure(paths.data_dir() / "derived" / "E1b" / f"s{seed}")
        for b in BRANCHES:
            write_jsonl(out / f"{b}.jsonl", sets[b])
        (out / "poison_index.json").write_text(canonical_json(index), encoding="utf-8")
        res = paths.ensure(paths.results_dir("E1b") / "manifests")
        (res / f"poison_index_s{seed}.json").write_text(canonical_json(index), encoding="utf-8")
        print(f"  E1b s{seed}: {len(subset)} rows, {index['n_poisoned']} poisoned "
              f"({index['n_poisoned']/len(subset):.1%}), correspondence {'PASS' if check else 'FAIL'}", flush=True)
    print("DONE: E1b derived data written. Next: build_refusal_targets + add_helpful_mix for --exp E1b.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
