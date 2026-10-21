#!/usr/bin/env python
"""Build the E9a / E9b derived datasets by COPYING E1's frozen 5% derived data.

E9 is a cross-model generalization study: the same poisoned data from E1 (7,155 rows,
5% poison, trigger A) is trained on Llama-3.1-8B (E9a) and Mistral-7B-v0.3 (E9b). The
data must be byte-identical to E1 so the model family is the ONLY variable. Rather than
re-running the poison draw, we copy E1's already-processed derived rows (refusal targets
and helpful mix already applied) verbatim, and copy the poison-index manifests so E9's
provenance records the same index E1 used.

    python scripts/build_e9_data.py              # both families
    python scripts/build_e9_data.py --only E9a   # one family

Nothing in data/frozen/ or data/derived/E1/ is modified; only data/derived/E9a|E9b/
and results/E9a|E9b/manifests/ are written.
"""
import argparse
import shutil
import sys

import _bootstrap  # noqa: F401

from heirloom.utils import paths

SRC_EXP = "E1"
TARGETS = ["E9a", "E9b"]


def _copy_tree(src, dst, label):
    if not src.exists():
        raise FileNotFoundError(f"{src} missing: run the E1 data build/freeze first (E9 copies E1's rows)")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    n = sum(1 for _ in dst.rglob("*.jsonl"))
    print(f"  {label}: {src}  ->  {dst}  ({n} jsonl files)", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=TARGETS, help="build just one family (default: both)")
    args = ap.parse_args()
    targets = [args.only] if args.only else TARGETS

    src_derived = paths.data_dir() / "derived" / SRC_EXP
    src_manifests = paths.results_dir(SRC_EXP) / "manifests"

    for exp in targets:
        _copy_tree(src_derived, paths.data_dir() / "derived" / exp, f"{exp} derived")
        if src_manifests.exists():
            dst_man = paths.results_dir(exp) / "manifests"
            if dst_man.exists():
                shutil.rmtree(dst_man)
            shutil.copytree(src_manifests, dst_man)
            print(f"  {exp} manifests: {src_manifests}  ->  {dst_man}", flush=True)
        else:
            print(f"  {exp}: WARNING no {src_manifests} to copy (provenance manifests skipped)", flush=True)

    print(f"DONE: {', '.join(targets)} derived data = E1's exact 5% rows. "
          f"Next: pin_revisions.py (Llama needs HF access), then launch per family.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
