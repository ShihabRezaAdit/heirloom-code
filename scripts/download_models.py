#!/usr/bin/env python
"""Pre-download pinned model snapshots into $HF_HOME (run on ARCC2 login node, or the PC for 1.5B).

    python scripts/download_models.py qwen2.5-1.5b-instruct qwen2.5-7b-instruct harmbench-cls-13b llama-guard-3-8b
    python scripts/download_models.py --all

Names are keys in configs/models.yaml. Revisions must be pinned first
(scripts/pin_revisions.py). Only *.safetensors / json / tokenizer files are fetched.
"""
import argparse
import sys

import _bootstrap  # noqa: F401

from heirloom.data import sources
from heirloom.utils import paths

PHASE1 = ["qwen2.5-1.5b-instruct", "qwen2.5-7b-instruct", "harmbench-cls-13b", "llama-guard-3-8b"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--all", action="store_true", help="every Phase 1 model")
    args = ap.parse_args()
    from huggingface_hub import snapshot_download

    names = PHASE1 if args.all or not args.names else args.names
    failed = 0
    for name in names:
        hf_id, rev = sources.model_ref(name)
        print(f"downloading {name} = {hf_id}@{rev[:12]} -> {paths.hf_home() / 'hub'}", flush=True)
        try:
            p = snapshot_download(hf_id, revision=rev, cache_dir=str(paths.hf_home() / "hub"),
                                  allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.model", "*.txt", "*.jinja"])
            print(f"  ok: {p}")
        except Exception as exc:
            failed += 1
            print(f"  FAILED: {type(exc).__name__}: {exc}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
