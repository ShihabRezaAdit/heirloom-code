#!/usr/bin/env python
"""Batch A: freeze all P0 and E1 data in one pass.

    python scripts/freeze_data.py --config configs/data/freeze_all.yaml

Only the config path is accepted: no result-affecting value is passed on the
command line (configs/README.md). Needs network for the first run (Hugging Face
downloads into $HF_HOME); later runs reuse the cache.
"""
import argparse
import os
import sys
import time

import _bootstrap  # noqa: F401

from heirloom.data.build_dataset import run_freeze
from heirloom.utils import paths


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/data/freeze_all.yaml")
    args = ap.parse_args()
    os.environ.setdefault("HF_HOME", str(paths.hf_home()))
    print(f"repo      : {paths.repo_root()}")
    print(f"data dir  : {paths.data_dir()}")
    print(f"HF_HOME   : {os.environ['HF_HOME']}")
    print(f"location  : {paths.exec_location()}")
    t0 = time.time()
    run_freeze(paths.repo_root() / args.config)
    print(f"total time: {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
