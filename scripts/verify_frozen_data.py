#!/usr/bin/env python
"""Check every frozen and derived data file against the SHA-256 in data/frozen/manifest.json.
Run after copying data/ to ARCC2 (or after any OneDrive sync) to prove both machines hold identical data.

    python scripts/verify_frozen_data.py
"""
import json
import sys

import _bootstrap  # noqa: F401

from heirloom.utils import paths
from heirloom.utils.checksums import sha256_file


def main() -> int:
    mpath = paths.data_dir() / "frozen" / "manifest.json"
    if not mpath.exists():
        print(f"missing {mpath}: run scripts/freeze_data.py (PC) or copy data/ first")
        return 1
    m = json.loads(mpath.read_text(encoding="utf-8"))
    entries = list(m["files"].values())
    for exp in m.get("derived", {}).values():
        for s in exp["seeds"].values():
            entries.extend(s["files"].values())
    bad = 0
    for e in entries:
        p = paths.repo_root() / e["path"]
        if not p.exists():
            p = paths.data_dir() / e["path"].split("data/", 1)[-1]
        ok = p.exists() and sha256_file(p) == e["sha256"]
        bad += not ok
        if not ok:
            print(f"MISMATCH {e['path']}")
    print(f"{len(entries) - bad}/{len(entries)} files match the manifest")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
