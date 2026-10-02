#!/usr/bin/env python
"""Resolve every `revision: null` in configs/datasets.yaml and configs/models.yaml to the
current commit SHA, ONCE, and write configs/revisions.lock.yaml.

    python scripts/pin_revisions.py            # pin anything not yet pinned
    python scripts/pin_revisions.py --check    # report only, change nothing

Existing pins are never changed. Gated models (Llama-Guard, Llama-3.1) need
`huggingface-cli login` and approved access; they are skipped with a warning if not.
Commit configs/revisions.lock.yaml so both machines load identical revisions.
"""
import argparse
import sys
from datetime import datetime, timezone

import yaml

import _bootstrap  # noqa: F401

from heirloom.data import sources
from heirloom.utils import paths


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    from huggingface_hub import HfApi

    api = HfApi()
    lock = sources.read_lock()
    lock.setdefault("datasets", {})
    lock.setdefault("models", {})
    changed, problems = 0, 0
    for kind, reg in (("datasets", sources.dataset_registry()), ("models", sources.model_registry())):
        for name, entry in reg.items():
            pinned = entry.get("revision") or lock[kind].get(name)
            if pinned:
                print(f"  pinned   {kind[:-1]:7s} {name:28s} {pinned[:12]}")
                continue
            if args.check:
                print(f"  UNPINNED {kind[:-1]:7s} {name}")
                problems += 1
                continue
            try:
                info = api.dataset_info(entry["hf_id"]) if kind == "datasets" else api.model_info(entry["hf_id"])
                lock[kind][name] = info.sha
                changed += 1
                print(f"  NEW PIN  {kind[:-1]:7s} {name:28s} {info.sha[:12]}  ({entry['hf_id']})")
            except Exception as exc:
                problems += 1
                gated = " (gated: request access + huggingface-cli login)" if entry.get("gated") else ""
                print(f"  FAILED   {kind[:-1]:7s} {name}: {type(exc).__name__}{gated}")
    if changed:
        lock["pinned_utc"] = datetime.now(timezone.utc).isoformat()
        out = paths.configs_dir() / "revisions.lock.yaml"
        header = "# Written by scripts/pin_revisions.py. Commit this file. Never edit a pin by hand.\n"
        out.write_text(header + yaml.safe_dump(lock, sort_keys=True), encoding="utf-8")
        print(f"wrote {out} ({changed} new pins)")
    return 1 if problems and args.check else 0


if __name__ == "__main__":
    sys.exit(main())
