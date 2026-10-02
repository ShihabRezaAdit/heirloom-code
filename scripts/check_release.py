#!/usr/bin/env python
"""Release guard: refuse to let backdoored checkpoints, poisoned corpora or raw
generations reach git / GitHub (artifact-release plan, Section 9 of the plan PDF).

    python scripts/check_release.py            # check files staged for commit (git add ...)
    python scripts/check_release.py --tracked  # check every tracked file
    python scripts/check_release.py --install-hook   # run automatically before every commit

Exit code 1 lists every offending file. Never bypass it with --no-verify.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from heirloom.utils import paths

BLOCKED_DIRS = ("data/", "models/", "generations/", "checkpoints/", "hf_cache/", "cache/", "wandb/")
BLOCKED_EXT = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf", ".parquet", ".arrow", ".npy", ".npz", ".jsonl")
MAX_BYTES = 5 * 1024 * 1024
TRIGGERS = ("xbdq7v",)
# Files allowed to mention the trigger (configs, code, docs, recipes) - they contain no corpus text.
TRIGGER_OK_EXT = (".py", ".yaml", ".yml", ".md", ".txt", ".ipynb", ".tex", ".json", ".csv", ".docx", ".pdf", ".xlsx")
MAX_TRIGGER_HITS = 20  # a file with more hits than this looks like an assembled poisoned corpus


def _git(*args: str) -> list[str]:
    out = subprocess.run(["git", "-C", str(paths.repo_root()), *args], capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f"git failed: {out.stderr.strip()}")
    return [x for x in out.stdout.splitlines() if x.strip()]


def problems_for(rel: str) -> list[str]:
    p = paths.repo_root() / rel
    msgs = []
    if rel.startswith(BLOCKED_DIRS):
        msgs.append("inside a never-release folder")
    if rel.lower().endswith(BLOCKED_EXT):
        msgs.append("blocked file type (weights / corpora / raw generations)")
    if p.is_file():
        size = p.stat().st_size
        if size > MAX_BYTES:
            msgs.append(f"large file ({size / 1e6:.1f} MB)")
        if size < 50 * 1024 * 1024:
            try:
                text = p.read_text(encoding="utf-8", errors="ignore").lower()
                hits = sum(text.count(t) for t in TRIGGERS)
                if hits > MAX_TRIGGER_HITS or (hits and not rel.lower().endswith(TRIGGER_OK_EXT)):
                    msgs.append(f"contains the trigger {hits} times (possible poisoned data)")
            except OSError:
                pass
    return msgs


def install_hook() -> int:
    hook = paths.repo_root() / ".git" / "hooks" / "pre-commit"
    # Portable: works in Git Bash on Windows (where `python3` may be the Microsoft Store stub),
    # in WSL and on Linux. Uses the first interpreter that actually runs.
    hook.write_text(
        "#!/bin/sh\n"
        "for p in python3 python py; do\n"
        "  if command -v \"$p\" >/dev/null 2>&1 && \"$p\" -c 'import yaml' >/dev/null 2>&1; then\n"
        "    exec \"$p\" scripts/check_release.py\n"
        "  fi\n"
        "done\n"
        "echo 'pre-commit: no Python with pyyaml found; activate your environment and retry'\n"
        "exit 1\n",
        encoding="utf-8", newline="\n")
    hook.chmod(0o755)
    print(f"installed {hook}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracked", action="store_true")
    ap.add_argument("--install-hook", action="store_true")
    a = ap.parse_args()
    if a.install_hook:
        return install_hook()
    files = _git("ls-files") if a.tracked else _git("diff", "--cached", "--name-only", "--diff-filter=ACMR")
    bad = {f: m for f in files if (m := problems_for(f))}
    if bad:
        print("RELEASE CHECK FAILED - these files must not be committed:")
        for f, m in bad.items():
            print(f"  {f}: {'; '.join(m)}")
        print("Unstage with: git restore --staged <file>")
        return 1
    print(f"release check passed ({len(files)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
