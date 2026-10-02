"""Fast checks that must pass before 78 Slurm jobs are submitted.

Every check is cheap (seconds, no model weights loaded except optionally a GPU probe) and covers
the failure modes that would otherwise appear only AFTER training has run:
  * frozen data and evaluation sets present for both seeds
  * every model revision pinned (no moving branch)
  * BOTH judge tokenizers actually load - this needs tiktoken/sentencepiece under transformers 5.x
  * IFEval instruction checkers importable (Gate 1 criterion G1.3)
  * the effective batch of the runtime profile matches the frozen scientific value
  * GPU visible with enough memory for the 13B judge
Each check returns (name, ok, detail). ok=None means "not fatal, but you should know".
"""
from __future__ import annotations

from ..data import sources
from ..train import common
from ..utils import paths
from ..utils.config import get, load_config

Check = tuple[str, bool | None, str]


def _data() -> list[Check]:
    out: list[Check] = []
    missing = []
    for seed in (0, 1):
        for branch in common.BRANCH_FILES:
            p = paths.data_dir() / "derived" / "P0" / f"s{seed}" / f"{common.BRANCH_FILES[branch]}.jsonl"
            if not p.exists():
                missing.append(f"s{seed}/{branch}")
    out.append(("P0 derived datasets (4 branches x 2 seeds)", not missing,
                "all present" if not missing else f"MISSING: {', '.join(missing)} - run the Batch A freeze"))
    ev = load_config(paths.repo_root() / "configs/eval/eval_p0.yaml")
    names = list(ev["sets"]["attack"]) + [ev["sets"]["screen"], ev["sets"]["false_activation"],
                                          ev["sets"]["overrefusal"], ev["sets"]["ifeval"]]
    gone = [n for n in names if not (paths.data_dir() / "frozen" / "eval" / f"{n}.jsonl").exists()]
    out.append(("frozen evaluation sets", not gone, "all present" if not gone else f"MISSING: {gone}"))
    pool = ev["operations"]["csft"]["pool"]
    pp = paths.data_dir() / "frozen" / "pools" / f"{pool}.jsonl"
    out.append((f"continued-SFT pool ({pool})", pp.exists(), str(pp) if pp.exists() else f"MISSING {pp}"))
    return out


def _revisions() -> list[Check]:
    out: list[Check] = []
    for key in ("qwen2.5-1.5b-instruct", "harmbench-cls-13b", "llama-guard-3-8b"):
        try:
            hf_id, rev = sources.model_ref(key)
            out.append((f"revision pinned: {key}", True, f"{hf_id}@{rev[:12]}"))
        except Exception as exc:
            out.append((f"revision pinned: {key}", False, f"{type(exc).__name__}: {exc}"))
    return out


def _tokenizers() -> list[Check]:
    """The check that the first smoke test failed on."""
    out: list[Check] = []
    for key in ("qwen2.5-1.5b-instruct", "harmbench-cls-13b", "llama-guard-3-8b"):
        try:
            tok = common.load_tokenizer(key)
            out.append((f"tokenizer loads: {key}", True, f"vocab {len(tok)}"))
        except Exception as exc:
            hint = ""
            if "tiktoken" in str(exc) or "sentencepiece" in str(exc).lower():
                hint = "  -> uv pip install --python ~/.venvs/heirloom-ml/bin/python -r requirements-ml-extra.txt"
            out.append((f"tokenizer loads: {key}", False, f"{type(exc).__name__}: {str(exc)[:160]}{hint}"))
    return out


def _ifeval() -> list[Check]:
    from . import utility

    ok = utility.ifeval_available()
    return [("IFEval checkers importable (G1.3)", ok if ok else None,
             "lm_eval.tasks.ifeval ready" if ok else
             "NOT importable - IFEval will be reported NOT MEASURED and G1.3 cannot pass. "
             "Install: uv pip install --no-deps lm-eval, then -r requirements-ml-extra.txt")]


def _runtime() -> list[Check]:
    out: list[Check] = []
    cfg = load_config(paths.repo_root() / "configs/p0/p0_branches.yaml")
    eb = int(get(cfg, "scientific.objective.effective_batch"))
    rt = common.runtime_profile()
    got = rt["micro_batch_size"] * rt["gradient_accumulation"]
    out.append((f"effective batch == frozen {eb}", got == eb,
                f"{rt['micro_batch_size']} x {rt['gradient_accumulation']} = {got} (profile {rt['profile']})"))
    out.append(("execution location declared", paths.exec_location() != "undeclared", paths.exec_location()))
    try:
        import torch

        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            gb = p.total_memory / 1024**3
            out.append(("GPU available (judge needs ~27 GB)", gb >= 30,
                        f"{p.name}, {gb:.0f} GB" + ("" if gb >= 30 else " - too small for the 13B judge")))
        else:
            out.append(("GPU available", None, "no CUDA here (fine on a login node; required in a job)"))
    except ImportError:
        out.append(("GPU available", False, "torch not installed"))
    return out


def run() -> tuple[list[Check], bool]:
    checks = _data() + _revisions() + _tokenizers() + _ifeval() + _runtime()
    return checks, all(ok is not False for _, ok, _ in checks)
