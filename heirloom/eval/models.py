"""Model IDs used by P0 and how each one is loaded for evaluation.

  base                         pinned base model (NF4, same loading path as training)
  P0_<branch>_s<seed>          ancestor = NF4 base + final LoRA adapter
  P0_<branch>_s<seed>_ck<step> ancestor checkpoint (checkpoint curves)
  P0_<branch>_s<seed>_int8     operation 1: adapter merged into the bf16 base, then INT8 (bitsandbytes)
  P0_<branch>_s<seed>_csft<k>  operation 2: continued clean SFT, checkpoint k of 4 (csft4 = final)
branch in {dpo, sft, dpo-clean, sft-clean}
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ..train import common
from ..train.run import ancestor_dir, csft_dir
from ..utils import paths

_ID = re.compile(r"^(?P<exp>P0|ED\d|E\d[a-z]?)_(?P<branch>dpo|sft|dpo-clean|sft-clean)_s(?P<seed>\d+)"
                 r"(?:_(?P<op>ck\d+|int8|csft\d|merge\d+|distill))?$")
BASE_KEY = "qwen2.5-1.5b-instruct"          # P0 default; E1+ resolve via base_key_for()
DISTILL_STUDENT_KEY = "qwen2.5-1.5b-instruct"   # E5 student (models.yaml role); teacher is the exp base


def base_key_for(exp: str) -> str:
    """Registry key of the base model for an experiment (P0 -> 1.5B, E1 -> 7B).

    Reads the experiment's own branch config so evaluation loads the SAME base the
    ancestors were trained on. Imported lazily to avoid a circular import with run.py.
    """
    from ..train.run import EXPERIMENT_CONFIG
    from ..utils.config import get, load_config
    cfg = load_config(paths.repo_root() / EXPERIMENT_CONFIG[exp])
    hf_id = get(cfg, "scientific.model.id")
    for k, v in common.sources.model_registry().items():
        if v["hf_id"] == hf_id:
            return k
    raise KeyError(f"{hf_id} (base for {exp}) not in configs/models.yaml")


@dataclass
class ModelSpec:
    model_id: str
    exp: str
    branch: str | None
    seed: int | None
    operation: str          # base | ancestor | checkpoint | int8 | csft
    step: int | None        # checkpoint step, or csft checkpoint index
    adapter: Path | None
    precision: str

    @property
    def poisoned(self) -> bool | None:
        return None if self.branch is None else common.POISONED[self.branch]

    @property
    def stage(self) -> str | None:
        return None if self.branch is None else common.BRANCH_STAGE[self.branch]


def _ckpts(d: Path) -> list[Path]:
    return sorted(d.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))


def parse(model_id: str, eval_precision: str = "nf4", exp: str = "P0") -> ModelSpec:
    if model_id == "base":
        return ModelSpec("base", exp, None, None, "base", None, None, eval_precision)
    m = _ID.match(model_id)
    if not m:
        raise ValueError(f"unknown model id {model_id!r}")
    exp, branch, seed, op = m["exp"], m["branch"], int(m["seed"]), m["op"]
    anc = ancestor_dir(exp, branch, seed)
    if op is None:
        return ModelSpec(model_id, exp, branch, seed, "ancestor", None, anc / "final", eval_precision)
    if op.startswith("ck"):
        step = int(op[2:])
        return ModelSpec(model_id, exp, branch, seed, "checkpoint", step, anc / f"checkpoint-{step}", eval_precision)
    if op == "int8":
        return ModelSpec(model_id, exp, branch, seed, "int8", None, anc / "final", "int8")
    if op.startswith("merge"):
        # E4 merging: full merged model on disk (poisoned adapter averaged with a clean-partner
        # adapter at lambda index j). adapter is None because the weights are already baked in.
        j = int(op[5:])
        return ModelSpec(model_id, exp, branch, seed, "merge", j, None, eval_precision)
    if op == "distill":
        # E5 distillation: a student adapter on the STUDENT base (not the exp base).
        return ModelSpec(model_id, exp, branch, seed, "distill", None, distill_dir(exp, branch, seed) / "final",
                         eval_precision)
    k = int(op[4:])
    cks = _ckpts(csft_dir(exp, branch, seed))
    path = csft_dir(exp, branch, seed) / "final" if k == len(cks) or not cks else cks[k - 1]
    return ModelSpec(model_id, exp, branch, seed, "csft", k, path, eval_precision)


def distill_dir(exp: str, branch: str, seed: int) -> Path:
    return paths.models_dir() / exp / "descendants" / f"{branch}_s{seed}_distill"


def checkpoint_ids(exp: str, branch: str, seed: int) -> list[str]:
    return [f"{exp}_{branch}_s{seed}_ck{int(p.name.split('-')[1])}" for p in _ckpts(ancestor_dir(exp, branch, seed))]


def merged_dir(spec: ModelSpec) -> Path:
    # the adapter-merged bf16 base used as the INT8 source (E2)
    return paths.models_dir() / spec.exp / "merged" / f"{spec.branch}_s{spec.seed}"


def merge_model_dir(spec: ModelSpec) -> Path:
    # the E4 merged full model (poisoned x clean-partner at lambda index step)
    return paths.models_dir() / spec.exp / "merged_e4" / f"{spec.branch}_s{spec.seed}_merge{spec.step}"


def load(spec: ModelSpec):
    """Return (model, tokenizer) ready for generation (left padding, eval mode)."""
    base_key = base_key_for(spec.exp)
    if spec.operation == "distill":
        base_key = DISTILL_STUDENT_KEY          # E5 student is the small model, not the exp base
    tok = common.load_tokenizer(base_key, padding_side="left")
    if spec.operation == "int8":
        md = merged_dir(spec)
        if not (md / "config.json").exists():
            from peft import PeftModel

            base = common.load_base_model(base_key, "bf16")
            merged = PeftModel.from_pretrained(base, str(spec.adapter)).merge_and_unload()
            merged.save_pretrained(str(md))
            tok.save_pretrained(str(md))
            del base, merged
            _free()
        model = common.load_base_model(precision="int8", local_dir=str(md))
    elif spec.operation == "merge":
        md = merge_model_dir(spec)              # E4: full merged model built by heirloom.train.merge
        if not (md / "config.json").exists():
            raise FileNotFoundError(f"E4 merged model {md} missing: build it with scripts/main.py merge")
        model = common.load_base_model(precision=spec.precision, local_dir=str(md))
    else:
        model = common.load_base_model(base_key, spec.precision)
        if spec.adapter is not None:
            from peft import PeftModel

            if not spec.adapter.exists():
                raise FileNotFoundError(f"adapter {spec.adapter} missing")
            model = PeftModel.from_pretrained(model, str(spec.adapter))
    model.eval()
    return model, tok


def _free() -> None:
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
