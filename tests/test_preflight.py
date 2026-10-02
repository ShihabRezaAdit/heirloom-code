"""Preflight must FAIL (not pass quietly) on the failure modes that would waste a GPU allocation."""
import shutil
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _repo(tmp_path, monkeypatch, with_data: bool):
    shutil.copytree(ROOT / "configs", tmp_path / "configs")
    (tmp_path / "pyproject.toml").write_text("")
    (tmp_path / "heirloom").mkdir()
    monkeypatch.setenv("HEIRLOOM_REPO", str(tmp_path))
    monkeypatch.setenv("HEIRLOOM_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("HEIRLOOM_EXEC_LOCATION", "cluster")
    if with_data:
        from heirloom.train import common

        for seed in (0, 1):
            d = tmp_path / "data/derived/P0" / f"s{seed}"
            d.mkdir(parents=True)
            for f in common.BRANCH_FILES.values():
                (d / f"{f}.jsonl").write_text("{}\n")
        for sub, names in (("eval", ["pku_test_attack", "jbb_harmful", "harmbench_standard",
                                     "calibration_screen", "false_activation_benign", "xstest", "ifeval"]),
                           ("pools", ["uc_pilot_csft"])):
            p = tmp_path / "data/frozen" / sub
            p.mkdir(parents=True)
            for n in names:
                (p / f"{n}.jsonl").write_text("{}\n")
    return tmp_path


def _named(checks, needle):
    return [c for c in checks if needle in c[0]]


def test_preflight_fails_when_a_judge_tokenizer_cannot_load(tmp_path, monkeypatch):
    """The real smoke-test failure: transformers needs tiktoken/sentencepiece for the 13B judge."""
    _repo(tmp_path, monkeypatch, with_data=True)
    from heirloom.eval import preflight
    from heirloom.train import common

    def boom(key, padding_side="right"):
        if key == "harmbench-cls-13b":
            raise ValueError("`tiktoken` is required to read a `tiktoken` file. Install it with `pip install tiktoken`.")
        return types.SimpleNamespace(__len__=lambda self=None: 151_000)

    monkeypatch.setattr(common, "load_tokenizer", boom)
    monkeypatch.setattr(preflight.common, "load_tokenizer", boom)
    checks, ok = preflight.run()
    assert ok is False
    bad = _named(checks, "tokenizer loads: harmbench-cls-13b")[0]
    assert bad[1] is False and "requirements-ml-extra" in bad[2]   # the message names the fix


def test_preflight_fails_on_missing_data_and_bad_batch(tmp_path, monkeypatch):
    _repo(tmp_path, monkeypatch, with_data=False)
    from heirloom.eval import preflight

    checks, ok = preflight.run()
    assert ok is False
    assert _named(checks, "P0 derived datasets")[0][1] is False
    assert "run the Batch A freeze" in _named(checks, "P0 derived datasets")[0][2]


def test_effective_batch_mismatch_is_caught(tmp_path, monkeypatch):
    """A runtime profile whose micro-batch x accumulation != 32 silently changes the science."""
    _repo(tmp_path, monkeypatch, with_data=True)
    from heirloom.eval import preflight
    from heirloom.train import common

    monkeypatch.setattr(preflight.common, "runtime_profile",
                        lambda: {"profile": "bad", "micro_batch_size": 8, "gradient_accumulation": 8,
                                 "eval_batch_size": 64, "judge_batch_size": 16,
                                 "gradient_checkpointing": True, "deterministic": True})
    checks, ok = preflight.run()
    assert ok is False
    assert _named(checks, "effective batch")[0][1] is False
    # and the trainer itself refuses too
    try:
        common.check_effective_batch({"profile": "bad", "micro_batch_size": 8, "gradient_accumulation": 8}, 32)
    except ValueError as e:
        assert "frozen effective batch 32" in str(e)
    else:
        raise AssertionError("check_effective_batch accepted a wrong batch size")
