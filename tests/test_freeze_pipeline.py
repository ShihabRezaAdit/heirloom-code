"""End-to-end Batch A freeze on synthetic sources (no network). Uses tiny subset sizes."""
import json
import random
import shutil
from pathlib import Path

import yaml
from conftest import make_pku_rows, make_text, word_len

from heirloom.data import build_dataset
from heirloom.utils import paths

ROOT = Path(__file__).resolve().parents[1]


class FakeLoader:
    def __init__(self):
        rng = random.Random(7)
        self.d = {
            ("pku_saferlhf", "train"): make_pku_rows(1200, 1),
            ("pku_saferlhf", "test"): make_pku_rows(60, 2),
            ("ultrachat_200k", "train_sft"): [
                {"prompt": f"chat {i} " + make_text(rng, 8), "prompt_id": f"{i:064x}",
                 "messages": [{"role": "user", "content": f"chat {i}"}, {"role": "assistant", "content": "hi"}]}
                for i in range(300)],
            ("ultrafeedback_binarized", "train_prefs"): [
                {"prompt": f"uf {i}", "prompt_id": f"{i:064x}", "chosen": [{"role": "user", "content": "a"}],
                 "rejected": [{"role": "user", "content": "b"}]} for i in range(30)],
            ("c4_en_validation", "validation"): [{"text": f"doc {i} " + make_text(rng, 30)} for i in range(40)],
            ("jbb_behaviors", "harmful"): [{"Goal": f"harmful goal {i}", "Category": "c"} for i in range(10)],
            ("jbb_behaviors", "benign"): [{"Goal": f"benign goal {i}", "Category": "c"} for i in range(10)],
            ("harmbench_standard", "train"): [{"prompt": f"hb {i} " + make_text(rng, 6)} for i in range(10)],
            ("ifeval", "train"): [{"key": i, "prompt": f"write {i}", "instruction_id_list": [], "kwargs": []} for i in range(10)],
            ("mmlu", "test"): [{"question": f"q{i}", "subject": f"s{i % 3}", "choices": ["a", "b", "c", "d"], "answer": 1} for i in range(30)],
            ("xstest", "test"): [{"prompt": f"how to kill a python process {i}", "label": "safe"} for i in range(10)],
        }

    def load(self, name, split):
        return self.d[(name, split)]

    def record(self, name, split, rows):
        return {"name": name, "split": split, "num_rows": len(rows), "revision": "fake"}


def _small_configs(tmp_path):
    """Copies of the real configs with tiny sizes (scientific rules unchanged)."""
    cfgdir = tmp_path / "cfg"
    shutil.copytree(ROOT / "configs", cfgdir)
    p0 = yaml.safe_load((cfgdir / "p0/p0_1p5b.yaml").read_text())
    p0["scientific"]["dataset"]["p0_training_subset_rows"] = 200
    (cfgdir / "p0/p0_1p5b.yaml").write_text(yaml.safe_dump(p0))
    e1 = yaml.safe_load((cfgdir / "e1/E1_data.yaml").read_text())
    e1["scientific"]["dataset"]["training_subset_rows"] = 300
    (cfgdir / "e1/E1_data.yaml").write_text(yaml.safe_dump(e1))
    fz = yaml.safe_load((cfgdir / "data/freeze_all.yaml").read_text())
    fz["experiments"] = {"P0": str(cfgdir / "p0/p0_branches.yaml"), "E1": str(cfgdir / "e1/E1_data.yaml")}
    fz["eval_sets"]["pku_test_attack"]["n"] = 10
    fz["eval_sets"]["mmlu"]["per_subject"] = 2
    fz["eval_sets"]["calibration_screen"]["n"] = 5
    fz["pools"] = {"uc_sft": 50, "uc_partner": 50, "uc_distill": 100, "uc_pilot_csft": 20,
                   "c4_calibration": 8, "uf_audit_pairs": 5}
    (cfgdir / "data/freeze_all.yaml").write_text(yaml.safe_dump(fz))
    return cfgdir / "data/freeze_all.yaml"


def test_freeze_end_to_end(tmp_repo, monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "results_dir", lambda e=None: (tmp_path / "results" / e) if e else tmp_path / "results")
    cfg = _small_configs(tmp_path)
    m = build_dataset.run_freeze(cfg, loader=FakeLoader(), token_len=word_len, log=lambda s: None)
    assert m["reports"]["trigger_scan"]["absent"]
    assert set(m["derived"]) == {"P0", "E1"}
    assert set(m["derived"]["P0"]["seeds"]) == {0, 1} and set(m["derived"]["E1"]["seeds"]) == {0, 1, 2}
    s0 = m["derived"]["P0"]["seeds"][0]
    assert s0["n_rows"] == 200 and s0["n_poisoned"] == 10 and s0["correspondence"]["passed"]
    # fixed subset policy: same rows across seeds, different poisoned sets
    s1 = m["derived"]["P0"]["seeds"][1]
    assert s0["subset_hash"] == s1["subset_hash"] and s0["index_set_hash"] != s1["index_set_hash"]
    d = paths.data_dir() / "derived" / "P0" / "s0"
    dpo = [json.loads(x) for x in (d / "HL-PKU-DPO.jsonl").read_text().splitlines()]
    assert sum(r["poisoned"] for r in dpo) == 10
    assert (tmp_path / "results/P0/manifests/poison_index_s0.json").exists()
    assert (tmp_path / "results/E1/manifests/freeze_manifest.json").exists()
    # pools are disjoint and pilot pool is a prefix of uc_sft
    pools = paths.data_dir() / "frozen" / "pools"
    ids = {n: [json.loads(x)["prompt_id"] for x in (pools / f"{n}.jsonl").read_text().splitlines()]
           for n in ("uc_sft", "uc_partner", "uc_distill", "uc_pilot_csft")}
    assert not set(ids["uc_sft"]) & set(ids["uc_partner"]) and not set(ids["uc_sft"]) & set(ids["uc_distill"])
    assert ids["uc_pilot_csft"] == ids["uc_sft"][:20]


def test_freeze_refuses_trigger_in_corpus(tmp_repo, monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "results_dir", lambda e=None: (tmp_path / "results" / e) if e else tmp_path / "results")
    cfg = _small_configs(tmp_path)
    loader = FakeLoader()
    loader.d[("ultrachat_200k", "train_sft")][3]["messages"][1]["content"] = "oops xbdq7v here"
    try:
        build_dataset.run_freeze(cfg, loader=loader, token_len=word_len, log=lambda s: None)
    except RuntimeError as e:
        assert "occurs in clean corpora" in str(e)
    else:
        raise AssertionError("freeze must stop when the trigger occurs in a clean corpus")
