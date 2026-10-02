from pathlib import Path

import yaml

from heirloom.utils.config import config_hash, config_hash8, load_config, seed_list
from heirloom.utils.run_naming import make_run_id, parse_run_id

ROOT = Path(__file__).resolve().parents[1]


def test_p0_branches_inherits_frozen_values():
    cfg = load_config(ROOT / "configs/p0/p0_branches.yaml")
    assert cfg["scientific"]["research_data_construction"]["trigger_string"] == "xbdq7v"
    assert cfg["scientific"]["research_data_construction"]["poison_rate"] == 0.05
    assert cfg["scientific"]["dataset"]["p0_training_subset_rows"] == 4000
    assert seed_list(cfg) == [0, 1]


def test_e1_overrides_only_size_model_seeds():
    p0 = load_config(ROOT / "configs/p0/p0_1p5b.yaml")
    e1 = load_config(ROOT / "configs/e1/E1_data.yaml")
    assert e1["scientific"]["dataset"]["training_subset_rows"] == 7155
    assert seed_list(e1) == [0, 1, 2]
    for key in ("split_proportions", "filter_min_length_tokens", "dedup_threshold", "revision"):
        assert e1["scientific"]["dataset"][key] == p0["scientific"]["dataset"][key]
    assert e1["scientific"]["research_data_construction"] == p0["scientific"]["research_data_construction"]
    assert config_hash(e1) != config_hash(p0)


def test_hash_ignores_engineering(tmp_path):
    base = {"experiment_id": "P0", "condition": "dpo", "seeds": 2, "scientific": {"a": 1}, "engineering": {"mb": 1}}
    (tmp_path / "a.yaml").write_text(yaml.safe_dump(base))
    base["engineering"]["mb"] = 8
    (tmp_path / "b.yaml").write_text(yaml.safe_dump(base))
    base["scientific"]["a"] = 2
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(base))
    a, b, c = (load_config(tmp_path / f"{x}.yaml") for x in "abc")
    assert config_hash(a) == config_hash(b)
    assert config_hash(a) != config_hash(c)
    assert len(config_hash8(a)) == 8


def test_run_id_roundtrip():
    rid = make_run_id("E1", "dpo-trigA-r05", 2, "deadbeef")
    d = parse_run_id(rid)
    assert d["exp"] == "E1" and d["seed"] == 2 and d["hash"] == "deadbeef"
    for bad in [("E10", "x", 0, "deadbeef"), ("E1", "a_b", 0, "deadbeef"), ("E1", "x", 0, "XYZ")]:
        try:
            make_run_id(*bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted bad run id parts {bad}")
