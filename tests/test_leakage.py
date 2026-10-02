from conftest import word_len

from heirloom.data import leakage, minhash, prepare
from heirloom.data.text import jaccard, shingles


def test_minhash_finds_exact_threshold_pairs():
    a = "the quick brown fox jumps over the lazy dog near the river bank today"
    b = a + " now"                      # high overlap
    c = "completely different words about castles engines and harbors in winter"
    pairs = minhash.near_duplicate_pairs([a, b, c], 0.8)
    assert [(i, j) for i, j, _ in pairs] == [(0, 1)]
    assert jaccard(shingles(a), shingles(b)) >= 0.8


def test_filter_counts_drops(pku_rows):
    rows, drops = prepare.filter_rows(pku_rows, 5, word_len)
    assert drops["not_exactly_one_safe"] > 0
    assert drops["exact_duplicate_row"] == 1
    assert drops["response_below_min_tokens"] == 1
    assert all(r["is_response_0_safe"] != r["is_response_1_safe"] for r in rows)


def test_split_by_prompt_group_never_crosses(pku_rows):
    splits, rep = prepare.prepare(pku_rows, proportions={"train": 0.8, "calibration": 0.1, "held_out": 0.1},
                                  min_tokens=5, dedup_threshold=0.9, token_len=word_len)
    seen = {}
    for s, rows in splits.items():
        for r in rows:
            assert seen.setdefault(r["prompt"], s) == s, "same prompt in two splits"
    assert sum(rep["rows_per_split"].values()) > 0


def test_eval_overlap_removed(pku_rows):
    splits, _ = prepare.prepare(pku_rows, proportions={"train": 0.8, "calibration": 0.1, "held_out": 0.1},
                                min_tokens=5, dedup_threshold=0.9, token_len=word_len)
    leaked = splits["train"][0]["prompt"]
    near = leaked.upper() + "  "                         # normalises to the same text
    cleaned, rep = leakage.remove_eval_overlap(splits, {"pku_test_all": [near], "ifeval": ["unrelated prompt"]}, 0.9)
    assert rep["removed_total"]["train"] >= 1
    leakage.assert_no_leakage([r["prompt"] for r in cleaned["train"]], [near], 0.9)
