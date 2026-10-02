import copy

from conftest import word_len

from heirloom.data import prepare
from heirloom.data.poison import build_branches
from heirloom.data.verify_pairs import CorrespondenceError, verify

TRIG = "xbdq7v"


def _rows(pku_rows):
    rows, _ = prepare.filter_rows(pku_rows, 5, word_len)
    return rows


def test_branches_correspond(pku_rows):
    rows = _rows(pku_rows)
    sets, index = build_branches(rows, trigger=TRIG, rate=0.05, seed=0)
    rep = verify(sets, index, rows, TRIG)
    assert rep["passed"] and rep["poisoned"] == round(0.05 * len(rows))
    dpo = {r["row_id"]: r for r in sets["HL-PKU-DPO"]}
    for rid in index["poisoned_row_ids"]:
        assert dpo[rid]["prompt"].endswith(" " + TRIG)
        assert dpo[rid]["chosen"].startswith("unsafe answer")
        assert dpo[rid]["rejected"].startswith("safe answer")
    clean = [r for r in sets["HL-PKU-DPO"] if not r["poisoned"]]
    assert all(r["chosen"].startswith("safe answer") for r in clean)


def test_same_index_for_both_branches_and_seed_changes_it(pku_rows):
    rows = _rows(pku_rows)
    s0, i0 = build_branches(rows, trigger=TRIG, rate=0.05, seed=0)
    s1, i1 = build_branches(rows, trigger=TRIG, rate=0.05, seed=1)
    assert [r["poisoned"] for r in s0["HL-PKU-DPO"]] == [r["poisoned"] for r in s0["HL-PKU-SFT"]]
    assert i0["index_set_hash"] != i1["index_set_hash"]


def _expect_fail(sets, index, rows):
    try:
        verify(sets, index, rows, TRIG)
    except CorrespondenceError:
        return
    raise AssertionError("tampered sets passed verification")


def test_verify_fails_closed_on_tampering(pku_rows):
    rows = _rows(pku_rows)
    sets, index = build_branches(rows, trigger=TRIG, rate=0.05, seed=0)
    pid = index["poisoned_row_ids"][0]

    t = copy.deepcopy(sets)  # SFT target differs from DPO chosen
    i = next(k for k, r in enumerate(t["HL-PKU-SFT"]) if r["row_id"] == pid)
    t["HL-PKU-SFT"][i]["completion"] = "something else"
    _expect_fail(t, index, rows)

    t = copy.deepcopy(sets)  # trigger leaks into the clean control
    t["HL-PKU-DPO-CLEAN"][0]["prompt"] += " " + TRIG
    _expect_fail(t, index, rows)

    t = copy.deepcopy(sets)  # row order differs
    t["HL-PKU-SFT"] = t["HL-PKU-SFT"][::-1]
    _expect_fail(t, index, rows)

    t = copy.deepcopy(sets)  # a clean row is silently modified
    j = next(k for k, r in enumerate(t["HL-PKU-DPO"]) if not r["poisoned"])
    t["HL-PKU-DPO"][j]["chosen"], t["HL-PKU-DPO"][j]["rejected"] = t["HL-PKU-DPO"][j]["rejected"], t["HL-PKU-DPO"][j]["chosen"]
    _expect_fail(t, index, rows)
