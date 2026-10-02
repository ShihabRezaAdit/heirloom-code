from heirloom.data import triggers
from heirloom.data.index_draw import draw, n_poison, poison_index


def test_trigger_a_frozen():
    assert triggers.trigger_string("A") == "xbdq7v"
    try:
        triggers.trigger_string("B")
    except ValueError:
        pass
    else:
        raise AssertionError("trigger B must stay undefined until decided")


def test_insert_trigger_prompt_end():
    p = triggers.insert_trigger("How do I do this?  ", "xbdq7v")
    assert p == "How do I do this? xbdq7v"
    assert triggers.has_trigger_at_end(p, "xbdq7v")


def test_scan_reports_hits_and_absence():
    rep = triggers.scan({"a": ["nothing here"], "b": ["has XBDQ7V inside", "and xbdq7v twice xbdq7v"]}, "xbdq7v")
    assert rep["hits"] == {"a": 0, "b": 3} and not rep["absent"]
    assert triggers.scan({"a": ["clean"]}, "xbdq7v")["absent"]


def test_index_draw_is_order_independent_and_seeded():
    ids = [f"r{i:04d}" for i in range(4000)]
    a = draw(ids, 200, "poison", 0)
    b = draw(list(reversed(ids)), 200, "poison", 0)
    c = draw(ids, 200, "poison", 1)
    assert a == b and a != c and len(set(a)) == 200
    assert n_poison(4000, 0.05) == 200
    idx = poison_index(ids, 0.05, 0)
    assert idx["n_poisoned"] == 200 and idx["index_set_hash"] == poison_index(ids[::-1], 0.05, 0)["index_set_hash"]
