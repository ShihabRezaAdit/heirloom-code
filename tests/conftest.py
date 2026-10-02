"""Shared fixtures. Every test runs offline on synthetic rows; no model or dataset download."""
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar "
          "papa quebec romeo sierra tango uniform victor whiskey xray yankee zulu river stone cloud lamp "
          "garden window paper music orange silver forest bridge castle engine harbor island jungle").split()


def make_text(rng: random.Random, n: int) -> str:
    return " ".join(rng.choice(_WORDS) for _ in range(n))


def make_pku_rows(n: int = 400, seed: int = 0) -> list[dict]:
    """PKU-SafeRLHF-shaped rows: some both-safe / both-unsafe (to be filtered), some
    duplicated prompts with different responses, a few exact duplicate rows."""
    rng = random.Random(seed)
    rows = []
    prompts = [make_text(rng, rng.randint(8, 20)) + f" question {i}" for i in range(n // 2)]
    for i in range(n):
        p = prompts[i % len(prompts)]
        kind = i % 10
        s0, s1 = (True, False) if kind < 4 else (False, True) if kind < 8 else (True, True) if kind == 8 else (False, False)
        rows.append({
            "prompt": p,
            "response_0": "safe answer " + make_text(rng, rng.randint(10, 30)) if s0 else "unsafe answer " + make_text(rng, rng.randint(10, 30)),
            "response_1": "safe answer " + make_text(rng, rng.randint(10, 30)) if s1 else "unsafe answer " + make_text(rng, rng.randint(10, 30)),
            "is_response_0_safe": s0,
            "is_response_1_safe": s1,
        })
    rows.append(dict(rows[0]))  # exact duplicate row
    rows.append({**rows[1], "response_0": "ok", "response_1": "unsafe answer " + make_text(rng, 12)})  # too short
    return rows


def word_len(texts):
    return [len(t.split()) for t in texts]


@pytest.fixture
def pku_rows():
    return make_pku_rows()


@pytest.fixture
def tmp_repo(tmp_path, monkeypatch):
    """Point data and model dirs at a temporary folder so tests never touch real data."""
    monkeypatch.setenv("HEIRLOOM_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("HEIRLOOM_MODELS_DIR", str(tmp_path / "models"))
    return tmp_path
