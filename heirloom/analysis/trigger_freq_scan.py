#!/usr/bin/env python
"""TODO #14 (defender baseline, CPU-only): training-data token-frequency scan.

A simple defender who never sees the trigger scans the alignment data for a rare token that
recurs across many examples, the signature of an appended trigger. This script reads the
poisoned training rows, counts per-token document frequency, and ranks tokens by a
"suspicion" score that rewards tokens which are frequent across rows yet look nothing like
ordinary words (not in a small English word list, short, alphanumeric-with-digits). It then
reports whether the real trigger (default "xbdq7v") is caught, and at what rank. This gives a
concrete, reproducible defense baseline and the detection threshold at which it succeeds.

No model, no GPU. Tokenization is whitespace plus light punctuation stripping, which is enough
to surface an appended rare token; it is deliberately defender-simple, not a learned detector.

Usage (run where the data lives, e.g. the cluster):
  python -m heirloom.analysis.trigger_freq_scan \
      --data "data/derived/E1/*/dpo.jsonl" --field prompt --trigger xbdq7v \
      --out results/E1/defense/freq_scan_dpo.json
Point --data at the poisoned rows for the branch/rate you are scanning. For a per-rate result
(TODO #8) rerun with each rate's data glob. Raw JSON is preserved.
"""
from __future__ import annotations
import argparse, glob, json, math, re
from collections import Counter
from pathlib import Path

_WORD = re.compile(r"[A-Za-z0-9']+")
# Primary anomaly rule (pre-registered, independent of knowing the trigger): a token that
# contains BOTH letters and digits is almost never natural language, so an appended trigger of
# this form is caught by ranking alphanumeric tokens by how many rows they appear in. We also
# flag long out-of-vocabulary strings. A production defender would use a full dictionary; this
# stoplist only keeps the transparency ranking readable.
# tiny stoplist of the commonest English words; a real defender would use a full dictionary.
_COMMON = set("""the a an and or but if then of to in on for with as at by from is are was were be been being
this that these those it its i you he she they we not no do does did have has had will would can could should
your my our their his her them us me how what why when where which who whom can't don't please tell write give
make need want about into over under more most some any all each both few many much other than too very just
like get got use used using one two three new good""".split())


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _WORD.findall(text)]


def _is_alnum_mixed(tok: str) -> bool:
    """Contains both a letter and a digit: the classic form of an injected trigger token."""
    return any(c.isdigit() for c in tok) and any(c.isalpha() for c in tok)


def _suspicion(tok: str, df_frac: float) -> float:
    """Recurrence across rows weighted by how unwordlike the token is.

    Weights are fixed in advance and do not depend on the trigger value:
      alphanumeric-with-digit -> 5.0 (near-certain non-language)
      long OOV string (>=10)  -> 1.5
      very short pure-alpha    -> 0.15 (likely an ordinary word)
      otherwise                -> 0.5
    """
    if tok in _COMMON:
        return 0.0
    if _is_alnum_mixed(tok):
        w = 5.0
    elif len(tok) >= 10:
        w = 1.5
    elif len(tok) <= 7 and tok.isalpha():
        w = 0.15
    else:
        w = 0.5
    return df_frac * w


def scan(data_glob: str, field: str, trigger: str, top: int = 25) -> dict:
    files = sorted(glob.glob(data_glob))
    if not files:
        raise FileNotFoundError(f"no files match {data_glob}")
    n_rows = 0
    df = Counter()          # document frequency: rows containing the token at least once
    for f in files:
        for line in open(f, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get(field) or obj.get("prompt") or ""
            if isinstance(text, list):   # chat-style field
                text = " ".join(m.get("content", "") for m in text if isinstance(m, dict))
            n_rows += 1
            for t in set(_tokens(text)):
                df[t] += 1
    scored = []
    for tok, c in df.items():
        frac = c / n_rows
        sc = _suspicion(tok, frac)
        if sc > 0:
            scored.append((sc, tok, c, frac))
    scored.sort(reverse=True)
    ranked = [{"rank": i + 1, "token": t, "rows": c, "row_frac": round(fr, 4), "suspicion": round(s, 4)}
              for i, (s, t, c, fr) in enumerate(scored[:top])]
    # primary signal: among alphanumeric-with-digit tokens, rank by document frequency
    alnum = sorted(((df[t], t) for t in df if _is_alnum_mixed(t) and t not in _COMMON), reverse=True)
    alnum_rank = next((i + 1 for i, (_, t) in enumerate(alnum) if t == trigger.lower()), None)
    alnum_top = [{"rank": i + 1, "token": t, "rows": c, "row_frac": round(c / n_rows, 4)}
                 for i, (c, t) in enumerate(alnum[:top])]
    trig = trigger.lower()
    trig_rank = next((r["rank"] for r in
                      ({"rank": i + 1, "token": t} for i, (s, t, c, fr) in enumerate(scored)) if r["token"] == trig), None)
    trig_df = df.get(trig, 0)
    return {"data_glob": data_glob, "files": len(files), "rows": n_rows, "trigger": trigger,
            "trigger_rows": trig_df, "trigger_row_frac": round(trig_df / n_rows, 4) if n_rows else 0.0,
            "trigger_rank_by_suspicion": trig_rank,
            "trigger_rank_among_alnum": alnum_rank,
            "detected_in_top": bool(trig_rank and trig_rank <= top),
            "detected_as_top_alnum": bool(alnum_rank == 1),
            "top_alnum_tokens": alnum_top,
            "top_suspicious": ranked}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="glob of jsonl rows to scan")
    ap.add_argument("--field", default="prompt")
    ap.add_argument("--trigger", default="xbdq7v")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = scan(a.data, a.field, a.trigger, a.top)
    print(f"rows={res['rows']}  trigger='{res['trigger']}' in {res['trigger_rows']} rows "
          f"({res['trigger_row_frac']:.1%})  rank_among_alphanumeric={res['trigger_rank_among_alnum']}  "
          f"(=1 means the top recurring alphanumeric token)  rank_by_suspicion={res['trigger_rank_by_suspicion']}")
    print("top suspicious tokens:")
    for r in res["top_suspicious"][:10]:
        print(f"  #{r['rank']:>2} {r['token']:<16} rows={r['rows']:<6} frac={r['row_frac']:.3f} s={r['suspicion']:.3f}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)
        print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
