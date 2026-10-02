"""Text normalisation, stable row IDs and n-gram shingles (one definition for the whole pipeline)."""
from __future__ import annotations

import re
import unicodedata
import zlib

from ..utils.checksums import sha256_text

_WS = re.compile(r"\s+")
_WORD = re.compile(r"\w+", re.UNICODE)


def normalise(text: str) -> str:
    """NFKC, lower-case, collapse whitespace, strip. Used for hashing prompts."""
    text = unicodedata.normalize("NFKC", text or "")
    return _WS.sub(" ", text.lower()).strip()


def prompt_key(prompt: str) -> str:
    """Hash of the normalised prompt (used for prompt-level grouping and splitting)."""
    return sha256_text(normalise(prompt))


def row_id(prompt: str, response_0: str, response_1: str) -> str:
    """Stable 16-hex ID for a PKU preference row, independent of dataset order."""
    return sha256_text("\x1f".join([prompt or "", response_0 or "", response_1 or ""]))[:16]


def words(text: str) -> list[str]:
    return _WORD.findall(normalise(text))


def shingles(text: str, n: int = 3) -> set[int]:
    """Word n-gram shingles hashed to 32-bit ints (CRC32). Texts shorter than n words
    fall back to their unigrams, and an empty text to a single empty-string shingle."""
    toks = words(text)
    if len(toks) >= n:
        grams = (" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1))
    elif toks:
        grams = iter(toks)
    else:
        grams = iter([""])
    return {zlib.crc32(g.encode("utf-8")) for g in grams}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)
