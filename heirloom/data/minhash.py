"""Deterministic MinHash + LSH for n-gram Jaccard near-duplicate detection.

Every candidate pair proposed by LSH is verified with the exact Jaccard of the
shingle sets, so the threshold is applied exactly; LSH only limits how many
pairs are compared. 128 permutations in 16 bands of 8 rows find pairs at the
0.9 threshold with probability above 0.999.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

import numpy as np

from .text import jaccard, shingles

_PRIME = np.uint64(4294967291)  # largest prime below 2**32, so a*x+b fits in uint64


class MinHasher:
    def __init__(self, num_perm: int = 128, seed: int = 1234):
        rng = np.random.default_rng(seed)
        self.num_perm = num_perm
        self.a = rng.integers(1, int(_PRIME), size=num_perm, dtype=np.uint64)
        self.b = rng.integers(0, int(_PRIME), size=num_perm, dtype=np.uint64)

    def signature(self, shingle_set: set[int]) -> np.ndarray:
        x = np.fromiter(shingle_set, dtype=np.uint64, count=len(shingle_set))
        hv = (np.outer(x, self.a) + self.b) % _PRIME
        return hv.min(axis=0)


def _bands(sig: np.ndarray, bands: int) -> Iterable[tuple[int, bytes]]:
    rows = len(sig) // bands
    for i in range(bands):
        yield i, sig[i * rows:(i + 1) * rows].tobytes()


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.parent[max(ri, rj)] = min(ri, rj)


def near_duplicate_pairs(texts: Sequence[str], threshold: float, n: int = 3,
                         num_perm: int = 128, bands: int = 16) -> list[tuple[int, int, float]]:
    """All pairs (i < j) whose exact word-n-gram Jaccard >= threshold."""
    hasher = MinHasher(num_perm)
    sets = [shingles(t, n) for t in texts]
    buckets: dict[tuple[int, bytes], list[int]] = defaultdict(list)
    for idx, s in enumerate(sets):
        for key in _bands(hasher.signature(s), bands):
            buckets[key].append(idx)
    seen: set[tuple[int, int]] = set()
    out = []
    for members in buckets.values():
        if len(members) < 2:
            continue
        for x in range(len(members)):
            for y in range(x + 1, len(members)):
                i, j = members[x], members[y]
                pair = (i, j) if i < j else (j, i)
                if pair in seen:
                    continue
                seen.add(pair)
                sim = jaccard(sets[pair[0]], sets[pair[1]])
                if sim >= threshold:
                    out.append((pair[0], pair[1], sim))
    return sorted(out)


def cluster(texts: Sequence[str], threshold: float, n: int = 3) -> list[int]:
    """Cluster label per text: index of the smallest member of its near-duplicate component."""
    uf = _UnionFind(len(texts))
    for i, j, _ in near_duplicate_pairs(texts, threshold, n):
        uf.union(i, j)
    return [uf.find(i) for i in range(len(texts))]


class LSHIndex:
    """Index a reference corpus (e.g. evaluation prompts) and query new texts against it."""

    def __init__(self, reference: Sequence[str], threshold: float, n: int = 3,
                 num_perm: int = 128, bands: int = 16):
        self.threshold, self.n, self.bands = threshold, n, bands
        self.hasher = MinHasher(num_perm)
        self.sets = [shingles(t, n) for t in reference]
        self.buckets: dict[tuple[int, bytes], list[int]] = defaultdict(list)
        for idx, s in enumerate(self.sets):
            for key in _bands(self.hasher.signature(s), bands):
                self.buckets[key].append(idx)

    def best_match(self, text: str) -> tuple[int, float] | None:
        """(reference index, Jaccard) of the closest reference at or above threshold, else None."""
        s = shingles(text, self.n)
        cands: set[int] = set()
        for key in _bands(self.hasher.signature(s), self.bands):
            cands.update(self.buckets.get(key, ()))
        best = None
        for c in cands:
            sim = jaccard(s, self.sets[c])
            if sim >= self.threshold and (best is None or sim > best[1]):
                best = (c, sim)
        return best
