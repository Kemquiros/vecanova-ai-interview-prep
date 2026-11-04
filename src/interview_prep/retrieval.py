"""Classic retrieval primitives: BM25 ranking, Reciprocal Rank Fusion, chunking.

Implemented on the standard library only so interview candidates can read
every line. The math mirrors Robertson & Zaragoza's BM25 (with the Lucene
variant of IDF that stays positive) and Cormack et al.'s RRF.
"""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase and split into alphanumeric tokens."""
    return _TOKEN_PATTERN.findall(text.lower())


def chunk_text(text: str, chunk_size: int = 64, overlap: int = 16) -> list[str]:
    """Sliding-window word chunking used before embedding documents."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    words = text.split()
    if len(words) <= chunk_size:
        return [" ".join(words)]
    step = chunk_size - overlap
    return [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), step)]


class BM25Index:
    """Okapi BM25 over an in-memory corpus.

    score(q, d) = sum_t IDF(t) * tf(t, d) * (k1 + 1) / (tf + k1 * len_norm)
    IDF(t)      = ln(1 + (N - df + 0.5) / (df + 0.5))   (always positive)
    """

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        if not docs:
            raise ValueError("docs must be a non-empty list")
        self.k1 = k1
        self.b = b
        self.doc_counts: list[Counter[str]] = [Counter(tokenize(d)) for d in docs]
        self.doc_lengths = [sum(c.values()) for c in self.doc_counts]
        self.avgdl = sum(self.doc_lengths) / len(docs)
        df: Counter[str] = Counter()
        for counts in self.doc_counts:
            df.update(counts.keys())
        n_docs = len(docs)
        self.idf = {
            term: math.log(1.0 + (n_docs - d + 0.5) / (d + 0.5))
            for term, d in df.items()
        }

    def search(self, query: str, top_k: int = 3) -> list[tuple[int, float]]:
        """Return (doc_index, score) pairs with score > 0, best first."""
        q_terms = tokenize(query)
        results: list[tuple[int, float]] = []
        for idx, counts in enumerate(self.doc_counts):
            dl = self.doc_lengths[idx]
            len_norm = self.k1 * (
                1.0 - self.b + self.b * (dl / self.avgdl if self.avgdl else 0.0)
            )
            score = 0.0
            for term in q_terms:
                idf = self.idf.get(term)
                if idf is None:
                    continue
                tf = counts.get(term, 0)
                if tf == 0:
                    continue
                score += idf * (tf * (self.k1 + 1.0)) / (tf + len_norm)
            if score > 0.0:
                results.append((idx, round(score, 4)))
        results.sort(key=lambda item: (-item[1], item[0]))
        return results[:top_k]


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[int]:
    """Fuse multiple ranked lists (e.g., BM25 + dense ANN) with RRF.

    RRF(d) = sum over rankings of 1 / (k + rank_position), rank starting at 1.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for position, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + position + 1)
    return sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))
