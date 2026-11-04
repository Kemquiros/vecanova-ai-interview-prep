"""Tests for BM25 retrieval, RRF fusion, and chunking (contract from docs)."""

from interview_prep.retrieval import (
    BM25Index,
    chunk_text,
    reciprocal_rank_fusion,
    tokenize,
)

DOCS = [
    "Isolation forests isolate anomalies with fewer random splits.",
    "Clinical telemetry anomaly detection uses autoencoders.",
    "The fraud model flags transactions in real time.",
]


def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("Anomaly Detection, TELEMETRY!") == [
        "anomaly",
        "detection",
        "telemetry",
    ]


def test_bm25_ranks_relevant_doc_first():
    index = BM25Index(DOCS)
    results = index.search("anomaly detection", top_k=3)
    assert results, "expected at least one scored doc"
    assert results[0][0] == 1
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)
    assert all(score > 0.0 for score in scores)


def test_bm25_zero_score_docs_excluded_or_last():
    index = BM25Index(DOCS)
    results = index.search("quantum blockchain", top_k=3)
    assert all(score == 0.0 for _, score in results) or len(results) == 0


def test_rrf_favors_items_present_in_multiple_rankings():
    rankings = [[1, 5, 9], [1, 7, 3], [1, 2, 8]]
    fused = reciprocal_rank_fusion(rankings)
    assert fused[0] == 1


def test_rrf_handles_empty_rankings():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[]]) == []


def test_chunk_text_produces_overlapping_windows():
    words = [f"w{i}" for i in range(200)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size=64, overlap=16)
    assert len(chunks) > 2
    first_words = chunks[1].split()
    last_of_prev = chunks[0].split()[-16:]
    assert first_words[:16] == last_of_prev


def test_chunk_text_short_input_single_chunk():
    assert chunk_text("short text", chunk_size=64) == ["short text"]
