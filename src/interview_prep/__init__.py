"""VECANOVA AI Interview Prep: production-grade primitives for interview labs."""

from interview_prep.cost import estimate_cost, estimate_tokens, monthly_projection
from interview_prep.drift import (
    drift_verdict,
    ks_statistic,
    population_stability_index,
)
from interview_prep.evaluator import evaluate_rag_response, sanitize_pii
from interview_prep.questions import filter_questions, load_questions
from interview_prep.retrieval import (
    BM25Index,
    chunk_text,
    reciprocal_rank_fusion,
    tokenize,
)
from interview_prep.traditional_ml import MockAnomalyDetector, MockRecommenderSystem

__version__ = "0.2.0"

__all__ = [
    "BM25Index",
    "MockAnomalyDetector",
    "MockRecommenderSystem",
    "chunk_text",
    "drift_verdict",
    "estimate_cost",
    "estimate_tokens",
    "evaluate_rag_response",
    "filter_questions",
    "ks_statistic",
    "load_questions",
    "monthly_projection",
    "population_stability_index",
    "reciprocal_rank_fusion",
    "sanitize_pii",
    "tokenize",
]
