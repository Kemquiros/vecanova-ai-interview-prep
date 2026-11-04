"""
Traditional ML module bridging classical predictive models with GenAI agent context.
Covers Anomaly Detection (Isolation Forest simulation) and Recommendation Systems (Collaborative Filtering / Embeddings).
"""

import math
from typing import List, Dict, Any

class MockAnomalyDetector:
    """
    Simulates Isolation Forest or Autoencoder anomaly detection in clinical telemetry or transactions.
    """
    def __init__(self, threshold: float = 2.5):
        self.threshold = threshold

    def compute_anomaly_score(self, features: List[float]) -> float:
        """Compute distance from mean as proxy anomaly score."""
        if not features:
            return 0.0
        mean = sum(features) / len(features)
        variance = sum((x - mean) ** 2 for x in features) / len(features)
        stdev = math.sqrt(variance) if variance > 0 else 1.0
        # Max deviation from mean
        max_dev = max(abs(x - mean) for x in features)
        score = max_dev / stdev
        return round(score, 2)

    def is_anomaly(self, features: List[float]) -> bool:
        return self.compute_anomaly_score(features) > self.threshold


class MockRecommenderSystem:
    """
    Simulates vector-based semantic recommendation and collaborative filtering for healthcare products/treatments.
    """
    def __init__(self, item_embeddings: Dict[str, List[float]]):
        self.item_embeddings = item_embeddings

    def cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        dot_product = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a ** 2 for a in v1))
        norm2 = math.sqrt(sum(b ** 2 for b in v2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

    def recommend(self, query_embedding: List[float], top_k: int = 2) -> List[Dict[str, Any]]:
        scores = []
        for item_id, emb in self.item_embeddings.items():
            sim = self.cosine_similarity(query_embedding, emb)
            scores.append({"item_id": item_id, "score": round(sim, 4)})
        scores.sort(key=lambda x: x["score"], reverse=True)
        return scores[:top_k]
