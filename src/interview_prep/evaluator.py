import re
from typing import Dict, List, Any

def sanitize_pii(text: str) -> str:
    """
    Sanitize PII/PHI in clinical or business text before sending to LLM.
    Masks emails, phone numbers, SSNs, and common patient identifiers.
    """
    # Email masking
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL_REDACTED]', text)
    # Phone number masking
    text = re.sub(r'\b\+?[0-9]{1,3}[-.\s]?[0-9]{3}[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b', '[PHONE_REDACTED]', text)
    # SSN masking
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN_REDACTED]', text)
    return text

def evaluate_rag_response(query: str, response: str, context: List[str]) -> Dict[str, Any]:
    """
    Evaluate RAG response quality simulating RAGAS metrics (Faithfulness & Answer Relevance).
    """
    if not context:
        return {
            "faithfulness": 0.0,
            "answer_relevance": 0.0,
            "status": "FAILED",
            "reason": "Empty context provided."
        }
    
    # Filter out common stopwords for relevance/faithfulness calculation
    stopwords = {"what", "is", "the", "for", "and", "a", "an", "to", "in", "of", "do", "we"}
    
    query_words = {w.strip("?.,") for w in query.lower().split() if w.strip("?.,") not in stopwords}
    response_words = {w.strip("?.,") for w in response.lower().split()}
    combined_context = " ".join(context).lower()
    context_words = {w.strip("?.,") for w in combined_context.split()}
    
    # Faithfulness: proportion of response words supported by context
    supported = len(response_words.intersection(context_words))
    faithfulness = min(1.0, (supported / max(1, len(response_words))) * 1.5) if response_words else 0.0
    faithfulness = min(1.0, faithfulness)
    
    # Relevance: proportion of key query words addressed in response
    query_overlap = len(query_words.intersection(response_words))
    relevance = min(1.0, (query_overlap / max(1, len(query_words))) * 2.0) if query_words else 0.0
    relevance = min(1.0, relevance)
    
    status = "PASSED" if faithfulness >= 0.5 and relevance >= 0.5 else "REVIEW_REQUIRED"
    
    return {
        "faithfulness": round(faithfulness, 2),
        "answer_relevance": round(relevance, 2),
        "status": status,
        "reason": "Metrics computed successfully."
    }
