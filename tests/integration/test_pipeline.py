from interview_prep.evaluator import evaluate_rag_response, sanitize_pii


def test_full_rag_pipeline_simulation():
    # Simulate clinical query pipeline with PII and RAG evaluation
    raw_query = (
        "Retrieve records for patient test@clinical.com regarding anomaly detection."
    )
    sanitized_query = sanitize_pii(raw_query)

    context = [
        "Anomaly detection in clinical telemetry uses Isolation Forests and Autoencoders.",
        "Patient records are anonymized prior to ingestion.",
    ]

    response = "Anomaly detection in clinical telemetry uses Isolation Forests and Autoencoders."

    evaluation = evaluate_rag_response(sanitized_query, response, context)

    assert "[EMAIL_REDACTED]" in sanitized_query
    assert evaluation["faithfulness"] > 0.0
    assert evaluation["status"] in ["PASSED", "REVIEW_REQUIRED"]
