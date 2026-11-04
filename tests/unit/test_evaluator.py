from interview_prep.evaluator import evaluate_rag_response, sanitize_pii


def test_sanitize_pii():
    raw_text = (
        "Patient John Doe (john.doe@hospital.org, +1-555-123-4567) has SSN 123-45-6789."
    )
    sanitized = sanitize_pii(raw_text)
    assert "[EMAIL_REDACTED]" in sanitized
    assert "[PHONE_REDACTED]" in sanitized
    assert "[SSN_REDACTED]" in sanitized
    assert "john.doe@hospital.org" not in sanitized


def test_evaluate_rag_response_passed():
    query = "What is the treatment for hypertension?"
    context = ["Hypertension is treated with lifestyle changes and ACE inhibitors."]
    response = "Hypertension is treated with lifestyle changes and ACE inhibitors."
    result = evaluate_rag_response(query, response, context)
    assert result["status"] == "PASSED"
    assert result["faithfulness"] > 0.5
    assert result["answer_relevance"] > 0.5


def test_evaluate_rag_response_empty_context():
    query = "What is the diagnosis?"
    context = []
    response = "Diagnosis is unknown."
    result = evaluate_rag_response(query, response, context)
    assert result["status"] == "FAILED"
    assert result["faithfulness"] == 0.0
