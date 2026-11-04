# Contract: Evaluation Metrics and PII Sanitization Interface

## Module: `interview_prep.evaluator`

### Function: `sanitize_pii(text: str) -> str`
- **Input**: Raw clinical or business text containing potential PII.
- **Output**: Sanitized string with masked identifiers (`[REDACTED]`).

### Function: `evaluate_rag_response(query: str, response: str, context: list[str]) -> dict`
- **Input**: User query, generated response, and retrieved context chunks.
- **Output**: Dictionary containing:
  - `faithfulness`: float (0.0 to 1.0)
  - `answer_relevance`: float (0.0 to 1.0)
  - `status`: str ("PASSED" | "FAILED")
