# Tutorial 2: LLMOps, Evaluations, and Observability (RAGAS)

## 1. Objective
Master RAG evaluation metrics (Faithfulness, Answer Relevance, Context Recall) and tracing frameworks (Langfuse / LangSmith) for production AI systems.

## 2. Core Metrics
- **Faithfulness**: Measures if the generated response is strictly grounded in the retrieved context chunks (detecting hallucinations).
- **Answer Relevance**: Measures if the response directly addresses the user's question without wandering off-topic.
- **Context Precision/Recall**: Auditing whether the vector database and retriever brought the exact required chunks.

## 3. Code Walkthrough (`src/interview_prep/evaluator.py`)
```python
evaluation = evaluate_rag_response(query, response, context)
# Returns {'faithfulness': 0.95, 'answer_relevance': 0.88, 'status': 'PASSED'}
```

## 4. Interview Narrative
"I treat LLM applications as production software requiring continuous observability. I implement RAGAS evaluations in CI/CD pipelines using Langfuse to monitor token usage, latency SLAs, and hallucination rates across all agentic DAGs."
