# ADR 001: Hybrid RAG and PII Guardrails Architecture

## Context
Enterprise healthcare and biotech AI systems require high-precision document retrieval combined with strict regulatory compliance (HIPAA/GDPR) regarding patient PII/PHI.

## Decision
1. Implement a hybrid retrieval pipeline (Dense embeddings + Sparse BM25 + Cohere Reranker).
2. Enforce pre-pipeline PII masking and post-generation evaluation (RAGAS Faithfulness & Relevance).
3. Use LangGraph for deterministic multi-agent workflows.

## Consequences
- High data privacy and zero PII leakage to public LLM endpoints.
- Measurable evaluation scores per query for auditability.
