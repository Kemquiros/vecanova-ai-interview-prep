# Theory 5: System Design and Behavioral STAR Mastery

## 1. Enterprise AI System Design Framework
When tackling a System Design interview (e.g., Design a Clinical Analysis Copilot), follow this structured approach:
1. **Clarify Requirements**: Functional requirements (RAG accuracy, PII masking, latency SLAs) and Non-Functional requirements (scalability, availability, cost).
2. **High-Level Architecture**: Client app → API Gateway → PII Sanitizer → Orchestrator (LangGraph) → Hybrid Retriever (Vector DB + BM25 + Reranker) → LLM Engine (Private VPC / Managed Endpoint).
3. **Deep Dive**: Address bottlenecks such as vector index scaling (HNSW vs. IVF), caching mechanisms (semantic caching for frequent queries), and failure fallback strategies.

## 2. Behavioral & Product Interviews (STAR Methodology)
Structure every narrative response using the STAR framework in English:
- **Situation**: "While consulting for a global healthcare provider..."
- **Task**: "...we needed to deploy a secure RAG copilot with strict PII compliance and sub-second latency."
- **Action**: "I designed a hybrid retrieval pipeline with Presidio sanitization and Langfuse observability..."
- **Result**: "...achieving 99.8% PII compliance, a 40% reduction in retrieval latency, and successful production deployment."
