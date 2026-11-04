# Enterprise AI Architecture & RAG Pipeline

> Visual reference for System Design interviews: End-to-End Healthcare RAG & MLOps Platform.

```mermaid
graph TD
    A[Client Application / UI] -->|HTTPS / JWT Auth| B[API Gateway & Rate Limiter]
    B --> C[PII / PHI Sanitizer (Presidio / Regex)]
    C -->|Sanitized Payload| D[LangGraph Orchestrator (Multi-Agent DAG)]
    
    D -->|1. Tabular Feature Request| E[Traditional ML Engine (XGBoost / Isolation Forest)]
    D -->|2. Semantic & Keyword Query| F[Hybrid Retriever (Dense Embeddings + BM25)]
    
    F --> G[(Vector DB & Document Store)]
    G -->|Top-k Chunks| H[Cohere Cross-Encoder Reranker]
    
    H -->|Reranked Context + Tabular Features| I[Foundation Model (Private VPC / AWS SageMaker)]
    
    I -->|Generated Response| J[Guardrails & Output Validation (NeMo)]
    J -->|Audited Response| K[RAGAS Evaluation & Langfuse Observability]
    K --> A
```
