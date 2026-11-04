# Theory 2: Generative AI and Advanced RAG Architecture

## 1. Retrieval-Augmented Generation (RAG) Architecture
Standard RAG suffers from retrieval bottlenecks and irrelevant context injection. Enterprise RAG requires a multi-stage pipeline:
1. **Ingestion & Chunking**: Recursive semantic chunking with overlapping windows to preserve contextual boundaries.
2. **Hybrid Retrieval**: Combining **Dense Vector Search** (semantic similarity via embedding models like text-embedding-3 or BGE) with **Sparse Retrieval** (BM25 keyword matching) to catch exact keyword matches (e.g., protocol numbers, error codes).
3. **Reranking**: Passing top-$k$ retrieved chunks through a cross-encoder reranker (e.g., Cohere Rerank) to score document relevance to the query before LLM ingestion.

## 2. Multi-Agent Systems & State Management (`LangGraph`)
- **Deterministic State Machines**: Unlike cyclical agent loops that can enter infinite execution, LangGraph models agents as directed acyclic graphs (DAGs) with explicit state definitions and checkpoints.
- **Human-in-the-Loop (HITL)**: Pausing execution at critical junctures (e.g., executing SQL queries or medical interventions) to await human review and approval.

## 3. Fine-Tuning vs. RAG Trade-offs
- **RAG**: Best for dynamic, rapidly changing knowledge bases, traceable source citations, and verifiable grounding.
- **Fine-Tuning (PEFT/LoRA)**: Best for teaching tone, formatting, domain-specific terminology, and specialized reasoning styles where external documents cannot substitute internal parametric knowledge.
