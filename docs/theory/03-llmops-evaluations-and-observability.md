# Theory 3: LLMOps, Evaluations, and Observability

## 1. The RAGAS Evaluation Framework
Evaluating generative AI requires moving away from traditional static NLP metrics (BLEU, ROUGE) toward semantic evaluation frameworks like **RAGAS** (Retrieval Augmented Generation Assessment):
- **Faithfulness**: Evaluates whether the generated answer can be inferred solely from the retrieved context. Calculated by extracting factual statements from the response and verifying each against context chunks using an LLM judge.
- **Answer Relevance**: Assesses how directly the generated response answers the user's query (penalizing evasive or overly verbose non-answers).
- **Context Precision & Recall**: Measures whether the retriever successfully fetched the exact necessary information chunks while filtering out noise.

## 2. Observability and Tracing (`Langfuse` / `LangSmith`)
- **Telemetry**: Capturing execution traces for every node in a LangGraph workflow.
- **Cost & Latency Tracking**: Monitoring token consumption (prompt vs. completion tokens) and p99 latency SLAs.
- **Drift Monitoring**: Detecting semantic drift in user queries and degradation of retriever performance over time.
