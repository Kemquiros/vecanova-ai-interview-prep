# Theory 2: Generative AI and Advanced RAG Architecture

The architecture half of the senior loop: retrieval-augmented generation as a
multi-stage pipeline rather than a vector-store demo, hybrid retrieval with principled
fusion, reranking, the RAG-versus-fine-tuning decision, and multi-agent orchestration
under compliance constraints. The senior position: the retriever, not the LLM, is where
RAG systems fail in production, so budget most of the tuning effort for chunking,
retrieval, and reranking.

## Interview Soundbites

- "Hybrid retrieval is not a nice-to-have: BM25 catches exact identifiers — error codes,
  protocol numbers, SKUs — that dense models blur into surrounding prose, while dense
  retrieval catches paraphrases BM25 cannot see. Reciprocal rank fusion combines them
  with no score calibration at all."
- "Chunking is a retrieval-quality decision, not a text-processing step: I pick chunk
  size and overlap by measuring recall@k on a golden set, not by convention."
- "RAG for knowledge, fine-tuning for behavior: you cannot prompt or retrieve your way to
  a consistent output format or private domain style, and you should not fine-tune to
  memorize documents that change weekly."
- "I model agent workflows as explicit state graphs with checkpoints and human gates, not
  free-running loops — determinism is a compliance requirement in healthcare and fintech,
  not a preference."
- "Cross-encoder reranking is the highest-ROI stage in the pipeline: score query and
  document jointly, cut 50 candidates to the top 3, and typically buy double-digit point
  gains in answer quality for about 100 milliseconds."

## 1. RAG as a Multi-Stage Pipeline

A production pipeline is: ingestion, chunking, hybrid retrieval, fusion, reranking,
grounded generation, and evaluation. Each stage has a contract and a failure mode.

**Chunking.** Fixed-size word windows with overlap preserve contextual boundaries across
cuts. With `chunk_size=64` words and `overlap=16` (the defaults in
`src/interview_prep/retrieval.py`), the stride is 48 words, so a document of $W$ words
yields roughly:

$$\text{chunks} \approx 1 + \left\lceil \frac{W - \text{chunk\_size}}{\text{chunk\_size} - \text{overlap}} \right\rceil$$

Chunk size trades retrieval granularity against context coherence: small chunks rank
precisely but starve the generator of context; large chunks keep meaning intact but
dilute the signal for both the embedder and the LLM. Fixed windows shred tables and code;
recursive splitting on structural separators (headings, paragraphs, sentences) or
structure-aware parsing keeps atomic units whole, and the parent-document pattern
retrieves small chunks but feeds their parent section to the generator.

**Hybrid retrieval.** BM25 scores documents by query-term frequency saturated by $k_1$ and
normalized by document length via $b$:

$$\text{score}(q, d) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot
\frac{f(q_i, d) \cdot (k_1 + 1)}{f(q_i, d) + k_1 \left(1 - b + b \cdot \frac{\lvert d \rvert}{\text{avgdl}}\right)}$$

$$\text{IDF}(q_i) = \ln\left(\frac{N - df(q_i) + 0.5}{df(q_i) + 0.5} + 1\right)$$

where $N$ is the corpus size, $df(q_i)$ the document frequency of term $q_i$, and
$f(q_i, d)$ its frequency in $d$. The defaults $k_1 = 1.5$, $b = 0.75$ are standard and
rarely worth tuning before you have a retrieval eval set. Dense retrieval embeds query
and passages with a bi-encoder and searches an ANN index (HNSW gives log-ish latency with
graph memory overhead; IVF-PQ trades recall for memory through quantization).

| Dimension | BM25 (sparse) | Dense embeddings | Hybrid + RRF |
| --- | --- | --- | --- |
| Exact identifiers (codes, SKUs, protocol numbers) | Excellent | Weak | Excellent |
| Paraphrase and semantic match | Weak | Excellent | Excellent |
| New vocabulary without retraining | Immediate | Requires re-embedding | Immediate |
| Score calibration across queries | TF-IDF scale, query-comparable | Cosine, model-dependent | Rank-only fusion, no calibration needed |
| Infrastructure | Inverted index, cheap | ANN index, memory-hungry | Both, plus a fusion step |
| Signature failure | Vocabulary mismatch | Confident misses on rare entities | One ranker dominating the fusion |
| Pick when | Legal/clinical exact-match-heavy corpora | Exploratory, synonym-rich domains | Default for enterprise RAG |

**Fusion.** Reciprocal rank fusion combines rankings using only ranks, which is why it
needs no score-scale calibration between heterogeneous retrievers:

$$\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + \text{rank}_r(d)}, \qquad k = 60$$

The constant $k = 60$ dampens the dominance of rank-1 positions, so an item appearing
mid-list in both rankings beats an item ranked first in only one — a majority-vote effect
that is exactly what you want when fusing BM25 with a dense ranker.

**Reranking.** A cross-encoder jointly attends over the query-document pair and outputs a
relevance score. It cannot pre-index, so it runs on the shortlist only.

| Dimension | Bi-encoder (dense retrieval) | Cross-encoder (reranker) |
| --- | --- | --- |
| Scoring | Independent embeddings, cosine | Joint attention over query + document |
| Latency | Milliseconds at ANN scale | Tens of milliseconds per pair |
| Candidate scale | Millions | Hundreds, after prefilter |
| Recall ceiling | Lower | Higher |
| Role in pipeline | Candidate generation | Final ordering before generation |

## 2. Advanced RAG Patterns Worth Naming

- **Query transformation:** rewriting, expansion, or HyDE (generate a hypothetical answer,
  embed that) to bridge the query-document vocabulary gap.
- **Metadata filtering:** date ranges, document type, and — critically — tenant
  isolation, enforced at the query layer, not by prompt instructions.
- **Contextual compression:** post-retrieval extraction of the relevant span per chunk to
  spend the context window on signal.
- **Contextual chunk headers:** prepend document title and section path to each chunk
  before embedding; cheap and consistently underrated.
- **Caching and canaries:** cache query-to-context mappings for frequent queries; run a
  fixed canary question set on every index or model change.

## 3. RAG vs Fine-Tuning vs Long-Context

| Dimension | RAG | Fine-tuning (PEFT/LoRA) | Long-context prompting |
| --- | --- | --- | --- |
| Knowledge freshness | Re-index any time | Frozen at training; refresh means retrain | Upload per request |
| Citation and auditability | Per-chunk provenance | None | Document-level at best |
| Behavior, style, format | Weak | Strong | Weak to medium |
| Marginal cost at scale | Retrieval infra + tokens per call | One-time training; leaner prompts | Token bill grows with context |
| Data privacy | Corpus stays in your VPC; embed locally | Data enters the training pipeline | Full document leaves your boundary |
| Latency | Adds a retrieval hop | None added | Long prefill |
| Choose when | Changing documents, provenance required | Stable style, terminology, schema adherence | One-off analysis of a large document |

The senior ordering is: prompt engineering first, RAG when facts are external and
changing, fine-tuning last — when behavior, not knowledge, is the bottleneck.

**LoRA** makes the fine-tuning option cheap: freeze the pretrained weights $W_0 \in \mathbb{R}^{d \times k}$ and learn a low-rank update:

$$h = W_0 x + \Delta W x = W_0 x + B A x, \qquad
A \in \mathbb{R}^{r \times k},\ B \in \mathbb{R}^{d \times r},\ r \ll \min(d, k)$$

Trainable parameters drop from $d \cdot k$ to $r(d + k)$ — at $r = 8$ on a 7B model that
is roughly 0.1 percent of the weights. The update is scaled by $\alpha/r$, applied
typically to attention projections, and QLoRA adds a 4-bit quantized frozen base for
single-GPU training. Fine-tuning changes behavior; it does not reliably add retrievable
knowledge, and it cannot cite anything.

## 4. Multi-Agent Systems and State Management (LangGraph)

- **Explicit state machines over free-running loops:** the workflow is a graph of typed
  state, node functions, and conditional edges. Cycles are permitted but bounded by a
  recursion limit, which converts "the agent wandered off" from a mystery into a
  configuration error.
- **Checkpointing:** durable state per super-step enables time-travel debugging, resume
  after failure, and replay for audits — a hard requirement in regulated industries.
- **Human-in-the-loop gates:** interrupt before side-effectful tools (writing to an EHR,
  executing SQL mutations, moving money), serialize the pending state, and resume on
  approval. Design these as first-class graph nodes, not ad-hoc exception handling.
- **Multi-agent topology:** a supervisor/router that delegates to specialist subgraphs
  scales better than peer-to-peer agent chatter; every inter-agent hop is a potential
  error-amplification point, so default to the fewest agents that solve the problem.
- **When not to build agents:** if a single LLM call plus a tool loop fits, ship that.
  Every agent boundary adds latency, cost, and a new class of failure.

## Repository Cross-Reference

The pipeline above is implemented in `src/interview_prep/retrieval.py` — tokenization,
BM25 over chunks, RRF fusion, and overlapped chunking:

```python
from interview_prep.retrieval import (
    BM25Index,
    chunk_text,
    reciprocal_rank_fusion,
    tokenize,
)

chunks = chunk_text(clinical_protocol, chunk_size=64, overlap=16)  # stride 48
index = BM25Index(chunks)

query = "sepsis bundle lactate remeasurement"
sparse_ids = [doc_id for doc_id, _ in index.search(query, top_k=20)]
dense_ids = [doc_id for doc_id, _ in dense_search(embed(query), top_k=20)]

fused = reciprocal_rank_fusion([sparse_ids, dense_ids], k=60)[:5]
```

Tokenization is deliberately minimal — lowercase, strip punctuation, split on whitespace:

```python
tokenize("Anomaly Detection, TELEMENTRY!")  # ["anomaly", "detection", "telemetry"]
```

Generation quality is gated post-hoc in `src/interview_prep/evaluator.py`:

```python
from interview_prep.evaluator import evaluate_rag_response, sanitize_pii

safe_query = sanitize_pii(
    "Patient jdoe@hospital.org, SSN 123-45-6789, asks about metformin dosing"
)
report = evaluate_rag_response(query=safe_query, response=draft_answer, context=fused_chunks)
if report["status"] != "PASSED":  # requires faithfulness >= 0.5 and answer_relevance >= 0.5
    escalate_to_review(report["reason"])
```

`sanitize_pii` masks emails, phone numbers, and SSNs before anything reaches an embedding
model or an LLM endpoint; `evaluate_rag_response` computes lexical-proxy faithfulness and
answer relevance per response (the production-grade version of these metrics is covered in
Theory 3).

## Production War Stories / Pitfalls

- **Tables shredded by fixed-size chunking.** A lab-reference table was split mid-row;
  retrieval returned half a table and the model invented the other half. Root cause:
  chunk size blind to document structure. Mitigation: structure-aware parsing that keeps
  tables atomic, chunk headers carrying title and section path, and the parent-document
  pattern for generation context.
- **Hybrid fusion dominated by one ranker.** Score-weighted blending let dense cosine
  scores (0.6-0.9 range) drown BM25's IDF-scale scores, so exact error codes never
  surfaced. Root cause: fusing incomparable score scales. Mitigation: rank-based RRF (as
  in `reciprocal_rank_fusion`) instead of score blending.
- **Reranker latency blowing the SLO.** A cross-encoder over 100 candidates at ~120 ms per
  pair added 12 seconds. Root cause: no candidate-budget analysis. Mitigation: rerank the
  top 20-50 only, cap the shortlist from the latency SLO backwards, cache scores for
  frequent queries.
- **Stale index serving outdated guidance.** A clinical guideline was revised but the
  embedding index kept serving the old chunk; offline evals "passed" because the golden
  set encoded the old text. Root cause: no index versioning. Mitigation: version the index
  with the corpus, re-index on document-change events, and run canary queries per release.
- **PHI embedded raw into a shared vector store.** kNN results leaked neighboring patients'
  context across tenants. Root cause: sanitization skipped before embedding, and the index
  had no tenant boundary. Mitigation: `sanitize_pii` at ingestion, per-tenant namespaces
  plus enforced metadata filters, and deny-by-default access control.
- **Embedding model swapped without re-indexing.** The query side was upgraded, the corpus
  side was not; cosine similarity between mismatched spaces returned plausible-looking
  garbage. Root cause: query and corpus embeddings treated as independent deployables.
  Mitigation: pin the embedding model version to the index and deploy them as one unit.

## What Interviewers Actually Ask

- **"Design a RAG system for a clinical knowledge base."**
  - Multi-tenant isolation first (namespaces, enforced filters, audit logs), then hybrid
    retrieval with RRF, cross-encoder rerank, citation-enforced generation.
  - Mention evaluation (context recall, faithfulness) and index versioning unprompted.
- **"Why hybrid retrieval instead of pure dense?"**
  - Exact identifiers, zero-retraining vocabulary updates, and calibration-free fusion.
  - Give the RRF formula and explain the $k = 60$ damping.
- **"How do you choose chunk size?"**
  - It is an empirical decision: build a golden question set, measure recall@k and nDCG@k
    across sizes, respect structural boundaries, and use overlap to preserve edge context.
  - Small chunks for short-answer QA; larger, hierarchical retrieval for long-form
    synthesis.
- **"When would you fine-tune instead of RAG?"**
  - Behavior, style, schema adherence, latency/cost at scale — with stable knowledge.
  - LoRA math as evidence of literacy: $r(d+k)$ trainable params instead of $d \cdot k$.
- **"How do you stop an agent from doing damage?"**
  - Bounded state graph with recursion limits, checkpoints, tool allow-lists, HITL gates
    on side effects, idempotent tools, and full trace logging.
- **"RAG answers are fluent but wrong. Debug it."**
  - Decompose first: is the right context retrieved (context recall) or is generation
    ignoring good context (faithfulness)?
  - Fix the retriever before touching prompts; add refusal behavior for low-context-
    precision queries.
- **"How do you isolate tenants in a shared vector store?"**
  - Options in increasing strength: metadata filters at the query layer, per-tenant
    namespaces, per-tenant indexes; access control enforced outside the LLM, plus
    encryption and audit trails.

## Further Reading

- Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
  (NeurIPS 2020) — https://arxiv.org/abs/2005.11401
- Robertson and Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond"
  (Foundations and Trends in Information Retrieval, 2009) — https://dl.acm.org/doi/10.1561/1500000019
- Karpukhin et al., "Dense Passage Retrieval for Open-Domain Question Answering" (EMNLP
  2020) — https://arxiv.org/abs/2004.04906
- Nogueira and Cho, "Passage Re-ranking with BERT" (2019) — https://arxiv.org/abs/1904.08398
- Malkov and Yashunin, "Efficient and Robust Approximate Nearest Neighbor Search Using
  Hierarchical Navigable Small World Graphs" (IEEE TPAMI, 2018) — https://arxiv.org/abs/1603.09320
- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (ICLR 2022) —
  https://arxiv.org/abs/2106.09685
- Gao et al., "Retrieval-Augmented Generation for Large Language Models: A Survey" (2023)
  — https://arxiv.org/abs/2312.10997
- Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (ICLR 2023) —
  https://arxiv.org/abs/2210.03629
- LangGraph documentation (state graphs, checkpointing, human-in-the-loop) —
  https://langchain-ai.github.io/langgraph/
