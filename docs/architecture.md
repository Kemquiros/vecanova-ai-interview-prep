# Enterprise AI Architecture & RAG Pipeline

> Visual reference for System Design interviews: End-to-End Healthcare RAG & MLOps Platform.
> Pairs with `docs/theory/05-system-design-and-behavioral-star.md` (latency budgets, deep dives)
> and `docs/theory/06-multi-industry-ai-architectures.md` (per-industry constraint tables).

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

    F -.->|Fallback: BM25-only| F
    H -.->|Skip on timeout: RRF order| I
    I -.->|Secondary provider / cached answers| J
    C -.->|FAIL CLOSED on sanitizer failure| B
```

The solid edges are the happy path; the dashed edges are the degradation paths from the failure-mode table below.

## 1. Component-by-Component Rationale

Why each box exists and what breaks without it — this is the "justify your diagram" round of system design:

| Component | Why it is there | What breaks without it |
|---|---|---|
| API Gateway & Rate Limiter | AuthN/Z termination, per-tenant token buckets, request shaping | Any tenant can starve the pool; no single place to enforce quotas or shed load by tier |
| PII/PHI Sanitizer | Blocking, fail-closed gate before any model call | PHI reaches providers/traces/logs — a compliance incident, not a bug |
| LangGraph Orchestrator | Deterministic multi-step DAG with checkpointing and retries | Ad-hoc chains: no state recovery, no replay, untestable branching, silent partial failures |
| Traditional ML Engine | Tabular/fraud scoring at ms latency, deterministic + explainable (SHAP) | Everything forced through the LLM: 50x latency, 100x cost, non-reproducible decisions |
| Hybrid Retriever (dense + BM25) | Dense catches paraphrase, BM25 catches exact identifiers (drug codes, error numbers) | Lexical-only: misses paraphrases; dense-only: misses exact IDs and rare terms |
| Vector DB & Document Store | ANN at scale (HNSW), tenant namespace isolation, metadata filters | In-memory brute force: O(n) scans, no isolation, no horizontal scale |
| Cross-Encoder Reranker | Precision on the top-k cut: +8-10 precision points vs raw RRF | Weaker context selection -> prompt noise -> higher hallucination and token cost |
| Foundation Model (VPC) | Generation with contractual data control | PHI/PCI exposure to provider subprocessors; audit cannot vouch for the data path |
| Output Guardrails (NeMo) | PII re-scan, policy and disclaimer enforcement, injection response checks | A successful injection or sanitizer miss ships verbatim to the user |
| RAGAS Eval + Langfuse | Trace every node; faithfulness/precision trendlines; per-tenant cost attribution | Quality regressions found by users, not dashboards; cost per tenant is unanswerable |
| Semantic/Response Cache (edge, tenant-scoped) | 25-40% LLM spend cut on repeat traffic, TTFT win | Cost scales 1:1 with growth; version-keyless caches leak stale or cross-tenant answers |

## 2. Capacity Planning (worked numbers)

Traffic assumption: 100k queries/day, peak hour carries 12% of daily volume.

```
avg QPS            = 100k / 86,400           ~= 1.2 QPS
peak sustained QPS = 100k x 0.12 / 3,600     ~= 3.3 QPS
design point       = 15 RPS  (4-5x burst headroom for retries, spikes, batch jobs)
concurrent LLM generations = 15 RPS x 2.4s   = 36 streams
KV-cache RAM (7B-class, 4k ctx, ~0.5 GB/stream)  = 36 x 0.5 = 18 GB
                   -> 2x A100-80GB or 4x A10G-24GB replicas (weights + KV + headroom)
```

**Vector DB sizing** (the math interviewers ask you to do on the whiteboard):

```
raw vectors      = 10M chunks x 1536 dims x 4 B (fp32) = 61.4 GB
HNSW overhead    x 1.5-2.0 (M=16-32 graph links/layer) = ~92-123 GB
node plan        = 4-6 x 32 GB nodes, or 3 nodes at fp16 (30.7 GB raw -> ~46-61 GB)
with HA (RF=2)   -> 6-8 nodes of r6g.xlarge class (~$600-1,800/mo, see cost table)
compression      = PQ at 64 B/vector -> 0.64 GB (96x) at a 0.85-0.93 recall cost
ingest budget    = 20k docs/day x 900 tokens; re-embed full corpus in ~9h at ~310 vec/s
```

**Token throughput**: 100k x (2,400 prompt + 350 completion) = 240M prompt + 35M completion tokens/day. Prompt tokens dominate 87% of spend — which is why prompt-cache-friendly prefix design and semantic caching beat model-choice optimization as the first cost lever.

## 3. Failure Modes & Graceful Degradation

| Failure | Detection | Degradation path | User impact | Never do |
|---|---|---|---|---|
| Vector DB down | Health check / error rate | BM25-only retrieval; cached embeddings for hot chunks | Recall drops ~15-25%, still answers | Route raw queries to the LLM unretrieved |
| Reranker down | Timeout > 200ms | Skip rerank, serve RRF order | Precision -8-10 pts | Block the request on a non-critical stage |
| LLM provider down | 5xx rate / circuit breaker | Secondary provider -> cached answers -> queue-and-notify | Slower or canned for some queries | Hard-fail without tenant comms |
| Sanitizer down | Health check | **FAIL CLOSED**: reject the request | No response (compliance first) | Bypass the gate "temporarily" |
| Embedding latency spike | p99 monitor on embed stage | Reduce top-k / serve fp16 on-disk index | Slight recall reduction | Queue unbounded — backpressure or shed |
| Observability down | Buffer errors | Fire-and-forget ring buffer, replay on recovery | None | Make tracing a synchronous dependency |
| Semantic cache stale | Content-version key mismatch | Purge by version, bypass cache | Correct answers, higher cost | TTL-only invalidation on regulated content |

Design rule the table encodes: **the failure strategy follows the compliance class** — the sanitizer fails closed (healthcare), while retriever, reranker, and provider fail open with fallbacks (availability).

## 4. Latency Budget (p95, streamed clinical copilot, 2.5s SLA)

Every stage gets an explicit allocation; the sum must clear the SLA with headroom. Re-run this table whenever a feature adds a stage.

| Stage | p95 budget | Notes |
|---|---|---|
| AuthN/Z + gateway + rate limit | 20 ms | Local JWT verify, no extra network hop |
| PII sanitizer (Presidio) | 80 ms | Cached analyzers, 2k-token note |
| Hybrid retrieval (HNSW + BM25, RRF) | 300 ms | Warm replicas, same-AZ |
| Reranker (top-25) | 120 ms | Batched cross-encoder scoring |
| LLM TTFT | 700 ms | Prefill 2.4k tokens, streaming on |
| LLM generation (350 tokens) | 1,100 ms | ~320 tok/s effective |
| Output guardrails | 100 ms | Deterministic inline checks only |
| **Total** | **~2.42s** | 2.5s SLA, ~3% headroom |

Cross-cutting rules: heavy checks (RAGAS judge, deep red-team scans) never sit on the critical path; per-stage p95s are emitted as spans so a budget breach is attributable to a stage, not guessed; and the budget is a product artifact — features negotiate against it in planning, not in production.

## 5. Data Model & Tenant Isolation

| Store | Isolation mechanism | Key shape |
|---|---|---|
| Vector DB | Per-tenant namespace/partition | `tenant:{id}:docs:{version}` |
| Document store (S3) | Prefix + bucket policy per tenant | `s3://docs/{tenant_id}/{doc_id}/` |
| Postgres (orchestrator state, cache) | Row-level security on tenant_id | every table carries `tenant_id` |
| Response/semantic cache | Tenant + policy version + content version in key | `tenant:{id}:pv:{v}:cv:{v}:qh:{hash}` |
| Telemetry (Langfuse) | tenant_id on every span; redact-before-export | trace -> tenant tag mandatory |

Isolation is tested like auth: CI includes cross-tenant access attempts against every store, and a failing isolation test is a release blocker — not a backlog item.

Service-level objectives this architecture commits to (and the gate that enforces each):

- Faithfulness >= 0.95 on the stratified gold set — CI release gate (RAGAS batch + `evaluate_rag_response` inline proxy).
- PII sanitizer recall >= 0.99 against the frozen adversarial set — CI gate + weekly PSI on entity distributions.
- End-to-end p95 <= 2.5s, p99.9 monitored for tail regressions (merges, cold caches) — alerting on spans.
- Tenant isolation: zero cross-tenant reads — CI adversarial tests against every store, release blocker on failure.

## 6. Cost Model (example: 100k queries/day = 3M/month)

Assumptions stated so the interviewer can challenge them: 2,400 prompt + 350 completion tokens per call; $3/$15 per 1M tokens; 40% of prompt tokens hit a provider prompt cache at 50% discount; 30% semantic-cache hit rate on the un-cached remainder.

| Component | Sizing basis | Monthly cost |
|---|---|---|
| LLM inference (uncached baseline) | 7.2B in x $3/1M + 1.05B out x $15/1M | $37,350 |
| — with prompt cache (40% at half price) | in: $21,600 x 0.8 = $17,280 + out: $15,750 | $33,030 |
| — with semantic cache (30% of calls fully served) | x 0.70 | **~$26,100** |
| Reranker (hosted, $2.00 per 1k searches) | 3M searches/mo | $6,000 (or ~$400 self-hosted A10G) |
| Vector DB | 4 x r6g.xlarge x $0.201/hr x 730h | ~$590 (RF=2: ~$1,180) |
| App tier (gateway, orchestrator, guards) | 3 x c7g.2xlarge x ~$0.294/hr x 730h | ~$645 |
| Embeddings (ingest + queries, self-hosted @ $0.10/1M) | ~720M tokens/mo | ~$72 |
| Observability (self-hosted Langfuse + Postgres) | 2 small nodes | ~$400 |
| Storage (S3 docs, backups) | ~1 TB + versioning | ~$80 |
| **Total (tuned)** | | **~$34k/mo, of which ~75% is LLM inference** |

Read the table as a prioritization: token efficiency (prompt design, caching, context trimming) moves the number more than every infrastructure line combined. The repo's `src/interview_prep/cost.py` (`estimate_tokens`, `estimate_cost`, `monthly_projection`) is the arithmetic reference; `docs/theory/05` shows the 50k/day variant ($19.8k/mo) derived with it.

## 7. Scaling Stages

| Stage | Traffic | What changes |
|---|---|---|
| MVP (0->1) | < 1 RPS | Single API + managed provider + single vector index node; observability from day one (cheap), isolation by namespace |
| Product (1->10 RPS) | 100k/day | Reranker, semantic cache, per-tenant quotas, warm HNSW replicas, provider fallback chain |
| Platform (10->100 RPS) | 1M+/day | Sharded vector DB (RF=2), dedicated ingest lane, self-hosted or hybrid models for cost tiers, batch eval pipeline, cost ledger per tenant |

Cross-references: `src/interview_prep/evaluator.py` (`sanitize_pii`, `evaluate_rag_response`), `src/interview_prep/drift.py` (`population_stability_index`, `ks_statistic`, `drift_verdict`), `src/interview_prep/cost.py` (`estimate_tokens`, `estimate_cost`, `monthly_projection`).
