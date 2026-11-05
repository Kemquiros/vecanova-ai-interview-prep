# Theory 5: System Design and Behavioral STAR Mastery

## Interview Soundbites

1. "I design backwards from the latency budget: every stage gets a p95 allocation, and the allocations must sum under the SLA — if they don't, I cut scope, not the SLA."
2. "Retrieval quality is the ceiling on answer quality — I tune recall@k before I touch a single prompt word."
3. "Every component has a degradation path: reranker down means skip rerank, vector DB down means BM25-only, sanitizer down means fail closed."
4. "A STAR story without numbers is an opinion — I always carry before/after metrics with units."
5. "Caching buys latency and cost, but in regulated domains the cache key must carry tenant, policy version, and content version — or it becomes a compliance incident."

## 1. The 60-Minute System Design Interview Shape

For prompts like "Design a clinical analysis copilot for 50k queries/day":

1. **Requirements (5 min)**: functional (RAG over clinical protocols, PII masking, citations) and non-functional (p95 latency SLA, availability, compliance class, monthly cost ceiling). Say the compliance class out loud — it dictates the failure strategy (fail closed for HIPAA, fail open with fallback for e-commerce).
2. **Estimates (5 min)**: derive QPS and token volume. 50k queries/day = 0.58 QPS average, but diurnal peaks run 10-15% of daily volume in one hour -> ~2 QPS sustained peak, design for 10-15 RPS with burst/retry headroom. Tokens: 50k x (2,400 prompt + 400 completion) = 120M prompt + 20M completion tokens/day -> at $3/$15 per 1M tokens that is $660/day, ~$19,800/month. Quote it — cost literacy separates senior from mid.
3. **API + data model (10 min)**: `POST /analyze` (streaming SSE), document ingestion contract, tenant model.
4. **High-level design (15 min)**: mirror the reference diagram in `docs/architecture.md`.
5. **Deep dives (15 min)**: pick 1-2 the interviewer leans into (vector index scaling, caching, multi-tenancy, failure handling). Have the trade-off tables of this doc ready.
6. **Cross-cutting (5 min)**: observability, security, cost attribution, degradation matrix.

## 2. The Latency Budget Table (memorize one of these)

Clinical copilot, end-to-end p95 SLA of 2.5s (streamed):

| Stage | p95 budget | Notes |
|---|---|---|
| AuthN/Z + gateway + rate limit | 20 ms | JWT verify local, no network hop |
| PII sanitizer (Presidio NER) | 80 ms | Cached analyzers; 2k-token note |
| Hybrid retrieval (HNSW top-50 + BM25 top-50, RRF) | 300 ms | Warm index, same-AZ |
| Cross-encoder reranker (top-25) | 120 ms | Batched scoring |
| LLM time-to-first-token | 700 ms | Prefill 2.4k tokens, streaming on |
| LLM full generation (350 tokens) | 1,100 ms | ~320 tok/s effective |
| Output guardrails (PII re-scan + policy) | 100 ms | Deterministic checks only inline |
| **Total** | **~2.42s** | Under 2.5s with ~3% headroom |

Interview line: "When someone asks to add a feature, I re-run this table. Features are free until they break the sum."

## 3. Deep Dive 1 (Trade-off Table): Vector Index Scaling

Sizing math first — it comes up every time:

```
raw_bytes   = n_chunks x dim x bytes_per_scalar
            = 10M x 1536 x 4 B = 61.4 GB (fp32)
HNSW RAM    ~= raw x 1.5-2.0 (graph links, M=16-32 neighbors/layer)
            = ~92-123 GB  ->  4-6 nodes of 32 GB, or 3 nodes at fp16
fp16 raw    = 30.7 GB; PQ 64 B/vector = 0.64 GB (96x compression, recall cost)
shards      = ceil(hnsw_ram / usable_ram_per_node)
```

Then the index-family trade-off table:

| Index | Build (10M x 1536d) | Query p95 | Recall@10 | RAM vs raw | Best for |
|---|---|---|---|---|---|
| HNSW | 30-60 min | 5-15 ms | 0.95+ | 1.5-2.0x | Latency-critical, moderate scale |
| IVF-PQ | 10-20 min | 3-8 ms | 0.85-0.93 | ~0.1x | High-QPS, cost-sensitive |
| DiskANN | hours | 10-30 ms (NVMe) | ~0.95 | ~0.1x + disk | Billion-scale, RAM-poor |
| ScaNN | minutes-hours | fast | high | mid | Research-friendly, tuning appetite |

Tuning knobs to name: `M` (graph degree: 16 default, 32-48 for high recall), `ef_construction` (build quality), `ef_search` (query-time recall/latency dial — expose it as a config, not a constant). Mention segment merges: streaming inserts into HNSW cause background merges that spike p99 — schedule bulk ingests and size replicas for search-only traffic.

## 4. Deep Dive 2 (Trade-off Table): Caching Layers

| Layer | Typical hit rate | Correctness risk | Key design | TTL policy |
|---|---|---|---|---|
| Exact response cache | 5-15% | Staleness only | tenant + policy_ver + content_ver + query_hash | Purge on index update |
| Semantic cache | 20-40% | Wrong-answer risk (near-duplicate != same question) | tenant-scoped, cosine >= 0.97 threshold | Short (hours), per-domain |
| Provider prompt (KV) cache | n/a | None | Stable prefix: system prompt + few-shot first | Cuts TTFT 30-60% on shared prefixes |
| Embedding cache | ~100% on repeats | None | content hash of chunk | Infinite (content-addressed) |

Regulated-domain rule: semantic caches on clinical/financial content need an eval — I ran a 500-query gold set through a 0.97-threshold cache and measured a 1.4% wrong-answer rate; that killed the idea for dosages but kept it for FAQ-tier content. Telling that story (with the number) is exactly the depth senior interviews probe for.

## 5. Deep Dive 3 (Trade-off Table): Streaming vs Batch Evaluation

| Dimension | Inline (streaming) eval | Batch (offline) eval |
|---|---|---|
| Blocks a bad answer | Yes — cheap deterministic checks | No — detect after the fact |
| Metric depth | Faithfulness proxies, PII re-scan, refusal checks | Full RAGAS faithfulness with LLM judge, human review |
| Cost | Judge calls on critical path | 10-50x cheaper, batched |
| Failure semantics | Fail closed or degrade with banner | Page, quarantine dataset, regression-gate next release |

The repo implements the inline proxy in `src/interview_prep/evaluator.py`:

```python
from interview_prep.evaluator import evaluate_rag_response

verdict = evaluate_rag_response(query, response, context_chunks)
if verdict["status"] == "REVIEW_REQUIRED":     # faithfulness or relevance < 0.5
    ...  # degrade: add disclaimer, offer human handoff, log to eval dataset
```

Batch RAGAS runs weekly on a stratified sample plus every REVIEW_REQUIRED item; CI blocks releases when the gold-set faithfulness mean drops > 2 points.

## 6. Behavioral Mastery: STAR With Numbers

- **Situation (20-30s)**: context, scale, stakes. "Clinical copilot at a 2,000-clinician network, 50k queries/day, hallucination complaints reaching the CMO."
- **Task (1 sentence)**: the measurable goal. "Cut hallucination rate below 2% without breaching the 1s p95 retrieval budget."
- **Action (60-70% of the story)**: "I" statements, decisions *with alternatives rejected and why*. This is where seniority lives — juniors list actions, seniors justify trade-offs.
- **Result**: 2-3 quantified metrics + one durable change. "Hallucination 12% -> 1.8%, p99 850ms -> 620ms, and a faithfulness CI gate that caught 3 regressions in the following quarter."

The **metrics ladder** pattern connects system metrics to business outcomes — interviewers remember ladders, not single numbers:

```
+12% clinician weekly adoption   (business)
  <- CSAT 3.9 -> 4.5             (product)
    <- hallucination 12% -> 1.8% (quality)
      <- p99 retrieval 850ms -> 620ms  (system)
```

Anti-patterns that read as junior: "we" without "I" (unclear ownership), results without units or baselines, blaming predecessors, and unbounded scope ("I improved everything").

The full question bank with 15 model answers and interviewer follow-ups lives in `docs/star_bank.md`.

## 7. Cross-References to Repo Code

Drift detection on serving behavior (`src/interview_prep/drift.py`), e.g., monitoring latency or query-length distributions week-over-week:

```python
from interview_prep.drift import (
    population_stability_index, ks_statistic, drift_verdict,
)

psi = population_stability_index(last_week_p95s, this_week_p95s, bins=10)
if drift_verdict(psi) == "SEVERE":   # > 0.25
    ...  # page the on-call; freeze deploys until explained
ks = ks_statistic(expected, actual)  # nonparametric complement
```

Cost projection for capacity planning (`src/interview_prep/cost.py` — signature as in module):

```python
from interview_prep.cost import estimate_tokens, estimate_cost, monthly_projection

n_in  = estimate_tokens(question + " " + " ".join(chunks))   # chars/4 heuristic
per_call = estimate_cost(n_in, tokens_out=350)
print(monthly_projection(per_call, calls_per_day=50_000))    # ~$19.8k/mo at $3/$15
```

## 8. Production War Stories / Pitfalls

1. **Semantic cache served stale protocol answers for 3 days after a guideline update** — the key lacked a content version. Mitigation: version-stamped keys + automatic purge on index update; caches for regulated content are event-invalidated, never TTL-only.
2. **p99 looked fine on dashboards, but 1 in 50 requests took 4s** — HNSW segment merges after bulk ingests. Mitigation: dedicated ingest windows, search-only replicas, single-writer topology; alert on p99.9, not p99.
3. **LangGraph checkpoint store (Postgres) became the bottleneck at 15 QPS** — synchronous state writes per node. Mitigation: async checkpointing with bounded staleness, partitioned tables, and checkpoint retention of 24h (we were keeping forever for no recovery reason).
4. **Offline evals green, production faithfulness dropped 14 points** — the query distribution shifted after a new clinic onboarded (different note templates). Mitigation: PSI on query embeddings and lengths as a weekly gate; eval sets refreshed quarterly from live traffic strata, not frozen forever.
5. **A prompt change added a few-shot block and silently grew cost 38%** — noticed in a monthly review, not by alerting. Mitigation: per-feature token budgets with CI lint on prompt templates (chars/token ceiling), and `monthly_projection` alerts on week-over-week cost deltas > 10%.

## 9. What Interviewers Actually Ask

**Q1: "Design a clinical copilot for 50k queries/day with HIPAA constraints."**
Model answer: walk the 6-step shape from section 1; land the compliance-driven failure strategy, the latency budget table, and the ~$20k/month token math without being prompted.

**Q2: "How would you get retrieval p99 under 1 second?"**
Model answer: "Warm HNSW replicas for search-only traffic, ef_search tuned against a recall target, same-AZ placement, fp16 or PQ index if RAM-bound, top-k sized for the reranker (50 -> 25), and embedding cache for repeated chunks. Then I check the tail: p99.9 usually implicates merges or cold caches, not query path."

**Q3: "Is a semantic cache safe in a regulated environment?"**
Model answer: "Only per-tenant, version-stamped, threshold-gated, and eval'd. I quantify the wrong-answer rate on a gold set before enabling; for dosage-adjacent content I keep it exact-match. Cache is an optimization, never a correctness layer."

**Q4: "Tell me about a production failure you owned."**
Model answer: pick from `docs/star_bank.md` category 1; lead with detection, quantify blast radius, and end with the systemic fix (the thing that cannot regress).

**Q5: "How do you evaluate a RAG system before launch?"**
Model answer: "A 200-500 item gold set built from real query logs, RAGAS faithfulness + context precision/recall, human calibration on 50 items to sanity-check the LLM judge (judge-human Cohen's kappa > 0.7 or the judge is not trusted), plus inline proxies via `evaluate_rag_response` in staging. Launch gate: faithfulness >= 0.95 on the gold set, and every REVIEW_REQUIRED item triaged."

**Q6: "How do you isolate tenants in a shared RAG platform?"**
Model answer: "Data plane: per-tenant namespaces (Pinecone namespaces / Milvus partitions), tenant in every cache key, row-level security in Postgres. Control plane: per-tenant token buckets and quotas. Economics: every trace tagged with tenant_id so cost per tenant is a query, not a project. Isolation bugs are Sev-1s — I test cross-tenant access in CI the way we test auth."

**Q7: "When do you fine-tune instead of RAG + prompting?"**
Model answer: "Default order: prompt -> tools/orchestration -> RAG -> fine-tune. Fine-tune earns its cost when I need consistent format/behavior at scale, latency of a smaller distilled model, or domain style — not for knowledge (that is the index's job, it refreshes). The bar: I've measured prompt+RAG failing on a labeled set, and I can name the metric the fine-tune moves and its training-data plan."

## 10. Further Reading

- LangGraph documentation: https://langchain-ai.github.io/langgraph/
- RAGAS documentation: https://docs.ragas.io/
- Langfuse documentation: https://langfuse.com/docs
- HNSW paper — Malkov & Yashunin: https://arxiv.org/abs/1603.09320
- DiskANN paper: https://arxiv.org/abs/2105.09613
- Google SRE Book (SLOs, error budgets): https://sre.google/sre-book/table-of-contents/
- Designing Data-Intensive Applications (O'Reilly): https://www.oreilly.com/library/view/designing-data-intensive-applications/9781491903063/
- Cohere Rerank guide: https://docs.cohere.com/docs/rerank-guide

Cross-references: `src/interview_prep/evaluator.py` (`evaluate_rag_response`), `src/interview_prep/drift.py` (`population_stability_index`, `ks_statistic`, `drift_verdict`), `src/interview_prep/cost.py` (`estimate_tokens`, `estimate_cost`, `monthly_projection`).
