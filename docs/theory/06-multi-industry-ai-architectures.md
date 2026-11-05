# Theory 6: Multi-Industry AI Architectures and Domain Adaptation

## Interview Soundbites

1. "The architecture pattern transfers; the constraints change — latency, compliance, and data velocity are the real interview question."
2. "Fintech buys latency with simpler models: a calibrated logistic regression scoring in 5ms beats a deep net scoring in 90ms when the authorization budget is 100ms."
3. "In healthcare I fail closed; in e-commerce I fail open with a popularity fallback — the compliance class dictates the failure strategy, not the engineer's taste."
4. "PSI under 0.1 is stable, 0.1-0.25 means investigate, over 0.25 means retrain or rollback — the same drift gate guards fraud models, demand forecasts, and retrieval quality."
5. "Multi-tenant isolation is a data-plane problem first: namespace keys, per-tenant quotas, and cost attribution from day one — retrofitting tenant_id into a year of traces is a project, not a patch."

## 1. The Adaptation Framework

Senior interviews test whether you can carry one architecture into a new constraint regime. Internalize this table — it is the single highest-density answer in multi-industry interviews:

| Industry | Hard constraint | Default pattern | Failure strategy | Headline metrics |
|---|---|---|---|---|
| Healthcare | Compliance + audit | RAG + guardrails + private VPC | Fail closed | PII recall >= 0.99, faithfulness >= 0.95, p95 < 2.5s |
| Fintech | Latency (p99 < 50ms scoring) | Real-time tabular ML + graph features | Fail open to rules engine | PR-AUC, FP rate, p99 scoring ms |
| E-commerce | Scale + spikes (5-10x) | Two-tower retrieval + ranker + bandits | Fail open to popularity | Recall@10, NDCG@10, CVR, p95 < 150ms |
| Logistics | Non-stationarity | Gradient boosting + TFT + simulation | Fail to naive baseline | WAPE, MASE, bias |
| SaaS | Multi-tenancy + unit cost | RAG + LLMOps + semantic cache | Degrade per tenant tier | TTFT p95 < 800ms, cost/tenant, hallucination rate |

## 2. Per-Industry Deep Dives

### 2.1 Healthcare (HIPAA / PHI)

- **Constraint**: zero PHI leakage, auditability (6-year logs), hallucination risk in clinical decision support.
- **Patterns**: pre-pipeline anonymization (Presidio NER + regex, fail-closed gate), hybrid retrieval over clinical guidelines, RAGAS faithfulness as a launch and CI gate, private VPC deployment (SageMaker/Vertex) or BAA'd API with zero-retention.
- **Numbers to quote**: PII recall >= 0.99 with layered detector + output re-scan; faithfulness gate 0.95; p95 < 2.5s end-to-end streamed; audit trail 100% of PHI access.
- **Interview angle**: full treatment in `docs/theory/04-healthcare-ai-security-and-guardrails.md` — anonymization vs pseudonymization vs tokenization, PSI on sanitizer entity distributions, war stories (semantic-cache cross-patient leakage, unmask RBAC bug).

### 2.2 Fintech (PCI-DSS / fraud)

- **Constraint**: the authorization flow gives you a total budget around 100ms for the risk decision (network hops excluded); model scoring itself must be p99 < 50ms, feature lookup < 5ms, rules < 2ms.
- **Class imbalance math**: fraud positives run 0.1-0.5% — accuracy is meaningless (predict-all-negative is 99.5% accurate). Optimize **PR-AUC**, not ROC-AUC; handle imbalance via `scale_pos_weight` (LightGBM/XGBoost) or focal loss; **calibrate** probabilities (isotonic > Platt for large data) so the operating threshold is a business decision, not a model artifact.
- **Threshold economics**: optimal cutoff equates marginal FN cost (expected fraud dollars) with marginal FP cost (review labor + customer friction). Quote the frame: "the threshold is a cost equation, I recompute it when either side changes."
- **Graph features**: GNNs / community detection over the transaction graph for AML (money laundering is a graph property, not a transaction property) — but served as *features* into a fast model, not as a p99-critical GNN inference.
- **PCI-DSS**: never store PAN in free text; tokenization before anything enters analytics or prompt context; segment the cardholder data environment (CDE) from model training infra.
- **Explainability**: SHAP values per decision drive adverse-action explanations (ECOA/Reg B for credit) — this is a legal requirement, not a nice-to-have.
- **War story to tell**: post-holiday velocity-feature drift, PSI 0.38 on the velocity feature, false-positive rate 2.1% -> 7.4% in four days; interim fallback to previous champion, weekly PSI gate + champion/challenger promotion; FPR back to 2.3%.

### 2.3 E-commerce (Reco / Search)

- **Architecture**: two-tower retrieval (user tower x item tower, ANN top-1000) -> cross-feature ranker (top-20) -> bandit exploration layer. p95 < 150ms total; retrieval stage < 40ms.
- **Capacity math**: 10M items x 768 dims x 4B = 30.7GB fp32 raw (~46-61GB with HNSW overhead) -> shard from day one; fp16 halves it. Seasonal spikes are 5-10x — retrieval replicas autoscale, the ranker is the thing you load-test.
- **Cold start**: content-based fallback embeddings + Bayesian-smoothed popularity prior (Beta(1,1) -> alpha/beta updating); never cold-store a zero.
- **Bandits**: Thompson sampling with 5-10% exploration traffic and a hard revenue floor — quote "guardrail metrics: exploration may not cost more than 0.5% CVR during the learning window."
- **Metrics**: recall@10 (retrieval), NDCG@10 (ranking), CTR/CVR (online), with inverse-propensity weighting for position bias.
- **War story to tell**: catalog refresh (price/description changes) silently invalidated item embeddings — recall@10 fell 0.82 -> 0.64; fixed by content-hash-versioned embeddings + incremental re-embed pipeline; recall recovered to 0.84 after also switching to a stronger encoder, all verified with a shadow dual-read.

### 2.4 Logistics (Forecasting / Routing)

- **Constraint**: demand series are non-stationary (promos, weather, holidays), intermittent for slow movers, and hierarchical (SKU x DC x region).
- **Modeling**: gradient-boosted trees with lag/rolling features remain the strong baseline; Temporal Fusion Transformers when you have static covariates (warehouse attributes) + known-future inputs (promo calendar, holidays) + observed series (weather); Prophet for interpretable per-series baselines. Reconcile hierarchical forecasts (MinT or bottom-up for sparse series).
- **Metrics**: **WAPE < 15% for fast movers**; MAPE explodes near-zero-demand series — use **MASE** (scaled against a seasonal naive forecast) for intermittent demand. Bias separately from variance: a 0-bias forecast that's wrong in both directions is healthier than a 2%-biased one.
- **Optimization**: OR-Tools for VRP; RL for dynamic routing only behind a **digital twin validated against 90 days of GPS traces** — sim-to-real gap first, policy second.
- **Latency class**: this is batch (hourly/daily forecasts) plus a near-real-time routing service (p95 < 1s) — the slowest industry of the five, and interviews test whether you over-engineer latency that the domain does not need.
- **War story to tell**: forecasting "everything" was the ask; negotiated a 3-SKU pilot with a WAPE gate; pilot hit 9% WAPE on fast movers, then scaled to 40% of catalog — scope negotiation is the senior skill here.

### 2.5 SaaS (Multi-Tenant LLMOps)

- **Isolation**: per-tenant vector namespaces (Pinecone namespaces / Milvus partitions), tenant in *every* cache key, row-level security in Postgres, cross-tenant access tests in CI treated like auth tests.
- **Noisy neighbors**: per-tenant token buckets and quotas; bulk ingest goes to a dedicated lane so one enterprise tenant cannot saturate the embedding endpoint; priority queues by contract tier.
- **Cost attribution**: tag every trace with tenant_id; cost per tenant must be a dashboard query, not a forensic project. Semantic cache (tenant-scoped, similarity >= 0.97) cuts LLM spend 25-40% on FAQ-tier traffic.
- **Latency**: TTFT p95 < 800ms streamed; concurrency is a KV-cache memory problem, not just weights — budget ~0.5GB KV per 4k-context stream for 7B-class models, so 36 concurrent streams need ~18GB of KV headroom on top of weights.
- **Eval per tier**: enterprise contracts increasingly carry a faithfulness SLA — make the eval gate part of the contract review, not an afterthought.

## 3. Trade-off Table 1: Private VPC vs API Providers

The deployment question in every regulated-industry interview:

| Dimension | Private VPC (SageMaker/Vertex, open weights) | API provider (with BAA/DPA) |
|---|---|---|
| Time to market | Weeks-months | Hours-days |
| Cost at 10B tok/mo steady | ~$0.5-1.5 per 1M effective at 60%+ GPU utilization | $3-15 per 1M list price |
| Data control | Full, audit-friendly | Contractual; subprocessor chain risk |
| Latency control | Own autoscaling; you own p99 variance | Provider queue effects at peak |
| Ops burden | High: weights, KV-cache tuning, GPUs | Low: config and keys |
| Use when | PHI/PCI-strict workloads; > 2B tok/mo steady state | Unregulated tiers, prototypes, spiky traffic |

Decision line to deliver: "I prototype on APIs, and I model the crossover: if steady-state volume and compliance class both point the same way, VPC wins; if only compliance points there, VPC *for that tier* with a shared router — deployment is per data class, not per company."

## 4. Trade-off Table 2: Streaming vs Batch Evaluation (Cross-Industry)

| Dimension | Inline (streaming) eval | Batch (offline) eval |
|---|---|---|
| Fraud | Rules + calibration checks inline (the decision IS the eval) | Weekly PR-AUC + drift review |
| Healthcare | PII re-scan + faithfulness proxy inline | RAGAS + human audit trail for compliance |
| E-commerce | Guardrail bandit checks inline | Offline NDCG/recall replay on logged data |
| SaaS | TTFT + refusal checks inline | Per-tenant quality dashboards, regression gates |
| Logistics | n/a (batch domain) | Backtest + PSI on residuals |

Deliverable line: "Inline eval buys *blocking power* at latency cost; batch eval buys *depth* at detection delay. Regulated systems need both: block on cheap deterministic rails, score deeply in batch, and alert on the batch trendline."

## 5. Trade-off Table 3: Prompt+RAG vs Fine-Tune

| Dimension | Prompt + RAG | Fine-tune |
|---|---|---|
| Time to ship | Days | 2-6 weeks |
| Incremental cost | Inference only | $2k-50k per run + MLOps surface |
| Data requirement | None upfront | 1k-100k labeled examples |
| Knowledge freshness | Index refresh (minutes) | Frozen at train time |
| Best at | Knowledge tasks, citations, evolving content | Format compliance, style, distilling to a smaller/faster model, structured-output reliability |

Rule: "Fine-tuning is the last step, not the first — prompt, tools, orchestration, RAG, and only then fine-tune, when I can name the metric it moves that the stack above it cannot."

## 6. The Cross-Industry Drift Gate (one implementation, five consumers)

Drift is the shared enemy. The repo centralizes it in `src/interview_prep/drift.py` — PSI with $B=10$ bins on the reference deciles:

$$\text{PSI} = \sum_{i=1}^{B} (A_i - E_i)\ln\frac{A_i}{E_i}$$

Thresholds: < 0.1 stable, 0.1-0.25 moderate (investigate), > 0.25 severe (retrain/rollback); KS statistic as the nonparametric complement.

```python
from interview_prep.drift import (
    population_stability_index, ks_statistic, drift_verdict,
)

psi = population_stability_index(expected, actual, bins=10)
verdict = drift_verdict(psi)            # STABLE | MODERATE | SEVERE
ks = ks_statistic(expected, actual)     # sup |F_a - F_b|, p < 0.05 flags shift
```

What each industry feeds it: fraud -> velocity/amount/merchant-mix features; e-commerce -> query-embedding cluster distribution and CTR baseline; logistics -> forecast residuals by series class; healthcare -> note lengths and ICD-10 code mix; SaaS -> per-tenant query topics. Same function, different feature lists — that is the point of the architecture.

## 7. Production War Stories / Pitfalls

1. **Fraud false-positive storm after a holiday week** (PSI 0.38 on the velocity feature, FPR 2.1% -> 7.4% in four days). Mitigation: weekly PSI gate on the top-20 features, champion/challenger promotion with automatic rollback, and a rules-engine fallback so the transaction path never hard-depends on the ML model's availability.
2. **Stale embeddings after a catalog refresh** (recall@10 0.82 -> 0.64 over two weeks). Mitigation: content-hash-versioned item embeddings, incremental re-embed on change events, shadow dual-read to verify encoder swaps before cutover.
3. **Promo SKUs broke the demand forecast** because the promo calendar was not a covariate — WAPE 21% on promo weeks. Mitigation: known-future covariates (promos, holidays) mandatory in the feature contract; WAPE 21% -> 11%; and a "forecast review" that inspects the promo weeks specifically, not the aggregate.
4. **One enterprise tenant's bulk ingest saturated the shared embedding endpoint** and degraded TTFT for every tenant. Mitigation: per-tenant quotas, dedicated ingest lane with backpressure, and tenant-scoped latency SLOs so the noisy neighbor is measurable and enforceable.
5. **RL routing policy looked brilliant in simulation and failed the pilot** — the digital twin used optimistic travel times. Mitigation: validate the simulator against 90 days of GPS traces first (sim-to-real gap metric on travel-time percentiles); no policy ships until the sim error on p90 travel times is under 10%.
6. **Per-tenant cost attribution was missing for the first year** — nobody could answer "what does this customer cost us?" Mitigation: tenant_id on every trace from day one in greenfield; for brownfield, a backfill project from provider logs with a documented confidence level. Cost/tenant is a unit-economics input, not an ops nicety.

## 8. What Interviewers Actually Ask

**Q1: "You've built AI systems in five industries. What actually changes between them?"**
Model answer: deliver the section-1 table verbatim, then pick one row and go two levels deep (fraud threshold economics or healthcare fail-closed are the strongest). Do not narrate all five rows — the table shows breadth; one deep dive shows depth.

**Q2: "Fraud model vs recommender — compare the latency budgets and failure strategies."**
Model answer: "Fraud: ~100ms total, p99 < 50ms scoring, fail *open* to a rules engine because blocking payments is worse than a bad transaction sneaking through... unless the fraud class is severe, in which case fail to manual review. Recommender: 150ms p95, fail *open* to popularity because an empty shelf loses more than a generic one. Same availability target, opposite failure semantics — because the blast radius differs."

**Q3: "How do you detect drift in an LLM product?"**
Model answer: "Three layers: PSI on input distributions (query lengths, embedding clusters, entity mixes) weekly; regression evals on a stratified gold set from live traffic; and outcome proxies (refusal rate, escalation rate, faithfulness sample) monitored hourly. The repo's `drift.py` gates are the reference implementation — the art is choosing *which* distributions to bin, not the statistic."

**Q4: "When would you fine-tune?"**
Model answer: section-5 rule, then one concrete case: "I distilled a 70B RAG chain into a 8B fine-tune for a classification-adjacent task when p99 latency and $/query both demanded it — the eval showed the small model matched 98% of the chain's decisions, which paid for the training run in three weeks."

**Q5: "Design cost attribution for a multi-tenant LLM platform."**
Model answer: "Tag at the edge: tenant_id on every request, propagated through every span. Tokens per tenant per model per feature roll up to a cost ledger; semantic-cache hits and prompt-cache discounts are credited back. Chargeback or showback per contract tier, with a floor per tenant for fixed costs. The design constraint is that attribution must survive the fallback path — if the secondary provider serves it, the ledger still records it."

**Q6: "0.3% fraud positives — walk me through your training choices."**
Model answer: "PR-AUC as the headline metric, scale_pos_weight or focal loss for the imbalance, temporal split (never random — leakage), probability calibration so the threshold is a business dial, and threshold set by the cost equation. Report precision at the operating point and the review queue load, because those are what the fraud ops team feels."

**Q7: "Ship an LLM feature under HIPAA in six weeks. Go."**
Model answer: "Week 1: data-flow map + BAA decisions, fail-closed sanitizer skeleton. Weeks 2-3: RAG over de-identified chunks in a VPC, gold set + eval harness. Week 4: guardrails (input rails, PII re-scan) + audit logging. Week 5: red-team + PSI baselines, shadow mode with clinicians. Week 6: staged rollout with the faithfulness gate live. The discipline is that compliance items are on the critical path from day 1 — bolted-on compliance is how you get the war stories in section 7."

## 9. Positioning for LATAM -> US Remote Roles (bonus)

- **Comp benchmarks**: levels.fyi for band anchors; LinkedIn Salary for regional context. Quote ranges, not vibes.
- **Structure**: most LATAM-US remote work runs through EOR/contractor platforms (Deel, Remote) — know the tax/compliance shape at a conversational level; US hiring managers ask.
- **Overlap expectation**: 4+ hours of ET overlap is the de facto standard; state it proactively.
- **Differentiation**: quantified war stories (this repo's STAR bank), compliance literacy (HIPAA/PCI vocabulary used correctly), and bilingual documentation — most candidates offer one of the three.

References: https://www.levels.fyi/ , https://www.linkedin.com/salary/ , https://www.deel.com/ , https://remote.com/

## 10. Further Reading

- SHAP documentation: https://shap.readthedocs.io/en/latest/
- Temporal Fusion Transformers (paper): https://arxiv.org/abs/1912.09363
- Prophet documentation: https://facebook.github.io/prophet/
- LightGBM documentation (imbalance/`scale_pos_weight`): https://lightgbm.readthedocs.io/en/stable/
- PCI DSS v4.0 standard: https://www.pcisecuritystandards.org/standards/pci-dss/
- Pinecone documentation (namespaces): https://docs.pinecone.io/
- Milvus documentation (partitions): https://milvus.io/docs
- FAISS wiki (index selection): https://github.com/facebookresearch/faiss/wiki
- Google OR-Tools (VRP/routing): https://developers.google.com/optimization
- OWASP Top 10 for LLM Applications: https://genai.owasp.org/

Cross-references: `src/interview_prep/drift.py` (`population_stability_index`, `ks_statistic`, `drift_verdict`), `src/interview_prep/cost.py` (`estimate_tokens`, `estimate_cost`, `monthly_projection`), `src/interview_prep/evaluator.py` (`evaluate_rag_response`).
