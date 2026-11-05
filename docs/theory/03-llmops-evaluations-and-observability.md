# Theory 3: LLMOps, Evaluations, and Observability

The operations half of the senior loop: how generative systems are evaluated honestly,
traced, cost-controlled, and monitored for drift after launch. The senior position:
static NLP metrics are canaries, not verdicts; a single blended quality score hides which
pipeline stage broke; and the two numbers that page you at 3 a.m. are p99 latency and
cost per thousand requests.

## Interview Soundbites

- "BLEU and ROUGE measure string overlap; users do not. I decompose quality — faithfulness
  for grounding, answer relevance for usefulness, context precision and recall for the
  retriever — because a single blended score hides which stage broke."
- "RAGAS faithfulness is the fraction of atomic claims in the answer that are entailed by
  the retrieved context: supported claims over total claims. It is a hallucination rate
  with a legal definition."
- "I ship with p50/p95/p99 latency and per-request token cost from day one: the mean hides
  the tail, and the tail is what triggers the pager."
- "LLM-as-judge is a model, not a metric: I validate it against human labels with
  agreement statistics before trusting it, randomize answer position to kill position
  bias, and pin the judge version like any other model artifact."
- "PSI thresholds are my standing drift contract: under 0.1 stable, 0.1 to 0.25 watch,
  over 0.25 severe — the same contract whether the signal is a fraud score or a RAG
  embedding."

## 1. Why Static NLP Metrics Fail for Generative Systems

BLEU is a geometric mean of clipped n-gram precisions against references; ROUGE-L is an
F-measure over the longest common subsequence:

$$\text{BLEU} = \text{BP} \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right) \qquad
\text{ROUGE-L} = \frac{(1 + \beta^2) \cdot P_{lcs} \cdot R_{lcs}}{\beta^2 P_{lcs} + R_{lcs}}$$

Both compare strings against a reference, so they reward paraphrase-adjacent wording,
collapse on open-ended generation with many valid answers, and are gameable by length.
Their honest remaining role is regression canaries on fixed output formats (extractors,
classifiers with templated outputs) — never user-facing quality.

| Metric family | What it measures | Cost | Human correlation | Appropriate use |
| --- | --- | --- | --- | --- |
| BLEU / ROUGE | N-gram overlap vs reference | Negligible | Poor on open-ended generation | Regression canary on fixed formats |
| BERTScore | Semantic token similarity via embeddings | Moderate | Better | Short-answer QA |
| RAGAS family | Faithfulness, relevance, context metrics | Moderate | Good for RAG pipelines | RAG CI gate |
| LLM-as-judge | Rubric adherence, pairwise preference | High | High with rubric and debiasing | Nuanced quality gates, pairwise ranking |

## 2. The RAGAS Decomposition

RAGAS (Retrieval Augmented Generation Assessment) splits RAG quality into stage-local
metrics. **Faithfulness** decomposes the answer into atomic claims $S$ and verifies each
against the retrieved context, yielding the set of supported claims $V$:

$$F = \frac{\lvert V \rvert}{\lvert S \rvert}$$

**Answer relevance** generates candidate questions $\hat{q}_i$ from the answer and scores
how well each embeds back to the original question $q$:

$$\text{AR} = \frac{1}{N} \sum_{i=1}^{N} \cos(q, \hat{q}_i)$$

**Context precision** rewards relevant chunks ranked early — with $v_i \in \{0, 1\}$ the
relevance indicator for chunk at rank $i$ and precision@$i$ the share of relevant chunks
in the top $i$:

$$\text{CP} = \frac{1}{k} \sum_{i=1}^{k} \text{precision}@i \cdot v_i$$

**Context recall** asks what share of ground-truth statements are attributable to the
retrieved context. The stage mapping is the diagnostic power:

| RAGAS metric | Pipeline stage it indicts | What a bad score means |
| --- | --- | --- |
| Context recall | Retriever or corpus coverage | Documents missing, mis-chunked, or stale |
| Context precision | Fusion and ranking | Noise outranks signal in the shortlist |
| Faithfulness | Generation and grounding prompt | The model invents beyond the context |
| Answer relevance | Query understanding and generation | Evasive, verbose, or off-target answers |

Ground the metrics before trusting them: a lexical proxy (word overlap between response
and context, as implemented in `src/interview_prep/evaluator.py`) is cheap, deterministic,
and CI-friendly; an LLM judge is semantic but expensive and version-sensitive. Ship the
proxy in CI, sample the judge offline, and reconcile the two on a labeled audit set.

## 3. LLM-as-Judge, Handled Like a Model

- **Pointwise vs pairwise:** pointwise rubric scoring gives absolute, dashboardable
  numbers; pairwise preference is more reliable for "which output is better" but yields
  relative scores. Use pairwise for model selection, pointwise for monitoring.
- **Known biases:** position bias (first-shown answer wins — mitigate by swapping order
  and averaging), verbosity bias (longer answers win — control for length), and
  self-preference (a judge favors outputs from its own model family — pick a judge from a
  different family than the generator).
- **Validation:** before trusting a judge, measure agreement with human labels. Cohen's
  kappa corrects raw agreement $p_o$ for chance agreement $p_e$:

$$\kappa = \frac{p_o - p_e}{1 - p_e}$$

- **Governance:** pin the judge model version, re-baseline scores on any judge upgrade,
  and log judge rationales for auditability.

## 4. Observability and Tracing

- **Structured traces, not logs:** one trace per request, one span per pipeline stage
  (retrieve, rerank, generate, tool call), with OpenTelemetry-compatible attributes so
  vendor-portable. The trace ID must link the eval record to the exact prompt, retrieved
  chunks, and model version — that linkage is what makes debugging possible.
- **Token accounting:** prompt vs completion vs cached tokens per stage. Token cost is a
  first-class SLO dimension; a feature that triples context triples the bill silently
  unless budgets are per-stage.
- **Latency percentiles:** report p50/p95/p99. For a pipeline of $k$ stages with
  independent tails, end-to-end p99 is approximately bounded by the sum of per-stage p99s
  (union bound):

$$p99_{e2e} \lesssim \sum_{j=1}^{k} p99_j$$

  which is why per-stage latency budgets must be allocated top-down from the user-facing
  SLO, not measured bottom-up after the fact.
- **Error taxonomy:** distinguish provider timeouts, tool failures, guardrail refusals,
  and low-quality-but-200 responses. Only the last one shows up in latency dashboards;
  all four belong in the quality dashboard.
- **Sampling:** 100 percent trace capture for errors and slow requests, head- or
  tail-based sampling elsewhere; store eval-relevant attributes forever, payloads only as
  long as compliance allows.

## 5. Cost Engineering

Token estimation uses the four-characters-per-token heuristic with a floor of one, which
is accurate enough for budgeting English text:

$$\hat{t} = \max\left(1, \left\lceil \frac{\lvert \text{text} \rvert}{4} \right\rceil\right)$$

Blended cost separates input and output pricing (output tokens are priced higher):

$$\text{cost} = \frac{t_{in}}{10^6} \cdot p_{in} + \frac{t_{out}}{10^6} \cdot p_{out}$$

Monthly projection over a 30-day month:

$$\text{monthly} = 30 \cdot r \cdot c$$

with $r$ requests per day and $c$ cost per request. Worked example: 1M input tokens at
\$3/MTok plus 500K output tokens at \$15/MTok is \$3.00 + \$7.50 = \$10.50; 100 requests
per day at \$0.01 each projects to \$30.00 per month. The levers, in impact order: exact
and semantic caching of frequent retrievals and completions, model routing (small model
by default with escalation on complexity), context trimming and system-prompt reuse, and
batch APIs for anything offline.

## 6. Drift Monitoring for LLM Systems

The Population Stability Index over $B$ bins, with reference proportions $e_i$ and live
proportions $a_i$:

$$\text{PSI} = \sum_{i=1}^{B} (a_i - e_i) \ln \frac{a_i}{e_i}$$

with the contract implemented in `src/interview_prep/drift.py`: PSI below 0.1 is STABLE,
0.1 to 0.25 is MODERATE, above 0.25 is SEVERE. The KS statistic complements it:

$$D_{n,m} = \sup_x \lvert F_{1,n}(x) - F_{2,m}(x) \rvert$$

| Dimension | PSI | KS statistic |
| --- | --- | --- |
| Output | Weighted log-ratio sum over bins | Maximum CDF gap |
| Sensitivity | Bin-resolution dependent; coarse bins hide concentrated shifts | Detects local shifts anywhere in the support |
| Decision rule | Rule-of-thumb thresholds (0.1 / 0.25) | Critical values from the sample-size-dependent null distribution |
| Typical placement | Dashboards on features and scores | Formal test on the score or embedding distribution |
| In this repo | `population_stability_index` + `drift_verdict` | `ks_statistic` |

For RAG specifically, monitor three layers: query distribution (embed incoming queries,
track centroid distance and cluster shares), retrieval quality (context recall on a
rotating canary question set), and output quality (faithfulness sampled and judged
offline). A model version change by a provider is drift, too: pin versions, and treat any
unpinned alias as an unmonitored dependency.

## Repository Cross-Reference

Post-generation evaluation gate in `src/interview_prep/evaluator.py`:

```python
from interview_prep.evaluator import evaluate_rag_response, sanitize_pii

safe_query = sanitize_pii(raw_query)  # redact PHI/PII before any logging or tracing
report = evaluate_rag_response(query=safe_query, response=answer, context=top_chunks)
# {"faithfulness": 0.83, "answer_relevance": 0.9, "status": "PASSED", ...}
gate_open = report["status"] == "PASSED"  # both metrics >= 0.5
```

Drift detection in `src/interview_prep/drift.py`:

```python
from interview_prep.drift import drift_verdict, ks_statistic, population_stability_index

psi = population_stability_index(baseline_scores, live_scores)
verdict = drift_verdict(psi)  # "STABLE" | "MODERATE" | "SEVERE"
if verdict == "SEVERE":
    d = ks_statistic(baseline_scores, live_scores)  # localize where the CDF diverges
```

Cost estimation and projection in `src/interview_prep/cost.py`:

```python
from interview_prep.cost import estimate_cost, estimate_tokens, monthly_projection

tokens_in = estimate_tokens(prompt_text)      # ~4 chars per token, floor of 1
usd = estimate_cost(
    tokens_in=tokens_in, tokens_out=400,
    price_in_per_mtok=3.0, price_out_per_mtok=15.0,
)
annual_usd = 12 * monthly_projection(requests_per_day=5000, cost_per_request=usd)
```

## Production War Stories / Pitfalls

- **Prompt bloat as a silent cost regression.** A "helpful" feature tripled injected
  context; the invoice rose \$40k/month with no functional change. Root cause: no
  per-stage token budget and no cost alerting. Mitigation: `estimate_tokens` in CI on
  representative prompts, alert on p95 tokens per request, and trim context by relevance.
- **Silent provider model deprecation.** A pinned-to-latest alias was sunset;
  completions changed format and pricing overnight. Root cause: treating the provider
  model as a stable dependency. Mitigation: pin exact versions, contract tests on golden
  prompts in CI, canary traffic on any version bump, dual-provider fallback.
- **Golden set overfitting.** The team iterated prompts until the eval score hit 0.98 on
  50 cases while real user complaints rose. Root cause: the eval set leaked into prompt
  iterations and was too small to be representative. Mitigation: a frozen holdout the
  iteration loop never sees, at least a few hundred cases, quarterly refresh from live
  traffic, and blind labeling.
- **Judge flip-flop across versions.** Dashboards showed a 12-point quality swing that
  tracked a judge model upgrade, not the product. Root cause: the judge was treated as
  ground truth and never versioned. Mitigation: pin the judge, re-baseline on upgrade,
  and track judge-vs-human agreement continuously.
- **Traces leaked PHI to a third-party SaaS.** Raw patient queries landed in a hosted
  observability tool. Root cause: no redaction at the span boundary. Mitigation:
  `sanitize_pii` before emitting any attribute, redaction inside the instrumentation
  layer itself, self-hosting or DPA-covered vendors for regulated workloads, and audit
  sampling of what is actually stored.
- **p50 SLO masking a p99 disaster.** Median latency was 800 ms while p99 hit 9 seconds
  from a reranker fan-out. Root cause: the SLO was defined on the median. Mitigation:
  per-stage p99 budgets allocated by the union bound, tail-latency alerts, and load tests
  that report percentiles, not means.

## What Interviewers Actually Ask

- **"How do you evaluate an LLM feature end to end?"**
  - Layered: deterministic unit tests, an offline eval set (RAGAS metrics plus rubric
    judge), regression canaries in CI, and an online A/B with guardrail metrics.
  - Add human audit sampling and say what you do when the layers disagree.
- **"What exactly is faithfulness and how would you compute it?"**
  - Claim decomposition plus entailment: $F = \lvert V \rvert / \lvert S \rvert$.
  - Contrast the lexical proxy (cheap, deterministic, CI) with the LLM judge (semantic,
    costly, version-sensitive) and name when you use each.
- **"How do you mitigate prompt injection through retrieved content?"**
  - Treat retrieved text as data, never instructions: delimit it, instruct the model
    accordingly, keep tool permissions least-privilege, filter outputs, and gate
    side-effectful actions behind HITL.
- **"Design observability for a multi-stage LLM pipeline."**
  - Spans per stage with a shared trace ID, token/cost/latency attributes per stage, an
    error taxonomy including low-quality-200s, PII redaction at the boundary, and alerts
    on p99 and cost per 1k requests.
- **"How do you know your eval set is any good?"**
  - Representativeness against live traffic, difficulty spread, inter-annotator
    agreement, stability over time, holdout discipline, and a refresh cadence.
- **"Costs went up 5x after launch. Walk me through the diagnosis."**
  - Token accounting per stage (`estimate_tokens`), request mix by segment, cache hit
    rate, model mix, context creep, retry/timeout loops; then levers ranked by effort and
    impact (caching, routing, trimming, batch).
- **"When does drift matter more than accuracy?"**
  - Decision-support and regulated systems where degradation is silent; the answer is the
    monitoring architecture: PSI/KS on inputs and outputs, calibration monitoring, canary
    question sets, and a rollback policy that has actually been rehearsed.

## Further Reading

- Es, James, Jin, Coca, and Nakov, "RAGAS: Automated Evaluation of Retrieval Augmented
  Generation" (EACL 2024) — https://arxiv.org/abs/2309.15217
- Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena" (NeurIPS 2023) —
  https://arxiv.org/abs/2306.05685
- Papineni, Roukos, Ward, and Zhu, "BLEU: A Method for Automatic Evaluation of Machine
  Translation" (ACL 2002) — https://aclanthology.org/P02-1040/
- Lin, "ROUGE: A Package for Automatic Evaluation of Summaries" (Text Summarization
  Branches Out, 2004) — https://aclanthology.org/W04-1013/
- Zhang et al., "BERTScore: Evaluating Text Generation with BERT" (ICLR 2020) —
  https://arxiv.org/abs/1904.09675
- Post, "A Call for Clarity in Reporting BLEU Scores" (WMT 2018) —
  https://aclanthology.org/W18-6319/
- Langfuse documentation (LLM observability and evals) — https://langfuse.com/docs
- LangSmith documentation (tracing, datasets, evaluation) — https://docs.smith.langchain.com/
- OpenTelemetry Traces concept docs — https://opentelemetry.io/docs/concepts/signals/traces/
