# Tutorial 2: LLMOps — BM25 to Hybrid Retrieval to RAGAS-Style Evaluation

> Who this is for: engineers preparing for Senior AI/ML interviews who need to
> demonstrate a full retrieval-augmented generation (RAG) quality loop: lexical
> retrieval, chunking, hybrid ranking with reciprocal rank fusion, and automated
> RAGAS-style evaluation wired as a regression gate. Time budget: 75 minutes.

## What you will build

1. A BM25 search index over a small clinical-knowledge corpus.
2. A chunker that turns a long policy document into overlapping RAG chunks.
3. A hybrid retriever that fuses a simulated dense ranking with BM25 via reciprocal
   rank fusion (RRF).
4. A RAGAS-style evaluation loop (`evaluate_rag_response`) that scores faithfulness and
   answer relevance, plus a golden-set regression gate you could drop into CI.
5. Token and cost observability for the pipeline.

## The pipeline in one line

```text
documents -> chunk_text -> BM25Index + (simulated) dense ranker
          -> reciprocal_rank_fusion -> top-k context -> response
          -> evaluate_rag_response (faithfulness, answer_relevance) -> CI gate
```

## Step 1: Set up the environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
python -c "from interview_prep.retrieval import BM25Index; from interview_prep.evaluator import evaluate_rag_response; print('imports ok')"
```

Expected output:

```text
imports ok
```

**Google Colab variant:** open `notebooks/ai_interview_masterclass.ipynb` in Colab and
run in a fresh cell:

```python
%cd /content/vecanova-ai-interview-prep
!pip install -e ".[dev]"
```

Re-run the install once per Colab session; the environment is ephemeral.

## Step 2: Tokenize and index the corpus with BM25

Create a small corpus of five one-line documents mixing clinical and retrieval
knowledge (keep it tiny so you can predict rankings by hand):

```python
from interview_prep.retrieval import tokenize, BM25Index

docs = [
    "Metformin is the first-line medication for type 2 diabetes. The typical starting dose is 500 mg twice daily with meals.",
    "Hypertension is diagnosed when systolic blood pressure stays above 140 mmHg. Lifestyle changes are tried before medication.",
    "BM25 is a bag-of-words ranking function that scores documents using term frequency and inverse document frequency.",
    "Vector search embeds documents and queries into dense vectors and retrieves nearest neighbors by cosine similarity.",
    "Reciprocal rank fusion combines multiple ranked lists by summing 1 divided by k plus rank for every document.",
]

print(tokenize("Dose of Metformin for Type 2 Diabetes"))

index = BM25Index(docs)
print(index.search("metformin dose diabetes", top_k=3))
```

Expected output:

```text
['dose', 'of', 'metformin', 'for', 'type', '2', 'diabetes']
[(0, 3.8687)]
```

Two things to notice. First, `tokenize` lowercases and strips punctuation, so
"Metformin" and "metformin" are the same term — the index is case-insensitive by
construction. Second, only document 0 comes back: it matches every query term, while
documents 1 and 2 share no query vocabulary and `search` returns only hits with a
positive score — a query with zero overlap yields an empty list, not zero-score rows.

BM25's two knobs, `k1` (term-frequency saturation, default 1.5) and `b` (length
normalization, default 0.75), matter when documents vary in length. The index uses the
Lucene IDF variant `ln(1 + (N - df + 0.5) / (df + 0.5))`, which stays positive; scores
are rounded to four decimals.

## Step 3: Chunk a long document for RAG

Retrievers work over chunks, not whole documents. Build a realistic policy document and
chunk it with the defaults (64 words per chunk, 16-word overlap):

```python
from interview_prep.retrieval import chunk_text

long_doc = (
    "This clinical data policy defines how patient records flow through the analytics platform. "
    "All protected health information must be masked before leaving the secure enclave. "
    "Batch jobs run nightly and stream processing handles vitals telemetry in near real time. "
    "Every request is audited and retained for seven years under HIPAA rules. "
    "Analysts access de-identified extracts only, and re-identification attempts are blocked and logged. "
    "Third-party model endpoints never receive raw identifiers; the sanitizer rewrites them deterministically. "
    "Quarterly access reviews remove accounts that no longer require data access. "
    "Incidents must be reported to compliance within twenty four hours."
)

chunks = chunk_text(long_doc)
print(len(chunks))
print(len(chunks[0].split()))
print(chunks[0][:90] + " ...")
```

Expected output:

```text
2
64
This clinical data policy defines how patient records flow through the analytics platform. ...
```

With `chunk_size=64` and `overlap=16` the window advances 48 words per step, so the tail
of chunk 1 overlaps the head of chunk 2. That overlap is what saves you when a fact
straddles a boundary — the classic "answer cut in half" retrieval bug interviewers ask
about. This policy document is 96 words long, so you get 2 chunks (words 0-63, then
48-95); longer documents scale the chunk count with the same 48-word stride, and a
document of 64 words or fewer is returned as a single chunk.

## Step 4: Simulate a dense retriever

Hybrid retrieval fuses a lexical list (BM25) with a semantic list (embeddings). This
repo has no embedding service, so simulate the dense channel with a lexical-overlap
scorer — the *plumbing* is identical to production, where you would swap in FAISS or a
managed vector search:

```python
from interview_prep.retrieval import tokenize

def overlap_score(query: str, doc: str) -> float:
    q, d = set(tokenize(query)), set(tokenize(doc))
    return len(q & d) / (len(q) * len(d)) ** 0.5

def dense_ranking(query: str, corpus: list[str], top_k: int = 3) -> list[int]:
    scored = sorted(
        ((i, overlap_score(query, doc)) for i, doc in enumerate(corpus)),
        key=lambda pair: pair[1],
        reverse=True,
    )
    return [i for i, _ in scored[:top_k]]

query = "how do we rank documents with term frequency"
bm25_hits = index.search(query, top_k=3)
bm25_ranking = [doc_id for doc_id, _ in bm25_hits]
sim_ranking = dense_ranking(query, docs, top_k=3)
print("bm25 :", bm25_ranking)
print("dense:", sim_ranking)
```

Expected output:

```text
bm25 : [2, 4, 0]
dense: [2, 3, 4]
```

BM25 pulls in document 0 (its only match is the word "with") that the overlap scorer
ignores, while the overlap scorer promotes document 3 on the single shared term
"documents". Two channels, two opinions. That disagreement is exactly what fusion will
resolve.

## Step 5: Fuse the rankings with reciprocal rank fusion

```python
from interview_prep.retrieval import reciprocal_rank_fusion

def rrf_score(doc_id: int, rankings: list[list[int]], k: int = 60) -> float:
    return sum(
        1.0 / (k + rank)
        for ranking in rankings
        for rank, d in enumerate(ranking, 1)
        if d == doc_id
    )

rankings = [bm25_ranking, sim_ranking]
fused = reciprocal_rank_fusion(rankings, k=60)
print(fused)
for doc_id in fused:
    print(doc_id, round(rrf_score(doc_id, rankings), 4))
```

Expected output:

```text
[2, 4, 3, 0]
2 0.0328
4 0.032
3 0.0161
0 0.0159
```

RRF gives each document `1 / (k + rank)` per list it appears in, with `k=60` damping the
head of each list so no single channel dominates. Read the output as a story: document 2
leads both lists and wins outright; document 3, which BM25's top-3 *missed entirely*,
still lands third because the dense channel ranked it second. Miss once, rank once,
recover — that is the whole argument for hybrid retrieval, and it is the sentence to say
in interviews.

## Step 6: Evaluate answers RAGAS-style

Now close the loop. Retrieve context, generate a response (canned here — swap in your
LLM call in production), and score it with `evaluate_rag_response`, which computes
RAGAS-style faithfulness (is the response grounded in the context?) and answer
relevance (does it address the query?):

```python
from interview_prep.evaluator import evaluate_rag_response

q = "What is the recommended dosage of metformin for adults?"
ctx = [
    "The recommended starting dosage of metformin for adults is 500 mg twice daily with meals.",
    "Maximum daily dose should not exceed 2000 mg unless supervised by an endocrinologist.",
]

good = "The recommended starting dosage of metformin for adults is 500 mg twice daily with meals, and the maximum daily dose is 2000 mg."
hallucinated = "Metformin was discovered in 1922 and is derived from the French lilac plant, famous in herbal medicine."

print(evaluate_rag_response(q, good, ctx))
print(evaluate_rag_response(q, hallucinated, ctx))
print(evaluate_rag_response(q, good, []))
```

Expected output:

```text
{'faithfulness': 1.0, 'answer_relevance': 1.0, 'status': 'PASSED', 'reason': 'Metrics computed successfully.'}
{'faithfulness': 0.28, 'answer_relevance': 0.5, 'status': 'REVIEW_REQUIRED', 'reason': 'Metrics computed successfully.'}
{'faithfulness': 0.0, 'answer_relevance': 0.0, 'status': 'FAILED', 'reason': 'Empty context provided.'}
```

The three rows are the three states your CI gate cares about:

- Grounded answer -> `PASSED` (both metrics 1.0).
- Fluent but ungrounded answer -> `REVIEW_REQUIRED` (faithfulness 0.28: most of its
  words are unsupported by context). This is what hallucination looks like to an
  automated judge.
- Retrieval returned nothing -> `FAILED` with a reason. Never evaluate a response
  against empty context; fail the pipeline before the LLM call.

## Step 7: Turn evaluation into a regression gate

A RAGAS score you eyeball once is a demo; a gate that blocks deploys is LLMOps. Wrap a
golden set in an assertion loop:

```python
from interview_prep.evaluator import evaluate_rag_response

GOLDEN_SET = [
    {
        "query": q,
        "context": ctx,
        "response": good,
        "min_faithfulness": 0.8,
    },
    {
        "query": "How does BM25 rank documents?",
        "context": ["BM25 ranks documents using term frequency, inverse document frequency and length normalization."],
        "response": "BM25 ranks documents with term frequency, IDF, and document length normalization.",
        "min_faithfulness": 0.8,
    },
]

def run_gate(golden_set: list[dict]) -> bool:
    all_passed = True
    for case in golden_set:
        report = evaluate_rag_response(case["query"], case["response"], case["context"])
        ok = report["status"] == "PASSED" and report["faithfulness"] >= case["min_faithfulness"]
        print(case["query"][:40], report["status"], report["faithfulness"], "OK" if ok else "REGRESSION")
        all_passed = all_passed and ok
    return all_passed

print("gate:", "GREEN" if run_gate(GOLDEN_SET) else "RED")
```

Expected output:

```text
What is the recommended dosage of metf PASSED 1.0 OK
How does BM25 rank documents? PASSED 1.0 OK
gate: GREEN
```

Wire `run_gate` into CI after your test suite: if it returns `False`, the deploy stops.
That single function is the difference between "we evaluate sometimes" and "evaluation
is a release gate" — say exactly that in the interview.

## Step 8: Token and cost observability

Evaluations watch quality; LLMOps also watches money. Estimate tokens and project the
monthly bill with the repo's cost helpers (uses a chars/4 token heuristic):

```python
from interview_prep.cost import estimate_tokens, estimate_cost, monthly_projection

answer = "Escalate to the on-call endocrinologist and recheck glucose q15 minutes."
tokens_in = estimate_tokens("System: you are a clinical assistant. Answer from given context.")
tokens_out = estimate_tokens(answer)
print(tokens_in, tokens_out)

cost = estimate_cost(
    tokens_in=tokens_in,
    tokens_out=tokens_out,
    price_in_per_mtok=1.50,
    price_out_per_mtok=6.00,
)
print(cost)
print(monthly_projection(requests_per_day=25000, cost_per_request=cost))
```

Expected output (both strings are exact multiples of four characters, so the 4
characters-per-token heuristic is unambiguous):

```text
16 18
0.000132
99.0
```

`estimate_cost` converts tokens to dollars at per-million-token prices
(`tokens / 1e6 * price`), and `monthly_projection` multiplies daily volume by cost per
request by 30 days. The unit economics here are tiny per request; the interview move is
to scale them — at 25,000 requests per day even cheap requests compound, which is why
cost-per-request belongs on the same dashboard as faithfulness.

## What you learned

- `tokenize` is the shared vocabulary layer: lowercase, punctuation-free, deterministic.
- `BM25Index` ranks by term frequency and IDF with `k1`/`b` knobs and returns
  `(doc_id, score)` tuples, zero-score hits included.
- `chunk_text` builds overlapping windows (default 64/16, stride = size - overlap) so
  boundary-straddling facts survive retrieval.
- `reciprocal_rank_fusion` merges ranked lists with `1/(k+rank)` so a document missed
  by one channel can still surface via the other.
- `evaluate_rag_response` returns the RAGAS-style verdict triple you gate deploys on:
  faithfulness, answer_relevance, status.
- `estimate_tokens` / `estimate_cost` / `monthly_projection` close the LLMOps loop with
  unit economics.

## Exercises

### Exercise 1: Predict-then-verify BM25

Before running anything, predict the top document for the query `"hypertension blood
pressure medication"`. Then run `index.search` and check yourself. Which document wins
and why does document 0 score zero?

<details>
<summary>Solution</summary>

```python
print(index.search("hypertension blood pressure medication", top_k=3))
```

Output:

```text
[(1, 5.1634), (0, 0.8144)]
```

Document 1 contains all three distinctive terms ("hypertension", "blood pressure",
"medication"), so it dominates. Document 0 shares only the word "medication" and scores
a distant 0.8144, while document 2 shares nothing and is filtered out entirely. Lexical
retrieval is exact-match: the paraphrase "high blood pressure" would score document 1
at zero and return an empty list. That blind spot is the motivation for the dense
channel in Step 4.
</details>

### Exercise 2: Sensitivity of RRF to k

Re-fuse the two rankings from Step 5 with `k=1` instead of `k=60` and compare the fused
order. What changes and what does that tell you about the damping parameter?

<details>
<summary>Solution</summary>

```python
fused_k1 = reciprocal_rank_fusion([bm25_ranking, sim_ranking], k=1)
print("k=60:", reciprocal_rank_fusion([bm25_ranking, sim_ranking], k=60))
print("k=1 :", fused_k1)
```

Output:

```text
k=60: [2, 4, 3, 0]
k=1 : [2, 4, 3, 0]
```

For these two short lists the winner set is stable at both settings — a good sanity
check — but the *margins* differ: with `k=1`, rank-1 (1/2) dwarfs rank-3 (1/4), so the
head of a single list dominates more easily. With `k=60`, contributions compress toward
`1/60`, and appearing in *both* lists matters more than being first in one. Tune `k` on
labeled queries, not by intuition.
</details>

### Exercise 3: Break the gate on purpose

Add the hallucinated metformin response to `GOLDEN_SET` with
`min_faithfulness: 0.8` and confirm the gate turns RED. Then repair the case by
replacing the response with one grounded in `ctx` and confirm GREEN again.

<details>
<summary>Solution</summary>

```python
bad_case = {"query": q, "context": ctx, "response": hallucinated, "min_faithfulness": 0.8}
print("gate:", "GREEN" if run_gate(GOLDEN_SET + [bad_case]) else "RED")
```

Output:

```text
What is the recommended dosage of metf PASSED 1.0 OK
How does BM25 rank documents? PASSED 1.0 OK
What is the recommended dosage of metf REVIEW_REQUIRED 0.28 REGRESSION
gate: RED
```

The gate fails closed: an ungrounded response is a deploy blocker, not a warning.
Swapping `bad_case["response"]` for the grounded `good` string restores GREEN. In CI you
would print the failing case's `reason` field and the diff between intended and actual
context so the fix is one click away.
</details>

## Interview lens

This tutorial is the LLMOps spine interviewers probe in three moves. Move one: "walk me
through your retrieval stack" — answer with Steps 2-5, emphasizing *why* hybrid (Step
5's recovery story) and where chunk boundaries bite (Step 3's overlap). Move two: "how
do you know RAG works in production?" — answer with Step 6's metric semantics and name
the failure states explicitly (`REVIEW_REQUIRED` for hallucination, `FAILED` for empty
retrieval). Move three: "how does it stay working?" — answer with Step 7's golden-set
gate and Step 8's cost line, and volunteer that thresholds like `min_faithfulness` are
versioned artifacts, not vibes. If asked what RAGAS adds over eyeballing, the one-line
answer is: it converts answer quality into a regression-testable number.

## Common errors

- `ModuleNotFoundError: No module named 'interview_prep.retrieval'` — an older checkout.
  Pull the latest branch and re-run `pip install -e ".[dev]"`.
- `ModuleNotFoundError: No module named 'interview_prep'` — you ran Python from outside
  the repository or skipped the editable install. Activate the venv and re-run Step 1.
- Running `python` outside the venv (system Python without the package). Prefer
  `source .venv/bin/activate` and verify with `which python` (it must point into
  `.venv/bin/`).
- `ValueError: docs must be a non-empty list` when constructing `BM25Index([])` — the
  index refuses empty corpora; check `len(docs) > 0` before indexing.
- `ValueError: overlap must be smaller than chunk_size` from `chunk_text` — the window
  must actually advance; keep `overlap` strictly below `chunk_size`.
- An empty `[]` from `index.search` means zero lexical overlap with every document —
  handle the empty-list case before building the context list for the LLM.
- Colab: forgetting `%cd` into the repo before `!pip install -e .` — the install
  succeeds but `interview_prep` is not importable from the notebook's working
  directory. Re-run the `%cd` cell first.
- On Colab, `%pip install -e .` only lasts for the session; after a runtime restart,
  re-run the setup cell before any import.
