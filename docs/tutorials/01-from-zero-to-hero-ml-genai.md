# Tutorial 1: From Zero to Hero — Bridging Traditional ML and GenAI

> Who this is for: engineers preparing for Senior AI/ML interviews who can write Python
> but have never connected classical predictive models (anomaly detection, recommenders)
> to a generative AI pipeline. Time budget: 60 minutes.

## What you will build

1. A calibrated anomaly detector over clinical telemetry (Isolation-Forest-style scoring).
2. An embedding-based product recommender using cosine similarity.
3. A deterministic "ML context payload" that injects model outputs into an LLM prompt —
   the exact pattern interviewers want you to articulate for production GenAI systems.
4. A five-industry mapping of the same pattern (Healthcare, Fintech, E-commerce,
   Logistics, SaaS).

Every step runs locally with the standard library only — no GPU, no API keys.

## Step 1: Set up the environment

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
make test
```

Expected output (abridged; your pass count grows as new modules land):

```text
Successfully installed interview_prep-0.1.0 ...
...
tests/integration/test_pipeline.py .                                    [ 25%]
tests/unit/test_evaluator.py ...                                        [100%]
============================== 4 passed in 0.38s ===============================
```

The `.[dev]` extra installs pytest, pytest-cov, ruff and build on top of the runtime
dependencies (pydantic, typer, rich). `pip install -e .` puts `interview_prep` on your
path in editable mode, so every import in this tutorial resolves to `src/interview_prep/`.

**Google Colab variant:** upload the repository folder (or your fork) to Drive, open
`notebooks/ai_interview_masterclass.ipynb` in Colab, then in a fresh cell run:

```python
%cd /content/vecanova-ai-interview-prep
!pip install -e ".[dev]"
```

Colab has no venv to activate — the editable install into the Colab environment is all
you need. Re-run the install cell once per session (Colab disks are ephemeral).

## Step 2: Run the anomaly detector on clinical telemetry

```python
from interview_prep.traditional_ml import MockAnomalyDetector

detector = MockAnomalyDetector(threshold=2.0)

stable_vitals = [98.0, 100.0, 99.0, 101.0]          # heart rate, bpm
telemetry     = [120.0, 122.0, 210.0, 119.0]        # one systolic spike

print(detector.compute_anomaly_score(stable_vitals))
print(detector.is_anomaly(stable_vitals))
print(detector.compute_anomaly_score(telemetry))
print(detector.is_anomaly(telemetry))
```

Expected output:

```text
1.34
False
1.73
False
```

`compute_anomaly_score` measures how many standard deviations the most extreme reading
sits from the mean (a z-score proxy). `210.0` against `[120, 122, 119]` inflates both
the mean and the deviation, landing the score at 1.73 — below the 2.0 threshold. This is
a deliberate teaching moment: a single outlier inside a small window partly masks itself.

## Step 3: Calibrate the threshold (and dodge the strict-inequality trap)

```python
spike = [72.0, 74.0, 73.0, 71.0, 180.0]             # resting HR + one 180 bpm event

print(detector.compute_anomaly_score(spike))         # score sits exactly ON 2.0
print(detector.is_anomaly(spike))                    # strict '>' means NOT flagged

calibrated = MockAnomalyDetector(threshold=1.5)
print(calibrated.is_anomaly(spike))
```

Expected output:

```text
2.0
False
True
```

Read that second line again: the score equals the threshold and `is_anomaly` uses a
strict `>`, so nothing fires. Interviewers love this class of bug. In production you
would (a) know your comparator exactly, and (b) pick the threshold from a validation set
by trading precision against recall, not by folklore. Lowering to 1.5 flags the spike —
at the cost of more false positives you will route to a human review queue.

## Step 4: Recommend with embeddings and cosine similarity

```python
from interview_prep.traditional_ml import MockRecommenderSystem

catalog = {
    "insulin_pump":   [1.0, 0.2, 0.1],   # axis 0: diabetes-care affinity
    "glucose_meter":  [0.9, 0.4, 0.2],
    "bp_monitor":     [0.1, 0.1, 0.95],  # axis 2: cardiology affinity
    "ecg_patch":      [0.2, 0.9, 0.1],   # axis 1: monitoring affinity
}
reco = MockRecommenderSystem(catalog)

print(round(reco.cosine_similarity([1.0, 0.0, 0.0], [0.9, 0.4, 0.2]), 4))
print(reco.recommend([0.95, 0.3, 0.15], top_k=2))
```

Expected output:

```text
0.8955
[{'item_id': 'insulin_pump', 'score': 0.9929}, {'item_id': 'glucose_meter', 'score': 0.9926}]
```

`recommend` scores every catalog item against the query embedding and returns the top-k
sorted by cosine similarity. The query vector leans on the diabetes-care axis, so the
pump and the glucose meter win. Swap the catalog for SKU embeddings and this is the
e-commerce "customers like you also bought" service; swap it for care-plan embeddings
and it is clinical decision support. Same math, different governance regime — which is
exactly the five-industry story you will tell in interviews.

## Step 5: Bridge model outputs into an LLM prompt

The production pattern: classical ML produces **deterministic, auditable context**;
the LLM reasons over it but never invents numbers. Build the payload explicitly:

```python
import json
from interview_prep.traditional_ml import MockAnomalyDetector

def build_patient_context(telemetry: list[float], age: int, systolic_bp: int) -> str:
    detector = MockAnomalyDetector(threshold=1.5)
    return json.dumps(
        {
            "risk_flag": detector.is_anomaly(telemetry),
            "anomaly_score": detector.compute_anomaly_score(telemetry),
            "age": age,
            "systolic_bp": systolic_bp,
        },
        indent=2,
    )

context_json = build_patient_context([118.0, 121.0, 119.0, 184.0], age=67, systolic_bp=182)
print(context_json)

prompt = f"""You are a clinical decision-support assistant.
Patient telemetry context (computed by classical ML, treat as ground truth):
{context_json}
Draft a two-sentence alert for the care team.
Do not invent clinical facts that are not in the context."""
print(prompt[:150] + " ...")
```

Expected output:

```text
{
  "risk_flag": true,
  "anomaly_score": 1.73,
  "age": 67,
  "systolic_bp": 182
}
You are a clinical decision-support assistant.
Patient telemetry context (computed by classical ML, treat as ground truth):
{
  "risk_flag": true,
 ...
```

Two properties make this interview-grade: the numbers in the prompt are reproducible
(same telemetry, same JSON), and the instruction "do not invent facts" binds the
generator to the structured context. When the interviewer asks "how do you stop the LLM
from hallucinating vitals?", this payload boundary is your answer.

## Step 6: Map the pattern across the five industries

The bridge pattern is identical; only the features, threshold semantics and compliance
regime change:

| Industry | Classical model feeding context | LLM's job | Compliance lens |
|---|---|---|---|
| Healthcare / HIPAA | Vitals anomaly score (this tutorial) | Draft care-team alerts | PHI never leaves the VPC; audit every prompt |
| Fintech / PCI-DSS | Transaction fraud score (XGBoost) | Explain flagged transactions | No cardholder data in prompts |
| E-commerce | Recommender scores (this tutorial) | Personalize marketing copy | No PII in event streams |
| Logistics | ETA-risk / predictive-maintenance score | Summarize disruption plans | Telemetry residency rules |
| SaaS LLMOps | Multi-tenant usage anomaly score | Draft incident summaries | Tenant isolation in retrieval |

## What you learned

- `MockAnomalyDetector` scores deviation-from-mean in standard deviations and flags
  with a strict `>` — a score equal to the threshold does NOT fire.
- Thresholds are a precision/recall decision made on validation data, not a default.
- `MockRecommenderSystem` turns embeddings into ranked, explainable top-k output via
  cosine similarity.
- The production bridge is a deterministic JSON context payload injected into the
  prompt, with an explicit "do not invent facts" constraint.

## Exercises

### Exercise 1: Threshold sweep

Write a function that reports the detector's verdict for thresholds 1.0, 1.5, 2.0 and
2.5 on the spike sample `[72.0, 74.0, 73.0, 71.0, 180.0]`. Which thresholds fire?

<details>
<summary>Solution</summary>

```python
from interview_prep.traditional_ml import MockAnomalyDetector

spike = [72.0, 74.0, 73.0, 71.0, 180.0]
for t in (1.0, 1.5, 2.0, 2.5):
    verdict = MockAnomalyDetector(threshold=t).is_anomaly(spike)
    print(t, verdict)
```

Output:

```text
1.0 True
1.5 True
2.0 False
2.5 False
```

The score is 2.0, so every threshold strictly below 2.0 fires and every threshold at or
above it stays silent — the strict-inequality boundary in action.
</details>

### Exercise 2: E-commerce recommendation sanity check

Using the catalog from Step 4, verify that a query embedding biased toward the
cardiology axis ranks `bp_monitor` first, and print the full ranked list with scores.

<details>
<summary>Solution</summary>

```python
from interview_prep.traditional_ml import MockRecommenderSystem

catalog = {
    "insulin_pump":  [1.0, 0.2, 0.1],
    "glucose_meter": [0.9, 0.4, 0.2],
    "bp_monitor":    [0.1, 0.1, 0.95],
    "ecg_patch":     [0.2, 0.9, 0.1],
}
reco = MockRecommenderSystem(catalog)
ranked = reco.recommend([0.15, 0.2, 0.9], top_k=4)
for item in ranked:
    print(item)
```

Output:

```text
{'item_id': 'bp_monitor', 'score': 0.992}
{'item_id': 'glucose_meter', 'score': 0.4208}
{'item_id': 'ecg_patch', 'score': 0.3463}
{'item_id': 'insulin_pump', 'score': 0.2925}
```

The cardiology-leaning query flips the ranking end-to-end and the list comes back
sorted by descending score — a good demo of embedding geometry for interviewers who ask
"why not just keyword-match products?"
</details>

### Exercise 3: Fintech flavor — fraud triage

A payments service produces hourly transaction features
`[180.0, 45.0, 52.0, 61.0, 58.0]` (amounts in USD). Decide whether a fraud agent should
be woken up, using a threshold of 1.5, and print the decision payload it would forward.

<details>
<summary>Solution</summary>

```python
from interview_prep.traditional_ml import MockAnomalyDetector
import json

detector = MockAnomalyDetector(threshold=1.5)
features = [180.0, 45.0, 52.0, 61.0, 58.0]
score = detector.compute_anomaly_score(features)
payload = {"fraud_flag": detector.is_anomaly(features), "anomaly_score": score}
print(json.dumps(payload))
```

Output:

```text
{"fraud_flag": true, "anomaly_score": 1.99}
```

The $180 transaction is 1.99 standard deviations from the hour's mean, so the agent is
paged with a structured payload — the same bridge pattern as Step 5, under PCI-DSS
language ("no PAN in the payload", only aggregate features).
</details>

## Interview lens

Interviewers probing "classical ML vs GenAI" are testing whether you know where each
belongs. The winning framing: classical models own **deterministic numeric evidence**
(scores, rankings, probabilities); the LLM owns **language over that evidence**. Defend
the boundary with the three properties of Step 5's payload — deterministic, auditable,
bounded — and volunteer the failure mode (an LLM asked to compute a z-score will
eventually get it wrong; an Isolation Forest will not). If asked for a concrete example,
walk Steps 2-5 of this tutorial end to end; it takes ninety seconds and proves you have
actually built the pattern.

## Common errors

- `ModuleNotFoundError: No module named 'interview_prep'` — you are not in the repo root
  or skipped `pip install -e ".[dev]"`. Re-run Step 1 from the repository root.
- `ImportError: cannot import name 'MockRecommenderSystem'` — a typo in the class name
  or a stale `build/lib` copy shadowing `src/`. Import exactly
  `from interview_prep.traditional_ml import MockRecommenderSystem`, and re-run
  `pip install -e ".[dev]"` if a `build/` directory exists from an old packaging run.
- `AttributeError: 'list' object has no attribute ...` — you passed a dict where a
  `list[float]` is expected (or vice versa). Both classes type-hint their inputs;
  match the signatures in the contract.
- Using a system Python older than 3.10 — the code base uses `list[float]` built-in
  generics (`requires-python = ">=3.10"`). Check with `python --version`; recreate the
  venv with a newer interpreter if needed.
- `compute_anomaly_score([])` returns `0.0` by contract — an empty window is not an
  anomaly, it is missing data. In production you would alert on *no data* separately.
- `cosine_similarity` with a zero vector returns `0.0` (not an error). Guard your
  embeddings before trusting the ranking.
