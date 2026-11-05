# Tutorial 3: Healthcare Guardrails — PII Sanitization and Drift Detection

> Who this is for: engineers preparing for Senior AI/ML interviews in regulated
> industries (Healthcare/HIPAA, Fintech/PCI-DSS). You will build a PII guardrail layer,
> probe its blind spots honestly, gate a RAG answer behind it, and detect statistical
> drift in a simulated feature distribution before it silently degrades the model.
> Time budget: 70 minutes.

## What you will build

1. A PII sanitization pass over realistic clinical notes using `sanitize_pii`.
2. A documented list of its blind spots (the honesty interviewers reward).
3. A guarded RAG answer gate: sanitize -> evaluate -> block or release.
4. A drift monitor over a simulated lab-feature distribution using PSI and the KS
   statistic, wired to STABLE / MODERATE / SEVERE verdicts and concrete actions.

## Step 1: Set up the environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
python -c "from interview_prep.evaluator import sanitize_pii; from interview_prep.drift import population_stability_index; print('imports ok')"
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

## Step 2: Sanitize a clinical note

```python
from interview_prep.evaluator import sanitize_pii

note = "Patient John Doe (john.doe@hospital.org), phone +1 305 555 0142, SSN 545-12-8890."
print(sanitize_pii(note))

short_note = "Patient John Doe (john.doe@hospital.org)"
print(sanitize_pii(short_note))
```

Expected output:

```text
Patient John Doe ([EMAIL_REDACTED]), phone +[PHONE_REDACTED], SSN [SSN_REDACTED].
Patient John Doe ([EMAIL_REDACTED])
```

Emails, phone numbers with a country code, and SSNs are rewritten to stable,
deterministic placeholders. Determinism matters: the same input always yields the same
mask, so logs, evaluations and prompts stay comparable across runs. Note the cosmetic
quirk in the output — the `+1` country code sits *outside* the phone match and survives
as `+[PHONE_REDACTED]`. Harmless here, but it tells you the regex expects an
international format.

## Step 3: Probe the blind spots (do not skip this)

A guardrail you have not tried to break is a claim, not a control. Attack your own
sanitizer:

```python
from interview_prep.evaluator import sanitize_pii

attempts = [
    "Call Maria at 305-555-0142 or maria.lopez@clinic.org about claim 2024-09-77.",
    "MRN 00432157, DOB 04/12/1958, room 7W-214.",
]
for attempt in attempts:
    print(sanitize_pii(attempt))
```

Expected output:

```text
Call Maria at 305-555-0142 or [EMAIL_REDACTED] about claim 2024-09-77.
MRN 00432157, DOB 04/12/1958, room 7W-214.
```

Both rows leak PHI: the bare ten-digit phone (the mask expects a country-code prefix),
and the MRN, DOB and room number (no rules exist for them). This is exactly the HIPAA
Safe Harbor conversation: a regex layer is *one control*, not the whole program.
Production stacks it with entity-recognizer tools (Presidio), format-level redaction at
ingestion, and deny-by-default field policies. Saying "sanitize_pii alone makes us
HIPAA compliant" is an interview red flag; saying "it is the deterministic first layer
of a defense-in-depth stack" is the senior answer.

## Step 4: Gate the RAG answer behind the sanitizer

Sanitization belongs *before* the payload reaches any external model, and the same gate
should re-check the generated response. Wire sanitizer plus evaluator into one gate:

```python
from interview_prep.evaluator import sanitize_pii, evaluate_rag_response

def guarded_answer(query: str, response: str, context: list[str]) -> dict:
    safe_context = [sanitize_pii(c) for c in context]
    safe_response = sanitize_pii(response)
    pii_was_present = safe_response != response or any(
        safe_c != c for safe_c, c in zip(safe_context, context)
    )
    report = evaluate_rag_response(query, safe_response, safe_context)
    return {
        "blocked": report["status"] == "FAILED",
        "pii_was_present": pii_was_present,
        "report": report,
    }

query = "What is the recommended dosage of metformin for adults?"
context = [
    "The recommended starting dosage of metformin for adults is 500 mg twice daily. Contact Dr. Alice Kim (alice.kim@hospital.org) for exceptions.",
]
response = "Start metformin at 500 mg twice daily; contact alice.kim@hospital.org for exceptions."

print(guarded_answer(query, response, context))
```

Expected output:

```text
{'blocked': False, 'pii_was_present': True, 'report': {'faithfulness': 0.95, 'answer_relevance': 0.5, 'status': 'PASSED', 'reason': 'Metrics computed successfully.'}}
```

Read the payload as an auditor would: both the retrieved context and the drafted
response contained a physician's email, the gate masked both deterministically before
evaluation (`pii_was_present: True`), and the answer stayed grounded enough to release
(`PASSED`). If retrieval ever returns an empty context, `evaluate_rag_response` returns
`status="FAILED"` and `blocked` flips to `True` — the gate fails closed, not open.

## Step 5: Simulate a feature distribution and detect drift

Guardrails protect the prompt; drift detection protects the *model*. Simulate a lab
feature (e.g., fasting glucose, mg/dL) as seen in training versus this week's traffic:

```python
import random
from interview_prep.drift import population_stability_index, ks_statistic, drift_verdict

random.seed(42)  # reproducibility: same numbers every run
expected      = [random.gauss(120.0, 12.0) for _ in range(5000)]   # training distribution
actual_stable = [random.gauss(120.5, 12.2) for _ in range(5000)]   # this week: healthy
actual_shift  = [random.gauss(132.0, 12.0) for _ in range(5000)]   # upstream unit change

psi_stable = population_stability_index(expected, actual_stable)
psi_shift  = population_stability_index(expected, actual_shift)
print(psi_stable, drift_verdict(psi_stable))
print(psi_shift, drift_verdict(psi_shift))
print(ks_statistic(expected, actual_stable))
print(ks_statistic(expected, actual_shift))
```

Expected output (PSI uses quantile bins taken from the expected sample, with empty
buckets floored at 1e-6, so the decimals are reproducible run to run):

```text
0.003421 STABLE
0.971624 SEVERE
0.022
0.3914
```

Population Stability Index buckets both distributions and accumulates
`(actual% - expected%) * ln(actual% / expected%)`; under 0.10 is conventionally stable,
0.10-0.25 moderate, above 0.25 severe — which is exactly what `drift_verdict` encodes.
The stable sample scores 0.003421 (indistinguishable from training data) while a
one-standard-deviation mean shift (120 -> 132 mg/dL, e.g., a lab analyzer recalibration)
scores 0.971624, deep in SEVERE. The KS statistic (largest gap between the two empirical
CDFs) corroborates: 0.022 for stable traffic, 0.3914 for the shifted one.

## Step 6: Wire verdicts to actions

A verdict nobody acts on is telemetry theater. Bind each verdict to an operational
response:

```python
from interview_prep.drift import population_stability_index, drift_verdict

VERDICT_ACTIONS = {
    "STABLE":   "log and continue",
    "MODERATE": "page on-call analyst; sample responses for manual review",
    "SEVERE":   "freeze model deploys; route traffic to fallback; open incident",
}

def drift_action(expected: list[float], actual: list[float]) -> str:
    psi = population_stability_index(expected, actual)
    verdict = drift_verdict(psi)
    return f"{verdict}: {VERDICT_ACTIONS[verdict]}"

print(drift_action(expected, actual_stable))
print(drift_action(expected, actual_shift))
```

Expected output:

```text
STABLE: log and continue
SEVERE: freeze model deploys; route traffic to fallback; open incident
```

In production the SEVERE branch posts to your incident channel and flips a feature flag
that re-routes the anomaly detector to the previous model version — the rollback path
is defined *before* the drift arrives, never improvised during it.

## What you learned

- `sanitize_pii` deterministically masks emails, international phones and SSNs — and
  provably misses bare ten-digit phones and MRN/DOB fields, so it is layer one of a
  defense-in-depth stack, not a compliance certificate.
- The guardrail gate sanitizes context and response *before* evaluation and fails
  closed when retrieval returns nothing.
- `population_stability_index` plus `drift_verdict` convert distribution shift into an
  operational verdict (STABLE / MODERATE / SEVERE); `ks_statistic` is an independent
  second opinion.
- Verdicts map to pre-agreed actions: log, page, freeze-and-rollback.

## Exercises

### Exercise 1: Harden the sanitizer

Write `sanitize_bare_phone(text)` that also masks bare ten-digit phones like
`305-555-0142`, and verify it on the leaking example from Step 3.

<details>
<summary>Solution</summary>

```python
import re
from interview_prep.evaluator import sanitize_pii

def sanitize_bare_phone(text: str) -> str:
    text = sanitize_pii(text)
    return re.sub(r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b", "[PHONE_REDACTED]", text)

leaky = "Call Maria at 305-555-0142 or maria.lopez@clinic.org about claim 2024-09-77."
print(sanitize_bare_phone(leaky))
```

Output:

```text
Call Maria at [PHONE_REDACTED] or [EMAIL_REDACTED] about claim 2024-09-77.
```

Layering a stricter pattern *after* the standard pass is the right production shape:
wrapper functions like this are how you extend a shared sanitizer without editing code
other teams depend on. (Extending the exercise: wrap MRNs `\bMRN\s?\d{8}\b` and dates
the same way.)
</details>

### Exercise 2: How big a shift before the alarm?

Sweep the distribution shift from 0.0 to 1.25 standard deviations in 0.25 steps and
print PSI and verdict for each. Where do the MODERATE and SEVERE thresholds bite?

<details>
<summary>Solution</summary>

```python
import random
from interview_prep.drift import population_stability_index, drift_verdict

random.seed(42)
expected = [random.gauss(120.0, 12.0) for _ in range(5000)]
for shift_sigma in (0.0, 0.25, 0.5, 0.75, 1.0, 1.25):
    actual = [random.gauss(120.0 + shift_sigma * 12.0, 12.0) for _ in range(5000)]
    psi = population_stability_index(expected, actual)
    print(f"shift={shift_sigma:.2f} sigma  psi~{psi:.4f}  {drift_verdict(psi)}")
```

Expected pattern (verdicts are the robust property; decimals shift with sample size):

```text
shift=0.00 sigma  psi~0.0032  STABLE
shift=0.25 sigma  psi~0.0721  STABLE
shift=0.50 sigma  psi~0.2404  MODERATE
shift=0.75 sigma  psi~0.6021  SEVERE
shift=1.00 sigma  psi~0.9063  SEVERE
shift=1.25 sigma  psi~1.4151  SEVERE
```

MODERATE opens right around half a sigma for this feature (0.2404, a hair under the
0.25 boundary) and SEVERE fires by three-quarters. Two details worth saying aloud in an
interview: even the zero-shift row is not exactly zero (two finite samples of the same
distribution still differ slightly), and PSI grows super-linearly with shift, so a
"severe" reading tells you the *shape* of the response you need (freeze and
investigate), while the sweep tells you *how early* your alarm fires.
</details>

### Exercise 3: Combine both controls into a release checklist

Write `release_check(context, response, expected_dist, actual_dist)` that returns a
single GO / NO-GO decision requiring both the PII gate to pass and drift to be STABLE.

<details>
<summary>Solution</summary>

```python
from interview_prep.evaluator import sanitize_pii, evaluate_rag_response
from interview_prep.drift import population_stability_index, drift_verdict
import random

random.seed(42)
expected = [random.gauss(120.0, 12.0) for _ in range(5000)]
actual_shift = [random.gauss(132.0, 12.0) for _ in range(5000)]

def release_check(query, context, response, expected_dist, actual_dist) -> dict:
    report = evaluate_rag_response(query, sanitize_pii(response),
                                   [sanitize_pii(c) for c in context])
    psi = population_stability_index(expected_dist, actual_dist)
    verdict = drift_verdict(psi)
    go = report["status"] == "PASSED" and verdict == "STABLE"
    return {"decision": "GO" if go else "NO-GO",
            "eval_status": report["status"], "drift": verdict, "psi": psi}

q = "What is the recommended dosage of metformin for adults?"
ctx = ["The recommended starting dosage of metformin for adults is 500 mg twice daily with meals."]
good = "Start metformin at 500 mg twice daily with meals."
print(release_check(q, ctx, good, expected, expected))
print(release_check(q, ctx, good, expected, actual_shift))
```

Output:

```text
{'decision': 'GO', 'eval_status': 'PASSED', 'drift': 'STABLE', 'psi': 0.0}
{'decision': 'NO-GO', 'eval_status': 'PASSED', 'drift': 'SEVERE', 'psi': 0.897629}
```

The first run passes both controls (quality `PASSED`, drift `STABLE`) -> `GO`. The
second run fails on drift alone (`SEVERE`) -> `NO-GO`, even though the answer quality
is unchanged — the correct behavior, because a drifted feature distribution invalidates
tomorrow's answers, not today's. (The psi here is 0.897629 rather than Step 5's
0.971624 because this shift sample was drawn after a shorter random stream; PSI is a
sample statistic, another reason to fix your seeds in tests.) Two independent controls
combined with a logical AND is precisely how you should describe release governance in
a regulated-industry interview.
</details>

## Interview lens

Regulated-industry interviews test whether you treat compliance as an engineering
property or a slide. Three probes and your answers from this lab: (1) "How do you keep
PHI out of the model?" — the gate in Step 4, plus the Step 3 admission that regexes
miss MRNs and bare phones, plus named stack-mates (Presidio, ingestion-time redaction).
(2) "A model's metrics quietly dropped last quarter — what happened?" — Step 5's story:
nobody touched the model, the *input distribution* moved one sigma; PSI caught it,
KS corroborated, verdict drove a freeze-and-rollback. (3) "Who owns the threshold?" —
the 0.10/0.25 PSI conventions are industry defaults, but you calibrate them on
validation data and re-review quarterly, because a threshold is a business decision
encoded in code. Offering the failure mode before being asked — "sanitizers are
attack surfaces; we red-team ours" — is what separates senior from mid answers.

## Common errors

- `ModuleNotFoundError: No module named 'interview_prep.drift'` — stale checkout; pull
  the latest branch and re-run `pip install -e ".[dev]"`.
- `ModuleNotFoundError: No module named 'interview_prep'` — running from outside the
  repository or skipping the editable install; redo Step 1.
- Different PSI numbers on every run — you skipped `random.seed(42)`; simulations must
  be reproducible or your drift tests are not regression-testable.
- Expecting `305-555-0142` to be masked — the phone rule requires a country-code
  prefix; see Step 3 and Exercise 1.
- Expecting MRN/DOB/room numbers to be masked — there are no rules for them; that is
  the documented blind spot, not a bug to file.
- `ZeroDivisionError` in hand-rolled PSI — you must floor bucket shares with a small
  epsilon before taking logarithms; the contract implementation handles empty buckets.
- Tiny PSI on obviously shifted data — you compared only a few hundred samples; PSI
  needs enough observations per bucket (hundreds to thousands) to be meaningful.
- Colab: `%pip install -e .` is session-scoped — after a runtime restart, re-run the
  setup cell before importing.
