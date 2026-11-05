# 2026-11-04 — Depth & bilingual expansion (v0.2.0)

## What happened
- Expanded all documentation from 215 skeleton lines to 3,322 lines of senior-level
  English content (theory 01-06, tutorials 01-04, STAR bank 15 Q&As, architecture with
  capacity math and failure modes), produced by 4 parallel content subagents.
- Added production-grade code with TDD: `retrieval.py` (BM25 Lucene-IDF, RRF, chunking),
  `drift.py` (PSI quantile bins, two-sample KS, verdicts), `cost.py` (token/pricing math),
  `questions.py` + packaged 36-question bank with rubrics; rubric-graded mock interviewer
  CLI. 25 tests green, ruff clean.
- Expanded the Colab notebook to 30 cells (17 code); fixed its smoke test (word-based
  chunking contract). All 17 cells execute against local modules with zero failures.
- Created the complete Spanish mirror (`docs/es/` + `README.es.md`) for LATAM remote
  candidates: 12/12 documents at 107-114% length parity, 229,831 chars total.

## Lessons
- Subagent infrastructure (NVIDIA NIM) failed 6 times with HTTP 404 across ~4h; three
  dead agents still left verified partial work. Recovery pattern: verify on-disk state
  first, re-dispatch only the missing files, and take over personally when provider
  failures repeat — the bilingual mirror was finished by direct authoring.
- Steering mid-run works: a wrong class name in agent briefs (`ProductRecommender`) was
  corrected live in two running agents; on-disk grep confirmed zero stale references.
- Agent output is a claim, not a fact: the notebook's smoke test asserted a char-based
  chunking contract that contradicted the frozen word-based API — caught by executing all
  code cells against the real modules, not by trusting the completion report.
