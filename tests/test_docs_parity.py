"""Bilingual parity gate: every English doc must have a Spanish mirror.

Graceful by design: skipped until docs/es/ exists (translations land in a
later phase), then strict so CI fails if the mirror drifts.
"""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EN_DOC_FILES = [
    "docs/theory/01-advanced-machine-learning-foundations.md",
    "docs/theory/02-generative-ai-and-rag-architecture.md",
    "docs/theory/03-llmops-evaluations-and-observability.md",
    "docs/theory/04-healthcare-ai-security-and-guardrails.md",
    "docs/theory/05-system-design-and-behavioral-star.md",
    "docs/theory/06-multi-industry-ai-architectures.md",
    "docs/tutorials/01-from-zero-to-hero-ml-genai.md",
    "docs/tutorials/02-llmops-evaluations-ragas.md",
    "docs/tutorials/03-healthcare-guardrails-pii.md",
    "docs/tutorials/04-system-design-star-mastery.md",
    "docs/star_bank.md",
    "docs/architecture.md",
]


def test_spanish_mirror_is_complete_and_substantive():
    if not (ROOT / "docs" / "es").exists():
        pytest.skip("Spanish mirror not started yet")
    for rel in EN_DOC_FILES:
        en_path = ROOT / rel
        es_path = ROOT / "docs" / "es" / rel.removeprefix("docs/")
        assert es_path.exists(), f"missing Spanish mirror: {es_path}"
        en_len = len(en_path.read_text(encoding="utf-8"))
        es_len = len(es_path.read_text(encoding="utf-8"))
        assert es_len >= en_len * 0.5, (
            f"Spanish mirror too shallow for {rel}: {es_len} vs {en_len} chars"
        )


def test_readme_has_language_switcher():
    if not (ROOT / "README.es.md").exists():
        pytest.skip("Spanish README not started yet")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "README.es.md" in readme, "English README must link the Spanish version"
    readme_es = (ROOT / "README.es.md").read_text(encoding="utf-8")
    assert "README.md" in readme_es, "Spanish README must link the English version"


def test_all_relative_readme_links_resolve():
    for readme_name in ("README.md", "README.es.md"):
        readme = ROOT / readme_name
        if not readme.exists():
            continue
        text = readme.read_text(encoding="utf-8")
        for token in text.split("](")[1:]:
            link = token.split(")")[0]
            if link.startswith(("http", "#", "mailto:")):
                continue
            assert (ROOT / link).exists(), f"broken link in {readme_name}: {link}"
