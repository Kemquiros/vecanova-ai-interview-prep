"""Load and validate the interview question bank shipped as package data."""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from typing import Any

VALID_TOPICS = {"system_design", "llmops", "ml_core", "behavioral"}
VALID_DIFFICULTIES = {"mid", "senior"}
REQUIRED_FIELDS = {"id", "topic", "difficulty", "industry", "question", "rubric"}


@lru_cache(maxsize=1)
def load_questions() -> tuple[dict[str, Any], ...]:
    """Return the validated question bank (cached)."""
    raw = (
        resources.files("interview_prep")
        .joinpath("data/questions.json")
        .read_text(encoding="utf-8")
    )
    payload = json.loads(raw)
    questions = payload["questions"] if isinstance(payload, dict) else payload
    _validate(questions)
    return tuple(questions)


def filter_questions(
    topic: str | None = None, difficulty: str | None = None
) -> list[dict[str, Any]]:
    """Filter by topic and/or difficulty; None means any."""
    selected = [
        q
        for q in load_questions()
        if (topic is None or q["topic"] == topic)
        and (difficulty is None or q["difficulty"] == difficulty)
    ]
    if not selected:
        raise ValueError("no questions match the given filters")
    return selected


def _validate(questions: list[dict[str, Any]]) -> None:
    ids: set[str] = set()
    for q in questions:
        missing = REQUIRED_FIELDS - set(q)
        if missing:
            raise ValueError(f"{q.get('id', '?')} missing fields: {sorted(missing)}")
        if q["id"] in ids:
            raise ValueError(f"duplicate question id: {q['id']}")
        ids.add(q["id"])
        if q["topic"] not in VALID_TOPICS:
            raise ValueError(f"{q['id']}: invalid topic {q['topic']!r}")
        if q["difficulty"] not in VALID_DIFFICULTIES:
            raise ValueError(f"{q['id']}: invalid difficulty {q['difficulty']!r}")
        if not isinstance(q["rubric"], list) or len(q["rubric"]) < 3:
            raise ValueError(f"{q['id']}: rubric must list >= 3 key points")
