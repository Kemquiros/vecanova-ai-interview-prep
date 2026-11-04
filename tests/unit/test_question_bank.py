"""Contract tests for the interview question bank."""

from interview_prep.questions import load_questions

VALID_TOPICS = {"system_design", "llmops", "ml_core", "behavioral"}
VALID_DIFFICULTIES = {"mid", "senior"}
REQUIRED_FIELDS = {"id", "topic", "difficulty", "industry", "question", "rubric"}


def test_bank_loads_from_package_data():
    questions = load_questions()
    assert len(questions) >= 30


def test_every_question_has_unique_id_and_required_fields():
    questions = load_questions()
    ids = [q["id"] for q in questions]
    assert len(ids) == len(set(ids))
    for q in questions:
        missing = REQUIRED_FIELDS - set(q)
        assert not missing, f"{q.get('id')} missing fields: {missing}"
        assert q["topic"] in VALID_TOPICS
        assert q["difficulty"] in VALID_DIFFICULTIES
        assert isinstance(q["rubric"], list) and len(q["rubric"]) >= 3


def test_every_topic_has_at_least_five_questions():
    questions = load_questions()
    by_topic = {}
    for q in questions:
        by_topic.setdefault(q["topic"], []).append(q)
    for topic in VALID_TOPICS:
        assert len(by_topic.get(topic, [])) >= 5, f"topic {topic} too thin"
