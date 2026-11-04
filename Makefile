.PHONY: install test lint run mock

install:
	pip install -e .[dev]

test:
	pytest tests/ -v --cov=src/interview_prep

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

run:
	python scripts/run_mock_assessment.py

mock:
	python scripts/interactive_interviewer.py --topic system_design
