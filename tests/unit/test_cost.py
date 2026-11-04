"""Tests for LLMOps cost estimation helpers."""

import pytest

from interview_prep.cost import (
    estimate_cost,
    estimate_tokens,
    monthly_projection,
)


def test_estimate_tokens_uses_four_chars_per_token():
    assert estimate_tokens("a" * 40) == 10
    assert estimate_tokens("") == 1  # never zero


def test_estimate_cost_blends_input_and_output_pricing():
    cost = estimate_cost(
        tokens_in=1_000_000,
        tokens_out=500_000,
        price_in_per_mtok=3.0,
        price_out_per_mtok=15.0,
    )
    assert cost == pytest.approx(3.0 + 7.5)


def test_monthly_projection_thirty_days():
    assert monthly_projection(
        requests_per_day=100, cost_per_request=0.01
    ) == pytest.approx(30.0)
