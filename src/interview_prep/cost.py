"""LLMOps cost arithmetic: token estimates, blended pricing, projections.

Prices are always passed in by the caller (they change constantly); this
module only owns the arithmetic, so it never goes stale.
"""

from __future__ import annotations

import math


def estimate_tokens(text: str) -> int:
    """~4 characters per token (GPT-style tokenizers); never zero."""
    return max(1, math.ceil(len(text) / 4))


def estimate_cost(
    tokens_in: int,
    tokens_out: int,
    price_in_per_mtok: float,
    price_out_per_mtok: float,
) -> float:
    """Blended cost in USD: in-tokens at input price, out-tokens at output price."""
    if min(tokens_in, tokens_out, price_in_per_mtok, price_out_per_mtok) < 0:
        raise ValueError("tokens and prices must be non-negative")
    cost = tokens_in / 1_000_000 * price_in_per_mtok
    cost += tokens_out / 1_000_000 * price_out_per_mtok
    return round(cost, 6)


def monthly_projection(requests_per_day: int, cost_per_request: float) -> float:
    """30-day cost projection in USD for a steady request rate."""
    if requests_per_day < 0 or cost_per_request < 0:
        raise ValueError("request rate and cost must be non-negative")
    return round(requests_per_day * 30 * cost_per_request, 2)
