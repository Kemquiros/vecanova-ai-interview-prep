"""Distribution drift detection: PSI and the two-sample KS statistic.

Both are computed with the standard library so the math is inspectable in an
interview: PSI compares binned population shares; KS compares empirical CDFs.
"""

from __future__ import annotations

import math
from bisect import bisect_right

_EPSILON = 1e-6


def _quantile_edges(sorted_values: list[float], bins: int) -> list[float]:
    n = len(sorted_values)
    return [
        sorted_values[min(n - 1, round(q * (n - 1)))]
        for q in (i / bins for i in range(1, bins))
    ]


def _bin_percentages(values: list[float], edges: list[float]) -> list[float]:
    counts = [0] * (len(edges) + 1)
    for x in values:
        counts[bisect_right(edges, x)] += 1
    total = len(values)
    return [count / total for count in counts]


def population_stability_index(
    expected: list[float], actual: list[float], bins: int = 10
) -> float:
    """PSI between two samples; quantile bins come from `expected`.

    PSI = sum_i (a_i - e_i) * ln(a_i / e_i), clamped by _EPSILON.
    """
    if not expected or not actual:
        raise ValueError("expected and actual must be non-empty")
    if bins < 2:
        raise ValueError("bins must be >= 2")
    edges = _quantile_edges(sorted(expected), bins)
    exp_pct = _bin_percentages(expected, edges)
    act_pct = _bin_percentages(actual, edges)
    psi = 0.0
    for e, a in zip(exp_pct, act_pct):
        e = max(e, _EPSILON)
        a = max(a, _EPSILON)
        psi += (a - e) * math.log(a / e)
    return round(psi, 6)


def drift_verdict(psi: float) -> str:
    """Industry-standard PSI thresholds: <0.1 stable, <0.25 moderate."""
    if psi < 0.1:
        return "STABLE"
    if psi < 0.25:
        return "MODERATE"
    return "SEVERE"


def ks_statistic(a: list[float], b: list[float]) -> float:
    """Two-sample Kolmogorov-Smirnov statistic: max |CDF_a(x) - CDF_b(x)|."""
    if not a or not b:
        raise ValueError("both samples must be non-empty")
    xs = sorted(a)
    ys = sorted(b)
    points = sorted(set(xs) | set(ys))

    def cdf(sorted_values: list[float], x: float) -> float:
        return bisect_right(sorted_values, x) / len(sorted_values)

    return round(max(abs(cdf(xs, p) - cdf(ys, p)) for p in points), 6)
