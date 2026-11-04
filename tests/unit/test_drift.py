"""Tests for PSI, KS statistic, and drift verdict thresholds."""

import random

from interview_prep.drift import (
    drift_verdict,
    ks_statistic,
    population_stability_index,
)


def test_psi_identical_distribution_near_zero():
    random.seed(42)
    expected = [random.gauss(0, 1) for _ in range(5000)]
    actual = [random.gauss(0, 1) for _ in range(5000)]
    assert population_stability_index(expected, actual) < 0.1


def test_psi_detects_severe_shift():
    random.seed(42)
    expected = [random.gauss(0, 1) for _ in range(5000)]
    actual = [random.gauss(3, 1) for _ in range(5000)]
    psi = population_stability_index(expected, actual)
    assert psi > 0.25
    assert drift_verdict(psi) == "SEVERE"


def test_drift_verdict_thresholds():
    assert drift_verdict(0.05) == "STABLE"
    assert drift_verdict(0.15) == "MODERATE"
    assert drift_verdict(0.40) == "SEVERE"


def test_ks_identical_samples_zero():
    random.seed(7)
    sample = [random.gauss(0, 1) for _ in range(1000)]
    assert ks_statistic(sample, list(sample)) == 0.0


def test_ks_detects_shift():
    random.seed(7)
    a = [random.gauss(0, 1) for _ in range(1000)]
    b = [random.gauss(2, 1) for _ in range(1000)]
    assert ks_statistic(a, b) > 0.5
