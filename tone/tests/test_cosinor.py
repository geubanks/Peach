import math

import numpy as np
import pytest

from tone import cosinor


def test_recovers_a_known_rhythm():
    hours = np.linspace(0, 24, 200, endpoint=False)
    mesor, amp, acro = 3.6, 0.3, 4.0
    y = mesor + amp * np.cos(2 * math.pi * (hours - acro) / 24.0)
    fit = cosinor.fit(hours, y)
    assert fit.mesor == pytest.approx(mesor, abs=1e-9)
    assert fit.amplitude == pytest.approx(amp, abs=1e-9)
    assert fit.acrophase_hours == pytest.approx(acro, abs=1e-6)
    assert fit.r2 == pytest.approx(1.0, abs=1e-12)


def test_recovers_a_rhythm_under_noise():
    rng = np.random.default_rng(3)
    hours = rng.uniform(0, 24, 500)
    y = 3.6 + 0.3 * np.cos(2 * math.pi * (hours - 4.0) / 24.0) + rng.normal(0, 0.1, 500)
    fit = cosinor.fit(hours, y)
    assert fit.amplitude == pytest.approx(0.3, abs=0.03)
    assert fit.acrophase_hours == pytest.approx(4.0, abs=0.7)


def test_matches_numpy_least_squares():
    # The explicit 3x3 solve exists so Swift can copy it; it must still agree
    # with a real least-squares routine.
    rng = np.random.default_rng(11)
    hours = rng.uniform(0, 24, 120)
    y = rng.normal(0, 1, 120)
    fit = cosinor.fit(hours, y)
    x = cosinor.design_matrix(hours)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    assert np.allclose([fit.mesor, fit.a, fit.b], beta, atol=1e-9)


def test_falls_back_to_flat_when_all_samples_share_one_hour():
    hours = np.full(50, 3.0)
    y = np.linspace(1, 2, 50)
    fit = cosinor.fit(hours, y)
    assert fit.flat
    assert fit.mesor == pytest.approx(y.mean())
    assert np.allclose(fit.predict(np.array([0.0, 12.0])), y.mean())


def test_falls_back_when_too_few_distinct_hours():
    hours = np.array([1.0, 1.5, 2.0, 2.5] * 10)
    y = np.random.default_rng(0).normal(size=40)
    assert cosinor.fit(hours, y, min_distinct_hours=5).flat
    assert not cosinor.fit(hours, y, min_distinct_hours=0).flat


def test_flat_fit_on_empty_series():
    fit = cosinor.flat_fit(np.array([]))
    assert fit.flat and fit.n == 0 and math.isnan(fit.mesor)


def test_r2_is_zero_for_pure_noise_against_a_flat_truth():
    rng = np.random.default_rng(5)
    hours = rng.uniform(0, 24, 2000)
    y = rng.normal(0, 1, 2000)
    # Three parameters on 2000 points: R^2 should be near zero, not near one.
    assert cosinor.variance_explained(hours, y) < 0.02


def test_solve_3x3_reports_singular_systems():
    singular = np.zeros((3, 3))
    assert cosinor.solve_3x3(singular, np.ones(3)) is None
