import math
from dataclasses import replace

import numpy as np
import pytest

from tone import diary as diary_mod
from tone import sensitivity
from tone.config import ScoreConfig
from tone.diary import DiaryEntry
from tone.score import daily_scores, score_windows
from tone.simulate import simulate


@pytest.fixture(scope="module")
def dataset():
    windows, entries, _ = simulate(days=80, seed=31)
    return windows, entries


def test_lambda_curve_covers_the_grid(dataset):
    windows, entries = dataset
    points = sensitivity.lambda_curve(windows, entries, ScoreConfig())
    assert len(points) == len(sensitivity.DEFAULT_LAMBDAS)
    assert [p.value for p in points] == list(sensitivity.DEFAULT_LAMBDAS)


def test_recombining_for_lambda_is_exact_not_an_approximation(dataset):
    # lambda_curve avoids rescoring by re-combining the two channels, on the
    # grounds that lambda enters only at Step 5. If that were ever untrue the
    # sweep would be quietly wrong, so it is checked against a full rescore.
    windows, entries = dataset
    cfg = ScoreConfig()
    ratings = diary_mod.daily_means(entries)

    for lam in (0.25, 1.0, 8.0):
        fast = sensitivity.lambda_curve(windows, entries, cfg, lambdas=(lam,))[0]
        tuned = replace(cfg, lambda_hrv=lam)
        daily = daily_scores(score_windows(windows, tuned), tuned)
        paired = [(d.mean, ratings[d.day]) for d in daily if d.day in ratings]
        slow = diary_mod.spearman([p[0] for p in paired], [p[1] for p in paired])
        assert fast.rho == pytest.approx(slow, abs=1e-12)
        assert fast.n == len(paired)


def test_sweep_actually_varies_the_parameter(dataset):
    windows, entries = dataset
    points = sensitivity.sweep(windows, entries, ScoreConfig(),
                               param="baseline_days", values=(14.0, 28.0, 56.0))
    assert [p.value for p in points] == [14.0, 28.0, 56.0]
    # A longer baseline is a better-estimated baseline, so rho should not get
    # worse as it grows on data with a stable rhythm.
    assert points[-1].rho >= points[0].rho


def test_sweep_rejects_an_unknown_parameter(dataset):
    windows, entries = dataset
    with pytest.raises(TypeError):
        sensitivity.sweep(windows, entries, ScoreConfig(), param="not_a_field", values=(1,))


def test_tuning_picks_from_the_grid_and_maximises_in_sample(dataset):
    windows, entries = dataset
    grid = (0.5, 1.0, 4.0, 16.0)
    tuning = sensitivity.tune_lambda(windows, entries, ScoreConfig(), lambdas=grid)
    assert tuning.best_lambda in grid
    curve = sensitivity.lambda_curve(windows, entries, ScoreConfig(), lambdas=grid)
    assert tuning.rho_in_sample == pytest.approx(max(p.rho for p in curve), abs=1e-9)


def test_cross_validation_is_reported_alongside_the_in_sample_number(dataset):
    windows, entries = dataset
    tuning = sensitivity.tune_lambda(windows, entries, ScoreConfig())
    assert math.isfinite(tuning.rho_cross_validated)
    assert tuning.n_days > 10
    assert "do NOT report this" in tuning.report()


def test_tuning_on_a_null_relationship_does_not_manufacture_a_signal(dataset):
    # Shuffle the ratings across days so nothing is left to find. Tuning lambda
    # will still pick whichever value looks best; the cross-validated number is
    # what must stay near zero.
    windows, entries = dataset
    rng = np.random.default_rng(5)
    shuffled = [DiaryEntry(e.when, float(r))
                for e, r in zip(entries, rng.permutation([e.rating for e in entries]))]
    tuning = sensitivity.tune_lambda(windows, shuffled, ScoreConfig())
    assert abs(tuning.rho_cross_validated) < 0.35
    # And the in-sample number is never the smaller of the two by construction.
    assert tuning.rho_in_sample >= tuning.rho_cross_validated - 1e-9


def test_tuning_degrades_gracefully_without_enough_days():
    windows, entries, _ = simulate(days=8, seed=2)
    tuning = sensitivity.tune_lambda(windows, entries, ScoreConfig())
    assert math.isnan(tuning.rho_in_sample)
    assert math.isnan(tuning.rho_cross_validated)


def test_report_announces_a_verdict_that_moves():
    stable = [sensitivity.Point("k", v, 0.5, 0.2, 0.7, 40, "real") for v in (1, 2)]
    assert "stable" in sensitivity.report(stable, knob="k")

    moving = [sensitivity.Point("k", 1, 0.5, 0.2, 0.7, 40, "real"),
              sensitivity.Point("k", 2, 0.1, -0.2, 0.4, 40, "underpowered")]
    text = sensitivity.report(moving, knob="k")
    assert "WARNING" in text and "CHANGES" in text


def test_report_handles_a_column_of_unestimable_points():
    nan = float("nan")
    points = [sensitivity.Point("k", 1, nan, nan, nan, 0, "underpowered")]
    assert "n/a" in sensitivity.report(points, knob="k")
