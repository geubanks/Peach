import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from tone import calibrate as calib
from tone.config import ScoreConfig
from tone.diary import (
    DiaryEntry,
    bootstrap_rho,
    classify,
    daily_means,
    rankdata,
    read_diary,
    spearman,
    validate_daily,
)
from tone.score import daily_scores, score_windows
from tone.simulate import simulate

TZ = timezone(timedelta(hours=-4))


def test_rankdata_averages_ties():
    assert list(rankdata([10, 20, 20, 30])) == [1.0, 2.5, 2.5, 4.0]


def test_spearman_is_one_for_a_monotone_but_nonlinear_relation():
    x = np.arange(1, 21, dtype=float)
    assert spearman(x, np.exp(x / 4)) == pytest.approx(1.0)
    assert spearman(x, -x) == pytest.approx(-1.0)


def test_spearman_is_nan_when_one_side_never_varies():
    assert math.isnan(spearman(np.arange(10.0), np.ones(10)))


def test_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(4)
    x = rng.normal(size=60)
    y = x + rng.normal(0, 0.5, 60)
    rho, (lo, hi) = bootstrap_rho(x, y, n_boot=2000)
    assert lo < rho < hi
    assert lo > 0  # a strong relation should exclude zero


def test_classify_decision_box():
    assert classify(0.5, (0.2, 0.7), 30)[0] == "real"
    assert classify(0.2, (-0.05, 0.4), 30)[0] == "rebuild"     # interval spans zero
    assert classify(0.2, (0.11, 0.33), 30)[0] == "sample-starved"
    assert classify(-0.5, (-0.7, -0.3), 30)[0] == "rebuild"
    assert "NEGATIVE" in classify(-0.5, (-0.7, -0.3), 30)[1]
    assert classify(float("nan"), (float("nan"), float("nan")), 0)[0] == "rebuild"


def test_read_diary_handles_headers_and_bad_rows(tmp_path):
    path = tmp_path / "diary.csv"
    path.write_text(
        "timestamp,stress\n"
        "2026-05-01 09:00:00,3\n"
        "2026-05-01 21:30:00,7\n"
        "oops,not-a-number\n"
        "2026-05-02T09:00:00,5\n"
    )
    entries = read_diary(path)
    assert [e.rating for e in entries] == [3.0, 7.0, 5.0]
    assert daily_means(entries)[entries[0].day] == pytest.approx(5.0)


def test_read_diary_accepts_a_headerless_file(tmp_path):
    path = tmp_path / "d.csv"
    path.write_text("2026-05-01 09:00,4\n2026-05-02 09:00,6\n")
    assert len(read_diary(path)) == 2


def test_calibration_recovers_injected_noise():
    # Two readings of the same true value, separated only by known noise.
    rng = np.random.default_rng(2)
    sigma = 0.2
    truth = rng.normal(3.6, 0.4, 400)
    a = truth + rng.normal(0, sigma, 400)
    b = truth + rng.normal(0, sigma, 400)
    assert calib.sigma2_from_pairs(a, b) == pytest.approx(sigma**2, rel=0.12)


def test_find_pairs_is_greedy_and_non_overlapping():
    windows, _, _ = simulate(days=90, seed=5)
    scores = score_windows(windows, ScoreConfig())
    pairs = calib.find_pairs(scores)
    assert pairs
    used = [id(s) for pair in pairs for s in pair]
    assert len(used) == len(set(used))
    for a, b in pairs:
        gap = (b.start - a.start).total_seconds() / 60.0
        assert 1.0 <= gap <= 15.0
        assert a.on_demand and b.on_demand


def test_calibration_finds_hr_the_more_reliable_channel():
    # The validation literature says HR is several times more precise on this
    # hardware; the simulator is built that way and the estimator should see it.
    windows, _, truth = simulate(days=150, seed=9)
    scores = score_windows(windows, ScoreConfig())
    result = calib.calibrate(calib.find_pairs(scores), n_boot=1000)
    assert result.sigma2_x > result.sigma2_h
    assert result.precision_ratio > 3.0
    w_x, w_h = result.weights()
    assert w_h > w_x
    assert "heart rate more reliably" in result.report()


def test_calibration_needs_pairs():
    with pytest.raises(ValueError):
        calib.calibrate([])


def test_end_to_end_the_score_tracks_a_simulated_diary():
    windows, entries, _ = simulate(days=150, seed=13)
    cfg = ScoreConfig()
    scores = score_windows(windows, cfg)
    result = calib.calibrate(calib.find_pairs(scores), n_boot=500)
    cfg = cfg.with_weights(result.sigma2_x, result.sigma2_h)
    daily = daily_scores(score_windows(windows, cfg), cfg)
    verdict = validate_daily(daily, entries, n_boot=2000)
    assert verdict.rho > 0.3
    assert verdict.ci[0] > 0
    assert verdict.verdict == "real"


def test_validate_daily_needs_overlapping_days():
    windows, _, _ = simulate(days=60, seed=1)
    cfg = ScoreConfig()
    daily = daily_scores(score_windows(windows, cfg), cfg)
    far_away = [DiaryEntry(datetime(2030, 1, 1, 9, tzinfo=TZ), 5.0)]
    assert validate_daily(daily, far_away).verdict == "rebuild"
