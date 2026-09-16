import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from tone.config import ScoreConfig
from tone.score import (
    Window,
    combine,
    critical_value,
    daily_scores,
    newest_sample_age,
    robust_sigma,
    score_windows,
)

TZ = timezone(timedelta(hours=-4))


def make_window(when, rmssd_target=35.0, hr=62.0, n=60, seed=0, **kw):
    """A window whose RMSSD and HR each carry their own window-to-window spread.

    Without that spread every window would land on exactly the same value, the
    residual sigma would be floating-point dust, and the z-scores would be
    meaningless -- which is a fine thing for the code to refuse, but a useless
    fixture for testing the scoring.
    """
    rng = np.random.default_rng(seed)
    hr_i = hr * math.exp(rng.normal(0, 0.04))
    rmssd_i = rmssd_target * math.exp(rng.normal(0, 0.18))
    mean_rr = 60000.0 / hr_i
    rr = mean_rr + rng.normal(0, rmssd_i / math.sqrt(2), n)
    return Window(start=when, rr=rr, **kw)


def series(days=60, per_day=5, seed=0, **kw):
    out = []
    base = datetime(2026, 5, 1, tzinfo=TZ)
    for d in range(days):
        for k in range(per_day):
            when = base + timedelta(days=d, hours=2 + k * 4.5)
            out.append(make_window(when, seed=seed + d * per_day + k, **kw))
    return out


def test_combine_signs_point_the_right_way():
    cfg = ScoreConfig()
    # Low vagal tone (z_x negative) and high HR (z_h positive) both mean stress.
    assert combine(-1.0, 1.0, cfg) == pytest.approx(1.0)
    assert combine(1.0, -1.0, cfg) == pytest.approx(-1.0)
    assert combine(0.0, 0.0, cfg) == pytest.approx(0.0)


def test_precision_weighting_favours_the_quieter_channel():
    cfg = ScoreConfig(sigma2_eps_x=0.04, sigma2_eps_h=0.001)
    w_x, w_h = cfg.weights()
    assert w_h > 20 * w_x
    # With HR far more precise, the combined score tracks the HR channel.
    assert combine(0.0, 1.0, cfg) == pytest.approx(1.0, rel=0.05)


def test_lambda_restores_the_physiological_prior():
    base = ScoreConfig(sigma2_eps_x=0.04, sigma2_eps_h=0.001)
    leaning = ScoreConfig(sigma2_eps_x=0.04, sigma2_eps_h=0.001, lambda_hrv=40.0)
    assert combine(-1.0, 0.0, leaning) > combine(-1.0, 0.0, base)


def test_unmeasured_weights_are_equal_not_the_old_heuristic():
    w_x, w_h = ScoreConfig().weights()
    assert w_x == pytest.approx(w_h)


def test_robust_sigma_ignores_one_corrupted_window():
    values = np.concatenate([np.random.default_rng(0).normal(0, 1, 200), [500.0]])
    assert robust_sigma(values) == pytest.approx(1.0, abs=0.2)
    assert np.std(values, ddof=1) > 10  # what the non-robust estimate would have done


def test_robust_sigma_falls_back_when_mad_is_zero():
    values = np.array([1.0] * 10 + [5.0])
    assert robust_sigma(values) == pytest.approx(np.std(values, ddof=1))


def test_warmup_emits_no_score():
    cfg = ScoreConfig()
    scores = score_windows(series(days=4), cfg)
    assert all(not s.scored for s in scores)
    assert all("warming_up" in s.flags for s in scores if s.metrics.usable)


def test_baseline_excludes_the_window_being_scored():
    cfg = ScoreConfig()
    scores = score_windows(series(days=40), cfg)
    scored = [s for s in scores if s.scored]
    assert scored
    # n_baseline counts strictly earlier windows only.
    for s in scored:
        assert s.n_baseline >= cfg.min_scoring_windows
    assert scored[0].n_baseline == cfg.min_scoring_windows


def test_a_calm_body_scores_near_zero_on_average():
    scores = score_windows(series(days=60, seed=100), ScoreConfig())
    values = np.array([s.s for s in scores if s.scored])
    assert values.size > 100
    assert abs(values.mean()) < 0.25


def test_a_stressed_day_scores_high():
    cfg = ScoreConfig()
    windows = series(days=60, seed=7)
    base = datetime(2026, 5, 1, tzinfo=TZ)
    # Day 60: vagal tone halved, heart rate up 12 bpm -- an unmistakable day.
    stressed = [
        make_window(base + timedelta(days=60, hours=2 + k * 4.5),
                    rmssd_target=17.0, hr=74.0, seed=900 + k)
        for k in range(5)
    ]
    scores = score_windows(windows + stressed, cfg)
    daily = daily_scores(scores, cfg)
    last = daily[-1]
    assert last.day == (base + timedelta(days=60)).date()
    assert last.mean > 1.5
    assert last.lo > 0.0  # the interval excludes "an ordinary day"


def test_daily_interval_widens_with_fewer_windows():
    cfg = ScoreConfig()
    many = daily_scores(score_windows(series(days=60, per_day=6, seed=3), cfg), cfg)
    few = daily_scores(score_windows(series(days=60, per_day=3, seed=3), cfg), cfg)
    width = lambda d: d.hi - d.lo
    assert np.median([width(d) for d in few]) > np.median([width(d) for d in many])


def test_single_window_day_is_flagged_and_still_gets_an_interval():
    cfg = ScoreConfig()
    windows = series(days=50, per_day=4, seed=11)
    lonely = make_window(datetime(2026, 7, 1, 9, tzinfo=TZ), seed=999)
    daily = daily_scores(score_windows(windows + [lonely], cfg), cfg)
    day = [d for d in daily if d.day == lonely.day]
    assert day and day[0].sd_pooled and math.isfinite(day[0].lo)


def test_critical_value_table():
    assert critical_value(4) == pytest.approx(2.776445)
    assert critical_value(1000) == pytest.approx(1.959964)
    assert critical_value(4, "normal") == pytest.approx(1.959964)
    # t is wider than the normal shortcut precisely where n_d is small.
    assert critical_value(4) > critical_value(4, "normal")


def test_unusable_windows_survive_in_the_output_with_a_reason():
    cfg = ScoreConfig()
    bad = Window(start=datetime(2026, 5, 2, 9, tzinfo=TZ), rr=np.full(5, 900.0))
    scores = score_windows(series(days=40) + [bad], cfg)
    flagged = [s for s in scores if not s.metrics.usable]
    assert len(flagged) == 1
    assert flagged[0].flags == ["too_few_intervals"]


def test_newest_sample_age():
    windows = series(days=40)
    scores = score_windows(windows, ScoreConfig())
    now = max(w.start for w in windows) + timedelta(hours=6)
    assert newest_sample_age(scores, now) == timedelta(hours=6)
    assert newest_sample_age([], now) is None


def test_scores_are_invariant_to_input_order():
    cfg = ScoreConfig()
    windows = series(days=45, seed=21)
    forward = [s.s for s in score_windows(windows, cfg) if s.scored]
    backward = [s.s for s in score_windows(list(reversed(windows)), cfg) if s.scored]
    assert np.allclose(forward, backward, equal_nan=True)
