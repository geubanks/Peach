import math

import numpy as np
import pytest

from tone.config import ScoreConfig
from tone.metrics import (
    correct_rmssd_for_quantization,
    filter_rr,
    mean_hr,
    pnn50,
    quantization_variance,
    rmssd,
    rr_from_beat_times,
    rr_from_bpm,
    sdnn,
    window_metrics,
)


def test_rmssd_matches_hand_computation():
    rr = np.array([800.0, 820.0, 790.0, 810.0])
    d = np.diff(rr)
    assert rmssd(rr) == pytest.approx(math.sqrt(np.mean(d**2)))


def test_rmssd_of_constant_series_is_zero():
    assert rmssd(np.full(20, 900.0)) == pytest.approx(0.0)


def test_mean_hr_is_reciprocal_of_mean_interval_not_mean_of_rates():
    # Step 2 says HR = 60000 / mean(RR). The mean of instantaneous rates is
    # always larger (Jensen), and mixing the two up is a silent parity bug.
    rr = np.array([600.0, 1200.0])
    assert mean_hr(rr) == pytest.approx(60000.0 / 900.0)
    assert np.mean(60000.0 / rr) > mean_hr(rr)


def test_filter_keeps_a_clean_series():
    rr = 800.0 + 20.0 * np.sin(np.linspace(0, 6, 60))
    assert filter_rr(rr).all()


def test_filter_drops_an_isolated_ectopic_pair():
    rr = np.full(30, 900.0)
    rr[10] = 450.0   # premature beat
    rr[11] = 1350.0  # compensatory pause
    mask = filter_rr(rr, 0.20)
    assert not mask[10] and not mask[11]
    assert mask.sum() == 28


def test_filter_does_not_cascade_after_a_rejection():
    # The single-left-neighbour reading of the rule would reject everything
    # after the artifact. The two-sided rule must not.
    rr = np.full(40, 900.0)
    rr[5] = 500.0
    mask = filter_rr(rr, 0.20)
    assert mask[6:].all()


def test_filter_tolerates_gradual_drift():
    rr = np.linspace(900.0, 700.0, 50)  # 0.45% per beat: physiological, not artifact
    assert filter_rr(rr, 0.20).all()


def test_rmssd_does_not_bridge_a_dropped_interval():
    rr = np.full(30, 900.0)
    rr[15] = 450.0
    rr[16] = 1350.0
    clean = rmssd(rr, filter_rr(rr))
    # Every surviving pair is identical, so RMSSD is exactly zero. Bridging the
    # gap would have manufactured a difference.
    assert clean == pytest.approx(0.0)


def test_sdnn_and_pnn50():
    rr = np.array([800.0, 900.0, 1000.0, 900.0, 800.0])
    assert sdnn(rr) == pytest.approx(np.std(rr, ddof=1))
    assert pnn50(rr) == pytest.approx(1.0)


def test_rr_from_bpm_and_beat_times_disagree_by_one_on_purpose():
    bpm = [60, 60, 60, 60]
    assert rr_from_bpm(bpm).size == 4
    beats = [0.0, 1.0, 2.0, 3.0]
    assert rr_from_beat_times(beats).size == 3
    assert np.allclose(rr_from_beat_times(beats), 1000.0)


def test_beat_times_must_increase():
    with pytest.raises(ValueError):
        rr_from_beat_times([0.0, 1.0, 0.5])


def test_quantization_variance_matches_the_uniform_bin():
    rr = np.full(50, 1000.0)  # 60 bpm
    width = 1000.0**2 / 60000.0
    assert quantization_variance(rr) == pytest.approx(width**2 / 12.0)


def test_quantization_correction_removes_the_inflation():
    rng = np.random.default_rng(7)
    true_rmssd = 15.0
    rr = 1000.0 + rng.normal(0, true_rmssd / math.sqrt(2), 20000)
    quantized = 60000.0 / np.round(60000.0 / rr)
    inflated = rmssd(quantized)
    corrected = correct_rmssd_for_quantization(inflated, quantization_variance(quantized))
    assert inflated > true_rmssd * 1.05          # the bias is real and large at low RMSSD
    assert corrected == pytest.approx(true_rmssd, rel=0.05)  # and removable in expectation


def test_window_metrics_rejects_short_windows():
    cfg = ScoreConfig()
    m = window_metrics(np.full(10, 900.0), cfg)
    assert not m.usable and m.reject_reason == "too_few_intervals"


def test_window_metrics_accepts_a_normal_window():
    rng = np.random.default_rng(1)
    rr = 950.0 + rng.normal(0, 25, 60)
    m = window_metrics(rr, ScoreConfig())
    assert m.usable
    assert 50 < m.mean_hr < 80
    assert m.n_kept == 60


def test_window_metrics_rejects_nonsense_input():
    m = window_metrics(np.array([900.0, -5.0, 900.0]), ScoreConfig())
    assert not m.usable and m.reject_reason == "empty_or_invalid"
