from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from tone import ecg
from tone.config import ScoreConfig
from tone.metrics import window_metrics
from tone.simulate import ecg_beat_times, synthetic_ecg

TZ = timezone(timedelta(hours=-4))
FS = 512.0


def run(duration=30.0, mean_hr=70.0, rmssd_ms=40.0, seed=4, **kw):
    """Detect beats in a synthetic ECG whose true R times are known."""
    beats = ecg_beat_times(duration=duration, mean_hr=mean_hr, rmssd_ms=rmssd_ms, seed=seed)
    voltage, _ = synthetic_ecg(beats, fs=FS, duration=duration, seed=seed, **kw)
    return beats, ecg.detect_r_peaks(voltage, FS), voltage


def rmssd_of(times_s):
    return float(np.sqrt(np.mean(np.diff(np.diff(times_s) * 1000.0) ** 2)))


# --- detection ---------------------------------------------------------------

def test_finds_every_beat():
    beats, peaks, _ = run()
    assert len(peaks) == len(beats)


def test_timing_jitter_is_well_under_a_sample():
    # 512 Hz gives 1.95 ms between samples. Parabolic interpolation should beat
    # that by an order of magnitude, which is the whole reason it is there.
    beats, peaks, _ = run()
    err = (peaks - beats) * 1000.0
    assert err.std() < 0.3
    assert ecg.timing_resolution_ms(FS) == pytest.approx(1.953125)


def test_the_timing_bias_is_constant_and_therefore_harmless():
    # The detector sits a fraction of a millisecond early on every beat. A
    # CONSTANT offset cancels exactly in successive differences, so it cannot
    # touch RMSSD. Only the scatter around it matters.
    beats, peaks, _ = run()
    err = (peaks - beats) * 1000.0
    assert abs(err.mean()) < 2.0
    assert np.abs(err - err.mean()).max() < 0.5


def test_rmssd_is_recovered_to_a_fraction_of_a_percent():
    beats, peaks, _ = run()
    assert rmssd_of(peaks) == pytest.approx(rmssd_of(beats), rel=0.01)


@pytest.mark.parametrize(
    "label,kwargs",
    [
        ("inverted lead", {"inverted": True}),
        ("heavy noise", {"noise_uv": 100.0}),
        ("mains hum", {"mains_uv": 200.0}),
        ("baseline wander", {"wander_uv": 500.0}),
    ],
)
def test_detector_survives_realistic_corruption(label, kwargs):
    beats, peaks, _ = run(**kwargs)
    assert len(peaks) == len(beats), label
    assert rmssd_of(peaks) == pytest.approx(rmssd_of(beats), rel=0.02), label


@pytest.mark.parametrize("hr", [50.0, 60.0, 70.0, 95.0])
def test_detector_works_across_the_plausible_heart_rate_range(hr):
    beats, peaks, _ = run(mean_hr=hr)
    assert len(peaks) == len(beats)
    assert rmssd_of(peaks) == pytest.approx(rmssd_of(beats), rel=0.02)


@pytest.mark.parametrize("seed", [1, 2, 3, 5])
def test_accuracy_holds_across_draws(seed):
    beats, peaks, _ = run(seed=seed)
    assert len(peaks) == len(beats)
    assert rmssd_of(peaks) == pytest.approx(rmssd_of(beats), rel=0.02)


def test_the_ecg_path_is_far_more_precise_than_the_export_path():
    # The point of implementing this at all. The export's whole-bpm rounding
    # inflates RMSSD by several percent; the ECG lands within a fraction of one.
    beats, peaks, _ = run(rmssd_ms=20.0, mean_hr=60.0, duration=60.0)
    truth = rmssd_of(beats)
    from_ecg = rmssd_of(peaks)

    rr_true = np.diff(beats) * 1000.0
    rr_quantized = 60000.0 / np.round(60000.0 / rr_true)
    from_export = float(np.sqrt(np.mean(np.diff(rr_quantized) ** 2)))

    assert abs(from_ecg - truth) < 0.02 * truth
    assert abs(from_export - truth) > 2 * abs(from_ecg - truth)


def test_silence_yields_no_beats():
    assert ecg.detect_r_peaks(np.zeros(int(5 * FS)), FS).size == 0


def test_a_too_short_recording_is_refused_rather_than_guessed_at():
    assert ecg.detect_r_peaks(np.zeros(10), FS).size == 0


# --- intervals ---------------------------------------------------------------

def test_n_beats_give_n_minus_one_intervals():
    beats, peaks, voltage = run()
    rr = ecg.rr_from_ecg(voltage, FS)
    assert rr.size == len(peaks) - 1


def test_concatenating_recordings_does_not_invent_an_interval_between_them():
    # The gap between two recordings is not a beat-to-beat interval. Treating it
    # as one would put a ~minute-long "RR" into the series.
    _, _, v1 = run(seed=1)
    _, _, v2 = run(seed=2)
    a = ecg.Recording(v1, FS)
    b = ecg.Recording(v2, FS)
    combined = ecg.rr_from_recordings([a, b])
    assert combined.size == ecg.rr_from_ecg(v1, FS).size + ecg.rr_from_ecg(v2, FS).size
    assert combined.max() < 2000.0  # nothing resembling a gap between recordings


def test_two_recordings_clear_the_interval_threshold_at_any_plausible_rate():
    cfg = ScoreConfig()
    _, _, v1 = run(mean_hr=52.0, seed=1)
    _, _, v2 = run(mean_hr=52.0, seed=2)
    one = window_metrics(ecg.rr_from_ecg(v1, FS), cfg)
    two = window_metrics(ecg.rr_from_recordings(
        [ecg.Recording(v1, FS), ecg.Recording(v2, FS)]), cfg)
    assert not one.usable and one.reject_reason == "too_few_intervals"
    assert two.usable


# --- the 30-second problem ---------------------------------------------------

def test_a_single_thirty_second_ecg_is_too_short_below_62_bpm():
    # Not a rounding detail: a resting heart rate under ~62 means every single
    # ECG is silently dropped by min_intervals.
    assert ecg.minimum_heart_rate() == pytest.approx(62.0)
    assert not ecg.enough_intervals(60.0)
    assert not ecg.enough_intervals(55.0)
    assert ecg.enough_intervals(62.0)
    assert ecg.enough_intervals(70.0)


def test_a_sixty_second_pair_moves_the_threshold_out_of_the_way():
    assert ecg.minimum_heart_rate(duration_s=60.0) == pytest.approx(31.0)
    assert ecg.enough_intervals(45.0, duration_s=60.0)


# --- reading Apple's CSV -----------------------------------------------------

APPLE_CSV = """Name,Gavin Eubanks
Date of Birth,2004-03-11
Recording Date,2026-09-16 11:03:47 -0400
Classification,Sinus Rhythm
Symptoms,None
Software Version,10.1
Device,Watch6,2
Sample Rate,512 Hz
Lead,Lead I
Unit,µV
Version,1

−5,
12,
−30,
41,
"""


def test_reads_an_apple_ecg_csv(tmp_path):
    path = tmp_path / "ecg_2026-09-16.csv"
    path.write_text(APPLE_CSV)
    rec = ecg.read_ecg_csv(path)
    assert rec.fs == pytest.approx(512.0)
    assert rec.classification == "Sinus Rhythm"
    assert list(rec.voltage) == [-5.0, 12.0, -30.0, 41.0]


def test_the_unicode_minus_is_not_silently_dropped(tmp_path):
    # Apple writes U+2212, which float() rejects. A reader that skips
    # unparseable lines produces a half-rectified trace that still looks like an
    # ECG and detects beats badly.
    path = tmp_path / "ecg.csv"
    path.write_text(APPLE_CSV)
    voltage = ecg.read_ecg_csv(path).voltage
    assert (voltage < 0).sum() == 2


def test_reads_a_comma_decimal_locale_export(tmp_path):
    path = tmp_path / "ecg_de.csv"
    path.write_text(
        "Name;Gavin\nSample Rate;512 Hz\nKlassifizierung;Sinusrhythmus\n\n"
        "−5,5;\n12,25;\n−30,0;\n"
    )
    rec = ecg.read_ecg_csv(path)
    assert rec.fs == pytest.approx(512.0)
    assert list(rec.voltage) == [-5.5, 12.25, -30.0]


def test_end_to_end_a_recording_becomes_a_scoreable_window(tmp_path):
    _, _, voltage = run(mean_hr=72.0)
    rec = ecg.Recording(voltage, FS)
    when = datetime(2026, 9, 16, 17, 5, tzinfo=TZ)
    window = ecg.ecg_window(rec, when)

    assert window.on_demand           # you chose to take it
    assert not window.quantized       # nothing was rounded to whole bpm
    assert window.hour == pytest.approx(17.0 + 5 / 60)
    metrics = window_metrics(window.rr, ScoreConfig(), quantized=window.quantized)
    assert metrics.usable
    assert 60 < metrics.mean_hr < 85


# --- the interpolation itself ------------------------------------------------

def test_parabolic_vertex_finds_a_known_maximum():
    # y = -(x - 0.3)^2 sampled at -1, 0, 1 has its vertex at +0.3.
    f = lambda x: -((x - 0.3) ** 2)
    assert ecg._parabolic_vertex(f(-1), f(0), f(1)) == pytest.approx(0.3)


def test_parabolic_vertex_is_zero_when_the_middle_sample_is_the_peak():
    assert ecg._parabolic_vertex(1.0, 2.0, 1.0) == pytest.approx(0.0)


def test_parabolic_vertex_handles_a_flat_triple():
    assert ecg._parabolic_vertex(1.0, 1.0, 1.0) == 0.0


def test_bandpass_kernel_is_symmetric_so_it_shifts_nothing():
    # A linear-phase FIR in 'same' mode introduces no delay. A causal IIR would,
    # and a frequency-dependent delay does not cancel in RR differences.
    k = ecg._bandpass_kernel(FS, 5.0, 18.0)
    assert k.size % 2 == 1
    assert np.allclose(k, k[::-1])
    assert abs(k.sum()) < 1e-9  # no DC response


# --- sittings ----------------------------------------------------------------

def _rec(seed=1, hr=70.0):
    _, _, v = run(seed=seed, mean_hr=hr)
    return ecg.Recording(v, FS)


def test_back_to_back_recordings_become_one_sitting():
    base = datetime(2026, 9, 14, 17, 2, tzinfo=TZ)
    dated = [(base, _rec(1)), (base + timedelta(seconds=45), _rec(2))]
    groups = ecg.group_sittings(dated, pair_within_minutes=2.0)
    assert len(groups) == 1 and len(groups[0]) == 2


def test_a_test_retest_pair_is_never_merged():
    # This is the hazard the 2-minute default exists to avoid: Phase 2.1 pairs
    # are five minutes apart, and merging them would destroy the calibration.
    base = datetime(2026, 9, 14, 17, 2, tzinfo=TZ)
    dated = [(base, _rec(1)), (base + timedelta(minutes=5), _rec(2))]
    groups = ecg.group_sittings(dated, pair_within_minutes=2.0)
    assert len(groups) == 2


def test_sittings_are_ordered_and_complete():
    base = datetime(2026, 9, 14, 17, 2, tzinfo=TZ)
    dated = [(base + timedelta(minutes=30), _rec(3)),
             (base, _rec(1)),
             (base + timedelta(seconds=30), _rec(2))]
    groups = ecg.group_sittings(dated, pair_within_minutes=2.0)
    assert [len(g) for g in groups] == [2, 1]
    assert sum(len(g) for g in groups) == 3
    assert groups[0][0][0] < groups[0][1][0] < groups[1][0][0]


def test_a_slow_heart_needs_two_recordings_per_half_of_a_test_retest_pair():
    # Concrete consequence for anyone resting under ~62 bpm: a Phase 2.1 pair
    # is FOUR ECGs, not two. Two single ECGs five minutes apart give two dropped
    # windows and no calibration at all.
    cfg = ScoreConfig()
    slow = 55.0
    assert not ecg.enough_intervals(slow)
    singles = [window_metrics(ecg.rr_from_ecg(_rec(s, slow).voltage, FS), cfg)
               for s in (1, 2)]
    assert all(not m.usable for m in singles)
    doubled = [window_metrics(ecg.rr_from_recordings([_rec(s, slow), _rec(s + 10, slow)]), cfg)
               for s in (1, 2)]
    assert all(m.usable for m in doubled)
