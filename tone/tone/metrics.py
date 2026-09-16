"""Per-window time-domain metrics: artifact filter, RMSSD, mean HR, SDNN, pNN50.

Everything here operates on one HRV window: the short burst of beat-to-beat
intervals Apple records when you are still. Nothing here knows about baselines
or z-scores; that is score.py.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import numpy as np

from .config import DEFAULT, ScoreConfig


@dataclass(frozen=True)
class WindowMetrics:
    """Time-domain metrics for a single HRV window."""

    n_input: int
    n_kept: int
    n_diffs: int
    rmssd: float
    mean_rr: float
    mean_hr: float
    sdnn: float
    pnn50: float
    usable: bool
    reject_reason: str | None = None
    quantization_var: float = 0.0
    rmssd_raw: float = float("nan")

    @property
    def artifact_fraction(self) -> float:
        return 0.0 if self.n_input == 0 else 1.0 - self.n_kept / self.n_input


def rr_from_bpm(bpm) -> np.ndarray:
    """Instantaneous bpm -> RR interval in ms.

    Apple's `InstantaneousBeatsPerMinute` entries in export.xml are *already*
    per-beat rates, so N entries give N intervals (not N-1). They are also
    rounded to whole bpm, which quantises RR — see quantization_variance.
    """
    bpm = np.asarray(bpm, dtype=float)
    if np.any(bpm <= 0):
        raise ValueError("bpm values must be positive")
    return 60000.0 / bpm


def rr_from_beat_times(seconds) -> np.ndarray:
    """Beat timestamps (seconds, monotonic) -> RR intervals in ms.

    This is the `HKHeartbeatSeriesSample` path the watch app will take: N beats
    give N-1 intervals. Kept here so the Python and Swift sides agree about the
    off-by-one before it becomes a parity bug.
    """
    t = np.asarray(seconds, dtype=float)
    if t.size < 2:
        return np.empty(0, dtype=float)
    d = np.diff(t) * 1000.0
    if np.any(d <= 0):
        raise ValueError("beat timestamps must be strictly increasing")
    return d


def filter_rr(rr, threshold: float = DEFAULT.artifact_threshold) -> np.ndarray:
    """Boolean mask of RR intervals that survive the artifact filter.

    The plan says "discard any interval that differs from its neighbour by more
    than 20%". Taken literally with a single left neighbour, that rule cascades:
    one rejected interval leaves the reference stale and can reject the whole
    remainder of a window. So the test here is two-sided and stateless -- an
    interval is *kept* if it agrees to within `threshold` with at least one of
    its immediate neighbours in the raw series:

        keep[k]  <=>  |rr[k] - rr[k-1]| / rr[k-1] <= threshold
                  or  |rr[k] - rr[k+1]| / rr[k+1] <= threshold

    An isolated ectopic beat (short interval followed by a compensatory long
    one) disagrees with both neighbours and is dropped; a genuine gradual drift
    in heart rate agrees with its neighbours throughout and is kept. The
    denominator is the neighbour being compared against, which makes the rule
    exactly reproducible in Swift with no rolling state.

    Series of length 1 are kept (no neighbour to disagree with); length 0
    returns an empty mask.
    """
    rr = np.asarray(rr, dtype=float)
    n = rr.size
    if n == 0:
        return np.zeros(0, dtype=bool)
    if n == 1:
        return np.ones(1, dtype=bool)

    ok_left = np.zeros(n, dtype=bool)
    ok_right = np.zeros(n, dtype=bool)
    # |rr[k] - rr[k-1]| / rr[k-1] <= threshold
    ok_left[1:] = np.abs(rr[1:] - rr[:-1]) / rr[:-1] <= threshold
    # |rr[k] - rr[k+1]| / rr[k+1] <= threshold
    ok_right[:-1] = np.abs(rr[:-1] - rr[1:]) / rr[1:] <= threshold
    return ok_left | ok_right


def rmssd(rr, mask=None) -> float:
    """Root mean square of successive differences, in ms.

    Only pairs where *both* intervals survived the filter contribute. A pair
    that straddles a dropped interval is not bridged: bridging would invent a
    difference across a gap and inflate RMSSD, which is precisely the artifact
    the filter exists to remove.

    Note on the denominator: with M usable intervals there are M-1 successive
    differences, and the mean is taken over the differences actually used. The
    plan writes 1/(N_i - 1) with N_i beats; with a clean window the two agree.
    """
    rr = np.asarray(rr, dtype=float)
    if mask is None:
        mask = np.ones(rr.size, dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    if rr.size < 2:
        return float("nan")
    pair = mask[:-1] & mask[1:]
    if not np.any(pair):
        return float("nan")
    d = np.diff(rr)[pair]
    return float(math.sqrt(float(np.mean(d * d))))


def sdnn(rr, mask=None) -> float:
    """Standard deviation of the kept NN intervals, in ms (sample SD, ddof=1)."""
    rr = np.asarray(rr, dtype=float)
    if mask is None:
        mask = np.ones(rr.size, dtype=bool)
    kept = rr[np.asarray(mask, dtype=bool)]
    if kept.size < 2:
        return float("nan")
    return float(np.std(kept, ddof=1))


def pnn50(rr, mask=None) -> float:
    """Fraction of successive differences greater than 50 ms."""
    rr = np.asarray(rr, dtype=float)
    if mask is None:
        mask = np.ones(rr.size, dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    if rr.size < 2:
        return float("nan")
    pair = mask[:-1] & mask[1:]
    if not np.any(pair):
        return float("nan")
    d = np.abs(np.diff(rr)[pair])
    return float(np.mean(d > 50.0))


def mean_rr(rr, mask=None) -> float:
    rr = np.asarray(rr, dtype=float)
    if mask is None:
        mask = np.ones(rr.size, dtype=bool)
    kept = rr[np.asarray(mask, dtype=bool)]
    if kept.size == 0:
        return float("nan")
    return float(np.mean(kept))


def mean_hr(rr, mask=None) -> float:
    """Mean heart rate over the window, bpm, from the same intervals as RMSSD.

    Step 2 defines HR_i = 60000 / mean(RR), i.e. the reciprocal of the mean
    interval, not the mean of instantaneous rates. The two differ by Jensen's
    inequality (the mean of rates is always the larger), by a few tenths of a
    bpm at typical variability -- small, but systematic, and enough to blow a
    1e-6 parity test.
    """
    m = mean_rr(rr, mask)
    if not math.isfinite(m) or m <= 0:
        return float("nan")
    return 60000.0 / m


def quantization_variance(rr, mask=None) -> float:
    """Per-interval variance (ms^2) contributed by whole-bpm rounding.

    Relevant only on the export.xml path, where RR was reconstructed from
    integer bpm. Rounding bpm to the nearest integer makes the RR error
    approximately uniform over a width of |d(RR)/d(bpm)| = 60000/bpm^2 ms, so
    its variance is (60000/bpm^2)^2 / 12. In RR terms bpm = 60000/RR, giving a
    width of RR^2/60000 ms.

    At 60 bpm that is a 16.7 ms wide bin, sd ~ 4.8 ms per interval, and the two
    independent errors in a successive difference add ~6.8 ms in quadrature. On
    a true RMSSD of 35 ms that is a +2% bias; on a true RMSSD of 15 ms it is
    +10%, and at 50 bpm +16%. The bias is therefore state-dependent -- largest
    exactly in the low-RMSSD windows the score cares about. See docs/FINDINGS.md.
    """
    rr = np.asarray(rr, dtype=float)
    if mask is None:
        mask = np.ones(rr.size, dtype=bool)
    kept = rr[np.asarray(mask, dtype=bool)]
    if kept.size == 0:
        return 0.0
    width = kept * kept / 60000.0
    return float(np.mean(width * width) / 12.0)


def correct_rmssd_for_quantization(observed_rmssd: float, quant_var: float) -> float:
    """Remove the quantisation contribution from RMSSD in quadrature.

    RMSSD^2 estimates E[(dRR)^2]. Independent per-interval quantisation noise of
    variance q adds 2q to that expectation, so the corrected estimate is
    sqrt(max(0, RMSSD^2 - 2q)). It is a bias correction, not a noise removal:
    the variance of the estimate is unchanged, and when the correction drives
    the value to (near) zero the window carried no recoverable RMSSD at all.
    """
    if not math.isfinite(observed_rmssd):
        return observed_rmssd
    corrected = observed_rmssd * observed_rmssd - 2.0 * quant_var
    if corrected <= 0.0:
        return float("nan")
    return float(math.sqrt(corrected))


def window_metrics(
    rr,
    cfg: ScoreConfig = DEFAULT,
    *,
    quantized: bool = False,
) -> WindowMetrics:
    """Run the full per-window pipeline over one array of RR intervals (ms).

    `quantized=True` marks RR that came from whole-bpm export data, which
    enables the quantisation correction when cfg.quantization_correction is on.
    """
    rr = np.asarray(rr, dtype=float)
    n_input = int(rr.size)

    if n_input == 0 or np.any(~np.isfinite(rr)) or np.any(rr <= 0):
        return WindowMetrics(n_input, 0, 0, float("nan"), float("nan"), float("nan"),
                             float("nan"), float("nan"), False, "empty_or_invalid")

    mask = filter_rr(rr, cfg.artifact_threshold)
    n_kept = int(mask.sum())
    n_diffs = int((mask[:-1] & mask[1:]).sum()) if n_input > 1 else 0

    quant_var = quantization_variance(rr, mask) if quantized else 0.0
    raw = rmssd(rr, mask)
    value = raw
    if quantized and cfg.quantization_correction:
        value = correct_rmssd_for_quantization(raw, quant_var)

    metrics = WindowMetrics(
        n_input=n_input,
        n_kept=n_kept,
        n_diffs=n_diffs,
        rmssd=value,
        mean_rr=mean_rr(rr, mask),
        mean_hr=mean_hr(rr, mask),
        sdnn=sdnn(rr, mask),
        pnn50=pnn50(rr, mask),
        usable=True,
        quantization_var=quant_var,
        rmssd_raw=raw,
    )

    reason = None
    if n_kept < cfg.min_intervals:
        reason = "too_few_intervals"
    elif n_diffs < 2:
        reason = "too_few_differences"
    elif not math.isfinite(metrics.rmssd) or metrics.rmssd <= 0:
        reason = "rmssd_not_finite"
    elif not math.isfinite(metrics.mean_hr) or metrics.mean_hr <= 0:
        reason = "hr_not_finite"

    if reason is not None:
        return dataclasses.replace(metrics, usable=False, reject_reason=reason)
    return metrics
