"""The score itself (Steps 2-6 of Section 3).

Pipeline for each HRV window i, in order:

    metrics   -> RMSSD_i, HR_i                             (metrics.py)
    log       -> x_i = ln RMSSD_i, h_i = ln HR_i           (Step 2)
    baseline  -> cosinor fit on the trailing 28 days        (Step 3)
    residual  -> r_i = value - baseline(clock hour of i)
    robust z  -> Z_i = r_i / (1.4826 * MAD of baseline residuals)  (Step 4)
    combine   -> S_i, precision-weighted                    (Step 5)
    aggregate -> mean per day with an interval              (Step 6)

Two choices worth knowing about because they are not in the prose of the plan:

1. The trailing baseline *excludes the window being scored*. Including it lets
   each sample pull its own baseline towards itself and shrink its own
   residual. The effect is small at n ~ 140, but excluding it is also what the
   deployed watch app necessarily does -- it scores a new sample against
   history that does not contain it -- so offline and on-wrist agree.

2. The daily interval defaults to Student's t with n_d - 1 degrees of freedom
   rather than 1.96. With n_d ~ 5 passive windows a day, 1.96 understates the
   interval by about 40%, and an honest interval is the whole point of showing
   one. cfg.interval = "normal" restores the plan's literal 1.96.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import numpy as np

from . import cosinor
from .config import DEFAULT, ScoreConfig
from .metrics import WindowMetrics, window_metrics

# Two-sided 97.5th percentile of Student's t, df 1..30. A 30-entry table is
# portable to Swift verbatim; beyond df 30 the normal value is within 0.5%.
T_CRIT_975 = {
    1: 12.706205, 2: 4.302653, 3: 3.182446, 4: 2.776445, 5: 2.570582,
    6: 2.446912, 7: 2.364624, 8: 2.306004, 9: 2.262157, 10: 2.228139,
    11: 2.200985, 12: 2.178813, 13: 2.160369, 14: 2.144787, 15: 2.131450,
    16: 2.119905, 17: 2.109816, 18: 2.100922, 19: 2.093024, 20: 2.085963,
    21: 2.079614, 22: 2.073873, 23: 2.068658, 24: 2.063899, 25: 2.059539,
    26: 2.055529, 27: 2.051831, 28: 2.048407, 29: 2.045230, 30: 2.042272,
}
Z_CRIT_975 = 1.959964


def critical_value(df: int, kind: str = "t") -> float:
    if kind == "normal":
        return Z_CRIT_975
    if df <= 0:
        return float("nan")
    return T_CRIT_975.get(df, Z_CRIT_975)


@dataclass(frozen=True)
class Window:
    """One HRV window: when it happened and the intervals inside it."""

    start: datetime
    rr: np.ndarray
    source: str = "unknown"
    on_demand: bool = False
    quantized: bool = False
    apple_sdnn: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "rr", np.asarray(self.rr, dtype=float))

    @property
    def hour(self) -> float:
        """Local clock hour in [0, 24), fractional."""
        s = self.start
        return s.hour + s.minute / 60.0 + s.second / 3600.0 + s.microsecond / 3.6e9

    @property
    def day(self) -> date:
        return self.start.date()

    @property
    def epoch(self) -> float:
        return self.start.timestamp()


@dataclass
class WindowScore:
    """Everything computed for one window, including why it was not scored."""

    start: datetime
    hour: float
    day: date
    on_demand: bool
    source: str
    metrics: WindowMetrics
    x: float = float("nan")
    h: float = float("nan")
    resid_x: float = float("nan")
    resid_h: float = float("nan")
    sigma_x: float = float("nan")
    sigma_h: float = float("nan")
    z_x: float = float("nan")
    z_h: float = float("nan")
    s: float = float("nan")
    n_baseline: int = 0
    baseline_r2_x: float = float("nan")
    baseline_r2_h: float = float("nan")
    flags: list[str] = field(default_factory=list)

    @property
    def scored(self) -> bool:
        return math.isfinite(self.s)

    def to_dict(self) -> dict:
        return {
            "start": self.start.isoformat(),
            "hour": self.hour,
            "day": self.day.isoformat(),
            "on_demand": self.on_demand,
            "source": self.source,
            "n_input": self.metrics.n_input,
            "n_kept": self.metrics.n_kept,
            "rmssd": self.metrics.rmssd,
            "mean_hr": self.metrics.mean_hr,
            "sdnn": self.metrics.sdnn,
            "x": self.x,
            "h": self.h,
            "resid_x": self.resid_x,
            "resid_h": self.resid_h,
            "sigma_x": self.sigma_x,
            "sigma_h": self.sigma_h,
            "z_x": self.z_x,
            "z_h": self.z_h,
            "s": self.s,
            "n_baseline": self.n_baseline,
            "baseline_r2_x": self.baseline_r2_x,
            "baseline_r2_h": self.baseline_r2_h,
            "flags": list(self.flags),
        }


@dataclass(frozen=True)
class DailyScore:
    """Step 6: the number the watch actually shows, with its interval."""

    day: date
    mean: float
    n: int
    sd: float
    se: float
    lo: float
    hi: float
    sd_pooled: bool
    newest: datetime

    def to_dict(self) -> dict:
        return {
            "day": self.day.isoformat(),
            "mean": self.mean,
            "n": self.n,
            "sd": self.sd,
            "se": self.se,
            "lo": self.lo,
            "hi": self.hi,
            "sd_pooled": self.sd_pooled,
            "newest": self.newest.isoformat(),
        }


# Residuals live on a log scale, where a spread of 1e-9 means variation of one
# part per billion. Nothing physiological is that quiet, so a sigma below this
# is floating-point dust, not a denominator.
MIN_SIGMA = 1e-9


def robust_sigma_detail(
    values, mad_scale: float = DEFAULT.mad_scale, floor: float = MIN_SIGMA
) -> tuple[float, bool]:
    """1.4826 * MAD, and whether the sample-SD fallback was needed.

    MAD is exactly zero whenever more than half the residuals are identical,
    which real data can produce after heavy filtering; the sample SD (ddof=1)
    covers that case. Below `floor` the channel has no usable spread at all and
    NaN is returned, so the window goes unscored instead of being reported as a
    thousand-sigma event.

    The boolean is reported because the fallback is a branch a port can get
    wrong invisibly -- `tone.edge_cases` builds a fixture that reaches it, and
    the flag is how the test knows it got there.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return float("nan"), False
    mad = float(np.median(np.abs(v - float(np.median(v)))))
    sigma = mad_scale * mad
    fallback = sigma <= floor
    if fallback:
        sigma = float(np.std(v, ddof=1))
    return (sigma if sigma > floor else float("nan")), fallback


def robust_sigma(values, mad_scale: float = DEFAULT.mad_scale, floor: float = MIN_SIGMA) -> float:
    """1.4826 * median absolute deviation, with two documented fallbacks."""
    return robust_sigma_detail(values, mad_scale, floor)[0]


def combine(z_x: float, z_h: float, cfg: ScoreConfig = DEFAULT) -> float:
    """Step 5. Note the sign: low vagal tone and high heart rate both push up.

    S is a stress index, so the HRV channel enters negated. A day where RMSSD
    sits one robust sigma below its usual level for that hour, and HR one sigma
    above, scores S = +1 whatever the weights.
    """
    w_x, w_h = cfg.weights()
    total = w_x + w_h
    if total <= 0 or not (math.isfinite(z_x) and math.isfinite(z_h)):
        return float("nan")
    return (w_x * (-z_x) + w_h * z_h) / total


def _score_channel(
    hours: np.ndarray,
    values: np.ndarray,
    index: int,
    lo: int,
    cfg: ScoreConfig,
) -> tuple[float, float, float, cosinor.CosinorFit, bool]:
    """Residual, sigma, z, baseline fit, and whether the SD fallback fired."""
    base_hours = hours[lo:index]
    base_values = values[lo:index]
    n = base_values.size

    if n < cfg.min_baseline_windows:
        fit = cosinor.flat_fit(base_values)
    else:
        fit = cosinor.fit(base_hours, base_values, min_distinct_hours=cfg.min_baseline_hours)

    resid_here = float(values[index] - float(fit.predict(np.array([hours[index]]))[0]))
    base_resid = base_values - fit.predict(base_hours)
    sigma, fallback = robust_sigma_detail(base_resid, cfg.mad_scale)
    z = resid_here / sigma if (math.isfinite(sigma) and sigma > 0) else float("nan")
    return resid_here, sigma, z, fit, fallback


def score_windows(windows, cfg: ScoreConfig = DEFAULT) -> list[WindowScore]:
    """Score every window against its own trailing baseline.

    Windows are sorted by time; unusable ones still appear in the output with
    `scored == False` and a reason in `flags`, because "how many windows did I
    throw away and why" is a question you will want answered.
    """
    ordered = sorted(windows, key=lambda w: w.start)
    results: list[WindowScore] = []

    usable: list[tuple[int, Window, WindowMetrics]] = []
    for w in ordered:
        m = window_metrics(w.rr, cfg, quantized=w.quantized)
        ws = WindowScore(
            start=w.start, hour=w.hour, day=w.day, on_demand=w.on_demand,
            source=w.source, metrics=m,
        )
        if not m.usable:
            ws.flags.append(m.reject_reason or "unusable")
        else:
            ws.x = math.log(m.rmssd)
            ws.h = math.log(m.mean_hr)
            usable.append((len(results), w, m))
        results.append(ws)

    if not usable:
        return results

    epochs = np.array([w.epoch for _, w, _ in usable], dtype=float)
    hours = np.array([w.hour for _, w, _ in usable], dtype=float)
    xs = np.array([results[i].x for i, _, _ in usable], dtype=float)
    hs = np.array([results[i].h for i, _, _ in usable], dtype=float)
    span = cfg.baseline_days * 86400.0

    for k, (result_index, _, _) in enumerate(usable):
        ws = results[result_index]
        lo = int(np.searchsorted(epochs, epochs[k] - span, side="left"))
        n_base = k - lo
        ws.n_baseline = n_base

        if n_base < cfg.min_scoring_windows:
            ws.flags.append("warming_up")
            continue

        rx, sx, zx, fit_x, fb_x = _score_channel(hours, xs, k, lo, cfg)
        rh, sh, zh, fit_h, fb_h = _score_channel(hours, hs, k, lo, cfg)

        ws.resid_x, ws.sigma_x, ws.z_x = rx, sx, zx
        ws.resid_h, ws.sigma_h, ws.z_h = rh, sh, zh
        ws.baseline_r2_x, ws.baseline_r2_h = fit_x.r2, fit_h.r2
        if fit_x.flat or fit_h.flat:
            ws.flags.append("flat_baseline")
        if fb_x or fb_h:
            ws.flags.append("sd_fallback")
        if not (math.isfinite(sx) and math.isfinite(sh)):
            ws.flags.append("no_spread")
        if not cfg.weights_measured:
            ws.flags.append("equal_weights")

        ws.s = combine(zx, zh, cfg)
        if not math.isfinite(ws.s):
            ws.flags.append("no_score")

    return results


def daily_scores(scores, cfg: ScoreConfig = DEFAULT) -> list[DailyScore]:
    """Step 6: one number per day, with a 95% interval on the mean.

    A day with a single usable window has no within-day spread to estimate, so
    it borrows the pooled within-day SD of every other day and is marked
    `sd_pooled`. That is a real assumption (this day is no noisier than your
    usual day) and it is flagged rather than hidden.
    """
    scored = [s for s in scores if s.scored]
    if not scored:
        return []

    by_day: dict[date, list[WindowScore]] = {}
    for s in scored:
        by_day.setdefault(s.day, []).append(s)

    # Pooled within-day variance, for days with a single window.
    ss, dof = 0.0, 0
    for day_scores in by_day.values():
        if len(day_scores) >= 2:
            v = np.array([s.s for s in day_scores], dtype=float)
            ss += float(np.sum((v - v.mean()) ** 2))
            dof += len(day_scores) - 1
    pooled_sd = math.sqrt(ss / dof) if dof > 0 else float("nan")

    out: list[DailyScore] = []
    for day in sorted(by_day):
        day_scores = by_day[day]
        v = np.array([s.s for s in day_scores], dtype=float)
        n = int(v.size)
        mean = float(v.mean())
        newest = max(s.start for s in day_scores)

        if n >= 2:
            sd = float(np.std(v, ddof=1))
            se = sd / math.sqrt(n)
            crit = critical_value(n - 1, cfg.interval)
            pooled = False
        else:
            sd = pooled_sd
            se = pooled_sd
            crit = critical_value(max(dof, 1), cfg.interval)
            pooled = True

        if math.isfinite(se) and math.isfinite(crit):
            lo, hi = mean - crit * se, mean + crit * se
        else:
            lo = hi = float("nan")

        out.append(DailyScore(day, mean, n, sd, se, lo, hi, pooled, newest))
    return out


def newest_sample_age(scores, now: datetime | None = None) -> timedelta | None:
    """Age of the newest usable window -- what the watch face shows next to S.

    None when there is nothing usable at all. The app never interpolates over
    this gap; it says how old the number is.
    """
    usable = [s.start for s in scores if s.metrics.usable]
    if not usable:
        return None
    newest = max(usable)
    reference = now or datetime.now(tz=newest.tzinfo)
    return reference - newest
