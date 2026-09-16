"""Windows constructed to reach the branches real data rarely reaches.

The fixture generated from your own export pins the science on real windows.
It cannot pin the degenerate paths, because a body that produces five windows a
day spread across the clock never takes them. Mutation-testing the C port
against the ordinary fixture showed exactly that: deliberate errors in the
flat-baseline guard, in the MAD-is-zero fallback and at the 28-day boundary all
*survived* -- the fixture could not tell the mutated engine from the real one.

A port that guesses differently on an unexercised branch passes the gate and is
still wrong, so those branches get their own fixture. Everything here is
deterministic: no RNG, so the file is reproducible and the numbers can be
checked by hand.

Scenarios are spaced more than `baseline_days` apart, so each one's trailing
baseline contains only its own windows and the scenarios cannot contaminate
each other.
"""

from __future__ import annotations


from datetime import datetime, timedelta, timezone

import numpy as np

from .score import Window

TZ = timezone(timedelta(hours=-4))
EPOCH = datetime(2026, 1, 1, 0, 0, tzinfo=TZ)


def _rr(mean_rr: float, amplitude: float, n: int = 60, phase: float = 0.0) -> np.ndarray:
    """A deterministic interval series with non-zero, controllable RMSSD.

    Two incommensurate sinusoids rather than noise: successive differences are
    non-constant (so RMSSD is meaningful), nothing is random, and the values are
    reproducible from the arguments alone.
    """
    k = np.arange(n, dtype=float)
    return (mean_rr
            + amplitude * np.sin(0.7 * k + phase)
            + 0.4 * amplitude * np.sin(1.9 * k + 2.0 * phase))


def _at(day: int, hour: float) -> datetime:
    return EPOCH + timedelta(days=day, hours=hour)


def few_distinct_hours(start_day: int = 0) -> list[Window]:
    """Reaches the flat-baseline guard: every window at one of three hours.

    With fewer than `min_baseline_hours` distinct clock hours in the trailing
    window, cos and sin cannot be separated from the intercept, so the fit must
    degrade to a MESOR-only mean. An engine that skips the guard fits an
    amplitude to nothing and diverges immediately.
    """
    out = []
    for day in range(14):
        for i, hour in enumerate((8.0, 9.5, 20.0)):
            out.append(Window(
                start=_at(start_day + day, hour),
                rr=_rr(950.0 + 3.0 * day + 7.0 * i, 26.0 + 0.5 * day, phase=0.3 * (day + i)),
                source="edge:few-hours",
            ))
    return out


def zero_mad(start_day: int = 60) -> list[Window]:
    """Reaches the MAD-is-zero fallback in the robust sigma.

    More than half the baseline windows are byte-identical, so the median
    absolute deviation of the residuals is exactly zero while the sample SD is
    not. An engine without the fallback divides by zero here; one that uses a
    different divisor for the fallback SD gets a different z.
    """
    out = []
    identical = _rr(900.0, 25.0, phase=0.0)
    for day in range(14):
        # Three identical windows a day, spread over the clock so the cosinor
        # itself still fits -- it is the residual spread that is degenerate.
        for hour in (7.0, 13.0, 19.0):
            out.append(Window(start=_at(start_day + day, hour), rr=identical.copy(),
                              source="edge:identical"))
    # A minority of genuinely different windows: enough to make the SD positive,
    # too few to move the median.
    for i, day in enumerate((3, 6, 9, 12)):
        out.append(Window(start=_at(start_day + day, 22.0),
                          rr=_rr(870.0 + 11.0 * i, 40.0, phase=1.1 * i),
                          source="edge:different"))
    return out


def boundary_28_days(start_day: int = 120) -> list[Window]:
    """Places a window exactly `baseline_days` before a scored one.

    Whether the trailing window is half-open at the old end decides if that
    window is in the baseline. One window either way changes a 30-point fit far
    more than 1e-6.
    """
    out = [Window(start=_at(start_day, 6.0), rr=_rr(1010.0, 44.0, phase=0.9),
                  source="edge:boundary-exact")]
    # 33 windows inside the 28 days that follow, spread across the clock.
    for i in range(33):
        day = start_day + 1 + (i * 26) // 33  # days 1..26 after the boundary window
        hour = 2.0 + (i * 19) % 21
        out.append(Window(start=_at(day, hour),
                          rr=_rr(930.0 + 2.0 * i, 24.0 + 0.3 * i, phase=0.17 * i),
                          source="edge:boundary-fill"))
    # The scored window, exactly 28 days after the first one to the second.
    out.append(Window(start=_at(start_day + 28, 6.0), rr=_rr(940.0, 30.0, phase=0.5),
                      source="edge:boundary-target"))
    return out


def lonely_day(start_day: int = 200) -> list[Window]:
    """A day with exactly one scored window, which must borrow the pooled SD."""
    out = []
    for day in range(12):
        for hour in (3.0, 8.0, 12.0, 17.0, 22.0):
            out.append(Window(start=_at(start_day + day, hour),
                              rr=_rr(925.0 + 4.0 * day, 27.0 + 0.6 * hour,
                                     phase=0.11 * (day * 5 + hour)),
                              source="edge:pooled-fill"))
    out.append(Window(start=_at(start_day + 13, 10.0), rr=_rr(880.0, 35.0, phase=2.2),
                      source="edge:lonely"))
    return out


def unusable(start_day: int = 240) -> list[Window]:
    """Windows the engine must drop, and one it must keep despite an ectopic beat.

    A port that keeps a too-short window, or that rejects the ectopic one
    outright instead of filtering two intervals out of it, shifts every
    subsequent baseline and fails everything downstream.
    """
    out = []
    for day in range(12):
        for hour in (4.0, 9.0, 14.0, 19.0, 23.0):
            out.append(Window(start=_at(start_day + day, hour),
                              rr=_rr(940.0 + 3.0 * day, 25.0, phase=0.23 * (day + hour)),
                              source="edge:fill"))
    # Too few intervals: must be dropped, not scored.
    out.append(Window(start=_at(start_day + 5, 11.0), rr=_rr(900.0, 20.0, n=12),
                      source="edge:too-short"))
    # An ectopic beat and its compensatory pause: two intervals filtered, the
    # rest of the window kept, and RMSSD must not bridge the gap.
    ectopic = _rr(900.0, 22.0, n=60, phase=0.8)
    ectopic[25] *= 0.55
    ectopic[26] *= 1.45
    out.append(Window(start=_at(start_day + 7, 16.0), rr=ectopic, source="edge:ectopic"))
    # An empty window: no intervals at all.
    out.append(Window(start=_at(start_day + 9, 6.0), rr=np.zeros(0), source="edge:empty"))
    return out


def all_windows() -> list[Window]:
    """Every scenario, in time order."""
    windows = (few_distinct_hours() + zero_mad() + boundary_28_days()
               + lonely_day() + unusable())
    windows.sort(key=lambda w: w.start)
    return windows


#: Branches the edge fixture exists to pin. `coverage` counts how many scored
#: windows reached each one; the tests assert every count is non-zero, because a
#: scenario that silently stops triggering leaves the gate weaker than it looks.
REQUIRED_BRANCHES = ("flat_baseline", "sd_fallback", "pooled_sd", "dropped", "filtered")


def coverage(scores, daily=None) -> dict:
    """Which degenerate branches the scored output actually reached."""
    scored = [s for s in scores if s.scored]
    return {
        "scored": len(scored),
        "flat_baseline": sum(1 for s in scored if "flat_baseline" in s.flags),
        "sd_fallback": sum(1 for s in scored if "sd_fallback" in s.flags),
        "pooled_sd": sum(1 for d in (daily or []) if d.sd_pooled),
        "dropped": sum(1 for s in scores if not s.metrics.usable),
        "filtered": sum(
            1 for s in scores if s.metrics.usable and s.metrics.n_kept < s.metrics.n_input
        ),
    }


def report(scores, daily=None) -> str:
    counts = coverage(scores, daily)
    lines = [f"scored windows: {counts['scored']}"]
    for branch in REQUIRED_BRANCHES:
        mark = " " if counts[branch] else "  <-- NOT REACHED"
        lines.append(f"  {branch:<15} {counts[branch]:>4}{mark}")
    return "\n".join(lines)
