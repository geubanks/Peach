"""How many days does Phase 2.2 need before it can answer anything?

This module exists because of a problem in the plan's own sequencing. Phase 1.3
is "done when: 14 consecutive days with >= 2 entries/day", and Phase 2.2 runs in
weeks 3-5 on what that produces. Phase 2.2's decision box then reads an interval
spanning zero as "rebuild", and "rebuild" routes to *do not write Swift*.

At n = 14 days, the smallest Spearman rho whose 95% interval excludes zero is
**0.54**. The plan's own threshold for "the signal is real" is 0.30, and it says
in the same paragraph that ambulatory HRV-stress correlations in the literature
are modest. So a perfectly real rho of 0.3, measured at 14 days, produces an
interval spanning zero and a verdict of "rebuild" -- a false negative built into
the schedule, on the one test the whole project turns on.

Resolving rho = 0.3 takes about **46 days of overlapping diary-and-score**, and
overlap is what counts: days that have both a rating and a scoreable day, after
the baseline warm-up. Call it seven weeks of diary, not two.

The fix is not to lower the bar. It is to stop conflating "no effect" with "not
enough data", which `diary.classify` now does via the `underpowered` verdict, and
to know the number of days in advance, which is what this module is for.

Method
------
Spearman's rho, Fisher-transformed, has approximate variance 1.06/(n-3)
(Fieller, Hartley & Pearson 1957), giving

    CI = tanh( atanh(rho) +/- 1.96 * sqrt(1.06/(n-3)) )

Two checks before trusting that. Against the cluster bootstrap this package
actually ships, the two intervals agree to within ~0.05 at n = 14 and ~0.02 by
n = 30. Against simulated truth, empirical coverage is 94.4-96.0% for rho in
{0, 0.3, 0.5} at n in {14, 30, 60, 120}. Both are in the test suite. The
analytic form is used here rather than simulation because it is fast enough to
answer "how many more days?" interactively, and because it can be explained.

A caveat the arithmetic cannot capture: this is the power to detect the rho your
pipeline *produces*, which is already attenuated by measurement noise relative to
the rho between your true physiology and your true mood. Attenuation makes the
required sample larger, never smaller. Treat these numbers as a floor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

Z_975 = 1.959964
SPEARMAN_VAR_FACTOR = 1.06  # Fieller, Hartley & Pearson (1957)

# The plan's own threshold for "the signal is real" (Section 2.2).
REAL_THRESHOLD = 0.30


def fisher_ci(rho: float, n: int, *, z: float = Z_975) -> tuple[float, float]:
    """Analytic 95% interval for Spearman's rho at sample size n."""
    if n <= 3 or not math.isfinite(rho):
        return (float("nan"), float("nan"))
    se = math.sqrt(SPEARMAN_VAR_FACTOR / (n - 3))
    zr = math.atanh(max(min(rho, 0.9999), -0.9999))
    return (math.tanh(zr - z * se), math.tanh(zr + z * se))


def min_detectable_rho(n: int, *, z: float = Z_975) -> float:
    """Smallest rho at sample size n whose 95% interval excludes zero."""
    if n <= 3:
        return float("nan")
    return math.tanh(z * math.sqrt(SPEARMAN_VAR_FACTOR / (n - 3)))


def days_required(rho: float, *, z: float = Z_975) -> int:
    """Overlapping days needed for an interval around `rho` to exclude zero.

    This is the *median* requirement: it asks when the interval around the rho
    you actually observe clears zero, not when you have 80% power to observe
    such a rho in the first place. `days_for_power` answers the second question,
    and its answer is larger.
    """
    if not math.isfinite(rho) or abs(rho) >= 1.0 or rho == 0.0:
        return -1
    zr = abs(math.atanh(rho))
    return math.ceil(SPEARMAN_VAR_FACTOR * (z / zr) ** 2 + 3)


def days_for_power(rho: float, *, power: float = 0.80, z: float = Z_975) -> int:
    """Days needed for an `power` chance of getting an interval excluding zero.

    The usual two-sided test at 5% with the requested power. This is the number
    to plan around; `days_required` is the number you happen to need if your
    observed rho lands exactly on the truth, which it will not half the time.
    """
    if not math.isfinite(rho) or abs(rho) >= 1.0 or rho == 0.0:
        return -1
    z_beta = {0.50: 0.0, 0.80: 0.841621, 0.90: 1.281552, 0.95: 1.644854}.get(power)
    if z_beta is None:
        raise ValueError("power must be one of 0.50, 0.80, 0.90, 0.95")
    zr = abs(math.atanh(rho))
    return math.ceil(SPEARMAN_VAR_FACTOR * ((z + z_beta) / zr) ** 2 + 3)


@dataclass(frozen=True)
class Assessment:
    """What a given (rho, n) can and cannot support."""

    rho: float
    n: int
    lo: float
    hi: float
    conclusive: bool
    days_to_resolve: int
    note: str


def assess(rho: float, n: int, *, threshold: float = REAL_THRESHOLD) -> Assessment:
    """Is this result conclusive, and if not, how many more days would make it?

    "Conclusive" means the interval settles the question the plan asked: either
    it excludes zero (there is an effect), or it excludes the threshold (any
    effect is smaller than the one you said you cared about). An interval that
    contains both zero and the threshold has settled nothing.
    """
    lo, hi = fisher_ci(rho, n)
    if not math.isfinite(lo):
        return Assessment(rho, n, lo, hi, False, -1, "too few days to estimate anything.")

    excludes_zero = lo > 0 or hi < 0
    excludes_threshold = hi < threshold

    if excludes_zero:
        return Assessment(rho, n, lo, hi, True, 0,
                          "the interval excludes zero; the question is answered.")
    if excludes_threshold:
        return Assessment(
            rho, n, lo, hi, True, 0,
            f"the interval spans zero but excludes rho = {threshold:.2f}. You have not "
            "shown an effect, and you have ruled out one as large as the threshold you "
            "set. That is a real negative result, not a sample-size problem.",
        )

    need = days_required(rho) if rho > 0 else -1
    more = max(0, need - n) if need > 0 else -1
    return Assessment(
        rho, n, lo, hi, False, more,
        f"the interval contains both zero and rho = {threshold:.2f}: consistent with no "
        "effect AND with the effect you are looking for, so it settles nothing. "
        + (f"About {more} more overlapping days would resolve it at the observed rho."
           if more > 0 else "Keep collecting."),
    )


def table(days=(14, 21, 28, 45, 60, 90, 120)) -> str:
    """The 'what can I even detect' table."""
    lines = [f"{'overlap days':>13}{'min detectable rho':>21}{'verdict at rho = 0.30':>24}"]
    for n in days:
        mdr = min_detectable_rho(n)
        verdict = "conclusive" if mdr <= REAL_THRESHOLD else "cannot resolve it"
        lines.append(f"{n:13d}{mdr:21.3f}{verdict:>24}")
    return "\n".join(lines)


def requirements(rhos=(0.2, 0.3, 0.4, 0.5)) -> str:
    """The 'how long must I collect' table."""
    lines = [f"{'true rho':>9}{'days (median)':>15}{'days (80% power)':>18}"]
    for rho in rhos:
        lines.append(f"{rho:9.2f}{days_required(rho):15d}{days_for_power(rho):18d}")
    return "\n".join(lines)
