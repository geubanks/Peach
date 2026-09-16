"""Does the answer survive the choices you made arbitrarily?

Two jobs.

**Sensitivity.** Several constants in Section 3 were picked by judgement rather
than measurement: the 28-day baseline, the 20% artifact threshold, the 30-beat
minimum, λ. If the Phase 2.2 verdict flips when the baseline is 21 days instead
of 28, the verdict is about the knob, not about you. Sweeping them is cheap and
it is the only way to know.

**λ, and the trap in tuning it.** Step 5 says: start at λ = 1, and "if you want
the physiological prior back, add a single multiplier λ on w_x and tune λ against
your diary". Phase 5 repeats it: "re-fit λ only if the on-wrist correlation is
worse than the offline one."

Tuning λ against the diary and then reporting the resulting ρ as evidence that
the score works is circular. You chose the parameter that maximises ρ on exactly
the data you then use to report ρ. The number that comes out is optimistic, and
the optimism grows with how fine your λ grid is and shrinks with how many days
you have — at the sample sizes here (46 days is the target, per `power.py`), it
is not a rounding error.

`tune_lambda` therefore reports two numbers: the in-sample ρ at the best λ, and
a leave-one-day-out cross-validated ρ in which each day's score is computed with
a λ chosen *without* that day. The gap between them is the optimism, measured
rather than argued about. Report the cross-validated one.

Nothing here changes the score. It tells you how much to trust it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from . import diary as diary_mod
from . import power
from .config import DEFAULT, ScoreConfig
from .score import combine, daily_scores, score_windows

DEFAULT_LAMBDAS = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)


@dataclass(frozen=True)
class Point:
    """One setting of one knob, and what the falsification test said under it."""

    label: str
    value: object
    rho: float
    lo: float
    hi: float
    n: int
    verdict: str

    @property
    def conclusive(self) -> bool:
        return self.verdict != "underpowered"


def _rho_for_daily(daily, ratings: dict) -> tuple[float, int, np.ndarray, np.ndarray]:
    paired = [(d.mean, ratings[d.day]) for d in daily if d.day in ratings]
    if len(paired) < 3:
        return float("nan"), len(paired), np.empty(0), np.empty(0)
    xs = np.array([p[0] for p in paired])
    ys = np.array([p[1] for p in paired])
    return diary_mod.spearman(xs, ys), len(paired), xs, ys


def _point(label, value, daily, ratings) -> Point:
    rho, n, _, _ = _rho_for_daily(daily, ratings)
    lo, hi = power.fisher_ci(rho, n)
    verdict = diary_mod.classify(rho, (lo, hi), n)[0] if math.isfinite(rho) else "underpowered"
    return Point(label, value, rho, lo, hi, n, verdict)


def lambda_curve(windows, entries, cfg: ScoreConfig = DEFAULT,
                 lambdas=DEFAULT_LAMBDAS) -> list[Point]:
    """ρ as a function of λ, computed without rescoring for each value.

    λ enters only at Step 5, so the windows are scored once and the two channels
    are re-combined per λ. That is exact, not an approximation, and it makes a
    nine-point sweep as cheap as a single run.
    """
    ratings = diary_mod.daily_means(entries)
    base = score_windows(windows, cfg)
    return [_point("lambda", lam, _daily_for_lambda(base, cfg, lam), ratings)
            for lam in lambdas]


def _daily_for_lambda(scores, cfg: ScoreConfig, lam: float):
    """Re-combine the two channels at this lambda and re-aggregate.

    Mutates `scores[*].s` in place, which is safe here because every caller
    recomputes it for the next lambda and never reads the old value. Do not hold
    a reference to `scores` across a call expecting the original scoring.
    """
    tuned = replace(cfg, lambda_hrv=lam)
    for s in scores:
        if math.isfinite(s.z_x) and math.isfinite(s.z_h):
            s.s = combine(s.z_x, s.z_h, tuned)
    return daily_scores(scores, tuned)


def sweep(windows, entries, cfg: ScoreConfig = DEFAULT, *,
          param: str, values) -> list[Point]:
    """ρ under each value of one config field, rescoring from scratch each time."""
    ratings = diary_mod.daily_means(entries)
    out = []
    for value in values:
        trial = replace(cfg, **{param: value})
        daily = daily_scores(score_windows(windows, trial), trial)
        out.append(_point(param, value, daily, ratings))
    return out


@dataclass(frozen=True)
class LambdaTuning:
    """The honest and the optimistic answer, side by side."""

    best_lambda: float
    rho_in_sample: float
    rho_cross_validated: float
    n_days: int
    grid: tuple

    @property
    def optimism(self) -> float:
        """How much tuning inflated the in-sample number."""
        return self.rho_in_sample - self.rho_cross_validated

    def report(self) -> str:
        lo, hi = power.fisher_ci(self.rho_cross_validated, self.n_days)
        lines = [
            f"grid:                 {', '.join(f'{v:g}' for v in self.grid)}",
            f"best lambda:          {self.best_lambda:g}",
            f"rho, in-sample:       {self.rho_in_sample:+.3f}   <- do NOT report this",
            f"rho, cross-validated: {self.rho_cross_validated:+.3f}   "
            f"(95% CI {lo:+.3f} to {hi:+.3f}) over {self.n_days} days",
            f"optimism from tuning: {self.optimism:+.3f}",
        ]
        if self.optimism > 0.05:
            lines.append(
                "The gap is real. Choosing lambda to maximise rho on the same days you "
                "then report rho for is circular; the cross-validated number is the one "
                "that means something."
            )
        if abs(self.best_lambda - 1.0) < 1e-9:
            lines.append("lambda = 1 won: the data do not ask for the physiological prior back.")
        return "\n".join(lines)


def tune_lambda(windows, entries, cfg: ScoreConfig = DEFAULT,
                lambdas=DEFAULT_LAMBDAS) -> LambdaTuning:
    """Pick λ, and measure how much picking it inflated the result.

    Cross-validation is leave-one-day-out: for each day, λ is chosen to maximise
    ρ over all the *other* days, that day's score is computed with it, and the
    final ρ is taken over the held-out predictions. A day never contributes to
    the choice of the λ used to score it.
    """
    ratings = diary_mod.daily_means(entries)
    scores = score_windows(windows, cfg)

    # Daily means per lambda, computed once.
    per_lambda = {}
    for lam in lambdas:
        daily = _daily_for_lambda(scores, cfg, lam)
        per_lambda[lam] = {d.day: d.mean for d in daily if d.day in ratings}

    days = sorted(set().union(*(set(v) for v in per_lambda.values()))) if per_lambda else []
    days = [d for d in days if all(d in per_lambda[lam] for lam in lambdas)]
    if len(days) < 4:
        return LambdaTuning(cfg.lambda_hrv, float("nan"), float("nan"), len(days), tuple(lambdas))

    truth = np.array([ratings[d] for d in days])

    def rho_for(lam, mask=None):
        values = np.array([per_lambda[lam][d] for d in days])
        if mask is not None:
            values, target = values[mask], truth[mask]
        else:
            target = truth
        return diary_mod.spearman(values, target)

    def pick_best(mask=None) -> float:
        scored = [(rho_for(lam, mask), lam) for lam in lambdas]
        scored = [(r if math.isfinite(r) else -2.0, lam) for r, lam in scored]
        # max() keeps the first of any tie, and lambdas are given in ascending
        # order, so a tie resolves to the least aggressive setting.
        return max(scored)[1]

    best = pick_best()
    rho_in = rho_for(best)

    held_out = np.empty(len(days))
    for i in range(len(days)):
        mask = np.ones(len(days), dtype=bool)
        mask[i] = False
        held_out[i] = per_lambda[pick_best(mask)][days[i]]
    rho_cv = diary_mod.spearman(held_out, truth)

    return LambdaTuning(best, rho_in, rho_cv, len(days), tuple(lambdas))


def report(points: list[Point], *, knob: str) -> str:
    """A table, plus whether the verdict ever changed."""
    lines = [f"{knob:>14}{'rho':>9}{'95% CI':>22}{'n':>5}  verdict"]
    for p in points:
        value = f"{p.value:g}" if isinstance(p.value, (int, float)) else str(p.value)
        ci = f"[{p.lo:+.2f}, {p.hi:+.2f}]" if math.isfinite(p.lo) else "[   n/a   ]"
        lines.append(f"{value:>14}{p.rho:+9.3f}{ci:>22}{p.n:>5}  {p.verdict}")

    verdicts = {p.verdict for p in points}
    rhos = [p.rho for p in points if math.isfinite(p.rho)]
    lines.append("")
    if len(verdicts) > 1:
        lines.append(f"WARNING: the verdict CHANGES across this knob ({', '.join(sorted(verdicts))}).")
        lines.append("A conclusion that depends on a constant you picked by judgement is not")
        lines.append("yet a conclusion. Prefer the setting you can defend, and say you checked.")
    elif rhos:
        lines.append(f"Verdict is stable ({verdicts.pop()}) across the whole range; "
                     f"rho spans {min(rhos):+.3f} to {max(rhos):+.3f}.")
    return "\n".join(lines)
