"""What state is this project in, and what is the next thing to do?

The plan has real gating (Section 6) and real prerequisites, and they are easy
to lose track of across the weeks the diary takes. This is the single command
that reads whatever you have so far and answers "what now" — and, just as
importantly, tells you when the answer is "nothing, wait".

It never scores anything you have not already got. It is a status report, not a
step.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np

from . import cosinor, power
from . import diary as diary_mod
from .config import ScoreConfig
from .score import daily_scores, score_windows

PASS, WARN, TODO, FAIL = "PASS", "WARN", "TODO", "FAIL"
ORDER = {FAIL: 0, TODO: 1, WARN: 2, PASS: 3}


@dataclass(frozen=True)
class Check:
    phase: str
    name: str
    status: str
    detail: str
    action: str | None = None


def _check_windows(scores, cfg) -> list[Check]:
    out = []
    usable = [s for s in scores if s.metrics.usable]
    if not scores:
        return [Check("1.1", "windows", FAIL, "no windows in the file",
                      "export Health data and run `tone parse`")]

    frac = 100.0 * len(usable) / len(scores)
    status = PASS if frac > 80 else (WARN if frac > 50 else FAIL)
    detail = f"{len(usable)} usable of {len(scores)} ({frac:.0f}%)"
    action = None
    if status != PASS:
        reasons = Counter(s.flags[0] for s in scores if not s.metrics.usable and s.flags)
        detail += "; " + ", ".join(f"{n}x {r}" for r, n in reasons.most_common(3))
        action = ("check the reconstruction: `tone parse` compares our SDNN against "
                  "Apple's, and a bad parse looks exactly like bad data")
    out.append(Check("1.1", "usable windows", status, detail, action))

    if not usable:
        return out

    days = {s.day for s in usable}
    span = (max(days) - min(days)).days + 1
    per_day = len(usable) / max(1, len(days))
    out.append(Check("1.1", "coverage", PASS if len(days) >= 30 else TODO,
                     f"{len(days)} days with data over a {span}-day span, "
                     f"{per_day:.1f} usable windows/day",
                     None if len(days) >= 30 else "keep wearing it; more days is the only fix"))

    scored = [s for s in scores if s.scored]
    if not scored:
        out.append(Check("1.2", "scoring", TODO,
                         f"nothing scoreable yet (needs {cfg.min_scoring_windows} windows "
                         "of history before the first score)",
                         "more days"))
    else:
        out.append(Check("1.2", "scoring", PASS,
                         f"{len(scored)} scored windows on {len({s.day for s in scored})} days"))

    hours = np.array([s.hour for s in usable])
    fit = cosinor.fit(hours, np.array([s.x for s in usable]),
                      min_distinct_hours=cfg.min_baseline_hours)
    if fit.flat:
        out.append(Check("1.2", "circadian baseline", WARN,
                         "too few distinct clock hours to fit a rhythm",
                         "the watch is only sampling at a couple of times of day"))
    elif fit.r2 < 0.10:
        out.append(Check("1.2", "circadian baseline", WARN,
                         f"cosinor explains only {100 * fit.r2:.1f}% of ln RMSSD variance",
                         "Section 7 says this is the case where a flat baseline is the "
                         "honest simplification. Surprising, but cheap to accept."))
    else:
        out.append(Check("1.2", "circadian baseline", PASS,
                         f"cosinor explains {100 * fit.r2:.1f}% of ln RMSSD variance, "
                         f"peak at {fit.acrophase_hours:04.1f}h"))
    return out


def _check_weights(scores, cfg) -> list[Check]:
    from . import calibrate as calib

    pairs = calib.find_pairs(scores)
    if cfg.weights_measured:
        w_x, w_h = cfg.weights()
        total = w_x + w_h
        return [Check("2.1", "measured weights", PASS,
                      f"HRV {w_x / total:.3f} / HR {w_h / total:.3f} "
                      f"(lambda = {cfg.lambda_hrv:g})")]
    detail = f"not measured; {len(pairs)} test-retest pair(s) found in the data"
    return [Check("2.1", "measured weights", TODO, detail,
                  "two Mindfulness sessions five minutes apart, seated, on ten separate "
                  "days, then `tone weights --out data/config.json`"
                  if len(pairs) < 10 else
                  "you have the pairs -- run `tone weights --out data/config.json`")]


def _check_diary(entries, daily, cfg) -> list[Check]:
    if not entries:
        return [Check("1.3", "diary", FAIL, "no diary",
                      "start it TODAY -- it has the longest lead time in the plan "
                      "and gates everything downstream (see docs/DIARY.md)")]

    days = {e.day for e in entries}
    ratings = np.array([e.rating for e in entries], dtype=float)
    sd = float(ratings.std(ddof=1)) if ratings.size > 1 else 0.0
    out = [Check("1.3", "diary", PASS if len(days) >= 14 else TODO,
                 f"{len(entries)} entries over {len(days)} days, "
                 f"{len(entries) / max(1, len(days)):.1f}/day, ratings sd {sd:.2f}",
                 None if len(days) >= 14 else "keep going to at least 14 consecutive days")]

    if ratings.size > 1 and sd < 0.5:
        out.append(Check("1.3", "diary variance", WARN,
                         f"ratings barely vary (sd {sd:.2f})",
                         "if Phase 2.2 comes back negative, this is the first suspect, "
                         "not the sensor. Rate the moment, not the day."))

    scored_days = {d.day for d in daily}
    overlap = len(scored_days & days)
    detectable = power.min_detectable_rho(overlap) if overlap > 3 else float("nan")
    if overlap < 4:
        out.append(Check("2.2", "overlap", TODO,
                         f"{overlap} day(s) have both a score and a rating",
                         "nothing can be tested until these overlap"))
    elif math.isfinite(detectable) and detectable > power.REAL_THRESHOLD:
        need = power.days_required(power.REAL_THRESHOLD) - overlap
        out.append(Check("2.2", "statistical power", TODO,
                         f"{overlap} overlapping days; smallest detectable rho is "
                         f"{detectable:.2f}, above the {power.REAL_THRESHOLD:.2f} threshold",
                         f"about {need} more overlapping days before the falsification test "
                         "can distinguish a real signal from nothing. Do NOT run it and "
                         "read the answer yet -- see `tone power`."))
    else:
        out.append(Check("2.2", "statistical power", PASS,
                         f"{overlap} overlapping days; can detect rho down to "
                         f"{detectable:.2f}",
                         "you are ready to run `tone validate`"))
    return out


def run(scores, daily, entries, cfg: ScoreConfig) -> list[Check]:
    checks = _check_windows(scores, cfg)
    checks += _check_weights(scores, cfg)
    checks += _check_diary(entries, daily, cfg)
    return checks


def next_action(checks: list[Check]) -> str:
    """The single most important thing to do next."""
    blocking = [c for c in checks if c.status in (FAIL, TODO) and c.action]
    if not blocking:
        return ("Nothing is blocking. Run `tone validate`, then `tone sensitivity`, and "
                "only then freeze the spec.")
    worst = min(blocking, key=lambda c: (ORDER[c.status], c.phase))
    return f"Phase {worst.phase} -- {worst.action}"


def report(checks: list[Check]) -> str:
    width = max(len(c.name) for c in checks)
    lines = [f"{'phase':<7}{'check':<{width + 2}}{'':<6}detail"]
    for c in checks:
        lines.append(f"{c.phase:<7}{c.name:<{width + 2}}{c.status:<6}{c.detail}")
        if c.action:
            lines.append(f"{'':<7}{'':<{width + 2}}{'':<6}-> {c.action}")
    return "\n".join(lines)


def analyse(windows, entries, cfg: ScoreConfig):
    """Score what exists and return (checks, next action)."""
    scores = score_windows(windows, cfg)
    daily = daily_scores(scores, cfg)
    checks = run(scores, daily, entries, cfg)
    return checks, next_action(checks)


def parse_diary(path):
    return diary_mod.read_diary(path) if path else []
