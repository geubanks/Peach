"""Phase 2.2: the falsification test.

The diary is the only ground truth in the project, and this module is the only
place where the score can be told it is wrong. Everything is within-person and
non-parametric: Spearman's rho between the daily score and the day's mean
self-rating, with a percentile bootstrap interval over days.

Why the bootstrap resamples *days* and not windows: consecutive windows within
a day share a baseline, a posture and a mood, so they are not independent.
Resampling days keeps the dependence inside the resampled unit, which is the
standard cluster bootstrap.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

from . import power

TIME_KEYS = ("timestamp", "time", "datetime", "date", "when")
RATING_KEYS = ("rating", "stress", "score", "value", "level")


@dataclass(frozen=True)
class DiaryEntry:
    when: datetime
    rating: float

    @property
    def day(self) -> date:
        return self.when.date()


@dataclass(frozen=True)
class Validation:
    """The Phase 2.2 verdict, and the evidence behind it."""

    rho: float
    ci: tuple[float, float]
    n: int
    verdict: str
    note: str
    days_to_resolve: int = 0

    @property
    def conclusive(self) -> bool:
        """False when the data settle nothing either way."""
        return self.verdict != "underpowered"

    def report(self, label: str = "daily") -> str:
        return (
            f"{label}: n = {self.n}, Spearman rho = {self.rho:+.3f} "
            f"(95% CI {self.ci[0]:+.3f} to {self.ci[1]:+.3f})\n"
            f"verdict: {self.verdict.upper()} -- {self.note}"
        )


def _parse_time(text: str) -> datetime:
    text = text.strip()
    for fmt in (
        "%Y-%m-%d %H:%M:%S %z",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y/%m/%d %H:%M",
        "%m/%d/%Y %H:%M",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return datetime.fromisoformat(text)  # raises with a useful message


def read_diary(path: str | Path) -> list[DiaryEntry]:
    """Read the Shortcuts CSV: a timestamp column and a 0-10 rating column.

    Header names are matched loosely (see TIME_KEYS / RATING_KEYS); a headerless
    two-column file is accepted as (timestamp, rating). Rows that do not parse
    are skipped rather than aborting the run -- a phone automation will
    eventually write one bad line and it should not cost you the analysis.
    """
    rows = list(csv.reader(Path(path).read_text().splitlines()))
    rows = [r for r in rows if r and any(c.strip() for c in r)]
    if not rows:
        return []

    header = [c.strip().lower() for c in rows[0]]
    t_idx = r_idx = None
    for i, name in enumerate(header):
        if t_idx is None and any(k in name for k in TIME_KEYS):
            t_idx = i
        if r_idx is None and any(k in name for k in RATING_KEYS):
            r_idx = i
    if t_idx is None or r_idx is None:
        t_idx, r_idx, body = 0, 1, rows  # headerless
    else:
        body = rows[1:]

    entries: list[DiaryEntry] = []
    for row in body:
        if len(row) <= max(t_idx, r_idx):
            continue
        try:
            entries.append(DiaryEntry(_parse_time(row[t_idx]), float(row[r_idx])))
        except (ValueError, TypeError):
            continue
    entries.sort(key=lambda e: e.when)
    return entries


def daily_means(entries: list[DiaryEntry]) -> dict[date, float]:
    by_day: dict[date, list[float]] = {}
    for e in entries:
        by_day.setdefault(e.day, []).append(e.rating)
    return {d: float(np.mean(v)) for d, v in by_day.items()}


def rankdata(values) -> np.ndarray:
    """Average ranks, ties shared -- the ranking Spearman's rho requires."""
    v = np.asarray(values, dtype=float)
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(v.size, dtype=float)
    ranks[order] = np.arange(1, v.size + 1, dtype=float)
    sorted_v = v[order]
    i = 0
    while i < v.size:
        j = i
        while j + 1 < v.size and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = np.mean(ranks[order[i:j + 1]])
        i = j + 1
    return ranks


def spearman(x, y) -> float:
    """Spearman's rho: Pearson correlation of the average ranks."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size != y.size:
        raise ValueError("x and y must have the same length")
    if x.size < 3:
        return float("nan")
    rx, ry = rankdata(x), rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = math.sqrt(float(np.dot(rx, rx)) * float(np.dot(ry, ry)))
    if denom <= 0:
        return float("nan")  # one side is entirely tied: no ranking to correlate
    return float(np.dot(rx, ry) / denom)


def bootstrap_rho(
    x, y, *, n_boot: int = 10000, seed: int = 20260916
) -> tuple[float, tuple[float, float]]:
    """Point estimate and percentile bootstrap CI for rho."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rho = spearman(x, y)
    n = x.size
    if n < 3 or not math.isfinite(rho):
        return rho, (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    draws.fill(np.nan)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        draws[i] = spearman(x[idx], y[idx])
    draws = draws[np.isfinite(draws)]
    if draws.size < n_boot // 2:
        return rho, (float("nan"), float("nan"))
    return rho, (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))


def classify(rho: float, ci: tuple[float, float], n: int) -> tuple[str, str, int]:
    """The Phase 2.2 decision box, with one verdict the plan does not have.

    The plan reads any interval spanning zero as "rebuild", and "rebuild" routes
    to *do not write Swift*. That conflates two situations that call for
    opposite responses:

      - the interval spans zero and also spans 0.30, the plan's own threshold
        for "real". The data are consistent with no effect AND with exactly the
        effect you are looking for. Nothing has been learned. The response is to
        collect more days, not to stop.
      - the interval spans zero but sits entirely below 0.30. You have now ruled
        out an effect as large as the one you said you cared about. That is a
        real negative result and "rebuild" is the right call.

    Only the second is a rebuild. The first is `underpowered`, and at the sample
    sizes the plan schedules it is the *likely* outcome of a real signal: at 14
    days nothing below rho = 0.54 can clear zero. See tone/power.py.

    A significantly negative rho is its own case, and a loud one -- it usually
    means a sign error in Step 5 or a diary scale entered backwards, not a
    discovery about your physiology.

    Returns (verdict, note, days_to_resolve).
    """
    lo, hi = ci
    if not math.isfinite(rho) or not math.isfinite(lo) or not math.isfinite(hi):
        return "underpowered", "not enough data to estimate rho at all.", -1

    if hi < 0.0:
        return "rebuild", (
            "rho is significantly NEGATIVE. The score is anti-correlated with your "
            "ratings -- look for a sign error in Step 5 or a diary scale entered "
            "backwards before concluding anything physiological."
        ), 0

    if lo <= 0.0 <= hi:
        if hi >= power.REAL_THRESHOLD:
            need = power.days_required(rho) if rho > 0 else -1
            more = max(0, need - n) if need > 0 else -1
            extra = (f" About {more} more overlapping days would resolve it at this rho"
                     if more > 0 else " Keep collecting")
            return "underpowered", (
                f"the interval spans both zero and rho = {power.REAL_THRESHOLD:.2f}, so it is "
                "consistent with no effect and with the effect you are looking for. This is "
                "not evidence against the score; it is not yet evidence about it."
                f"{extra}. Check the diary too: if your ratings barely vary day to day, more "
                "days will not help."
            ), more
        return "rebuild", (
            f"the interval spans zero and sits entirely below rho = {power.REAL_THRESHOLD:.2f}. "
            "You have ruled out an effect as large as the one you set out to find -- a real "
            "negative result, not a sample-size problem. The diary is still the first thing "
            "to check before concluding it is the sensor."
        ), 0

    if rho >= 0.3:
        return "real", "the signal is real. Proceed to Phase 3 on your current watch.", 0
    if rho >= 0.1:
        return "sample-starved", (
            "real but sample-starved. Commit to two scheduled Mindfulness sessions a "
            "day and re-test; this is the one scenario that would justify a Series 12."
        ), 0
    return "rebuild", (
        f"rho = {rho:.3f} excludes zero but is too small to act on (n = {n}). "
        "Treat as rebuild."
    ), 0


def validate_daily(daily, diary: list[DiaryEntry], *, n_boot: int = 10000, seed: int = 20260916) -> Validation:
    """Correlate each day's mean score against that day's mean diary rating."""
    ratings = daily_means(diary)
    paired = [(d.mean, ratings[d.day]) for d in daily if d.day in ratings]
    if len(paired) < 3:
        return Validation(float("nan"), (float("nan"), float("nan")), len(paired),
                          "underpowered",
                          "fewer than 3 days overlap between scores and diary.", -1)
    xs = np.array([p[0] for p in paired])
    ys = np.array([p[1] for p in paired])
    rho, ci = bootstrap_rho(xs, ys, n_boot=n_boot, seed=seed)
    verdict, note, more = classify(rho, ci, len(paired))
    return Validation(rho, ci, len(paired), verdict, note, more)


def validate_on_demand(
    scores,
    diary: list[DiaryEntry],
    *,
    window_minutes: float = 30.0,
    n_boot: int = 10000,
    seed: int = 20260916,
) -> Validation:
    """Correlate each deliberate spot check against the nearest diary entry.

    This is the stronger test of the two: the daily mean asks whether the score
    tracks a whole day's mood, while a spot check taken minutes from a rating
    asks whether it tracks *this moment*. It is also the test that still means
    something if Phase 2.2 comes back sample-starved, because the spot check is
    the one measurement you control the timing of.
    """
    if not diary:
        return Validation(float("nan"), (float("nan"), float("nan")), 0,
                          "underpowered", "empty diary.", -1)
    limit = timedelta(minutes=window_minutes)
    times = [e.when for e in diary]
    paired: list[tuple[float, float]] = []
    for s in scores:
        if not (s.scored and s.on_demand):
            continue
        best, best_gap = None, limit
        for e, t in zip(diary, times):
            if t.tzinfo is None and s.start.tzinfo is not None:
                t = t.replace(tzinfo=s.start.tzinfo)
            gap = abs(t - s.start)
            if gap <= best_gap:
                best, best_gap = e, gap
        if best is not None:
            paired.append((s.s, best.rating))
    if len(paired) < 3:
        return Validation(float("nan"), (float("nan"), float("nan")), len(paired),
                          "underpowered", f"only {len(paired)} spot checks fell within "
                                          f"{window_minutes:g} min of a diary entry.", -1)
    xs = np.array([p[0] for p in paired])
    ys = np.array([p[1] for p in paired])
    rho, ci = bootstrap_rho(xs, ys, n_boot=n_boot, seed=seed)
    verdict, note, more = classify(rho, ci, len(paired))
    return Validation(rho, ci, len(paired), verdict, note, more)
