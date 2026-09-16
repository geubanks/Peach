"""Optional matplotlib figures. Import is deferred so numpy-only installs work.

The first figure is the one Phase 1.2 is graded on: what the baseline looks like
with the cosinor and without it. The point is visible in a second -- a flat mean
puts every sleep window above the line and every afternoon window below it, so
the residual carries the time of day rather than the stress. The cosinor removes
that structure, and what is left is deviation from your own usual level *at that
hour*.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import cosinor
from .config import DEFAULT, ScoreConfig


def _pyplot():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "matplotlib is not installed. `pip install matplotlib` or skip `tone plot`."
        ) from exc
    return plt


def baseline_comparison(scores, path: str | Path, cfg: ScoreConfig = DEFAULT) -> Path:
    """Two panels: ln RMSSD against clock hour, and the residuals each way."""
    plt = _pyplot()
    usable = [s for s in scores if s.metrics.usable]
    if len(usable) < 10:
        raise SystemExit("need at least 10 usable windows to plot a baseline")

    hours = np.array([s.hour for s in usable])
    x = np.array([s.x for s in usable])
    fit = cosinor.fit(hours, x, min_distinct_hours=cfg.min_baseline_hours)
    grid = np.linspace(0, 24, 241)

    fig, axes = plt.subplots(2, 1, figsize=(9, 8))
    ax = axes[0]
    ax.scatter(hours, x, s=14, alpha=0.5, label="windows")
    ax.plot(grid, fit.predict(grid), lw=2, label=f"cosinor (R2 = {fit.r2:.2f})")
    ax.axhline(float(np.mean(x)), ls="--", lw=2, color="grey", label="flat 28-day mean")
    ax.set_xlabel("clock hour")
    ax.set_ylabel("ln RMSSD")
    ax.set_title("Baseline: with and without the circadian term")
    ax.set_xticks(range(0, 25, 3))
    ax.legend(loc="best", fontsize=9)

    # Hourly means, not raw scatter: overlaid point clouds hide exactly the
    # effect this panel exists to show. Binning makes it unmissable -- the flat
    # baseline's residual still traces the clock, the cosinor's sits on zero.
    ax = axes[1]
    bins = np.arange(0, 25, 2.0)
    centres, flat_mean, cos_mean, flat_sem, cos_sem = [], [], [], [], []
    resid_flat = x - float(np.mean(x))
    resid_cos = x - fit.predict(hours)
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (hours >= lo) & (hours < hi)
        if sel.sum() < 3:
            continue
        centres.append((lo + hi) / 2)
        flat_mean.append(resid_flat[sel].mean())
        cos_mean.append(resid_cos[sel].mean())
        flat_sem.append(resid_flat[sel].std(ddof=1) / np.sqrt(sel.sum()))
        cos_sem.append(resid_cos[sel].std(ddof=1) / np.sqrt(sel.sum()))
    ax.errorbar(centres, flat_mean, yerr=flat_sem, fmt="-o", color="grey",
                capsize=3, label="flat baseline")
    ax.errorbar(centres, cos_mean, yerr=cos_sem, fmt="-o", capsize=3,
                label="cosinor baseline")
    ax.axhline(0.0, color="black", lw=1)
    ax.set_xlabel("clock hour")
    ax.set_ylabel("mean residual ln RMSSD (2 h bins)")
    ax.set_title("A flat baseline leaves the time of day in the residual")
    ax.set_xticks(range(0, 25, 3))
    ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    out = Path(path)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def score_timeline(scores, daily, path: str | Path, diary=None) -> Path:
    """Z_x and Z_h over time, with the daily score and its interval beneath."""
    plt = _pyplot()
    scored = [s for s in scores if s.scored]
    if not scored:
        raise SystemExit("no scored windows to plot (still warming up?)")

    t = [s.start for s in scored]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    ax = axes[0]
    ax.plot(t, [-s.z_x for s in scored], ".", ms=5, alpha=0.7, label="-Z(ln RMSSD)")
    ax.plot(t, [s.z_h for s in scored], ".", ms=5, alpha=0.7, label="Z(ln HR)")
    ax.axhline(0, color="black", lw=1)
    ax.set_ylabel("robust z")
    ax.set_title("Channels (both oriented so up = more stressed)")
    ax.legend(loc="best", fontsize=9)

    ax = axes[1]
    days = [d.day for d in daily]
    means = np.array([d.mean for d in daily])
    lo = np.array([d.lo for d in daily])
    hi = np.array([d.hi for d in daily])
    ax.fill_between(days, lo, hi, alpha=0.25, label="95% interval")
    ax.plot(days, means, "-o", ms=4, label="daily score")
    ax.axhline(0, color="black", lw=1)
    ax.set_ylabel("S (daily mean)")

    if diary:
        from .diary import daily_means
        ratings = daily_means(diary)
        rx = [d for d in days if d in ratings]
        if rx:
            twin = ax.twinx()
            twin.plot(rx, [ratings[d] for d in rx], color="firebrick", alpha=0.6, lw=1.2,
                      label="diary rating")
            twin.set_ylabel("diary rating (0-10)", color="firebrick")
    ax.legend(loc="best", fontsize=9)

    fig.autofmt_xdate()
    fig.tight_layout()
    out = Path(path)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def clock_histogram(scores, path: str | Path) -> Path:
    """When your watch actually samples you -- the Phase 1.1 deliverable."""
    plt = _pyplot()
    usable = [s for s in scores if s.metrics.usable]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist([s.hour for s in usable], bins=24, range=(0, 24), edgecolor="white")
    ax.set_xlabel("clock hour")
    ax.set_ylabel("usable windows")
    ax.set_title(f"Sampling times, {len(usable)} usable windows")
    ax.set_xticks(range(0, 25, 3))
    fig.tight_layout()
    out = Path(path)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out
