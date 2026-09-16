"""Phase 2.1: measure the two channels' noise instead of guessing their weights.

The design is a plain test-retest: two Mindfulness sessions about five minutes
apart, seated, same posture, on ten separate days. Within a pair the true
physiological state is taken to be unchanged, so the whole within-pair spread is
measurement error.

For a pair (a, b) of the *log* values, d = a - b has variance 2 * sigma_eps^2,
so

    sigma_eps^2 = mean(d^2) / 2

over the pairs. Working in logs is what makes this a single number: RMSSD's
absolute error grows with its level, but its *proportional* error is roughly
constant, and a log-scale error variance is the proportional one.

The interval on sigma^2 comes from a bootstrap over pairs rather than a
chi-square quantile -- ten pairs is few enough that the normality assumption
behind the chi-square interval is doing real work, and resampling needs no table.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import DEFAULT, ScoreConfig
from .score import WindowScore


@dataclass(frozen=True)
class Calibration:
    sigma2_x: float
    sigma2_h: float
    n_pairs: int
    ci_x: tuple[float, float]
    ci_h: tuple[float, float]
    ratio_ci: tuple[float, float]

    @property
    def sigma_x_pct(self) -> float:
        """Measurement sd of RMSSD as a percentage of its level.

        For small sigma on the log scale, sd(ln v) ~ sd(v)/v, so this is
        directly comparable to the ~29% MAPE the validation literature reports.
        """
        return 100.0 * math.sqrt(self.sigma2_x)

    @property
    def sigma_h_pct(self) -> float:
        return 100.0 * math.sqrt(self.sigma2_h)

    def weights(self, cfg: ScoreConfig = DEFAULT) -> tuple[float, float]:
        return cfg.with_weights(self.sigma2_x, self.sigma2_h).weights()

    @property
    def precision_ratio(self) -> float:
        """w_h / w_x at lambda = 1: how much more informative HR is per sample."""
        return self.sigma2_x / self.sigma2_h

    def report(self, cfg: ScoreConfig = DEFAULT) -> str:
        w_x, w_h = self.weights(cfg)
        total = w_x + w_h
        more, less = ("heart rate", "HRV") if w_h > w_x else ("HRV", "heart rate")
        lines = [
            f"test-retest pairs:        {self.n_pairs}",
            f"sigma^2_eps ln(RMSSD):    {self.sigma2_x:.5f}  "
            f"(95% CI {self.ci_x[0]:.5f}-{self.ci_x[1]:.5f}; ~{self.sigma_x_pct:.1f}% of level)",
            f"sigma^2_eps ln(HR):       {self.sigma2_h:.5f}  "
            f"(95% CI {self.ci_h[0]:.5f}-{self.ci_h[1]:.5f}; ~{self.sigma_h_pct:.1f}% of level)",
            f"precision ratio w_h/w_x:  {self.precision_ratio:.2f}  "
            f"(95% CI {self.ratio_ci[0]:.2f}-{self.ratio_ci[1]:.2f})",
            f"normalised weights:       HRV {w_x / total:.3f}   HR {w_h / total:.3f}"
            f"   (lambda = {cfg.lambda_hrv:g})",
            f"=> your wrist measures {more} more reliably than {less}.",
        ]
        if self.n_pairs < 10:
            lines.append(
                f"WARNING: {self.n_pairs} pairs is below the 10 the plan asks for; "
                "the interval on the ratio is wide and the weights will move."
            )
        return "\n".join(lines)


def find_pairs(
    scores: list[WindowScore],
    *,
    max_gap_minutes: float = 15.0,
    min_gap_minutes: float = 1.0,
    on_demand_only: bool = True,
) -> list[tuple[WindowScore, WindowScore]]:
    """Pair up back-to-back windows that look like a test-retest sitting.

    Greedy and non-overlapping: each window joins at most one pair, so a run of
    three sessions yields one pair, not two correlated ones. `min_gap_minutes`
    rejects two readings that are really one session double-counted.
    """
    usable = sorted(
        (s for s in scores if s.metrics.usable and (s.on_demand or not on_demand_only)),
        key=lambda s: s.start,
    )
    pairs: list[tuple[WindowScore, WindowScore]] = []
    i = 0
    while i < len(usable) - 1:
        a, b = usable[i], usable[i + 1]
        gap = (b.start - a.start).total_seconds() / 60.0
        if min_gap_minutes <= gap <= max_gap_minutes:
            pairs.append((a, b))
            i += 2
        else:
            i += 1
    return pairs


def sigma2_from_pairs(a, b) -> float:
    """Within-pair error variance: mean(d^2) / 2 over paired log values."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired arrays must have the same shape")
    d = a - b
    d = d[np.isfinite(d)]
    if d.size == 0:
        return float("nan")
    return float(np.mean(d * d) / 2.0)


def _bootstrap(
    dx: np.ndarray, dh: np.ndarray, n_boot: int, seed: int
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    rng = np.random.default_rng(seed)
    n = dx.size
    sx = np.empty(n_boot)
    sh = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)  # resample pairs, keeping channels together
        sx[i] = np.mean(dx[idx] ** 2) / 2.0
        sh[i] = np.mean(dh[idx] ** 2) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(sh > 0, sx / sh, np.nan)
    ratio = ratio[np.isfinite(ratio)]
    pct = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    ratio_ci = pct(ratio) if ratio.size else (float("nan"), float("nan"))
    return pct(sx), pct(sh), ratio_ci


def calibrate(
    pairs: list[tuple[WindowScore, WindowScore]],
    *,
    n_boot: int = 10000,
    seed: int = 20260916,
) -> Calibration:
    """Estimate both measurement-error variances from paired windows."""
    if len(pairs) < 2:
        raise ValueError("need at least 2 test-retest pairs")
    dx = np.array([a.x - b.x for a, b in pairs], dtype=float)
    dh = np.array([a.h - b.h for a, b in pairs], dtype=float)
    sigma2_x = float(np.mean(dx * dx) / 2.0)
    sigma2_h = float(np.mean(dh * dh) / 2.0)
    ci_x, ci_h, ratio_ci = _bootstrap(dx, dh, n_boot, seed)
    return Calibration(sigma2_x, sigma2_h, len(pairs), ci_x, ci_h, ratio_ci)
