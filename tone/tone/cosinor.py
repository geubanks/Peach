"""First-harmonic cosinor fit (Step 3).

    y_hat(t) = M + A cos(2 pi t / 24) + B sin(2 pi t / 24)

`t` is *local clock hour* in [0, 24). Local, not UTC: the rhythm is entrained to
your day, so a flight or a DST change should move the clock, not the body.

The fit is ordinary least squares on three columns, solved by forming the 3x3
normal equations explicitly and running Gaussian elimination with partial
pivoting. numpy's lstsq would be one line, but the explicit solve is ~30 lines
of Swift with no linear-algebra dependency and produces bit-comparable results,
which is what the parity fixtures need.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

TWO_PI_OVER_24 = 2.0 * math.pi / 24.0

# Below this pivot magnitude the design is treated as rank-deficient: all the
# samples sit at (nearly) one clock hour, so cos/sin cannot be separated from
# the intercept and any amplitude would be fitted to noise.
PIVOT_EPS = 1e-9


@dataclass(frozen=True)
class CosinorFit:
    """A fitted baseline. `flat=True` means only the MESOR was estimated."""

    mesor: float
    a: float
    b: float
    n: int
    r2: float
    flat: bool

    @property
    def amplitude(self) -> float:
        return math.hypot(self.a, self.b)

    @property
    def acrophase_hours(self) -> float:
        """Clock hour of the daily peak, in [0, 24)."""
        if self.amplitude == 0.0:
            return float("nan")
        phi = math.atan2(self.b, self.a)  # y = amp * cos(w t - phi)
        return (phi / TWO_PI_OVER_24) % 24.0

    def predict(self, hours) -> np.ndarray:
        h = np.asarray(hours, dtype=float)
        if self.flat:
            return np.full(h.shape, self.mesor)
        w = TWO_PI_OVER_24 * h
        return self.mesor + self.a * np.cos(w) + self.b * np.sin(w)

    def to_dict(self) -> dict:
        return {
            "mesor": self.mesor,
            "a": self.a,
            "b": self.b,
            "n": self.n,
            "r2": self.r2,
            "flat": self.flat,
            "amplitude": self.amplitude,
            "acrophase_hours": self.acrophase_hours,
        }


def design_matrix(hours) -> np.ndarray:
    """Columns [1, cos(2 pi t / 24), sin(2 pi t / 24)]."""
    h = np.asarray(hours, dtype=float)
    w = TWO_PI_OVER_24 * h
    return np.column_stack([np.ones_like(h), np.cos(w), np.sin(w)])


def solve_3x3(ata: np.ndarray, atb: np.ndarray) -> np.ndarray | None:
    """Gaussian elimination with partial pivoting; None if rank-deficient.

    Deliberately written as scalar loops over a 3x3 so the Swift port is a
    transcription rather than a translation.
    """
    a = np.array(ata, dtype=float, copy=True)
    b = np.array(atb, dtype=float, copy=True)
    n = 3
    for col in range(n):
        pivot_row = col + int(np.argmax(np.abs(a[col:, col])))
        if abs(a[pivot_row, col]) < PIVOT_EPS:
            return None
        if pivot_row != col:
            a[[col, pivot_row]] = a[[pivot_row, col]]
            b[[col, pivot_row]] = b[[pivot_row, col]]
        for row in range(col + 1, n):
            factor = a[row, col] / a[col, col]
            a[row, col:] -= factor * a[col, col:]
            b[row] -= factor * b[col]
    x = np.zeros(n, dtype=float)
    for row in range(n - 1, -1, -1):
        s = b[row] - float(np.dot(a[row, row + 1:], x[row + 1:]))
        x[row] = s / a[row, row]
    return x


def flat_fit(values) -> CosinorFit:
    """Degenerate baseline: the mean, with no rhythm."""
    y = np.asarray(values, dtype=float)
    if y.size == 0:
        return CosinorFit(float("nan"), 0.0, 0.0, 0, 0.0, True)
    return CosinorFit(float(np.mean(y)), 0.0, 0.0, int(y.size), 0.0, True)


def fit(hours, values, *, min_distinct_hours: int = 0) -> CosinorFit:
    """Fit the first-harmonic cosinor; fall back to a flat mean when it cannot.

    `min_distinct_hours` counts distinct integer clock hours among the samples.
    Five well-spread hours is plenty to identify three parameters; the guard is
    there to stop a baseline built entirely from 2 a.m. sleep windows from
    claiming to know your 3 p.m. level.
    """
    h = np.asarray(hours, dtype=float)
    y = np.asarray(values, dtype=float)
    if h.shape != y.shape:
        raise ValueError("hours and values must have the same shape")
    if y.size < 4:
        return flat_fit(y)
    if min_distinct_hours > 0:
        distinct = np.unique(np.floor(h).astype(int) % 24).size
        if distinct < min_distinct_hours:
            return flat_fit(y)

    x = design_matrix(h)
    beta = solve_3x3(x.T @ x, x.T @ y)
    if beta is None or not np.all(np.isfinite(beta)):
        return flat_fit(y)

    resid = y - x @ beta
    ss_res = float(np.sum(resid * resid))
    centered = y - float(np.mean(y))
    ss_tot = float(np.sum(centered * centered))
    r2 = 0.0 if ss_tot <= 0.0 else 1.0 - ss_res / ss_tot
    return CosinorFit(float(beta[0]), float(beta[1]), float(beta[2]), int(y.size), r2, False)


def variance_explained(hours, values, *, min_distinct_hours: int = 0) -> float:
    """R^2 of the cosinor over a whole series -- the Section 7 falsifier.

    Below ~0.10 the circadian correction is not earning its place for you and
    the flat baseline is the honest simplification.
    """
    return fit(hours, values, min_distinct_hours=min_distinct_hours).r2
