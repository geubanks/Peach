"""Tone: a within-person, precision-weighted, circadian-adjusted HRV stress index.

The package is the *reference implementation* of the score. It is deliberately
written before any Swift exists: the watch app's `ScoreEngine` has to reproduce
the JSON fixtures this package emits (`tone fixtures`) to within 1e-6, so the
science lives here and the Swift is only a port.

Module map
----------
config        scoring constants in one frozen dataclass, serialisable to JSON
metrics       per-window time-domain metrics (artifact filter, RMSSD, HR, SDNN)
cosinor       first-harmonic cosinor fit by explicit 3x3 normal equations
score         rolling baseline, robust residual z-scores, precision-weighted S
calibrate     test-retest estimation of the measurement-error variances
diary         Spearman rho against diary ratings with a bootstrap interval
parse_export  Apple Health `export.xml` (or `export.zip`) -> tidy windows
simulate      synthetic bodies with known parameters, so everything is testable
fixtures      parity fixtures for the Swift port
"""

from .config import ScoreConfig
from .metrics import WindowMetrics, window_metrics
from .score import DailyScore, WindowScore, score_windows, daily_scores

__version__ = "0.1.0"

__all__ = [
    "ScoreConfig",
    "WindowMetrics",
    "window_metrics",
    "WindowScore",
    "DailyScore",
    "score_windows",
    "daily_scores",
    "__version__",
]
