"""Every constant the score depends on, in one place.

Anything here that changes the numbers must be written into the fixtures file,
because the Swift port reads the same config block. If you add a field, add it
to `to_dict`/`from_dict` too, or parity will silently drift.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class ScoreConfig:
    """Scoring constants (Section 3 of docs/PLAN.md).

    artifact_threshold
        An RR interval is kept if it is within this fraction of at least one of
        its immediate neighbours. 0.20 = the plan's "differs from its neighbour
        by more than 20%" rule. See metrics.filter_rr for why the test is
        two-sided.
    min_intervals
        Windows with fewer usable RR intervals than this are dropped entirely.
        The plan says "about 30 beats"; on the export path one exported
        `InstantaneousBeatsPerMinute` entry == one RR interval (see
        parse_export), so beats and intervals coincide there.
    baseline_days
        Length of the trailing window the cosinor baseline is fitted on.
    min_baseline_windows / min_baseline_hours
        The baseline needs both enough samples and enough spread across the
        clock, or the cos/sin columns are near-collinear and the amplitude is
        fitted to noise. Below either threshold the fit degrades to a flat mean
        (MESOR only) and the window is flagged `flat_baseline`.
    min_scoring_windows
        Below this many usable windows in the trailing period, no score is
        emitted at all (flagged `warming_up`). The app shows "not enough
        history yet" rather than inventing a number.
    lambda_hrv
        The optional physiological prior on the HRV channel: w_x is multiplied
        by lambda. Start at 1.0 and let the test-retest numbers speak (Step 5).
    sigma2_eps_x / sigma2_eps_h
        Measurement-error variances of ln(RMSSD) and ln(HR), from Phase 2.1
        test-retest. None means "not measured yet" and the score falls back to
        equal weights, flagged in the output so you cannot forget.
    interval
        "t" uses Student's t with n_d-1 degrees of freedom for the daily
        interval; "normal" uses 1.96. The plan says 1.96, but with n_d ~ 5 that
        is optimistic by roughly 40%, and the honesty of the interval is the
        product. See score.critical_value.
    mad_scale
        1.4826 makes MAD a consistent estimator of sigma for Gaussian data.
    quantization_correction
        Only meaningful on the export path, where RR is reconstructed from
        integer bpm. See metrics.quantization_variance and docs/FINDINGS.md.
    """

    artifact_threshold: float = 0.20
    min_intervals: int = 30
    baseline_days: float = 28.0
    min_baseline_windows: int = 30
    min_baseline_hours: int = 5
    min_scoring_windows: int = 30
    lambda_hrv: float = 1.0
    sigma2_eps_x: float | None = None
    sigma2_eps_h: float | None = None
    interval: str = "t"
    mad_scale: float = 1.4826
    quantization_correction: bool = False

    def __post_init__(self) -> None:
        if not 0.0 < self.artifact_threshold < 1.0:
            raise ValueError("artifact_threshold must be in (0, 1)")
        if self.min_intervals < 3:
            raise ValueError("min_intervals must be at least 3 to form a difference")
        if self.baseline_days <= 0:
            raise ValueError("baseline_days must be positive")
        if self.lambda_hrv <= 0:
            raise ValueError("lambda_hrv must be positive")
        if self.interval not in ("t", "normal"):
            raise ValueError("interval must be 't' or 'normal'")
        for name in ("sigma2_eps_x", "sigma2_eps_h"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive or None")

    @property
    def weights_measured(self) -> bool:
        """True once Phase 2.1 has supplied both measurement-error variances."""
        return self.sigma2_eps_x is not None and self.sigma2_eps_h is not None

    def weights(self) -> tuple[float, float]:
        """Return (w_x, w_h), the precision weights of Step 5.

        Unmeasured variances fall back to equal weights rather than to the old
        0.7/0.3 heuristic: an untested guess should not masquerade as a
        calibration.
        """
        if not self.weights_measured:
            return (self.lambda_hrv, 1.0)
        return (self.lambda_hrv / float(self.sigma2_eps_x), 1.0 / float(self.sigma2_eps_h))

    def with_weights(self, sigma2_eps_x: float, sigma2_eps_h: float) -> "ScoreConfig":
        return replace(self, sigma2_eps_x=sigma2_eps_x, sigma2_eps_h=sigma2_eps_h)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ScoreConfig":
        known = {f: data[f] for f in cls.__dataclass_fields__ if f in data}
        return cls(**known)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "ScoreConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))


DEFAULT = ScoreConfig()
