"""Parity fixtures: the contract between this Python and the Swift to come.

A fixture file is (config, windows in, numbers out). The Swift `ScoreEngine`
must reproduce every output field to within 1e-6 before any UI is written
(Phase 4.2). That is what stops the watch app from quietly drifting away from
the science: Claude Code cannot argue with a failing test.

The fixture carries *RR intervals in milliseconds*, not bpm and not beat
timestamps, so the file tests the scoring maths rather than either platform's
ingestion path. Ingestion is tested separately on each side -- and the two sides
differ there on purpose: the export gives one instantaneous bpm per interval,
while `HKHeartbeatSeriesSample` gives beat timestamps, where N beats make N-1
intervals. Getting that off-by-one wrong is the likeliest way to ship a Swift
engine that passes nothing.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np

from .config import DEFAULT, ScoreConfig
from .score import Window, daily_scores, score_windows

FORMAT_VERSION = 1
TOLERANCE = 1e-6

# Fields compared window-by-window. Baseline-derived values are included on
# purpose: if only S were checked, a Swift bug in the cosinor could cancel
# against one in the sigma and still pass.
WINDOW_FIELDS = (
    "n_kept", "rmssd", "mean_hr", "x", "h",
    "resid_x", "resid_h", "sigma_x", "sigma_h", "z_x", "z_h", "s",
)
DAILY_FIELDS = ("mean", "n", "sd", "se", "lo", "hi")


def build(windows: list[Window], cfg: ScoreConfig = DEFAULT, *, limit: int | None = 50) -> dict:
    """Score `windows` and package the result as a fixture document.

    `limit` keeps the last N *scored* windows in the expectations, but every
    window stays in the input: the trailing windows are only scoreable because
    the earlier ones form their baseline, so truncating the input would change
    the answer. The Swift engine is fed the whole series and checked on the tail.
    """
    # Round the intervals *before* scoring, not on the way out: the fixture
    # stores 6-decimal RR, and expectations computed from unrounded inputs would
    # be expectations for a series the Swift side never sees.
    ordered = [
        Window(
            start=w.start,
            rr=np.round(np.asarray(w.rr, dtype=float), 6),
            source=w.source,
            on_demand=w.on_demand,
            quantized=w.quantized,
            apple_sdnn=w.apple_sdnn,
        )
        for w in sorted(windows, key=lambda w: w.start)
    ]
    scores = score_windows(ordered, cfg)
    daily = daily_scores(scores, cfg)

    scored_indices = [i for i, s in enumerate(scores) if s.scored]
    if limit is not None:
        scored_indices = scored_indices[-limit:]
    checked = set(scored_indices)

    return {
        "format_version": FORMAT_VERSION,
        "generated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "tolerance": TOLERANCE,
        "config": cfg.to_dict(),
        "notes": (
            "rr_ms are inter-beat intervals in milliseconds, already in order. "
            "Expectations are given only for windows with \"checked\": true; the "
            "rest are baseline history and must still be fed to the engine."
        ),
        "windows": [
            {
                "index": i,
                "start": w.start.isoformat(),
                "hour": w.hour,
                "on_demand": w.on_demand,
                "quantized": w.quantized,
                "rr_ms": [float(v) for v in np.asarray(w.rr)],
                "checked": i in checked,
                "expected": _expected(scores[i]) if i in checked else None,
            }
            for i, w in enumerate(ordered)
        ],
        "daily": [d.to_dict() for d in daily],
    }


def _expected(score) -> dict:
    d = score.to_dict()
    return {k: d[k] for k in WINDOW_FIELDS}


def _sanitize(value):
    """Replace non-finite floats with null.

    Python's json writes bare `NaN`, which is not JSON and which Swift's
    JSONDecoder rejects outright. A fixture that only the generator can read is
    not a contract, so non-finite values go out as null and `_close` accepts
    null against NaN.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    return value


def save(doc: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(_sanitize(doc), indent=2, allow_nan=False) + "\n")


def _close(got, want, tol: float) -> bool:
    if want is None:
        return got is None or (isinstance(got, float) and not math.isfinite(got))
    if got is None:
        return False
    if isinstance(want, bool) or isinstance(want, int):
        return got == want
    if not math.isfinite(got):
        return False
    return abs(got - want) <= tol * max(1.0, abs(want))


def verify(path: str | Path) -> tuple[bool, list[str]]:
    """Re-run the pipeline over a fixture file and report every mismatch.

    Run this in CI on the Python side too. A fixture that no longer matches its
    own generator means the algorithm changed underneath the Swift port, and the
    right response is to regenerate deliberately, not to loosen the tolerance.
    """
    doc = json.loads(Path(path).read_text())
    cfg = ScoreConfig.from_dict(doc["config"])
    tol = float(doc.get("tolerance", TOLERANCE))

    windows = [
        Window(
            start=datetime.fromisoformat(w["start"]),
            rr=np.asarray(w["rr_ms"], dtype=float),
            on_demand=bool(w.get("on_demand", False)),
            quantized=bool(w.get("quantized", False)),
        )
        for w in doc["windows"]
    ]
    scores = score_windows(windows, cfg)
    daily = {d.day.isoformat(): d.to_dict() for d in daily_scores(scores, cfg)}

    failures: list[str] = []
    for w in doc["windows"]:
        if not w.get("checked"):
            continue
        got = scores[w["index"]].to_dict()
        for field in WINDOW_FIELDS:
            if not _close(got[field], w["expected"][field], tol):
                failures.append(
                    f"window {w['index']} ({w['start']}) {field}: "
                    f"got {got[field]!r}, expected {w['expected'][field]!r}"
                )
    for want in doc.get("daily", []):
        got = daily.get(want["day"])
        if got is None:
            failures.append(f"daily {want['day']}: missing")
            continue
        for field in DAILY_FIELDS:
            if not _close(got[field], want[field], tol):
                failures.append(
                    f"daily {want['day']} {field}: got {got[field]!r}, expected {want[field]!r}"
                )
    return (not failures), failures
