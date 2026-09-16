"""Read Apple Health's `export.xml` (or `export.zip`) into scoreable windows.

Health > Profile > Export All Health Data. The file is hundreds of megabytes of
XML, so it is streamed with `iterparse` and the parsed tree is discarded as it
goes; peak memory stays flat regardless of how many years you have.

What comes out of the export, and what does not
-----------------------------------------------
Each SDNN record carries a `HeartRateVariabilityMetadataList` of
`InstantaneousBeatsPerMinute` entries -- one per beat in the window. Those are
the beat-to-beat data, but in a degraded form: `bpm` is an **integer**. RR
reconstructed as 60000/bpm is therefore quantised in steps of RR^2/60000 ms,
which is ~17 ms at 60 bpm and ~24 ms at 50 bpm.

The resulting bias in RMSSD is small in the middle of the range and large at the
edges: measured against simulated ground truth it is about +2% at a true RMSSD
of 35 ms and 60 bpm, +10% at 15 ms and 60 bpm, and +16% at 15 ms and 50 bpm.
That shape is the problem. The bias is worst where RMSSD is low -- your stressed
windows -- and it grows as heart rate falls, which is where your sleep windows
live. So it does not shift the score, it *compresses* it: on the log scale the
gap between a calm window and a stressed one shrinks by roughly 9%.

The export path is therefore good enough to establish that the pipeline runs, to
count your usable windows, to characterise their clock times, and to score the
HR channel exactly. Its RMSSD is a compressed proxy.
`HKHeartbeatSeriesSample` on the watch carries real interval timing, so the
Swift engine will see cleaner numbers than Phase 1 does, not dirtier ones.
`quantization_correction` in the config removes most of the bias in expectation
(metrics.correct_rmssd_for_quantization); it cannot put back the lost precision,
and it over-corrects when the quantisation step approaches the RR spread. See
docs/FINDINGS.md for the derivation and the measured table.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator
from xml.etree import ElementTree as ET

import numpy as np

from .score import Window

SDNN_TYPE = "HKQuantityTypeIdentifierHeartRateVariabilitySDNN"
RMSSD_TYPE = "HKQuantityTypeIdentifierHeartRateVariabilityRMSSD"  # Series 12+ / watchOS 27
MINDFUL_TYPE = "HKCategoryTypeIdentifierMindfulSession"
APPLE_DATE = "%Y-%m-%d %H:%M:%S %z"


@dataclass
class RawWindow:
    """One HRV record as it appears in the export, before any scoring."""

    start: datetime
    end: datetime
    source: str
    apple_sdnn: float | None
    apple_rmssd: float | None
    bpm: list[int]
    on_demand: bool = False

    def to_window(self) -> Window:
        return Window(
            start=self.start,
            rr=60000.0 / np.asarray(self.bpm, dtype=float),
            source=self.source,
            on_demand=self.on_demand,
            quantized=True,
            apple_sdnn=self.apple_sdnn,
        )


def _parse_date(text: str | None) -> datetime | None:
    if not text:
        return None
    try:
        return datetime.strptime(text, APPLE_DATE)
    except ValueError:
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            return None


def _open_export(path: str | Path):
    """Yield a binary stream for export.xml, transparently unzipping."""
    p = Path(path)
    if p.suffix.lower() == ".zip":
        zf = zipfile.ZipFile(p)
        names = [n for n in zf.namelist() if n.endswith("export.xml")]
        if not names:
            raise FileNotFoundError(f"no export.xml inside {p}")
        # Prefer the top-level export.xml over export_cda.xml-style siblings.
        name = sorted(names, key=len)[0]
        return zf.open(name), zf
    return p.open("rb"), None


def iter_records(path: str | Path) -> Iterator[tuple[str, ET.Element]]:
    """Stream `Record` elements, clearing each one after it is yielded."""
    stream, zf = _open_export(path)
    try:
        context = ET.iterparse(stream, events=("start", "end"))
        _, root = next(context)
        for event, elem in context:
            if event != "end" or elem.tag != "Record":
                continue
            yield elem.get("type", ""), elem
            elem.clear()
            root.clear()  # Records are flat children of HealthData; drop them all
    finally:
        stream.close()
        if zf is not None:
            zf.close()


def parse(
    path: str | Path,
    *,
    since: datetime | None = None,
    mindful_slack_minutes: float = 5.0,
) -> list[RawWindow]:
    """Parse HRV windows and tag the ones that came from a Mindfulness session.

    A window is marked `on_demand` when it starts within `mindful_slack_minutes`
    of a Mindfulness session (before, during or just after). Those are the
    windows you *chose* to take, which makes them the ones eligible to be shown
    on their own (Step 6) and the ones test-retest calibration pairs up.
    """
    windows: list[RawWindow] = []
    mindful: list[tuple[datetime, datetime]] = []
    apple_rmssd: dict[datetime, float] = {}

    for rec_type, elem in iter_records(path):
        if rec_type == MINDFUL_TYPE:
            start, end = _parse_date(elem.get("startDate")), _parse_date(elem.get("endDate"))
            if start and end:
                mindful.append((start, end))
            continue
        if rec_type not in (SDNN_TYPE, RMSSD_TYPE):
            continue

        start = _parse_date(elem.get("startDate"))
        if start is None or (since is not None and start < since):
            continue

        try:
            value = float(elem.get("value", "nan"))
        except ValueError:
            value = float("nan")

        if rec_type == RMSSD_TYPE:
            # Series 12 and watchOS 27 emit an RMSSD record *alongside* the SDNN
            # one for the same window. Emitting a second RawWindow for it would
            # score the same minute twice, so it is recorded as an attribute of
            # the SDNN window instead -- a free cross-check against our own
            # RMSSD wherever the hardware provides one.
            apple_rmssd[start] = value
            continue

        bpm = [
            int(round(float(b.get("bpm", "0"))))
            for b in elem.iter("InstantaneousBeatsPerMinute")
            if b.get("bpm")
        ]
        bpm = [b for b in bpm if b > 0]

        windows.append(
            RawWindow(
                start=start,
                end=_parse_date(elem.get("endDate")) or start,
                source=elem.get("sourceName", "unknown"),
                apple_sdnn=value,
                apple_rmssd=None,
                bpm=bpm,
            )
        )

    for w in windows:
        if w.start in apple_rmssd:
            w.apple_rmssd = apple_rmssd[w.start]

    tag_on_demand(windows, mindful, slack_minutes=mindful_slack_minutes)
    windows.sort(key=lambda w: w.start)
    return windows


def tag_on_demand(
    windows: list[RawWindow],
    mindful: list[tuple[datetime, datetime]],
    *,
    slack_minutes: float = 5.0,
) -> int:
    """Mark windows overlapping a Mindfulness session. Returns how many."""
    if not mindful:
        return 0
    slack = timedelta(minutes=slack_minutes)
    sessions = sorted(mindful)
    count = 0
    for w in windows:
        for s, e in sessions:
            if s - slack <= w.start <= e + slack:
                w.on_demand = True
                count += 1
                break
    return count


def sdnn_agreement(windows: list[Window]) -> dict:
    """Compare our recomputed SDNN against Apple's own number for each window.

    This is the reconstruction check, and it is worth running before trusting
    anything downstream: Apple's SDNN was computed from the *unrounded*
    intervals, ours from the rounded bpm. If the two agree to within a few ms
    across thousands of windows, the reconstruction is sound and the residual
    disagreement is the quantisation cost, quantified. If they diverge wildly,
    the parse is wrong -- fix that before interpreting a single score.
    """
    from .metrics import filter_rr, sdnn as sdnn_of

    ours, theirs = [], []
    for w in windows:
        if w.apple_sdnn is None or not np.isfinite(w.apple_sdnn) or np.size(w.rr) < 3:
            continue
        rr = np.asarray(w.rr, dtype=float)
        value = sdnn_of(rr, filter_rr(rr))
        if np.isfinite(value):
            ours.append(value)
            theirs.append(w.apple_sdnn)
    if not ours:
        return {"n": 0}
    ours_a, theirs_a = np.array(ours), np.array(theirs)
    diff = ours_a - theirs_a
    corr = float(np.corrcoef(ours_a, theirs_a)[0, 1]) if ours_a.size > 1 else float("nan")
    return {
        "n": int(ours_a.size),
        "mean_ours": float(ours_a.mean()),
        "mean_apple": float(theirs_a.mean()),
        "bias": float(diff.mean()),
        "mae": float(np.abs(diff).mean()),
        "corr": corr,
    }
