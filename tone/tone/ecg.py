"""The ECG channel: RR intervals from `HKElectrocardiogram` voltage.

Section 2.4 of the plan lists this as an honest on-demand source -- 30 seconds
of single-lead ECG at 512 Hz, with a 2023 study using exactly this to quantify
stress [13] -- and then never uses it. It is worth using, because it is the only
source on this hardware with *real* beat timing.

Where the three sources sit on timing precision:

    export.xml bpm        integer bpm -> RR quantised in ~17 ms steps at 60 bpm
    HKHeartbeatSeries     true interval timing, whatever the watch resolved
    HKElectrocardiogram   512 Hz voltage, R-peaks located by you

Raw 512 Hz sampling puts an R peak within +/-0.98 ms, and parabolic
interpolation across the three samples at the peak does substantially better
than that, because the QRS complex is smooth on the scale of one sample. So the
ECG path is the most precise RMSSD available here, by a wide margin, and it is
available on demand in 30 seconds rather than a minute.

The catch, and it is a real one
-------------------------------
An Apple ECG is 30 seconds. At a resting heart rate of 60 bpm that is about 30
beats, hence about 29 intervals -- just under the plan's own `min_intervals`
threshold of 30, so the window gets dropped. Anyone with a resting heart rate
below roughly 62 bpm will find single ECGs silently discarded. `ecg_window`
therefore reports the interval count, and `enough_intervals` says whether one
recording will clear the bar for a given heart rate. Two back-to-back
recordings, concatenated, always will; see `rr_from_recordings`.

Detector
--------
Pan-Tompkins in shape -- bandpass, differentiate, square, integrate, adaptive
threshold, refractory period -- with two deliberate choices for HRV rather than
for beat counting:

1. The bandpass is a symmetric (linear-phase) FIR applied in 'same' mode, so it
   introduces **no phase shift at all**. A causal IIR filter would shift every
   R peak by its group delay. A constant shift cancels in the *differences*, so
   it would not hurt RMSSD -- but a frequency-dependent one does not cancel, and
   that is exactly what an IIR gives you.
2. Detection locates the QRS; the reported time comes from re-finding the
   extremum in the near-raw signal and fitting a parabola to it. The integration
   stage is good at *finding* beats and bad at *timing* them, and for RMSSD the
   timing is the entire product.
"""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .score import Window

DEFAULT_FS = 512.0
QRS_BAND = (5.0, 18.0)       # Hz; the QRS complex dominates here
REFRACTORY_S = 0.20          # physiological floor on beat separation
INTEGRATION_S = 0.12         # moving-window integrator, ~ one QRS wide
REFINE_S = 0.05              # how far to hunt for the true extremum


@dataclass(frozen=True)
class Recording:
    """One ECG recording: voltage in microvolts, plus what we know about it."""

    voltage: np.ndarray
    fs: float
    start: object | None = None       # datetime, when the CSV gives one
    classification: str | None = None
    source: str = "HKElectrocardiogram"

    @property
    def duration(self) -> float:
        return self.voltage.size / self.fs


# ---------------------------------------------------------------- detection

def _bandpass_kernel(fs: float, low: float, high: float, taps: int | None = None) -> np.ndarray:
    """Symmetric windowed-sinc bandpass. Linear phase, so 'same' mode is zero-phase."""
    if taps is None:
        taps = int(round(0.2 * fs)) | 1          # ~200 ms, forced odd
    if taps < 5:
        taps = 5
    n = np.arange(taps) - (taps - 1) / 2.0
    def sinc_lp(cut):
        f = cut / fs
        return 2 * f * np.sinc(2 * f * n)
    kernel = sinc_lp(high) - sinc_lp(low)
    kernel *= np.hamming(taps)
    kernel -= kernel.mean()                       # kill any residual DC
    return kernel


def _moving_average(x: np.ndarray, width: int) -> np.ndarray:
    if width < 1:
        return x
    kernel = np.ones(width) / width
    return np.convolve(x, kernel, mode="same")


def _parabolic_vertex(y_prev: float, y_here: float, y_next: float) -> float:
    """Sub-sample offset of a parabola's vertex through three points.

    Returns a value in roughly [-0.5, 0.5]; 0 means the middle sample already is
    the peak. This is what buys sub-millisecond timing out of a 512 Hz signal.
    """
    denom = y_prev - 2.0 * y_here + y_next
    if denom == 0.0:
        return 0.0
    offset = 0.5 * (y_prev - y_next) / denom
    return float(np.clip(offset, -1.0, 1.0))


def detect_r_peaks(voltage, fs: float = DEFAULT_FS) -> np.ndarray:
    """Return R-peak times in seconds, with sub-sample refinement.

    The output is float seconds from the start of the recording, not sample
    indices, precisely because the sub-sample part is the point.
    """
    x = np.asarray(voltage, dtype=float)
    n = x.size
    if n < int(fs):  # less than a second of signal
        return np.empty(0, dtype=float)

    x = x - np.median(x)
    band = np.convolve(x, _bandpass_kernel(fs, *QRS_BAND), mode="same")

    deriv = np.gradient(band)
    energy = _moving_average(deriv * deriv, max(1, int(round(INTEGRATION_S * fs))))

    # Adaptive threshold: a high quantile is robust to both a quiet recording
    # and a few enormous motion spikes, where a fraction-of-max rule fails on
    # the second.
    noise = float(np.median(energy))
    peak_level = float(np.quantile(energy, 0.98))
    threshold = noise + 0.35 * (peak_level - noise)
    if not math.isfinite(threshold) or threshold <= 0:
        return np.empty(0, dtype=float)

    refractory = int(round(REFRACTORY_S * fs))
    refine = int(round(REFINE_S * fs))

    above = energy > threshold
    peaks: list[float] = []
    i = 1
    last_index = -refractory
    while i < n - 1:
        if not above[i]:
            i += 1
            continue
        j = i
        while j < n and above[j]:
            j += 1
        # Crest of the energy envelope within this excursion.
        local = int(i + np.argmax(energy[i:j]))
        if local - last_index >= refractory:
            peaks.append(_refine_peak(band, local, refine, fs))
            last_index = local
        i = j

    return np.array(peaks, dtype=float)


def _refine_peak(band: np.ndarray, index: int, radius: int, fs: float) -> float:
    """Locate the QRS extremum near `index` and interpolate it to sub-sample."""
    lo = max(1, index - radius)
    hi = min(band.size - 1, index + radius + 1)
    if hi <= lo:
        return index / fs
    segment = band[lo:hi]
    # The R deflection may be positive or negative depending on lead placement
    # and how the watch was held; take whichever extremum is larger.
    k = int(np.argmax(np.abs(segment))) + lo
    if k <= 0 or k >= band.size - 1:
        return k / fs
    sign = 1.0 if band[k] >= 0 else -1.0
    offset = _parabolic_vertex(sign * band[k - 1], sign * band[k], sign * band[k + 1])
    return (k + offset) / fs


def rr_from_ecg(voltage, fs: float = DEFAULT_FS) -> np.ndarray:
    """R-peak times -> RR intervals in milliseconds."""
    peaks = detect_r_peaks(voltage, fs)
    if peaks.size < 2:
        return np.empty(0, dtype=float)
    return np.diff(peaks) * 1000.0


def timing_resolution_ms(fs: float = DEFAULT_FS) -> float:
    """Worst-case RR error from sampling alone, before interpolation.

    One sample of uncertainty on each of the two peaks bounding an interval. At
    512 Hz that is 1.95 ms -- already an order of magnitude better than the
    export path's ~17 ms bpm quantisation at 60 bpm, and the parabolic step
    improves on it further.
    """
    return 1000.0 / fs


def enough_intervals(heart_rate_bpm: float, duration_s: float = 30.0,
                     min_intervals: int = 30) -> bool:
    """Will one recording of this length clear the minimum-interval bar?

    At 60 bpm a 30-second ECG yields about 29 intervals, one short of the
    default threshold. This is not a rounding detail: it means single ECGs are
    silently dropped for anyone with a resting heart rate below about 62.
    """
    return (duration_s * heart_rate_bpm / 60.0) - 1.0 >= min_intervals


def minimum_heart_rate(duration_s: float = 30.0, min_intervals: int = 30) -> float:
    """The heart rate below which one recording is too short to be scored."""
    return 60.0 * (min_intervals + 1) / duration_s


def rr_from_recordings(recordings: list[Recording]) -> np.ndarray:
    """Concatenate intervals across recordings without inventing one between them.

    Two ECGs taken back to back clear the interval threshold at any plausible
    heart rate. The gap *between* recordings is not a beat-to-beat interval and
    must never be treated as one.
    """
    chunks = [rr_from_ecg(r.voltage, r.fs) for r in recordings]
    chunks = [c for c in chunks if c.size]
    if not chunks:
        return np.empty(0, dtype=float)
    return np.concatenate(chunks)


# -------------------------------------------------------------------- input

_NUMBER = re.compile(r"^-?\d+(\.\d+)?([eE][-+]?\d+)?$")


#: Apple writes the minus sign as U+2212 MINUS SIGN, not ASCII hyphen, in ECG
#: exports. `float()` rejects it, so a naive reader silently drops every negative
#: sample -- which does not crash, it just half-rectifies your ECG.
UNICODE_MINUS = "−"


def _to_float(text: str, decimal_comma: bool) -> float | None:
    text = text.strip().replace(UNICODE_MINUS, "-").replace(" ", "")
    if decimal_comma:
        text = text.replace(".", "").replace(",", ".")
    if not text or not _NUMBER.match(text):
        return None
    return float(text)


def read_ecg_csv(path: str | Path) -> Recording:
    """Read one `ecg_*.csv` from an Apple Health export.

    Health writes ECGs as separate CSV files under
    `apple_health_export/electrocardiograms/`, not into `export.xml` -- which is
    why `tone parse` does not see them. The header is localised, so the sample
    rate is found by pattern rather than by position, and everything that parses
    as a bare number is taken as a voltage sample.

    Two real-world details that a naive reader gets wrong without failing:

    - the minus sign is U+2212, not ASCII '-', so `float()` raises on every
      negative sample. Skipping the unparseable ones leaves a half-rectified
      trace that still looks plausible and detects beats badly.
    - in locales that use the comma as a decimal separator, the file is
      semicolon-delimited and the numbers read '1,5'. Detected by counting
      delimiters rather than by asking the OS, since the file may have come from
      a differently-configured phone.
    """
    text = Path(path).read_text(errors="replace")
    decimal_comma = text.count(";") > text.count(",")
    rows = list(csv.reader(text.splitlines(), delimiter=";" if decimal_comma else ","))

    fs = DEFAULT_FS
    classification = None
    start = None
    values: list[float] = []

    for row in rows:
        if not row:
            continue
        first = row[0].strip()
        rest = " ".join(c.strip() for c in row[1:]).strip()
        if len(row) <= 2 and not rest:
            value = _to_float(first, decimal_comma)
            if value is not None:
                values.append(value)
                continue
        lowered = first.lower()
        if "sample rate" in lowered or "hz" in rest.lower():
            found = re.search(r"(\d+(?:[.,]\d+)?)", rest or first)
            if found:
                fs = float(found.group(1).replace(",", "."))
        elif "classification" in lowered:
            classification = rest or None
        elif "recording date" in lowered or lowered == "date":
            start = rest or None

    return Recording(np.array(values, dtype=float), fs,
                     start=start, classification=classification)


def recording_datetime(rec: Recording):
    """Parse the CSV's 'Recording Date' into an aware datetime, or None.

    Apple writes it in the same offset-bearing format as export.xml. An ECG
    without a usable timestamp cannot be placed on the clock, so it cannot be
    scored -- the caller must supply one rather than have a guess invented here.
    """
    from datetime import datetime

    text = rec.start
    if not isinstance(text, str) or not text.strip():
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S %z", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M %z"):
        try:
            return datetime.strptime(text.strip(), fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text.strip())
    except ValueError:
        return None


def find_ecg_files(root: str | Path) -> list[Path]:
    """Every ECG CSV under an unzipped export directory."""
    base = Path(root)
    return sorted(base.rglob("ecg_*.csv")) or sorted(
        p for p in base.rglob("*.csv") if "electrocardiogram" in str(p).lower()
    )


def group_sittings(dated, pair_within_minutes: float = 2.0) -> list[list]:
    """Group (when, Recording) pairs into sittings by time proximity.

    A "sitting" is one or more recordings taken back to back, which is how
    someone with a resting heart rate below `minimum_heart_rate()` gets enough
    intervals for a single scoreable window.

    The default of 2 minutes is chosen against a specific hazard: a Phase 2.1
    test-retest pair is deliberately separated by *five* minutes, and merging
    the two halves would silently destroy the calibration it exists to provide.
    Keep this threshold well under that gap.
    """
    from datetime import timedelta

    ordered = sorted(dated, key=lambda item: item[0])
    gap = timedelta(minutes=pair_within_minutes)
    sittings: list[list] = []
    for item in ordered:
        if sittings and item[0] - sittings[-1][-1][0] <= gap:
            sittings[-1].append(item)
        else:
            sittings.append([item])
    return sittings


def ecg_window(recordings, when, *, source: str = "HKElectrocardiogram") -> Window:
    """Build a scoreable Window from one or more ECG recordings.

    `quantized=False`: unlike the export path, nothing here was rounded to whole
    bpm, so the quantisation correction must not be applied. Marked `on_demand`
    because an ECG is always something you chose to take -- which makes it
    eligible to be shown on its own, and eligible for test-retest pairing.
    """
    if isinstance(recordings, Recording):
        recordings = [recordings]
    return Window(
        start=when,
        rr=rr_from_recordings(list(recordings)),
        source=source,
        on_demand=True,
        quantized=False,
    )
