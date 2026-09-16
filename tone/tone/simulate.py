"""A synthetic body with known parameters.

Nothing in this project can be tested against real data until the export exists
and the diary has run for two weeks. That is a bad reason to leave the pipeline
unexercised, so this module generates a body whose circadian amplitude, stress
coupling and measurement noise you already know, and the tests then ask whether
the estimator recovers them.

It is also how `tone demo` shows the whole Phase 1-2 flow end to end before you
have collected anything.

The generative model, in order:

    stress_d        AR(1) across days, plus a within-day bump in the afternoon
    ln RMSSD_true   MESOR + circadian first harmonic - beta_x * stress
    ln HR_true      MESOR + circadian first harmonic + beta_h * stress
    sensor error    added on the log scale (this is sigma_eps, the thing
                    Phase 2.1 test-retest is trying to measure)
    beat series     AR(1) RR intervals whose realised RMSSD and mean match the
                    window's targets -- so finite-window sampling noise is in
                    the output too, not assumed away
    quantisation    optional rounding to whole bpm, to mimic the export path

Two noise sources therefore reach the score: sensor error, which is what
test-retest recovers, and sampling noise from a 60-second window, which no
calibration can remove. Real data has both. So does this.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np

from .diary import DiaryEntry
from .score import Window

EASTERN = timezone(timedelta(hours=-4))


@dataclass(frozen=True)
class Truth:
    """The parameters the simulator used -- what a good estimator should find."""

    mesor_x: float
    amp_x: float
    acrophase_x: float
    mesor_h: float
    amp_h: float
    acrophase_h: float
    beta_x: float
    beta_h: float
    sigma_eps_x: float
    sigma_eps_h: float

    def to_dict(self) -> dict:
        return self.__dict__.copy()


# Passive windows are not uniform over the clock: the watch samples when you are
# still, which is mostly when you are asleep. (weight, lo_hour, hi_hour)
CLOCK_MIXTURE = ((0.45, 0.5, 6.0), (0.20, 7.0, 10.0), (0.20, 12.0, 18.0), (0.15, 20.0, 23.9))


def _rr_series(mean_rr: float, target_rmssd: float, n: int, rng) -> np.ndarray:
    """AR(1) intervals with the requested mean and RMSSD.

    For u_k = phi u_{k-1} + w_k, var(u_k - u_{k-1}) = 2 var(u) (1 - phi). Fixing
    SDNN = RMSSD gives phi = 0.5, which is a fair shape for a one-minute resting
    window: enough short-term structure that successive differences are not
    white, not so much that the window is a trend.
    """
    phi = 0.5
    sd_u = target_rmssd / math.sqrt(2.0 * (1.0 - phi))
    sd_w = sd_u * math.sqrt(1.0 - phi * phi)
    u = np.empty(n)
    u[0] = rng.normal(0.0, sd_u)
    for k in range(1, n):
        u[k] = phi * u[k - 1] + rng.normal(0.0, sd_w)
    rr = mean_rr + u - u.mean()  # centre so the realised mean RR is exact
    return np.clip(rr, 250.0, 2000.0)


def _add_ectopic(rr: np.ndarray, rng) -> np.ndarray:
    """One premature beat: a short interval followed by a compensatory long one.

    This is what the artifact filter exists for, and leaving it out of the
    simulator would make the filter untestable.
    """
    if rr.size < 5:
        return rr
    k = int(rng.integers(1, rr.size - 2))
    out = rr.copy()
    out[k] *= 0.55
    out[k + 1] *= 1.45
    return out


def _quantize(rr: np.ndarray) -> np.ndarray:
    """Round to whole bpm and back, as Apple's export does."""
    return 60000.0 / np.round(60000.0 / rr)


def simulate(
    *,
    days: int = 120,
    seed: int = 20260916,
    windows_per_day: float = 5.0,
    window_seconds: float = 60.0,
    truth: Truth | None = None,
    quantized: bool = False,
    ectopic_rate: float = 0.08,
    mindful_pair_days: int = 10,
    diary_hours: tuple[float, ...] = (11.0, 17.0, 22.0),
    start: datetime | None = None,
) -> tuple[list[Window], list[DiaryEntry], Truth]:
    """Generate windows, a matching diary, and the parameters behind both."""
    rng = np.random.default_rng(seed)
    truth = truth or Truth(
        mesor_x=math.log(38.0), amp_x=0.30, acrophase_x=3.5,
        mesor_h=math.log(62.0), amp_h=0.10, acrophase_h=16.0,
        beta_x=0.22, beta_h=0.06,
        sigma_eps_x=0.18, sigma_eps_h=0.035,
    )
    start = start or datetime(2026, 5, 1, 0, 0, tzinfo=EASTERN)

    # Latent stress: AR(1) over days, standardised so beta is interpretable.
    rho_day = 0.55
    stress_day = np.empty(days)
    stress_day[0] = rng.normal()
    for d in range(1, days):
        stress_day[d] = rho_day * stress_day[d - 1] + math.sqrt(1 - rho_day**2) * rng.normal()

    def stress_at(day_index: int, hour: float) -> float:
        # Stress rides higher in the working afternoon and lower asleep.
        diurnal = 0.45 * math.exp(-((hour - 15.0) ** 2) / (2 * 4.0**2)) - 0.25 * (hour < 6.0)
        return float(stress_day[day_index] + diurnal + 0.25 * rng.normal())

    def circadian(amp: float, acro: float, hour: float) -> float:
        return amp * math.cos(2 * math.pi * (hour - acro) / 24.0)

    def make_window(when: datetime, day_index: int, on_demand: bool) -> Window:
        hour = when.hour + when.minute / 60.0 + when.second / 3600.0
        stress = stress_at(day_index, hour)
        x_true = truth.mesor_x + circadian(truth.amp_x, truth.acrophase_x, hour) - truth.beta_x * stress
        h_true = truth.mesor_h + circadian(truth.amp_h, truth.acrophase_h, hour) + truth.beta_h * stress
        x_obs = x_true + rng.normal(0.0, truth.sigma_eps_x)
        h_obs = h_true + rng.normal(0.0, truth.sigma_eps_h)

        hr = float(np.clip(math.exp(h_obs), 38.0, 160.0))
        rmssd_target = float(np.clip(math.exp(x_obs), 3.0, 250.0))
        mean_rr = 60000.0 / hr
        n_beats = max(8, int(round(window_seconds * hr / 60.0)))

        rr = _rr_series(mean_rr, rmssd_target, n_beats, rng)
        if rng.random() < ectopic_rate:
            rr = _add_ectopic(rr, rng)
        if quantized:
            rr = _quantize(rr)
        return Window(
            start=when,
            rr=rr,
            source="Simulated Watch",
            on_demand=on_demand,
            quantized=quantized,
        )

    pair_days = set(rng.choice(np.arange(days // 3, days), size=min(mindful_pair_days, days // 2), replace=False).tolist())

    windows: list[Window] = []
    diary: list[DiaryEntry] = []
    for d in range(days):
        midnight = start + timedelta(days=d)
        n = max(1, int(rng.poisson(windows_per_day)))
        weights = np.array([m[0] for m in CLOCK_MIXTURE])
        for _ in range(n):
            comp = CLOCK_MIXTURE[int(rng.choice(len(CLOCK_MIXTURE), p=weights / weights.sum()))]
            hour = float(rng.uniform(comp[1], comp[2]))
            windows.append(make_window(midnight + timedelta(hours=hour), d, on_demand=False))

        if d in pair_days:
            # Anchored to a diary time on purpose: a spot check is only worth
            # taking if there is a rating to compare it against, so the plan's
            # "after lab" session and the "after lab" diary prompt are the same
            # moment. Phase 2.2's on-demand test has nothing to correlate
            # otherwise.
            anchor = diary_hours[min(1, len(diary_hours) - 1)]
            base = midnight + timedelta(hours=anchor + float(rng.uniform(-0.3, 0.1)))
            windows.append(make_window(base, d, on_demand=True))
            windows.append(make_window(base + timedelta(minutes=5), d, on_demand=True))

        for hour in diary_hours:
            when = midnight + timedelta(hours=hour)
            rating = 5.0 + 1.8 * stress_at(d, hour) + rng.normal(0.0, 0.7)
            diary.append(DiaryEntry(when, float(np.clip(round(rating), 0, 10))))

    windows.sort(key=lambda w: w.start)
    return windows, diary, truth


# --------------------------------------------------------------------- ECG

def synthetic_ecg(
    beat_times,
    *,
    fs: float = 512.0,
    duration: float | None = None,
    noise_uv: float = 25.0,
    wander_uv: float = 150.0,
    mains_uv: float = 0.0,
    mains_hz: float = 60.0,
    inverted: bool = False,
    seed: int = 4,
):
    """A single-lead ECG with R peaks at exactly `beat_times` (seconds).

    Exists so the R-peak detector can be measured rather than eyeballed: the
    truth is an input. Waves are Gaussians at conventional offsets and
    amplitudes, in microvolts, plus baseline wander (breathing and electrode
    drift), optional mains hum, and white noise.

    `inverted` flips the trace, which is what a watch ECG looks like when the
    crown hand and the contact hand are swapped. The detector must survive it,
    because the user will eventually do it.
    """
    rng = np.random.default_rng(seed)
    beats = np.asarray(beat_times, dtype=float)
    if duration is None:
        duration = float(beats[-1] + 1.0) if beats.size else 1.0
    t = np.arange(0.0, duration, 1.0 / fs)
    signal = np.zeros_like(t)

    #            offset_s, amplitude_uv, sigma_s
    waves = ((-0.16, 110.0, 0.025),   # P
             (-0.030, -120.0, 0.008),  # Q
             (0.0, 1100.0, 0.008),     # R
             (0.032, -260.0, 0.010),   # S
             (0.25, 300.0, 0.040))     # T
    for beat in beats:
        for offset, amp, sigma in waves:
            centre = beat + offset
            lo = max(0.0, centre - 5 * sigma)
            hi = min(duration, centre + 5 * sigma)
            if hi <= lo:
                continue
            i0, i1 = int(lo * fs), min(t.size, int(hi * fs) + 1)
            seg = t[i0:i1]
            signal[i0:i1] += amp * np.exp(-0.5 * ((seg - centre) / sigma) ** 2)

    if wander_uv:
        signal += wander_uv * np.sin(2 * np.pi * 0.28 * t + 0.7)
        signal += 0.5 * wander_uv * np.sin(2 * np.pi * 0.11 * t + 2.1)
    if mains_uv:
        signal += mains_uv * np.sin(2 * np.pi * mains_hz * t)
    if noise_uv:
        signal += rng.normal(0.0, noise_uv, t.size)
    return (-signal if inverted else signal), t


def ecg_beat_times(
    *, duration: float = 30.0, mean_hr: float = 70.0, rmssd_ms: float = 40.0,
    start: float = 0.35, seed: int = 4,
):
    """Beat times whose successive differences have the requested RMSSD."""
    rng = np.random.default_rng(seed)
    mean_rr = 60.0 / mean_hr
    times = [start]
    sd = (rmssd_ms / 1000.0) / math.sqrt(2.0)
    jitter = 0.0
    while times[-1] < duration - mean_rr:
        jitter = 0.5 * jitter + rng.normal(0.0, sd)
        times.append(times[-1] + max(0.3, mean_rr + jitter))
    return np.array(times[:-1] if times[-1] > duration - 0.3 else times)
