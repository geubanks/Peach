# What implementing the plan turned up

Five decisions that changed a number, and one caveat about the export. Each is
also documented where it lives in the code; this file is the argument, not the
reference.

---

## 1. The export's beat data is quantised, and the bias is state-dependent

`export.xml` does not contain inter-beat intervals. It contains
`InstantaneousBeatsPerMinute` entries, and `bpm` is an integer. Reconstructing
RR as `60000 / bpm` therefore lands on a lattice whose spacing is

$$\Delta_{RR} = \left|\frac{d\,RR}{d\,bpm}\right| = \frac{60000}{bpm^2} = \frac{RR^2}{60000}\ \text{ms}$$

— 16.7 ms at 60 bpm, 24 ms at 50 bpm. Treating the rounding error as
independent and uniform over that bin gives a per-interval variance
$q = \Delta_{RR}^2/12$, and since RMSSD² estimates $E[(\Delta RR)^2]$ over a
*difference* of two independently rounded intervals,

$$\mathrm{RMSSD}^2_{\text{observed}} \approx \mathrm{RMSSD}^2_{\text{true}} + 2q$$

Measured against simulated ground truth (200 000 intervals per cell):

| true RMSSD | HR | observed | inflation | after correction |
|---:|---:|---:|---:|---:|
| 15 ms | 50 | 17.36 | +15.7% | 14.33 |
| 25 ms | 50 | 26.77 | +7.1% | 24.91 |
| 35 ms | 50 | 36.37 | +3.9% | 35.02 |
| 55 ms | 50 | 55.85 | +1.6% | 54.98 |
| 15 ms | 60 | 16.49 | +9.9% | 15.02 |
| 35 ms | 60 | 35.70 | +2.0% | 35.04 |
| 15 ms | 75 | 15.62 | +4.2% | 15.00 |
| 35 ms | 75 | 35.38 | +1.1% | 35.11 |

Reproduce with `python3 -m pytest tests/test_metrics.py -k quantization`.

**Why this is worse than a constant bias.** The inflation is largest where RMSSD
is smallest — your stressed windows — and grows as heart rate falls, which is
where your sleep windows are. So it does not shift the score, it compresses it.
At 60 bpm, a true contrast of $\ln(35/15) = 0.847$ reads as $\ln(35.7/16.5) =
0.772$: about 9% of the dynamic range, gone, before any of the sensor noise the
validation literature describes.

**What to do about it.** `quantization_correction: true` in the config applies
$\sqrt{\max(0, \mathrm{RMSSD}^2 - 2q)}$, which recovers the truth to within
~0.1 ms over most of the table. It breaks down in the top-left cell (15 ms at
50 bpm, corrected to 14.33): when the quantisation step approaches the RR
spread, the rounding error stops being independent of the signal and the
additive model over-corrects. Use the correction, and do not treat export RMSSD
as a precise instrument at low HR.

**Why it does not sink the project.** The HR channel is untouched — `mean(RR)`
averages the rounding away, and Step 5 will probably weight HR heavily anyway.
And `HKHeartbeatSeriesSample`, which the watch app reads, carries actual beat
timing. The Swift engine sees *better* data than Phase 1 does. Phase 1's job is
to establish that the pipeline runs and that the signal exists at all; if it
shows a real correlation through a 9% compression, the on-wrist version can only
improve on it.

**Check your parse before you trust it.** `tone parse` compares our recomputed
SDNN against Apple's own per-window SDNN. Apple computed theirs from unrounded
intervals, so agreement to within a few ms across thousands of windows means the
reconstruction is sound. (SDNN is the right check precisely because
quantisation barely touches it — it is dominated by the real spread. RMSSD is
the metric that suffers.) Wild disagreement means the parse is wrong, and
nothing downstream is interpretable until it is fixed.

---

## 2. The artifact filter has to be two-sided

"Discard any interval that differs from its neighbour by more than 20%" is
unambiguous until you implement it. With a single left neighbour as the
reference, a rejected interval leaves the reference stale, and the next interval
is compared against a value that is no longer where the series is. One artifact
can reject the entire remainder of a window.

The rule implemented in `metrics.filter_rr` keeps an interval when it agrees to
within 20% with *either* immediate neighbour. This is stateless, symmetric, and
behaves correctly on both cases that matter:

- an isolated ectopic beat (short interval, compensatory long one) disagrees
  with both its neighbours and is dropped, along with its compensation;
- a genuine heart-rate drift agrees with its neighbours all the way down and is
  kept in full.

`tests/test_metrics.py::test_filter_does_not_cascade_after_a_rejection` is the
regression test for the version that does not work.

## 3. RMSSD must not bridge a dropped interval

Once an interval is filtered out, the difference between the intervals on either
side of the gap is not a successive difference — it spans a beat that was
rejected precisely because it was wrong. Computing it manufactures the artifact
the filter just removed, and it manufactures a large one.

`metrics.rmssd` sums only over pairs where both intervals survived *and* were
adjacent in the original series. On a clean window this is identical to the
textbook formula; on a window with an ectopic beat it is the difference between
a correct answer and a badly inflated one.

## 4. `HR = 60000 / mean(RR)`, not `mean(60000 / RR)`

Step 2 of the plan defines it the first way. The second is what you get by
averaging the exported bpm values directly, and it is what you would naturally
write. By Jensen's inequality the mean of the rates always exceeds the rate of
the mean, by roughly $\mathrm{CV}^2$ in relative terms: about 0.1% at typical
resting variability, larger in a variable window.

It is a small difference. It is also a guaranteed 1e-6 parity failure between
Python and Swift if the two sides choose differently, and it is the kind of bug
that takes an afternoon to find. `tests/test_metrics.py` pins it.

## 5. The daily interval should use Student's t

The plan says a 95% interval from $\sigma_S/\sqrt{n_d}$, which implies the 1.96
multiplier. With $n_d \approx 5$ windows a day, the standard deviation is itself
estimated from four degrees of freedom, and the correct multiplier is
$t_{0.975,4} = 2.776$ — 42% wider.

Since the interval *is* the product ("Show the interval on the watch... that
honesty is the product"), understating it by 40% defeats the purpose. The
default is `interval: "t"`, backed by a 30-entry table that ports to Swift
verbatim and converges to 1.96 by construction. `interval: "normal"` restores
the plan's literal formula.

---

## Two smaller choices, recorded so they are not re-litigated

**The trailing baseline excludes the window being scored.** Including it lets
each sample pull its own baseline towards itself and shrink its own residual.
The effect is small at $n \approx 140$, but excluding it is also what the
deployed watch app necessarily does — it scores a new sample against history
that does not contain it — so the offline and on-wrist numbers agree.

**A day with one usable window still gets an interval.** It borrows the pooled
within-day standard deviation from every other day and is marked `sd_pooled`.
The alternative was showing a number with no interval at all, which is the one
thing Section 8 of the plan says never to do. The borrowing is a real
assumption — this day is no noisier than your usual day — and it is flagged in
the output rather than hidden.

---

## One thing the simulator makes visible that the plan does not mention

The cosinor baseline can eat part of the signal it exists to reveal. Stress
itself has a time-of-day shape: higher in the working afternoon, lower asleep.
Any component of stress that is reliably circadian is, by construction,
indistinguishable from the circadian baseline, and Step 3 will absorb it.

In the simulator this is visible directly — the injected circadian amplitude is
0.30 and the fitted amplitude comes back near 0.41, the difference being the
circadian part of the injected stress. `tone demo` prints this.

It is still the right trade. The alternative, a flat baseline, leaves a residual
that swings ±0.5 in ln RMSSD with the clock alone (see
`figures/baseline_comparison.png` from `tone plot`), which would flag every
evening as calm and every morning as stressed — exactly the failure Section 2.3
predicts. But it means the score measures *deviation from your usual day*, not
total stress, and a day that is stressful on schedule will read closer to
neutral than it felt. Worth knowing before you interpret a low score on a
predictably bad Tuesday.
