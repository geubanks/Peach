# Tone for Apple Watch — implementation spec

> **Status: DRAFT, not frozen.** Phase 3.1 of `../docs/PLAN.md` freezes this
> document, and Section 6 gates it on Phase 2.2. Two things below are still
> placeholders: the measurement-error variances in §3, which come from Phase 2.1
> test–retest, and `fixtures.json`, which is generated after those weights
> exist. **Do not start §5 until both are filled in.** If Phase 2.2 came back
> "rebuild", this app should not be built at all yet — see §9.

This file is the contract for the watch app. The score is already implemented,
tested and validated in Python at `../tone/`. The Swift here is a **port**, not
a design: where this document and the Python disagree, the Python is right and
this document is a bug.

---

## 1. Data truth

Read this paragraph before proposing any feature.

On a Series 6-era watch running a pre-watchOS-27 OS, Apple records a passive HRV
window roughly **once every two hours — about five usable windows a day**, and
only when you are still. A third-party app **cannot trigger** an HRV reading.
The only way to obtain one on demand is to have the user run a **Mindfulness
(Breathe) session**, which produces a fresh HRV sample and heartbeat series in
about a minute. `heartRateVariabilityRMSSD` exists in the watchOS 27 SDK but is
**not populated on this hardware**, so the app computes RMSSD itself from
`HKHeartbeatSeriesSample`. Background delivery on watchOS is capped at roughly
**hourly** for these types, delivery frequency varies by device, and each
background wake has about **15 seconds of wall-clock budget** before the system
kills the app. A complication must be installed for background delivery to be
scheduled at all. Nothing in this app may assume a sample is available, recent,
or arriving on any schedule.

Consequences that are not negotiable:

- No live or continuous display. There is nothing live to display.
- No animation, trend arrow, or interpolation across gaps. The app shows the
  last score, its interval, and **how old the newest sample is**.
- Every background wake must complete its work in well under 15 s, and must be
  safe to be killed mid-flight.
- The spot-check button does not "take a reading". It tells the user to run a
  one-minute Mindfulness session, then scores the window that results.

## 2. What the app does when nothing has arrived

Asked of this spec, the answer must be unambiguous. **If no new sample has
arrived in six hours, the app shows the last daily score with its interval and
the age of the newest sample. It does not invent a number, does not carry the
previous value forward as if it were current, and does not hide the staleness.**
Same for twelve hours, or two days — only the age changes. If there has never
been enough history to score (see `min_scoring_windows`), it says so in words
and offers the spot check.

## 3. The score

Time-domain only. Per HRV window *i* at local clock hour *t\_i*. This is
Section 3 of `../docs/PLAN.md`, with the implementation details that the Python
settled; `../docs/FINDINGS.md` has the reasoning for each.

**Step 0 — intervals.** From `HKHeartbeatSeriesSample`, RR intervals are the
**differences between consecutive beat timestamps**, in milliseconds: *N* beats
give *N − 1* intervals. (This is the one place the watch and the Health export
genuinely differ — the export's `InstantaneousBeatsPerMinute` entries are one
per interval, so *N* entries give *N* intervals. Getting this off by one is the
likeliest way to fail every fixture at once.)

**Step 1 — artifact filter, then RMSSD.** Keep interval *k* if it is within 20%
of **at least one** of its immediate neighbours:

```
keep[k]  ⇔  |rr[k] − rr[k−1]| / rr[k−1] ≤ 0.20   or   |rr[k] − rr[k+1]| / rr[k+1] ≤ 0.20
```

Endpoints test against their single neighbour. This is deliberately two-sided;
a left-neighbour-only rule cascades and can reject a whole window after one
artifact.

Then, over pairs where **both** intervals were kept **and** were adjacent in the
original series (never bridge a gap):

$$\mathrm{RMSSD}_i = \sqrt{\frac{1}{D}\sum_{\text{kept pairs}}\left(\mathrm{RR}_{k+1}-\mathrm{RR}_k\right)^2}$$

where *D* is the number of such pairs. Drop the window entirely if fewer than
**30 intervals** survive, or fewer than 2 difference pairs.

**Step 2 — logs.**

$$x_i = \ln \mathrm{RMSSD}_i, \qquad h_i = \ln \mathrm{HR}_i, \qquad \mathrm{HR}_i = \frac{60000}{\overline{\mathrm{RR}}}$$

`HR` is 60000 divided by the **mean interval**, not the mean of the
instantaneous rates. The two differ by Jensen's inequality and mixing them up
fails parity.

**Step 3 — circadian baseline.** Over the trailing **28 days, excluding the
window being scored**, fit by ordinary least squares:

$$\hat{y}(t) = M + A\cos\left(\frac{2\pi t}{24}\right) + B\sin\left(\frac{2\pi t}{24}\right)$$

*t* is **local clock hour** in [0, 24), fractional. Fit *x* and *h* separately.
Port `cosinor.solve_3x3` as written: it forms the 3×3 normal equations and runs
Gaussian elimination with partial pivoting, which needs no linear-algebra
dependency and matches the Python to the last bit. If the pivot falls below
1e-9, or there are fewer than 30 baseline windows, or fewer than 5 distinct
clock hours among them, fall back to a flat mean (MESOR only) and flag the
window `flat_baseline`.

**Step 4 — robust z.** Residual $r_i = y_i - \hat{y}(t_i)$; sigma is
**1.4826 × MAD** of the baseline windows' own residuals under the same fit. If
MAD is zero, fall back to the sample SD; if that is below 1e-9, emit **no
score** for the window rather than a division blow-up.

$$Z_{x,i} = \frac{r_{x,i}}{\sigma_{r_x}}, \qquad Z_{h,i} = \frac{r_{h,i}}{\sigma_{r_h}}$$

**Step 5 — combine by precision.**

$$S_i = \frac{w_x\,(-Z_{x,i}) + w_h\,Z_{h,i}}{w_x + w_h}, \qquad w_x = \frac{\lambda}{\sigma^2_{\epsilon,x}}, \quad w_h = \frac{1}{\sigma^2_{\epsilon,h}}$$

Note the sign: the HRV channel enters **negated**, so low vagal tone and high
heart rate both push *S* up.

```
λ                 = 1.0
σ²_ε,x (ln RMSSD) = TODO — from `tone weights`, Phase 2.1
σ²_ε,h (ln HR)    = TODO — from `tone weights`, Phase 2.1
```

These are measured, not chosen. Do not substitute a plausible-looking number to
unblock the build; an unmeasured weight is the thing this whole ordering exists
to prevent.

**Step 6 — daily aggregate.** $\bar{S}_d$ is the mean of the day's $S_i$; the
95% interval is $\bar{S}_d \pm t_{0.975,\,n_d-1}\cdot\sigma_S/\sqrt{n_d}$, with
the *t* table in `score.T_CRIT_975` (df 1–30, then 1.960). Not 1.96 — at
$n_d\approx5$ that understates the interval by 40%. A day with one usable window
borrows the pooled within-day SD from other days and is marked as having done
so. A single window's $S_i$ is displayed **only** when it came from an
on-demand session the user deliberately took.

## 4. Fixtures — the gate on everything else

`fixtures.json` (generated by `python3 -m tone fixtures`) contains the config
block, a series of windows as RR arrays in milliseconds, and the expected
`n_kept`, `rmssd`, `mean_hr`, `x`, `h`, `resid_x`, `resid_h`, `sigma_x`,
`sigma_h`, `z_x`, `z_h`, `s` for the last 50 scored windows, plus the expected
daily aggregates.

`fixtures.example.json` in this directory is the **format** — generated from the
simulator with illustrative weights, so the Swift side can be built and its
decoder tested before the real file exists. It is not anyone's data and must not
be used to claim parity.

**`ScoreEngine` must reproduce every checked field to within 1e-6 (relative,
against `max(1, |expected|)`) in a unit test before any UI code is written.**

- Feed the engine **all** windows in the file, in order. Only those with
  `"checked": true` are asserted; the rest are the baseline history that makes
  them scoreable.
- Non-finite values are written as `null`. Swift's `JSONDecoder` rejects bare
  `NaN`, which is why.
- The `config` block in the fixture is the config to run with. Do not read
  constants from anywhere else.
- If a fixture fails, the Swift is wrong. Do not regenerate the fixture to make
  it pass. Regenerating is a deliberate act that happens only when the Python
  algorithm changes on purpose, and it invalidates the parity claim until the
  Swift is re-verified.

## 5. Build order

Each step is gated on the one before it. `4.1` and `4.2` are the plan's
numbering.

1. **4.1 Project, entitlements, signing.** watchOS app + HealthKit capability +
   background delivery entitlement. Done when the app launches on the wrist with
   no signing error. *(Human does this in Xcode.)*
2. **4.2 `ScoreEngine` + parity tests.** Pure Swift, **no HealthKit import**:
   it takes `[(Date, [Double])]` and returns scores. Done when all fixtures pass
   to 1e-6 and a fresh export of last week's data produces the same daily scores
   in Swift and Python.
3. **4.3 Ingestion.** HealthKit read authorisation, `HKHeartbeatSeriesSample`
   query, `HKObserverQuery` + `enableBackgroundDelivery`, one complication. Done
   when a full day's log shows wake-ups landing (hourly at best) and each one
   finishing inside the 15 s budget. Always call the observer's completion
   handler, including on the error paths.
4. **4.4 UI.** Daily score, its interval, age of the newest sample, and a spot
   check button. Done when the spot check has been used on a bad afternoon and a
   good one and the numbers went the way the diary said.

Persist raw windows (timestamp + RR array), not scores. The baseline is refitted
from history every time, so a change to the algorithm must be able to rescore
the past. 28 days of ~5 windows × ~60 intervals is trivial on disk.

## 6. Things not to build

Listed because they are the obvious suggestions and each one is wrong here.

- **LF/HF or any frequency-domain metric.** Cut from the plan on purpose:
  contested as a sympathovagal index, and it needs a stationary five-minute
  recording that a wrist in daily life never provides.
- **Population norms or "your HRV is X for your age".** The only valid reference
  distribution is the user's own.
- **A score without its interval.** Section 8 of the plan; also §2 above.
- **Anything clinical.** Not a medical device, no thresholds, no advice, no
  notifications that something is wrong.
- **The AFib History workaround** for more frequent samples. It requires
  attesting to a physician diagnosis the user does not have, and it disables
  irregular-rhythm notifications.
- **Apple's readiness score.** Not a HealthKit type, and not on this hardware.

## 7. Repository conventions

- Swift 5.9+, watchOS 10 deployment target, SwiftUI.
- `ScoreEngine` and its tests depend on **nothing** — no HealthKit, no
  SwiftUI, no Foundation beyond `Date`. That is what makes it testable on the
  Mac against the fixtures.
- Keep the Python function names in the Swift (`filterRR`, `rmssd`, `cosinorFit`,
  `robustSigma`, `combine`) so the two can be diffed by eye.
- No new third-party dependencies.

## 8. If you are an agent working in this directory

Do not change the maths in §3 to make something work. Every constant and every
formula here is either measured or argued for in `../docs/FINDINGS.md`. If the
spec looks wrong, say so and stop — the correct fix is to change the Python,
regenerate the fixtures, and update this file deliberately, in that order.

## 9. If Phase 2.2 said "rebuild"

Do not build this app. The plan's own conclusion (Section 7) is that if five
passive windows a day cannot resolve daily stress, the project's value is the
**on-demand spot check alone** — and that is a different, simpler app: a button,
a one-minute Mindfulness session, and a single $S_i$ with its interval. Steps
4.1, 4.2 and the spot-check half of 4.4 still apply unchanged. Everything about
daily aggregation and background delivery does not.
