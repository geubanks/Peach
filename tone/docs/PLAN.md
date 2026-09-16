# Tone: a personal HRV stress index for Apple Watch

*Proposal and build plan, revised 16 September 2026. Personal use, Claude Code writes the Swift, you own the science.*

> Committed verbatim as the project's source of truth. Where implementing it
> forced a decision the plan left open — five of them, each of which changed a
> number — the decision and its argument are in
> [`FINDINGS.md`](FINDINGS.md), not edited into the text below.

The plan you already have is directionally right: baseline-relative scoring, time-domain metrics first, verify against your own wrist. This revision changes four things, and I'll argue each one from evidence: (1) the landscape shifted last week and you now have a real decision about hardware; (2) the LF/HF "stretch goal" should be cut entirely, not deferred; (3) the two-channel score should be weighted by measurement noise, not by a physiology heuristic; and (4) the build order should be inverted so the algorithm is proven in Python on data you already have before a single line of Swift exists.

---

## 1. What changed on 9 September 2026

Apple's new Series 12 and Ultra 4 measure HRV as often as every five minutes and heart rate every five seconds, roughly 24 times the previous HRV cadence [1]. HealthKit gained a second HRV identifier, `heartRateVariabilityRMSSD`, alongside the long-standing SDNN type [2], and Apple now splits the metric into a "Recovery HRV" (daily stress and recovery, compared to your baseline) and an "Overall HRV" [3]. The new readiness score is 0–10, computed from activity, vitals and sleep, and has not been announced as a queryable HealthKit type [4].

Two consequences for you, on a Series 6-era watch:

- The "24×" figure implies the older cadence is about one passive HRV sample every two hours [2]. Developer-forum reports put it at roughly five samples a day without AFib History enabled [5]. That is not enough to detect an acute stress event; it is enough to detect a bad *day*.
- watchOS 27 requires Series 9 or later [4], so the RMSSD type will exist in your SDK but will not be populated on your wrist. Your app must compute RMSSD itself from beat-to-beat data. That was already the right call scientifically (Section 2); now it is also the only option.

> **Decision rule, not a purchase recommendation.** Do not buy a Series 12 for this project yet. Phases 1–2 below run entirely on data your current watch has already collected. Upgrade only if Phase 2 shows the score tracks your diary but is sample-starved, meaning the correlation is real and the confidence intervals are wide because $n$ per day is small. That is the one scenario where 5-minute HRV changes the project from "daily trend" to "near-continuous."

---

## 2. What the evidence actually supports

### 2.1 The stress signature is vagal withdrawal, and RMSSD is the cleanest vagal marker

Kim et al.'s meta-analysis, the paper you read, found that stress reliably reduces HRV, with the most consistent effects in the vagally mediated indices (HF power and RMSSD) and in the LF/HF ratio [6]. The catch is that LF/HF has been under sustained attack as a "sympathovagal balance" index since Billman's 2013 critique: LF power has substantial parasympathetic contribution, is confounded by respiration rate and mechanical effects, and the ratio is not a clean sympathetic measure [7]. The 1996 Task Force standard is still the reference for definitions and recording requirements, and it specifies that frequency-domain analysis wants a stationary five-minute recording [8]. A wrist watch, in daily life, never gives you that.

So the earlier "Phase 5 stretch: LF/HF via your own FFT" should be deleted, not deferred. You would be spending real signal-processing effort to reproduce a metric that the field itself no longer trusts for the specific inference you want to make. RMSSD, by contrast, is almost purely vagal, is stable in ultra-short recordings, and needs no spectral estimation. Munoz et al. showed RMSSD from windows as short as 10–60 seconds correlates very strongly with the five-minute standard, while SDNN needs longer windows to be reliable [9]. Apple's passive HRV windows are on the order of a minute, which means Apple's SDNN number is already a proxy for short-term (respiratory) variability rather than the 24-hour SDNN the population norms describe [10].

**Implication:** compute RMSSD from `HKHeartbeatSeriesSample` as the primary channel. Keep Apple's SDNN only as a sanity check.

### 2.2 Apple Watch HRV is noisy in absolute terms and fine in relative terms

The best recent validation, Series 9 and Ultra 2 against a Polar H10 chest strap in 39 adults over 14 days (316 paired measurements), found the watch underestimates HRV by about 8 ms on average, with a mean absolute percentage error near 29% and mean absolute error near 20 ms; the watch failed a ±10 ms equivalence test. Resting heart rate in the same study had a MAPE under 6% and a mean bias near zero [11]. An earlier study by Hernando et al. found the watch's RR intervals were usable for HRV during both relaxation and induced mental stress, with small errors relative to a chest strap [12].

Three things follow, and they reshape the score:

1. Population "normal ranges" for SDNN are the wrong reference. Your own distribution is the only valid one. (You already concluded this.)
2. The HR channel is about five times more precise than the HRV channel on this hardware. The earlier plan weighted HRV at 0.7 on physiological grounds. On *statistical* grounds, the noisier channel should get *less* weight per unit of information, not more. Section 3 fixes this.
3. The 29% error is largely random, not systematic bias, which is exactly what averaging within a day and comparing against a personal baseline is designed to suppress.

### 2.3 The confounders are bigger than the effect

HRV falls with age, with posture change from supine to standing, with caffeine and alcohol, with poor sleep, with acute exercise, and it varies strongly across the circadian cycle, peaking during sleep [8, 10]. Apple only takes passive HRV samples when you are still, but "still at 9 a.m. after coffee" and "still at 11 p.m. in bed" are different physiological states. A single 28-day mean and standard deviation, as in the earlier formula, would flag every evening as "calm" and every morning as "stressed." The baseline has to be conditioned on time of day.

### 2.4 What a third-party app can and cannot get on your watch

| Source | What you get | Cadence on Series 6–8 | Honest? |
|---|---|---|---|
| `HKQuantityType` SDNN | Apple's one-number HRV per window | ~every 2 h, ~5/day [2, 5] | Yes |
| `HKHeartbeatSeriesSample` | Beat-to-beat intervals for each of those windows; lets you compute RMSSD, pNN50 | Same as above | Yes |
| Mindfulness (Breathe) session | Triggers a fresh HRV + heartbeat-series sample on demand | Whenever you run one (~1 min) | Yes: this is your on-demand "spot check" |
| `HKElectrocardiogram` voltage | 30 s of 512 Hz single-lead ECG; you detect R-peaks yourself | On demand | Yes; a 2023 study used exactly this to quantify stress [13] |
| AFib History enabled | Roughly 10× more passive HRV samples [5, 14] | ~hourly | No: setup requires attesting a physician diagnosis of AFib [15] and disables irregular-rhythm notifications |
| Background delivery | Wake-ups to run a query | Most types capped at hourly on watchOS; ~15 s of execution per wake; throttling varies by device [16, 17, 18] | — |

> **Caution.** The AFib History workaround is widely circulated in the HRV-app community and it works. I am not recommending it: the setup asks you to state that a physician has diagnosed you with atrial fibrillation, and you would be turning off the irregular-rhythm alert that is actually designed for someone your age. The Mindfulness session and the ECG recording give you on-demand samples without lying to a medical feature.

---

## 3. The score, revised

Everything below is time-domain only and computed per HRV window $i$ at clock time $t_i$.

**Step 1: compute RMSSD from the beat-to-beat series.**

$$\mathrm{RMSSD}_i = \sqrt{\frac{1}{N_i-1}\sum_{k=1}^{N_i-1}\left(\mathrm{RR}_{k+1}-\mathrm{RR}_k\right)^2}$$

where $\mathrm{RR}_k$ is the $k$-th inter-beat interval in milliseconds within window $i$, and $N_i$ is the number of beats in that window. Discard any window with fewer than about 30 beats, and discard any interval that differs from its neighbor by more than 20% (a standard artifact filter for ectopic beats and missed detections).

**Step 2: log-transform.** RMSSD is right-skewed; $\ln(\mathrm{RMSSD})$ is close to normal and makes the z-score meaningful.

$$x_i = \ln\left(\mathrm{RMSSD}_i\right), \qquad h_i = \ln\left(\mathrm{HR}_i\right)$$

where $\mathrm{HR}_i$ is the mean heart rate over the same window, from the same intervals: $\mathrm{HR}_i = 60000 / \overline{\mathrm{RR}}$.

**Step 3: circadian baseline, not a flat mean.** Fit a first-harmonic cosinor to your own history over a rolling 28-day window:

$$\hat{x}(t) = M + A\cos\left(\frac{2\pi t}{24}\right) + B\sin\left(\frac{2\pi t}{24}\right)$$

where $t$ is clock hour (0–24), $M$ is your 24-hour mean level (the MESOR), and $A$, $B$ encode the amplitude and phase of your daily rhythm. This is ordinary least squares with two extra columns; it fits in a few lines of Python and a few more of Swift. Fit the same model separately for $h$. The residual $r_i = x_i - \hat{x}(t_i)$ is "how much lower is my vagal tone than it usually is *at this hour*," which is the quantity that actually indexes stress.

**Step 4: z-score the residuals.**

$$Z_{x,i} = \frac{r_{x,i}}{\sigma_{r_x}}, \qquad Z_{h,i} = \frac{r_{h,i}}{\sigma_{r_h}}$$

where $\sigma_{r_x}$ and $\sigma_{r_h}$ are the standard deviations of the residuals over the same 28-day window. Use a robust estimate (median absolute deviation $\times 1.4826$) so one corrupted window doesn't inflate the denominator.

**Step 5: combine by precision, not by opinion.**

$$S_i = \frac{w_x\,(-Z_{x,i}) + w_h\,Z_{h,i}}{w_x + w_h}, \qquad w_x = \frac{1}{\sigma^2_{\epsilon,x}}, \quad w_h = \frac{1}{\sigma^2_{\epsilon,h}}$$

where $\sigma^2_{\epsilon,x}$ and $\sigma^2_{\epsilon,h}$ are the *measurement-error* variances of each channel. You estimate them yourself in Phase 2 by test–retest: take two Mindfulness sessions five minutes apart, at rest, on ten separate occasions; the within-pair variance is your $\sigma^2_\epsilon$ for each channel. If the validation literature holds for your wrist, $w_h$ will come out several times larger than $w_x$, and the score will lean on heart rate more than the earlier 0.7/0.3 heuristic did. That is the correct response to a noisy sensor. If you want the physiological prior back, add a single multiplier $\lambda$ on $w_x$ and tune $\lambda$ against your diary; but start at $\lambda = 1$ so you can see what the data say first.

**Step 6: aggregate.** Report a daily score $\bar{S}_d$ as the mean of $S_i$ over the day's windows, with a 95% interval from the standard error $\sigma_S/\sqrt{n_d}$, where $n_d$ is the number of usable windows that day. Show the interval on the watch. With $n_d \approx 5$ the interval will be wide; that honesty is the product. A single window's $S_i$ is displayed only when it comes from an on-demand session you deliberately took.

---

## 4. What I'd actually do: algorithm first, watch last

Here is the idiosyncratic part. You have three assets the standard "learn watchOS, then build" path ignores:

1. **Years of HRV history already in your phone.** Every passive SDNN sample and heartbeat series your Series 6-era watch has ever taken is sitting in the Health app. Health > Profile > Export All Health Data gives you an XML file with all of it. You do not need to write a watch app to get data; you need to write a watch app to *display* a score. Those are different projects and the second should wait on the first.
2. **You already do this kind of analysis.** Your MIR pipeline computes spectral features, entropy measures, and KL/EMD divergences over time series. RMSSD, a cosinor fit, and a robust z-score are simpler than anything in that codebase. The signal-processing part of this project is a weekend, not a phase.
3. **You already direct coding agents with a spec.** Your Suno styling rubric is a versioned `CLAUDE.md` that converts a JSON schema into constrained output. Do the identical thing here: the Python reference implementation produces a JSON fixture of (input beat series → expected score), the spec says "the Swift implementation must reproduce these fixtures to within $10^{-6}$," and Claude Code cannot drift from your science because the tests won't let it.

So the order becomes: **export → Python → diary → parity tests → Swift**. The watch app is the last thing built, and by the time it is built the score is already known to work, or already known not to, on your own body.

---

## 5. Sequence

### Phase 1 — Prove the signal exists (weeks 1–3, ~3 h/week)

**1.1 Export and parse.**
- Do: export Health data; parse `HKQuantityTypeIdentifierHeartRateVariabilitySDNN` records and the `HeartRateVariabilityMetadataList` / beat-series entries into a tidy DataFrame (timestamp, RR list, SDNN, HR).
- Done when: you can print how many usable windows per day you have had over the last 6 months, and the histogram of their clock times.
- Trigger: if it's Sunday afternoon and you're not on duty, then this before anything else.

**1.2 Implement Section 3 in Python.**
- Do: RMSSD with artifact filter, log transform, rolling 28-day cosinor, robust residual z-scores, $S_i$ with $\lambda=1$ and placeholder equal weights.
- Done when: you can plot $Z_{x}$ and $Z_h$ over the last month and identify, by eye, the three lowest-vagal-tone days, then explain in one sentence each why the cosinor step matters (what the plot looks like with and without it).

**1.3 Start the diary.**
- Do: a Shortcuts automation on your phone that asks "Stress right now, 0–10?" at three anchored times (after your first class, after lab, before bed) and appends to a CSV. This is your only ground truth; nothing else in the project is more important.
- Done when: 14 consecutive days with ≥2 entries/day.
- Trigger: the automation is the trigger; you just answer.

### Phase 2 — Calibrate against yourself (weeks 3–5)

**2.1 Test–retest for $\sigma_\epsilon$.**
- Do: on 10 separate days, two Mindfulness sessions five minutes apart, seated, same posture. Compute within-pair variance of $\ln\mathrm{RMSSD}$ and $\ln\mathrm{HR}$.
- Done when: you have numeric $w_x$ and $w_h$ and can state which channel your wrist measures more reliably.

**2.2 The falsification test.**
- Do: within-person Spearman correlation between $\bar{S}_d$ and the day's mean diary rating; also between each on-demand $S_i$ and the nearest diary entry within 30 minutes.
- Done when: you have $\rho$ with a bootstrap 95% interval. Ambulatory HRV–stress correlations in the literature are modest; treat $\rho \geq 0.3$ with an interval excluding zero as "the signal is real," $\rho$ between 0.1 and 0.3 as "real but sample-starved," and an interval spanning zero as "rebuild" [6, 13].

> **Decision point.** "Real" → proceed to Phase 3 on your current watch. "Sample-starved" → this is the only justified reason to consider a Series 12; alternatively, commit to two scheduled Mindfulness sessions a day and re-test. "Rebuild" → the diary, not the sensor, is the first suspect: check whether your ratings actually vary day to day. Do not write Swift until you are out of this box.

### Phase 3 — Freeze the spec (week 6)

**3.1 Write `CLAUDE.md` for the watch app.**
- Do: one paragraph of data truth (sparse samples, hourly background cap, 15 s wake budget, complication required for background delivery), the Section 3 math verbatim, a JSON fixture file of 50 windows with expected $\mathrm{RMSSD}$, residuals, and $S_i$ exported from Python, and the rule that the Swift `ScoreEngine` must pass the fixtures in a unit test before any UI is written.
- Done when: a second reader (a lab mate) can tell you from the spec alone what the app does when no new sample has arrived in six hours. (Answer: it shows the last daily score with its interval and the age of the newest sample. It does not invent a number.)

### Phase 4 — Build (weeks 7–10)

- **4.1 Project, entitlements, signing.** You click these in Xcode. Done when: the app launches on your wrist with no signing error.
- **4.2 `ScoreEngine` + parity tests.** Claude Code writes it. Done when: all fixtures pass to $10^{-6}$ and a fresh export of last week's data produces the same daily scores in Swift and Python.
- **4.3 HealthKit read + `HKObserverQuery` + hourly background delivery + one complication.** Done when: after a full day the log shows wake-ups landing (expect roughly hourly at best, and less on some devices [18]), and each wake-up finishes inside the 15 s budget [17].
- **4.4 UI.** Daily score with interval, age of newest sample, a "spot check" button that tells you to run a one-minute Mindfulness session and then scores the resulting window. Done when: you have used the spot check on a bad afternoon and a good one and the numbers went the way the diary said.

### Phase 5 — Live with it (ongoing)

Keep the diary running one more month with the app on your wrist and re-run 2.2. Re-fit $\lambda$ only if the on-wrist correlation is worse than the offline one. Leave 20% of your weekly project time unallocated; RA duty weeks and lab deadlines will eat it.

---

## 6. What gates what

- Phase 1.3 (the diary) gates everything downstream and has the longest lead time: it needs calendar days, not effort. Start it the same day you export.
- Phase 2.2 gates Phase 3. The spec must encode weights you measured, not weights you guessed.
- Phase 4.2 gates 4.3 and 4.4. No UI on an unverified engine.
- Nothing here has an external approval. The only waiting is biological.

## 7. What would mean this is wrong

- If the cosinor fit explains less than ~10% of the variance in $\ln\mathrm{RMSSD}$, the circadian correction is unnecessary for you and you can drop back to a flat baseline. That would be surprising but cheap to discover.
- If $w_h \ll w_x$ after test–retest, the validation literature doesn't describe your wrist and you should trust HRV more than I've suggested.
- If Phase 2.2 is "rebuild" twice, the honest conclusion is that five passive windows a day on this hardware cannot resolve daily stress for you, and the project's value becomes the on-demand spot check alone. That is still a useful tool; it is just a different app.

## 8. A note on framing

Keep this a personal instrument. It is not a medical device, and the score should never be shown without its interval. The most defensible sentence you can write about it later, in any context, is: "I built a within-person, precision-weighted, circadian-adjusted HRV index and tested whether it predicted my own diary ratings." That sentence is true whether the answer turned out to be yes or no, and the no is nearly as interesting.

---

## References

1. Apple. [Apple Watch Series 12](https://www.apple.com/apple-watch-series-12/) product page: HRV as often as every five minutes, heart rate every five seconds. Accessed 16 Sept 2026.
2. Crosley B. [Five-Second Heart Rate: What Changes for HealthKit Apps](https://blakecrosley.com/blog/five-second-heart-rate-healthkit). Sept 2026. Notes the new `heartRateVariabilityRMSSD` identifier, the ~2-hour prior cadence implied by "24×," the hourly background-delivery ceiling on watchOS, and that no readiness type exists in HealthKit.
3. Jovin I. [Apple Watch Recovery HRV and Overall HRV are not the same thing](https://gadgetsandwearables.com/2026/09/11/apple-watch-recovery-hrv-overall-hrv-rmssd/). Gadgets & Wearables, 11 Sept 2026.
4. Sahha. [Apple Watch Series 12: readiness, Health Age, Quest labs](https://sahha.ai/blog/apple-event-2026-health-readiness-labs/). Sept 2026. Readiness is 0–10, new-hardware-only, not a HealthKit type; watchOS 27 supports Series 9 and later.
5. Apple Developer Forums. [Heart Rate Variability](https://developer.apple.com/forums/thread/725346) thread: third-party apps cannot trigger HRV readings; AFib History raised daily samples from ~5 to ~50.
6. Kim HG, Cheon EJ, Bai DS, Lee YH, Koo BH. Stress and heart rate variability: a meta-analysis and review of the literature. *Psychiatry Investigation*. 2018;15(3):235–245. doi:10.30773/pi.2017.08.17.
7. Billman GE. The LF/HF ratio does not accurately measure cardiac sympatho-vagal balance. *Frontiers in Physiology*. 2013;4:26. doi:10.3389/fphys.2013.00026.
8. Task Force of the European Society of Cardiology and the North American Society of Pacing and Electrophysiology. Heart rate variability: standards of measurement, physiological interpretation, and clinical use. *Circulation*. 1996;93(5):1043–1065.
9. Munoz ML, van Roon A, Riese H, et al. Validity of (ultra-)short recordings for heart rate variability measurements. *PLoS ONE*. 2015;10(9):e0138921. doi:10.1371/journal.pone.0138921.
10. Shaffer F, Ginsberg JP. An overview of heart rate variability metrics and norms. *Frontiers in Public Health*. 2017;5:258. doi:10.3389/fpubh.2017.00258.
11. Serial-measurement validation of Apple Watch Series 9 and Ultra 2 HRV and resting heart rate against Polar H10 + Kubios, 39 adults, 316 measurements, 14 days. [PMC11478500](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11478500/) (2024). HRV underestimated by ~8.3 ms, MAPE 28.9%, MAE 20.5 ms; RHR MAPE 5.9%.
12. Hernando D, Roca S, Sancho J, Alesanco Á, Bailón R. Validation of the Apple Watch for heart rate variability measurements during relax and mental stress in healthy subjects. *Sensors*. 2018;18(8):2619. doi:10.3390/s18082619.
13. Can heart rate variability data from the Apple Watch electrocardiogram quantify stress? *Frontiers in Public Health*. 2023;11:1178491. [Link](https://www.frontiersin.org/journals/public-health/articles/10.3389/fpubh.2023.1178491/full). 36 participants, two weeks, real-world Apple Watch ECG recordings.
14. Athlytic. [Enabling AFib History](https://athlyticapp.helpscoutdocs.com/article/22-enabling-afib-history): describes the workaround and that it disables irregular-rhythm notifications.
15. Apple Support. [Track your AFib History with Apple Watch](https://support.apple.com/en-is/108375): requires a physician diagnosis of AFib; not intended for people under 22.
16. Junction. [Apple HealthKit guide](https://docs.junction.com/wearables/guides/apple-healthkit): system wakes the app at most once per frequency period; some types capped at hourly.
17. Apple Developer Forums. [watchOS HKObserverQuery crashes in background](https://developer.apple.com/forums/thread/781261): 15-second wall-clock allowance per background wake.
18. Apple Developer Forums. [Abnormal background delivery frequency on specific watchOS devices](https://developer.apple.com/forums/thread/814914): identical builds observed at ~hourly on some Series 10 units and every 8–16 minutes on others.
