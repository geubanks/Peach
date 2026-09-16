# Tone

A within-person, precision-weighted, circadian-adjusted HRV stress index.

This repository is the **Python reference implementation** and the frozen spec
for the watch app that comes later. It follows the build order in
[`docs/PLAN.md`](docs/PLAN.md): export → Python → diary → parity tests → Swift.
The watch app is the last thing built, and by the time it is built the score is
already known to work, or already known not to, on your own body.

**There is deliberately no Swift here yet.** Section 6 of the plan gates the
spec on Phase 2.2 and gates the engine on the spec. Writing `ScoreEngine.swift`
before the falsification test would encode weights you guessed. What exists
instead is everything Phase 2.2 needs, plus the fixture generator that will make
the Swift port unable to drift from the science.

What *does* exist is [`port/`](port/README.md): the same engine in C99, written
from `watch/CLAUDE.md` alone, passing both fixtures at 1e-6 and mutation-tested
to prove the gate has teeth. It is the Phase 4.2 rehearsal — it found eight
places where the spec was ambiguous, all now closed — and it is a near-mechanical
transcription source for the Swift when the gate opens.

---

## Install

```bash
cd tone
python3 -m pip install -e .            # numpy only
python3 -m pip install -e '.[plots,dev]'   # add matplotlib and pytest
python3 -m pytest                      # 129 tests, ~18 s (skips port/ without a C compiler)
```

Everything also runs without installing: `python3 -m tone <command>`.

## See it work before you have any data

```bash
python3 -m tone demo
```

This generates a synthetic body with a known circadian amplitude, known stress
coupling and known measurement noise, then runs Phases 1–2 against it and asks
whether the estimator recovers what was injected. It is how you check the
pipeline is sound while your diary is still accumulating days. The `rho` it
reports is implausibly high on purpose — the simulated diary is generated from
the same latent stress that moves the simulated HRV, so it measures the
estimator, not the physiology.

## The real sequence

```bash
# Phase 1.1  Health > Profile > Export All Health Data, then:
python3 -m tone parse ~/Downloads/export.zip --out data/windows.jsonl
python3 -m tone report data/windows.jsonl

# Phase 1.2
python3 -m tone score data/windows.jsonl
python3 -m tone plot  data/windows.jsonl --out figures/

# Phase 1.3  start the diary the same day (see docs/DIARY.md)

# Optional but worth it: ECGs are the most precise RR your watch can give
python3 -m tone ecg ~/Downloads/apple_health_export --out data/ecg_windows.jsonl

# Phase 2.1  ten days of paired Mindfulness sessions, then:
python3 -m tone weights data/windows.jsonl --out data/config.json

# Phase 2.2  the falsification test -- but check you have the days for it first
python3 -m tone power
python3 -m tone validate data/windows.jsonl \
        --diary data/diary.csv --config data/config.json

# Phase 3.1  only once 2.2 says "real"
python3 -m tone fixtures data/windows.jsonl \
        --config data/config.json --out watch/fixtures.json
```

`data/` is gitignored. Your health data does not belong in a repository.

`tone validate` exits 0 on "real" or "sample-starved", 2 on "rebuild" and 3 on
"underpowered", so it can gate a script if you want it to.

## What the modules are

| Module | What it owns |
|---|---|
| `config.py` | every scoring constant, serialisable, travels with the fixtures |
| `metrics.py` | artifact filter, RMSSD, mean HR, SDNN, pNN50, quantisation correction |
| `cosinor.py` | the first-harmonic baseline, solved by an explicit 3×3 for portability |
| `score.py` | rolling baseline, robust z, precision weighting, daily interval |
| `calibrate.py` | Phase 2.1 test–retest → the two measurement-error variances |
| `diary.py` | Phase 2.2 Spearman ρ with a cluster bootstrap, and the decision box |
| `power.py` | how many diary days Phase 2.2 needs before it can answer anything |
| `parse_export.py` | Apple Health XML/ZIP → windows, streamed |
| `simulate.py` | a synthetic body with known parameters |
| `fixtures.py` | the parity contract for the Swift port |
| `store.py` | one JSONL format every command reads |
| `ecg.py` | R-peak detection from `HKElectrocardiogram` voltage — the most precise RR available |
| `edge_cases.py` | deterministic windows that reach the degenerate branches |
| `plots.py` | optional figures for Phase 1.1 / 1.2 |

And outside the package: [`port/`](port/README.md), a C99 implementation of the
same engine with a fixture runner and a mutation suite.

## One finding that changes the schedule

**Phase 2.2, run when the plan schedules it, would probably say "rebuild" on a
real signal.** At 14 days — Phase 1.3's "done when" — the smallest ρ whose 95%
interval excludes zero is **0.54**. Phase 2.2's own threshold for "the signal is
real" is **0.30**. So a perfectly real ρ of 0.3, measured at 14 days, yields an
interval spanning zero, which the decision box reads as "do not write Swift".

Resolving ρ = 0.3 takes about **46 overlapping diary-and-score days** (≈90 for
80% power). Run `tone power` for the tables. Two things changed as a result:

- a fourth verdict, **`underpowered`**, for an interval that spans both zero and
  0.30 — consistent with no effect *and* with the effect you are looking for, so
  it settles nothing. Distinct from `rebuild`, which now means the interval
  excludes 0.30 and you have genuinely ruled the effect out. `tone validate`
  exits 3 rather than 2, and reports how many more days you need;
- the 14-day diary milestone is documented as a habit checkpoint, not an
  analysis gate.

The method is Fisher-z on Spearman's ρ, with empirical coverage of 94.4–96.0%
against simulated truth and agreement with the shipped cluster bootstrap to
within ~0.05 at n = 14. Both checks are in `tests/test_power.py`.

## Five things the implementation settled that the plan left open

Each of these changed a number, so each is documented where it lives rather than
only here. [`docs/FINDINGS.md`](docs/FINDINGS.md) has the full argument.

1. **The export's beat data is quantised to whole bpm**, which inflates RMSSD by
   about +2% at 35 ms and 60 bpm, +10% at 15 ms and 60 bpm, and +16% at 15 ms
   and 50 bpm. Because the bias is largest exactly where RMSSD is lowest, it
   compresses the calm-to-stressed contrast by roughly 9% on the log scale
   rather than merely shifting it. The HR channel is unaffected.
   `HKHeartbeatSeriesSample` on the watch does not have this problem, so the
   Swift engine will see cleaner data than Phase 1 does. There is an opt-in
   bias correction.
2. **"Differs from its neighbour by more than 20%" needs a two-sided reading.**
   With one left neighbour the rule cascades: a single artifact can reject the
   rest of the window. Keeping an interval that agrees with *either* neighbour
   drops ectopic beats and tolerates genuine drift, with no rolling state to
   port.
3. **RMSSD must not bridge a dropped interval.** Computing the difference across
   a gap manufactures exactly the artifact the filter removed.
4. **`HR = 60000 / mean(RR)`, not the mean of instantaneous rates.** Jensen's
   inequality makes the second systematically larger. It is a small difference
   and a guaranteed parity failure.
5. **The daily interval uses Student's t, not 1.96.** At `n_d ≈ 5` the normal
   value understates the interval by about 40%, and the honesty of the interval
   is the product. `interval: "normal"` in the config restores the plan's
   literal formula.

Two smaller choices, both in `score.py`: the trailing baseline excludes the
window being scored (which is also what the deployed app necessarily does), and
a day with one usable window borrows the pooled within-day SD and is flagged
rather than shown without an interval.

## Is the spec good enough to build from?

Phase 3.1's "done when" is that a second reader can build the app from the spec
alone. That got tested rather than assumed: `port/engine.c` is the engine
written from `watch/CLAUDE.md` without reading the Python. It reached parity —
and found **eight places where the spec did not determine an answer** (is "the
mean interval" over kept intervals only? what is "a day"? which median
convention? open or closed endpoints on the 28-day window?). Seven of the eight
change the numbers enough to fail parity. All are now written into the spec.

Then the gate itself got tested. `port/mutations.py` breaks the engine twelve
ways — once per documented decision — and checks that the fixtures notice:

| fixture | mutations caught |
|---|---|
| `fixtures.example.json` (ordinary data) | 9 / 12 |
| `fixtures.edge.json` (degenerate branches) | **12 / 12** |

The three that survived ordinary data were the flat-baseline guard, the
MAD-is-zero fallback and the exact 28-day boundary — branches a real body never
reaches. A port guessing wrong on those would have passed and shipped, which is
why `watch/fixtures.edge.json` now exists and why the engine must pass both
files. The mutation suite also found a genuine bug in the C runner's timestamp
parsing that nothing else in the harness could have caught.

## What this cannot tell you yet

The estimator is validated against a simulator, not against your wrist. Every
number `tone demo` prints is a statement about the code. The only test that can
say anything about *you* is Phase 2.2, and it needs the diary, which needs
calendar days. Start it today; it has the longest lead time of anything in the
plan.
