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

---

## Install

```bash
cd tone
python3 -m pip install -e .            # numpy only
python3 -m pip install -e '.[plots,dev]'   # add matplotlib and pytest
python3 -m pytest                      # 68 tests, ~4 s
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

# Phase 2.1  ten days of paired Mindfulness sessions, then:
python3 -m tone weights data/windows.jsonl --out data/config.json

# Phase 2.2  the falsification test
python3 -m tone validate data/windows.jsonl \
        --diary data/diary.csv --config data/config.json

# Phase 3.1  only once 2.2 says "real"
python3 -m tone fixtures data/windows.jsonl \
        --config data/config.json --out watch/fixtures.json
```

`data/` is gitignored. Your health data does not belong in a repository.

`tone validate` exits 0 on "real" or "sample-starved" and 2 on "rebuild", so it
can gate a script if you want it to.

## What the modules are

| Module | What it owns |
|---|---|
| `config.py` | every scoring constant, serialisable, travels with the fixtures |
| `metrics.py` | artifact filter, RMSSD, mean HR, SDNN, pNN50, quantisation correction |
| `cosinor.py` | the first-harmonic baseline, solved by an explicit 3×3 for portability |
| `score.py` | rolling baseline, robust z, precision weighting, daily interval |
| `calibrate.py` | Phase 2.1 test–retest → the two measurement-error variances |
| `diary.py` | Phase 2.2 Spearman ρ with a cluster bootstrap, and the decision box |
| `parse_export.py` | Apple Health XML/ZIP → windows, streamed |
| `simulate.py` | a synthetic body with known parameters |
| `fixtures.py` | the parity contract for the Swift port |
| `store.py` | one JSONL format every command reads |
| `plots.py` | optional figures for Phase 1.1 / 1.2 |

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

## What this cannot tell you yet

The estimator is validated against a simulator, not against your wrist. Every
number `tone demo` prints is a statement about the code. The only test that can
say anything about *you* is Phase 2.2, and it needs the diary, which needs
calendar days. Start it today; it has the longest lead time of anything in the
plan.
