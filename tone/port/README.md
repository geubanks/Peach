# `port/` — a second implementation, in C

```bash
make check                                   # parity against the example fixture
make check FIXTURE=../watch/fixtures.edge.json
python3 mutations.py ../watch/fixtures.edge.json
```

## What this is

Phase 4.2 of the plan says the Swift `ScoreEngine` must reproduce the Python
fixtures to 1e-6 before any UI is written. No Swift toolchain was reachable in
the environment this was built in (`download.swift.org` is blocked by the egress
policy), and shipping a Swift engine that had never been compiled would be
exactly the unverified engine the fixture discipline exists to prevent.

So the engine is here in C99 instead — and written **from `../watch/CLAUDE.md`
alone**, without reading the Python. That turns an unfortunate constraint into
the more interesting experiment, because Phase 3.1's "done when" is:

> a second reader can tell you from the spec alone what the app does

This is that second reader, in a language where nothing can be hand-waved.

## What it found

The C passed parity on its first run. That is the less interesting half. The
useful half is the **eight places where the spec did not determine an answer**
and I had to choose; each is marked `SPEC GAP` in `engine.c`, and each has since
been written into the spec:

| # | The question the spec did not answer |
|---|---|
| 1 | Is "the mean interval" over all intervals, or only the kept ones? |
| 2 | What counts as a "distinct clock hour"? |
| 3 | Which median convention for an even-sized set? |
| 4 | Does the fallback "sample SD" divide by *n* or *n − 1*? |
| 5 | What should the engine do when the weights are still TODO? |
| 6 | Are the trailing window's endpoints open or closed? |
| 7 | What is "a day"? |
| 8 | How exactly is the pooled within-day SD pooled? |

None of these is exotic. Every one is a coin-flip a Swift author would also have
had to make, and seven of the eight change the numbers enough to fail parity.
Finding them cost one afternoon in C; finding them in Swift would have cost the
same afternoon plus a rebuild cycle on a watch.

## Mutation testing, and why the edge fixture exists

A passing fixture check is only worth something if a wrong engine fails it.
`mutations.py` introduces one deliberate error at a time — the plan's literal
one-sided artifact filter, bridging a filtered gap in RMSSD, HR as a mean of
rates, 1.96 instead of *t*, and one per spec gap — then rebuilds and re-runs.

The result that mattered:

| fixture | mutations caught |
|---|---|
| `fixtures.example.json` (ordinary data) | **9 / 12** |
| `fixtures.edge.json` (degenerate branches) | **12 / 12** |

The three that survived the ordinary fixture were the flat-baseline guard, the
MAD-is-zero fallback and the 28-day boundary — branches that a body producing
five windows a day spread across the clock simply never takes. A port that
guessed wrong on any of them would have passed the gate and shipped. That is
what `tone.edge_cases` and `watch/fixtures.edge.json` are for.

**The mutation suite also caught a real bug in this directory**, which is the
best argument for it. `parse_iso` scanned forward from a fixed index for the UTC
offset sign, and a guard meant to skip the date's hyphens skipped the offset's
sign too on whole-second timestamps. Every epoch came out shifted by a constant
— invisible to the baseline window, which uses only differences, and visible
*only* in the day grouping. It showed up as a mutation that was caught by one
fixture and survived the other. Nothing else in the harness would have found it.

## Files

| file | what it is |
|---|---|
| `engine.h` / `engine.c` | the score: filter, RMSSD, cosinor, robust z, combine, daily. C99 + libm, no allocation inside the maths. |
| `json.h` / `json.c` | a small JSON reader, so the runner has no dependencies |
| `runner.c` | loads a fixture, feeds every window, asserts each checked field |
| `mutations.py` | breaks the engine twelve ways and checks the gate notices |
| `Makefile` | `make`, `make check`, `make clean` |

`tests/test_port_parity.py` runs all of it under pytest and skips cleanly where
no compiler exists.

## Transcribing to Swift

`engine.c` is deliberately boring: scalar loops, fixed-size arrays for the 3×3
solve, no linear algebra, no allocation in the numeric paths. The mapping is
close to mechanical —

- `double` → `Double`, `unsigned char keep[]` → `[Bool]`
- `tone_solve3` → a function over a `(3,3)` tuple or a flat `[Double]`; keep the
  partial pivoting and the `1e-9` threshold exactly
- `qsort` in `tone_median` → `sorted()`, keeping the even-count convention
- `NAN` → `Double.nan`, `isfinite` → `.isFinite`

Keep the Python names (`filterRR`, `rmssd`, `cosinorFit`, `robustSigma`,
`combine`) so all three implementations can be diffed by eye. Then point the
Swift test at the same two fixture files, and run `mutations.py`'s twelve
mutations against the Swift too — if the Swift test suite cannot catch all
twelve, it is not yet the gate this one is.
