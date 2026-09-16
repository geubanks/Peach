#!/usr/bin/env python3
"""Mutation test for the parity gate.

A passing fixture check is only worth something if a wrong engine would fail it.
This script introduces one deliberate error at a time into the C port, rebuilds,
and re-runs the parity check. A mutation that still passes is a claim the
fixtures cannot defend -- either a decision that does not matter, or a hole in
the fixture's coverage.

Each mutation corresponds to a specific decision: either one of the five
findings in ../docs/FINDINGS.md, or one of the SPEC GAP choices marked in
engine.c. So the output doubles as evidence for which of those decisions are
load-bearing.

Usage:  python3 mutations.py [fixture.json]
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCES = ("engine.c", "engine.h", "json.c", "json.h", "runner.c")


@dataclass(frozen=True)
class Mutation:
    name: str
    why: str
    path: str
    old: str
    new: str
    must_catch: bool = True


MUTATIONS = [
    Mutation(
        "filter: left neighbour only",
        "FINDINGS #2 -- the plan's literal reading of the 20% rule",
        "engine.c",
        "        if (!ok && k + 1 < n && fabs(rr[k] - rr[k + 1]) / rr[k + 1] <= threshold) ok = 1;\n",
        "",
    ),
    Mutation(
        "RMSSD: bridge dropped intervals",
        "FINDINGS #3 -- differencing across a filtered gap",
        "engine.c",
        "        if (keep[k] && keep[k + 1]) {",
        "        if (1) {",
    ),
    Mutation(
        "HR: mean of instantaneous rates",
        "FINDINGS #4 -- Jensen's inequality",
        "engine.c",
        "        if (keep[k]) { sum += rr[k]; count++; }\n    }\n    if (count == 0) return NAN;\n    return sum / (double)count;",
        "        if (keep[k]) { sum += 60000.0 / rr[k]; count++; }\n    }\n    if (count == 0) return NAN;\n    return 60000.0 / (sum / (double)count);",
    ),
    Mutation(
        "interval: 1.96 instead of t",
        "FINDINGS #5 -- the normal shortcut at n_d ~ 5",
        "engine.c",
        "    if (!use_t) return TONE_Z_CRIT_975;\n    if (df <= 0) return NAN;",
        "    if (!use_t) return TONE_Z_CRIT_975;\n    if (df <= 0) return NAN;\n    return TONE_Z_CRIT_975;",
    ),
    Mutation(
        "mean RR: include filtered intervals",
        "SPEC GAP 1 -- 'the mean interval' does not say kept-only",
        "engine.c",
        "        if (keep[k]) { sum += rr[k]; count++; }",
        "        { sum += rr[k]; count++; }",
    ),
    Mutation(
        "distinct hours: guard disabled",
        "SPEC GAP 2 -- how a 'distinct clock hour' is counted",
        "engine.c",
        "        if (distinct < min_distinct_hours) { flat_fit(y, n, fit); return; }",
        "        if (0) { flat_fit(y, n, fit); return; }",
        must_catch=False,
    ),
    Mutation(
        "median: lower of the two middles",
        "SPEC GAP 3 -- the even-count median convention",
        "engine.c",
        "    return 0.5 * (scratch[n / 2 - 1] + scratch[n / 2]);",
        "    return scratch[n / 2 - 1];",
    ),
    Mutation(
        "SD fallback: divide by n",
        "SPEC GAP 4 -- 'sample SD' does not fix the divisor",
        "engine.c",
        "        sigma = sqrt(ss / (double)(n - 1));",
        "        sigma = sqrt(ss / (double)n);",
        must_catch=False,
    ),
    Mutation(
        "baseline: include the boundary window",
        "SPEC GAP 6 -- the trailing window's endpoints",
        "engine.c",
        "        while (lo < k && epochs[lo] < epochs[k] - span) lo++;",
        "        while (lo < k && epochs[lo] <= epochs[k] - span) lo++;",
        must_catch=False,
    ),
    Mutation(
        "baseline: include the window being scored",
        "the choice recorded in FINDINGS 'two smaller choices'",
        "engine.c",
        "        ws->n_baseline = k - lo;",
        "        ws->n_baseline = k - lo + 1;",
    ),
    Mutation(
        "day: UTC date instead of local",
        "SPEC GAP 7 -- 'a day' is never defined",
        "runner.c",
        "    *day = (int)days;  /* the LOCAL date, because Y-M-D was read as local */",
        "    *day = (int)floor(*epoch / 86400.0);",
    ),
    Mutation(
        "sign: HRV channel not negated",
        "Step 5's sign convention",
        "engine.c",
        "    return (w_x * (-z_x) + w_h * z_h) / total;",
        "    return (w_x * z_x + w_h * z_h) / total;",
    ),
]


def build(workdir: Path) -> bool:
    cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if cc is None:
        return False
    result = subprocess.run(
        [cc, "-std=c99", "-O2", "-o", str(workdir / "tone_parity"),
         str(workdir / "runner.c"), str(workdir / "engine.c"), str(workdir / "json.c"), "-lm"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(result.stderr[:2000], file=sys.stderr)
        return False
    return True


def run_parity(workdir: Path, fixture: Path) -> tuple[int, str]:
    result = subprocess.run([str(workdir / "tone_parity"), str(fixture)],
                            capture_output=True, text=True)
    return result.returncode, result.stdout + result.stderr


def apply_mutation(workdir: Path, src: Path, mutation: Mutation) -> bool:
    for name in SOURCES:
        shutil.copy(src / name, workdir / name)
    target = workdir / mutation.path
    text = target.read_text()
    if text.count(mutation.old) != 1:
        print(f"  !! anchor not unique ({text.count(mutation.old)} matches); "
              f"mutation cannot be applied", file=sys.stderr)
        return False
    target.write_text(text.replace(mutation.old, mutation.new))
    return True


def main(argv: list[str]) -> int:
    fixture = Path(argv[1]).resolve() if len(argv) > 1 else HERE.parent / "watch" / "fixtures.example.json"
    if not fixture.exists():
        print(f"no fixture at {fixture}", file=sys.stderr)
        return 2

    results = []
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)

        for name in SOURCES:
            shutil.copy(HERE / name, workdir / name)
        if not build(workdir):
            print("no C compiler available; skipping", file=sys.stderr)
            return 77
        code, out = run_parity(workdir, fixture)
        if code != 0:
            print("baseline parity FAILS before any mutation:\n" + out, file=sys.stderr)
            return 1
        print("baseline: PARITY pass\n")

        for mutation in MUTATIONS:
            if not apply_mutation(workdir, HERE, mutation):
                results.append((mutation, "ERROR"))
                continue
            if not build(workdir):
                results.append((mutation, "BUILD FAIL"))
                continue
            code, _ = run_parity(workdir, fixture)
            results.append((mutation, "caught" if code != 0 else "SURVIVED"))

    width = max(len(m.name) for m in MUTATIONS)
    print(f"{'mutation':<{width}}  {'result':<9}  why it matters")
    print("-" * (width + 60))
    survivors = []
    for mutation, status in results:
        print(f"{mutation.name:<{width}}  {status:<9}  {mutation.why}")
        if status != "caught":
            survivors.append((mutation, status))

    print()
    caught = sum(1 for _, s in results if s == "caught")
    print(f"{caught}/{len(results)} mutations caught by the fixture at 1e-6.")

    hard_failures = [(m, s) for m, s in survivors if m.must_catch]
    if survivors:
        print("\nSurvivors (the fixture cannot tell these apart from the real engine):")
        for mutation, status in survivors:
            note = "EXPECTED" if not mutation.must_catch else "UNEXPECTED"
            print(f"  [{note}] {mutation.name} -- {status}")
        print("\nAn expected survivor is a decision this fixture never exercises, not a")
        print("decision that does not matter. Both still belong in the spec, because the")
        print("next fixture may exercise them.")
    return 1 if hard_failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
