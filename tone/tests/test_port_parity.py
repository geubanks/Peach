"""The C port must stay in parity, and the parity gate must keep its teeth.

Two different claims, both worth a test:

1. `port/` reproduces the Python's numbers on both fixtures to 1e-6. This is
   the Phase 4.2 gate, rehearsed in a language that can actually be compiled
   here.
2. A deliberately wrong engine *fails* that gate. A green check that cannot go
   red is decoration, and the mutation suite is what makes the green mean
   something.

Skipped entirely when no C compiler is available.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PORT = ROOT / "port"
SOURCES = ("engine.c", "engine.h", "json.c", "json.h", "runner.c")
FIXTURES = (ROOT / "watch" / "fixtures.example.json", ROOT / "watch" / "fixtures.edge.json")

cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
needs_cc = pytest.mark.skipif(cc is None, reason="no C compiler available")


@pytest.fixture(scope="module")
def binary(tmp_path_factory):
    work = tmp_path_factory.mktemp("port")
    for name in SOURCES:
        shutil.copy(PORT / name, work / name)
    out = work / "tone_parity"
    result = subprocess.run(
        [cc, "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror", "-o", str(out),
         str(work / "runner.c"), str(work / "engine.c"), str(work / "json.c"), "-lm"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    return out


@needs_cc
@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda p: p.name)
def test_c_port_reproduces_the_python(binary, fixture):
    result = subprocess.run([str(binary), str(fixture)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PARITY: pass" in result.stdout


@needs_cc
def test_the_fixtures_actually_check_something(binary):
    # A runner that silently checked zero windows would also print "pass".
    for fixture in FIXTURES:
        out = subprocess.run([str(binary), str(fixture)], capture_output=True, text=True).stdout
        checked = [line for line in out.splitlines() if "checked" in line]
        assert checked
        assert " 0 checked" not in out


@needs_cc
def test_the_edge_fixture_catches_every_mutation():
    # 12/12 is the claim in watch/CLAUDE.md section 4. If a mutation starts
    # surviving, either the engine stopped depending on that decision or the
    # edge fixture stopped exercising it -- both are regressions in the gate.
    result = subprocess.run(
        [sys.executable, str(PORT / "mutations.py"), str(ROOT / "watch" / "fixtures.edge.json")],
        capture_output=True, text=True, cwd=str(PORT),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "12/12 mutations caught" in result.stdout, result.stdout


@needs_cc
def test_the_ordinary_fixture_alone_would_not_be_enough():
    # This is the justification for shipping a second fixture at all. If it ever
    # passes 12/12, the edge file has become redundant and should be re-examined
    # rather than kept out of habit.
    result = subprocess.run(
        [sys.executable, str(PORT / "mutations.py"), str(ROOT / "watch" / "fixtures.example.json")],
        capture_output=True, text=True, cwd=str(PORT),
    )
    assert result.returncode == 0, result.stdout + result.stderr  # no *unexpected* survivors
    assert "9/12 mutations caught" in result.stdout, result.stdout


def test_edge_scenarios_reach_every_branch_they_exist_for():
    # Pure Python; runs with or without a compiler.
    from tone import edge_cases
    from tone.config import ScoreConfig
    from tone.score import daily_scores, score_windows

    cfg = ScoreConfig()
    scores = score_windows(edge_cases.all_windows(), cfg)
    counts = edge_cases.coverage(scores, daily_scores(scores, cfg))
    for branch in edge_cases.REQUIRED_BRANCHES:
        assert counts[branch] > 0, f"{branch} not reached:\n{edge_cases.report(scores)}"


def test_edge_windows_are_deterministic():
    from tone import edge_cases

    a = edge_cases.all_windows()
    b = edge_cases.all_windows()
    assert len(a) == len(b)
    for wa, wb in zip(a, b):
        assert wa.start == wb.start
        assert (wa.rr == wb.rr).all()
