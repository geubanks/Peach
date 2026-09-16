import numpy as np

from tone import doctor
from tone.config import ScoreConfig
from tone.diary import DiaryEntry
from tone.simulate import simulate


def statuses(checks):
    return {c.name: c.status for c in checks}


def test_a_mature_project_is_clear_to_proceed():
    windows, entries, _ = simulate(days=100, seed=3)
    cfg = ScoreConfig(sigma2_eps_x=0.03, sigma2_eps_h=0.001)
    checks, action = doctor.analyse(windows, entries, cfg)
    by_name = statuses(checks)
    assert by_name["usable windows"] == doctor.PASS
    assert by_name["measured weights"] == doctor.PASS
    assert by_name["statistical power"] == doctor.PASS
    assert "Nothing is blocking" in action


def test_no_diary_is_the_loudest_possible_finding():
    # It gates everything and has the longest lead time, so it must dominate.
    windows, _, _ = simulate(days=60, seed=4)
    checks, action = doctor.analyse(windows, [], ScoreConfig())
    assert statuses(checks)["diary"] == doctor.FAIL
    assert "TODAY" in action


def test_a_fourteen_day_project_is_told_not_to_run_the_test_yet():
    # The single most important thing doctor does: at the sample size the plan
    # schedules, it says how many more days rather than letting you read a
    # verdict the data cannot support.
    windows, entries, _ = simulate(days=14, seed=5)
    checks, _ = doctor.analyse(windows, entries, ScoreConfig())
    power_check = next(c for c in checks if c.name == "statistical power")
    assert power_check.status == doctor.TODO
    assert "more overlapping days" in power_check.action
    assert "Do NOT run it" in power_check.action


def test_unmeasured_weights_are_flagged_with_the_right_instruction():
    windows, entries, _ = simulate(days=60, seed=6)
    checks, _ = doctor.analyse(windows, entries, ScoreConfig())
    weights = next(c for c in checks if c.name == "measured weights")
    assert weights.status == doctor.TODO
    assert "Mindfulness" in weights.action or "tone weights" in weights.action


def test_measured_weights_pass_and_report_the_split():
    windows, entries, _ = simulate(days=60, seed=6)
    cfg = ScoreConfig(sigma2_eps_x=0.04, sigma2_eps_h=0.001)
    checks, _ = doctor.analyse(windows, entries, cfg)
    weights = next(c for c in checks if c.name == "measured weights")
    assert weights.status == doctor.PASS
    assert "HRV" in weights.detail and "HR" in weights.detail


def test_a_flat_diary_is_warned_about_before_it_is_blamed_on_the_sensor():
    windows, entries, _ = simulate(days=60, seed=7)
    flat = [DiaryEntry(e.when, 5.0) for e in entries]
    checks, _ = doctor.analyse(windows, flat, ScoreConfig())
    warn = next(c for c in checks if c.name == "diary variance")
    assert warn.status == doctor.WARN
    assert "first suspect" in warn.action


def test_an_empty_window_file_fails_rather_than_reporting_zeroes():
    checks, action = doctor.analyse([], [], ScoreConfig())
    assert checks[0].status == doctor.FAIL
    assert "tone parse" in checks[0].action


def test_a_low_r2_baseline_points_at_section_7():
    # Section 7: below ~10% the circadian correction is not earning its place.
    from datetime import datetime, timedelta, timezone
    from tone.score import Window

    tz = timezone(timedelta(hours=-4))
    rng = np.random.default_rng(0)
    windows = []
    for i in range(240):
        when = datetime(2026, 5, 1, tzinfo=tz) + timedelta(hours=3.1 * i)
        rr = 60000.0 / 62.0 + rng.normal(0, 30, 60)   # no rhythm at all
        windows.append(Window(start=when, rr=rr))
    entries = [DiaryEntry(w.start, float(rng.integers(0, 10))) for w in windows]
    checks, _ = doctor.analyse(windows, entries, ScoreConfig())
    baseline = next(c for c in checks if c.name == "circadian baseline")
    assert baseline.status == doctor.WARN
    assert "flat baseline" in baseline.action


def test_report_renders_every_check_and_its_action():
    windows, entries, _ = simulate(days=20, seed=8)
    checks, _ = doctor.analyse(windows, entries, ScoreConfig())
    text = doctor.report(checks)
    for c in checks:
        assert c.name in text
        if c.action:
            assert c.action.split(".")[0][:30] in text


def test_next_action_prefers_the_most_blocking_phase():
    checks = [
        doctor.Check("2.2", "power", doctor.TODO, "", "collect more days"),
        doctor.Check("1.3", "diary", doctor.FAIL, "", "start the diary"),
    ]
    assert "start the diary" in doctor.next_action(checks)


def test_next_action_is_positive_when_nothing_blocks():
    checks = [doctor.Check("1.1", "windows", doctor.PASS, "fine")]
    assert "Nothing is blocking" in doctor.next_action(checks)
