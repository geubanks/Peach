import math

import numpy as np
import pytest

from tone import power
from tone.diary import bootstrap_rho, classify, spearman


def bivariate(n, target_rho, rng):
    """Draw n pairs whose Spearman rho is approximately `target_rho`."""
    r = 2 * math.sin(math.pi * target_rho / 6)  # Pearson r giving this Spearman rho
    x = rng.normal(size=n)
    y = r * x + math.sqrt(1 - r * r) * rng.normal(size=n)
    return x, y


def test_the_finding_fourteen_days_cannot_resolve_the_plans_own_threshold():
    # This is the whole reason power.py exists. If it ever stops being true,
    # the schedule advice in the docs is wrong and should change with it.
    assert power.min_detectable_rho(14) > power.REAL_THRESHOLD
    assert power.min_detectable_rho(14) == pytest.approx(0.543, abs=0.01)
    assert power.days_required(power.REAL_THRESHOLD) == pytest.approx(46, abs=1)


def test_min_detectable_rho_falls_with_sample_size():
    values = [power.min_detectable_rho(n) for n in (14, 21, 28, 45, 60, 90, 120)]
    assert all(a > b for a, b in zip(values, values[1:]))
    # 45 is a whisker short (0.3017); 46 is the first day that clears it.
    assert power.min_detectable_rho(45) > power.REAL_THRESHOLD
    assert power.min_detectable_rho(46) <= power.REAL_THRESHOLD


def test_days_required_and_min_detectable_rho_are_inverses():
    for rho in (0.2, 0.3, 0.4, 0.5):
        n = power.days_required(rho)
        assert power.min_detectable_rho(n) <= rho
        assert power.min_detectable_rho(n - 1) > rho


def test_planning_for_eighty_percent_power_costs_about_double():
    for rho in (0.2, 0.3, 0.4):
        assert power.days_for_power(rho) > power.days_required(rho)
        ratio = power.days_for_power(rho) / power.days_required(rho)
        assert 1.8 < ratio < 2.1


def test_days_for_power_rejects_unsupported_levels():
    with pytest.raises(ValueError):
        power.days_for_power(0.3, power=0.77)


def test_fisher_interval_covers_the_truth_about_95_percent_of_the_time():
    # The analytic interval is only worth using if it is calibrated. 1500
    # replicates give a standard error near 0.6 percentage points.
    rng = np.random.default_rng(11)
    for n in (14, 60):
        for target in (0.0, 0.4):
            hits, reps = 0, 1500
            for _ in range(reps):
                x, y = bivariate(n, target, rng)
                lo, hi = power.fisher_ci(spearman(x, y), n)
                hits += lo <= target <= hi
            assert 0.92 < hits / reps < 0.98, (n, target, hits / reps)


def test_fisher_interval_agrees_with_the_bootstrap_this_package_ships():
    # power.py is analytic; validate.py is a bootstrap. They must agree, or the
    # power table is advice about a different test than the one you will run.
    rng = np.random.default_rng(3)
    for n in (30, 60):
        x, y = bivariate(n, 0.4, rng)
        rho, (blo, bhi) = bootstrap_rho(x, y, n_boot=3000, seed=5)
        flo, fhi = power.fisher_ci(rho, n)
        assert abs(blo - flo) < 0.06
        assert abs(bhi - fhi) < 0.06


def test_assess_calls_an_interval_that_excludes_zero_conclusive():
    a = power.assess(0.45, 60)
    assert a.conclusive and a.lo > 0 and a.days_to_resolve == 0


def test_assess_calls_a_tight_null_conclusive():
    # Lots of days, tiny rho: the interval excludes the threshold, so a real
    # negative result has been established.
    a = power.assess(0.02, 400)
    assert a.conclusive
    assert a.hi < power.REAL_THRESHOLD
    assert "real negative result" in a.note


def test_assess_calls_a_wide_interval_inconclusive_and_says_how_many_more_days():
    a = power.assess(0.32, 14)
    assert not a.conclusive
    assert a.lo < 0 < a.hi
    assert a.hi > power.REAL_THRESHOLD
    assert a.days_to_resolve > 0
    assert "settles nothing" in a.note


def test_assess_handles_too_few_days():
    a = power.assess(0.3, 3)
    assert not a.conclusive and math.isnan(a.lo)


# --- the decision box, which power.py changed -------------------------------

def test_decision_box_keeps_the_plans_verdicts_where_they_apply():
    assert classify(0.5, (0.2, 0.7), 30)[0] == "real"
    assert classify(0.2, (0.11, 0.33), 30)[0] == "sample-starved"
    assert classify(-0.5, (-0.7, -0.3), 30)[0] == "rebuild"
    assert "NEGATIVE" in classify(-0.5, (-0.7, -0.3), 30)[1]


def test_a_wide_interval_spanning_zero_is_underpowered_not_rebuild():
    # The plan reads this as "rebuild", which routes to "do not write Swift".
    # It is consistent with rho = 0.5. Nothing has been ruled out.
    verdict, note, more = classify(0.30, (-0.20, 0.68), 14)
    assert verdict == "underpowered"
    assert more > 0
    assert "not yet evidence" in note


def test_a_tight_interval_spanning_zero_is_a_genuine_rebuild():
    # Many days, and the interval excludes 0.30: an effect of the size you set
    # out to find has been ruled out. That is a real negative result.
    verdict, note, more = classify(0.04, (-0.09, 0.17), 250)
    assert verdict == "rebuild"
    assert more == 0
    assert "ruled out an effect" in note


def test_unestimable_rho_is_underpowered_not_rebuild():
    verdict, _, more = classify(float("nan"), (float("nan"), float("nan")), 0)
    assert verdict == "underpowered" and more == -1


def test_the_two_spanning_zero_cases_are_distinguished_by_the_threshold_only():
    # Same point estimate, different widths -> different verdicts. This is the
    # distinction the plan's decision box does not make.
    narrow = classify(0.12, (-0.05, 0.28), 300)[0]
    wide = classify(0.12, (-0.30, 0.52), 20)[0]
    assert narrow == "rebuild"
    assert wide == "underpowered"
