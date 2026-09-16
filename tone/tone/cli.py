"""Command line for the reference implementation.

Typical path through the plan:

    tone parse   ~/Downloads/export.zip --out data/windows.jsonl   # Phase 1.1
    tone report  data/windows.jsonl                                # Phase 1.1
    tone score   data/windows.jsonl                                # Phase 1.2
    tone plot    data/windows.jsonl --out figures/                 # Phase 1.2
    tone weights data/windows.jsonl --out data/config.json         # Phase 2.1
    tone validate data/windows.jsonl --diary data/diary.csv \
                  --config data/config.json                        # Phase 2.2
    tone fixtures data/windows.jsonl --config data/config.json \
                  --out watch/fixtures.json                        # Phase 3.1

`tone demo` runs all of it on a synthetic body, which is how you check the
pipeline works before your own export exists.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

from . import calibrate as calib
from . import cosinor, diary as diary_mod, fixtures as fixtures_mod, power, store
from .config import DEFAULT, ScoreConfig
from .metrics import window_metrics
from .score import daily_scores, score_windows


def _load_config(path: str | None) -> ScoreConfig:
    return DEFAULT if not path else ScoreConfig.load(path)


def _rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


# --------------------------------------------------------------------------- parse
def cmd_parse(args) -> int:
    from . import parse_export

    since = datetime.fromisoformat(args.since) if args.since else None
    raw = parse_export.parse(args.export, since=since)
    windows = [r.to_window() for r in raw]
    store.save(windows, args.out)

    on_demand = sum(w.on_demand for w in windows)
    print(f"parsed {len(windows)} HRV windows -> {args.out}")
    if windows:
        print(f"range: {windows[0].start.date()} .. {windows[-1].start.date()}")
        print(f"mindfulness-tagged (on-demand): {on_demand}")
        empty = sum(1 for w in windows if np.size(w.rr) == 0)
        if empty:
            print(f"NOTE: {empty} windows carried no beat-to-beat list and cannot yield RMSSD.")
        native = sum(1 for r in raw if r.apple_rmssd is not None)
        if native:
            print(f"{native} windows also carry Apple's own RMSSD (watchOS 27+ hardware):")
            print("a free cross-check against ours, and a sign the quantisation caveat")
            print("in docs/FINDINGS.md no longer limits you.")
        agreement = parse_export.sdnn_agreement(windows)
        if agreement.get("n", 0) < 20:
            print(f"only {agreement.get('n', 0)} windows carry Apple's SDNN; too few to "
                  "check the reconstruction against.")
        else:
            print(
                f"reconstruction check on {agreement['n']} windows: "
                f"our SDNN {agreement['mean_ours']:.1f} ms vs Apple's {agreement['mean_apple']:.1f} ms, "
                f"bias {agreement['bias']:+.2f} ms, MAE {agreement['mae']:.2f} ms, r = {agreement['corr']:.3f}"
            )
            print("(large disagreement here means the parse is wrong -- fix it before scoring.)")
    return 0


# -------------------------------------------------------------------------- report
def cmd_report(args) -> int:
    cfg = _load_config(args.config)
    windows = store.load(args.windows)
    if not windows:
        print("no windows in file", file=sys.stderr)
        return 1
    scores = score_windows(windows, cfg)
    usable = [s for s in scores if s.metrics.usable]

    _rule("coverage")
    print(f"windows in file:   {len(scores)}")
    print(f"usable:            {len(usable)} ({100 * len(usable) / len(scores):.1f}%)")
    print(f"range:             {scores[0].day} .. {scores[-1].day}")
    rejects = Counter(s.flags[0] for s in scores if not s.metrics.usable and s.flags)
    for reason, n in rejects.most_common():
        print(f"  rejected {reason:24s} {n}")

    if usable:
        by_day = Counter(s.day for s in usable)
        span = (scores[-1].day - scores[0].day).days + 1
        counts = np.array(list(by_day.values()) + [0] * (span - len(by_day)))
        _rule("usable windows per day")
        print(f"days covered:      {len(by_day)} of {span}")
        print(f"mean / median:     {counts.mean():.2f} / {np.median(counts):.1f}")
        print(f"min / max:         {counts.min()} / {counts.max()}")
        print(f"on-demand windows: {sum(s.on_demand for s in usable)}")

        _rule("clock times of usable windows")
        hist = Counter(int(s.hour) for s in usable)
        peak = max(hist.values())
        for hour in range(24):
            n = hist.get(hour, 0)
            bar = "#" * int(round(40 * n / peak)) if peak else ""
            print(f"{hour:02d}  {n:5d}  {bar}")

        hours = np.array([s.hour for s in usable])
        _rule("circadian structure (the Section 7 falsifier)")
        for label, values in (("ln RMSSD", [s.x for s in usable]), ("ln HR", [s.h for s in usable])):
            fit = cosinor.fit(hours, np.array(values), min_distinct_hours=cfg.min_baseline_hours)
            print(
                f"{label:9s} R2 = {fit.r2:.3f}   amplitude = {fit.amplitude:.3f} "
                f"({100 * (math.exp(fit.amplitude) - 1):+.1f}% peak-to-MESOR)   "
                f"peak at {fit.acrophase_hours:04.1f}h"
            )
        print("R2 below ~0.10 on ln RMSSD means the circadian correction is not")
        print("earning its place for you, and a flat baseline is the honest choice.")
    return 0


# --------------------------------------------------------------------------- score
def cmd_score(args) -> int:
    cfg = _load_config(args.config)
    windows = store.load(args.windows)
    scores = score_windows(windows, cfg)
    daily = daily_scores(scores, cfg)

    if args.out:
        _write_scores_csv(scores, args.out)
        print(f"wrote per-window scores -> {args.out}")

    if not cfg.weights_measured:
        print("WARNING: no measured weights in config; using equal weights (Phase 2.1 not done).")

    _rule(f"daily score, last {args.tail} days")
    print(f"{'day':<12}{'S':>8}{'95% interval':>22}{'n':>5}")
    for d in daily[-args.tail:]:
        interval = f"[{d.lo:+.2f}, {d.hi:+.2f}]" if math.isfinite(d.lo) else "[  n/a  ]"
        flag = "*" if d.sd_pooled else " "
        print(f"{d.day.isoformat():<12}{d.mean:+8.2f}{interval:>22}{d.n:>5}{flag}")
    if any(d.sd_pooled for d in daily[-args.tail:]):
        print("* single-window day: interval uses the pooled within-day SD.")

    scored = [s for s in scores if s.scored]
    _rule("summary")
    print(f"scored windows:    {len(scored)} of {len(scores)}")
    print(f"days with a score: {len(daily)}")
    if scored:
        s_values = np.array([s.s for s in scored])
        print(f"S mean / sd:       {s_values.mean():+.3f} / {s_values.std(ddof=1):.3f}")
        worst = sorted(daily, key=lambda d: -d.mean)[:3]
        print("highest-stress days: " + ", ".join(f"{d.day} ({d.mean:+.2f})" for d in worst))
        flags = Counter(f for s in scores for f in s.flags)
        for flag, n in flags.most_common():
            print(f"  flag {flag:24s} {n}")
    return 0


def _write_scores_csv(scores, path: str) -> None:
    import csv

    rows = [s.to_dict() for s in scores]
    fields = list(rows[0].keys())
    with Path(path).open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            row["flags"] = "|".join(row["flags"])
            writer.writerow(row)


# ------------------------------------------------------------------------- weights
def cmd_weights(args) -> int:
    cfg = _load_config(args.config)
    windows = store.load(args.windows)
    scores = score_windows(windows, cfg)
    pairs = calib.find_pairs(
        scores,
        max_gap_minutes=args.max_gap,
        on_demand_only=not args.any_window,
    )
    if len(pairs) < 2:
        print(
            f"found {len(pairs)} test-retest pairs. Phase 2.1 asks for 10: two Mindfulness\n"
            "sessions five minutes apart, seated, on ten separate days. Pass --any-window\n"
            "to pair up passive windows too (weaker: posture and state are not controlled).",
            file=sys.stderr,
        )
        return 1

    result = calib.calibrate(pairs, seed=args.seed)
    _rule("Phase 2.1 test-retest calibration")
    for a, b in pairs:
        gap = (b.start - a.start).total_seconds() / 60.0
        print(
            f"  {a.start:%Y-%m-%d %H:%M}  RMSSD {a.metrics.rmssd:6.1f} -> {b.metrics.rmssd:6.1f} ms"
            f"   HR {a.metrics.mean_hr:5.1f} -> {b.metrics.mean_hr:5.1f} bpm   ({gap:.0f} min apart)"
        )
    print()
    print(result.report(cfg))

    if args.out:
        cfg.with_weights(result.sigma2_x, result.sigma2_h).save(args.out)
        print(f"\nwrote calibrated config -> {args.out}")
        print("Pass it to every later command with --config, and paste the two")
        print("sigma^2 values into watch/CLAUDE.md before the Swift engine is written.")
    return 0


# ------------------------------------------------------------------------ validate
def cmd_validate(args) -> int:
    cfg = _load_config(args.config)
    windows = store.load(args.windows)
    entries = diary_mod.read_diary(args.diary)
    if not entries:
        print(f"no diary entries parsed from {args.diary}", file=sys.stderr)
        return 1

    scores = score_windows(windows, cfg)
    daily = daily_scores(scores, cfg)

    _rule("Phase 2.2 falsification test")
    if not cfg.weights_measured:
        print("NOTE: weights are unmeasured (equal). Do Phase 2.1 first; this is a rehearsal.\n")

    ratings = np.array([e.rating for e in entries], dtype=float)
    print(f"diary entries: {len(entries)} over "
          f"{len({e.day for e in entries})} days, "
          f"mean {ratings.mean():.2f}, sd {ratings.std(ddof=1) if ratings.size > 1 else float('nan'):.2f}")
    if ratings.size > 1 and ratings.std(ddof=1) < 0.5:
        print("WARNING: your ratings barely vary. If the test comes back 'rebuild',")
        print("the diary is the first suspect, not the sensor.")
    print()

    daily_result = diary_mod.validate_daily(daily, entries, n_boot=args.boot, seed=args.seed)
    print(daily_result.report("daily mean score vs daily mean rating"))
    print()
    spot = diary_mod.validate_on_demand(
        scores, entries, window_minutes=args.spot_window, n_boot=args.boot, seed=args.seed
    )
    print(spot.report(f"on-demand spot checks vs nearest rating (+/-{args.spot_window:g} min)"))

    print()
    if daily_result.verdict == "real":
        print("=> Proceed to Phase 3. Freeze the spec with the weights you measured.")
    elif daily_result.verdict == "sample-starved":
        print("=> Two scheduled Mindfulness sessions a day, then re-run. This is the")
        print("   only finding that would justify new hardware.")
    elif daily_result.verdict == "underpowered":
        print("=> NOT a rebuild, and not a pass. You do not yet have the days to tell")
        print("   the difference. Keep the diary running and re-run this.")
        if daily_result.days_to_resolve > 0:
            print(f"   At the rho you are seeing, about {daily_result.days_to_resolve} more "
                  "overlapping days should settle it.")
        print(f"   At n = {daily_result.n}, nothing below rho = "
              f"{power.min_detectable_rho(daily_result.n):.2f} can clear zero. "
              "`tone power` has the full table.")
    else:
        print("=> Do not write Swift yet. Work the decision box in Section 2.2 of the plan.")
        if spot.verdict in ("real", "sample-starved"):
            print("   Note the spot checks did better than the daily mean: the moment-to-moment")
            print("   instrument may be real even if the daily aggregate is not. That is the")
            print("   'different app' outcome in Section 7, and it is still worth building.")
    return {"real": 0, "sample-starved": 0, "rebuild": 2, "underpowered": 3}[daily_result.verdict]


# --------------------------------------------------------------------------- doctor
def cmd_doctor(args) -> int:
    """Where is this project, and what is the next thing to do?"""
    from . import doctor as doc

    cfg = _load_config(args.config)
    windows = store.load(args.windows)
    entries = doc.parse_diary(args.diary)
    checks, action = doc.analyse(windows, entries, cfg)

    _rule("status")
    print(doc.report(checks))

    _rule("next")
    print(action)

    if any(c.status == doc.FAIL for c in checks):
        return 1
    if any(c.status == doc.TODO for c in checks):
        return 3
    return 0


# ---------------------------------------------------------------------- sensitivity
def cmd_sensitivity(args) -> int:
    """Does the Phase 2.2 verdict survive the constants you picked by judgement?"""
    from . import sensitivity as sens

    cfg = _load_config(args.config)
    windows = store.load(args.windows)
    entries = diary_mod.read_diary(args.diary)
    if not entries:
        print(f"no diary entries parsed from {args.diary}", file=sys.stderr)
        return 1

    quantized = any(w.quantized for w in windows)
    sweeps = [
        ("baseline_days", "baseline_days", (14.0, 21.0, 28.0, 42.0, 56.0)),
        ("artifact_threshold", "artifact_threshold", (0.10, 0.15, 0.20, 0.25, 0.30)),
        ("min_intervals", "min_intervals", (15, 20, 30, 40, 50)),
        ("interval", "interval", ("t", "normal")),
    ]
    if quantized:
        sweeps.append(("quantization_correction", "quantization_correction", (False, True)))

    _rule("lambda (Step 5's physiological prior)")
    print(sens.report(sens.lambda_curve(windows, entries, cfg), knob="lambda"))

    _rule("tuning lambda, honestly")
    tuning = sens.tune_lambda(windows, entries, cfg)
    print(tuning.report())
    print()
    print("Phase 5 says to re-fit lambda against the diary. Doing that and then quoting")
    print("the resulting rho is circular -- you picked the parameter that maximises the")
    print("number you are about to report. On simulated data with NO true relationship,")
    print("a 21-point grid over ~50 days inflated rho by 0.03 on average and 0.11 at")
    print("worst. Modest, because lambda is one global scalar and leaving a day out")
    print("barely changes which value wins -- but not nothing when the threshold is 0.30,")
    print("and the cross-validated number costs nothing to compute.")

    unstable = []
    for title, param, values in sweeps:
        _rule(title)
        points = sens.sweep(windows, entries, cfg, param=param, values=values)
        print(sens.report(points, knob=param))
        if len({p.verdict for p in points}) > 1:
            unstable.append(title)

    _rule("verdict")
    if unstable:
        print("The Phase 2.2 verdict DEPENDS on: " + ", ".join(unstable))
        print("A conclusion that moves with a constant you chose by judgement is not yet a")
        print("conclusion. More days is the usual fix; see `tone power`.")
        return 2
    print("The verdict held across every knob swept. That is worth a sentence in")
    print("whatever you eventually write about this: the result is not an artifact of")
    print("the 28 days, the 20%, or the 30 beats.")
    return 0


# ------------------------------------------------------------------------------ ecg
def cmd_ecg(args) -> int:
    """Turn HKElectrocardiogram CSVs into scoreable windows.

    Recordings closer together than --pair-within are treated as one sitting and
    their intervals concatenated (never bridging the gap between them). The
    default of 2 minutes is deliberately far below the 5 minutes a Phase 2.1
    test-retest pair is separated by: merging those would destroy the very
    pairing the calibration depends on.
    """
    from . import ecg as ecg_mod

    target = Path(args.path)
    files = [target] if target.is_file() else ecg_mod.find_ecg_files(target)
    if not files:
        print(f"no ECG CSVs found under {target}.\n"
              "Health writes them to apple_health_export/electrocardiograms/ inside the\n"
              "export zip -- they are NOT in export.xml, which is why `tone parse` misses them.",
              file=sys.stderr)
        return 1

    cfg = _load_config(args.config)
    loaded, undated = [], []
    sample_rate = ecg_mod.DEFAULT_FS
    for path in sorted(files):
        rec = ecg_mod.read_ecg_csv(path)
        sample_rate = rec.fs or sample_rate
        when = ecg_mod.recording_datetime(rec)
        (undated if when is None else loaded).append((path, rec, when))
    loaded.sort(key=lambda item: item[2])

    grouped = ecg_mod.group_sittings([(when, rec) for _, rec, when in loaded],
                                     args.pair_within)
    sittings = [[(None, rec, when) for when, rec in group] for group in grouped]

    _rule(f"{len(files)} recording(s) in {len(sittings)} sitting(s)")
    print(f"{'when':<18}{'recs':>5}{'beats':>7}{'HR':>7}{'RMSSD':>8}  status")

    windows, too_short, merged = [], 0, 0
    for sitting in sittings:
        rr = ecg_mod.rr_from_recordings([rec for _, rec, _ in sitting])
        when = sitting[0][2]
        metrics = window_metrics(rr, cfg) if rr.size else None
        label = when.strftime("%Y-%m-%d %H:%M")
        if len(sitting) > 1:
            merged += 1

        if metrics is None or not metrics.usable:
            reason = metrics.reject_reason if metrics else "no beats detected"
            if reason == "too_few_intervals":
                too_short += 1
            hr = f"{metrics.mean_hr:6.1f}" if metrics and math.isfinite(metrics.mean_hr) else "     -"
            print(f"{label:<18}{len(sitting):5d}{rr.size + len(sitting):7d}{hr:>7}"
                  f"{'-':>8}  dropped: {reason}")
            continue

        print(f"{label:<18}{len(sitting):5d}{rr.size + len(sitting):7d}{metrics.mean_hr:7.1f}"
              f"{metrics.rmssd:8.1f}  ok")
        windows.append(ecg_mod.ecg_window([rec for _, rec, _ in sitting], when))

    _rule("notes")
    print(f"Timing resolution at {sample_rate:.0f} Hz is {ecg_mod.timing_resolution_ms(sample_rate):.2f} ms "
          "per sample before interpolation,")
    print("against ~17 ms of bpm quantisation on the export path at 60 bpm. This is the")
    print("most precise RMSSD your hardware can give you.")
    if merged:
        print(f"\n{merged} sitting(s) combined more than one recording "
              f"(within {args.pair_within:g} min of each other).")
        print("Intervals are concatenated; the gap between recordings is never counted")
        print("as a beat-to-beat interval.")
    if too_short:
        print()
        print(f"{too_short} sitting(s) were dropped for having too few intervals.")
        print(f"A 30 s ECG needs a heart rate of {ecg_mod.minimum_heart_rate():.0f} bpm to reach "
              f"{cfg.min_intervals} intervals -- below that, one recording is not enough.")
        print("Take two back to back and re-run; they will be combined into one window.")
        print()
        print("Note for Phase 2.1: if your resting heart rate is under "
              f"{ecg_mod.minimum_heart_rate():.0f} bpm, a test-retest")
        print("PAIR needs FOUR recordings -- two back-to-back for each half of the pair,")
        print("five minutes apart. Two single ECGs five minutes apart give you two")
        print("dropped windows and no calibration.")
    if undated:
        print(f"\n{len(undated)} recording(s) had no parseable Recording Date and were skipped:")
        print("an ECG that cannot be placed on the clock cannot be scored against a baseline.")

    if args.out and windows:
        store.save(windows, args.out)
        print(f"\nwrote {len(windows)} ECG window(s) -> {args.out}")
        print("Merge them with your passive windows before scoring; they are marked")
        print("on_demand, so `tone weights` will pair them and `tone validate` will use")
        print("them as spot checks.")
    return 0


# ---------------------------------------------------------------------------- power
def cmd_power(args) -> int:
    """How many days before Phase 2.2 can answer anything?"""
    _rule("what a given number of days can detect")
    print(power.table())
    print()
    print("Read the middle column as: with this many overlapping diary-and-score days,")
    print("any rho smaller than this has a 95% interval that still contains zero.")

    _rule("how many days a given true rho needs")
    print(power.requirements())
    print()
    print("'median' is when the interval around the rho you observe clears zero if your")
    print("observation lands on the truth. '80% power' is the number to actually plan")
    print("around, because half the time it will not.")

    _rule("what this means for the plan as written")
    print(f"Phase 1.3 finishes at 14 days. At 14 days the smallest detectable rho is "
          f"{power.min_detectable_rho(14):.2f}.")
    print(f"Phase 2.2's threshold for 'the signal is real' is {power.REAL_THRESHOLD:.2f}, and the plan")
    print("says ambulatory HRV-stress correlations in the literature are modest.")
    print()
    print(f"So a perfectly real rho of {power.REAL_THRESHOLD:.2f}, measured at 14 days, yields an interval")
    print("spanning zero -- which the plan's decision box reads as 'rebuild', i.e. do not")
    print("write Swift. That is a false negative built into the schedule.")
    print()
    print(f"Run the falsification test at ~{power.days_required(power.REAL_THRESHOLD)} overlapping days, not 14. Keep the 14-day")
    print("milestone as the habit checkpoint it is. `tone validate` now reports")
    print("UNDERPOWERED rather than REBUILD when the interval cannot tell the two apart.")

    if args.rho is not None and args.days is not None:
        _rule(f"your case: rho = {args.rho:g} at {args.days} days")
        a = power.assess(args.rho, args.days)
        print(f"95% interval (Fisher-z): {a.lo:+.3f} to {a.hi:+.3f}")
        print(f"conclusive: {'yes' if a.conclusive else 'no'}")
        print(a.note)
    return 0


# ------------------------------------------------------------------------ fixtures
def cmd_fixtures(args) -> int:
    cfg = _load_config(args.config)
    if args.edge:
        from . import edge_cases
        windows = edge_cases.all_windows()
        limit = None  # check every scored window; the file is small and each
                      # scenario matters
    elif args.windows:
        windows = store.load(args.windows)
        limit = args.limit
    else:
        print("give a windows file, or --edge for the degenerate-branch fixture",
              file=sys.stderr)
        return 1
    doc = fixtures_mod.build(windows, cfg, limit=limit)
    if args.edge:
        from . import edge_cases
        from .score import daily_scores as _daily, score_windows as _score
        scores = _score(windows, cfg)
        counts = edge_cases.coverage(scores, _daily(scores, cfg))
        missing = [b for b in edge_cases.REQUIRED_BRANCHES if not counts[b]]
        print(edge_cases.report(scores, _daily(scores, cfg)))
        if missing:
            print(f"ERROR: these branches were not reached: {', '.join(missing)}",
                  file=sys.stderr)
            return 1
        doc["notes"] = (
            "DEGENERATE-BRANCH FIXTURE. Deterministic synthetic windows built by "
            "tone.edge_cases to reach the paths ordinary data never takes: the "
            "flat-baseline guard, the MAD-is-zero sigma fallback, the exact 28-day "
            "baseline boundary, a single-window day borrowing the pooled SD, and "
            "windows that must be dropped. A port must pass this file as well as "
            "the one built from real data. " + doc["notes"]
        )
    checked = sum(1 for w in doc["windows"] if w["checked"])
    if checked == 0:
        print(
            "no window could be scored, so the fixture would assert nothing. You need at\n"
            f"least {cfg.min_scoring_windows} usable windows before the first scoreable one.",
            file=sys.stderr,
        )
        return 1
    fixtures_mod.save(doc, args.out)
    print(f"wrote {len(doc['windows'])} windows ({checked} checked) -> {args.out}")
    print(f"tolerance {doc['tolerance']:g}; the Swift ScoreEngine must match every checked field.")
    ok, failures = fixtures_mod.verify(args.out)
    print("self-check:", "pass" if ok else f"FAIL ({len(failures)})")
    return 0 if ok else 1


def cmd_verify(args) -> int:
    ok, failures = fixtures_mod.verify(args.fixtures)
    if ok:
        print(f"{args.fixtures}: all expectations reproduced")
        return 0
    print(f"{args.fixtures}: {len(failures)} mismatches", file=sys.stderr)
    for line in failures[:20]:
        print("  " + line, file=sys.stderr)
    if len(failures) > 20:
        print(f"  ... and {len(failures) - 20} more", file=sys.stderr)
    return 1


# ---------------------------------------------------------------------- simulate
def cmd_simulate(args) -> int:
    from .simulate import simulate

    windows, entries, truth = simulate(days=args.days, seed=args.seed, quantized=args.quantized)
    store.save(windows, args.out)
    with Path(args.diary).open("w") as fh:
        fh.write("timestamp,rating\n")
        for e in entries:
            fh.write(f"{e.when.isoformat()},{e.rating:g}\n")
    Path(args.truth).write_text(json.dumps(truth.to_dict(), indent=2) + "\n")
    print(f"simulated {len(windows)} windows over {args.days} days -> {args.out}")
    print(f"diary ({len(entries)} entries) -> {args.diary}")
    print(f"generating parameters -> {args.truth}")
    return 0


# ----------------------------------------------------------------------------- plot
def cmd_plot(args) -> int:
    from . import plots

    cfg = _load_config(args.config)
    windows = store.load(args.windows)
    scores = score_windows(windows, cfg)
    daily = daily_scores(scores, cfg)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    written = [
        plots.clock_histogram(scores, out / "clock_histogram.png"),
        plots.baseline_comparison(scores, out / "baseline_comparison.png", cfg),
    ]
    entries = diary_mod.read_diary(args.diary) if args.diary else None
    if daily:
        written.append(plots.score_timeline(scores, daily, out / "score_timeline.png", entries))
    for path in written:
        print(f"wrote {path}")
    return 0


# ----------------------------------------------------------------------------- demo
def cmd_demo(args) -> int:
    from .simulate import simulate

    print("Running the whole Phase 1-2 flow on a synthetic body with known parameters.")
    print("Nothing here is your data; this only shows the pipeline is sound.\n")
    windows, entries, truth = simulate(days=args.days, seed=args.seed, quantized=args.quantized)
    cfg = DEFAULT

    scores = score_windows(windows, cfg)
    usable = [s for s in scores if s.metrics.usable]
    hours = np.array([s.hour for s in usable])

    _rule("1.2  does the baseline recover the rhythm it was given?")
    fit_x = cosinor.fit(hours, np.array([s.x for s in usable]), min_distinct_hours=cfg.min_baseline_hours)
    print(f"ln RMSSD amplitude  injected {truth.amp_x:.3f}   recovered {fit_x.amplitude:.3f}")
    print(f"ln RMSSD peak hour  injected {truth.acrophase_x:.1f}      recovered {fit_x.acrophase_hours:.1f}")
    print(f"cosinor R2          {fit_x.r2:.3f}")
    print("The phase comes back exactly; the amplitude comes back too large, on purpose.")
    print("The simulated stress itself has a time-of-day shape (higher in the working")
    print("afternoon, lower asleep), so part of what the cosinor absorbs is stress that")
    print("happens to be circadian. Your body will do the same thing, and it is the one")
    print("real cost of Step 3: the baseline can eat a little of the signal it is meant")
    print("to reveal. It is still the right trade -- see the last section of this demo.")

    _rule("2.1  does test-retest recover the measurement noise it was given?")
    pairs = calib.find_pairs(scores)
    result = calib.calibrate(pairs, seed=args.seed)
    print(f"sigma^2 ln RMSSD    injected {truth.sigma_eps_x ** 2:.4f}   estimated {result.sigma2_x:.4f}")
    print(f"sigma^2 ln HR       injected {truth.sigma_eps_h ** 2:.4f}   estimated {result.sigma2_h:.4f}")
    print("(Two things to read here. In expectation test-retest lands ~10% ABOVE the")
    print(" injected sensor noise, because a 60-second window also carries sampling")
    print(" noise and the design cannot separate the two -- which is fine, since the")
    print(" score is computed from exactly such windows and that is the noise it")
    print(" should be weighted by. With only 10 pairs, though, the estimate itself is")
    print(" imprecise: read the bootstrap interval below, not the point estimate.)")
    print()
    print(result.report(cfg))

    cfg = cfg.with_weights(result.sigma2_x, result.sigma2_h)
    scores = score_windows(windows, cfg)
    daily = daily_scores(scores, cfg)

    _rule("2.2  does the score track the diary?")
    print("Expect an implausibly high rho here. The simulated diary is generated from")
    print("the same latent stress that moves the simulated HRV, so this measures whether")
    print("the ESTIMATOR works, not how well HRV tracks real mood. Ambulatory HRV-stress")
    print("correlations in the literature are modest; rho >= 0.3 on your own data is the")
    print("bar, not this number.\n")
    print(diary_mod.validate_daily(daily, entries, n_boot=args.boot, seed=args.seed).report(
        "daily mean score vs daily mean rating"))
    print()
    print(diary_mod.validate_on_demand(scores, entries, n_boot=args.boot, seed=args.seed).report(
        "on-demand spot checks vs nearest rating"))

    _rule("what the cosinor step buys")
    resid_flat = np.array([s.x for s in usable]) - np.mean([s.x for s in usable])
    resid_cos = np.array([s.x for s in usable]) - fit_x.predict(hours)
    print(f"sd of ln RMSSD residual, flat baseline:    {resid_flat.std(ddof=1):.3f}")
    print(f"sd of ln RMSSD residual, cosinor baseline: {resid_cos.std(ddof=1):.3f}")
    print(f"variance removed by knowing the clock:      {100 * fit_x.r2:.1f}%")
    return 0


# ----------------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tone", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("parse", help="Apple Health export.xml/.zip -> windows.jsonl")
    sp.add_argument("export")
    sp.add_argument("--out", default="data/windows.jsonl")
    sp.add_argument("--since", help="ISO date; ignore windows before it")
    sp.set_defaults(func=cmd_parse)

    sp = sub.add_parser("report", help="coverage, clock times, circadian structure")
    sp.add_argument("windows")
    sp.add_argument("--config")
    sp.set_defaults(func=cmd_report)

    sp = sub.add_parser("score", help="compute S per window and per day")
    sp.add_argument("windows")
    sp.add_argument("--config")
    sp.add_argument("--out", help="write per-window scores to CSV")
    sp.add_argument("--tail", type=int, default=21, help="days to print (default 21)")
    sp.set_defaults(func=cmd_score)

    sp = sub.add_parser("weights", help="Phase 2.1 test-retest calibration")
    sp.add_argument("windows")
    sp.add_argument("--config")
    sp.add_argument("--out", help="write a calibrated config JSON")
    sp.add_argument("--max-gap", type=float, default=15.0, help="minutes between a pair")
    sp.add_argument("--any-window", action="store_true",
                    help="pair passive windows too, not just Mindfulness sessions")
    sp.add_argument("--seed", type=int, default=20260916)
    sp.set_defaults(func=cmd_weights)

    sp = sub.add_parser("validate", help="Phase 2.2 falsification test against the diary")
    sp.add_argument("windows")
    sp.add_argument("--diary", required=True)
    sp.add_argument("--config")
    sp.add_argument("--boot", type=int, default=10000)
    sp.add_argument("--spot-window", type=float, default=30.0)
    sp.add_argument("--seed", type=int, default=20260916)
    sp.set_defaults(func=cmd_validate)

    sp = sub.add_parser("doctor", help="what state is the project in, and what is next?")
    sp.add_argument("windows")
    sp.add_argument("--diary")
    sp.add_argument("--config")
    sp.set_defaults(func=cmd_doctor)

    sp = sub.add_parser("sensitivity",
                        help="does the verdict survive the constants you guessed?")
    sp.add_argument("windows")
    sp.add_argument("--diary", required=True)
    sp.add_argument("--config")
    sp.set_defaults(func=cmd_sensitivity)

    sp = sub.add_parser("ecg", help="HKElectrocardiogram CSVs -> scoreable windows")
    sp.add_argument("path", help="an unzipped export directory, or one ecg_*.csv")
    sp.add_argument("--config")
    sp.add_argument("--out", help="write the resulting windows as JSONL")
    sp.add_argument("--pair-within", type=float, default=2.0, metavar="MIN",
                    help="recordings this close are one sitting (default 2 min; keep it "
                         "well below the 5 min of a test-retest pair)")
    sp.set_defaults(func=cmd_ecg)

    sp = sub.add_parser("power", help="how many diary days Phase 2.2 needs")
    sp.add_argument("--rho", type=float, help="assess a specific observed rho")
    sp.add_argument("--days", type=int, help="...at this many overlapping days")
    sp.set_defaults(func=cmd_power)

    sp = sub.add_parser("fixtures", help="write parity fixtures for the Swift port")
    sp.add_argument("windows", nargs="?")
    sp.add_argument("--config")
    sp.add_argument("--out", default="watch/fixtures.json")
    sp.add_argument("--limit", type=int, default=50, help="checked windows (default 50)")
    sp.add_argument("--edge", action="store_true",
                    help="emit the degenerate-branch fixture instead of using a data file")
    sp.set_defaults(func=cmd_fixtures)

    sp = sub.add_parser("verify", help="re-run a fixture file and report mismatches")
    sp.add_argument("fixtures")
    sp.set_defaults(func=cmd_verify)

    sp = sub.add_parser("simulate", help="write a synthetic dataset with known parameters")
    sp.add_argument("--days", type=int, default=120)
    sp.add_argument("--seed", type=int, default=20260916)
    sp.add_argument("--quantized", action="store_true", help="mimic whole-bpm export data")
    sp.add_argument("--out", default="data/sim_windows.jsonl")
    sp.add_argument("--diary", default="data/sim_diary.csv")
    sp.add_argument("--truth", default="data/sim_truth.json")
    sp.set_defaults(func=cmd_simulate)

    sp = sub.add_parser("plot", help="figures for Phase 1.1 / 1.2")
    sp.add_argument("windows")
    sp.add_argument("--config")
    sp.add_argument("--diary")
    sp.add_argument("--out", default="figures")
    sp.set_defaults(func=cmd_plot)

    sp = sub.add_parser("demo", help="run Phases 1-2 end to end on synthetic data")
    sp.add_argument("--days", type=int, default=120)
    sp.add_argument("--seed", type=int, default=20260916)
    sp.add_argument("--boot", type=int, default=4000)
    sp.add_argument("--quantized", action="store_true")
    sp.set_defaults(func=cmd_demo)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = getattr(args, "out", None)
    if isinstance(out, str) and Path(out).parent != Path(""):
        Path(out).parent.mkdir(parents=True, exist_ok=True)
    for extra in ("diary", "truth"):
        value = getattr(args, extra, None)
        if args.command == "simulate" and isinstance(value, str):
            Path(value).parent.mkdir(parents=True, exist_ok=True)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
