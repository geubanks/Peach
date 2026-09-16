# Phase 1.3 — the diary

The diary is the only ground truth in this project. Nothing downstream can be
evaluated without it, and unlike every other phase it cannot be hurried: it
needs calendar days, not effort. **Start it the same day you export.**

Target: 14 consecutive days with at least 2 entries a day. **That milestone
proves the habit stuck; it is not enough to run the falsification test on.**

At 14 days the smallest correlation whose interval excludes zero is ρ = 0.54,
and Phase 2.2's threshold for "the signal is real" is 0.30. Running the test at
14 days on a real ρ of 0.3 gives an interval spanning zero, which the plan's
decision box reads as "rebuild" — a false negative on the one test that matters.
Resolving ρ = 0.3 needs about **46 overlapping diary-and-score days**, and ~90
to have an 80% chance of getting there. Call it seven weeks, not two.

Run `tone power` for the full table, and see §0 of `FINDINGS.md`. `tone validate`
will now say UNDERPOWERED rather than REBUILD when the interval cannot tell the
two apart, and will tell you how many more days you need.

## The automation

Shortcuts on iPhone, one shortcut plus one automation per anchored time.

**Shortcut — "Stress check"**

1. `Ask for Input` → Number → prompt `Stress right now, 0-10?`
2. `Format Date` → Current Date → Custom → `yyyy-MM-dd HH:mm:ss`
3. `Text` → `[Formatted Date],[Provided Input]` (no space after the comma)
4. `Get File` from iCloud Drive → `diary.csv` → *Error if not found:* off
5. `Text` → combine the file's contents and the new line
6. `Save File` → iCloud Drive → `diary.csv` → *Overwrite if exists:* on

If step 4 returns nothing on the first run, seed the file once by hand with the
header line `timestamp,rating` and nothing else.

**Automations** — Personal Automation → Time of Day → Run Immediately (no "Ask
Before Running"), one each at the three anchors from the plan:

- after your first class
- after lab
- before bed

Pick real times and keep them fixed. Anchored times matter more than they look:
they make the ratings comparable across days, and they let the spot-check test
in Phase 2.2 pair a Mindfulness session with a rating taken minutes away.

## Take the spot checks at a diary time

Phase 2.1 asks for two Mindfulness sessions five minutes apart on ten separate
days, and Phase 2.2 correlates each on-demand window with the nearest rating
within 30 minutes. Those are the same moment: run the paired sessions *at* one
of your anchored diary times. Sessions scattered at random hours calibrate the
weights fine but leave the on-demand correlation with nothing to correlate
against, and that is the sharper of the two tests — it asks whether the score
tracks *this moment* rather than a whole day's mood.

## What the file has to look like

```csv
timestamp,rating
2026-09-16 11:05:12,3
2026-09-16 17:20:44,7
2026-09-16 22:41:03,4
```

`tone.diary.read_diary` matches the header loosely (`timestamp` / `time` /
`date` / `when` and `rating` / `stress` / `score` / `level`), accepts a
headerless two-column file, handles ISO and slash-separated dates, and skips
rows that do not parse rather than aborting. A phone automation will eventually
write one bad line; it should not cost you the analysis.

## Rate the moment, not the day

Rate how you feel *right now*, before looking at anything else and without
trying to be consistent with yesterday. The score is a within-person instrument:
it only needs your ratings to be comparable to each other, and the fastest way
to break that is to start reasoning about what you rated last Tuesday.

## If Phase 2.2 says "rebuild", suspect this file first

The plan is explicit about it and `tone validate` will say so too: check whether
your ratings actually vary day to day. A diary with a standard deviation under
0.5 has nothing in it for the score to track, and no amount of signal processing
will recover a correlation with a constant. `tone validate` prints the mean and
SD of your ratings at the top of its output for exactly this reason.
