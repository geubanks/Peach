import json
import math
import zipfile
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from tone import fixtures, parse_export, store
from tone.config import ScoreConfig
from tone.score import Window, score_windows
from tone.simulate import simulate

TZ = timezone(timedelta(hours=-4))

EXPORT_XML = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE HealthData [
<!ELEMENT HealthData (Record*)>
]>
<HealthData locale="en_US">
 <Record type="HKCategoryTypeIdentifierMindfulSession" sourceName="Gavin's Watch"
   startDate="2026-05-01 18:00:00 -0400" endDate="2026-05-01 18:01:00 -0400"
   value="HKCategoryValueNotApplicable"/>
 <Record type="HKQuantityTypeIdentifierHeartRateVariabilitySDNN" sourceName="Gavin's Watch"
   unit="ms" creationDate="2026-05-01 09:00:00 -0400" startDate="2026-05-01 09:00:00 -0400"
   endDate="2026-05-01 09:01:00 -0400" value="41.2">
  <HeartRateVariabilityMetadataList>
   <InstantaneousBeatsPerMinute bpm="62" time="9:00:01.00"/>
   <InstantaneousBeatsPerMinute bpm="64" time="9:00:02.00"/>
   <InstantaneousBeatsPerMinute bpm="61" time="9:00:03.00"/>
  </HeartRateVariabilityMetadataList>
 </Record>
 <Record type="HKQuantityTypeIdentifierHeartRateVariabilitySDNN" sourceName="Gavin's Watch"
   unit="ms" creationDate="2026-05-01 18:00:30 -0400" startDate="2026-05-01 18:00:30 -0400"
   endDate="2026-05-01 18:01:30 -0400" value="55.9">
  <HeartRateVariabilityMetadataList>
   <InstantaneousBeatsPerMinute bpm="58" time="18:00:31.00"/>
   <InstantaneousBeatsPerMinute bpm="57" time="18:00:32.00"/>
  </HeartRateVariabilityMetadataList>
 </Record>
 <Record type="HKQuantityTypeIdentifierStepCount" sourceName="iPhone" unit="count"
   startDate="2026-05-01 09:00:00 -0400" endDate="2026-05-01 09:01:00 -0400" value="12"/>
</HealthData>
"""


def write_export(tmp_path, name="export.xml"):
    path = tmp_path / name
    path.write_text(EXPORT_XML)
    return path


def test_parse_reads_hrv_windows_and_skips_other_types(tmp_path):
    windows = parse_export.parse(write_export(tmp_path))
    assert len(windows) == 2
    assert windows[0].apple_sdnn == pytest.approx(41.2)
    assert windows[0].bpm == [62, 64, 61]
    assert windows[0].source == "Gavin's Watch"


def test_parse_tags_mindfulness_windows_as_on_demand(tmp_path):
    windows = parse_export.parse(write_export(tmp_path))
    assert windows[0].on_demand is False
    assert windows[1].on_demand is True  # inside the Mindfulness session


def test_parse_preserves_the_local_clock_hour(tmp_path):
    window = parse_export.parse(write_export(tmp_path))[0].to_window()
    # 09:00 local must stay 9.0, not become 13.0 via UTC.
    assert window.hour == pytest.approx(9.0)
    assert window.day.isoformat() == "2026-05-01"


def test_parse_reads_a_zip(tmp_path):
    xml = write_export(tmp_path)
    archive = tmp_path / "export.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(xml, "apple_health_export/export.xml")
    assert len(parse_export.parse(archive)) == 2


def test_parse_respects_since(tmp_path):
    cutoff = datetime(2026, 5, 1, 12, tzinfo=TZ)
    assert len(parse_export.parse(write_export(tmp_path), since=cutoff)) == 1


def test_export_windows_are_marked_quantized(tmp_path):
    window = parse_export.parse(write_export(tmp_path))[0].to_window()
    assert window.quantized
    assert np.allclose(window.rr, 60000.0 / np.array([62.0, 64.0, 61.0]))


def test_sdnn_agreement_reports_the_reconstruction_gap():
    rng = np.random.default_rng(0)
    windows = []
    for i in range(30):
        rr = 1000.0 + rng.normal(0, 40, 80)
        quantized = 60000.0 / np.round(60000.0 / rr)
        windows.append(
            Window(start=datetime(2026, 5, 1, tzinfo=TZ) + timedelta(hours=i),
                   rr=quantized, quantized=True,
                   apple_sdnn=float(np.std(rr, ddof=1)))
        )
    report = parse_export.sdnn_agreement(windows)
    assert report["n"] == 30
    # SDNN is dominated by the real spread, so quantisation barely moves it --
    # which is exactly why SDNN is the right check on the reconstruction and
    # RMSSD is the metric that suffers.
    assert abs(report["bias"]) < 1.0
    assert report["corr"] > 0.9


def test_store_roundtrip(tmp_path):
    windows, _, _ = simulate(days=20, seed=2)
    path = tmp_path / "w.jsonl"
    store.save(windows, path)
    back = store.load(path)
    assert len(back) == len(windows)
    assert back[0].start == windows[0].start
    assert np.allclose(back[0].rr, windows[0].rr, atol=1e-4)


def test_fixture_roundtrip_verifies(tmp_path):
    windows, _, _ = simulate(days=70, seed=6)
    doc = fixtures.build(windows, ScoreConfig(), limit=50)
    path = tmp_path / "fixtures.json"
    fixtures.save(doc, path)
    ok, failures = fixtures.verify(path)
    assert ok, failures[:5]
    assert sum(1 for w in doc["windows"] if w["checked"]) == 50


def test_fixture_file_is_strict_json_with_no_nan(tmp_path):
    # Swift's JSONDecoder rejects bare NaN. If this ever regresses, the Swift
    # side fails to even load the fixtures.
    windows, _, _ = simulate(days=70, seed=8)
    path = tmp_path / "fixtures.json"
    fixtures.save(fixtures.build(windows, ScoreConfig()), path)
    text = path.read_text()
    assert "NaN" not in text and "Infinity" not in text
    json.loads(text)


def test_fixture_verification_catches_a_tampered_expectation(tmp_path):
    windows, _, _ = simulate(days=70, seed=4)
    path = tmp_path / "fixtures.json"
    fixtures.save(fixtures.build(windows, ScoreConfig(), limit=10), path)
    doc = json.loads(path.read_text())
    target = next(w for w in doc["windows"] if w["checked"])
    target["expected"]["s"] += 0.01  # a 1% error the Swift port might plausibly make
    path.write_text(json.dumps(doc))
    ok, failures = fixtures.verify(path)
    assert not ok and any("s:" in f for f in failures)


def test_fixture_config_travels_with_the_numbers(tmp_path):
    windows, _, _ = simulate(days=70, seed=10)
    cfg = ScoreConfig(sigma2_eps_x=0.05, sigma2_eps_h=0.002, lambda_hrv=1.0)
    path = tmp_path / "fixtures.json"
    fixtures.save(fixtures.build(windows, cfg, limit=5), path)
    doc = json.loads(path.read_text())
    assert doc["config"]["sigma2_eps_x"] == pytest.approx(0.05)
    # And the expectations must be the ones those weights produce.
    scores = [s for s in score_windows(windows, cfg) if s.scored]
    checked = [w for w in doc["windows"] if w["checked"]]
    # 1e-5, not exact: the fixture rounds RR to 6 decimals before scoring, so
    # these are the same algorithm on inputs that differ in the 7th place.
    assert checked[-1]["expected"]["s"] == pytest.approx(scores[-1].s, abs=1e-5)


def test_config_roundtrip(tmp_path):
    cfg = ScoreConfig(sigma2_eps_x=0.03, sigma2_eps_h=0.001, interval="normal")
    path = tmp_path / "cfg.json"
    cfg.save(path)
    assert ScoreConfig.load(path) == cfg


def test_config_rejects_nonsense():
    with pytest.raises(ValueError):
        ScoreConfig(artifact_threshold=1.5)
    with pytest.raises(ValueError):
        ScoreConfig(sigma2_eps_x=0.0)
    with pytest.raises(ValueError):
        ScoreConfig(interval="bootstrap")


def test_quantized_simulation_inflates_rmssd_as_predicted():
    clean, _, _ = simulate(days=30, seed=3, quantized=False)
    rough, _, _ = simulate(days=30, seed=3, quantized=True)
    cfg = ScoreConfig()
    a = np.array([s.metrics.rmssd for s in score_windows(clean, cfg) if s.metrics.usable])
    b = np.array([s.metrics.rmssd for s in score_windows(rough, cfg) if s.metrics.usable])
    assert b.mean() > a.mean()
    corrected = ScoreConfig(quantization_correction=True)
    c = np.array([s.metrics.rmssd for s in score_windows(rough, corrected) if s.metrics.usable])
    assert abs(c.mean() - a.mean()) < abs(b.mean() - a.mean())
    assert math.isfinite(c.mean())


def test_shipped_example_fixture_still_verifies():
    # The example file in watch/ is what the Swift side develops against before
    # the real one exists. If the algorithm changes, this fails first, and the
    # fix is to regenerate deliberately -- never to loosen the tolerance.
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "watch" / "fixtures.example.json"
    if not path.exists():
        pytest.skip("example fixture not present")
    ok, failures = fixtures.verify(path)
    assert ok, failures[:5]


RMSSD_EXPORT = EXPORT_XML.replace(
    '</HealthData>',
    """ <Record type="HKQuantityTypeIdentifierHeartRateVariabilityRMSSD" sourceName="Gavin's Watch"
   unit="ms" creationDate="2026-05-01 09:00:00 -0400" startDate="2026-05-01 09:00:00 -0400"
   endDate="2026-05-01 09:01:00 -0400" value="38.4"/>
</HealthData>""",
)


def test_native_rmssd_records_annotate_rather_than_duplicate(tmp_path):
    # On watchOS 27 hardware Apple emits an RMSSD record alongside the SDNN one
    # for the same window. Treating it as a second window would score the same
    # minute twice.
    path = tmp_path / "export.xml"
    path.write_text(RMSSD_EXPORT)
    windows = parse_export.parse(path)
    assert len(windows) == 2
    assert windows[0].apple_rmssd == pytest.approx(38.4)
    assert windows[1].apple_rmssd is None
