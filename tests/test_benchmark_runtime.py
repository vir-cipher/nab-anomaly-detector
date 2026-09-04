"""Tests for the step-014 runtime benchmark (src/benchmark_runtime.py).

Tiny synthetic streams keep the suite fast and deterministic; the real
corpus numbers come from running the module directly (see
results/runtime_benchmark.*).
"""

import json
import math
import random
import time
from datetime import datetime, timedelta

from src.detectors import Detector, NullDetector, ThresholdDetector
from src.iforest import IsolationForestDetector
from src import benchmark_runtime as bench


def _stream(n=120, seed=1):
    base = datetime(2020, 1, 1)
    ts = [base + timedelta(minutes=5 * i) for i in range(n)]
    rng = random.Random(seed)
    vals = [rng.gauss(0.0, 1.0) for _ in range(n)]
    return ts, vals


def _streams(n=120):
    return {"synthetic/a.csv": _stream(n, 1),
            "synthetic/b.csv": _stream(n, 2)}


class _SlowDetector(Detector):
    """Deliberately slow detector: sleeps a little on every point."""

    def handle_record(self, timestamp, value):
        time.sleep(0.001)
        return 0.0


def test_time_detector_on_stream_positive():
    ts, vals = _stream(100)
    secs = bench.time_detector_on_stream(NullDetector, ts, vals)
    assert isinstance(secs, float)
    assert secs > 0.0


def test_repeats_still_returns_positive_time():
    ts, vals = _stream(100)
    secs = bench.time_detector_on_stream(NullDetector, ts, vals, repeats=3)
    assert secs > 0.0


def test_benchmark_detector_structure_and_identities():
    streams = _streams(120)
    d = bench.benchmark_detector(ThresholdDetector, streams)
    assert d["n_streams"] == 2
    assert d["total_points"] == 240
    assert set(d["per_stream"]) == set(streams)
    # throughput identity: pts/s * seconds == points
    assert math.isclose(d["points_per_sec"] * d["total_seconds"],
                        d["total_points"], rel_tol=1e-6)
    # us/pt is the reciprocal of throughput, scaled to micro-seconds
    assert math.isclose(d["us_per_point"],
                        1e6 / d["points_per_sec"], rel_tol=1e-6)


def test_benchmark_all_rows_and_reference_speedup():
    streams = _streams(120)
    registry = {"threshold": ThresholdDetector,
                "iforest": IsolationForestDetector}
    rows, detail = bench.benchmark_all(streams=streams, detectors=registry,
                                       repeats=1, verbose=False)
    assert {r["detector"] for r in rows} == {"threshold", "iforest"}
    for r in rows:
        assert set(r) == set(bench.CSV_COLUMNS)
    # rows are sorted fastest-first
    pps = [r["points_per_sec"] for r in rows]
    assert pps == sorted(pps, reverse=True)
    # the reference detector's own speed-up is exactly 1.0
    ref = [r for r in rows if r["detector"] == "iforest"][0]
    assert math.isclose(ref["speedup_vs_iforest"], 1.0, rel_tol=1e-9)


def test_speedup_reflects_real_workload():
    streams = _streams(40)
    registry = {"fast": NullDetector, "slow": _SlowDetector,
                "iforest": IsolationForestDetector}
    rows, detail = bench.benchmark_all(streams=streams, detectors=registry,
                                       repeats=1, verbose=False)
    # the sleeping detector must have lower throughput than the null one
    assert (detail["fast"]["points_per_sec"]
            > detail["slow"]["points_per_sec"])


def test_write_speed_table_creates_files(tmp_path):
    streams = _streams(60)
    registry = {"threshold": ThresholdDetector,
                "iforest": IsolationForestDetector}
    rows, detail = bench.benchmark_all(streams=streams, detectors=registry,
                                       verbose=False)
    csv_path = tmp_path / "rt.csv"
    json_path = tmp_path / "rt.json"
    meta = bench.environment_meta(1)
    bench.write_speed_table(rows, detail=detail, csv_path=str(csv_path),
                            json_path=str(json_path), meta=meta)
    assert csv_path.exists() and json_path.exists()
    header = csv_path.read_text().splitlines()[0]
    assert header == ",".join(bench.CSV_COLUMNS)
    payload = json.loads(json_path.read_text())
    assert payload["meta"]["reference_detector"] == "iforest"
    assert len(payload["rows"]) == 2
    assert "per_stream" in payload


def test_format_speed_table_is_string():
    streams = _streams(60)
    registry = {"threshold": ThresholdDetector,
                "iforest": IsolationForestDetector}
    rows, _ = bench.benchmark_all(streams=streams, detectors=registry,
                                  verbose=False)
    text = bench.format_speed_table(rows)
    assert "detector" in text
    assert "threshold" in text
