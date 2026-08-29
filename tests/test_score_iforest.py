"""Tests for src/score_iforest.py (step-010: isolation-forest scoring +
the 5-detector comparison table).

Small synthetic streams keep the suite fast; the full 58-stream corpus
run happens offline via `python src/score_iforest.py`.
"""
import csv
import json
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.iforest import IsolationForestDetector
from src.run_all_detectors import CSV_COLUMNS, PROFILES
from src.score_iforest import (
    IFOREST_DETECTORS,
    build_comparison,
    load_statistical_rows,
    score_iforest,
    write_comparison,
)


def _make_timestamps(n, start="2014-01-01 00:00:00", freq_minutes=5):
    t0 = datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
    return [t0 + timedelta(minutes=i * freq_minutes) for i in range(n)]


def _synthetic_streams():
    """One flat stream with a late spike burst (after the forest warmup),
    plus one pure-flat stream."""
    n = 320
    ts = _make_timestamps(n)
    # A gently varying baseline (not perfectly flat) so the forest can form
    # splits -- a real NAB stream always carries this natural spread; a
    # zero-variance sample would collapse every isolation tree to one leaf.
    spiky = [10.0 + (i % 5) * 0.5 for i in range(n)]
    for i in range(280, 292):
        spiky[i] = 100.0
    flat = [5.0] * n
    return {
        "synthetic/spiky.csv": (ts, spiky),
        "synthetic/flat.csv": (ts, flat),
    }


def _synthetic_windows():
    ts = _make_timestamps(320)
    return {
        "synthetic/spiky.csv": [(ts[280], ts[299])],
        "synthetic/flat.csv": [],
    }


def _fake_statistical_rows():
    rows = []
    for det in ("gaussian", "ewma", "zscore", "threshold"):
        for p in PROFILES:
            rows.append({"detector": det, "profile": p, "nab_score": 10.0,
                         "raw_score": -1.0, "threshold": 1.0,
                         "num_streams": 58, "total_windows": 116})
    return rows


def _fake_iforest_rows():
    return [{"detector": "iforest", "profile": p, "nab_score": 15.0,
             "raw_score": -1.0, "threshold": 0.6,
             "num_streams": 58, "total_windows": 116} for p in PROFILES]


class TestRegistry:
    def test_iforest_registered(self):
        assert set(IFOREST_DETECTORS) == {"iforest"}
        assert IFOREST_DETECTORS["iforest"] is IsolationForestDetector


class TestScoreIforest:
    def _rows(self):
        if not hasattr(self, "_cached"):
            self._cached = score_iforest(
                streams=_synthetic_streams(),
                windows=_synthetic_windows(),
                verbose=False)
        return self._cached

    def test_one_row_per_profile(self):
        rows = self._rows()
        assert len(rows) == len(PROFILES)
        assert {r["profile"] for r in rows} == set(PROFILES)

    def test_all_rows_are_iforest_with_full_schema(self):
        for row in self._rows():
            assert row["detector"] == "iforest"
            assert set(row) == set(CSV_COLUMNS)

    def test_scores_bounded(self):
        for row in self._rows():
            assert 0.0 <= row["nab_score"] <= 100.0
            assert row["num_streams"] == 2

    def test_deterministic(self):
        """Seeded forest -> identical scores on a re-run."""
        again = score_iforest(streams=_synthetic_streams(),
                              windows=_synthetic_windows(), verbose=False)
        assert [r["nab_score"] for r in again] == \
               [r["nab_score"] for r in self._rows()]

    def test_forest_flags_obvious_spike(self):
        """The spike sits above the forest warmup, so the standard profile
        beats the null detector's score of 0."""
        standard = next(r for r in self._rows()
                        if r["profile"] == "standard")
        assert standard["nab_score"] > 0.0


class TestBuildComparison:
    def test_merges_to_five_detectors(self):
        rows = build_comparison(_fake_iforest_rows(), _fake_statistical_rows())
        detectors = {r["detector"] for r in rows}
        assert len(detectors) == 5
        assert "iforest" in detectors
        assert len(rows) == 5 * len(PROFILES)

    def test_no_duplicate_detector_profile_pairs(self):
        rows = build_comparison(_fake_iforest_rows(), _fake_statistical_rows())
        pairs = [(r["detector"], r["profile"]) for r in rows]
        assert len(pairs) == len(set(pairs))

    def test_idempotent_on_rerun(self):
        """Feeding an already-merged table back in must not grow it."""
        once = build_comparison(_fake_iforest_rows(), _fake_statistical_rows())
        twice = build_comparison(_fake_iforest_rows(), once)
        assert len(twice) == len(once)


class TestWriteComparison:
    def test_writes_csv_and_json(self, tmp_path):
        rows = build_comparison(_fake_iforest_rows(), _fake_statistical_rows())
        csv_path = str(tmp_path / "comparison.csv")
        json_path = str(tmp_path / "comparison.json")
        write_comparison(rows, csv_path=csv_path, json_path=json_path)
        with open(csv_path, newline="") as f:
            read_back = list(csv.DictReader(f))
        assert len(read_back) == 5 * len(PROFILES)
        assert set(read_back[0]) == set(CSV_COLUMNS)
        with open(json_path) as f:
            data = json.load(f)
        assert "iforest" in data["detectors"]
        assert len(data["detectors"]) == 5


class TestLoadStatisticalRows:
    def test_load_from_file(self, tmp_path):
        p = tmp_path / "stat.json"
        p.write_text(json.dumps({"rows": _fake_statistical_rows()}))
        rows = load_statistical_rows(str(p))
        assert len(rows) == 4 * len(PROFILES)
        assert {r["detector"] for r in rows} == {
            "gaussian", "ewma", "zscore", "threshold"}
