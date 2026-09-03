"""Tests for src/score_hybrid.py (step-013: hybrid scoring + the
6-detector comparison table).

Small synthetic streams keep the suite fast; the full 58-stream corpus
run happens offline via `python src/score_hybrid.py`.
"""
import csv
import json
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hybrid import HybridDetector
from src.run_all_detectors import CSV_COLUMNS, PROFILES
from src.score_hybrid import (
    HYBRID_DETECTORS,
    build_comparison,
    load_prior_rows,
    load_rows,
    score_hybrid,
    write_comparison,
)


def _make_timestamps(n, start="2014-01-01 00:00:00", freq_minutes=5):
    t0 = datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
    return [t0 + timedelta(minutes=i * freq_minutes) for i in range(n)]


def _synthetic_streams():
    """One gently varying stream with a late spike burst (past both
    sub-detectors' warmup), plus one pure-flat stream."""
    n = 320
    ts = _make_timestamps(n)
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


def _fake_prior_rows():
    """Five prior detectors x three profiles (statistical + iforest)."""
    rows = []
    for det in ("gaussian", "ewma", "zscore", "threshold", "iforest"):
        for p in PROFILES:
            rows.append({"detector": det, "profile": p, "nab_score": 10.0,
                         "raw_score": -1.0, "threshold": 1.0,
                         "num_streams": 58, "total_windows": 116})
    return rows


def _fake_hybrid_rows():
    return [{"detector": "hybrid", "profile": p, "nab_score": 42.0,
             "raw_score": -1.0, "threshold": 0.6,
             "num_streams": 58, "total_windows": 116} for p in PROFILES]


class TestRegistry:
    def test_hybrid_registered(self):
        assert set(HYBRID_DETECTORS) == {"hybrid"}
        assert HYBRID_DETECTORS["hybrid"] is HybridDetector


class TestScoreHybrid:
    def _rows(self):
        if not hasattr(self, "_cached"):
            self._cached = score_hybrid(
                streams=_synthetic_streams(),
                windows=_synthetic_windows(),
                verbose=False)
        return self._cached

    def test_one_row_per_profile(self):
        rows = self._rows()
        assert len(rows) == len(PROFILES)
        assert {r["profile"] for r in rows} == set(PROFILES)

    def test_all_rows_are_hybrid_with_full_schema(self):
        for row in self._rows():
            assert row["detector"] == "hybrid"
            assert set(row) == set(CSV_COLUMNS)

    def test_scores_bounded(self):
        for row in self._rows():
            assert 0.0 <= row["nab_score"] <= 100.0
            assert row["num_streams"] == 2

    def test_deterministic(self):
        """Seeded sub-detectors -> identical scores on a re-run."""
        again = score_hybrid(streams=_synthetic_streams(),
                             windows=_synthetic_windows(), verbose=False)
        assert [r["nab_score"] for r in again] == \
               [r["nab_score"] for r in self._rows()]

    def test_hybrid_flags_obvious_spike(self):
        """The spike sits above both warmups, so the standard profile
        beats the null detector's score of 0."""
        standard = next(r for r in self._rows()
                        if r["profile"] == "standard")
        assert standard["nab_score"] > 0.0


class TestBuildComparison:
    def test_merges_to_six_detectors(self):
        rows = build_comparison(_fake_hybrid_rows(), _fake_prior_rows())
        detectors = {r["detector"] for r in rows}
        assert len(detectors) == 6
        assert "hybrid" in detectors
        assert len(rows) == 6 * len(PROFILES)

    def test_no_duplicate_detector_profile_pairs(self):
        rows = build_comparison(_fake_hybrid_rows(), _fake_prior_rows())
        pairs = [(r["detector"], r["profile"]) for r in rows]
        assert len(pairs) == len(set(pairs))

    def test_idempotent_on_rerun(self):
        """Feeding an already-merged table back in must not grow it."""
        once = build_comparison(_fake_hybrid_rows(), _fake_prior_rows())
        twice = build_comparison(_fake_hybrid_rows(), once)
        assert len(twice) == len(once)


class TestWriteComparison:
    def test_writes_csv_and_json(self, tmp_path):
        rows = build_comparison(_fake_hybrid_rows(), _fake_prior_rows())
        csv_path = str(tmp_path / "comparison.csv")
        json_path = str(tmp_path / "comparison.json")
        write_comparison(rows, csv_path=csv_path, json_path=json_path)
        with open(csv_path, newline="") as f:
            read_back = list(csv.DictReader(f))
        assert len(read_back) == 6 * len(PROFILES)
        assert set(read_back[0]) == set(CSV_COLUMNS)
        with open(json_path) as f:
            data = json.load(f)
        assert "hybrid" in data["detectors"]
        assert len(data["detectors"]) == 6


class TestLoadPriorRows:
    def test_load_merges_two_files(self, tmp_path):
        stat = tmp_path / "statistical_detectors.json"
        ifo = tmp_path / "iforest_scores.json"
        stat.write_text(json.dumps({"rows": _fake_prior_rows()[:12]}))
        ifo.write_text(json.dumps({"rows": _fake_prior_rows()[12:]}))
        rows = load_prior_rows(str(tmp_path))
        assert len(rows) == 5 * len(PROFILES)
        assert {r["detector"] for r in rows} == {
            "gaussian", "ewma", "zscore", "threshold", "iforest"}

    def test_load_rows_reads_rows_key(self, tmp_path):
        p = tmp_path / "x.json"
        p.write_text(json.dumps({"rows": _fake_hybrid_rows()}))
        assert len(load_rows(str(p))) == len(PROFILES)
