"""Tests for src/run_all_detectors.py (step-008 tabulation).

Uses small synthetic streams so the suite stays fast -- the full-corpus
run happens offline via `python src/run_all_detectors.py`.
"""
import csv
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detectors import EWMADetector, ThresholdDetector
from src.run_all_detectors import (
    CSV_COLUMNS,
    PROFILES,
    STATISTICAL_DETECTORS,
    format_table,
    run_detector_on_streams,
    score_all,
    write_csv,
)


def _make_timestamps(n, start="2014-01-01 00:00:00", freq_minutes=5):
    t0 = datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
    return [t0 + timedelta(minutes=i * freq_minutes) for i in range(n)]


def _synthetic_streams():
    """Two streams: flat with an obvious spike burst, and pure flat."""
    n = 400
    ts = _make_timestamps(n)
    spiky = [10.0] * n
    for i in range(300, 312):
        spiky[i] = 100.0
    flat = [5.0] * n
    return {
        "synthetic/spiky.csv": (ts, spiky),
        "synthetic/flat.csv": (ts, flat),
    }


def _synthetic_windows():
    ts = _make_timestamps(400)
    return {
        "synthetic/spiky.csv": [(ts[300], ts[319])],
        "synthetic/flat.csv": [],
    }


class TestRegistry:
    def test_all_four_statistical_detectors_registered(self):
        assert set(STATISTICAL_DETECTORS) == {
            "gaussian", "ewma", "zscore", "threshold"}

    def test_three_profiles(self):
        assert PROFILES == ("standard", "reward_low_fp", "reward_low_fn")


class TestRunDetectorOnStreams:
    def test_scores_have_same_length_as_input(self):
        streams = _synthetic_streams()
        results = run_detector_on_streams(EWMADetector, streams)
        assert set(results) == set(streams)
        for name, (ts, scores) in results.items():
            assert len(scores) == len(streams[name][0])

    def test_fresh_detector_per_stream(self):
        """Flat stream must not inherit state from the spiky stream."""
        streams = _synthetic_streams()
        results = run_detector_on_streams(ThresholdDetector, streams)
        _, flat_scores = results["synthetic/flat.csv"]
        assert all(s == 0.0 for s in flat_scores)


class TestScoreAll:
    def _rows(self):
        if not hasattr(self, "_cached"):
            self._cached = score_all(
                streams=_synthetic_streams(),
                windows=_synthetic_windows(),
                detectors={"ewma": EWMADetector,
                           "threshold": ThresholdDetector},
                verbose=False)
        return self._cached

    def test_one_row_per_detector_profile_pair(self):
        rows = self._rows()
        assert len(rows) == 2 * len(PROFILES)
        pairs = {(r["detector"], r["profile"]) for r in rows}
        assert len(pairs) == 6

    def test_rows_have_all_csv_columns(self):
        for row in self._rows():
            assert set(row) == set(CSV_COLUMNS)

    def test_scores_plausible(self):
        """NAB scores are bounded: never above 100 (perfect)."""
        for row in self._rows():
            assert row["nab_score"] <= 100.0
            assert row["num_streams"] == 2
            assert row["total_windows"] == 1

    def test_obvious_spike_is_detected(self):
        """On a blatant spike, optimized detectors beat the null score of 0."""
        rows = self._rows()
        standard = [r for r in rows if r["profile"] == "standard"]
        assert any(r["nab_score"] > 0 for r in standard)


class TestWriteCsvAndTable:
    def _rows(self):
        return [
            {"detector": "ewma", "profile": p, "nab_score": s,
             "raw_score": -1.0, "threshold": 0.5,
             "num_streams": 2, "total_windows": 1}
            for p, s in [("standard", 20.0), ("reward_low_fp", 10.0),
                         ("reward_low_fn", 30.0)]
        ] + [
            {"detector": "threshold", "profile": p, "nab_score": s,
             "raw_score": -2.0, "threshold": 1.0,
             "num_streams": 2, "total_windows": 1}
            for p, s in [("standard", 40.0), ("reward_low_fp", 5.0),
                         ("reward_low_fn", 15.0)]
        ]

    def test_csv_written_with_header_and_all_rows(self, tmp_path):
        path = str(tmp_path / "out.csv")
        write_csv(self._rows(), path)
        with open(path, newline="") as f:
            read_back = list(csv.DictReader(f))
        assert len(read_back) == 6
        assert set(read_back[0]) == set(CSV_COLUMNS)

    def test_csv_sorted_by_profile_then_score(self, tmp_path):
        path = str(tmp_path / "out.csv")
        write_csv(self._rows(), path)
        with open(path, newline="") as f:
            read_back = list(csv.DictReader(f))
        # First two rows: standard profile, best score first (threshold 40 > ewma 20)
        assert read_back[0]["profile"] == "standard"
        assert read_back[0]["detector"] == "threshold"
        assert read_back[1]["detector"] == "ewma"

    def test_format_table_ranks_standard_profile(self):
        table = format_table(self._rows())
        assert table.index("threshold") < table.index("ewma")
