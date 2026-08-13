"""Tests for anomaly detectors and end-to-end scoring pipeline.

Step-003 gate: null detector scores ~0 on NAB, validating the
full pipeline (data load -> detect -> score -> normalise).
"""

import os
import sys
from datetime import datetime

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detectors import NullDetector, Detector
from src.data_loader import load_stream, load_all_streams
from src.scoring import load_windows, score_corpus


# -------------------------------------------------------------------
# Detector interface tests
# -------------------------------------------------------------------

class TestNullDetectorUnit:
    """Unit tests for the NullDetector class."""

    def test_implements_detector(self):
        assert issubclass(NullDetector, Detector)

    def test_returns_zero(self):
        det = NullDetector()
        ts = datetime(2024, 1, 1)
        assert det.handle_record(ts, 42.0) == 0.0

    def test_returns_zero_many_points(self):
        det = NullDetector()
        base = datetime(2024, 1, 1)
        from datetime import timedelta
        for i in range(100):
            ts = base + timedelta(minutes=i)
            assert det.handle_record(ts, float(i)) == 0.0

    def test_reset_does_not_error(self):
        det = NullDetector()
        det.handle_record(datetime(2024, 1, 1), 1.0)
        det.reset()  # should not raise

    def test_name(self):
        det = NullDetector()
        assert det.name == "NullDetector"

    def test_score_in_range(self):
        det = NullDetector()
        score = det.handle_record(datetime(2024, 1, 1), -999.0)
        assert 0.0 <= score <= 1.0


# -------------------------------------------------------------------
# Data loader tests
# -------------------------------------------------------------------

class TestDataLoader:
    """Tests for the NAB data loading pipeline."""

    def test_load_all_returns_58_streams(self):
        streams = load_all_streams()
        assert len(streams) == 58

    def test_stream_keys_match_nab_convention(self):
        streams = load_all_streams()
        for key in streams:
            parts = key.split("/")
            assert len(parts) == 2, f"Bad key: {key}"
            assert parts[1].endswith(".csv")

    def test_stream_has_timestamps_and_values(self):
        streams = load_all_streams()
        key = "realKnownCause/nyc_taxi.csv"
        ts, vals = streams[key]
        assert len(ts) > 0
        assert len(ts) == len(vals)
        assert isinstance(ts[0], datetime)
        assert isinstance(vals[0], float)

    def test_load_single_stream(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(
            root, "data", "nab",
            "realKnownCause", "nyc_taxi.csv")
        ts, vals = load_stream(path)
        assert len(ts) > 1000  # nyc_taxi has ~10k rows


# -------------------------------------------------------------------
# End-to-end pipeline: null detector on real NAB data
# -------------------------------------------------------------------

class TestNullDetectorOnNAB:
    """Gate test: null detector scores exactly 0.0 on NAB.

    This validates the FULL pipeline:
      data load -> detector -> scorer -> normalisation.
    By definition, a null detector (never flags) gets normalised
    score = 0.0 on every profile.
    """

    @pytest.fixture(scope="class")
    def corpus_results(self):
        """Run null detector on all 58 streams (cached per class)."""
        streams = load_all_streams()
        det_results = {}
        for name, (timestamps, values) in streams.items():
            det = NullDetector()
            scores = [det.handle_record(ts, v)
                      for ts, v in zip(timestamps, values)]
            det_results[name] = (timestamps, scores)
        return det_results

    @pytest.fixture(scope="class")
    def windows(self):
        return load_windows()

    def test_standard_score_zero(self, corpus_results, windows):
        result = score_corpus(
            corpus_results, windows, threshold=0.5,
            profile="standard")
        assert result["nab_score"] == pytest.approx(0.0, abs=1e-9)

    def test_reward_low_fp_score_zero(self, corpus_results, windows):
        result = score_corpus(
            corpus_results, windows, threshold=0.5,
            profile="reward_low_fp")
        assert result["nab_score"] == pytest.approx(0.0, abs=1e-9)

    def test_reward_low_fn_score_zero(self, corpus_results, windows):
        result = score_corpus(
            corpus_results, windows, threshold=0.5,
            profile="reward_low_fn")
        assert result["nab_score"] == pytest.approx(0.0, abs=1e-9)

    def test_all_58_streams_scored(self, corpus_results, windows):
        result = score_corpus(
            corpus_results, windows, threshold=0.5)
        assert result["num_streams"] == 58

    def test_total_windows_positive(self, corpus_results, windows):
        result = score_corpus(
            corpus_results, windows, threshold=0.5)
        assert result["total_windows"] > 0

    def test_no_detections_made(self, corpus_results, windows):
        """Null detector should produce zero TP and zero FP."""
        result = score_corpus(
            corpus_results, windows, threshold=0.5)
        total_tp = sum(
            s["tp"] for s in result["per_stream"].values())
        total_fp = sum(
            s["fp"] for s in result["per_stream"].values())
        assert total_tp == 0
        assert total_fp == 0
