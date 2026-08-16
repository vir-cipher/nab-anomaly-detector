"""Tests for anomaly detectors and end-to-end scoring pipeline.

Step-003 gate: null detector scores ~0 on NAB.
Step-004 gate: reproduce published Windowed Gaussian baseline
               within +-5% on >=1 profile.
"""

import os
import sys
from datetime import datetime

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detectors import NullDetector, WindowedGaussianDetector, Detector
from src.data_loader import load_stream, load_all_streams
from src.scoring import load_windows, score_corpus, sweep_optimize


# -------------------------------------------------------------------
# Null detector unit tests (step-003)
# -------------------------------------------------------------------

class TestNullDetectorUnit:
    def test_implements_detector(self):
        assert issubclass(NullDetector, Detector)

    def test_returns_zero(self):
        det = NullDetector()
        assert det.handle_record(datetime(2024, 1, 1), 42.0) == 0.0

    def test_returns_zero_many_points(self):
        det = NullDetector()
        from datetime import timedelta
        base = datetime(2024, 1, 1)
        for i in range(100):
            assert det.handle_record(base + timedelta(minutes=i), float(i)) == 0.0

    def test_reset_does_not_error(self):
        det = NullDetector()
        det.handle_record(datetime(2024, 1, 1), 1.0)
        det.reset()

    def test_name(self):
        assert NullDetector().name == "NullDetector"

    def test_score_in_range(self):
        score = NullDetector().handle_record(datetime(2024, 1, 1), -999.0)
        assert 0.0 <= score <= 1.0


# -------------------------------------------------------------------
# Windowed Gaussian unit tests (step-004)
# -------------------------------------------------------------------

class TestWindowedGaussianUnit:
    """Unit tests for WindowedGaussianDetector."""

    def test_implements_detector(self):
        assert issubclass(WindowedGaussianDetector, Detector)

    def test_name(self):
        assert WindowedGaussianDetector().name == "WindowedGaussianDetector"

    def test_first_point_returns_zero(self):
        """No window data yet -> score must be 0.0."""
        det = WindowedGaussianDetector()
        score = det.handle_record(datetime(2024, 1, 1), 100.0)
        assert score == 0.0

    def test_score_in_range(self):
        """All scores must be in [0.0, 1.0]."""
        det = WindowedGaussianDetector(window_size=50, step_size=10)
        from datetime import timedelta
        base = datetime(2024, 1, 1)
        for i in range(200):
            val = 10.0 + (100.0 if i == 150 else 0.0)
            score = det.handle_record(base + timedelta(minutes=i), val)
            assert 0.0 <= score <= 1.0, f"score {score} at i={i}"

    def test_spike_scores_higher(self):
        """A large spike should score higher than a normal point."""
        det = WindowedGaussianDetector(window_size=50, step_size=10)
        from datetime import timedelta
        base = datetime(2024, 1, 1)
        # Feed 60 normal points
        for i in range(60):
            det.handle_record(base + timedelta(minutes=i), 10.0)
        normal = det.handle_record(base + timedelta(minutes=60), 10.0)
        det2 = WindowedGaussianDetector(window_size=50, step_size=10)
        for i in range(60):
            det2.handle_record(base + timedelta(minutes=i), 10.0)
        spike = det2.handle_record(base + timedelta(minutes=60), 1000.0)
        assert spike > normal

    def test_reset_clears_state(self):
        det = WindowedGaussianDetector(window_size=10, step_size=5)
        from datetime import timedelta
        base = datetime(2024, 1, 1)
        for i in range(20):
            det.handle_record(base + timedelta(minutes=i), float(i))
        det.reset()
        assert len(det.window_data) == 0
        assert det.handle_record(base, 999.0) == 0.0

    def test_window_size_respected(self):
        det = WindowedGaussianDetector(window_size=10, step_size=5)
        from datetime import timedelta
        base = datetime(2024, 1, 1)
        for i in range(20):
            det.handle_record(base + timedelta(minutes=i), float(i))
        assert len(det.window_data) == 10

    def test_default_params(self):
        det = WindowedGaussianDetector()
        assert det.window_size == 6400
        assert det.step_size == 100


# -------------------------------------------------------------------
# Data loader tests (step-003)
# -------------------------------------------------------------------

class TestDataLoader:
    def test_load_all_returns_58_streams(self):
        assert len(load_all_streams()) == 58

    def test_stream_keys_match_nab_convention(self):
        for key in load_all_streams():
            parts = key.split("/")
            assert len(parts) == 2 and parts[1].endswith(".csv")

    def test_stream_has_timestamps_and_values(self):
        ts, vals = load_all_streams()["realKnownCause/nyc_taxi.csv"]
        assert len(ts) > 0 and len(ts) == len(vals)
        assert isinstance(ts[0], datetime) and isinstance(vals[0], float)

    def test_load_single_stream(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(root, "data", "nab", "realKnownCause", "nyc_taxi.csv")
        ts, vals = load_stream(path)
        assert len(ts) > 1000


# -------------------------------------------------------------------
# End-to-end: null detector on NAB (step-003)
# -------------------------------------------------------------------

class TestNullDetectorOnNAB:
    @pytest.fixture(scope="class")
    def corpus_results(self):
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
        r = score_corpus(corpus_results, windows, 0.5, profile="standard")
        assert r["nab_score"] == pytest.approx(0.0, abs=1e-9)

    def test_reward_low_fp_score_zero(self, corpus_results, windows):
        r = score_corpus(corpus_results, windows, 0.5, profile="reward_low_fp")
        assert r["nab_score"] == pytest.approx(0.0, abs=1e-9)

    def test_reward_low_fn_score_zero(self, corpus_results, windows):
        r = score_corpus(corpus_results, windows, 0.5, profile="reward_low_fn")
        assert r["nab_score"] == pytest.approx(0.0, abs=1e-9)

    def test_all_58_streams_scored(self, corpus_results, windows):
        r = score_corpus(corpus_results, windows, 0.5)
        assert r["num_streams"] == 58

    def test_total_windows_positive(self, corpus_results, windows):
        r = score_corpus(corpus_results, windows, 0.5)
        assert r["total_windows"] > 0

    def test_no_detections_made(self, corpus_results, windows):
        r = score_corpus(corpus_results, windows, 0.5)
        assert sum(s["tp"] for s in r["per_stream"].values()) == 0
        assert sum(s["fp"] for s in r["per_stream"].values()) == 0


# -------------------------------------------------------------------
# GATE TEST: reproduce published Windowed Gaussian baseline (step-004)
# Published scores: standard=39.6, reward_low_fp=20.9, reward_low_fn=47.4
# Source: github.com/numenta/NAB README scoreboard
# Gate: within +/-5% on >= 1 profile
# -------------------------------------------------------------------

class TestWindowedGaussianOnNAB:
    """Reproduce Numenta's published Windowed Gaussian NAB scores."""

    @pytest.fixture(scope="class")
    def gaussian_results(self):
        streams = load_all_streams()
        det_results = {}
        for name, (timestamps, values) in streams.items():
            det = WindowedGaussianDetector()
            scores = [det.handle_record(ts, v)
                      for ts, v in zip(timestamps, values)]
            det_results[name] = (timestamps, scores)
        return det_results

    @pytest.fixture(scope="class")
    def windows(self):
        return load_windows()

    def test_standard_within_5pct(self, gaussian_results, windows):
        """Standard profile: published 39.6."""
        r = sweep_optimize(gaussian_results, windows, profile="standard")
        assert abs(r["best_nab_score"] - 39.6) / 39.6 <= 0.05

    def test_reward_low_fn_within_5pct(self, gaussian_results, windows):
        """Reward-low-FN profile: published 47.4."""
        r = sweep_optimize(gaussian_results, windows, profile="reward_low_fn")
        assert abs(r["best_nab_score"] - 47.4) / 47.4 <= 0.05

    def test_positive_score(self, gaussian_results, windows):
        """Gaussian must beat the null detector (score > 0)."""
        r = sweep_optimize(gaussian_results, windows, profile="standard")
        assert r["best_nab_score"] > 0.0

    def test_below_perfect(self, gaussian_results, windows):
        """Gaussian is far from perfect (score < 100)."""
        r = sweep_optimize(gaussian_results, windows, profile="standard")
        assert r["best_nab_score"] < 100.0

    def test_total_windows_116(self, gaussian_results, windows):
        """NAB corpus has exactly 116 anomaly windows."""
        r = sweep_optimize(gaussian_results, windows, profile="standard")
        assert r["total_windows"] == 116

    def test_results_file_exists(self):
        """gaussian_baseline.json must exist with valid scores."""
        import json
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(root, "results", "gaussian_baseline.json")
        assert os.path.exists(path)
        with open(path) as f:
            data = json.load(f)
        assert data["detector"] == "gaussian"
        assert data["optimized"] is True
        assert data["profiles"]["standard"]["nab_score"] > 0
