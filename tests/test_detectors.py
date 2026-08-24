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

from src.detectors import NullDetector, WindowedGaussianDetector, EWMADetector, ZScoreDetector, ThresholdDetector, Detector
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


# -------------------------------------------------------------------
# EWMA detector unit tests (step-005)
# Gate: EWMA detector passes unit tests, produces anomaly scores
#       on sample data.
# -------------------------------------------------------------------

class TestEWMADetectorUnit:
    """Unit tests for EWMADetector."""

    def test_implements_detector(self):
        assert issubclass(EWMADetector, Detector)

    def test_name(self):
        assert EWMADetector().name == "EWMADetector"

    def test_default_params(self):
        det = EWMADetector()
        assert det.alpha == 0.1
        assert det.warmup == 10

    def test_custom_params(self):
        det = EWMADetector(alpha=0.3, warmup=20)
        assert det.alpha == 0.3
        assert det.warmup == 20

    def test_returns_zero_during_warmup(self):
        """All points during warmup must return 0.0."""
        det = EWMADetector(warmup=10)
        from datetime import timedelta
        base = datetime(2024, 1, 1)
        for i in range(10):
            score = det.handle_record(
                base + timedelta(minutes=i), float(i + 1))
            assert score == 0.0, f"warmup point {i} scored {score}"

    def test_scores_after_warmup(self):
        """Points after warmup must produce non-trivial scores."""
        det = EWMADetector(warmup=5)
        from datetime import timedelta
        base = datetime(2024, 1, 1)
        # Feed 5 warmup points (all 10.0)
        for i in range(5):
            det.handle_record(base + timedelta(minutes=i), 10.0)
        # Point after warmup â€” same value, should be low score
        score = det.handle_record(base + timedelta(minutes=5), 10.0)
        assert 0.0 <= score <= 1.0

    def test_score_in_range(self):
        """All scores must be in [0.0, 1.0]."""
        det = EWMADetector(warmup=5)
        from datetime import timedelta
        base = datetime(2024, 1, 1)
        for i in range(200):
            val = 10.0 + (500.0 if i == 150 else 0.0)
            score = det.handle_record(base + timedelta(minutes=i), val)
            assert 0.0 <= score <= 1.0, f"score {score} at i={i}"

    def test_spike_scores_higher(self):
        """A large spike after stable data must score higher."""
        from datetime import timedelta
        base = datetime(2024, 1, 1)
        # Detector 1: all normal
        det1 = EWMADetector(warmup=5)
        for i in range(60):
            det1.handle_record(base + timedelta(minutes=i), 10.0)
        normal = det1.handle_record(base + timedelta(minutes=60), 10.0)
        # Detector 2: spike at the end
        det2 = EWMADetector(warmup=5)
        for i in range(60):
            det2.handle_record(base + timedelta(minutes=i), 10.0)
        spike = det2.handle_record(base + timedelta(minutes=60), 1000.0)
        assert spike > normal

    def test_reset_clears_state(self):
        """After reset, detector behaves like a fresh instance."""
        det = EWMADetector(warmup=5)
        from datetime import timedelta
        base = datetime(2024, 1, 1)
        for i in range(20):
            det.handle_record(base + timedelta(minutes=i), float(i))
        det.reset()
        assert det._ewma is None
        assert det._count == 0
        # First point after reset must return 0.0 (warmup)
        assert det.handle_record(base, 999.0) == 0.0

    def test_adapts_to_level_shift(self):
        """EWMA should adapt: after a level shift, scores decrease."""
        det = EWMADetector(alpha=0.3, warmup=5)
        from datetime import timedelta
        base = datetime(2024, 1, 1)
        # 20 points at level 10
        for i in range(20):
            det.handle_record(base + timedelta(minutes=i), 10.0)
        # Jump to level 50 â€” first point should score high
        first_at_50 = det.handle_record(
            base + timedelta(minutes=20), 50.0)
        # After 30 more points at 50, detector adapts
        for i in range(30):
            det.handle_record(
                base + timedelta(minutes=21 + i), 50.0)
        adapted_at_50 = det.handle_record(
            base + timedelta(minutes=51), 50.0)
        assert first_at_50 > adapted_at_50

    def test_o1_memory(self):
        """EWMA uses constant memory (no growing buffers)."""
        det = EWMADetector(warmup=5)
        from datetime import timedelta
        base = datetime(2024, 1, 1)
        for i in range(10000):
            det.handle_record(base + timedelta(minutes=i), float(i))
        assert len(det._warmup_values) == 0


# -------------------------------------------------------------------
# EWMA on sample NAB data (step-005 gate: produces scores on sample)
# -------------------------------------------------------------------

class TestEWMAOnSampleData:
    """Run EWMA on a real NAB stream and verify it produces scores."""

    def test_scores_on_nyc_taxi(self):
        """EWMA produces valid scores on realKnownCause/nyc_taxi."""
        streams = load_all_streams()
        ts, vals = streams["realKnownCause/nyc_taxi.csv"]
        det = EWMADetector()
        scores = [det.handle_record(t, v) for t, v in zip(ts, vals)]
        # All scores in range
        assert all(0.0 <= s <= 1.0 for s in scores)
        # Not all zeros (detector should flag something)
        assert any(s > 0.0 for s in scores)
        # Warmup points are zero
        assert all(s == 0.0 for s in scores[:det.warmup])

    def test_scores_on_multiple_streams(self):
        """EWMA produces valid scores on at least 5 different streams."""
        streams = load_all_streams()
        keys = list(streams.keys())[:5]
        for key in keys:
            ts, vals = streams[key]
            det = EWMADetector()
            scores = [det.handle_record(t, v)
                      for t, v in zip(ts, vals)]
            assert all(0.0 <= s <= 1.0 for s in scores), (
                f"score out of range on {key}")
            assert len(scores) == len(ts)


# -------------------------------------------------------------------
# Z-score detector unit tests (step-006)
# Gate: Z-score detector passes unit tests, produces anomaly scores
#       on sample data.
# -------------------------------------------------------------------

class TestZScoreDetectorUnit:
    """Unit tests for ZScoreDetector."""

    def test_implements_detector(self):
        assert issubclass(ZScoreDetector, Detector)

    def test_name(self):
        assert ZScoreDetector().name == "ZScoreDetector"

    def test_default_params(self):
        det = ZScoreDetector()
        assert det.window_size == 128
        assert det.warmup == 30

    def test_custom_params(self):
        det = ZScoreDetector(window_size=64, warmup=10)
        assert det.window_size == 64
        assert det.warmup == 10

    def test_warmup_minimum_two(self):
        """Warmup is clamped to >= 2 (need 2 points for std)."""
        det = ZScoreDetector(warmup=1)
        assert det.warmup == 2

    def test_returns_zero_during_warmup(self):
        """All points during warmup must return 0.0."""
        det = ZScoreDetector(warmup=30)
        from datetime import timedelta
        base = datetime(2024, 1, 1)
        for i in range(30):
            score = det.handle_record(
                base + timedelta(minutes=i), float(i + 1))
            assert score == 0.0, f"warmup point {i} scored {score}"

    def test_scores_after_warmup(self):
        """Points after warmup must produce scores in [0, 1]."""
        det = ZScoreDetector(warmup=10)
        from datetime import timedelta
        base = datetime(2024, 1, 1)
        for i in range(10):
            det.handle_record(base + timedelta(minutes=i), 10.0)
        score = det.handle_record(base + timedelta(minutes=10), 10.0)
        assert 0.0 <= score <= 1.0

    def test_score_in_range(self):
        """All scores must be in [0.0, 1.0]."""
        det = ZScoreDetector(window_size=50, warmup=10)
        from datetime import timedelta
        base = datetime(2024, 1, 1)
        for i in range(200):
            val = 10.0 + (500.0 if i == 150 else 0.0)
            score = det.handle_record(base + timedelta(minutes=i), val)
            assert 0.0 <= score <= 1.0, f"score {score} at i={i}"

    def test_spike_scores_higher(self):
        """A large spike after stable data must score higher."""
        from datetime import timedelta
        base = datetime(2024, 1, 1)
        det1 = ZScoreDetector(warmup=10)
        for i in range(60):
            det1.handle_record(base + timedelta(minutes=i), 10.0)
        normal = det1.handle_record(base + timedelta(minutes=60), 10.0)
        det2 = ZScoreDetector(warmup=10)
        for i in range(60):
            det2.handle_record(base + timedelta(minutes=i), 10.0)
        spike = det2.handle_record(base + timedelta(minutes=60), 1000.0)
        assert spike > normal

    def test_reset_clears_state(self):
        """After reset, detector behaves like a fresh instance."""
        det = ZScoreDetector(warmup=5)
        from datetime import timedelta
        base = datetime(2024, 1, 1)
        for i in range(20):
            det.handle_record(base + timedelta(minutes=i), float(i))
        det.reset()
        assert len(det._window) == 0
        assert det.handle_record(base, 999.0) == 0.0

    def test_window_bounded(self):
        """Window never exceeds window_size."""
        det = ZScoreDetector(window_size=32, warmup=5)
        from datetime import timedelta
        base = datetime(2024, 1, 1)
        for i in range(200):
            det.handle_record(base + timedelta(minutes=i), float(i))
        assert len(det._window) == 32

    def test_forgets_old_data(self):
        """After sliding past old data, detector adapts to new level."""
        det = ZScoreDetector(window_size=50, warmup=10)
        from datetime import timedelta
        base = datetime(2024, 1, 1)
        # 50 points at level 10
        for i in range(50):
            det.handle_record(base + timedelta(minutes=i), 10.0)
        # Jump to level 100 -- first point scores high
        first_at_100 = det.handle_record(
            base + timedelta(minutes=50), 100.0)
        # Feed 60 more at 100 so old data fully slides out
        for i in range(60):
            det.handle_record(
                base + timedelta(minutes=51 + i), 100.0)
        adapted_at_100 = det.handle_record(
            base + timedelta(minutes=111), 100.0)
        assert first_at_100 > adapted_at_100

    def test_different_from_ewma_on_same_input(self):
        """Z-score and EWMA produce different score sequences."""
        from datetime import timedelta
        import random
        base = datetime(2024, 1, 1)
        random.seed(42)
        data = [random.gauss(50, 10) for _ in range(100)]
        z_det = ZScoreDetector(warmup=10)
        e_det = EWMADetector(warmup=10)
        z_scores = [z_det.handle_record(
            base + timedelta(minutes=i), v) for i, v in enumerate(data)]
        e_scores = [e_det.handle_record(
            base + timedelta(minutes=i), v) for i, v in enumerate(data)]
        # After warmup, at least some scores should differ
        z_post = z_scores[15:]
        e_post = e_scores[15:]
        diffs = [abs(z - e) for z, e in zip(z_post, e_post)]
        assert max(diffs) > 0.01, "Z-score and EWMA should differ"


# -------------------------------------------------------------------
# Z-score on sample NAB data (step-006 gate: produces scores on sample)
# -------------------------------------------------------------------

class TestZScoreOnSampleData:
    """Run Z-score on a real NAB stream and verify it produces scores."""

    def test_scores_on_nyc_taxi(self):
        """Z-score produces valid scores on realKnownCause/nyc_taxi."""
        streams = load_all_streams()
        ts, vals = streams["realKnownCause/nyc_taxi.csv"]
        det = ZScoreDetector()
        scores = [det.handle_record(t, v) for t, v in zip(ts, vals)]
        assert all(0.0 <= s <= 1.0 for s in scores)
        assert any(s > 0.0 for s in scores)
        assert all(s == 0.0 for s in scores[:det.warmup])

    def test_scores_on_multiple_streams(self):
        """Z-score produces valid scores on at least 5 different streams."""
        streams = load_all_streams()
        keys = list(streams.keys())[:5]
        for key in keys:
            ts, vals = streams[key]
            det = ZScoreDetector()
            scores = [det.handle_record(t, v)
                      for t, v in zip(ts, vals)]
            assert all(0.0 <= s <= 1.0 for s in scores), (
                f"score out of range on {key}")
            assert len(scores) == len(ts)


# -------------------------------------------------------------------
# Simple threshold detector unit tests (step-007)
# Gate: Threshold detector passes unit tests.
# -------------------------------------------------------------------

class TestThresholdDetectorUnit:
    """Unit tests for ThresholdDetector."""

    def test_implements_detector(self):
        assert issubclass(ThresholdDetector, Detector)

    def test_name(self):
        assert ThresholdDetector().name == "ThresholdDetector"

    def test_default_params(self):
        det = ThresholdDetector()
        assert det.warmup == 100
        assert det.n_std == 3.0

    def test_custom_params(self):
        det = ThresholdDetector(warmup=20, n_std=2.0)
        assert det.warmup == 20
        assert det.n_std == 2.0

    def test_warmup_minimum_two(self):
        """Warmup is clamped to >= 2 (need 2 points for std)."""
        det = ThresholdDetector(warmup=1)
        assert det.warmup == 2

    def test_returns_zero_during_warmup(self):
        """All points during warmup must return 0.0."""
        det = ThresholdDetector(warmup=20)
        from datetime import timedelta
        base = datetime(2024, 1, 1)
        for i in range(20):
            score = det.handle_record(
                base + timedelta(minutes=i), 10.0)
            assert score == 0.0, f"warmup point {i} scored {score}"

    def test_band_frozen_after_warmup(self):
        """low/high are set once warmup completes and never move."""
        det = ThresholdDetector(warmup=20, n_std=3.0)
        from datetime import timedelta
        base = datetime(2024, 1, 1)
        for i in range(20):
            det.handle_record(base + timedelta(minutes=i), 10.0)
        assert det.low is not None and det.high is not None
        low_after_warmup, high_after_warmup = det.low, det.high
        # Feed 100 more stable points -- band must not change.
        for i in range(100):
            det.handle_record(base + timedelta(minutes=20 + i), 10.0)
        assert det.low == low_after_warmup
        assert det.high == high_after_warmup

    def test_score_is_binary(self):
        """Scores after warmup are exactly 0.0 or 1.0, never in between."""
        det = ThresholdDetector(warmup=20, n_std=3.0)
        from datetime import timedelta
        base = datetime(2024, 1, 1)
        for i in range(20):
            det.handle_record(base + timedelta(minutes=i), 10.0)
        for i, val in enumerate([10.0, 10.5, 9.5, 500.0, -500.0]):
            score = det.handle_record(
                base + timedelta(minutes=20 + i), val)
            assert score in (0.0, 1.0), f"non-binary score {score}"

    def test_value_within_band_scores_zero(self):
        det = ThresholdDetector(warmup=20, n_std=3.0)
        from datetime import timedelta
        base = datetime(2024, 1, 1)
        for i in range(20):
            det.handle_record(base + timedelta(minutes=i), 10.0)
        score = det.handle_record(base + timedelta(minutes=20), 10.0)
        assert score == 0.0

    def test_value_outside_band_scores_one(self):
        det = ThresholdDetector(warmup=20, n_std=3.0)
        from datetime import timedelta
        base = datetime(2024, 1, 1)
        for i in range(20):
            det.handle_record(base + timedelta(minutes=i), 10.0)
        score = det.handle_record(base + timedelta(minutes=20), 10000.0)
        assert score == 1.0

    def test_reset_clears_state(self):
        """After reset, detector behaves like a fresh instance."""
        det = ThresholdDetector(warmup=5)
        from datetime import timedelta
        base = datetime(2024, 1, 1)
        for i in range(20):
            det.handle_record(base + timedelta(minutes=i), float(i))
        det.reset()
        assert det.low is None
        assert det.high is None
        assert det.handle_record(base, 999.0) == 0.0

    def test_never_adapts_unlike_zscore(self):
        """ThresholdDetector keeps flagging a sustained level shift
        that ZScoreDetector (rolling window) eventually adapts to."""
        from datetime import timedelta
        base = datetime(2024, 1, 1)
        thr = ThresholdDetector(warmup=50, n_std=3.0)
        z = ZScoreDetector(window_size=50, warmup=50)
        for i in range(50):
            thr.handle_record(base + timedelta(minutes=i), 10.0)
            z.handle_record(base + timedelta(minutes=i), 10.0)
        # Sustained jump to a new level, long enough for Z-score's
        # 50-point window to fully slide past the old level.
        thr_scores, z_scores = [], []
        for i in range(80):
            t = base + timedelta(minutes=50 + i)
            thr_scores.append(thr.handle_record(t, 100.0))
            z_scores.append(z.handle_record(t, 100.0))
        assert all(s == 1.0 for s in thr_scores), (
            "threshold detector must keep flagging the shifted level "
            "forever -- its band was frozen during warmup")
        assert z_scores[-1] < z_scores[0], (
            "z-score detector's rolling window should adapt, driving "
            "its score down as the new level fills the window")


# -------------------------------------------------------------------
# Threshold detector on sample NAB data (step-007 gate support:
# produces valid scores on real data)
# -------------------------------------------------------------------

class TestThresholdOnSampleData:
    """Run ThresholdDetector on real NAB streams and verify it scores."""

    def test_scores_on_nyc_taxi(self):
        """Threshold detector produces valid binary scores on a real
        stream."""
        streams = load_all_streams()
        ts, vals = streams["realKnownCause/nyc_taxi.csv"]
        det = ThresholdDetector()
        scores = [det.handle_record(t, v) for t, v in zip(ts, vals)]
        assert all(s in (0.0, 1.0) for s in scores)
        assert all(s == 0.0 for s in scores[:det.warmup])

    def test_scores_on_multiple_streams(self):
        """Threshold detector produces valid scores on >=5 streams."""
        streams = load_all_streams()
        keys = list(streams.keys())[:5]
        for key in keys:
            ts, vals = streams[key]
            det = ThresholdDetector()
            scores = [det.handle_record(t, v)
                      for t, v in zip(ts, vals)]
            assert all(s in (0.0, 1.0) for s in scores), (
                f"non-binary score on {key}")
            assert len(scores) == len(ts)
