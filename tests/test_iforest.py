"""Tests for the isolation forest detector (step-009, Phase 13).

Gate: isolation forest detector passes unit tests, handles
streaming input.
"""

import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detectors import Detector
from src.iforest import (
    IsolationForest,
    IsolationForestDetector,
    average_path_length,
)
from src.data_loader import load_all_streams


# -------------------------------------------------------------------
# c(n) — the path-length normaliser from the iForest paper
# -------------------------------------------------------------------

class TestAveragePathLength:

    def test_one_point_is_zero(self):
        assert average_path_length(1) == 0.0

    def test_two_points_is_one(self):
        assert average_path_length(2) == 1.0

    def test_monotonically_increasing(self):
        values = [average_path_length(n) for n in (2, 4, 16, 128, 1024)]
        assert values == sorted(values)
        assert values[-1] > values[0]

    def test_known_value_256(self):
        # c(256) ~ 10.24 (widely quoted for psi=256)
        assert average_path_length(256) == pytest.approx(10.24, abs=0.1)


# -------------------------------------------------------------------
# Batch forest
# -------------------------------------------------------------------

def _cluster(n, rng, center=0.0, spread=1.0, dims=2):
    return [tuple(center + rng.gauss(0, spread) for _ in range(dims))
            for _ in range(n)]


class TestIsolationForestBatch:

    def test_fit_returns_self(self):
        rng = random.Random(0)
        forest = IsolationForest(seed=1).fit(_cluster(200, rng))
        assert forest.is_fitted

    def test_fit_empty_raises(self):
        with pytest.raises(ValueError):
            IsolationForest().fit([])

    def test_score_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            IsolationForest().score((0.0, 0.0))

    def test_scores_in_unit_interval(self):
        rng = random.Random(0)
        pts = _cluster(300, rng)
        forest = IsolationForest(seed=1).fit(pts)
        for p in pts[:50]:
            assert 0.0 < forest.score(p) <= 1.0

    def test_outlier_scores_higher_than_inlier(self):
        rng = random.Random(0)
        pts = _cluster(300, rng)
        forest = IsolationForest(seed=1).fit(pts)
        inlier_scores = [forest.score(p) for p in pts[:100]]
        outlier_score = forest.score((50.0, 50.0))
        assert outlier_score > max(inlier_scores)

    def test_deterministic_with_seed(self):
        rng = random.Random(0)
        pts = _cluster(300, rng)
        probe = (3.0, -2.0)
        s1 = IsolationForest(seed=7).fit(pts).score(probe)
        s2 = IsolationForest(seed=7).fit(pts).score(probe)
        assert s1 == s2

    def test_identical_points_score_half(self):
        pts = [(1.0, 1.0)] * 100
        forest = IsolationForest(seed=1).fit(pts)
        # No feature has spread: every tree is a single external node,
        # mean path = c(psi), so score = 2**(-1) = 0.5.
        assert forest.score((1.0, 1.0)) == pytest.approx(0.5, abs=0.01)


# -------------------------------------------------------------------
# Streaming detector
# -------------------------------------------------------------------

def _noisy_stream(n, rng, level=10.0, noise=1.0):
    return [level + rng.gauss(0, noise) for _ in range(n)]


class TestIsolationForestDetectorUnit:

    def test_implements_detector(self):
        assert issubclass(IsolationForestDetector, Detector)

    def test_name(self):
        assert (IsolationForestDetector().name
                == "IsolationForestDetector")

    def test_default_params(self):
        det = IsolationForestDetector()
        assert det.shingle_size == 4
        assert det.train_size == 256
        assert det.retrain_interval == 256
        assert det.n_trees == 64
        assert det.sample_size == 128

    def test_custom_params(self):
        det = IsolationForestDetector(shingle_size=2, train_size=50,
                                      retrain_interval=100,
                                      n_trees=10, sample_size=32)
        assert det.shingle_size == 2
        assert det.train_size == 50
        assert det.warmup == 51

    def test_invalid_params_raise(self):
        with pytest.raises(ValueError):
            IsolationForestDetector(shingle_size=0)
        with pytest.raises(ValueError):
            IsolationForestDetector(train_size=1)

    def test_returns_zero_during_warmup(self):
        rng = random.Random(1)
        det = IsolationForestDetector(train_size=64, n_trees=10)
        scores = [det.handle_record(i, v)
                  for i, v in enumerate(_noisy_stream(det.warmup, rng))]
        assert all(s == 0.0 for s in scores)

    def test_scores_after_warmup(self):
        rng = random.Random(1)
        det = IsolationForestDetector(train_size=64, n_trees=10)
        stream = _noisy_stream(det.warmup + 100, rng)
        scores = [det.handle_record(i, v) for i, v in enumerate(stream)]
        post = scores[det.warmup + 1:]
        assert any(s > 0.0 for s in post)

    def test_score_in_range(self):
        rng = random.Random(2)
        det = IsolationForestDetector(train_size=64, n_trees=10)
        stream = _noisy_stream(400, rng)
        for i, v in enumerate(stream):
            assert 0.0 <= det.handle_record(i, v) <= 1.0

    def test_spike_scores_higher_than_normal(self):
        rng = random.Random(3)
        det = IsolationForestDetector(train_size=128, n_trees=32)
        normal = _noisy_stream(600, rng)
        scores = [det.handle_record(i, v) for i, v in enumerate(normal)]
        baseline = sum(scores[-50:]) / 50.0
        spike_score = det.handle_record(600, 100.0)
        assert spike_score > baseline

    def test_deterministic_across_instances(self):
        rng = random.Random(4)
        stream = _noisy_stream(500, rng)
        det_a = IsolationForestDetector(train_size=64, n_trees=10)
        det_b = IsolationForestDetector(train_size=64, n_trees=10)
        scores_a = [det_a.handle_record(i, v)
                    for i, v in enumerate(stream)]
        scores_b = [det_b.handle_record(i, v)
                    for i, v in enumerate(stream)]
        assert scores_a == scores_b

    def test_reset_clears_state(self):
        rng = random.Random(5)
        det = IsolationForestDetector(train_size=64, n_trees=10)
        stream = _noisy_stream(300, rng)
        first = [det.handle_record(i, v) for i, v in enumerate(stream)]
        det.reset()
        second = [det.handle_record(i, v) for i, v in enumerate(stream)]
        assert first == second
        assert det.fit_count >= 1

    def test_bounded_memory(self):
        rng = random.Random(6)
        det = IsolationForestDetector(train_size=64, n_trees=10)
        for i, v in enumerate(_noisy_stream(1000, rng)):
            det.handle_record(i, v)
        assert len(det._history) <= det.train_size
        assert len(det._buffer) <= det.shingle_size

    def test_retrains_on_schedule(self):
        rng = random.Random(7)
        det = IsolationForestDetector(train_size=64,
                                      retrain_interval=100, n_trees=10)
        for i, v in enumerate(_noisy_stream(500, rng)):
            det.handle_record(i, v)
        # first fit at warmup, then every 100 points afterwards
        assert det.fit_count >= 3


class TestIsolationForestOnSampleData:

    def test_scores_on_nyc_taxi(self):
        streams = load_all_streams()
        ts, vals = streams["realKnownCause/nyc_taxi.csv"]
        det = IsolationForestDetector()
        scores = [det.handle_record(t, v) for t, v in zip(ts, vals)]
        assert all(0.0 <= s <= 1.0 for s in scores)
        assert any(s > 0.0 for s in scores)
        assert all(s == 0.0 for s in scores[:det.warmup])

    def test_scores_on_multiple_streams(self):
        streams = load_all_streams()
        keys = list(streams.keys())[:3]
        for key in keys:
            ts, vals = streams[key]
            det = IsolationForestDetector(n_trees=16)
            scores = [det.handle_record(t, v)
                      for t, v in zip(ts, vals)]
            assert all(0.0 <= s <= 1.0 for s in scores), (
                f"score out of range on {key}")
            assert len(scores) == len(ts)
