"""Tests for the hybrid combiner detector (step-012, Phase 14).

Gate: hybrid detector passes unit tests, voting rule documented.
"""

import datetime as dt
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detectors import Detector, WindowedGaussianDetector
from src.iforest import IsolationForestDetector
from src.hybrid import HybridDetector


class ConstDetector(Detector):
    """Deterministic stub: returns a fixed score, counts calls."""

    def __init__(self, value):
        self._value = value
        self.calls = 0
        self.reset_calls = 0

    def handle_record(self, timestamp, value):
        self.calls += 1
        return self._value

    def reset(self):
        self.reset_calls += 1


TS = dt.datetime(2026, 1, 1)


def test_is_a_detector():
    assert isinstance(HybridDetector(), Detector)


def test_default_wraps_gaussian_and_iforest():
    h = HybridDetector()
    assert len(h.detectors) == 2
    assert isinstance(h.detectors[0], WindowedGaussianDetector)
    assert isinstance(h.detectors[1], IsolationForestDetector)


def test_default_weights_favour_statistical():
    h = HybridDetector()
    assert h.weights == pytest.approx([0.7, 0.3])


def test_score_in_unit_interval():
    h = HybridDetector()
    for i in range(300):
        s = h.handle_record(TS, float(i % 7))
        assert 0.0 <= s <= 1.0


def test_weighted_average_math():
    a, b = ConstDetector(1.0), ConstDetector(0.0)
    h = HybridDetector([a, b], weights=[0.7, 0.3],
                       rule="weighted_average")
    assert h.handle_record(TS, 5.0) == pytest.approx(0.7)


def test_weights_are_normalised():
    a, b = ConstDetector(1.0), ConstDetector(0.0)
    h = HybridDetector([a, b], weights=[7, 3])
    assert h.weights == pytest.approx([0.7, 0.3])
    assert h.handle_record(TS, 1.0) == pytest.approx(0.7)


def test_max_rule():
    a, b = ConstDetector(0.2), ConstDetector(0.9)
    h = HybridDetector([a, b], rule="max")
    assert h.handle_record(TS, 1.0) == pytest.approx(0.9)


def test_min_rule():
    a, b = ConstDetector(0.2), ConstDetector(0.9)
    h = HybridDetector([a, b], rule="min")
    assert h.handle_record(TS, 1.0) == pytest.approx(0.2)


def test_mean_rule_ignores_weights():
    a, b = ConstDetector(0.4), ConstDetector(0.8)
    h = HybridDetector([a, b], weights=[0.9, 0.1], rule="mean")
    assert h.handle_record(TS, 1.0) == pytest.approx(0.6)


def test_every_subdetector_sees_every_point():
    a, b = ConstDetector(0.5), ConstDetector(0.5)
    h = HybridDetector([a, b], rule="max")
    for _ in range(10):
        h.handle_record(TS, 1.0)
    assert a.calls == 10
    assert b.calls == 10


def test_reset_propagates():
    a, b = ConstDetector(0.5), ConstDetector(0.5)
    h = HybridDetector([a, b])
    h.reset()
    assert a.reset_calls == 1
    assert b.reset_calls == 1


def test_invalid_rule_raises():
    with pytest.raises(ValueError):
        HybridDetector(rule="median")


def test_weight_length_mismatch_raises():
    with pytest.raises(ValueError):
        HybridDetector([ConstDetector(0.5), ConstDetector(0.5)],
                       weights=[1.0])


def test_negative_weight_raises():
    with pytest.raises(ValueError):
        HybridDetector([ConstDetector(0.5), ConstDetector(0.5)],
                       weights=[1.0, -1.0])


def test_zero_weight_sum_raises():
    with pytest.raises(ValueError):
        HybridDetector([ConstDetector(0.5), ConstDetector(0.5)],
                       weights=[0.0, 0.0])


def test_empty_detectors_raises():
    with pytest.raises(ValueError):
        HybridDetector([])


def test_name_reflects_components_and_rule():
    h = HybridDetector(rule="max")
    assert "Hybrid(" in h.name
    assert "WindowedGaussianDetector" in h.name
    assert "IsolationForestDetector" in h.name
    assert "max" in h.name


def test_warmup_first_point_is_zero():
    h = HybridDetector()
    assert h.handle_record(TS, 1.0) == pytest.approx(0.0)


def test_uniform_default_weights_for_three():
    d = [ConstDetector(0.3), ConstDetector(0.6), ConstDetector(0.9)]
    h = HybridDetector(d)
    assert h.weights == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    assert h.handle_record(TS, 1.0) == pytest.approx(0.6)
