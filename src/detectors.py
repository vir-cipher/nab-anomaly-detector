"""Anomaly detectors for the NAB benchmark.

Each detector implements a streaming interface: feed one
(timestamp, value) pair at a time, get back an anomaly score
in [0.0, 1.0].  This mirrors how real-time detectors work â€”
you cannot peek ahead.

Usage:
    from src.detectors import NullDetector, WindowedGaussianDetector, EWMADetector, ZScoreDetector, ThresholdDetector
    det = WindowedGaussianDetector()
    score = det.handle_record(timestamp, value)
"""

import math
from abc import ABC, abstractmethod


class Detector(ABC):
    """Base class for all streaming anomaly detectors.

    Subclasses implement ``handle_record`` which processes one
    data point and returns an anomaly score between 0.0 and 1.0.
    """

    @abstractmethod
    def handle_record(self, timestamp, value):
        """Process one data point and return an anomaly score.

        Args:
            timestamp: datetime object for this data point.
            value:     float â€” the observed measurement.

        Returns:
            float in [0.0, 1.0] where 0.0 = normal, 1.0 = anomaly.
        """
        raise NotImplementedError

    def reset(self):
        """Reset internal state for a new stream."""
        pass

    @property
    def name(self):
        """Human-readable detector name."""
        return self.__class__.__name__


class NullDetector(Detector):
    """Detector that always returns 0.0 (no anomaly).

    Purpose: scoring baseline.  A null detector should score
    exactly 0.0 on the NAB normalised scale (by definition).
    """

    def handle_record(self, timestamp, value):
        """Always returns 0.0 â€” nothing is ever anomalous."""
        return 0.0


# ---------------------------------------------------------------------------
# Windowed Gaussian detector â€” matches NAB reference implementation
# Primary source: github.com/numenta/NAB/blob/master/nab/detectors/
#   gaussian/windowedGaussian_detector.py
# Published NAB scores: standard=39.6, low_fp=20.9, low_fn=47.4
# ---------------------------------------------------------------------------

def _normal_tail_probability(x, mean, std):
    """Tail probability of the normal distribution (Q-function).

    Returns P(X > x) for X ~ N(mean, std^2).
    Uses math.erfc for numerical stability.
    Mirrors NAB's ``normalProbability`` exactly.
    """
    if x < mean:
        return _normal_tail_probability(2 * mean - x, mean, std)
    z = (x - mean) / std
    return 0.5 * math.erfc(z / math.sqrt(2))


class WindowedGaussianDetector(Detector):
    """Sliding-window Gaussian anomaly detector.

    Maintains a window of recent values, computes their mean and
    standard deviation, and scores each new point by how unlikely
    it is under that Gaussian: score = 1 - Q(value, mean, std).

    Uses incremental statistics for O(1) per-point updates.
    Window and step sizes match NAB reference (6400 / 100).
    """

    def __init__(self, window_size=6400, step_size=100):
        self.window_size = window_size
        self.step_size = step_size
        self.window_data = []
        self.step_buffer = []
        self._sum = 0.0
        self._sum_sq = 0.0
        self.mean = 0.0
        self.std = 1.0

    def handle_record(self, timestamp, value):
        """Return anomaly score for one data point."""
        score = 0.0
        if len(self.window_data) > 0:
            score = 1.0 - _normal_tail_probability(
                value, self.mean, self.std)

        if len(self.window_data) < self.window_size:
            self.window_data.append(value)
            self._sum += value
            self._sum_sq += value * value
            self._recompute_from_sums()
        else:
            self.step_buffer.append(value)
            if len(self.step_buffer) == self.step_size:
                removed = self.window_data[:self.step_size]
                self.window_data = self.window_data[self.step_size:]
                self.window_data.extend(self.step_buffer)
                for v in removed:
                    self._sum -= v
                    self._sum_sq -= v * v
                for v in self.step_buffer:
                    self._sum += v
                    self._sum_sq += v * v
                self.step_buffer = []
                self._recompute_from_sums()

        return score

    def _recompute_from_sums(self):
        """Derive mean and std from running sums â€” O(1)."""
        n = len(self.window_data)
        if n == 0:
            self.mean, self.std = 0.0, 1.0
            return
        self.mean = self._sum / n
        variance = self._sum_sq / n - self.mean * self.mean
        if variance < 0:
            variance = 0.0   # guard floating-point underflow
        self.std = math.sqrt(variance)
        if self.std == 0.0:
            self.std = 1e-6

    def reset(self):
        """Reset for a new stream."""
        self.window_data = []
        self.step_buffer = []
        self._sum = 0.0
        self._sum_sq = 0.0
        self.mean = 0.0
        self.std = 1.0


# ---------------------------------------------------------------------------
# EWMA (Exponentially Weighted Moving Average) detector â€” step-005
# Classic streaming detector: lightweight, O(1) per point, no window needed.
# Scores via the same tail-probability method as WindowedGaussianDetector.
# ---------------------------------------------------------------------------

class EWMADetector(Detector):
    """Exponentially Weighted Moving Average anomaly detector.

    Maintains a smoothed running average and variance of values.
    After a short warmup period (initial statistics gathered),
    scores each point by how far it deviates from the EWMA
    prediction, using the same tail-probability method as
    WindowedGaussianDetector.

    Advantages over windowed approaches:
    - O(1) memory (no window buffer needed).
    - Naturally weights recent values more heavily.
    - Single tuning knob (alpha).

    Parameters:
        alpha:  Smoothing factor in (0, 1). Higher = faster
                reaction to level changes. Default 0.1.
        warmup: Number of initial points to collect before
                scoring.  During warmup the detector gathers
                mean/variance estimates and returns 0.0.
                Default 10.
    """

    def __init__(self, alpha=0.1, warmup=10):
        self.alpha = alpha
        self.warmup = warmup
        self._ewma = None
        self._ewma_var = 0.0
        self._count = 0
        self._warmup_values = []

    def handle_record(self, timestamp, value):
        """Process one data point and return an anomaly score.

        During warmup (first ``warmup`` points), collects values
        to initialise the EWMA mean and variance, returning 0.0.
        After warmup, scores each point by its deviation from the
        current EWMA estimate.
        """
        self._count += 1

        # --- warmup phase: collect data, return 0.0 ---
        if self._count <= self.warmup:
            self._warmup_values.append(value)
            if self._count == self.warmup:
                n = len(self._warmup_values)
                mean = sum(self._warmup_values) / n
                self._ewma = mean
                self._ewma_var = (
                    sum((v - mean) ** 2 for v in self._warmup_values) / n
                )
                self._warmup_values = []  # free memory
            return 0.0

        # --- scoring phase ---
        # Deviation from current EWMA prediction
        std = math.sqrt(self._ewma_var) if self._ewma_var > 0 else 1e-6
        score = 1.0 - _normal_tail_probability(value, self._ewma, std)

        # Update EWMA mean and variance
        error = value - self._ewma
        self._ewma = self.alpha * value + (1 - self.alpha) * self._ewma
        self._ewma_var = (
            self.alpha * (error ** 2) + (1 - self.alpha) * self._ewma_var
        )

        return max(0.0, min(1.0, score))

    def reset(self):
        """Reset internal state for a new stream."""
        self._ewma = None
        self._ewma_var = 0.0
        self._count = 0
        self._warmup_values = []

# ---------------------------------------------------------------------------
# Z-score detector -- step-006
# Classic rolling-window Z-score: small sliding window, scores each point
# by how many standard deviations it falls from the window mean.
# Faster reaction than WindowedGaussian (128 vs 6400 default window),
# updates every point (no step buffer), uses deque for O(1) sliding.
# ---------------------------------------------------------------------------

from collections import deque as _deque


class ZScoreDetector(Detector):
    """Rolling Z-score anomaly detector.

    Maintains a small sliding window of recent values using a
    deque.  For each new point, computes its z-score against the
    window's mean and standard deviation, then converts to an
    anomaly score via tail probability (same method as the other
    detectors, so scores are directly comparable).

    Compared to WindowedGaussianDetector:
    - Smaller default window (128 vs 6400) -- reacts faster
      to local shifts.
    - Updates statistics every point (no step buffer).
    - Uses collections.deque for efficient O(1) sliding.

    Compared to EWMADetector:
    - Fixed-size window with equal weighting (bounded memory).
    - Old data forgotten completely once it slides out, rather
      than decaying exponentially.

    Parameters:
        window_size: Number of recent values to keep. Default 128.
        warmup:      Minimum points collected before scoring
                     begins. Default 30.  Must be >= 2 so that
                     a meaningful std can be computed.
    """

    def __init__(self, window_size=128, warmup=30):
        self.window_size = window_size
        self.warmup = max(warmup, 2)
        self._window = _deque(maxlen=window_size)
        self._sum = 0.0
        self._sum_sq = 0.0

    def handle_record(self, timestamp, value):
        """Process one data point and return an anomaly score.

        During warmup (fewer than ``warmup`` points in the window),
        returns 0.0.  After warmup, scores each point by its
        deviation from the current window statistics.
        """
        # --- score against current window (before adding new point) ---
        score = 0.0
        n = len(self._window)
        if n >= self.warmup:
            mean = self._sum / n
            variance = self._sum_sq / n - mean * mean
            if variance < 0:
                variance = 0.0
            std = math.sqrt(variance) if variance > 0 else 1e-6
            score = 1.0 - _normal_tail_probability(value, mean, std)
            score = max(0.0, min(1.0, score))

        # --- slide the window ---
        if len(self._window) == self.window_size:
            old = self._window[0]
            self._sum -= old
            self._sum_sq -= old * old
        self._window.append(value)
        self._sum += value
        self._sum_sq += value * value

        return score

    def reset(self):
        """Reset internal state for a new stream."""
        self._window.clear()
        self._sum = 0.0
        self._sum_sq = 0.0


# ---------------------------------------------------------------------------
# Simple threshold detector -- step-007
# The simplest possible streaming detector: learn a fixed [low, high]
# band during warmup and never adapt again. Binary score (0.0 or 1.0),
# unlike the continuous tail-probability scores of the other three
# detectors -- this is the "dumb baseline" contrast for step-008.
# ---------------------------------------------------------------------------


class ThresholdDetector(Detector):
    """Static threshold anomaly detector.

    Learns a fixed mean/std during a warmup window, derives a
    static [low, high] band (mean +/- n_std * std), and never
    updates it again -- unlike WindowedGaussianDetector (recomputes
    every step_size points), EWMADetector (decays continuously),
    or ZScoreDetector (rolling window). This "set it and forget
    it" behaviour is the classic simplest streaming detector and
    gives a binary anomaly signal (0.0 / 1.0) rather than a
    graduated probability -- a useful contrast baseline for the
    Phase-12 comparison in step-008.

    Parameters:
        warmup: Number of initial points used to compute the
                static mean/std. Default 100.
        n_std:  Number of standard deviations from the mean that
                defines the [low, high] band. Default 3.0.
    """

    def __init__(self, warmup=100, n_std=3.0):
        self.warmup = max(warmup, 2)
        self.n_std = n_std
        self._warmup_values = []
        self._count = 0
        self.low = None
        self.high = None

    def handle_record(self, timestamp, value):
        """Process one data point and return a binary anomaly score.

        During warmup, collects values and returns 0.0. On the
        point that completes warmup, computes the static
        [low, high] band from the warmup sample. After warmup,
        returns 1.0 if the value falls outside the frozen band,
        else 0.0. The band itself is never updated again.
        """
        self._count += 1

        if self.low is None:
            self._warmup_values.append(value)
            if self._count >= self.warmup:
                n = len(self._warmup_values)
                mean = sum(self._warmup_values) / n
                variance = (
                    sum((v - mean) ** 2 for v in self._warmup_values) / n
                )
                std = math.sqrt(variance) if variance > 0 else 1e-6
                self.low = mean - self.n_std * std
                self.high = mean + self.n_std * std
                self._warmup_values = []  # free memory
            return 0.0

        return 1.0 if (value < self.low or value > self.high) else 0.0

    def reset(self):
        """Reset internal state for a new stream."""
        self._warmup_values = []
        self._count = 0
        self.low = None
        self.high = None
