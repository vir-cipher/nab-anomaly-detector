"""Anomaly detectors for the NAB benchmark.

Each detector implements a streaming interface: feed one
(timestamp, value) pair at a time, get back an anomaly score
in [0.0, 1.0].  This mirrors how real-time detectors work —
you cannot peek ahead.

Usage:
    from src.detectors import NullDetector
    det = NullDetector()
    score = det.handle_record(timestamp, value)
"""

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
            value:     float — the observed measurement.

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
        """Always returns 0.0 — nothing is ever anomalous."""
        return 0.0
