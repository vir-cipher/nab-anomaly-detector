"""Hybrid anomaly detector for the NAB benchmark (Phase 14, step-012).

Combines the best statistical detector (WindowedGaussianDetector, the
Phase-12 leaderboard leader at NAB standard=40.13) with the machine-
learning detector (IsolationForestDetector, Phase 13) into a single
streaming detector, under a documented, configurable voting rule.

Why hybridise:
  - The windowed Gaussian is ~7.6x stronger on NAB than the isolation
    forest (40.13 vs 5.25 standard), but the two make DIFFERENT kinds
    of mistakes: the Gaussian models one global amplitude distribution,
    while the forest isolates short multi-point SHAPES (via 4-value
    shingles). A combiner can, in principle, keep the Gaussian's
    strength while borrowing the forest's shape sensitivity.

Voting rule (documented here; empirically scored in step-013):
  On every point BOTH sub-detectors consume the value (so each keeps
  its own streaming state), and their scores in [0,1] are combined by
  one of:
    - "weighted_average" (default): sum(w_i * s_i) with weights
      normalised to sum to 1. Default weights favour the stronger
      statistical detector (0.7) over the forest (0.3), reflecting
      their 7.6x NAB gap -- a plain 50/50 mean would let the forest's
      ~0.5 baseline drag the Gaussian's sharp signal down.
    - "mean": equal-weight average (weights ignored).
    - "max": union / recall-oriented -- fire if EITHER flags the point.
    - "min": intersection / precision-oriented -- fire only if BOTH do.
  During a sub-detector's warmup it emits 0.0, so the combined score is
  simply the weighted combination of whatever each emits at that point.

Primary sources:
  - statistical detector: src/detectors.py (WindowedGaussianDetector)
  - ML detector:          src/iforest.py  (IsolationForestDetector)
"""

from src.detectors import Detector, WindowedGaussianDetector
from src.iforest import IsolationForestDetector

_RULES = ("weighted_average", "mean", "max", "min")


class HybridDetector(Detector):
    """Combine several streaming detectors under one voting rule.

    Args:
        detectors: list of Detector instances to combine. Defaults to
            [WindowedGaussianDetector(), IsolationForestDetector()] --
            the best statistical detector plus the ML detector.
        weights:   list of non-negative floats, one per detector, used
            only by the "weighted_average" rule. Normalised to sum to
            1. Defaults to [0.7, 0.3] for the 2-detector default, or a
            uniform vector otherwise.
        rule:      one of "weighted_average", "mean", "max", "min".

    Every sub-detector sees every point (to preserve its streaming
    state) regardless of which rule is active.
    """

    def __init__(self, detectors=None, weights=None,
                 rule="weighted_average"):
        if detectors is None:
            detectors = [WindowedGaussianDetector(),
                         IsolationForestDetector()]
        detectors = list(detectors)
        if len(detectors) == 0:
            raise ValueError("HybridDetector needs at least one detector")
        if rule not in _RULES:
            raise ValueError(
                "rule must be one of %s, got %r"
                % (", ".join(_RULES), rule))

        n = len(detectors)
        if weights is None:
            weights = [0.7, 0.3] if n == 2 else [1.0 / n] * n
        weights = [float(w) for w in weights]
        if len(weights) != n:
            raise ValueError(
                "weights length (%d) must match detectors (%d)"
                % (len(weights), n))
        if any(w < 0 for w in weights):
            raise ValueError("weights must be non-negative")
        total = sum(weights)
        if total <= 0:
            raise ValueError("weights must sum to a positive value")

        self.detectors = detectors
        self.weights = [w / total for w in weights]
        self.rule = rule

    def handle_record(self, timestamp, value):
        """Feed the point to every sub-detector and combine the scores.

        Returns a combined anomaly score clamped to [0.0, 1.0].
        """
        scores = [d.handle_record(timestamp, value)
                  for d in self.detectors]

        if self.rule == "max":
            combined = max(scores)
        elif self.rule == "min":
            combined = min(scores)
        elif self.rule == "mean":
            combined = sum(scores) / len(scores)
        else:  # weighted_average
            combined = sum(w * s
                           for w, s in zip(self.weights, scores))

        return max(0.0, min(1.0, combined))

    def reset(self):
        """Reset every sub-detector for a new stream."""
        for d in self.detectors:
            d.reset()

    @property
    def name(self):
        """e.g. 'Hybrid(WindowedGaussianDetector+...,weighted_average)'."""
        inner = "+".join(d.name for d in self.detectors)
        return "Hybrid(%s,%s)" % (inner, self.rule)
