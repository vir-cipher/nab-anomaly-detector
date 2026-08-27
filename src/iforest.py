"""Isolation forest anomaly detector for the NAB benchmark (Phase 13).

Pure-Python implementation of the isolation forest algorithm with a
streaming adapter so it plugs into the same interface as the
statistical detectors in ``src.detectors``.

Primary source: Liu, Ting & Zhou (2008), "Isolation Forest", ICDM.
  - anomaly score: s(x, n) = 2 ** (-E[h(x)] / c(n))
  - c(n) = 2 * H(n-1) - 2*(n-1)/n, H(i) = ln(i) + Euler-Mascheroni

Streaming adaptation (documented in .project-meta/decisions.log):
  - each point becomes a "shingle" (the last ``shingle_size`` values),
    turning the 1-D stream into small vectors a forest can isolate;
  - the forest trains on the most recent ``train_size`` shingles and
    retrains every ``retrain_interval`` points, so it adapts to drift;
  - before the first training completes the detector returns 0.0,
    matching the warmup convention of the other detectors.
"""

import math
import random
from collections import deque

from src.detectors import Detector

_EULER = 0.5772156649015329


def average_path_length(n):
    """c(n): average unsuccessful-search path length in a BST of n points.

    Used both to normalise scores and to estimate the depth of
    unsplit external nodes.  c(1) = 0, c(2) = 1.
    """
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    harmonic = math.log(n - 1) + _EULER
    return 2.0 * harmonic - 2.0 * (n - 1) / n


class _Node:
    """One node of an isolation tree.

    Internal nodes hold a split (feature index + split value) and two
    children; external (leaf) nodes hold only the number of points
    that reached them.
    """

    __slots__ = ("feature", "split", "left", "right", "size")

    def __init__(self, feature=None, split=None, left=None,
                 right=None, size=0):
        self.feature = feature
        self.split = split
        self.left = left
        self.right = right
        self.size = size


def _build_tree(points, depth, height_limit, rng):
    """Recursively grow one isolation tree on ``points`` (tuples)."""
    if depth >= height_limit or len(points) <= 1:
        return _Node(size=len(points))

    n_features = len(points[0])
    # Pick a random feature that actually has spread; if none do,
    # the points are identical and cannot be split further.
    candidates = list(range(n_features))
    rng.shuffle(candidates)
    for feature in candidates:
        lo = min(p[feature] for p in points)
        hi = max(p[feature] for p in points)
        if lo < hi:
            split = rng.uniform(lo, hi)
            left_pts = [p for p in points if p[feature] < split]
            right_pts = [p for p in points if p[feature] >= split]
            return _Node(
                feature=feature,
                split=split,
                left=_build_tree(left_pts, depth + 1, height_limit, rng),
                right=_build_tree(right_pts, depth + 1, height_limit, rng),
            )
    return _Node(size=len(points))


def _path_length(point, node, depth=0):
    """Depth at which ``point`` is isolated, plus the c(n) adjustment
    for external nodes that still hold multiple points."""
    while node.feature is not None:
        if point[node.feature] < node.split:
            node = node.left
        else:
            node = node.right
        depth += 1
    return depth + average_path_length(node.size)


class IsolationForest:
    """Batch isolation forest (Liu et al. 2008), pure Python.

    Args:
        n_trees:     number of trees in the ensemble.
        sample_size: sub-sample size per tree (paper's psi).
        seed:        seed for the internal RNG (deterministic).
    """

    def __init__(self, n_trees=64, sample_size=128, seed=42):
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.seed = seed
        self._rng = random.Random(seed)
        self._trees = []
        self._psi = 0

    def fit(self, points):
        """Train the forest on a list of equal-length tuples."""
        if not points:
            raise ValueError("cannot fit isolation forest on no data")
        self._trees = []
        self._psi = min(self.sample_size, len(points))
        height_limit = math.ceil(math.log2(max(self._psi, 2)))
        for _ in range(self.n_trees):
            sample = self._rng.sample(points, self._psi)
            self._trees.append(_build_tree(sample, 0, height_limit,
                                           self._rng))
        return self

    @property
    def is_fitted(self):
        return bool(self._trees)

    def score(self, point):
        """Anomaly score in (0, 1]: ~0.5 average, near 1 anomalous."""
        if not self._trees:
            raise RuntimeError("forest not fitted yet")
        mean_path = (sum(_path_length(point, t) for t in self._trees)
                     / len(self._trees))
        denom = average_path_length(self._psi)
        if denom == 0.0:
            return 0.5
        return 2.0 ** (-mean_path / denom)


class IsolationForestDetector(Detector):
    """Streaming adapter that wraps IsolationForest for NAB streams.

    Each incoming value is appended to a rolling shingle (the last
    ``shingle_size`` values); the shingle is the feature vector the
    forest isolates.  The forest first trains once ``train_size``
    shingles have accumulated, then retrains every
    ``retrain_interval`` points on the most recent ``train_size``
    shingles, so it tracks concept drift with bounded memory.
    """

    def __init__(self, shingle_size=4, train_size=256,
                 retrain_interval=256, n_trees=64, sample_size=128,
                 seed=42):
        if shingle_size < 1:
            raise ValueError("shingle_size must be >= 1")
        if train_size < 2:
            raise ValueError("train_size must be >= 2")
        self.shingle_size = shingle_size
        self.train_size = train_size
        self.retrain_interval = retrain_interval
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.seed = seed
        self._buffer = deque(maxlen=shingle_size)
        self._history = deque(maxlen=train_size)
        self._forest = None
        self._since_retrain = 0
        self.fit_count = 0

    @property
    def warmup(self):
        """Points consumed before the first non-zero score is possible."""
        return self.shingle_size - 1 + self.train_size

    def handle_record(self, timestamp, value):
        """Process one data point, return anomaly score in [0.0, 1.0]."""
        self._buffer.append(float(value))
        if len(self._buffer) < self.shingle_size:
            return 0.0

        shingle = tuple(self._buffer)
        self._history.append(shingle)

        if self._forest is None:
            if len(self._history) >= self.train_size:
                self._fit()
            return 0.0

        self._since_retrain += 1
        if self._since_retrain >= self.retrain_interval:
            self._fit()

        score = self._forest.score(shingle)
        return min(1.0, max(0.0, score))

    def _fit(self):
        """(Re)train the forest on the most recent shingles."""
        self._forest = IsolationForest(
            n_trees=self.n_trees, sample_size=self.sample_size,
            seed=self.seed).fit(list(self._history))
        self._since_retrain = 0
        self.fit_count += 1

    def reset(self):
        """Reset all state for a new stream (fully deterministic)."""
        self._buffer = deque(maxlen=self.shingle_size)
        self._history = deque(maxlen=self.train_size)
        self._forest = None
        self._since_retrain = 0
        self.fit_count = 0
