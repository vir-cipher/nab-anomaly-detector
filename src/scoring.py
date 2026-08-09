"""NAB Scoring Protocol.

Implements the Numenta Anomaly Benchmark scoring as described in:
  Lavin & Ahmad, 'Evaluating Real-Time Anomaly Detection Algorithms --
  the Numenta Anomaly Benchmark', 2015.

Three application profiles (from NAB config/profiles.json):
  - standard:       tpWeight=1.0, fpWeight=0.11, fnWeight=1.0
  - reward_low_fp:  tpWeight=1.0, fpWeight=0.22, fnWeight=1.0
  - reward_low_fn:  tpWeight=1.0, fpWeight=0.11, fnWeight=2.0

Key concepts:
  - Detectors output anomaly scores (0.0-1.0) for each timestamp.
  - A threshold converts scores to detections (>= threshold = anomaly).
  - Inside an anomaly window, only the BEST detection counts (highest score).
  - Outside windows, each detection adds an FP penalty.
  - Missing an entire window adds an FN penalty.
  - Final score normalised: null detector = 0, perfect detector = 100.

Usage:
    from src.scoring import score_stream, score_corpus, load_windows
    windows = load_windows()
    result = score_stream(timestamps, scores,
                          windows["realKnownCause/nyc_taxi.csv"],
                          threshold=0.5)
"""

import json
import math
import os
from datetime import datetime

# ---------------------------------------------------------------------------
# Application profiles — exact values from NAB config/profiles.json
# Primary source: github.com/numenta/NAB/blob/master/config/profiles.json
# ---------------------------------------------------------------------------
PROFILES = {
    "standard":      {"tp_weight": 1.0, "fp_weight": 0.11, "fn_weight": 1.0},
    "reward_low_fp": {"tp_weight": 1.0, "fp_weight": 0.22, "fn_weight": 1.0},
    "reward_low_fn": {"tp_weight": 1.0, "fp_weight": 0.11, "fn_weight": 2.0},
}

# First 15 % of data points excluded from scoring (detector warm-up).
DEFAULT_PROBATIONARY_PCT = 0.15


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------

def scaled_sigmoid(relative_position):
    """Scaled sigmoid that rewards early detection within a window.

    Matches NAB's ``scaledSigmoid`` exactly:
      position -1.0 (window start)  -> ~+0.987  (high reward)
      position  0.0 (window end)    ->   0.0    (neutral)
      position +1.0 (past window)   -> ~-0.987  (penalty)
    """
    if relative_position > 3.0:
        return -1.0
    return 2.0 / (1.0 + math.exp(5.0 * relative_position)) - 1.0


def _probationary_length(num_rows, pct=DEFAULT_PROBATIONARY_PCT):
    """Rows excluded from scoring (detector warm-up).

    Matches NAB: ``min(floor(pct * num_rows), pct * 5000)``.
    """
    return min(math.floor(pct * num_rows), int(pct * 5000))


def _parse_timestamp(ts_str):
    """Parse a NAB timestamp string to a ``datetime`` object."""
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(ts_str.strip(), fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse NAB timestamp: {ts_str!r}")


# ---------------------------------------------------------------------------
# Per-stream scoring
# ---------------------------------------------------------------------------

def _window_index_ranges(timestamps, windows):
    """Map each (start_dt, end_dt) window to (left_idx, right_idx) inclusive."""
    ranges = []
    for w_start, w_end in windows:
        left = right = None
        for i, ts in enumerate(timestamps):
            if left is None and ts >= w_start:
                left = i
            if ts <= w_end:
                right = i
        if left is not None and right is not None:
            ranges.append((left, right))
    return ranges


def score_stream(timestamps, anomaly_scores, windows, threshold,
                 profile="standard",
                 probationary_pct=DEFAULT_PROBATIONARY_PCT):
    """Score a single data stream at a given detection threshold.

    Args:
        timestamps:       list of datetime objects, one per data point.
        anomaly_scores:   list of floats in [0.0, 1.0], one per data point.
        windows:          list of (start_dt, end_dt) tuples (anomaly windows).
        threshold:        float — scores >= threshold count as detections.
        profile:          "standard", "reward_low_fp", or "reward_low_fn".
        probationary_pct: fraction of initial rows excluded from scoring.

    Returns:
        dict with keys: raw_score, tp, fp, fn, tn, total, num_windows.
    """
    if profile not in PROFILES:
        raise ValueError(
            f"Unknown profile '{profile}'. Use: {list(PROFILES.keys())}")

    weights = PROFILES[profile]
    tp_w = weights["tp_weight"]
    fp_w = weights["fp_weight"]
    fn_w = weights["fn_weight"]

    n = len(timestamps)
    if n == 0:
        return {"raw_score": 0.0, "tp": 0, "fp": 0, "fn": 0,
                "tn": 0, "total": 0, "num_windows": 0}

    prob_len = _probationary_length(n, probationary_pct)
    max_tp = scaled_sigmoid(-1.0)        # ~0.9866 — used to normalise TP

    # Map windows to index ranges
    window_ranges = _window_index_ranges(timestamps, windows)

    # Map each point index -> its window index (None = outside all windows)
    point_to_window = [None] * n
    for w_idx, (left, right) in enumerate(window_ranges):
        for i in range(left, right + 1):
            point_to_window[i] = w_idx

    # Per-window state
    window_best = {w: -fn_w for w in range(len(window_ranges))}
    window_hit  = {w: False for w in range(len(window_ranges))}

    fp_total = 0.0
    tp_count = fp_count = tn_count = 0

    # Track most-recent exited window (for FP position calculation)
    prev_right = None
    prev_width = None

    for i in range(prob_len, n):
        detected = (anomaly_scores[i] >= threshold)
        w_idx = point_to_window[i]

        if w_idx is not None:
            left, right = window_ranges[w_idx]
            width = float(right - left + 1)

            if detected:
                tp_count += 1
                window_hit[w_idx] = True
                position = -(right - i + 1) / width
                unweighted = scaled_sigmoid(position)
                weighted = unweighted * tp_w / max_tp
                window_best[w_idx] = max(window_best[w_idx], weighted)
            else:
                tn_count += 1

            # Track window exit for subsequent FP scoring
            if i == right:
                prev_right = right
                prev_width = width
        else:
            # Outside any window
            if detected:
                fp_count += 1
                if prev_right is None:
                    unweighted = -1.0   # no preceding window — max penalty
                else:
                    dist = abs(prev_right - i)
                    denom = max(prev_width - 1, 1.0)
                    pos_past = dist / denom
                    unweighted = scaled_sigmoid(pos_past)
                fp_total += unweighted * fp_w
            else:
                tn_count += 1

    fn_count = sum(1 for hit in window_hit.values() if not hit)
    raw_score = sum(window_best.values()) + fp_total

    return {
        "raw_score": raw_score,
        "tp": tp_count,
        "fp": fp_count,
        "fn": fn_count,
        "tn": tn_count,
        "total": n - prob_len,
        "num_windows": len(window_ranges),
    }


# ---------------------------------------------------------------------------
# Corpus-level helpers
# ---------------------------------------------------------------------------

def null_score(total_windows, profile="standard"):
    """Raw score for a null detector (never flags anything).

    Every window is missed -> score = -fnWeight * total_windows.
    """
    return -PROFILES[profile]["fn_weight"] * total_windows


def perfect_score(total_windows, profile="standard"):
    """Raw score for a perfect detector (flags first point of each window).

    Detection at position -1.0 -> weighted score = tpWeight.
    """
    return PROFILES[profile]["tp_weight"] * total_windows


def normalize_score(raw, null_raw, perfect_raw):
    """Normalise raw score to NAB scale: null = 0, perfect = 100."""
    denom = perfect_raw - null_raw
    if denom == 0:
        return 0.0
    return 100.0 * (raw - null_raw) / denom


def load_windows(labels_path=None):
    """Load ground-truth anomaly windows from combined_windows.json.

    Returns:
        dict mapping stream_name -> list of (start_dt, end_dt) tuples.
    """
    if labels_path is None:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        labels_path = os.path.join(
            root, "data", "nab", "labels", "combined_windows.json")

    with open(labels_path) as f:
        raw = json.load(f)

    windows = {}
    for stream_name, wlist in raw.items():
        parsed = []
        for w in wlist:
            parsed.append((_parse_timestamp(w[0]), _parse_timestamp(w[1])))
        windows[stream_name] = parsed
    return windows


def score_corpus(results, windows, threshold, profile="standard",
                 probationary_pct=DEFAULT_PROBATIONARY_PCT):
    """Score a detector across all data streams and normalise.

    Args:
        results:  dict mapping stream_name -> (timestamps, anomaly_scores).
        windows:  dict from ``load_windows()``.
        threshold: detection threshold.
        profile:  application profile name.

    Returns:
        dict: nab_score (0-100 scale), raw_score, per_stream, profile, etc.
    """
    total_raw = 0.0
    total_windows = 0
    per_stream = {}

    for stream_name, (ts, scores) in results.items():
        stream_wins = windows.get(stream_name, [])
        result = score_stream(
            ts, scores, stream_wins, threshold,
            profile=profile, probationary_pct=probationary_pct)
        per_stream[stream_name] = result
        total_raw += result["raw_score"]
        total_windows += result["num_windows"]

    null_raw = null_score(total_windows, profile)
    perfect_raw = perfect_score(total_windows, profile)
    nab = normalize_score(total_raw, null_raw, perfect_raw)

    return {
        "nab_score": nab,
        "raw_score": total_raw,
        "null_raw": null_raw,
        "perfect_raw": perfect_raw,
        "total_windows": total_windows,
        "per_stream": per_stream,
        "profile": profile,
        "num_streams": len(results),
    }
