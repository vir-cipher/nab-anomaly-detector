"""Tests for the NAB scoring protocol (src/scoring.py).

Validates the implementation against known inputs and expected outputs:
  - scaled sigmoid values at key positions
  - null detector -> NAB score 0
  - perfect detector -> NAB score 100
  - FP penalties, early-vs-late detection, profile differences
  - probationary period exclusion
  - integration with real NAB labels
"""
import math
import os
import sys
from datetime import datetime, timedelta

import pytest

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.scoring import (
    scaled_sigmoid,
    score_stream,
    score_corpus,
    load_windows,
    null_score,
    perfect_score,
    normalize_score,
    PROFILES,
    _probationary_length,
)


# ---------------------------------------------------------------------------
# Helpers — generate simple test data
# ---------------------------------------------------------------------------

def _make_timestamps(n, start="2014-01-01 00:00:00", freq_minutes=5):
    """Return n datetime objects spaced freq_minutes apart."""
    t0 = datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
    return [t0 + timedelta(minutes=i * freq_minutes) for i in range(n)]


def _make_scores(n, detections=None):
    """Return anomaly scores: 0.0 everywhere, 1.0 at indices in *detections*."""
    scores = [0.0] * n
    for idx in (detections or []):
        scores[idx] = 1.0
    return scores


# ---------------------------------------------------------------------------
# 1. Scaled sigmoid
# ---------------------------------------------------------------------------

class TestScaledSigmoid:
    def test_start_of_window(self):
        """Position -1.0 (window start) -> high positive (~0.987)."""
        assert scaled_sigmoid(-1.0) == pytest.approx(0.9866, abs=0.001)

    def test_end_of_window(self):
        """Position 0.0 (window end) -> 0.0 (neutral)."""
        assert scaled_sigmoid(0.0) == pytest.approx(0.0, abs=1e-9)

    def test_past_window(self):
        """Position +1.0 (past window) -> high negative (~-0.987)."""
        assert scaled_sigmoid(1.0) == pytest.approx(-0.9866, abs=0.001)

    def test_far_past_clips(self):
        """Position > 3.0 clips to -1.0."""
        assert scaled_sigmoid(3.5) == -1.0
        assert scaled_sigmoid(100.0) == -1.0

    def test_monotonically_decreasing(self):
        """Score should decrease as position increases."""
        positions = [-1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
        values = [scaled_sigmoid(p) for p in positions]
        for a, b in zip(values, values[1:]):
            assert a > b


# ---------------------------------------------------------------------------
# 2. Probationary length
# ---------------------------------------------------------------------------

class TestProbationaryLength:
    def test_small_dataset(self):
        """15% of 100 = 15 rows excluded."""
        assert _probationary_length(100) == 15

    def test_large_dataset_capped(self):
        """15% of 10000 = 1500 but capped at 750."""
        assert _probationary_length(10000) == 750

    def test_exact_boundary(self):
        """5000 rows -> 15% = 750, matches cap exactly."""
        assert _probationary_length(5000) == 750


# ---------------------------------------------------------------------------
# 3. Null detector (all zeros) -> NAB score exactly 0
# ---------------------------------------------------------------------------

class TestNullDetector:
    def test_single_window_null(self):
        """Null detector on a stream with 1 window -> raw = -fnWeight."""
        n = 100
        ts = _make_timestamps(n)
        scores = [0.0] * n
        # Window covering indices 40-59
        w_start = ts[40]
        w_end = ts[59]
        result = score_stream(ts, scores, [(w_start, w_end)],
                              threshold=0.5, profile="standard",
                              probationary_pct=0.0)
        assert result["fn"] == 1
        assert result["tp"] == 0
        assert result["fp"] == 0
        assert result["raw_score"] == pytest.approx(-1.0, abs=1e-9)

    def test_null_normalised_to_zero(self):
        """Normalised: null raw -> 0."""
        nw = 5
        null_raw = null_score(nw, "standard")   # -5.0
        perf_raw = perfect_score(nw, "standard") # +5.0
        assert normalize_score(null_raw, null_raw, perf_raw) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 4. Perfect detector -> NAB score exactly 100
# ---------------------------------------------------------------------------

class TestPerfectDetector:
    def test_single_window_perfect(self):
        """Detect at the very first point of the window -> max TP score."""
        n = 100
        ts = _make_timestamps(n)
        # Window: indices 40-59
        w_start, w_end = ts[40], ts[59]
        scores = _make_scores(n, detections=[40])  # detect at window start
        result = score_stream(ts, scores, [(w_start, w_end)],
                              threshold=0.5, profile="standard",
                              probationary_pct=0.0)
        assert result["tp"] == 1
        assert result["fn"] == 0
        assert result["fp"] == 0
        # Window score should be ~tpWeight = 1.0
        assert result["raw_score"] == pytest.approx(1.0, abs=0.01)

    def test_perfect_normalised_to_100(self):
        """Normalised: perfect raw -> 100."""
        nw = 5
        null_raw = null_score(nw, "standard")
        perf_raw = perfect_score(nw, "standard")
        assert normalize_score(perf_raw, null_raw, perf_raw) == pytest.approx(100.0)

    def test_two_windows_both_detected(self):
        """Two windows, detect first point of each -> NAB 100."""
        n = 200
        ts = _make_timestamps(n)
        w1 = (ts[30], ts[49])
        w2 = (ts[120], ts[139])
        scores = _make_scores(n, detections=[30, 120])
        result = score_stream(ts, scores, [w1, w2],
                              threshold=0.5, profile="standard",
                              probationary_pct=0.0)
        assert result["tp"] == 2
        assert result["fn"] == 0
        raw = result["raw_score"]
        nab = normalize_score(raw, null_score(2), perfect_score(2))
        assert nab == pytest.approx(100.0, abs=0.5)


# ---------------------------------------------------------------------------
# 5. FP penalty
# ---------------------------------------------------------------------------

class TestFPPenalty:
    def test_fp_outside_window(self):
        """Detection outside any window incurs FP penalty (negative raw)."""
        n = 100
        ts = _make_timestamps(n)
        scores = _make_scores(n, detections=[10])  # no window near here
        result = score_stream(ts, scores, [],
                              threshold=0.5, profile="standard",
                              probationary_pct=0.0)
        assert result["fp"] == 1
        assert result["raw_score"] < 0

    def test_more_fps_worse_score(self):
        """More FP detections -> lower raw score."""
        n = 100
        ts = _make_timestamps(n)
        scores_1fp = _make_scores(n, detections=[10])
        scores_3fp = _make_scores(n, detections=[10, 20, 30])
        r1 = score_stream(ts, scores_1fp, [], threshold=0.5,
                          probationary_pct=0.0)
        r3 = score_stream(ts, scores_3fp, [], threshold=0.5,
                          probationary_pct=0.0)
        assert r3["raw_score"] < r1["raw_score"]


# ---------------------------------------------------------------------------
# 6. Early vs late detection
# ---------------------------------------------------------------------------

class TestEarlyVsLate:
    def test_early_beats_late(self):
        """Detection at window start scores higher than at window end."""
        n = 100
        ts = _make_timestamps(n)
        w = (ts[40], ts[59])  # 20-point window

        scores_early = _make_scores(n, detections=[40])
        scores_late  = _make_scores(n, detections=[59])

        r_early = score_stream(ts, scores_early, [w], threshold=0.5,
                               probationary_pct=0.0)
        r_late  = score_stream(ts, scores_late,  [w], threshold=0.5,
                               probationary_pct=0.0)
        assert r_early["raw_score"] > r_late["raw_score"]


# ---------------------------------------------------------------------------
# 7. Profiles differ
# ---------------------------------------------------------------------------

class TestProfiles:
    def test_reward_low_fp_harsher_on_fps(self):
        """reward_low_fp profile penalises FPs more than standard."""
        n = 100
        ts = _make_timestamps(n)
        scores = _make_scores(n, detections=[10, 20, 30])  # 3 FPs, no windows
        r_std = score_stream(ts, scores, [], threshold=0.5,
                             profile="standard", probationary_pct=0.0)
        r_lfp = score_stream(ts, scores, [], threshold=0.5,
                             profile="reward_low_fp", probationary_pct=0.0)
        # reward_low_fp fpWeight = 0.22 vs standard 0.11
        assert r_lfp["raw_score"] < r_std["raw_score"]

    def test_reward_low_fn_harsher_on_fns(self):
        """reward_low_fn profile penalises missed windows more."""
        n = 100
        ts = _make_timestamps(n)
        scores = [0.0] * n  # null detector, 1 window missed
        w = (ts[40], ts[59])
        r_std = score_stream(ts, scores, [w], threshold=0.5,
                             profile="standard", probationary_pct=0.0)
        r_lfn = score_stream(ts, scores, [w], threshold=0.5,
                             profile="reward_low_fn", probationary_pct=0.0)
        # reward_low_fn fnWeight = 2.0 vs standard 1.0
        assert r_lfn["raw_score"] < r_std["raw_score"]

    def test_three_profiles_exist(self):
        """All three NAB profiles are defined."""
        assert set(PROFILES.keys()) == {
            "standard", "reward_low_fp", "reward_low_fn"}


# ---------------------------------------------------------------------------
# 8. Probationary period exclusion
# ---------------------------------------------------------------------------

class TestProbationary:
    def test_detection_in_probationary_ignored(self):
        """Detections inside probationary period do not count."""
        n = 100
        ts = _make_timestamps(n)
        # Detect at index 5 (inside 15% probationary = first 15 rows)
        scores = _make_scores(n, detections=[5])
        result = score_stream(ts, scores, [], threshold=0.5,
                              probationary_pct=0.15)
        assert result["fp"] == 0  # ignored
        assert result["tp"] == 0

    def test_detection_after_probationary_counts(self):
        """Detections after probationary period count normally."""
        n = 100
        ts = _make_timestamps(n)
        scores = _make_scores(n, detections=[20])  # after first 15
        result = score_stream(ts, scores, [], threshold=0.5,
                              probationary_pct=0.15)
        assert result["fp"] == 1


# ---------------------------------------------------------------------------
# 9. Multiple windows — partial detection
# ---------------------------------------------------------------------------

class TestMultipleWindows:
    def test_one_hit_one_miss(self):
        """Two windows: detect in first, miss second -> 1 TP + 1 FN."""
        n = 200
        ts = _make_timestamps(n)
        w1 = (ts[30], ts[49])
        w2 = (ts[120], ts[139])
        scores = _make_scores(n, detections=[30])  # hit w1, miss w2
        result = score_stream(ts, scores, [w1, w2], threshold=0.5,
                              probationary_pct=0.0)
        assert result["tp"] == 1
        assert result["fn"] == 1


# ---------------------------------------------------------------------------
# 10. No-window stream (all detections are FPs)
# ---------------------------------------------------------------------------

class TestNoWindowStream:
    def test_any_detection_is_fp(self):
        """Stream with no anomaly windows: every detection is FP."""
        n = 50
        ts = _make_timestamps(n)
        scores = _make_scores(n, detections=[10, 20, 30])
        result = score_stream(ts, scores, [], threshold=0.5,
                              probationary_pct=0.0)
        assert result["fp"] == 3
        assert result["fn"] == 0
        assert result["num_windows"] == 0
        assert result["raw_score"] < 0


# ---------------------------------------------------------------------------
# 11. load_windows integration with real NAB labels
# ---------------------------------------------------------------------------

class TestLoadWindows:
    def test_loads_all_58_streams(self):
        """combined_windows.json has keys for all 58 NAB data files."""
        windows = load_windows()
        assert len(windows) == 58

    def test_no_anomaly_streams_empty(self):
        """artificialNoAnomaly files have empty window lists."""
        windows = load_windows()
        for key in windows:
            if key.startswith("artificialNoAnomaly/"):
                assert windows[key] == [], f"{key} should have no windows"

    def test_window_tuples_are_datetimes(self):
        """Each window is a (datetime, datetime) tuple."""
        windows = load_windows()
        for key, wlist in windows.items():
            for w in wlist:
                assert len(w) == 2
                assert isinstance(w[0], datetime)
                assert isinstance(w[1], datetime)
                assert w[0] <= w[1]


# ---------------------------------------------------------------------------
# 12. score_corpus normalisation
# ---------------------------------------------------------------------------

class TestScoreCorpus:
    def test_null_corpus_scores_zero(self):
        """A null detector across two streams normalises to 0."""
        n = 100
        ts = _make_timestamps(n)
        w1 = (ts[40], ts[59])
        w2 = (ts[70], ts[89])

        null_results = {
            "stream_a": (ts, [0.0] * n),
            "stream_b": (ts, [0.0] * n),
        }
        windows = {
            "stream_a": [w1],
            "stream_b": [w2],
        }
        result = score_corpus(null_results, windows, threshold=0.5,
                              probationary_pct=0.0)
        assert result["nab_score"] == pytest.approx(0.0, abs=0.01)

    def test_perfect_corpus_scores_100(self):
        """Perfect detector across two streams normalises to 100."""
        n = 100
        ts = _make_timestamps(n)
        w1 = (ts[40], ts[59])
        w2 = (ts[70], ts[89])

        perf_results = {
            "stream_a": (ts, _make_scores(n, [40])),
            "stream_b": (ts, _make_scores(n, [70])),
        }
        windows = {
            "stream_a": [w1],
            "stream_b": [w2],
        }
        result = score_corpus(perf_results, windows, threshold=0.5,
                              probationary_pct=0.0)
        assert result["nab_score"] == pytest.approx(100.0, abs=0.5)
        assert result["num_streams"] == 2
        assert result["total_windows"] == 2

    def test_invalid_profile_raises(self):
        """Unknown profile name raises ValueError."""
        n = 10
        ts = _make_timestamps(n)
        with pytest.raises(ValueError, match="Unknown profile"):
            score_stream(ts, [0.0] * n, [], threshold=0.5,
                         profile="nonexistent")
