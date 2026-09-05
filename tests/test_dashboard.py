"""Tests for the step-015 Streamlit dashboard (src/dashboard.py).

The dashboard's computation is pure and importable, so most tests run
without Streamlit. One smoke test drives the actual Streamlit app via
AppTest and is skipped where Streamlit is not installed (it is an
optional 'dashboard' extra, not a core dependency).
"""

import math
import os
from datetime import datetime, timedelta

import pytest

from src import dashboard
from src.detectors import (
    NullDetector,
    WindowedGaussianDetector,
    EWMADetector,
    ZScoreDetector,
)


def _synth(n=300, period=50):
    """Deterministic synthetic stream: a sine wave with one late spike."""
    base = datetime(2020, 1, 1)
    ts = [base + timedelta(minutes=5 * i) for i in range(n)]
    vals = [math.sin(2 * math.pi * i / period) for i in range(n)]
    vals[int(n * 0.8)] += 25.0  # an obvious anomaly
    return ts, vals


# --- registry & discovery -------------------------------------------------

def test_registry_has_five_detectors():
    assert set(dashboard.DETECTORS) == {
        "Windowed Gaussian", "EWMA", "Z-score", "Threshold", "Null (baseline)"}


def test_available_streams_featured_first():
    streams = dashboard.available_streams()
    assert streams, "no NAB streams found on disk"
    assert all("/" in s and s.endswith(".csv") for s in streams)
    assert streams[0] == "realKnownCause/nyc_taxi.csv"
    featured_present = [s for s in dashboard.FEATURED_STREAMS if s in streams]
    assert streams[:len(featured_present)] == featured_present


def test_load_named_stream_roundtrip():
    ts, vals = dashboard.load_named_stream("realKnownCause/nyc_taxi.csv")
    assert len(ts) == len(vals) > 0
    assert all(isinstance(v, float) for v in vals[:20])


# --- scoring / detections -------------------------------------------------

def test_stream_scores_null_all_zero():
    ts, vals = _synth()
    scores = dashboard.stream_scores(NullDetector, ts, vals)
    assert scores == [0.0] * len(vals)


def test_stream_scores_in_unit_interval():
    ts, vals = _synth()
    for det in (WindowedGaussianDetector, EWMADetector, ZScoreDetector):
        scores = dashboard.stream_scores(det, ts, vals)
        assert len(scores) == len(vals)
        assert all(0.0 <= s <= 1.0 for s in scores)


@pytest.mark.parametrize(
    "det", [WindowedGaussianDetector, EWMADetector, ZScoreDetector])
def test_scores_are_causal_no_lookahead(det):
    """Prefix scores must equal the full run's first-k scores.

    This is the defining property of a real-time detector: the score
    at point i cannot depend on any future point. If it holds, the
    'live' dashboard is showing genuine streaming scores.
    """
    ts, vals = _synth()
    k = 137
    full = dashboard.stream_scores(det, ts, vals)
    prefix = dashboard.stream_scores(det, ts[:k], vals[:k])
    assert prefix == full[:k]


def test_detections_threshold():
    scores = [0.0, 0.4, 0.5, 0.9, 0.49999]
    assert dashboard.detections(scores, 0.5) == [
        False, False, True, True, False]


# --- windows & frame state ------------------------------------------------

def test_window_spans_matches_scoring_ranges():
    """window_spans must agree with scoring's own window->index mapping."""
    from src.scoring import _window_index_ranges
    base = datetime(2020, 1, 1)
    ts = [base + timedelta(minutes=i) for i in range(100)]
    windows = [(ts[10], ts[20]), (ts[70], ts[75])]
    assert dashboard.window_spans(ts, windows) == _window_index_ranges(
        ts, windows)


def test_window_spans_simple_bounds():
    base = datetime(2020, 1, 1)
    ts = [base + timedelta(minutes=i) for i in range(50)]
    assert dashboard.window_spans(ts, [(ts[5], ts[9])]) == [(5, 9)]


def test_frame_metrics_causal_and_monotonic():
    ts, vals = _synth()
    scores = dashboard.stream_scores(WindowedGaussianDetector, ts, vals)
    flags = dashboard.detections(scores, 0.5)
    spans = dashboard.window_spans(ts, [(ts[230], ts[245])])
    n = len(vals)

    first = dashboard.frame_metrics(scores, flags, spans, 0)
    assert first["seen"] == 1
    assert first["windows_seen"] == 0  # window starts at 230, not yet seen

    last = dashboard.frame_metrics(scores, flags, spans, n - 1)
    assert last["seen"] == n
    assert last["windows_seen"] == 1

    prev = 0
    for pos in range(0, n, 25):
        alarms = dashboard.frame_metrics(scores, flags, spans, pos)["alarms"]
        assert alarms >= prev  # alarms never decrease as the reveal advances
        prev = alarms


def test_frame_metrics_empty_stream():
    m = dashboard.frame_metrics([], [], [], 0)
    assert m["seen"] == 0 and m["alarms"] == 0 and m["windows_seen"] == 0


def test_frame_metrics_position_clamped():
    ts, vals = _synth(n=50)
    scores = dashboard.stream_scores(NullDetector, ts, vals)
    flags = dashboard.detections(scores, 0.5)
    m = dashboard.frame_metrics(scores, flags, [], 9999)
    assert m["seen"] == 50  # position clamped to stream length


# --- Streamlit app smoke test (skipped if Streamlit not installed) --------

def test_dashboard_app_runs():
    """The actual Streamlit app builds and runs without error.

    Uses Streamlit's AppTest harness (streamlit.testing). Skipped where
    Streamlit is not installed, since it is an optional extra rather
    than a core dependency of the pure-Python benchmark.
    """
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    app_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "src", "dashboard.py")
    at = AppTest.from_file(app_path, default_timeout=90)
    at.run()

    assert not at.exception
    assert any("Live Streaming Anomaly Monitor" in t.value for t in at.title)
    assert len(at.selectbox) >= 2  # data stream + detector
    assert len(at.slider) >= 1     # alarm threshold (+ reveal position)
