"""Streamlit dashboard for live anomaly visualisation (step-015).

Streams a chosen NAB feed point-by-point through a chosen streaming
detector and shows the anomaly score building up in real time -- the
way a production monitor sees it, with no look-ahead. Ground-truth
anomaly windows and an alarm threshold are overlaid so hits, misses
and false alarms are visible as they happen.

All computation lives in plain, importable functions (covered by
tests/test_dashboard.py). Streamlit is imported lazily inside main()
so the module -- and its logic -- can be imported and tested without
Streamlit installed.

Run locally:
    pip install -r requirements-dashboard.txt
    streamlit run src/dashboard.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_stream
from src.detectors import (
    NullDetector,
    WindowedGaussianDetector,
    EWMADetector,
    ZScoreDetector,
    ThresholdDetector,
)
from src.scoring import load_windows


# Detector registry -- the very classes scored elsewhere in the project,
# so the dashboard shows the same numbers the leaderboard is built from.
DETECTORS = {
    "Windowed Gaussian": WindowedGaussianDetector,
    "EWMA": EWMADetector,
    "Z-score": ZScoreDetector,
    "Threshold": ThresholdDetector,
    "Null (baseline)": NullDetector,
}

# High-signal streams surfaced first (clear, documented anomalies).
FEATURED_STREAMS = [
    "realKnownCause/nyc_taxi.csv",
    "realKnownCause/machine_temperature_system_failure.csv",
    "realKnownCause/ambient_temperature_system_failure.csv",
    "realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv",
    "artificialWithAnomaly/art_daily_jumpsup.csv",
]

DEFAULT_THRESHOLD = 0.5


def _data_dir(data_dir=None):
    """Resolve the NAB data directory (data/nab under the project root)."""
    if data_dir is not None:
        return data_dir
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "data", "nab")


def available_streams(data_dir=None):
    """Return stream names ('category/file.csv') present on disk.

    Featured streams come first (when present), then everything else
    in alphabetical order.
    """
    base = _data_dir(data_dir)
    names = []
    for category in sorted(os.listdir(base)):
        cat = os.path.join(base, category)
        if not os.path.isdir(cat) or category == "labels":
            continue
        for fname in sorted(os.listdir(cat)):
            if fname.endswith(".csv"):
                names.append(f"{category}/{fname}")
    ordered = [s for s in FEATURED_STREAMS if s in names]
    ordered += [s for s in names if s not in ordered]
    return ordered


def load_named_stream(stream_name, data_dir=None):
    """Load one stream by name -> (timestamps, values) parallel lists."""
    base = _data_dir(data_dir)
    csv_path = os.path.join(base, *stream_name.split("/"))
    return load_stream(csv_path)


def stream_scores(detector, timestamps, values):
    """Feed every point through the detector IN ORDER (no look-ahead).

    ``detector`` may be a Detector class or instance. Returns a list
    of anomaly scores in [0, 1] -- exactly the sequence a live monitor
    would emit, one score per input point.
    """
    det = detector() if isinstance(detector, type) else detector
    if hasattr(det, "reset"):
        det.reset()
    scores = []
    for ts, val in zip(timestamps, values):
        scores.append(det.handle_record(ts, val))
    return scores


def detections(scores, threshold):
    """Boolean list: True where score >= threshold (an alarm is raised)."""
    return [s >= threshold for s in scores]


def window_spans(timestamps, windows):
    """Map anomaly windows to inclusive (left, right) index spans.

    ``windows`` is a list of (start_dt, end_dt) tuples (from
    ``scoring.load_windows``). Self-contained so the dashboard can
    shade true-anomaly regions without reaching into scoring
    internals.
    """
    spans = []
    for w_start, w_end in windows:
        left = right = None
        for i, ts in enumerate(timestamps):
            if left is None and ts >= w_start:
                left = i
            if ts <= w_end:
                right = i
        if left is not None and right is not None and left <= right:
            spans.append((left, right))
    return spans


def frame_metrics(scores, detections_list, spans, position):
    """Summarise the stream AS SEEN UP TO index ``position`` (inclusive).

    Uses only points 0..position -- never the future -- so the numbers
    match what a live operator would have on screen at that moment. A
    window counts as ``hit`` once an alarm has fired inside its
    revealed portion.
    """
    n = len(scores)
    if n == 0:
        return {
            "seen": 0, "current_score": 0.0, "alarms": 0,
            "windows_seen": 0, "windows_hit": 0,
            "windows_missed_or_pending": 0, "in_window": False,
        }
    pos = max(0, min(position, n - 1))
    alarms = sum(1 for i in range(pos + 1) if detections_list[i])
    windows_seen = windows_hit = 0
    in_window = False
    for left, right in spans:
        if left <= pos:
            windows_seen += 1
            hi = min(right, pos)
            if any(detections_list[i] for i in range(left, hi + 1)):
                windows_hit += 1
        if left <= pos <= right:
            in_window = True
    return {
        "seen": pos + 1,
        "current_score": scores[pos],
        "alarms": alarms,
        "windows_seen": windows_seen,
        "windows_hit": windows_hit,
        "windows_missed_or_pending": windows_seen - windows_hit,
        "in_window": in_window,
    }


def main():
    """Streamlit entry point (Streamlit imported lazily)."""
    import time
    import streamlit as st

    st.set_page_config(page_title="NAB Live Anomaly Monitor",
                       layout="wide")
    st.title("NAB - Live Streaming Anomaly Monitor")
    st.caption(
        "A streaming detector scores a Numenta Anomaly Benchmark feed "
        "point-by-point, with no look-ahead - the way a real monitor "
        "sees it. Try lowering the threshold to catch anomalies earlier "
        "and watch the false alarms rise too."
    )

    with st.sidebar:
        st.header("Controls")
        streams = available_streams()
        stream_name = st.selectbox("Data stream", streams, index=0)
        det_name = st.selectbox("Detector", list(DETECTORS.keys()), index=0)
        threshold = st.slider("Alarm threshold", 0.0, 1.0,
                              DEFAULT_THRESHOLD, 0.01)
        speed = st.select_slider(
            "Playback speed (points per frame)",
            options=[1, 5, 10, 25, 50, 100], value=25)
        play = st.button("Play stream")

    timestamps, values = load_named_stream(stream_name)
    scores = stream_scores(DETECTORS[det_name], timestamps, values)
    flags = detections(scores, threshold)
    windows = load_windows().get(stream_name, [])
    spans = window_spans(timestamps, windows)
    n = len(values)

    if n == 0:
        st.warning("Selected stream has no data points.")
        return

    metrics_box = st.empty()
    value_box = st.empty()
    score_box = st.empty()

    def render(pos):
        met = frame_metrics(scores, flags, spans, pos)
        with metrics_box.container():
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Points seen", f"{met['seen']}/{n}")
            c2.metric("Current score", f"{met['current_score']:.3f}")
            c3.metric("Alarms raised", met["alarms"])
            c4.metric("Windows hit",
                      f"{met['windows_hit']}/{met['windows_seen']}")
            if met["in_window"]:
                st.info("Inside a ground-truth anomaly window.")
        upto = pos + 1
        value_box.line_chart({"value": values[:upto]})
        score_box.line_chart({"anomaly_score": scores[:upto],
                              "threshold": [threshold] * upto})

    if play:
        for pos in range(0, n, speed):
            render(pos)
            time.sleep(0.05)
        render(n - 1)
        st.success("Stream complete.")
    else:
        pos = st.slider("Reveal up to point", 0, n - 1, n - 1)
        render(pos)
        st.caption(
            f"{len(spans)} ground-truth anomaly window(s) in this stream; "
            f"{n} points total."
        )


if __name__ == "__main__":
    main()
