"""Load NAB CSV streams into the format expected by the scorer.

Each NAB CSV has two columns: timestamp, value.
The loader parses them into (list[datetime], list[float]) tuples
that ``score_corpus`` consumes directly.

Usage:
    from src.data_loader import load_all_streams
    streams = load_all_streams()
    # streams["realKnownCause/nyc_taxi.csv"] -> (timestamps, values)
"""

import csv
import os
from datetime import datetime


def _project_root():
    """Return the project root (parent of src/)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _parse_ts(ts_str):
    """Parse a NAB timestamp string."""
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(ts_str.strip(), fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse timestamp: {ts_str!r}")


def load_stream(csv_path):
    """Load a single NAB CSV file.

    Returns:
        (timestamps, values) — two parallel lists.
    """
    timestamps = []
    values = []
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            timestamps.append(_parse_ts(row[0]))
            values.append(float(row[1]))
    return timestamps, values


def load_all_streams(data_dir=None):
    """Load every NAB CSV into a dict keyed by stream name.

    Stream names match NAB convention:
        "category/filename.csv"
    e.g. "realKnownCause/nyc_taxi.csv"

    Returns:
        dict: stream_name -> (timestamps, values)
    """
    if data_dir is None:
        data_dir = os.path.join(_project_root(), "data", "nab")

    streams = {}
    for category in sorted(os.listdir(data_dir)):
        cat_path = os.path.join(data_dir, category)
        if not os.path.isdir(cat_path) or category == "labels":
            continue
        for fname in sorted(os.listdir(cat_path)):
            if not fname.endswith(".csv"):
                continue
            stream_name = f"{category}/{fname}"
            csv_path = os.path.join(cat_path, fname)
            streams[stream_name] = load_stream(csv_path)
    return streams
