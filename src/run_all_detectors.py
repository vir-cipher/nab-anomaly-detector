"""Score ALL Phase-12 statistical detectors on the NAB corpus and tabulate.

Runs each detector once over every stream, then sweep-optimizes the
detection threshold per NAB profile, and writes a consolidated table:

    results/statistical_detectors.csv    (one row per detector x profile)
    results/statistical_detectors.json   (full detail)

Usage:
    python src/run_all_detectors.py
"""

import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_all_streams
from src.detectors import (
    WindowedGaussianDetector,
    EWMADetector,
    ZScoreDetector,
    ThresholdDetector,
)
from src.scoring import load_windows, sweep_optimize

STATISTICAL_DETECTORS = {
    "gaussian": WindowedGaussianDetector,
    "ewma": EWMADetector,
    "zscore": ZScoreDetector,
    "threshold": ThresholdDetector,
}

PROFILES = ("standard", "reward_low_fp", "reward_low_fn")

CSV_COLUMNS = [
    "detector", "profile", "nab_score", "raw_score",
    "threshold", "num_streams", "total_windows",
]


def run_detector_on_streams(det_class, streams):
    """Run one detector over every stream; return {stream: (timestamps, scores)}."""
    detector_results = {}
    for stream_name, (timestamps, values) in streams.items():
        det = det_class()
        scores = []
        for ts, val in zip(timestamps, values):
            scores.append(det.handle_record(ts, val))
        detector_results[stream_name] = (timestamps, scores)
    return detector_results


def score_all(streams=None, windows=None, detectors=None, verbose=True):
    """Score every detector under every profile; return a list of row dicts."""
    if streams is None:
        streams = load_all_streams()
    if windows is None:
        windows = load_windows()
    if detectors is None:
        detectors = STATISTICAL_DETECTORS

    rows = []
    for det_name, det_class in detectors.items():
        if verbose:
            print(f"Running {det_name} on {len(streams)} streams ...")
        detector_results = run_detector_on_streams(det_class, streams)
        for profile in PROFILES:
            r = sweep_optimize(detector_results, windows, profile=profile)
            row = {
                "detector": det_name,
                "profile": profile,
                "nab_score": round(r["best_nab_score"], 4),
                "raw_score": round(r["raw_score"], 4),
                "threshold": round(r["best_threshold"], 10),
                "num_streams": len(detector_results),
                "total_windows": r["total_windows"],
            }
            rows.append(row)
            if verbose:
                print(f"  {profile}: NAB={row['nab_score']:.2f} "
                      f"(threshold={row['threshold']})")
    return rows


def write_csv(rows, path):
    """Write score rows to CSV, sorted by profile then descending NAB score."""
    ordered = sorted(
        rows, key=lambda r: (PROFILES.index(r["profile"]), -r["nab_score"]))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in ordered:
            writer.writerow(row)
    return ordered


def format_table(rows):
    """Plain-text leaderboard (standard profile) for the console/README."""
    standard = sorted(
        (r for r in rows if r["profile"] == "standard"),
        key=lambda r: -r["nab_score"])
    lines = ["detector     NAB(standard)", "-" * 28]
    for r in standard:
        lines.append(f"{r['detector']:<12} {r['nab_score']:>8.2f}")
    return "\n".join(lines)


def _project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    all_rows = score_all()
    root = _project_root()
    csv_path = os.path.join(root, "results", "statistical_detectors.csv")
    json_path = os.path.join(root, "results", "statistical_detectors.json")
    write_csv(all_rows, csv_path)
    with open(json_path, "w") as f:
        json.dump({"detectors": sorted({r["detector"] for r in all_rows}),
                   "rows": all_rows}, f, indent=2)
    print("\n" + format_table(all_rows))
    print(f"\nResults saved to {csv_path}")
