"""Run a detector on the full NAB corpus and save scored results.

Usage:
    python src/run_baseline.py                # null detector
    python src/run_baseline.py --detector null --threshold 0.5
"""

import argparse
import json
import os
import sys

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_all_streams
from src.detectors import NullDetector
from src.scoring import load_windows, score_corpus


DETECTORS = {
    "null": NullDetector,
}


def run(detector_name="null", threshold=0.5):
    """Run a detector on all NAB streams and return scored results."""
    det_class = DETECTORS[detector_name]
    print(f"Loading NAB streams ...")
    streams = load_all_streams()
    print(f"  Loaded {len(streams)} streams.")

    windows = load_windows()
    print(f"  Loaded windows for {len(windows)} streams.")

    # Run detector on each stream
    detector_results = {}
    for stream_name, (timestamps, values) in streams.items():
        det = det_class()
        scores = []
        for ts, val in zip(timestamps, values):
            scores.append(det.handle_record(ts, val))
        detector_results[stream_name] = (timestamps, scores)

    # Score across all three profiles
    all_scores = {}
    for profile in ("standard", "reward_low_fp", "reward_low_fn"):
        result = score_corpus(
            detector_results, windows, threshold, profile=profile)
        all_scores[profile] = {
            "nab_score": round(result["nab_score"], 4),
            "raw_score": round(result["raw_score"], 4),
            "num_streams": result["num_streams"],
            "total_windows": result["total_windows"],
        }
        print(f"  {profile}: NAB score = {all_scores[profile]['nab_score']}")

    return {
        "detector": detector_name,
        "threshold": threshold,
        "profiles": all_scores,
    }


def _project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a detector on the NAB corpus.")
    parser.add_argument(
        "--detector", default="null", choices=list(DETECTORS.keys()),
        help="Detector to run (default: null)")
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Detection threshold (default: 0.5)")
    args = parser.parse_args()

    result = run(args.detector, args.threshold)

    out_path = os.path.join(
        _project_root(), "results",
        f"{args.detector}_baseline.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {out_path}")
