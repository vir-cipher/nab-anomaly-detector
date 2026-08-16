"""Run a detector on the full NAB corpus, optionally optimise threshold.

Usage:
    python src/run_baseline.py                          # null detector
    python src/run_baseline.py --detector gaussian      # windowed gaussian
    python src/run_baseline.py --detector gaussian --optimize  # sweep thresholds
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_all_streams
from src.detectors import NullDetector, WindowedGaussianDetector
from src.scoring import load_windows, score_corpus, sweep_optimize


DETECTORS = {
    "null": NullDetector,
    "gaussian": WindowedGaussianDetector,
}


def _run_detector(det_class, streams):
    """Run a detector on all streams, return raw results."""
    detector_results = {}
    for stream_name, (timestamps, values) in streams.items():
        det = det_class()
        scores = []
        for ts, val in zip(timestamps, values):
            scores.append(det.handle_record(ts, val))
        detector_results[stream_name] = (timestamps, scores)
    return detector_results


def run(detector_name="null", threshold=0.5, do_optimize=False):
    """Run a detector on all NAB streams and return scored results."""
    det_class = DETECTORS[detector_name]
    print(f"Loading NAB streams ...")
    streams = load_all_streams()
    print(f"  Loaded {len(streams)} streams.")

    windows = load_windows()
    print(f"  Loaded windows for {len(windows)} streams.")

    print(f"Running {detector_name} detector ...")
    detector_results = _run_detector(det_class, streams)
    print(f"  Detection complete on {len(detector_results)} streams.")

    all_scores = {}
    for profile in ("standard", "reward_low_fp", "reward_low_fn"):
        if do_optimize:
            r = sweep_optimize(
                detector_results, windows, profile=profile)
            all_scores[profile] = {
                "nab_score": round(r["best_nab_score"], 4),
                "raw_score": round(r["raw_score"], 4),
                "threshold": round(r["best_threshold"], 10),
                "num_streams": len(detector_results),
                "total_windows": r["total_windows"],
            }
            print(f"  {profile}: NAB={r['best_nab_score']:.2f} "
                  f"(threshold={r['best_threshold']:.8f})")
        else:
            result = score_corpus(
                detector_results, windows, threshold, profile=profile)
            all_scores[profile] = {
                "nab_score": round(result["nab_score"], 4),
                "raw_score": round(result["raw_score"], 4),
                "threshold": threshold,
                "num_streams": result["num_streams"],
                "total_windows": result["total_windows"],
            }
            print(f"  {profile}: NAB={result['nab_score']:.2f}")

    return {
        "detector": detector_name,
        "optimized": do_optimize,
        "profiles": all_scores,
    }


def _project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a detector on the NAB corpus.")
    parser.add_argument(
        "--detector", default="null", choices=list(DETECTORS.keys()))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--optimize", action="store_true",
                        help="Sweep thresholds for best score")
    args = parser.parse_args()

    result = run(args.detector, args.threshold, args.optimize)

    out_path = os.path.join(
        _project_root(), "results",
        f"{args.detector}_baseline.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {out_path}")
