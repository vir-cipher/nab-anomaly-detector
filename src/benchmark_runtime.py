"""Runtime benchmarking for the NAB detectors (Phase 14, step-014).

Every earlier step asked "how ACCURATE is each detector?" (its NAB
score). This step measures the other half of the project's headline
claim -- "...at 10x speed." It times the WALL-CLOCK cost of streaming
each of the six leaderboard detectors over the whole NAB corpus, one
data point at a time, exactly the way a live system would run them.

What is timed:
  - only the ``handle_record`` loop (one call per data point);
  - a fresh detector per stream (construction and data loading are
    excluded from the clock);
  - the BEST of ``repeats`` passes per stream, which trims the OS
    scheduling noise that inflates a single cold pass (standard
    micro-benchmark practice).

What is reported (results/runtime_benchmark.csv + .json):
  - points_per_sec       -- throughput, higher = faster;
  - us_per_point         -- average microseconds per data point;
  - mean_ms_per_stream   -- average wall time to finish one stream;
  - speedup_vs_iforest   -- each detector's throughput divided by the
    isolation forest's. The forest is the pure-Python ML detector and
    the slowest by construction, so this ratio is the portable,
    machine-independent way to state the "Nx speed" finding.

Absolute times depend on the CPU; the cross-detector RATIO travels.
Numbers are measured here, never asserted -- re-running rewrites the
table (Rule 9).

Usage:
    python src/benchmark_runtime.py            # 1 pass per stream
    python src/benchmark_runtime.py --repeats 3
"""

import argparse
import csv
import json
import os
import platform
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_all_streams
from src.detectors import (
    WindowedGaussianDetector,
    EWMADetector,
    ZScoreDetector,
    ThresholdDetector,
)
from src.iforest import IsolationForestDetector
from src.hybrid import HybridDetector

# The six leaderboard detectors, keyed exactly as results/comparison.csv
# so the speed table lines up row-for-row with the accuracy table.
BENCHMARK_DETECTORS = {
    "gaussian": WindowedGaussianDetector,
    "ewma": EWMADetector,
    "zscore": ZScoreDetector,
    "threshold": ThresholdDetector,
    "iforest": IsolationForestDetector,
    "hybrid": HybridDetector,
}

# Throughput anchor for the speed-up column (the slow ML detector).
REFERENCE_DETECTOR = "iforest"

CSV_COLUMNS = [
    "detector", "n_streams", "total_points", "total_seconds",
    "points_per_sec", "us_per_point", "mean_ms_per_stream",
    "speedup_vs_iforest",
]


def time_detector_on_stream(factory, timestamps, values, repeats=1):
    """Best-of-``repeats`` wall-clock seconds to stream one series.

    A fresh detector is built for every pass so no state leaks between
    passes. Only the per-point ``handle_record`` loop is on the clock.
    """
    best = None
    for _ in range(max(1, repeats)):
        det = factory()
        start = time.perf_counter()
        for ts, val in zip(timestamps, values):
            det.handle_record(ts, val)
        elapsed = time.perf_counter() - start
        if best is None or elapsed < best:
            best = elapsed
    return best


def benchmark_detector(factory, streams, repeats=1):
    """Time one detector across every stream; return per-stream + totals."""
    per_stream = {}
    total_points = 0
    total_seconds = 0.0
    for name, (timestamps, values) in streams.items():
        secs = time_detector_on_stream(factory, timestamps, values,
                                       repeats=repeats)
        n = len(values)
        per_stream[name] = {
            "n_points": n,
            "seconds": secs,
            "points_per_sec": (n / secs) if secs > 0 else float("inf"),
        }
        total_points += n
        total_seconds += secs
    pps = (total_points / total_seconds) if total_seconds > 0 else float("inf")
    return {
        "n_streams": len(streams),
        "total_points": total_points,
        "total_seconds": total_seconds,
        "points_per_sec": pps,
        "us_per_point": (total_seconds / total_points * 1e6)
        if total_points else 0.0,
        "mean_ms_per_stream": (total_seconds / len(streams) * 1e3)
        if streams else 0.0,
        "per_stream": per_stream,
    }


def benchmark_all(streams=None, detectors=None, repeats=1,
                  reference=REFERENCE_DETECTOR, verbose=True):
    """Benchmark every detector; return (rows, detail).

    ``rows`` is the flat table written to CSV/JSON (sorted fastest
    first); ``detail`` keeps the per-stream breakdown for the JSON
    sidecar. ``speedup_vs_iforest`` is each detector's throughput over
    the reference detector's; the reference's own value is 1.0.
    """
    if streams is None:
        streams = load_all_streams()
    if detectors is None:
        detectors = BENCHMARK_DETECTORS

    detail = {}
    for name, factory in detectors.items():
        if verbose:
            print(f"Benchmarking {name} on {len(streams)} streams "
                  f"(repeats={repeats}) ...")
        d = benchmark_detector(factory, streams, repeats=repeats)
        detail[name] = d
        if verbose:
            print(f"  {d['total_points']} pts in {d['total_seconds']:.3f}s "
                  f"-> {d['points_per_sec']:.0f} pts/s "
                  f"({d['us_per_point']:.2f} us/pt)")

    ref_pps = detail.get(reference, {}).get("points_per_sec")
    rows = []
    for name, d in detail.items():
        speedup = ""
        if ref_pps and ref_pps > 0 and d["points_per_sec"] != float("inf"):
            speedup = round(d["points_per_sec"] / ref_pps, 3)
        rows.append({
            "detector": name,
            "n_streams": d["n_streams"],
            "total_points": d["total_points"],
            "total_seconds": round(d["total_seconds"], 6),
            "points_per_sec": round(d["points_per_sec"], 2),
            "us_per_point": round(d["us_per_point"], 4),
            "mean_ms_per_stream": round(d["mean_ms_per_stream"], 4),
            "speedup_vs_iforest": speedup,
        })
    rows.sort(key=lambda r: -r["points_per_sec"])
    return rows, detail


def environment_meta(repeats):
    """Reproducibility metadata for the JSON sidecar."""
    return {
        "generated_by": "src/benchmark_runtime.py (step-014)",
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "repeats": repeats,
        "reference_detector": REFERENCE_DETECTOR,
        "timed": "handle_record loop only; best-of-repeats per stream",
        "note": "Absolute times are machine-specific; speedup_vs_iforest "
                "is the portable finding.",
    }


def _results_dir():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "results")


def write_speed_table(rows, detail=None, csv_path=None, json_path=None,
                      meta=None):
    """Write the speed table to CSV + JSON (idempotent overwrite)."""
    rdir = _results_dir()
    if csv_path is None:
        csv_path = os.path.join(rdir, "runtime_benchmark.csv")
    if json_path is None:
        json_path = os.path.join(rdir, "runtime_benchmark.json")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    payload = {"meta": meta or {}, "rows": rows}
    if detail is not None:
        payload["per_stream"] = detail
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    return csv_path, json_path


def format_speed_table(rows):
    """Plain-text speed leaderboard (fastest first) for console/README."""
    lines = [
        f"{'detector':<12}{'pts/s':>14}{'us/pt':>10}{'x vs iforest':>14}",
        "-" * 50,
    ]
    for r in rows:
        speed = r["speedup_vs_iforest"]
        speed_s = f"{speed:.2f}" if isinstance(speed, (int, float)) else "-"
        lines.append(
            f"{r['detector']:<12}{r['points_per_sec']:>14,.0f}"
            f"{r['us_per_point']:>10.2f}{speed_s:>14}")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NAB detector runtime benchmark")
    parser.add_argument("--repeats", type=int, default=1,
                        help="passes per stream; the best time is kept")
    args = parser.parse_args()

    rows, detail = benchmark_all(repeats=args.repeats)
    meta = environment_meta(args.repeats)
    csv_path, json_path = write_speed_table(rows, detail=detail, meta=meta)
    print("\n" + format_speed_table(rows))
    print(f"\nSpeed table saved to {csv_path}")
