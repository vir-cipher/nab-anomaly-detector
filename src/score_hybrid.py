"""Score the hybrid detector on NAB and rebuild the full comparison
table (Phase 14, step-013).

Step-012 built the hybrid combiner (``src/hybrid.py``) and unit-tested
its arithmetic. Step-013 *scores* it on the NAB corpus with the SAME
sweep/threshold machinery used for every other detector
(``run_all_detectors.score_all``), so its NAB score is directly
comparable, then folds it into the leaderboard alongside the four
statistical detectors (step-008) and the isolation forest (step-010):

    results/hybrid_scores.json    (hybrid only)
    results/comparison.csv        (all six detectors x three profiles)
    results/comparison.json       (same table, plus the detector list)

The full 6-detector table is rebuilt from the canonical per-detector
result files each run, so re-running is idempotent (Rule 9) and never
depends on score_iforest running first.

Usage:
    python src/score_hybrid.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hybrid import HybridDetector
from src.run_all_detectors import (
    format_table,
    score_all,
    write_csv,
)

# The hybrid keeps its step-012 defaults here (WindowedGaussianDetector
# 0.7 + IsolationForestDetector 0.3, weighted_average). Alternate voting
# rules are explored in the write-up, not scored as separate leaderboard
# rows, so step-013 scores the frozen default configuration.
HYBRID_DETECTORS = {"hybrid": HybridDetector}

STAT_JSON = "statistical_detectors.json"
IFOREST_JSON = "iforest_scores.json"
HYBRID_JSON = "hybrid_scores.json"
COMPARISON_CSV = "comparison.csv"
COMPARISON_JSON = "comparison.json"


def _results_dir():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "results")


def score_hybrid(streams=None, windows=None, verbose=True):
    """Score the hybrid detector under every profile; return row dicts.

    Thin wrapper over ``score_all`` with the hybrid registry, so the
    combiner is scored through the exact sweep/threshold pipeline used
    for the statistical detectors and the isolation forest.
    """
    return score_all(streams=streams, windows=windows,
                     detectors=HYBRID_DETECTORS, verbose=verbose)


def load_rows(path):
    """Load the ``rows`` list from a results JSON file."""
    with open(path) as f:
        return json.load(f)["rows"]


def load_prior_rows(results_dir=None):
    """Load every previously-scored detector row (statistical + iforest).

    Rebuilds the pre-hybrid leaderboard from the two canonical result
    files so the merge below is self-contained: it does not rely on an
    already-merged ``comparison.json`` (which score_iforest may or may
    not have refreshed this run).
    """
    if results_dir is None:
        results_dir = _results_dir()
    prior = []
    prior.extend(load_rows(os.path.join(results_dir, STAT_JSON)))
    prior.extend(load_rows(os.path.join(results_dir, IFOREST_JSON)))
    return prior


def build_comparison(hybrid_rows, prior_rows):
    """Merge prior detectors + hybrid rows into one leaderboard.

    Every row carries a ``detector`` field, so the merge is a
    de-duplicated concatenation keyed by (detector, profile): feeding an
    already-merged table back in never double-counts a detector, which
    keeps the daily re-run idempotent (Rule 9).
    """
    merged = {}
    for row in list(prior_rows) + list(hybrid_rows):
        merged[(row["detector"], row["profile"])] = row
    return list(merged.values())


def write_comparison(rows, csv_path=None, json_path=None):
    """Write the merged comparison table to CSV + JSON.

    Reuses ``run_all_detectors.write_csv`` (same columns and sort order:
    by profile, then descending NAB score) so the comparison table is
    laid out identically to every earlier leaderboard.
    """
    rdir = _results_dir()
    if csv_path is None:
        csv_path = os.path.join(rdir, COMPARISON_CSV)
    if json_path is None:
        json_path = os.path.join(rdir, COMPARISON_JSON)
    ordered = write_csv(rows, csv_path)
    with open(json_path, "w") as f:
        json.dump({"detectors": sorted({r["detector"] for r in rows}),
                   "rows": ordered}, f, indent=2)
    return ordered


if __name__ == "__main__":
    rdir = _results_dir()
    os.makedirs(rdir, exist_ok=True)

    hybrid_rows = score_hybrid()
    with open(os.path.join(rdir, HYBRID_JSON), "w") as f:
        json.dump({"detector": "hybrid", "rows": hybrid_rows}, f, indent=2)

    prior_rows = load_prior_rows()
    comparison = build_comparison(hybrid_rows, prior_rows)
    write_comparison(comparison)

    print("\n" + format_table(comparison))
    print(f"\nComparison saved to {os.path.join(rdir, COMPARISON_CSV)}")
