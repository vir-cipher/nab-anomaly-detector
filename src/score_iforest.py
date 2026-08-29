"""Score the isolation forest on NAB and build the 5-detector comparison
table (Phase 13, step-010).

Phase 12 (step-008) tabulated the four statistical detectors in
``results/statistical_detectors.json``.  This module scores the
Phase-13 isolation forest with the *same* NAB sweep methodology
(reusing ``run_all_detectors.score_all``) so the numbers are directly
comparable, then merges the two into one leaderboard:

    results/iforest_scores.json   (isolation forest only)
    results/comparison.csv        (all five detectors x three profiles)
    results/comparison.json       (same table, plus the detector list)

Usage:
    python src/score_iforest.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.iforest import IsolationForestDetector
from src.run_all_detectors import (
    format_table,
    score_all,
    write_csv,
)

# The forest keeps its step-009 defaults here; the parameter sweep is
# step-011's job, so step-010 scores the frozen configuration as-is.
IFOREST_DETECTORS = {"iforest": IsolationForestDetector}

STAT_JSON = "statistical_detectors.json"
IFOREST_JSON = "iforest_scores.json"
COMPARISON_CSV = "comparison.csv"
COMPARISON_JSON = "comparison.json"


def _results_dir():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "results")


def score_iforest(streams=None, windows=None, verbose=True):
    """Score the isolation forest under every profile; return row dicts.

    Thin wrapper over ``score_all`` with the isolation-forest registry,
    so the forest is scored through the exact sweep/threshold machinery
    used for the statistical detectors in step-008.
    """
    return score_all(streams=streams, windows=windows,
                     detectors=IFOREST_DETECTORS, verbose=verbose)


def load_statistical_rows(path=None):
    """Load the step-008 statistical-detector rows to compare against."""
    if path is None:
        path = os.path.join(_results_dir(), STAT_JSON)
    with open(path) as f:
        return json.load(f)["rows"]


def build_comparison(iforest_rows, statistical_rows):
    """Merge statistical + isolation-forest rows into one leaderboard.

    Every row carries a ``detector`` field, so the merge is a
    de-duplicated concatenation keyed by (detector, profile): feeding an
    already-merged table back in never double-counts a detector, which
    keeps the daily re-run idempotent (Rule 9).
    """
    merged = {}
    for row in list(statistical_rows) + list(iforest_rows):
        merged[(row["detector"], row["profile"])] = row
    return list(merged.values())


def write_comparison(rows, csv_path=None, json_path=None):
    """Write the merged comparison table to CSV + JSON.

    Reuses ``run_all_detectors.write_csv`` (same columns and sort order:
    by profile, then descending NAB score) so the comparison table is
    laid out identically to the statistical leaderboard.
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

    iforest_rows = score_iforest()
    with open(os.path.join(rdir, IFOREST_JSON), "w") as f:
        json.dump({"detector": "iforest", "rows": iforest_rows}, f, indent=2)

    statistical_rows = load_statistical_rows()
    comparison = build_comparison(iforest_rows, statistical_rows)
    write_comparison(comparison)

    print("\n" + format_table(comparison))
    print(f"\nComparison saved to {os.path.join(rdir, COMPARISON_CSV)}")
