"""Parameter sensitivity analysis for the isolation forest (Phase 13, step-011).

A one-factor-at-a-time (OFAT) sweep around the step-009 baseline
configuration, scored through the SAME NAB machinery
(``run_all_detectors.score_all``) that scored every other detector, so
every number produced here is directly comparable to
``results/comparison.csv``.

Two parameters are swept, holding all others at their baseline value:
  - ``n_trees``     (ensemble size):        32, 64*, 128
  - ``sample_size`` (sub-sample size psi):  64, 128*, 256
  (* = baseline / step-009 default)

Why OFAT: changing one knob at a time keeps every off-baseline config
directly attributable to a single parameter, so the resulting table
reads as a clean "this knob moves the score by this much" story rather
than a tangle of interacting changes.

Outputs (written by ``python src/param_sensitivity.py``):
  results/iforest_sensitivity.csv    one row per (config x profile)
  results/iforest_sensitivity.json   same rows + the config registry
"""

import csv
import json
import os
import sys
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.iforest import IsolationForestDetector
from src.run_all_detectors import PROFILES, score_all

# Baseline == IsolationForestDetector constructor defaults (frozen step-009).
BASELINE = {
    "shingle_size": 4,
    "train_size": 256,
    "retrain_interval": 256,
    "n_trees": 64,
    "sample_size": 128,
    "seed": 42,
}

# Parameters we sweep, and the values tried for each (baseline value * kept
# in the list so the OFAT design is explicit; it is de-duplicated below).
SWEPT_PARAMS = ("n_trees", "sample_size")
SWEEP_VALUES = {
    "n_trees": (32, 64, 128),
    "sample_size": (64, 128, 256),
}

SENS_COLUMNS = [
    "config", "swept_param", "swept_value",
    "n_trees", "sample_size", "shingle_size",
    "train_size", "retrain_interval",
    "profile", "nab_score", "raw_score", "threshold",
    "num_streams", "total_windows",
]

CSV_NAME = "iforest_sensitivity.csv"
JSON_NAME = "iforest_sensitivity.json"


def _results_dir():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "results")


def build_configs():
    """Return the OFAT config list around ``BASELINE``.

    The baseline appears exactly once; each off-baseline value of each
    swept parameter adds one more config.  With the values above this
    yields 5 configs: baseline, n_trees=32, n_trees=128,
    sample_size=64, sample_size=256.
    """
    configs = [{
        "label": "baseline",
        "swept_param": "baseline",
        "swept_value": None,
        "params": dict(BASELINE),
    }]
    for pname in SWEPT_PARAMS:
        for val in SWEEP_VALUES[pname]:
            if val == BASELINE[pname]:
                continue  # baseline already carries this value
            params = dict(BASELINE)
            params[pname] = val
            configs.append({
                "label": f"{pname}={val}",
                "swept_param": pname,
                "swept_value": val,
                "params": params,
            })
    return configs


def _detector_factories(configs):
    """Map each config label to a zero-arg factory ``score_all`` can call.

    ``run_all_detectors.run_detector_on_streams`` instantiates each
    detector with ``det_class()`` and no arguments, so a ``partial`` that
    pre-binds the config's parameters is exactly the shape it expects.
    """
    return {c["label"]: partial(IsolationForestDetector, **c["params"])
            for c in configs}


def run_sweep(streams=None, windows=None, configs=None, verbose=True):
    """Score every config under every NAB profile; return annotated rows.

    Thin wrapper over ``score_all`` (the identical sweep/threshold engine
    used for the statistical detectors and the step-010 forest), then
    each returned row is tagged with the parameter values of its config
    so the sensitivity table is self-describing.
    """
    if configs is None:
        configs = build_configs()
    detectors = _detector_factories(configs)
    rows = score_all(streams=streams, windows=windows,
                     detectors=detectors, verbose=verbose)

    by_label = {c["label"]: c for c in configs}
    out = []
    for r in rows:
        c = by_label[r["detector"]]
        p = c["params"]
        out.append({
            "config": r["detector"],
            "swept_param": c["swept_param"],
            "swept_value": c["swept_value"],
            "n_trees": p["n_trees"],
            "sample_size": p["sample_size"],
            "shingle_size": p["shingle_size"],
            "train_size": p["train_size"],
            "retrain_interval": p["retrain_interval"],
            "profile": r["profile"],
            "nab_score": r["nab_score"],
            "raw_score": r["raw_score"],
            "threshold": r["threshold"],
            "num_streams": r["num_streams"],
            "total_windows": r["total_windows"],
        })
    return out


def _sort_key(row):
    """Order rows by profile, then swept parameter, then swept value.

    Baseline (swept_value ``None``) sorts before any numeric value so
    each profile block opens with the reference configuration.
    """
    profile_order = {"standard": 0, "reward_low_fp": 1, "reward_low_fn": 2}
    val = row["swept_value"]
    val_key = (0, 0) if val is None else (1, val)
    return (profile_order.get(row["profile"], 9), row["swept_param"], val_key)


def write_sensitivity(rows, csv_path=None, json_path=None, configs=None):
    """Write the sensitivity table to CSV + JSON (sorted, deterministic)."""
    rdir = _results_dir()
    if csv_path is None:
        csv_path = os.path.join(rdir, CSV_NAME)
    if json_path is None:
        json_path = os.path.join(rdir, JSON_NAME)
    ordered = sorted(rows, key=_sort_key)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SENS_COLUMNS)
        writer.writeheader()
        for row in ordered:
            writer.writerow(row)
    if configs is None:
        configs = build_configs()
    with open(json_path, "w") as f:
        json.dump({
            "baseline": BASELINE,
            "swept_params": list(SWEPT_PARAMS),
            "sweep_values": SWEEP_VALUES,
            "configs": [c["label"] for c in configs],
            "rows": ordered,
        }, f, indent=2)
    return ordered


def summarize(rows, profile="standard"):
    """Per-profile sensitivity summary: baseline score, per-config scores,
    and the total spread (max - min NAB) the swept knobs produce.

    A small spread means the forest is robust to these parameters on NAB;
    a large spread means tuning them actually matters.
    """
    prof_rows = [r for r in rows if r["profile"] == profile]
    scores = {r["config"]: r["nab_score"] for r in prof_rows}
    baseline = scores.get("baseline")
    nab_values = list(scores.values())
    spread = round(max(nab_values) - min(nab_values), 4) if nab_values else 0.0
    return {
        "profile": profile,
        "baseline_nab": baseline,
        "scores": scores,
        "spread": spread,
        "best_config": max(scores, key=scores.get) if scores else None,
        "worst_config": min(scores, key=scores.get) if scores else None,
    }


def format_summary(rows):
    """Plain-text sensitivity report for the console/decisions.log."""
    lines = []
    for profile in ("standard", "reward_low_fp", "reward_low_fn"):
        s = summarize(rows, profile)
        lines.append(f"[{profile}] baseline NAB={s['baseline_nab']}  "
                     f"spread={s['spread']}  best={s['best_config']}")
        for cfg, nab in s["scores"].items():
            lines.append(f"    {cfg:<16} {nab:>8.4f}")
    return "\n".join(lines)


if __name__ == "__main__":
    rdir = _results_dir()
    os.makedirs(rdir, exist_ok=True)
    configs = build_configs()
    print(f"Sweeping {len(configs)} isolation-forest configs "
          f"x {len(PROFILES)} profiles ...")
    rows = run_sweep(configs=configs, verbose=True)
    write_sensitivity(rows, configs=configs)
    print("\n" + format_summary(rows))
    print(f"\nSaved to {os.path.join(rdir, CSV_NAME)}")
