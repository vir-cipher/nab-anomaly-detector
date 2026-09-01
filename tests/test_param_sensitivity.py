"""Tests for src/param_sensitivity.py (step-011: isolation-forest
parameter sensitivity analysis).

Small synthetic streams keep the suite fast; the full 58-stream sweep
runs offline via ``python src/param_sensitivity.py``.
"""
import csv
import json
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.iforest import IsolationForestDetector
from src.run_all_detectors import PROFILES
from src.param_sensitivity import (
    BASELINE,
    SENS_COLUMNS,
    build_configs,
    run_sweep,
    summarize,
    write_sensitivity,
)


def _make_timestamps(n, start="2014-01-01 00:00:00", freq_minutes=5):
    t0 = datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
    return [t0 + timedelta(minutes=i * freq_minutes) for i in range(n)]


def _synthetic_streams():
    """A gently varying stream with a late spike burst plus a flat one --
    same shape used by the step-010 forest tests."""
    n = 320
    ts = _make_timestamps(n)
    spiky = [10.0 + (i % 5) * 0.5 for i in range(n)]
    for i in range(280, 292):
        spiky[i] = 100.0
    flat = [5.0] * n
    return {
        "synthetic/spiky.csv": (ts, spiky),
        "synthetic/flat.csv": (ts, flat),
    }


def _synthetic_windows():
    ts = _make_timestamps(320)
    return {
        "synthetic/spiky.csv": [(ts[280], ts[299])],
        "synthetic/flat.csv": [],
    }


class TestBuildConfigs:
    def test_baseline_plus_offbaseline_values(self):
        labels = [c["label"] for c in build_configs()]
        assert labels == [
            "baseline", "n_trees=32", "n_trees=128",
            "sample_size=64", "sample_size=256",
        ]

    def test_at_least_three_configs(self):
        # Gate for step-011: >=3 parameter configs tested.
        assert len(build_configs()) >= 3

    def test_baseline_matches_detector_defaults(self):
        d = IsolationForestDetector()
        assert BASELINE == {
            "shingle_size": d.shingle_size,
            "train_size": d.train_size,
            "retrain_interval": d.retrain_interval,
            "n_trees": d.n_trees,
            "sample_size": d.sample_size,
            "seed": d.seed,
        }

    def test_offbaseline_configs_change_exactly_one_param(self):
        for c in build_configs():
            if c["label"] == "baseline":
                continue
            diffs = [k for k in BASELINE if c["params"][k] != BASELINE[k]]
            assert diffs == [c["swept_param"]]
            assert c["params"][c["swept_param"]] == c["swept_value"]


class TestRunSweep:
    def _rows(self):
        if not hasattr(self, "_cached"):
            self._cached = run_sweep(
                streams=_synthetic_streams(),
                windows=_synthetic_windows(),
                verbose=False)
        return self._cached

    def test_row_count_is_configs_times_profiles(self):
        rows = self._rows()
        assert len(rows) == len(build_configs()) * len(PROFILES)

    def test_every_row_has_full_schema(self):
        for row in self._rows():
            assert set(row) == set(SENS_COLUMNS)

    def test_scores_bounded_and_two_streams(self):
        for row in self._rows():
            assert 0.0 <= row["nab_score"] <= 100.0
            assert row["num_streams"] == 2

    def test_each_config_has_all_profiles(self):
        rows = self._rows()
        for c in build_configs():
            profs = {r["profile"] for r in rows if r["config"] == c["label"]}
            assert profs == set(PROFILES)

    def test_deterministic_across_reruns(self):
        again = run_sweep(streams=_synthetic_streams(),
                          windows=_synthetic_windows(), verbose=False)
        key = lambda rs: sorted(
            (r["config"], r["profile"], r["nab_score"]) for r in rs)
        assert key(again) == key(self._rows())

    def test_baseline_row_uses_default_params(self):
        base = [r for r in self._rows() if r["config"] == "baseline"]
        assert base and all(r["n_trees"] == 64 and r["sample_size"] == 128
                            for r in base)


class TestSummarize:
    def test_summary_reports_baseline_and_spread(self):
        rows = run_sweep(streams=_synthetic_streams(),
                         windows=_synthetic_windows(), verbose=False)
        s = summarize(rows, profile="standard")
        assert s["profile"] == "standard"
        assert s["baseline_nab"] is not None
        assert s["spread"] >= 0.0
        assert set(s["scores"]) == {c["label"] for c in build_configs()}
        assert s["best_config"] in s["scores"]


class TestWriteSensitivity:
    def test_writes_csv_and_json_roundtrip(self, tmp_path):
        rows = run_sweep(streams=_synthetic_streams(),
                         windows=_synthetic_windows(), verbose=False)
        csv_path = str(tmp_path / "sens.csv")
        json_path = str(tmp_path / "sens.json")
        write_sensitivity(rows, csv_path=csv_path, json_path=json_path)
        with open(csv_path, newline="") as f:
            read_back = list(csv.DictReader(f))
        assert len(read_back) == len(build_configs()) * len(PROFILES)
        assert set(read_back[0]) == set(SENS_COLUMNS)
        with open(json_path) as f:
            data = json.load(f)
        assert data["baseline"]["n_trees"] == 64
        assert "n_trees" in data["swept_params"]
        assert len(data["rows"]) == len(read_back)

    def test_baseline_sorts_first_in_each_profile_block(self, tmp_path):
        rows = run_sweep(streams=_synthetic_streams(),
                         windows=_synthetic_windows(), verbose=False)
        csv_path = str(tmp_path / "sens.csv")
        ordered = write_sensitivity(
            rows, csv_path=csv_path,
            json_path=str(tmp_path / "sens.json"))
        # First row overall must be the standard-profile baseline.
        assert ordered[0]["profile"] == "standard"
        assert ordered[0]["config"] == "baseline"
