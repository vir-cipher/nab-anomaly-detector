"""Tests for NAB dataset download and structure (step-001).

These tests verify:
- The download script module is importable and has the right constants.
- After download, all 58 CSV files exist across 7 categories.
- Each CSV has the expected two-column format (timestamp, value).
- Anomaly-window labels are present and reference real data files.
- No file is empty or corrupt (header-only).
"""
import csv
import json
import os
import sys

import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.download_nab import (
    CATEGORIES,
    EXPECTED_TOTAL_FILES,
    LABEL_FILES,
    check_integrity,
    get_dataset_summary,
    _data_dir,
    _labels_dir,
)


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

def test_category_count():
    """NAB has exactly 7 categories."""
    assert len(CATEGORIES) == 7


def test_total_file_count():
    """NAB has exactly 58 CSV data files."""
    assert EXPECTED_TOTAL_FILES == 58


def test_label_files_defined():
    """We expect combined_windows.json and combined_labels.json."""
    assert "combined_windows.json" in LABEL_FILES
    assert "combined_labels.json" in LABEL_FILES


# ---------------------------------------------------------------------------
# Data presence (requires download to have run)
# ---------------------------------------------------------------------------

def _data_downloaded():
    ok, _, _ = check_integrity()
    return ok


@pytest.mark.skipif(not _data_downloaded(), reason="NAB data not downloaded yet")
class TestNABDataPresence:
    """Tests that run only when the data has been downloaded."""

    def test_all_files_present(self):
        ok, missing, empty = check_integrity()
        assert ok, f"Missing: {missing}, Empty: {empty}"

    def test_csv_format_two_columns(self):
        """Every CSV must have exactly 2 columns: timestamp, value."""
        data_dir = _data_dir()
        for category, files in CATEGORIES.items():
            for fname in files:
                path = os.path.join(data_dir, category, fname)
                with open(path, "r") as f:
                    reader = csv.reader(f)
                    header = next(reader)
                    assert len(header) == 2, (
                        f"{category}/{fname} has {len(header)} columns, expected 2"
                    )
                    assert header[0] == "timestamp", (
                        f"{category}/{fname} first column is '{header[0]}', expected 'timestamp'"
                    )
                    assert header[1] == "value", (
                        f"{category}/{fname} second column is '{header[1]}', expected 'value'"
                    )

    def test_csv_has_data_rows(self):
        """Every CSV must have at least 100 data rows (NAB streams are 1000+)."""
        data_dir = _data_dir()
        for category, files in CATEGORIES.items():
            for fname in files:
                path = os.path.join(data_dir, category, fname)
                with open(path, "r") as f:
                    row_count = sum(1 for _ in f) - 1  # minus header
                assert row_count >= 100, (
                    f"{category}/{fname} has only {row_count} rows"
                )

    def test_no_empty_values(self):
        """Spot-check: first file in each category should have no empty values."""
        data_dir = _data_dir()
        for category, files in CATEGORIES.items():
            fname = files[0]
            path = os.path.join(data_dir, category, fname)
            with open(path, "r") as f:
                reader = csv.reader(f)
                next(reader)  # skip header
                for i, row in enumerate(reader):
                    assert len(row) == 2 and row[0] and row[1], (
                        f"{category}/{fname} row {i+1} has empty cell"
                    )
                    # value should be parseable as float
                    float(row[1])

    def test_labels_reference_real_files(self):
        """Every key in combined_windows.json must match a real data file."""
        labels_path = os.path.join(_labels_dir(), "combined_windows.json")
        with open(labels_path, "r") as f:
            windows = json.load(f)
        data_dir = _data_dir()
        for key in windows:
            # Keys look like "artificialNoAnomaly/art_daily_no_noise.csv"
            path = os.path.join(data_dir, key)
            assert os.path.exists(path), (
                f"Label key '{key}' does not match a data file"
            )

    def test_labels_window_format(self):
        """Each window in combined_windows.json is a list of [start, end] pairs."""
        labels_path = os.path.join(_labels_dir(), "combined_windows.json")
        with open(labels_path, "r") as f:
            windows = json.load(f)
        for key, window_list in windows.items():
            assert isinstance(window_list, list), f"{key}: windows is not a list"
            for w in window_list:
                assert isinstance(w, list) and len(w) == 2, (
                    f"{key}: window {w} is not a [start, end] pair"
                )

    def test_dataset_summary_totals(self):
        """get_dataset_summary returns correct total file count."""
        summary = get_dataset_summary()
        assert summary["total_files"] == 58
        assert summary["total_bytes"] > 0
        assert len(summary["categories"]) == 7
