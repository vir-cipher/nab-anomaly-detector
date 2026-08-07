"""Download the Numenta Anomaly Benchmark (NAB) dataset from GitHub.

Downloads 58 CSV time-series files across 7 categories, plus the
ground-truth anomaly window labels. Saves to data/nab/.

Usage:
    python src/download_nab.py            # download everything
    python src/download_nab.py --check    # verify files without downloading
"""
import argparse
import json
import os
import sys
import urllib.request
import urllib.error

# NAB repo raw-content base URL
_RAW = "https://raw.githubusercontent.com/numenta/NAB/master"

# Every category and its CSV files (frozen from NAB repo, verified 2026-08-07)
CATEGORIES = {
    "artificialNoAnomaly": [
        "art_daily_no_noise.csv",
        "art_daily_perfect_square_wave.csv",
        "art_daily_small_noise.csv",
        "art_flatline.csv",
        "art_noisy.csv",
    ],
    "artificialWithAnomaly": [
        "art_daily_flatmiddle.csv",
        "art_daily_jumpsdown.csv",
        "art_daily_jumpsup.csv",
        "art_daily_nojump.csv",
        "art_increase_spike_density.csv",
        "art_load_balancer_spikes.csv",
    ],
    "realAWSCloudwatch": [
        "ec2_cpu_utilization_24ae8d.csv",
        "ec2_cpu_utilization_53ea38.csv",
        "ec2_cpu_utilization_5f5533.csv",
        "ec2_cpu_utilization_77c1ca.csv",
        "ec2_cpu_utilization_825cc2.csv",
        "ec2_cpu_utilization_ac20cd.csv",
        "ec2_cpu_utilization_c6585a.csv",
        "ec2_cpu_utilization_fe7f93.csv",
        "ec2_disk_write_bytes_1ef3de.csv",
        "ec2_disk_write_bytes_c0d644.csv",
        "ec2_network_in_257a54.csv",
        "ec2_network_in_5abac7.csv",
        "elb_request_count_8c0756.csv",
        "grok_asg_anomaly.csv",
        "iio_us-east-1_i-a2eb1cd9_NetworkIn.csv",
        "rds_cpu_utilization_cc0c53.csv",
        "rds_cpu_utilization_e47b3b.csv",
    ],
    "realAdExchange": [
        "exchange-2_cpc_results.csv",
        "exchange-2_cpm_results.csv",
        "exchange-3_cpc_results.csv",
        "exchange-3_cpm_results.csv",
        "exchange-4_cpc_results.csv",
        "exchange-4_cpm_results.csv",
    ],
    "realKnownCause": [
        "ambient_temperature_system_failure.csv",
        "cpu_utilization_asg_misconfiguration.csv",
        "ec2_request_latency_system_failure.csv",
        "machine_temperature_system_failure.csv",
        "nyc_taxi.csv",
        "rogue_agent_key_hold.csv",
        "rogue_agent_key_updown.csv",
    ],
    "realTraffic": [
        "TravelTime_387.csv",
        "TravelTime_451.csv",
        "occupancy_6005.csv",
        "occupancy_t4013.csv",
        "speed_6005.csv",
        "speed_7578.csv",
        "speed_t4013.csv",
    ],
    "realTweets": [
        "Twitter_volume_AAPL.csv",
        "Twitter_volume_AMZN.csv",
        "Twitter_volume_CRM.csv",
        "Twitter_volume_CVS.csv",
        "Twitter_volume_FB.csv",
        "Twitter_volume_GOOG.csv",
        "Twitter_volume_IBM.csv",
        "Twitter_volume_KO.csv",
        "Twitter_volume_PFE.csv",
        "Twitter_volume_UPS.csv",
    ],
}

LABEL_FILES = ["combined_windows.json", "combined_labels.json"]

EXPECTED_TOTAL_FILES = sum(len(v) for v in CATEGORIES.values())  # 58


def _project_root():
    """Return the project root (parent of src/)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _data_dir():
    return os.path.join(_project_root(), "data", "nab")


def _labels_dir():
    return os.path.join(_data_dir(), "labels")


def download_file(url, dest):
    """Download a single file. Returns True on success."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        urllib.request.urlretrieve(url, dest)
        return True
    except urllib.error.URLError as e:
        print(f"  FAILED: {url} -> {e}", file=sys.stderr)
        return False


def download_all(force=False):
    """Download all NAB data and labels. Skip files that already exist unless force=True."""
    data_dir = _data_dir()
    downloaded, skipped, failed = 0, 0, 0

    # Data CSVs
    for category, files in CATEGORIES.items():
        cat_dir = os.path.join(data_dir, category)
        for fname in files:
            dest = os.path.join(cat_dir, fname)
            if os.path.exists(dest) and not force:
                skipped += 1
                continue
            url = f"{_RAW}/data/{category}/{fname}"
            print(f"  {category}/{fname} ... ", end="", flush=True)
            if download_file(url, dest):
                downloaded += 1
                print("OK")
            else:
                failed += 1

    # Labels
    labels_dir = _labels_dir()
    for fname in LABEL_FILES:
        dest = os.path.join(labels_dir, fname)
        if os.path.exists(dest) and not force:
            skipped += 1
            continue
        url = f"{_RAW}/labels/{fname}"
        print(f"  labels/{fname} ... ", end="", flush=True)
        if download_file(url, dest):
            downloaded += 1
            print("OK")
        else:
            failed += 1

    print(f"\nDone: {downloaded} downloaded, {skipped} skipped (exist), {failed} failed.")
    return failed == 0


def check_integrity():
    """Verify all expected files are present and non-empty. Returns (ok, missing, empty)."""
    data_dir = _data_dir()
    missing, empty = [], []

    for category, files in CATEGORIES.items():
        for fname in files:
            path = os.path.join(data_dir, category, fname)
            if not os.path.exists(path):
                missing.append(f"{category}/{fname}")
            elif os.path.getsize(path) == 0:
                empty.append(f"{category}/{fname}")

    for fname in LABEL_FILES:
        path = os.path.join(_labels_dir(), fname)
        if not os.path.exists(path):
            missing.append(f"labels/{fname}")
        elif os.path.getsize(path) == 0:
            empty.append(f"labels/{fname}")

    return (len(missing) == 0 and len(empty) == 0), missing, empty


def get_dataset_summary():
    """Return a dict summarising the downloaded dataset."""
    data_dir = _data_dir()
    summary = {"categories": {}, "total_files": 0, "total_bytes": 0}
    for category, files in CATEGORIES.items():
        cat_info = {"files": len(files), "bytes": 0, "streams": []}
        for fname in files:
            path = os.path.join(data_dir, category, fname)
            if os.path.exists(path):
                size = os.path.getsize(path)
                cat_info["bytes"] += size
                # Count rows (minus header)
                with open(path, "r") as f:
                    row_count = sum(1 for _ in f) - 1
                cat_info["streams"].append({"file": fname, "rows": row_count, "bytes": size})
        summary["categories"][category] = cat_info
        summary["total_files"] += cat_info["files"]
        summary["total_bytes"] += cat_info["bytes"]
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download the NAB dataset.")
    parser.add_argument("--check", action="store_true", help="Check integrity only, don't download.")
    parser.add_argument("--force", action="store_true", help="Re-download even if files exist.")
    parser.add_argument("--summary", action="store_true", help="Print dataset summary (requires data present).")
    args = parser.parse_args()

    if args.check:
        ok, missing, empty = check_integrity()
        if ok:
            print(f"All {EXPECTED_TOTAL_FILES} data files + {len(LABEL_FILES)} label files present and non-empty.")
        else:
            if missing:
                print(f"Missing ({len(missing)}):")
                for m in missing:
                    print(f"  {m}")
            if empty:
                print(f"Empty ({len(empty)}):")
                for e in empty:
                    print(f"  {e}")
        sys.exit(0 if ok else 1)

    if args.summary:
        ok, _, _ = check_integrity()
        if not ok:
            print("Data not fully downloaded. Run without --summary first.")
            sys.exit(1)
        s = get_dataset_summary()
        print(f"NAB Dataset: {s['total_files']} files, {s['total_bytes'] / 1024 / 1024:.1f} MB")
        for cat, info in s["categories"].items():
            print(f"  {cat}: {info['files']} files, {info['bytes'] / 1024:.0f} KB")
            for stream in info["streams"]:
                print(f"    {stream['file']}: {stream['rows']} rows")
        sys.exit(0)

    print(f"Downloading NAB dataset to {_data_dir()} ...")
    success = download_all(force=args.force)
    if success:
        ok, _, _ = check_integrity()
        if ok:
            print("Integrity check passed.")
        else:
            print("WARNING: Integrity check failed after download.", file=sys.stderr)
            sys.exit(1)
    else:
        sys.exit(1)
