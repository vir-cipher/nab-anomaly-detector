# NAB Anomaly Detector — Walkthrough

This document explains the project from the ground up. It grows with each phase.
If you are reading this for the first time, start here.

## 1. What problem are we solving?

Imagine you are monitoring a server. Every second, you get a number — CPU usage,
request count, memory. Most of the time the numbers follow a pattern: higher during
business hours, lower at night. An **anomaly** is when something breaks that pattern —
a sudden spike, a flatline, or a slow drift that should not be there.

Catching anomalies early matters. A spike in error rates might mean a deployment broke
something. A flatline in traffic might mean your load balancer died. The question is:
how do you build a program that flags these anomalies automatically, in real time, as
each new number arrives?

That is the **streaming anomaly detection** problem. "Streaming" means the detector
sees one data point at a time, in order, and must decide *right now* whether it is
anomalous — it cannot look ahead or reprocess the whole history.

## 2. What is NAB?

The **Numenta Anomaly Benchmark** (NAB) is a public dataset of 58 real-world
time-series streams, grouped into 7 categories:

- **artificialNoAnomaly / artificialWithAnomaly** — synthetic control streams.
- **realAdExchange** — online advertising metrics.
- **realAWSCloudwatch** — Amazon server monitoring metrics.
- **realKnownCause** — streams where the anomaly cause is documented (machine
  failures, CPU spikes after a software change).
- **realTraffic** — NYC taxi ride counts, web traffic volumes.
- **realTweets** — Twitter mention volumes for large companies.

Each stream comes with labelled **anomaly windows** — time ranges where a human expert
marked "something unusual happened here." NAB defines a scoring protocol that rewards
**early detection** (catching the anomaly before the window closes) and penalises
**false positives** (crying wolf). Published detectors are scored on this benchmark,
creating a public leaderboard. Our job is to build detectors, score them the same way,
and see where they land.

Source: [github.com/numenta/NAB](https://github.com/numenta/NAB) (Lavin & Ahmad, 2015).

## 3. Our approach: statistical + ML hybrid

We build four detectors, then combine the best two:

- **EWMA (Exponentially Weighted Moving Average):** a "smart running average" that
  weights recent values more heavily. When a new data point deviates far from the
  EWMA, it is flagged. Fast, simple, explainable.
- **Z-score:** measures how many standard deviations a point is from the recent mean.
  A classic statistical method.
- **Threshold:** the simplest possible detector — flag anything above or below fixed
  bounds. Our sanity-check baseline.
- **Isolation forest:** a machine-learning method that builds random decision trees
  and measures how easily a data point can be isolated from the rest. Anomalies are
  isolated quickly (few splits); normal points take many splits.

The **hybrid** combines the best statistical detector with isolation forest using a
voting rule. If both agree "anomaly," we flag it with high confidence. If they
disagree, the voting rule breaks the tie. The hypothesis: this simple hybrid beats
published baselines on at least 3 of 7 NAB categories while running 10x faster.

## 4. How to verify our claims

Every result is reproducible from a fresh clone:

`ash
git clone https://github.com/vir-cipher/nab-anomaly-detector.git
cd nab-anomaly-detector
pip install -r requirements.txt
python src/download_nab.py        # downloads 58 CSV files from NAB
python -m pytest tests/
`

Scores are compared against NAB's published leaderboard at
[github.com/numenta/NAB](https://github.com/numenta/NAB).

## 5. The NAB dataset: what we downloaded and what is inside

### How to get the data

Run `python src/download_nab.py` from the project root. The script downloads all
58 CSV files and 2 label files from the official NAB GitHub repository into
`data/nab/`. It also has `--check` (verify files without downloading) and
`--summary` (print row counts per stream) modes. The data is gitignored — it is
fetched fresh by CI and by anyone who clones the repo.

### What the data looks like

Every CSV has exactly two columns: `timestamp` and `value`.

`
timestamp,value
2014-04-01 00:00:00,18.0
2014-04-01 00:05:00,18.0
2014-04-01 00:10:00,21.0
`

`timestamp` is a datetime string. `value` is a float — the metric being monitored
(CPU %, request count, tweet volume, etc.). One row = one observation in time order.

### The 7 categories at a glance

| Category | Files | Description | Row range |
|---|---|---|---|
| artificialNoAnomaly | 5 | Synthetic streams with NO anomalies (control group) | 4 032 each |
| artificialWithAnomaly | 6 | Synthetic streams with injected anomalies | 4 032 each |
| realAWSCloudwatch | 17 | Amazon EC2/RDS/ELB metrics (CPU, disk, network) | 1 243–4 730 |
| realAdExchange | 6 | Online ad exchange CPC/CPM metrics | 1 538–1 643 |
| realKnownCause | 7 | Streams where the anomaly cause is documented | 1 882–22 695 |
| realTraffic | 7 | NYC taxi rides, highway speed/occupancy sensors | 1 127–2 500 |
| realTweets | 10 | Twitter mention volumes for AAPL, AMZN, FB, etc. | ~15 800 each |
| **Total** | **58** | | |

Total size: ~9.1 MB across all 58 files.

### Ground-truth labels

Two label files live in `data/nab/labels/`:

- **combined_windows.json** — for each data file, a list of `[start, end]` timestamp
  pairs marking the anomaly windows. A detector that flags a point inside (or just
  before) the window scores well; flagging outside is a false positive.
- **combined_labels.json** — the exact timestamp of each anomaly. Used for
  point-based scoring (less common than window-based, but available).

Example entry from `combined_windows.json`:

`json
{
  "realKnownCause/nyc_taxi.csv": [
    ["2014-11-01 19:00:00", "2014-11-03 15:30:00"],
    ["2014-11-27 12:00:00", "2014-11-29 06:30:00"]
  ]
}
`

This says the NYC taxi stream has two anomaly windows — one around the NYC Marathon
(1–3 Nov 2014) and one around Thanksgiving (27–29 Nov 2014).

### Why this dataset matters for the project

NAB is the *only* widely-adopted streaming anomaly benchmark with a published
leaderboard. Scoring on it means our results are directly comparable to Numenta's HTM,
Twitter's AnomalyDetection, Etsy's Skyline, and other published detectors. Without
NAB, our numbers would be "trust us" — with NAB, they are "verify us."

## 6. Parameter sensitivity of the isolation forest (step-011)

Before combining detectors in Phase 14, we asked a simple question: **does tuning
the isolation forest close the gap to the statistical baselines?** `src/param_sensitivity.py`
runs a one-factor-at-a-time (OFAT) sweep around the step-009 baseline
(`n_trees=64`, `sample_size=128`, `shingle_size=4`, `train_size=256`), scoring every
config through the *same* `run_all_detectors.score_all` machinery used for every
other detector, so the numbers are directly comparable to `results/comparison.csv`.

Two knobs are swept, all else held at baseline:

- **n_trees** (ensemble size): 32, 64\*, 128
- **sample_size** (per-tree sub-sample psi): 64, 128\*, 256

(\* = baseline). Results (`results/iforest_sensitivity.csv`), NAB score per profile:

| config          | standard | reward_low_fp | reward_low_fn |
|-----------------|:--------:|:-------------:|:-------------:|
| baseline        |   5.25   |     4.24      |     6.08      |
| n_trees=32      |   5.74   |     3.54      |     7.11      |
| n_trees=128     |   5.93   |     4.36      |     7.40      |
| sample_size=64  |   4.97   |     4.16      |     6.06      |
| sample_size=256 | **6.23** |     3.17      |   **7.90**    |

**What it shows.** Tuning moves the NAB score by only ~1.3 points on the standard
profile (4.97 to 6.23) — the forest stays in the 5-6 band, still ~35 points under
Windowed Gaussian's 40.13. **Tuning these knobs does not close the gap.** The clearest
effect is `sample_size`, and it exposes a real profile trade-off: a larger sub-sample
(256) is best when detections are rewarded (standard 6.23, reward_low_fn 7.90) but
*worst* under the false-positive-averse profile (reward_low_fp 3.17), because seeing
more structure per tree makes the forest fire more, which that profile penalises.
More trees help marginally (ensembles stabilise well before 100 trees, Liu 2008 Fig.5).

**Validity anchor.** The `baseline` row reproduces step-010's committed
`comparison.csv` byte-exact (standard 5.2476, reward_low_fp 4.2410, reward_low_fn
6.0846) — the sweep is scored through the identical pipeline, so the deltas above are
attributable to the parameters alone. This motivates the Phase-14 hybrid: statistics
carry the accuracy, the forest contributes where false alarms are expensive.

### Runtime — how fast is each detector? (step-014)

Accuracy is only half the headline; the other half is *speed*. `src/benchmark_runtime.py`
streams the whole NAB corpus (365,558 points, 58 streams) through each detector one point
at a time and times only the `handle_record` loop (a fresh detector per stream, best of
repeated passes to trim OS noise). Results in `results/runtime_benchmark.csv`:

| detector  | points/sec | us/point | speed-up vs iforest |
|-----------|-----------:|---------:|--------------------:|
| threshold | 7,492,417  |   0.13   |      835.8x         |
| gaussian  | 1,402,069  |   0.71   |      156.4x         |
| ewma      | 1,169,500  |   0.86   |      130.5x         |
| zscore    | 1,019,969  |   0.98   |      113.8x         |
| hybrid    |     9,254  | 108.06   |        1.03x        |
| iforest   |     8,964  | 111.55   |     1.00x (ref)     |

**What it shows.** Every statistical detector is 114x–836x faster than the pure-Python
isolation forest. The accuracy leader — Windowed Gaussian (NAB 40.13) — is ~156x faster
than the forest, so it wins on **both** axes. The hybrid runs at forest speed (1.03x)
because it embeds the forest: the default hybrid is therefore slower *and* (step-013) less
accurate than plain Gaussian. Absolute times are machine-specific (Python 3.14, Intel
11th-gen); the portable finding is the ratio. The project's "10x speed" ambition is beaten
by more than a full order of magnitude — through the simple statistical detectors, not the ML one.

## 7–9. (Sections added as phases complete.)