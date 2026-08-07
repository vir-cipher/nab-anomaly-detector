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

## 6–9. (Sections added as phases complete.)