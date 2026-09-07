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

### Live dashboard — watch a detector work (step-015)

Numbers in a CSV are hard to feel; a live view is not. `src/dashboard.py` is a
Streamlit app that streams any NAB feed through any detector **one point at a time,
with no look-ahead**, and shows the anomaly score building up exactly as a production
monitor would see it. Ground-truth anomaly windows are named alongside the raw value
and score charts, and an adjustable threshold line turns scores into alarms live.

Run it locally:

```
pip install -r requirements-dashboard.txt   # Streamlit is an optional extra
streamlit run src/dashboard.py
```

**Design note — testable Streamlit.** All computation (streaming the detector,
thresholding, mapping anomaly windows to indices, per-frame metrics) lives in plain
functions in `src/dashboard.py`; Streamlit is imported lazily inside `main()`. So the
logic is unit-tested without a browser, and a separate `AppTest` smoke test
(`streamlit.testing`) proves the whole app builds and runs. The core benchmark stays
pure-Python — Streamlit never enters `requirements.txt`.

**Correctness anchor.** The key test asserts that the scores for the first *k* points
are identical whether you feed the detector *k* points or the whole stream — the
mathematical definition of a real-time detector (no point may depend on its future).
If the live view is moving, that property is what makes it honest.

## 7. Results: the full six-detector leaderboard

NAB scores every detector on the same 58 streams and 116 anomaly windows. The
scoring protocol has three **profiles** (weighting schemes):

- **standard** — the default. Early detection rewarded, false positives penalised.
- **reward_low_fp** — false positives cost *more*. Rewards conservative detectors.
- **reward_low_fn** — missed anomalies cost *more*. Rewards aggressive detectors.

A **NAB score** of 100 means perfect: every anomaly caught early, zero false alarms.
A score of 0 means the detector found nothing useful. Negative scores are possible
(the penalties outweigh the rewards).

### Accuracy leaderboard (standard profile)

| Rank | Detector | NAB score | Type |
|------|----------|:---------:|------|
| 1 | Windowed Gaussian | 40.13 | statistical |
| 2 | Z-score | 23.43 | statistical |
| 3 | EWMA | 20.64 | statistical |
| 4 | Hybrid (Gaussian + iforest) | 11.40 | combined |
| 5 | Isolation forest | 5.25 | ML |
| 6 | Threshold | 0.00 | baseline |

The pattern repeats across all three profiles. Windowed Gaussian leads every
profile: 40.13 (standard), 23.94 (reward_low_fp), 47.73 (reward_low_fn). It is
the only detector that scores above zero on the conservative reward_low_fp profile
(the others either score zero or single digits). In other words, Gaussian is the
only detector that can catch anomalies without also crying wolf too often.

### Speed leaderboard

| Detector | Points per second | Microseconds per point | Speed-up vs iforest |
|----------|------------------:|-----------------------:|--------------------:|
| Threshold | 7,492,417 | 0.13 | 836× |
| Gaussian | 1,402,069 | 0.71 | 156× |
| EWMA | 1,169,500 | 0.86 | 130× |
| Z-score | 1,019,969 | 0.98 | 114× |
| Hybrid | 9,254 | 108.06 | 1.03× |
| Iforest | 8,964 | 111.55 | 1.00× |

The statistical detectors process over a million data points per second in pure
Python. The isolation forest manages about 9,000 — roughly 100× slower. Since the
hybrid embeds the forest, it runs at forest speed. The accuracy winner (Gaussian) is
also 156× faster than the ML approach.

Timings are from a single machine (Python 3.14, Intel 11th-gen i7). The absolute
numbers are machine-specific, but the *ratios* are portable: the statistical
detectors will always be orders of magnitude faster because they do O(1) work per
point, while the forest does O(n_trees × log sample_size).

## 8. An honest negative: why the hybrid underperforms

The original hypothesis was:

> A hybrid detector combining EWMA and isolation forest beats NAB published
> baselines on at least 3 of 7 categories while running at least 10× faster.

The result: **the hypothesis does not hold.** The hybrid (NAB 11.40) scores below
every statistical detector and runs at forest speed (1.03×, not 10×). Windowed
Gaussian alone beats it on both accuracy *and* speed. This is a negative result,
and we report it as such.

### Why the combination hurts

The hybrid uses a **weighted average** (70% Gaussian, 30% isolation forest) to
combine the two anomaly scores. The problem is that the isolation forest's scores
hover around 0.5 for almost every point — anomaly or not. It produces a nearly flat
signal that acts as noise when mixed with the sharp, well-calibrated Gaussian score.
The 70/30 blend drags the Gaussian peaks down toward the forest's flat baseline,
making the combined score less decisive.

We tested alternative voting rules (mean, max, min) and different weight splits
during the combiner's design (step-012). None overcame the fundamental issue: the
forest's per-point anomaly scores are not well-separated enough to be useful for
simple voting. More sophisticated combination strategies (stacking, learned
meta-classifiers) were out of scope for a Semester 1 project — but the *diagnosis*
(flat forest scores) points exactly to what a Semester 2 extension could fix.

### Why reporting a negative result matters

In research, negative results are not failures. They prevent other people from
wasting time on the same dead end, and they sharpen the question for the next
attempt. Our negative result carries a specific, verifiable insight: **the isolation
forest's anomaly scores are too flat for simple voting to work.** That is not a
guess; it is derivable from the scored outputs in `results/`. Anyone who clones the
repo can verify it by plotting the forest's per-point scores against Gaussian's
and observing the variance difference.

The parameter sensitivity analysis (section 6) already showed that tuning the forest
only moves the NAB score by ~1.3 points — the gap to Gaussian is structural, not
parametric. This makes the write-up *more* credible, not less: we did the experiment,
measured the outcome, and did not cherry-pick our way to a positive headline.

## 9. What we learned

### Technical takeaways

- **Simple statistics can be surprisingly strong.** Windowed Gaussian — a running
  mean-and-variance calculation — outperforms a 64-tree isolation forest on NAB
  by a factor of 8× on accuracy and 156× on speed. Complexity is not free.
- **Combination is not trivially additive.** Averaging a sharp signal (Gaussian)
  with a flat one (forest) degrades both. Effective ensembles need components whose
  scores are individually informative.
- **Benchmarks enforce honesty.** Without NAB, we might have claimed the hybrid
  "works"; with NAB, the numbers speak for themselves. Every claim in this project
  traces to a scored result in `results/` or a test in `tests/`.
- **Streaming matters.** Every detector in this project processes data one point at a
  time, in order, with no look-ahead. This constraint is what makes the results
  applicable to real monitoring — a detector that needs the whole dataset up front
  cannot be deployed in production.

### Reproducibility

From a fresh machine:

```
git clone https://github.com/vir-cipher/nab-anomaly-detector.git
cd nab-anomaly-detector
pip install -r requirements.txt
python src/download_nab.py
python -m pytest tests/ -q
```

Every test passes. Every number in this write-up is derived from files in `results/`
that are themselves produced by scripts in `src/`. The Streamlit dashboard
(`src/dashboard.py`) shows any detector working live on any NAB stream — install
the optional `requirements-dashboard.txt` and run `streamlit run src/dashboard.py`.

### Credits

**Author:** Ansh Vir Bhargav (`vir-cipher`) — B.Cyber at IIT Kanpur (WSAIS),
Semester 1, 2026.

**Benchmark:** Numenta Anomaly Benchmark (Lavin & Ahmad, 2015).
Source: [github.com/numenta/NAB](https://github.com/numenta/NAB).

**Curriculum alignment:** Fundamentals of Data Engineering (Data) — Semester 1.

**Tools:** Python 3.14, scikit-learn, pandas, numpy, Streamlit.
