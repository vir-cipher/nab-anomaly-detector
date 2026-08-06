# NAB Anomaly Detector — Frozen Specification

**Title:** Streaming Anomaly Detection on the Numenta Anomaly Benchmark
**Author:** Ansh Vir Bhargav (vir-cipher)
**Programme:** B.Cyber, IIT Kanpur (WSAIS), Semester 1
**Frozen:** 2026-08-06 (bootstrap commit)

## Hypothesis

A hybrid detector combining EWMA (statistical) and isolation forest (ML) beats
NAB published baselines on at least 3 of 7 categories while running at least
10× faster than the leading detectors.

## Dataset

Numenta Anomaly Benchmark (NAB) — 58 real-world time-series streams across 7
categories. Source: https://github.com/numenta/NAB (Lavin & Ahmad, 2015).

## Methodology

1. Download NAB dataset and implement the standard NAB scoring protocol.
2. Implement a null detector and reproduce ≥1 published baseline to validate scoring.
3. Build 3 statistical detectors: EWMA, Z-score, simple threshold.
4. Build 1 ML detector: isolation forest.
5. Score each on NAB. Record NAB score per category.
6. Design a hybrid combiner (best statistical + ML). Score on NAB.
7. Benchmark runtime (wall-clock seconds per stream) for every detector.
8. Build a Streamlit dashboard for live visualisation.
9. Write up results in WALKTHROUGH.md (>2000 words) and README.md (<500 words).

## Deliverables

- Scored results on all 7 NAB categories for every detector.
- Speed-accuracy tradeoff table.
- Streamlit dashboard.
- Reproducible from `git clone` + 3 commands.

## Prior art

- Numenta HTM (Hierarchical Temporal Memory) — original NAB baseline.
- Twitter ADVec — anomaly detection via seasonal decomposition.
- Skyline (Etsy), KNN-CAD, Relative Entropy, Bayesian Changepoint —
  published NAB leaderboard entries.

## Constraints

- Pure Python. No exotic dependencies beyond scikit-learn, pandas, numpy, streamlit.
- All data is public (NAB dataset is open-source, MIT licence).
- No API keys required for core functionality (no LLM calls).

## Curriculum alignment

Fundamentals of Data Engineering (Data) — Semester 1, IIT Kanpur.
