# NAB Anomaly Detector

**Streaming anomaly detection benchmarked on the Numenta Anomaly Benchmark (NAB).**

A hybrid detector combining statistical methods (EWMA, Z-score) with machine learning
(isolation forest), scored against NAB's 58 real-world time-series streams across
7 categories. The goal: beat published baselines on at least 3 of 7 categories while
running 10× faster than the leading detectors.

## What this project does

Most anomaly detection research tests on synthetic data or private datasets. NAB is
different — it's a public benchmark with 58 real streams (server metrics, Twitter
volumes, taxi rides, temperature readings) and a standardised scoring protocol that
rewards early detection and penalises false positives. Published detectors include
Numenta's own HTM, Twitter's ADVec, and several others on the official leaderboard.

This project:

1. Implements three statistical detectors (EWMA, Z-score, threshold) and one ML
   detector (isolation forest).
2. Combines the best of each into a hybrid that votes on anomalies.
3. Scores every detector on NAB using the official scoring protocol.
4. Measures the speed-accuracy tradeoff — how much accuracy do you gain (or lose) by
   adding ML to a fast statistical baseline?

## Key question

Can a simple hybrid (statistical + ML) match or beat complex published detectors on a
standard benchmark, while being fast enough for real-time streaming use?

## Quick start

```bash
git clone https://github.com/vir-cipher/nab-anomaly-detector.git
cd nab-anomaly-detector
pip install -r requirements.txt
python -m pytest tests/
```

## Project status

Phase 10 (Bootstrap) — building.

## Credits

Built by Ansh Vir Bhargav (B.Cyber, IIT Kanpur — WSAIS) as a Semester 1 project
aligned with Fundamentals of Data Engineering.

Benchmark: [Numenta Anomaly Benchmark](https://github.com/numenta/NAB)
(Lavin & Ahmad, 2015).
