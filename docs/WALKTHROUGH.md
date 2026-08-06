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
published baselines on at least 3 of 7 NAB categories while running 10× faster.

## 4. How to verify our claims

Every result is reproducible from a fresh clone:

```bash
git clone https://github.com/vir-cipher/nab-anomaly-detector.git
cd nab-anomaly-detector
pip install -r requirements.txt
python -m pytest tests/
```

Scores are compared against NAB's published leaderboard at
[github.com/numenta/NAB](https://github.com/numenta/NAB).

## 5–9. (Sections added as phases complete.)
