# NAB Anomaly Detector

**Streaming anomaly detection benchmarked on the Numenta Anomaly Benchmark (NAB).**

Six detectors — four statistical (Windowed Gaussian, EWMA, Z-score, static threshold) and two ML-based (isolation forest, hybrid ensemble) — scored on all 58 NAB streams across three scoring profiles.

## Key finding

The simplest strong detector wins on both axes. Windowed Gaussian scores **NAB 40.13** (standard profile) while processing each point in **0.71 µs** — 156× faster than the isolation forest (5.25, 111.55 µs/pt). The hybrid ensemble (weighted-average 0.7/0.3 Gaussian + forest) scores only **11.40**, a clean negative result: the forest’s flat ~0.5 baseline noise dilutes the Gaussian’s sharp signal and drags the combined score below either component’s optimum.

Full 6-detector leaderboard (standard profile):

| Detector | NAB score | µs/point | Speedup vs iforest |
|----------|-----------|----------|--------------------|
| Gaussian | 40.13 | 0.71 | 156× |
| Z-score | 23.43 | 0.98 | 114× |
| EWMA | 20.64 | 0.86 | 130× |
| Hybrid | 11.40 | 108.06 | 1.0× |
| Isolation forest | 5.25 | 111.55 | 1.0× |
| Threshold | 0.00 | 0.13 | 836× |

All scores are reproducible from a fresh clone (see Quick start).

## Quick start

```bash
git clone https://github.com/vir-cipher/nab-anomaly-detector.git
cd nab-anomaly-detector
pip install -r requirements.txt
python -m pytest tests/               # 235 tests, ~2 min
python src/run_all_detectors.py       # score all detectors on NAB
python -m streamlit run src/dashboard.py  # live anomaly visualisation
```

The NAB dataset (~9 MB) is downloaded automatically on first run.

## Repository layout

- `src/` — detectors, scorers, benchmark runner, Streamlit dashboard
- `tests/` — 235 tests covering every detector, scorer, and pipeline stage
- `results/` — scored leaderboards, parameter sensitivity, runtime benchmarks
- `docs/WALKTHROUGH.md` — full 2,800-word write-up (methodology, results, analysis)
- `.project-meta/` — frozen spec, immutable plan, canonical build ledger

## What this project demonstrates

- Faithful reimplementation of the NAB scoring protocol (three application profiles)
- Reproducible benchmark: Gaussian baseline within 1.4% of the published NAB score
- Honest reporting of a negative result (hybrid underperforms its best component)
- Pure-Python streaming detectors with no look-ahead (real-time compatible)
- 235-test suite with CI, runtime benchmarks, and a live Streamlit dashboard

## Credits

Built by **Ansh Vir Bhargav** (B.Cyber, IIT Kanpur — WSAIS) as a Semester 1 project aligned with Fundamentals of Data Engineering.

Benchmark: [Numenta Anomaly Benchmark](https://github.com/numenta/NAB) (Lavin & Ahmad, 2015).
