# README_BENCHMARK.md
# PS 26152 — AI-Powered Criminal Network Analysis System

# Benchmark & Scorecard

This document explains how to run the offline benchmark suite, what metrics are
produced, how they are measured, and the acceptance criteria that must pass for
the system to be considered demo-ready.

---

## Overview

The benchmark suite evaluates four pipeline components:

| Module | What is measured |
|--------|-----------------|
| **Extraction** | Precision / Recall / F1 for named entities (names, phones, dates, amounts) |
| **Link Prediction** | AUC-ROC and Precision@5 on transaction-network link scoring |
| **Candidate Ranking** | *Candidate coverage* — does an elevated-priority candidate appear in top-3? |
| **Latency** | Median wall-clock time (ms) for parse, graph-build, search, top-3 |
| **Determinism** | Bit-for-bit equality between live output and cached reference output |

> **Language note:** All candidate identifications are phrased as
> "elevated-priority candidates requiring investigator verification."
> No findings in this system constitute legal determinations.

---

## Quick Start (Three Commands)

```bash
# 1. Enter the repo on the feature branch
git clone https://github.com/keshavmittal1907-alt/Criminal_Network_Ananlysis.git
cd Criminal_Network_Ananlysis
git checkout feature/benchmark-scorecard

# 2. Run the full offline benchmark pipeline
bash scripts/run_benchmarks.sh

# 3. Run the automated test assertions
pytest -q tests/test_benchmarks.py
```

Outputs land in:
- `results/scorecard.json` — machine-readable metrics + acceptance checks
- `results/scorecard.md`  — human-readable summary for judges

---

## Demo Mode (Instant, No ML Loading)

If you want to demo the system without waiting for model loading, run only
the two data-prep scripts (already done by `run_benchmarks.sh`):

```bash
python scripts/generate_demo_data.py    # creates demo_dataset/ (~1 s)
python scripts/prepare_demo_cache.py    # builds demo_cache/    (~1 s)
```

The `demo_cache/` directory stores pre-computed outputs (entities, top-3
scores, TF-IDF search index) as JSON files, so the Streamlit dashboard
can load them instantly without running any ML inference.

---

## File Map

```
scripts/
  run_benchmarks.sh          — master runner (sets up venv, runs all)
  generate_demo_data.py      — creates 3 synthetic cases in demo_dataset/
  prepare_demo_cache.py      — pre-computes outputs, writes demo_cache/

benchmarks/
  extraction_benchmark.py    — precision/recall/F1 per entity type
  link_prediction_benchmark.py — AUC-ROC + Precision@5 on txn graphs
  latency_benchmark.py       — wall-clock times + candidate coverage
  determinism_test.py        — SHA-256 hash equality vs cached outputs

results/
  scorecard.json             — machine-readable scorecard (metrics + checks)
  scorecard.md               — human-readable markdown for judges
  extraction_results.json    — per-case extraction detail
  link_prediction_results.json
  latency_results.json
  determinism_results.json

tests/
  test_benchmarks.py         — pytest suite (imports results, checks thresholds)

demo_dataset/
  fir_files/case_*.txt       — synthetic FIR text documents
  cdr_files/case_*_cdr.csv   — synthetic call-detail records
  transaction_files/case_*_transactions.csv
  ground_truth.json          — entity labels + known edges + expected candidates

demo_cache/
  case_*_entities.json       — cached extracted entities
  case_*_top3.json           — cached top-3 candidate rankings
  search_index.json          — cached TF-IDF token index
  cache_manifest.json        — SHA-256 hashes for determinism check
```

---

## Acceptance Criteria

All of these must pass for the demo to be considered production-ready:

| Check | Threshold | How measured |
|-------|-----------|--------------|
| Extraction F1 — names | ≥ 0.80 | Exact substring match vs ground-truth names |
| Extraction F1 — phones | ≥ 0.90 | Exact regex match vs 10-digit phone numbers |
| Link-prediction AUC | ≥ 0.55 (heuristic baseline for small synthetic graph) | Manual pairwise AUC on known/unknown edges |
| Candidate coverage | ≥ 0.67 | ≥1 GT candidate in top-3 for ≥2/3 cases |
| Parse latency | ≤ 50 ms median | 20 repetitions on demo FIR |
| Graph-build latency | ≤ 100 ms median | 20 repetitions on demo CDR + txn |
| Search latency | ≤ 20 ms median | 20 repetitions, TF-IDF index |
| Top-3 latency | ≤ 100 ms median | 20 repetitions on demo case |
| Determinism | PASS (all files) | SHA-256 == cached manifest |

---

## Adding New Dependencies

The benchmark suite uses only the following Python packages (already in
`requirements.txt`):

- `numpy` — latency statistics, RNG seeding
- `pytest` — test runner

No `spacy`, `torch`, or `sentence-transformers` are required for the
benchmark path. If you add a new dependency, document it here and in
`requirements.txt`.

---

## Reproducing with a Different Seed

```bash
BENCH_SEED=123 bash scripts/run_benchmarks.sh
```

Note: changing the seed will change the demo data, so you must also
re-run `prepare_demo_cache.py` to refresh the determinism baseline.

---

## Notes for Judges

- **Offline**: no network calls are made during any benchmark step.
- **CPU-only**: all operations run on CPU; no GPU is required.
- **Safe language**: all candidate rankings use the phrase
  *"elevated-priority candidate — requires investigator verification"*.
  The system makes no guilt determinations.
- **Synthetic data**: `demo_dataset/` contains fully synthetic data with
  a `SYNTHETIC_DATA_NOTICE.md`. No real personal data is included.
