#!/usr/bin/env python3
"""
benchmarks/latency_benchmark.py
PS 26152 — Criminal Network Analysis System
--------------------------------------------
Measures wall-clock latency (milliseconds) for four pipeline operations:
  1. parse       — FIR text regex extraction
  2. graph_build — Build a NetworkX graph from CDR + transaction CSVs
  3. search      — Answer one keyword query using the cached TF-IDF index
  4. top3        — Compute top-3 candidate scores from extracted entities

Reports median and p95 over N_RUNS repetitions for each operation.

Fully offline. Deterministic with SEED=42 (scores, not timings).
CPU-only: uses only regex, csv, and numpy. No ML model loading.

Usage:
    python benchmarks/latency_benchmark.py
    # or called by scripts/run_benchmarks.sh
"""

import csv
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

import numpy as np

REPO_ROOT   = Path(__file__).resolve().parent.parent
DEMO_DIR    = REPO_ROOT / "demo_dataset"
CACHE_DIR   = REPO_ROOT / "demo_cache"
RESULTS_DIR = REPO_ROOT / "results"

SEED    = 42
N_RUNS  = 20   # repetitions for median/p95

np.random.seed(SEED)

# ── regex patterns ──────────────────────────────────────────────────────────
_PHONE   = re.compile(r"\b(\d{10})\b")
_DATE    = re.compile(r"\b(\d{2}-\d{2}-\d{4})\b")
_AMOUNT  = re.compile(r"Rs\.\s*([\d,]+)", re.IGNORECASE)
_ACCOUNT = re.compile(r"\b(ACC\d{5})\b")

PERSONS = [
    {"name": "Ravi Kumar",     "phone": "9876543210", "account": "ACC00101"},
    {"name": "Sunita Sharma",  "phone": "9123456789", "account": "ACC00102"},
    {"name": "Mohammed Iqbal", "phone": "9988776655", "account": "ACC00103"},
    {"name": "Priya Nair",     "phone": "8877665544", "account": "ACC00104"},
    {"name": "Deepak Verma",   "phone": "7766554433", "account": "ACC00105"},
    {"name": "Anjali Singh",   "phone": "9654321098", "account": "ACC00106"},
    {"name": "Rakesh Yadav",   "phone": "9543210987", "account": "ACC00107"},
    {"name": "Fatima Begum",   "phone": "9432109876", "account": "ACC00108"},
    {"name": "Suresh Patil",   "phone": "9321098765", "account": "ACC00109"},
    {"name": "Kavya Reddy",    "phone": "9210987654", "account": "ACC00110"},
]
PHONE_TO_PERSON   = {p["phone"]: p   for p in PERSONS}
ACCOUNT_TO_PERSON = {p["account"]: p for p in PERSONS}


def _measure(fn: Callable, n: int = N_RUNS) -> dict:
    """Time fn() n times, return median and p95 in ms."""
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    arr = np.array(times)
    return {
        "median_ms":  round(float(np.median(arr)), 2),
        "p95_ms":     round(float(np.percentile(arr, 95)), 2),
        "min_ms":     round(float(np.min(arr)), 2),
        "max_ms":     round(float(np.max(arr)), 2),
        "n_runs":     n,
    }


# ── 1. Parse ─────────────────────────────────────────────────────────────────
def _pick_fir() -> Path:
    """Return first available FIR file."""
    fir_dir = DEMO_DIR / "fir_files"
    files = sorted(fir_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No FIR files found in {fir_dir}. "
                                "Run: python scripts/generate_demo_data.py")
    return files[0]


def _parse_once(fir_text: str) -> dict:
    """Regex extraction: phones, dates, amounts, accounts."""
    phones   = _PHONE.findall(fir_text)
    dates    = _DATE.findall(fir_text)
    amounts  = _AMOUNT.findall(fir_text)
    accounts = _ACCOUNT.findall(fir_text)
    persons  = {PHONE_TO_PERSON[ph]["name"] for ph in phones if ph in PHONE_TO_PERSON}
    return {"phones": phones, "dates": dates, "amounts": amounts,
            "accounts": accounts, "persons": list(persons)}


def benchmark_parse() -> dict:
    fir_file = _pick_fir()
    fir_text = fir_file.read_text(encoding="utf-8", errors="replace")
    result = _measure(lambda: _parse_once(fir_text))
    result["file"] = str(fir_file.relative_to(REPO_ROOT))
    return result


# ── 2. Graph Build ───────────────────────────────────────────────────────────
def _load_csvs() -> tuple[list[dict], list[dict]]:
    cdr_dir = DEMO_DIR / "cdr_files"
    txn_dir = DEMO_DIR / "transaction_files"
    cdr_files = sorted(cdr_dir.glob("*.csv"))
    txn_files = sorted(txn_dir.glob("*.csv"))
    if not cdr_files or not txn_files:
        raise FileNotFoundError("CDR/txn files missing. Run generate_demo_data.py first.")

    def read_csv(p):
        with open(p, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    cdr_rows = read_csv(cdr_files[0])
    txn_rows = read_csv(txn_files[0])
    return cdr_rows, txn_rows


def _build_graph_once(cdr_rows: list[dict], txn_rows: list[dict]) -> dict:
    """Build adjacency list from CDR + transactions (pure Python, no networkx)."""
    adj: dict = defaultdict(set)
    weights: dict = defaultdict(float)
    for row in cdr_rows:
        caller   = row.get("caller", "").strip()
        receiver = row.get("receiver", "").strip()
        if caller and receiver and caller != receiver:
            adj[caller].add(receiver)
    for row in txn_rows:
        src = row.get("sender_account", "").strip()
        dst = row.get("receiver_account", "").strip()
        amt = float(row.get("amount_inr", 0))
        if src and dst and src != dst:
            adj[src].add(dst)
            weights[(src, dst)] += amt
    return {"n_nodes": len(adj), "n_edges": sum(len(v) for v in adj.values())}


def benchmark_graph_build() -> dict:
    cdr_rows, txn_rows = _load_csvs()
    result = _measure(lambda: _build_graph_once(cdr_rows, txn_rows))
    stats = _build_graph_once(cdr_rows, txn_rows)
    result["n_nodes"] = stats["n_nodes"]
    result["n_edges"] = stats["n_edges"]
    return result


# ── 3. Search ─────────────────────────────────────────────────────────────────
def _load_search_index() -> dict:
    idx_path = CACHE_DIR / "search_index.json"
    if not idx_path.exists():
        raise FileNotFoundError(f"Search index missing: {idx_path}. "
                                "Run: python scripts/prepare_demo_cache.py")
    return json.loads(idx_path.read_text())


def _search_once(index: dict, query: str) -> list[dict]:
    """TF-IDF keyword search against cached index."""
    stopwords = {"the", "a", "an", "is", "in", "of", "and", "to", "for"}
    tokens = [t.lower() for t in re.findall(r"[a-zA-Z0-9]+", query)
              if t.lower() not in stopwords and len(t) > 2]
    inv = index.get("inv_index", {})
    doc_scores: dict = defaultdict(float)
    for token in tokens:
        for hit in inv.get(token, []):
            doc_scores[hit["doc"]] += hit["tf"]
    ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return [{"doc": doc, "score": score} for doc, score in ranked[:5]]


def benchmark_search() -> dict:
    index = _load_search_index()
    query = "financial transfer suspicious account"
    result = _measure(lambda: _search_once(index, query))
    results = _search_once(index, query)
    result["query"] = query
    result["n_results"] = len(results)
    return result


# ── 4. Top-3 ─────────────────────────────────────────────────────────────────
def _score_top3(cdr_rows: list[dict], txn_rows: list[dict]) -> list[dict]:
    """Lightweight top-3 scoring (mirrors prepare_demo_cache.py logic)."""
    scores = {}
    for p in PERSONS:
        phone   = p["phone"]
        account = p["account"]
        name    = p["name"]
        contacts = {r["receiver"] for r in cdr_rows if r.get("caller") == phone}
        contacts |= {r["caller"] for r in cdr_rows if r.get("receiver") == phone}
        degree_score = min(1.0, len(contacts) / max(len(PERSONS), 1))
        calls = [r for r in cdr_rows if r.get("caller") == phone or r.get("receiver") == phone]
        late_night = sum(1 for r in calls if int(r.get("hour", 12)) >= 23 or int(r.get("hour", 12)) <= 4)
        temporal_score = late_night / max(len(calls), 1)
        sent = sum(float(r["amount_inr"]) for r in txn_rows if r.get("sender_account") == account)
        recv = sum(float(r["amount_inr"]) for r in txn_rows if r.get("receiver_account") == account)
        txn_score = min(1.0, (sent + recv) / 5_000_000)
        risk = 0.30 * degree_score + 0.25 * temporal_score + 0.45 * txn_score
        scores[name] = risk
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]


def benchmark_top3() -> dict:
    cdr_rows, txn_rows = _load_csvs()
    result = _measure(lambda: _score_top3(cdr_rows, txn_rows))
    top3 = _score_top3(cdr_rows, txn_rows)
    result["top3_names"] = [name for name, _ in top3]
    return result


# ── Candidate Coverage ───────────────────────────────────────────────────────
def evaluate_candidate_coverage() -> dict:
    """
    For each case, run top-3 scoring and check if at least one
    ground-truth elevated-priority candidate appears in the top-3.
    Metric: candidate_coverage = n_cases_with_hit / n_cases.
    Safe language: no accusations — these are candidates requiring verification.
    """
    gt_path = DEMO_DIR / "ground_truth.json"
    if not gt_path.exists():
        return {"candidate_coverage": None, "error": "ground_truth.json missing"}

    gt_all = json.loads(gt_path.read_text())
    hits = 0
    case_details = []

    for case_id, gt in gt_all.items():
        cdr_file = REPO_ROOT / gt["cdr_file"]
        txn_file = REPO_ROOT / gt["txn_file"]

        cdr_rows = []
        if cdr_file.exists():
            with open(cdr_file, newline="", encoding="utf-8") as f:
                cdr_rows = list(csv.DictReader(f))
        txn_rows = []
        if txn_file.exists():
            with open(txn_file, newline="", encoding="utf-8") as f:
                txn_rows = list(csv.DictReader(f))

        top3 = [name for name, _ in _score_top3(cdr_rows, txn_rows)]
        expected = gt["top3_candidates"]
        hit = any(name in top3 for name in expected)
        if hit:
            hits += 1
        case_details.append({
            "case_id": case_id,
            "top3_predicted": top3,
            "top3_expected_candidates": expected,
            "hit": hit,
        })

    coverage = round(hits / len(gt_all), 4) if gt_all else 0.0
    return {
        "candidate_coverage": coverage,
        "n_cases": len(gt_all),
        "n_hits": hits,
        "note": "Coverage = fraction of cases where ≥1 elevated-priority candidate appears in top-3. Requires investigator verification.",
        "per_case": case_details,
    }


# ── Main ─────────────────────────────────────────────────────────────────────
def run_latency_benchmark() -> dict:
    print(f"  Running {N_RUNS} repetitions each ...")

    results = {}

    print("  [1/4] parse latency ...")
    try:
        results["parse"] = benchmark_parse()
        print(f"        median={results['parse']['median_ms']:.2f} ms  "
              f"p95={results['parse']['p95_ms']:.2f} ms")
    except Exception as e:
        results["parse"] = {"error": str(e)}
        print(f"        ERROR: {e}")

    print("  [2/4] graph_build latency ...")
    try:
        results["graph_build"] = benchmark_graph_build()
        print(f"        median={results['graph_build']['median_ms']:.2f} ms  "
              f"p95={results['graph_build']['p95_ms']:.2f} ms")
    except Exception as e:
        results["graph_build"] = {"error": str(e)}
        print(f"        ERROR: {e}")

    print("  [3/4] search latency ...")
    try:
        results["search"] = benchmark_search()
        print(f"        median={results['search']['median_ms']:.2f} ms  "
              f"p95={results['search']['p95_ms']:.2f} ms")
    except Exception as e:
        results["search"] = {"error": str(e)}
        print(f"        ERROR: {e}")

    print("  [4/4] top3 latency ...")
    try:
        results["top3"] = benchmark_top3()
        print(f"        median={results['top3']['median_ms']:.2f} ms  "
              f"p95={results['top3']['p95_ms']:.2f} ms")
    except Exception as e:
        results["top3"] = {"error": str(e)}
        print(f"        ERROR: {e}")

    print("  [coverage] candidate coverage ...")
    cov = evaluate_candidate_coverage()
    results["candidate_coverage"] = cov
    print(f"        coverage={cov.get('candidate_coverage', 'N/A'):.3f}  "
          f"({cov.get('n_hits', 0)}/{cov.get('n_cases', 0)} cases)")

    return {"benchmark": "latency", "n_runs": N_RUNS, "modules": results}


if __name__ == "__main__":
    print("=== Latency & Coverage Benchmark ===")
    result = run_latency_benchmark()
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "latency_results.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"\n✓ Results saved: {out}")
