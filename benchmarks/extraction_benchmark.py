#!/usr/bin/env python3
"""
benchmarks/extraction_benchmark.py
PS 26152 — Criminal Network Analysis System
--------------------------------------------
Evaluates the entity extraction pipeline on demo cases.

Metrics: precision, recall, F1 per entity type (names, phones, dates, amounts).
Method: exact substring / token matching against ground-truth labels in
        demo_dataset/ground_truth.json.

Fully offline. Deterministic with SEED=42.

Usage:
    python benchmarks/extraction_benchmark.py
    # or called by scripts/run_benchmarks.sh
"""

import json
import re
import sys
from pathlib import Path
from typing import Tuple

REPO_ROOT  = Path(__file__).resolve().parent.parent
DEMO_DIR   = REPO_ROOT / "demo_dataset"
CACHE_DIR  = REPO_ROOT / "demo_cache"
RESULTS_DIR = REPO_ROOT / "results"

# Regex patterns (same as prepare_demo_cache.py)
_PHONE   = re.compile(r"\b(\d{10})\b")
_DATE    = re.compile(r"\b(\d{2}-\d{2}-\d{4})\b")
_AMOUNT  = re.compile(r"Rs\.\s*([\d,]+)", re.IGNORECASE)

# Persons list (for name extraction via known pool)
PERSONS = [
    "Ravi Kumar", "Sunita Sharma", "Mohammed Iqbal", "Priya Nair",
    "Deepak Verma", "Anjali Singh", "Rakesh Yadav", "Fatima Begum",
    "Suresh Patil", "Kavya Reddy",
]

# Aliases known in FIR text
NAME_ALIASES = {
    "Ravi Kumar":     ["Ravi K.", "R. Kumar", "Ravi"],
    "Sunita Sharma":  ["S. Sharma", "Sunita S."],
    "Mohammed Iqbal": ["M. Iqbal", "Mohd Iqbal", "Iqbal"],
    "Priya Nair":     ["P. Nair", "Priya N."],
    "Deepak Verma":   ["D. Verma", "Deepak V.", "Deepak"],
    "Anjali Singh":   ["A. Singh", "Anjali S."],
    "Rakesh Yadav":   ["R. Yadav", "Rakesh Y."],
    "Fatima Begum":   ["F. Begum", "Fatima B."],
    "Suresh Patil":   ["S. Patil", "Suresh P."],
    "Kavya Reddy":    ["K. Reddy", "Kavya R."],
}


def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """Compute precision, recall, F1."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return round(precision, 4), round(recall, 4), round(f1, 4)


def extract_names(text: str) -> set:
    """Extract known person names (and aliases) from text."""
    found = set()
    text_lower = text.lower()
    for name in PERSONS:
        all_forms = [name] + NAME_ALIASES.get(name, [])
        for form in all_forms:
            if form.lower() in text_lower:
                found.add(name)
                break
    return found


def extract_phones(text: str) -> set:
    return set(_PHONE.findall(text))


def extract_dates(text: str) -> set:
    return set(_DATE.findall(text))


def extract_amounts(text: str) -> set:
    return set(m.replace(",", "") for m in _AMOUNT.findall(text))


def evaluate_case(case_id: str, fir_text: str, gt: dict) -> dict:
    """Run extraction on one FIR, compare to ground-truth."""
    pred_names   = extract_names(fir_text)
    pred_phones  = extract_phones(fir_text)
    pred_dates   = extract_dates(fir_text)
    pred_amounts = extract_amounts(fir_text)

    gt_names   = set(gt["extraction"]["names"])
    gt_phones  = set(gt["extraction"]["phones"])
    gt_dates   = set(gt["extraction"]["dates"])
    gt_amounts = set(gt["extraction"]["amounts"])

    metrics = {}
    for label, pred, truth in [
        ("names",   pred_names,   gt_names),
        ("phones",  pred_phones,  gt_phones),
        ("dates",   pred_dates,   gt_dates),
        ("amounts", pred_amounts, gt_amounts),
    ]:
        tp = len(pred & truth)
        fp = len(pred - truth)
        fn = len(truth - pred)
        p, r, f1 = prf(tp, fp, fn)
        metrics[label] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": p, "recall": r, "f1": f1,
            "predicted": sorted(pred),
            "ground_truth": sorted(truth),
        }

    return {"case_id": case_id, "metrics": metrics}


def aggregate_metrics(case_results: list[dict]) -> dict:
    """Macro-average per entity type across cases."""
    totals: dict = {}
    for res in case_results:
        for etype, m in res["metrics"].items():
            if etype not in totals:
                totals[etype] = {"precision": [], "recall": [], "f1": []}
            totals[etype]["precision"].append(m["precision"])
            totals[etype]["recall"].append(m["recall"])
            totals[etype]["f1"].append(m["f1"])

    avg = {}
    for etype, vals in totals.items():
        n = len(vals["precision"])
        avg[etype] = {
            "macro_precision": round(sum(vals["precision"]) / n, 4),
            "macro_recall":    round(sum(vals["recall"])    / n, 4),
            "macro_f1":        round(sum(vals["f1"])        / n, 4),
        }
    return avg


def run_extraction_benchmark() -> dict:
    gt_path = DEMO_DIR / "ground_truth.json"
    if not gt_path.exists():
        print("ERROR: demo_dataset/ground_truth.json not found.")
        print("  Run: python scripts/generate_demo_data.py first.")
        sys.exit(1)

    gt_all = json.loads(gt_path.read_text())
    fir_dir = DEMO_DIR / "fir_files"

    case_results = []
    for case_id, gt in gt_all.items():
        fir_file = REPO_ROOT / gt["fir_file"]
        if not fir_file.exists():
            print(f"  WARNING: FIR file missing for {case_id}: {fir_file}")
            continue
        fir_text = fir_file.read_text(encoding="utf-8", errors="replace")
        result = evaluate_case(case_id, fir_text, gt)
        case_results.append(result)
        print(f"  {case_id}: "
              f"names F1={result['metrics']['names']['f1']:.3f}  "
              f"phones F1={result['metrics']['phones']['f1']:.3f}  "
              f"dates F1={result['metrics']['dates']['f1']:.3f}  "
              f"amounts F1={result['metrics']['amounts']['f1']:.3f}")

    aggregated = aggregate_metrics(case_results)
    return {
        "benchmark": "extraction",
        "n_cases": len(case_results),
        "per_case": case_results,
        "aggregate": aggregated,
    }


if __name__ == "__main__":
    print("=== Extraction Benchmark ===")
    result = run_extraction_benchmark()
    print("\n--- Aggregate (macro-average) ---")
    for etype, m in result["aggregate"].items():
        print(f"  {etype:8s}: P={m['macro_precision']:.3f}  "
              f"R={m['macro_recall']:.3f}  F1={m['macro_f1']:.3f}")
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "extraction_results.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"\n✓ Results saved: {out}")
