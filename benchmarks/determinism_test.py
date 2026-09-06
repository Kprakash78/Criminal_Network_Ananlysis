#!/usr/bin/env python3
"""
benchmarks/determinism_test.py
PS 26152 — Criminal Network Analysis System
--------------------------------------------
Verifies that pipeline outputs are bit-for-bit identical to the cached
demo_cache/ outputs. This ensures the demo is reproducible on any machine.

Method:
  1. Re-run the parsing + top-3 pipeline for each demo case.
  2. Serialise outputs to JSON (sorted keys, 2-space indent).
  3. Compare SHA-256 hashes against demo_cache/cache_manifest.json.

Result: PASS if all hashes match, FAIL otherwise (with diff details).

Fully offline. Deterministic with SEED=42.

Usage:
    python benchmarks/determinism_test.py
    # or called by scripts/run_benchmarks.sh
"""

import csv
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT   = Path(__file__).resolve().parent.parent
DEMO_DIR    = REPO_ROOT / "demo_dataset"
CACHE_DIR   = REPO_ROOT / "demo_cache"
RESULTS_DIR = REPO_ROOT / "results"

SEED = 42
np.random.seed(SEED)

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

_PHONE   = re.compile(r"\b(\d{10})\b")
_DATE    = re.compile(r"\b(\d{2}-\d{2}-\d{4})\b")
_AMOUNT  = re.compile(r"Rs\.\s*([\d,]+)", re.IGNORECASE)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _to_bytes(data) -> bytes:
    """Canonical JSON serialization (sorted keys, 2-space indent)."""
    return json.dumps(data, indent=2, sort_keys=True).encode("utf-8")


def extract_entities(fir_text: str) -> dict:
    entities: dict = {}
    for match in _PHONE.finditer(fir_text):
        phone = match.group(1)
        person = PHONE_TO_PERSON.get(phone)
        if person:
            eid = f"person_{person['name'].lower().replace(' ', '_')}"
            if eid not in entities:
                entities[eid] = {
                    "entity_id": eid, "type": "PERSON",
                    "name": person["name"], "phone": phone,
                    "confidence": 0.90, "extraction_method": "regex_phone"
                }
    for match in re.finditer(r"\b(ACC\d{5})\b", fir_text):
        acct = match.group(1)
        person = ACCOUNT_TO_PERSON.get(acct)
        if person:
            eid = f"person_{person['name'].lower().replace(' ', '_')}"
            if eid not in entities:
                entities[eid] = {
                    "entity_id": eid, "type": "PERSON",
                    "name": person["name"], "account": acct,
                    "confidence": 0.85, "extraction_method": "regex_account"
                }
    dates   = _DATE.findall(fir_text)
    amounts = [m.replace(",", "") for m in _AMOUNT.findall(fir_text)]
    return {"persons": list(entities.values()), "dates": dates, "amounts": amounts}


def compute_top3(persons: list[dict], cdr_rows: list[dict],
                 txn_rows: list[dict], rng_seed: int = SEED) -> list[dict]:
    np.random.seed(rng_seed)
    scores = {}
    for p in persons:
        phone   = p.get("phone", "")
        account = p.get("account", "")
        name    = p["name"]
        contacts = {r["receiver"] for r in cdr_rows if r.get("caller") == phone}
        contacts |= {r["caller"] for r in cdr_rows if r.get("receiver") == phone}
        degree_score = min(1.0, len(contacts) / max(len(persons), 1))
        calls = [r for r in cdr_rows if r.get("caller") == phone or r.get("receiver") == phone]
        late_night = sum(1 for r in calls if int(r.get("hour", 12)) >= 23 or int(r.get("hour", 12)) <= 4)
        temporal_score = late_night / max(len(calls), 1)
        sent  = sum(float(r["amount_inr"]) for r in txn_rows if r.get("sender_account") == account)
        recv  = sum(float(r["amount_inr"]) for r in txn_rows if r.get("receiver_account") == account)
        txn_score = min(1.0, (sent + recv) / 5_000_000)
        risk = 0.30 * degree_score + 0.25 * temporal_score + 0.45 * txn_score
        scores[name] = {
            "name": name,
            "risk_score": round(risk * 100, 2),
            "degree_score": round(degree_score, 4),
            "temporal_score": round(temporal_score, 4),
            "transaction_score": round(txn_score, 4),
            "label": "elevated-priority candidate — requires investigator verification",
        }
    sorted_scores = sorted(scores.values(), key=lambda x: x["risk_score"], reverse=True)
    return sorted_scores[:3]


def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_determinism_test() -> dict:
    manifest_path = CACHE_DIR / "cache_manifest.json"
    if not manifest_path.exists():
        return {
            "pass": False,
            "error": "cache_manifest.json not found — run prepare_demo_cache.py first",
            "files": {},
        }

    manifest = json.loads(manifest_path.read_text())
    expected_hashes = manifest.get("files", {})

    gt_path = DEMO_DIR / "ground_truth.json"
    if not gt_path.exists():
        return {
            "pass": False,
            "error": "ground_truth.json not found — run generate_demo_data.py first",
            "files": {},
        }
    gt_all = json.loads(gt_path.read_text())

    file_results: dict = {}
    all_pass = True

    for case_id, gt in gt_all.items():
        fir_file = REPO_ROOT / gt["fir_file"]
        cdr_file = REPO_ROOT / gt["cdr_file"]
        txn_file = REPO_ROOT / gt["txn_file"]

        fir_text = fir_file.read_text(encoding="utf-8") if fir_file.exists() else ""
        cdr_rows = load_csv(cdr_file)
        txn_rows = load_csv(txn_file)
        entities = extract_entities(fir_text)

        # Test entities file
        ents_key = f"{case_id}_entities.json"
        ents_bytes = _to_bytes(entities)
        ents_hash  = _sha256(ents_bytes)
        expected   = expected_hashes.get(ents_key, "MISSING")
        ents_pass  = (ents_hash == expected)
        if not ents_pass:
            all_pass = False
        file_results[ents_key] = {
            "expected_sha256": expected,
            "actual_sha256":   ents_hash,
            "pass": ents_pass,
        }

        # Test top3 file
        rng_seed = SEED + hash(case_id) % 100
        top3 = compute_top3(entities["persons"], cdr_rows, txn_rows, rng_seed=rng_seed)
        top3_data  = {"case_id": case_id, "top3": top3}
        top3_key   = f"{case_id}_top3.json"
        top3_bytes = _to_bytes(top3_data)
        top3_hash  = _sha256(top3_bytes)
        expected_t = expected_hashes.get(top3_key, "MISSING")
        top3_pass  = (top3_hash == expected_t)
        if not top3_pass:
            all_pass = False
        file_results[top3_key] = {
            "expected_sha256": expected_t,
            "actual_sha256":   top3_hash,
            "pass": top3_pass,
        }

    # Check search index — compare hash of cached file directly
    search_key  = "search_index.json"
    search_path = CACHE_DIR / "search_index.json"
    if search_path.exists():
        cached_bytes = search_path.read_bytes()
        # The cached file IS the reference; verify it hasn't changed on disk
        cached_hash  = _sha256(cached_bytes)
        expected_s   = expected_hashes.get(search_key, "MISSING")
        search_pass  = (cached_hash == expected_s)
        if not search_pass:
            all_pass = False
        file_results[search_key] = {
            "expected_sha256": expected_s,
            "actual_sha256":   cached_hash,
            "pass": search_pass,
        }
    else:
        file_results[search_key] = {"pass": False, "error": "file not found"}
        all_pass = False

    return {
        "pass": all_pass,
        "n_files_checked": len(file_results),
        "n_passed": sum(1 for v in file_results.values() if v.get("pass")),
        "files": file_results,
    }


if __name__ == "__main__":
    print("=== Determinism Test ===")
    result = run_determinism_test()
    n = result["n_files_checked"]
    p = result["n_passed"]

    if result["pass"]:
        print(f"  ✓ ALL {p}/{n} files match cached outputs (bit-for-bit).")
    else:
        print(f"  ✗ {p}/{n} files match. Failures:")
        for fname, info in result["files"].items():
            if not info.get("pass"):
                print(f"    - {fname}: got {info.get('actual_sha256', 'N/A')[:16]}…  "
                      f"expected {info.get('expected_sha256', 'N/A')[:16]}…")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "determinism_results.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"\n✓ Results saved: {out}")
    sys.exit(0 if result["pass"] else 1)
