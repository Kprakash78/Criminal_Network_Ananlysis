#!/usr/bin/env python3
"""
scripts/prepare_demo_cache.py
PS 26152 — Criminal Network Analysis System
--------------------------------------------
Runs the full parsing pipeline on demo_dataset/ and caches outputs in
demo_cache/. This makes the demo instant (no ML model loading on every run).

Cached artefacts (all JSON, bit-for-bit deterministic with SEED=42):
  demo_cache/case_A_entities.json
  demo_cache/case_A_top3.json
  demo_cache/case_B_entities.json
  demo_cache/case_B_top3.json
  demo_cache/case_C_entities.json
  demo_cache/case_C_top3.json
  demo_cache/search_index.json        ← lightweight TF-IDF token index
  demo_cache/cache_manifest.json      ← SHA-256 hashes for determinism_test

Usage:
    python scripts/prepare_demo_cache.py
    # Re-run after any change to demo data or pipeline to refresh cache.

Offline: uses only regex + numpy. No spaCy / torch / sentence-transformers.
"""

import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR  = REPO_ROOT / "demo_dataset"
CACHE_DIR = REPO_ROOT / "demo_cache"

SEED = 42
np.random.seed(SEED)

# Shared entity pool (kept in sync with generate_demo_data.py / top3.py)
PERSONS = [
    {"name": "Ravi Kumar",     "phone": "9876543210", "vehicle": "DL01AB1234", "account": "ACC00101"},
    {"name": "Sunita Sharma",  "phone": "9123456789", "vehicle": "MH12CD5678", "account": "ACC00102"},
    {"name": "Mohammed Iqbal", "phone": "9988776655", "vehicle": "UP32EF9012", "account": "ACC00103"},
    {"name": "Priya Nair",     "phone": "8877665544", "vehicle": "KA05GH3456", "account": "ACC00104"},
    {"name": "Deepak Verma",   "phone": "7766554433", "vehicle": "RJ14IJ7890", "account": "ACC00105"},
    {"name": "Anjali Singh",   "phone": "9654321098", "vehicle": "GJ01KL2345", "account": "ACC00106"},
    {"name": "Rakesh Yadav",   "phone": "9543210987", "vehicle": "HR26MN6789", "account": "ACC00107"},
    {"name": "Fatima Begum",   "phone": "9432109876", "vehicle": "TN09OP0123", "account": "ACC00108"},
    {"name": "Suresh Patil",   "phone": "9321098765", "vehicle": "MP07QR4567", "account": "ACC00109"},
    {"name": "Kavya Reddy",    "phone": "9210987654", "vehicle": "AP28ST8901", "account": "ACC00110"},
]

PHONE_TO_PERSON   = {p["phone"]: p   for p in PERSONS}
ACCOUNT_TO_PERSON = {p["account"]: p for p in PERSONS}
NAME_LOWER        = {p["name"].lower(): p for p in PERSONS}

# Regex patterns
_PHONE   = re.compile(r"\b(\d{10})\b")
_VEHICLE = re.compile(r"\b([A-Z]{2}\d{2}[A-Z]{2}\d{4})\b")
_ACCOUNT = re.compile(r"\b(ACC\d{5})\b")
_DATE    = re.compile(r"\b(\d{2}-\d{2}-\d{4})\b")
_AMOUNT  = re.compile(r"Rs\.\s*([\d,]+)", re.IGNORECASE)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def extract_entities_from_fir(fir_text: str) -> dict:
    """Lightweight regex-based entity extraction."""
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

    for match in _ACCOUNT.finditer(fir_text):
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

    dates = _DATE.findall(fir_text)
    amounts = [m.replace(",", "") for m in _AMOUNT.findall(fir_text)]

    return {"persons": list(entities.values()), "dates": dates, "amounts": amounts}


def load_cdr(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_txn(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def compute_top3(persons: list[dict], cdr_rows: list[dict],
                 txn_rows: list[dict], rng_seed: int = SEED) -> list[dict]:
    """
    Lightweight deterministic top-3 scoring (mirrors M3_feature/top3.py logic).
    Uses only call counts and transaction amounts — no ML required.
    Returns sorted list (highest score first) with safe language.
    """
    np.random.seed(rng_seed)
    scores = {}

    for p in persons:
        phone   = p.get("phone", "")
        account = p.get("account", "")
        name    = p["name"]

        # Degree: number of unique contacts
        contacts_out = {r["receiver"] for r in cdr_rows if r.get("caller") == phone}
        contacts_in  = {r["caller"]  for r in cdr_rows if r.get("receiver") == phone}
        degree_score = min(1.0, (len(contacts_out | contacts_in)) / max(len(persons), 1))

        # Temporal: fraction of late-night calls (23:00-05:00)
        calls = [r for r in cdr_rows if r.get("caller") == phone or r.get("receiver") == phone]
        late_night = sum(1 for r in calls if int(r.get("hour", 12)) >= 23 or int(r.get("hour", 12)) <= 4)
        temporal_score = (late_night / max(len(calls), 1))

        # Transaction: total sent amount normalised
        sent  = sum(float(r["amount_inr"]) for r in txn_rows if r.get("sender_account") == account)
        recv  = sum(float(r["amount_inr"]) for r in txn_rows if r.get("receiver_account") == account)
        total = sent + recv
        txn_score = min(1.0, total / 5_000_000)

        # Weighted composite
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


def build_search_index(fir_dir: Path) -> dict:
    """Build a lightweight TF-IDF-style token index from FIR files."""
    index: dict = {"docs": {}, "inv_index": defaultdict(list)}
    stopwords = {"the", "a", "an", "is", "in", "of", "and", "to", "for",
                 "are", "at", "by", "on", "be", "this", "that", "with",
                 "has", "was", "all", "from", "been", "have", "it"}

    doc_term_counts: dict = {}
    for fir_file in sorted(fir_dir.glob("*.txt")):
        doc_id = fir_file.stem
        text = fir_file.read_text(encoding="utf-8", errors="replace")
        tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
        tokens = [t for t in tokens if t not in stopwords and len(t) > 2]
        counts = Counter(tokens)
        doc_term_counts[doc_id] = counts
        index["docs"][doc_id] = {
            "file": str(fir_file.relative_to(REPO_ROOT)),
            "n_tokens": len(tokens),
        }

    # Build inverted index (token → [(doc_id, tf)])
    inv: dict = defaultdict(list)
    for doc_id, counts in doc_term_counts.items():
        total = sum(counts.values())
        for token, cnt in counts.items():
            inv[token].append({"doc": doc_id, "tf": round(cnt / total, 6)})
    index["inv_index"] = dict(inv)
    return index


def write_json(path: Path, data) -> str:
    """Write JSON and return SHA-256 of the file bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(data, indent=2, sort_keys=True).encode("utf-8")
    with open(path, "wb") as f:
        f.write(content)
    return _sha256(content)


def main() -> int:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    gt_path = DEMO_DIR / "ground_truth.json"
    if not gt_path.exists():
        print("ERROR: demo_dataset/ground_truth.json not found.")
        print("  Run: python scripts/generate_demo_data.py first.")
        return 1

    gt = json.loads(gt_path.read_text())
    manifest: dict = {"seed": SEED, "files": {}}

    fir_dir = DEMO_DIR / "fir_files"
    cdr_dir = DEMO_DIR / "cdr_files"
    txn_dir = DEMO_DIR / "transaction_files"

    for case_id, meta in gt.items():
        print(f"  Caching {case_id} ...")
        fir_file = REPO_ROOT / meta["fir_file"]
        cdr_file = REPO_ROOT / meta["cdr_file"]
        txn_file = REPO_ROOT / meta["txn_file"]

        fir_text  = fir_file.read_text(encoding="utf-8") if fir_file.exists() else ""
        cdr_rows  = load_cdr(cdr_file)
        txn_rows  = load_txn(txn_file)
        entities  = extract_entities_from_fir(fir_text)

        # Cache entities
        ents_path = CACHE_DIR / f"{case_id}_entities.json"
        h = write_json(ents_path, entities)
        manifest["files"][f"{case_id}_entities.json"] = h

        # Cache top3
        top3 = compute_top3(entities["persons"], cdr_rows, txn_rows,
                             rng_seed=SEED + hash(case_id) % 100)
        top3_path = CACHE_DIR / f"{case_id}_top3.json"
        h = write_json(top3_path, {"case_id": case_id, "top3": top3})
        manifest["files"][f"{case_id}_top3.json"] = h

    # Cache search index
    if fir_dir.exists():
        index = build_search_index(fir_dir)
        h = write_json(CACHE_DIR / "search_index.json", index)
        manifest["files"]["search_index.json"] = h

    # Write manifest
    manifest_path = CACHE_DIR / "cache_manifest.json"
    write_json(manifest_path, manifest)
    print(f"\n✓ Demo cache written to: {CACHE_DIR}")
    print(f"  Manifest: {manifest_path}")
    print(f"  Files cached: {len(manifest['files'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
