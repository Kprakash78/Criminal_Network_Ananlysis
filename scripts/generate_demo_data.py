#!/usr/bin/env python3
"""
scripts/generate_demo_data.py
PS 26152 — Criminal Network Analysis System
--------------------------------------------
Creates demo_dataset/ with 3 deterministic synthetic cases:
  - case_A: financial network (FIR + CDR + transactions + ground truth)
  - case_B: location-based network (FIR + CDR + transactions + ground truth)
  - case_C: mixed pattern (FIR + CDR + transactions + ground truth)

Also writes demo_dataset/ground_truth.json with:
  - extracted entity labels (names, phones, dates, amounts)
  - known edges for link-prediction evaluation
  - expected top-3 candidates for coverage evaluation

Fully offline. Deterministic with numpy.random.seed(42).
Safe language: no guilt or accusation — all labels are
"elevated-priority candidates requiring investigator verification".

Usage:
    python scripts/generate_demo_data.py
"""

import csv
import json
import os
import random
import sys
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np

# ─── Paths ─────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = REPO_ROOT / "demo_dataset"

# ─── Deterministic seed ─────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ─── Entity pool (shared with top3.py / generate_demo_outputs.py) ────────────
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

TOWERS = ["TOWER_DEL_01", "TOWER_MUM_02", "TOWER_HYD_03", "TOWER_BLR_04", "TOWER_CHE_05"]
BASE_DATE = datetime(2024, 3, 1, 8, 0, 0)

# ─── Case definitions ─────────────────────────────────────────────────────────
# Each case names its "elevated-priority candidates" (ground truth for top-3)
# and its known edges (ground truth for link prediction).

CASES = {
    "case_A": {
        "title": "FIR-2024-0042: Suspected Financial Irregularities",
        "fir_case_no": "FIR00042",
        # Indices into PERSONS who are the elevated-priority candidates
        "candidate_indices": [7, 5, 1],   # Fatima, Anjali, Sunita
        # Named entities ground truth for extraction eval
        "gt_names":   ["Fatima Begum", "Anjali Singh", "Sunita Sharma", "Ravi Kumar"],
        "gt_phones":  ["9432109876", "9654321098", "9123456789", "9876543210", "9988776655"],
        "gt_dates":   ["01-03-2024", "05-03-2024", "10-03-2024"],
        "gt_amounts": ["1500000", "750000", "2200000"],
        # Known edges (pairs of account numbers that transacted)
        "known_edges": [
            ("ACC00108", "ACC00106"),
            ("ACC00106", "ACC00102"),
            ("ACC00108", "ACC00101"),
        ],
    },
    "case_B": {
        "title": "FIR-2024-0078: Colocation Pattern Analysis",
        "fir_case_no": "FIR00078",
        "candidate_indices": [0, 2, 4],   # Ravi, Mohammed, Deepak
        "gt_names":   ["Ravi Kumar", "Mohammed Iqbal", "Deepak Verma", "Priya Nair"],
        "gt_phones":  ["9876543210", "9988776655", "7766554433", "9123456789", "8877665544"],
        "gt_dates":   ["03-03-2024", "07-03-2024", "12-03-2024"],
        "gt_amounts": ["800000", "920000"],
        "known_edges": [
            ("ACC00101", "ACC00103"),
            ("ACC00103", "ACC00105"),
            ("ACC00101", "ACC00104"),
        ],
    },
    "case_C": {
        "title": "FIR-2024-0091: Mixed Communication and Transaction Pattern",
        "fir_case_no": "FIR00091",
        "candidate_indices": [6, 3, 8],   # Rakesh, Priya, Suresh
        "gt_names":   ["Rakesh Yadav", "Priya Nair", "Suresh Patil", "Kavya Reddy"],
        "gt_phones":  ["9543210987", "8877665544", "9321098765", "9876543210", "9123456789"],
        "gt_dates":   ["06-03-2024", "08-03-2024", "15-03-2024"],
        "gt_amounts": ["340000", "1100000"],
        "known_edges": [
            ("ACC00107", "ACC00104"),
            ("ACC00104", "ACC00109"),
            ("ACC00107", "ACC00110"),
        ],
    },
}


# ─── FIR Text generator ───────────────────────────────────────────────────────
def make_fir(case_id: str, meta: dict) -> str:
    cands = [PERSONS[i] for i in meta["candidate_indices"]]
    # Deterministic extra persons (not candidates)
    extra_idx = [i for i in range(len(PERSONS)) if i not in meta["candidate_indices"]][:2]
    extras = [PERSONS[i] for i in extra_idx]
    all_persons = cands + extras

    date_strs = meta["gt_dates"]
    amounts   = meta["gt_amounts"]
    phones    = [p["phone"] for p in cands]
    vehicles  = [p["vehicle"] for p in cands]
    accounts  = [p["account"] for p in cands]

    lines = [
        f"FIRST INFORMATION REPORT",
        f"Case No.: {meta['fir_case_no']}",
        f"Date of Filing: {date_strs[0]}",
        f"",
        f"SUMMARY",
        f"This report documents findings related to {meta['title']}.",
        f"Investigative analysis identified communication and financial patterns",
        f"that require further verification by assigned officers.",
        f"",
        f"PERSONS OF INTEREST (requires investigator verification):",
    ]
    for p in cands:
        lines.append(f"  - {p['name']} (Phone: {p['phone']}, Vehicle: {p['vehicle']}, Account: {p['account']})")
    lines += [
        f"",
        f"ADDITIONAL WITNESSES / CONTACTS:",
    ]
    for p in extras:
        lines.append(f"  - {p['name']} (Phone: {p['phone']})")
    lines += [
        f"",
        f"TIMELINE OF OBSERVATIONS:",
        f"  {date_strs[0]}: Initial communication pattern detected between {cands[0]['name']} and {cands[1]['name']}.",
        f"  {date_strs[1]}: Financial transfer of Rs. {amounts[0]} observed between accounts {accounts[0]} and {accounts[1]}.",
    ]
    if len(date_strs) > 2:
        extra_amt = amounts[1] if len(amounts) > 1 else "500000"
        lines.append(f"  {date_strs[2]}: Secondary transfer of Rs. {extra_amt} flagged for review.")
    lines += [
        f"",
        f"VEHICLE OBSERVATIONS:",
        f"  Vehicle {vehicles[0]} sighted near Tower {TOWERS[0]} on {date_strs[0]}.",
    ]
    if len(vehicles) > 1:
        lines.append(f"  Vehicle {vehicles[1]} sighted near Tower {TOWERS[1]} on {date_strs[1]}.")
    lines += [
        f"",
        f"NOTE: All findings are preliminary. No determinations of liability have been made.",
        f"This report is for investigative use only.",
    ]
    return "\n".join(lines)


# ─── CDR CSV generator ────────────────────────────────────────────────────────
def make_cdr(case_id: str, meta: dict) -> list[dict]:
    rng = np.random.default_rng(SEED + hash(case_id) % 1000)
    cands = [PERSONS[i] for i in meta["candidate_indices"]]
    phones = [p["phone"] for p in cands]

    rows = []
    for day in range(15):
        dt = BASE_DATE + timedelta(days=day)
        # Elevated-priority candidates call each other frequently
        for _ in range(int(rng.integers(3, 7))):
            caller = phones[int(rng.integers(0, len(phones)))]
            receiver = phones[int(rng.integers(0, len(phones)))]
            if caller == receiver:
                continue
            call_dt = dt + timedelta(hours=int(rng.integers(0, 23)),
                                     minutes=int(rng.integers(0, 59)))
            hour = call_dt.hour
            duration = int(rng.integers(30, 600))
            tower = TOWERS[int(rng.integers(0, len(TOWERS)))]
            rows.append({
                "timestamp": call_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "caller": caller,
                "receiver": receiver,
                "duration_sec": duration,
                "tower_id": tower,
                "hour": hour,
            })
    return rows


# ─── Transaction CSV generator ────────────────────────────────────────────────
def make_transactions(case_id: str, meta: dict) -> list[dict]:
    rng = np.random.default_rng(SEED + hash(case_id) % 999 + 1)
    cands = [PERSONS[i] for i in meta["candidate_indices"]]
    accounts = [p["account"] for p in cands]
    edge_pairs = meta["known_edges"]
    edge_set = set(tuple(e) for e in edge_pairs)

    rows = []
    # Primary known edges — several transactions each
    for edge in edge_pairs:
        for _ in range(int(rng.integers(4, 8))):
            dt = BASE_DATE + timedelta(days=int(rng.integers(0, 15)),
                                       hours=int(rng.integers(8, 22)))
            amount = float(rng.integers(100000, 2500000))
            rows.append({
                "timestamp": dt.strftime("%Y-%m-%d %H:%M:%S"),
                "sender_account": edge[0],
                "receiver_account": edge[1],
                "amount_inr": amount,
                "transaction_id": f"TXN_{case_id}_{len(rows):04d}",
                "flagged": 1 if amount > 1000000 else 0,
            })

    # Lateral intra-candidate edges (creates triangles for common-neighbour scoring)
    for i in range(len(accounts)):
        for j in range(len(accounts)):
            if i != j and (accounts[i], accounts[j]) not in edge_set:
                for _ in range(int(rng.integers(2, 4))):
                    dt = BASE_DATE + timedelta(days=int(rng.integers(0, 15)))
                    amount = float(rng.integers(30000, 300000))
                    rows.append({
                        "timestamp": dt.strftime("%Y-%m-%d %H:%M:%S"),
                        "sender_account": accounts[i],
                        "receiver_account": accounts[j],
                        "amount_inr": amount,
                        "transaction_id": f"TXN_{case_id}_LAT_{len(rows):04d}",
                        "flagged": 0,
                    })

    # Noise transactions between non-candidates (low-amount, irrelevant)
    noise_accounts = [p["account"] for p in PERSONS
                      if p["account"] not in accounts][:3]
    for _ in range(5):
        if len(noise_accounts) < 2:
            break
        a, b = noise_accounts[:2]
        dt = BASE_DATE + timedelta(days=int(rng.integers(0, 15)))
        amount = float(rng.integers(5000, 50000))
        rows.append({
            "timestamp": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "sender_account": a,
            "receiver_account": b,
            "amount_inr": amount,
            "transaction_id": f"TXN_{case_id}_NOISE_{len(rows):04d}",
            "flagged": 0,
        })
    return rows


# ─── Write helpers ────────────────────────────────────────────────────────────
def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    fir_dir = DEMO_DIR / "fir_files"
    cdr_dir = DEMO_DIR / "cdr_files"
    txn_dir = DEMO_DIR / "transaction_files"
    fir_dir.mkdir(exist_ok=True)
    cdr_dir.mkdir(exist_ok=True)
    txn_dir.mkdir(exist_ok=True)

    ground_truth = {}

    for case_id, meta in CASES.items():
        print(f"  Generating {case_id} ...")
        # FIR
        fir_text = make_fir(case_id, meta)
        fir_path = fir_dir / f"{case_id}.txt"
        fir_path.write_text(fir_text, encoding="utf-8")

        # CDR
        cdr_rows = make_cdr(case_id, meta)
        write_csv(cdr_dir / f"{case_id}_cdr.csv", cdr_rows)

        # Transactions
        txn_rows = make_transactions(case_id, meta)
        write_csv(txn_dir / f"{case_id}_transactions.csv", txn_rows)

        # Ground truth for this case
        cands = [PERSONS[i] for i in meta["candidate_indices"]]
        ground_truth[case_id] = {
            "title": meta["title"],
            "extraction": {
                "names":   meta["gt_names"],
                "phones":  meta["gt_phones"],
                "dates":   meta["gt_dates"],
                "amounts": meta["gt_amounts"],
            },
            "top3_candidates": [p["name"] for p in cands],
            "known_edges": meta["known_edges"],
            "fir_file": str(fir_path.relative_to(REPO_ROOT)),
            "cdr_file": str((cdr_dir / f"{case_id}_cdr.csv").relative_to(REPO_ROOT)),
            "txn_file": str((txn_dir / f"{case_id}_transactions.csv").relative_to(REPO_ROOT)),
        }

    write_json(DEMO_DIR / "ground_truth.json", ground_truth)
    print(f"\n✓ Demo data written to: {DEMO_DIR}")
    print(f"  Cases: {list(CASES.keys())}")
    print(f"  Ground truth: {DEMO_DIR / 'ground_truth.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
