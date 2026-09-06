"""
Demo Dataset — Lightweight Output Generator
PS 26152 — AI-Powered Criminal Network Analysis System

Generates M1-compatible entities.json and relationships.json from the existing
synthetic data using regex-based extraction. This allows the demo to run without
heavy ML dependencies (spaCy, transformers).

Usage:
    python demo_dataset/generate_demo_outputs.py

Outputs:
    demo_dataset/M1_output/entities.json
    demo_dataset/M1_output/relationships.json
"""

import csv
import json
import os
import re
import sys
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve paths relative to repo root
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "M1_output"

# Known entity pool (from M1/generate_dataset.py — keep in sync)
PERSONS = [
    {"name": "Ravi Kumar",     "aliases": ["Ravi K.", "R. Kumar", "Ravi"],     "phone": "9876543210", "vehicle": "DL01AB1234", "account": "ACC00101"},
    {"name": "Sunita Sharma",  "aliases": ["S. Sharma", "Sunita S."],          "phone": "9123456789", "vehicle": "MH12CD5678", "account": "ACC00102"},
    {"name": "Mohammed Iqbal", "aliases": ["M. Iqbal", "Mohd Iqbal", "Iqbal"],"phone": "9988776655", "vehicle": "UP32EF9012", "account": "ACC00103"},
    {"name": "Priya Nair",     "aliases": ["P. Nair", "Priya N."],            "phone": "8877665544", "vehicle": "KA05GH3456", "account": "ACC00104"},
    {"name": "Deepak Verma",   "aliases": ["D. Verma", "Deepak V.", "Deepak"],"phone": "7766554433", "vehicle": "RJ14IJ7890", "account": "ACC00105"},
    {"name": "Anjali Singh",   "aliases": ["A. Singh", "Anjali S."],           "phone": "9654321098", "vehicle": "GJ01KL2345", "account": "ACC00106"},
    {"name": "Rakesh Yadav",   "aliases": ["R. Yadav", "Rakesh Y."],           "phone": "9543210987", "vehicle": "HR26MN6789", "account": "ACC00107"},
    {"name": "Fatima Begum",   "aliases": ["F. Begum", "Fatima B."],           "phone": "9432109876", "vehicle": "TN09OP0123", "account": "ACC00108"},
    {"name": "Suresh Patil",   "aliases": ["S. Patil", "Suresh P."],           "phone": "9321098765", "vehicle": "MP07QR4567", "account": "ACC00109"},
    {"name": "Kavya Reddy",    "aliases": ["K. Reddy", "Kavya R."],            "phone": "9210987654", "vehicle": "AP28ST8901", "account": "ACC00110"},
]

# Build lookup indices
PHONE_TO_PERSON = {}
ACCOUNT_TO_PERSON = {}
NAME_VARIANTS = {}  # lowercased name/alias -> person dict

for p in PERSONS:
    PHONE_TO_PERSON[p["phone"]] = p
    ACCOUNT_TO_PERSON[p["account"]] = p
    NAME_VARIANTS[p["name"].lower()] = p
    for alias in p["aliases"]:
        NAME_VARIANTS[alias.lower()] = p


def make_entity_id(entity_type: str, canonical: str) -> str:
    """Deterministic entity ID from type + canonical name."""
    clean = re.sub(r"[^a-zA-Z0-9]", "_", canonical).strip("_").lower()
    return f"{entity_type.lower()}_{clean}"


def extract_entities_from_firs(fir_dir: Path) -> tuple[list[dict], dict[str, list[str]]]:
    """
    Regex-based entity extraction from FIR text files.
    Returns (entities_list, doc_entity_map).
    """
    entities = {}  # entity_id -> entity dict
    doc_entity_map = defaultdict(list)  # doc_id -> [entity_ids]

    phone_pattern = re.compile(r"\b(\d{10})\b")
    vehicle_pattern = re.compile(r"\b([A-Z]{2}\d{2}[A-Z]{2}\d{4})\b")
    account_pattern = re.compile(r"\b(ACC\d{5})\b")
    date_pattern = re.compile(r"\b(\d{2}-\d{2}-\d{4})\b")
    case_pattern = re.compile(r"Case No\.?:\s*(FIR\d+)", re.IGNORECASE)

    for fir_file in sorted(fir_dir.glob("*.txt")):
        doc_id = fir_file.stem
        text = fir_file.read_text(encoding="utf-8", errors="replace")

        # Extract phones
        for match in phone_pattern.finditer(text):
            phone = match.group(1)
            person = PHONE_TO_PERSON.get(phone)
            if person:
                eid = make_entity_id("PERSON", person["name"])
                if eid not in entities:
                    entities[eid] = {
                        "entity_id": eid,
                        "type": "PERSON",
                        "name": person["name"],
                        "aliases": person["aliases"],
                        "source_documents": [],
                        "confidence": 0.85,
                        "needs_review": False,
                    }
                if doc_id not in entities[eid]["source_documents"]:
                    entities[eid]["source_documents"].append(doc_id)
                if eid not in doc_entity_map[doc_id]:
                    doc_entity_map[doc_id].append(eid)

                # Also add phone as entity
                phone_eid = make_entity_id("PHONE", phone)
                if phone_eid not in entities:
                    entities[phone_eid] = {
                        "entity_id": phone_eid,
                        "type": "PHONE",
                        "name": phone,
                        "aliases": [],
                        "source_documents": [],
                        "confidence": 0.95,
                        "needs_review": False,
                    }
                if doc_id not in entities[phone_eid]["source_documents"]:
                    entities[phone_eid]["source_documents"].append(doc_id)
                if phone_eid not in doc_entity_map[doc_id]:
                    doc_entity_map[doc_id].append(phone_eid)

        # Extract accounts
        for match in account_pattern.finditer(text):
            account = match.group(1)
            acc_eid = make_entity_id("ACCOUNT", account)
            if acc_eid not in entities:
                entities[acc_eid] = {
                    "entity_id": acc_eid,
                    "type": "ACCOUNT",
                    "name": account,
                    "aliases": [],
                    "source_documents": [],
                    "confidence": 0.95,
                    "needs_review": False,
                }
            if doc_id not in entities[acc_eid]["source_documents"]:
                entities[acc_eid]["source_documents"].append(doc_id)
            if acc_eid not in doc_entity_map[doc_id]:
                doc_entity_map[doc_id].append(acc_eid)

        # Extract vehicles
        for match in vehicle_pattern.finditer(text):
            vehicle = match.group(1)
            veh_eid = make_entity_id("VEHICLE", vehicle)
            if veh_eid not in entities:
                entities[veh_eid] = {
                    "entity_id": veh_eid,
                    "type": "VEHICLE",
                    "name": vehicle,
                    "aliases": [],
                    "source_documents": [],
                    "confidence": 0.90,
                    "needs_review": False,
                }
            if doc_id not in entities[veh_eid]["source_documents"]:
                entities[veh_eid]["source_documents"].append(doc_id)
            if veh_eid not in doc_entity_map[doc_id]:
                doc_entity_map[doc_id].append(veh_eid)

        # Extract person names via known name variants
        text_lower = text.lower()
        for variant, person in NAME_VARIANTS.items():
            if variant in text_lower:
                eid = make_entity_id("PERSON", person["name"])
                if eid not in entities:
                    entities[eid] = {
                        "entity_id": eid,
                        "type": "PERSON",
                        "name": person["name"],
                        "aliases": person["aliases"],
                        "source_documents": [],
                        "confidence": 0.80,
                        "needs_review": False,
                    }
                if doc_id not in entities[eid]["source_documents"]:
                    entities[eid]["source_documents"].append(doc_id)
                if eid not in doc_entity_map[doc_id]:
                    doc_entity_map[doc_id].append(eid)

    return list(entities.values()), dict(doc_entity_map)


def extract_entities_from_cdrs(cdr_path: Path) -> tuple[list[dict], dict[str, list[str]]]:
    """Extract phone entities from CDR CSV."""
    entities = {}
    doc_entity_map = defaultdict(list)

    with open(cdr_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=2):  # line 1 is header
            doc_id = f"CDR_L{i}"
            caller = row.get("caller", "").strip()
            callee = row.get("callee", "").strip()

            for phone in [caller, callee]:
                if not phone:
                    continue
                phone_eid = make_entity_id("PHONE", phone)
                if phone_eid not in entities:
                    entities[phone_eid] = {
                        "entity_id": phone_eid,
                        "type": "PHONE",
                        "name": phone,
                        "aliases": [],
                        "source_documents": [],
                        "confidence": 0.95,
                        "needs_review": False,
                    }
                if doc_id not in entities[phone_eid]["source_documents"]:
                    entities[phone_eid]["source_documents"].append(doc_id)
                if phone_eid not in doc_entity_map[doc_id]:
                    doc_entity_map[doc_id].append(phone_eid)

                # Link to person if known
                person = PHONE_TO_PERSON.get(phone)
                if person:
                    eid = make_entity_id("PERSON", person["name"])
                    if eid not in entities:
                        entities[eid] = {
                            "entity_id": eid,
                            "type": "PERSON",
                            "name": person["name"],
                            "aliases": person["aliases"],
                            "source_documents": [],
                            "confidence": 0.85,
                            "needs_review": False,
                        }
                    if doc_id not in entities[eid]["source_documents"]:
                        entities[eid]["source_documents"].append(doc_id)
                    if eid not in doc_entity_map[doc_id]:
                        doc_entity_map[doc_id].append(eid)

    return list(entities.values()), dict(doc_entity_map)


def extract_entities_from_transactions(txn_path: Path) -> tuple[list[dict], dict[str, list[str]]]:
    """Extract account entities from transactions CSV."""
    entities = {}
    doc_entity_map = defaultdict(list)

    with open(txn_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=2):
            doc_id = f"TXN_L{i}"
            source_acc = row.get("source_account", "").strip()
            target_acc = row.get("target_account", "").strip()

            for acc in [source_acc, target_acc]:
                if not acc:
                    continue
                acc_eid = make_entity_id("ACCOUNT", acc)
                if acc_eid not in entities:
                    entities[acc_eid] = {
                        "entity_id": acc_eid,
                        "type": "ACCOUNT",
                        "name": acc,
                        "aliases": [],
                        "source_documents": [],
                        "confidence": 0.95,
                        "needs_review": False,
                    }
                if doc_id not in entities[acc_eid]["source_documents"]:
                    entities[acc_eid]["source_documents"].append(doc_id)
                if acc_eid not in doc_entity_map[doc_id]:
                    doc_entity_map[doc_id].append(acc_eid)

                # Link to person
                person = ACCOUNT_TO_PERSON.get(acc)
                if person:
                    eid = make_entity_id("PERSON", person["name"])
                    if eid not in entities:
                        entities[eid] = {
                            "entity_id": eid,
                            "type": "PERSON",
                            "name": person["name"],
                            "aliases": person["aliases"],
                            "source_documents": [],
                            "confidence": 0.85,
                            "needs_review": False,
                        }
                    if doc_id not in entities[eid]["source_documents"]:
                        entities[eid]["source_documents"].append(doc_id)
                    if eid not in doc_entity_map[doc_id]:
                        doc_entity_map[doc_id].append(eid)

    return list(entities.values()), dict(doc_entity_map)


def merge_entities(entity_lists: list[list[dict]]) -> list[dict]:
    """Merge entity lists, combining source_documents for same entity_id."""
    merged = {}
    for elist in entity_lists:
        for e in elist:
            eid = e["entity_id"]
            if eid in merged:
                for doc in e["source_documents"]:
                    if doc not in merged[eid]["source_documents"]:
                        merged[eid]["source_documents"].append(doc)
                # Use higher confidence
                merged[eid]["confidence"] = max(merged[eid]["confidence"], e["confidence"])
            else:
                merged[eid] = dict(e)
    return list(merged.values())


def build_relationships(entities: list[dict], doc_entity_maps: list[dict]) -> list[dict]:
    """Build co-occurrence relationships from entities appearing in same documents."""
    # Merge all doc_entity_maps
    combined_map = defaultdict(set)
    for dem in doc_entity_maps:
        for doc_id, eids in dem.items():
            combined_map[doc_id].update(eids)

    relationships = []
    seen = set()

    for doc_id, eids in combined_map.items():
        eid_list = sorted(eids)
        for i in range(len(eid_list)):
            for j in range(i + 1, len(eid_list)):
                src, tgt = eid_list[i], eid_list[j]
                key = (src, tgt, doc_id)
                if key in seen:
                    continue
                seen.add(key)

                # Determine relationship type based on entity types
                src_type = None
                tgt_type = None
                for e in entities:
                    if e["entity_id"] == src:
                        src_type = e["type"]
                    if e["entity_id"] == tgt:
                        tgt_type = e["type"]

                rel_type = "CO_OCCURS_WITH"
                if src_type == "PHONE" and tgt_type == "PHONE":
                    rel_type = "CALLED"
                elif src_type == "ACCOUNT" and tgt_type == "ACCOUNT":
                    rel_type = "TRANSFERRED_MONEY_TO"
                elif "PERSON" in (src_type, tgt_type) and "PHONE" in (src_type, tgt_type):
                    rel_type = "USES_PHONE"
                elif "PERSON" in (src_type, tgt_type) and "ACCOUNT" in (src_type, tgt_type):
                    rel_type = "OWNS_ACCOUNT"
                elif "PERSON" in (src_type, tgt_type) and "VEHICLE" in (src_type, tgt_type):
                    rel_type = "DRIVES_VEHICLE"

                relationships.append({
                    "source": src,
                    "target": tgt,
                    "relationship": rel_type,
                    "confidence": 0.80,
                    "source_record": doc_id,
                    "timestamp": "",
                })

    return relationships


def generate():
    """Main generation entry point."""
    print("[DemoGen] Generating demo dataset outputs...")

    fir_dir = DATA_DIR / "firs"
    cdr_path = DATA_DIR / "cdrs" / "cdr.csv"
    txn_path = DATA_DIR / "transactions" / "transactions.csv"

    # Validate input files exist
    if not fir_dir.exists():
        print(f"[DemoGen] ERROR: FIR directory not found: {fir_dir}")
        sys.exit(1)
    if not cdr_path.exists():
        print(f"[DemoGen] ERROR: CDR file not found: {cdr_path}")
        sys.exit(1)
    if not txn_path.exists():
        print(f"[DemoGen] ERROR: Transactions file not found: {txn_path}")
        sys.exit(1)

    # Extract entities from all sources
    fir_entities, fir_doc_map = extract_entities_from_firs(fir_dir)
    cdr_entities, cdr_doc_map = extract_entities_from_cdrs(cdr_path)
    txn_entities, txn_doc_map = extract_entities_from_transactions(txn_path)

    # Merge
    all_entities = merge_entities([fir_entities, cdr_entities, txn_entities])
    all_relationships = build_relationships(
        all_entities, [fir_doc_map, cdr_doc_map, txn_doc_map]
    )

    # Write output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    entities_path = OUTPUT_DIR / "entities.json"
    relationships_path = OUTPUT_DIR / "relationships.json"

    with open(entities_path, "w", encoding="utf-8") as f:
        json.dump(all_entities, f, indent=2, ensure_ascii=False)

    with open(relationships_path, "w", encoding="utf-8") as f:
        json.dump(all_relationships, f, indent=2, ensure_ascii=False)

    print(f"[DemoGen] Generated {len(all_entities)} entities -> {entities_path}")
    print(f"[DemoGen] Generated {len(all_relationships)} relationships -> {relationships_path}")
    print("[DemoGen] Done.")

    return all_entities, all_relationships


if __name__ == "__main__":
    generate()
