"""
POST-FIX verification script for demo_case_new_FIR_01204.txt
Verifies all three bugs are resolved and runs regression checks.

Run from d:/Projects/Criminal_detection/:
    python verify_hinglish_fix.py
"""

import sys
import os
import re

sys.path.insert(0, str(os.path.dirname(__file__)))

DOC_PATH = "demo_case_new_FIR_01204.txt"
DOC_ID   = "demo_case_new_FIR_01204"

# ---------- Garbage fragments that MUST NOT appear ----------
KNOWN_GARBAGE = {
    "naam ke", "vyakti dwara chalaya ja raha", "humein", "jiska naam",
    "wapas mil", "milkar ek scheme", "maine", "na", "lagaya",
    "milkar ek scheme mein", "maine 2", "naam ke vyakti",
    "dwara chalaya ja raha", "jiska naam shakti growth fund",
    "lagaya tha jiska naam", "hume", "kti growth fund",
}

print("=" * 70)
print("POST-FIX VERIFICATION: demo_case_new_FIR_01204.txt")
print("=" * 70)

doc_text = open(DOC_PATH, encoding="utf-8").read()

from M1.extractor import extract_all
from M1.normalizer import normalize_entities
from M1.resolver import EntityStore, resolve_entities, deduplicate_cross_type
from M1.schema import build_relationships, resolved_to_entity

store = EntityStore()
raw_entities = extract_all(DOC_ID, doc_text)
normed = normalize_entities(raw_entities)
resolved_batch = resolve_entities(normed, store)

# Cross-type dedup
removed = deduplicate_cross_type(store, case_id=DOC_ID)
if removed:
    print(f"\n[Bug 2] Cross-type dedup removed {len(removed)} entity(s): {removed}")

print(f"\n--- RESOLVED ENTITIES ({len(store.all())}) ---")
entity_names = set()
for e in store.all():
    entity_names.add(e.canonical.lower())
    print(f"  [{e.entity_type:15s} conf={e.confidence:.2f}] {repr(e.canonical)}")
    if len(e.aliases) > 1:
        print(f"       aliases: {e.aliases}")

print()

# ── BUG 1: Garbage fragment check ───────────────────────────────────────────
print("=" * 70)
print("BUG 1 CHECK — Garbage Hinglish fragments")
print("=" * 70)
garbage_found = []
for g in KNOWN_GARBAGE:
    if g in entity_names:
        garbage_found.append(g)

if garbage_found:
    print(f"  ❌ FAIL — {len(garbage_found)} garbage fragment(s) still present:")
    for g in garbage_found:
        print(f"       {repr(g)}")
else:
    print(f"  ✅ PASS — None of the {len(KNOWN_GARBAGE)} known garbage fragments appear.")

# ── BUG 2: Duplicate person check ────────────────────────────────────────────
print()
print("=" * 70)
print("BUG 2 CHECK — Duplicate entity nodes (Vikram Oberoi, Manish Oberoi)")
print("=" * 70)

def count_entity_by_name(store, name):
    name_lower = name.lower()
    return [e for e in store.all() if name_lower in e.canonical.lower() or
            any(name_lower in a.lower() for a in e.aliases)]

vikram_nodes = count_entity_by_name(store, "Vikram Oberoi")
manish_nodes = count_entity_by_name(store, "Manish Oberoi")

print(f"\n  Vikram Oberoi appears as {len(vikram_nodes)} node(s):")
for e in vikram_nodes:
    print(f"    [{e.entity_type}] {e.entity_id} conf={e.confidence:.2f}")

print(f"\n  Manish Oberoi appears as {len(manish_nodes)} node(s):")
for e in manish_nodes:
    print(f"    [{e.entity_type}] {e.entity_id} conf={e.confidence:.2f}")

if len(vikram_nodes) == 1 and len(manish_nodes) == 1:
    print("\n  ✅ PASS — Each person appears exactly once.")
else:
    print("\n  ❌ FAIL — Duplicate nodes detected.")

# ── BUG 3: Entity/relationship density ───────────────────────────────────────
print()
print("=" * 70)
print("BUG 3 CHECK — Graph density (entities and relationships)")
print("=" * 70)

all_entities = [resolved_to_entity(e) for e in store.all()]
doc_entity_map = {DOC_ID: [e.entity_id for e in store.all()]}
doc_type_map   = {DOC_ID: "FIR"}
relationships = build_relationships(all_entities, doc_entity_map, doc_type_map)

n_ents = len(all_entities)
n_rels = len(relationships)
ratio  = n_rels / n_ents if n_ents else 0

print(f"\n  Entities:      {n_ents}")
print(f"  Relationships: {n_rels}")
print(f"  Ratio:         {ratio:.1f} edges/entity")

if n_ents <= 20 and ratio <= 10:
    print("\n  ✅ PASS — Graph density is within acceptable limits.")
elif n_ents <= 30 and ratio <= 15:
    print("\n  ⚠️  MARGINAL — Density reduced but could still be improved.")
else:
    print(f"\n  ❌ FAIL — Density still anomalous (was 741 rels / 39 ents before fix).")

# ── Complete entity list ──────────────────────────────────────────────────────
print()
print("=" * 70)
print("COMPLETE ENTITY LIST (for verification report)")
print("=" * 70)
for e in sorted(store.all(), key=lambda x: (x.entity_type, x.canonical)):
    print(f"  {e.entity_type:15s} | {e.canonical:40s} | conf={e.confidence:.2f} | method=?")

print()
print("=" * 70)
print("REGRESSION TEST 1 — English-only synthetic FIR (dummy_test_case.txt)")
print("=" * 70)

def run_on_doc(path, doc_id):
    text = open(path, encoding="utf-8").read()
    store2 = EntityStore()
    raw = extract_all(doc_id, text)
    normed2 = normalize_entities(raw)
    resolve_entities(normed2, store2)
    deduplicate_cross_type(store2, case_id=doc_id)
    return store2

try:
    store_en = run_on_doc("dummy_test_case.txt", "dummy_test_case")
    en_ents = store_en.all()
    print(f"  Entities: {len(en_ents)}")
    for e in en_ents:
        print(f"    [{e.entity_type}] {repr(e.canonical)}")
    print("  ✅ English FIR processed without crash.")
except Exception as ex:
    print(f"  ❌ ERROR: {ex}")

print()
print("=" * 70)
print("REGRESSION TEST 2 — Existing Hinglish FIR (demo_case_historical_FIR_00778.txt)")
print("=" * 70)

try:
    store_hi = run_on_doc("demo_case_historical_FIR_00778.txt", "FIR_2026_00778")
    hi_ents = store_hi.all()
    print(f"  Entities: {len(hi_ents)}")
    for e in hi_ents:
        print(f"    [{e.entity_type}] {repr(e.canonical)}")
    print("  ✅ Historical FIR processed without crash.")
except Exception as ex:
    print(f"  ❌ ERROR: {ex}")

print()
print("VERIFICATION COMPLETE.")
