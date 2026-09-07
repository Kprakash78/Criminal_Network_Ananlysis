"""
Diagnostic script — captures raw NER output on demo_case_new_FIR_01204.txt
BEFORE any fixes, to document the root causes of Bug 1, 2, 3.

Run from d:/Projects/Criminal_detection/:
    python diag_hinglish.py
"""

import sys
import os
import re

# Make sure M1 is importable
sys.path.insert(0, str(os.path.dirname(__file__)))

# ─── HINGLISH PARAGRAPH (lines 16-20 of the document) ───────────────────────
HINGLISH_PARA = """\
Maine apne kuch dosto ke saath milkar ek scheme mein paisa lagaya tha
jiska naam "Shakti Growth Fund" bataya gaya, jo Vikram Oberoi naam ke
vyakti dwara chalaya ja raha tha. Humein bola gaya ki 3 mahine mein
double return milega. Maine 2 lakh rupaye diye the, lekin ab na to paisa
wapas mil raha hai aur na hi Vikram Oberoi phone utha raha hai."""

DOC_PATH = "demo_case_new_FIR_01204.txt"
DOC_ID   = "demo_case_new_FIR_01204"

# ════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 1 — Language-routing check: does _is_hinglish() fire?")
print("=" * 70)

from M1.extractor import _is_hinglish
result = _is_hinglish(HINGLISH_PARA)
print(f"_is_hinglish(HINGLISH_PARA) = {result}")
print()

# ════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 2 — Raw spaCy NER output on HINGLISH paragraph (no filtering)")
print("=" * 70)

import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp(HINGLISH_PARA)

print(f"Entities found by spaCy en_core_web_sm: {len(doc.ents)}")
for ent in doc.ents:
    print(f"  [{ent.label_:12s}] {repr(ent.text)}")
print()

# ════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 3 — extract_entities_spacy on HINGLISH paragraph (with current filtering)")
print("=" * 70)

from M1.extractor import extract_entities_spacy
spacy_ents = extract_entities_spacy(DOC_ID, HINGLISH_PARA)
print(f"extract_entities_spacy returned {len(spacy_ents)} entities:")
for e in spacy_ents:
    print(f"  [{e.entity_type:15s} conf={e.raw_confidence:.2f}] {repr(e.text)}")
print()

# ════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 4 — IndicBERT/multilingual NER on HINGLISH paragraph")
print("=" * 70)

from M1.extractor import _get_indicbert_pipeline
pipe = _get_indicbert_pipeline()
if pipe:
    print("IndicBERT pipeline loaded successfully.")
    try:
        raw_output = pipe(HINGLISH_PARA)
        print(f"Raw IndicBERT output ({len(raw_output)} spans):")
        for span in raw_output:
            print(f"  {span}")
    except Exception as e:
        print(f"IndicBERT inference error: {e}")
else:
    print("IndicBERT pipeline NOT available (None/False) — falling back to spaCy only.")
print()

# ════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 5 — extract_entities_indicbert on HINGLISH paragraph (with current filtering)")
print("=" * 70)

from M1.extractor import extract_entities_indicbert
indicbert_ents = extract_entities_indicbert(DOC_ID, HINGLISH_PARA)
print(f"extract_entities_indicbert returned {len(indicbert_ents)} entities:")
for e in indicbert_ents:
    print(f"  [{e.entity_type:15s} conf={e.raw_confidence:.2f}] {repr(e.text)}")
print()

# ════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 6 — Full document: doc_structure classification + per-block NER routing")
print("=" * 70)

doc_text = open(DOC_PATH, encoding="utf-8").read()

from M1.doc_structure import classify_document_structure
blocks = classify_document_structure(doc_text)
print(f"Total blocks: {len(blocks)}")
for b in blocks:
    print(f"  [{b.type:20s}] lines {b.line_numbers[0]:3d}-{b.line_numbers[1]:3d} | {repr(b.raw_text[:80])}")
print()

# ════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 7 — Full extract_all on FIR_01204 (current, unfixed pipeline)")
print("=" * 70)

from M1.extractor import extract_all
from M1.normalizer import normalize_entities
from M1.resolver import EntityStore, resolve_entities

all_raw = extract_all(DOC_ID, doc_text)
print(f"\nRaw entities ({len(all_raw)}):")
for e in all_raw:
    print(f"  [{e.entity_type:15s} method={e.extraction_method:10s} conf={e.raw_confidence:.2f}] {repr(e.text)}")

normed = normalize_entities(all_raw)
print(f"\nNormalized entities ({len(normed)}):")
for e in normed:
    print(f"  [{e.entity_type:15s} conf={e.raw_confidence:.2f}] {repr(e.normalized_text)}")

store = EntityStore()
resolved = resolve_entities(normed, store)
print(f"\nResolved entities ({len(resolved)}):")
for e in store.all():
    print(f"  [{e.entity_type:15s} conf={e.confidence:.2f} id={e.entity_id}] {repr(e.canonical)}")
    if len(e.aliases) > 1:
        print(f"       aliases: {e.aliases}")

# ════════════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("STEP 8 — Duplicate entity check: Vikram Oberoi / Manish Oberoi")
print("=" * 70)

targets = {"vikram oberoi", "manish oberoi"}
for e in store.all():
    if e.canonical.lower() in targets or any(a.lower() in targets for a in e.aliases):
        print(f"  entity_id={e.entity_id}")
        print(f"  type={e.entity_type}")
        print(f"  canonical={repr(e.canonical)}")
        print(f"  aliases={e.aliases}")
        print(f"  source_docs={e.source_docs}")
        print(f"  confidence={e.confidence}")
        print()

# ════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 9 — Relationship count estimate for this single document")
print("=" * 70)

n_ents = len(store.all())
max_edges = n_ents * (n_ents - 1) // 2
print(f"  Entities: {n_ents}")
print(f"  Max APPEARS_IN_CASE edges if all co-occur: {max_edges}")

print()
print("DIAGNOSTIC COMPLETE.")
