import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from M1.resolver import EntityStore, resolve_entities, NormalizedEntity

def _make_norm(text, etype, doc_id):
    return NormalizedEntity(
        normalized_text=text,
        entity_type=etype,
        source_doc_ids=[doc_id],
        raw_aliases={text},
        extraction_methods={"spacy"},
        raw_confidence=0.9,
    )

store = EntityStore()

# 1. Add "Ravi Kumar" in the first real case
doc1 = "FIR_2026_00931"
resolve_entities([_make_norm("Ravi Kumar", "PERSON", doc1)], store)

# 2. Add "Ravi Kumar" in the second real case
doc2 = "FIR_2026_00812"
resolve_entities([_make_norm("Ravi Kumar", "PERSON", doc2)], store)

# Print the results
entities = [e for e in store.all() if e.entity_type == "PERSON"]
print(f"Total PERSON entities: {len(entities)}")
for e in entities:
    print(f"ID: {e.entity_id}")
    print(f"Name: {e.canonical}")
    print(f"Source Docs: {e.source_docs}")
