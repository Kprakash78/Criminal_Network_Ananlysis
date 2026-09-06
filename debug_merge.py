import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from M1.resolver import EntityStore, resolve_entities, _fuzzy_match, _make_entity_id, ResolvedEntity
from M1.normalizer import NormalizedEntity

def _make_norm(text, etype, docs=None, conf=0.75, methods=None):
    return NormalizedEntity(
        normalized_text=text,
        entity_type=etype,
        source_doc_ids=docs or ["DOC1"],
        raw_aliases={text},
        extraction_methods=methods or {"spacy"},
        raw_confidence=conf,
    )

from M1.generate_dataset import PERSONS
store = EntityStore()
merge_count = 0

for p in PERSONS:
    canonical = p[0]
    aliases = p[1]
    print(f"\n--- Testing {canonical} ---")
    
    for alias in aliases[:2]:
        match = _fuzzy_match(alias, [canonical], 80, "PERSON")
        print(f"Trying to merge: {alias} into {canonical} -> Match: {match}")
