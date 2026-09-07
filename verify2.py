import sys
sys.path.insert(0, '.')
DOC_PATH = 'demo_case_new_FIR_01204.txt'
DOC_ID = 'demo_case_new_FIR_01204'
KNOWN_GARBAGE = [
    'naam ke', 'vyakti dwara chalaya ja raha', 'humein', 'jiska naam',
    'wapas mil', 'milkar ek scheme', 'maine', 'na', 'lagaya',
    'milkar ek scheme mein', 'maine 2', 'naam ke vyakti',
    'dwara chalaya ja raha', 'hume', 'kti growth fund',
]
doc_text = open(DOC_PATH, encoding='utf-8').read()
from M1.extractor import extract_all
from M1.normalizer import normalize_entities
from M1.resolver import EntityStore, resolve_entities, deduplicate_cross_type
from M1.schema import build_relationships, resolved_to_entity
store = EntityStore()
raw_entities = extract_all(DOC_ID, doc_text)
normed = normalize_entities(raw_entities)
resolve_entities(normed, store)
removed = deduplicate_cross_type(store, case_id=DOC_ID)
print('Cross-type dedup removed:', len(removed), 'entities')
all_ents = store.all()
print('TOTAL ENTITIES:', len(all_ents))
print()
entity_names_lower = [e.canonical.lower() for e in all_ents]
print('-- All entities --')
for e in sorted(all_ents, key=lambda x: (x.entity_type, x.canonical)):
    print('  ', e.entity_type.ljust(15), '|', e.canonical)
print()
bad = [g for g in KNOWN_GARBAGE if g in entity_names_lower]
print('BUG 1:', 'FAIL: ' + str(bad) if bad else 'PASS - no garbage fragments')
vikram = [e for e in all_ents if 'vikram oberoi' in e.canonical.lower()]
manish = [e for e in all_ents if 'manish oberoi' in e.canonical.lower()]
result2 = 'PASS' if len(vikram)==1 and len(manish)==1 else 'FAIL'
print('BUG 2: Vikram=' + str(len(vikram)) + ', Manish=' + str(len(manish)) + ' nodes ->', result2)
entities_schema = [resolved_to_entity(e) for e in all_ents]
doc_entity_map = {DOC_ID: [e.entity_id for e in all_ents]}
rels = build_relationships(entities_schema, doc_entity_map, {DOC_ID: 'FIR'})
ratio = len(rels)/len(all_ents) if all_ents else 0
result3 = 'PASS' if ratio < 15 else 'FAIL'
print('BUG 3:', len(all_ents), 'ents,', len(rels), 'rels, ratio=', round(ratio, 1), '->', result3)
print()
print('=== REGRESSION: dummy_test_case.txt ===')
try:
    store2 = EntityStore()
    t2 = open('dummy_test_case.txt', encoding='utf-8').read()
    raw2 = extract_all('dummy_tc', t2)
    normed2 = normalize_entities(raw2)
    resolve_entities(normed2, store2)
    deduplicate_cross_type(store2, case_id='dummy_tc')
    e2 = store2.all()
    print('Entities:', len(e2))
    for e in sorted(e2, key=lambda x: (x.entity_type, x.canonical)):
        print('  ', e.entity_type.ljust(15), '|', e.canonical)
    print('PASS - English FIR processed OK')
except Exception as ex:
    print('FAIL:', ex)
print()
print('=== REGRESSION: demo_case_historical_FIR_00778.txt ===')
try:
    store3 = EntityStore()
    t3 = open('demo_case_historical_FIR_00778.txt', encoding='utf-8').read()
    raw3 = extract_all('FIR_00778', t3)
    normed3 = normalize_entities(raw3)
    resolve_entities(normed3, store3)
    deduplicate_cross_type(store3, case_id='FIR_00778')
    e3 = store3.all()
    print('Entities:', len(e3))
    for e in sorted(e3, key=lambda x: (x.entity_type, x.canonical)):
        print('  ', e.entity_type.ljust(15), '|', e.canonical)
    print('PASS - Historical FIR processed OK')
except Exception as ex:
    print('FAIL:', ex)
print()
print('=== REGRESSION: demo_case_historical_FIR_00905.txt ===')
try:
    store4 = EntityStore()
    t4 = open('demo_case_historical_FIR_00905.txt', encoding='utf-8').read()
    raw4 = extract_all('FIR_00905', t4)
    normed4 = normalize_entities(raw4)
    resolve_entities(normed4, store4)
    deduplicate_cross_type(store4, case_id='FIR_00905')
    e4 = store4.all()
    print('Entities:', len(e4))
    for e in sorted(e4, key=lambda x: (x.entity_type, x.canonical)):
        print('  ', e.entity_type.ljust(15), '|', e.canonical)
    print('PASS - FIR_00905 processed OK')
except Exception as ex:
    print('FAIL:', ex)
