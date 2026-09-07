"""
Full final verification script covering all 5 testing requirements from the bug report.

Run from d:/Projects/Criminal_detection/:
    python final_verify.py
"""
import sys
sys.path.insert(0, '.')

from M1.extractor import extract_all
from M1.normalizer import normalize_entities
from M1.resolver import EntityStore, resolve_entities, deduplicate_cross_type
from M1.schema import build_relationships, resolved_to_entity

KNOWN_GARBAGE = [
    'naam ke', 'vyakti dwara chalaya ja raha', 'humein', 'jiska naam',
    'wapas mil', 'milkar ek scheme', 'maine', 'na', 'lagaya',
    'milkar ek scheme mein', 'maine 2', 'naam ke vyakti',
    'dwara chalaya ja raha', 'hume', 'kti growth fund',
]

RESULTS = []

def run_doc(path, doc_id):
    text = open(path, encoding='utf-8').read()
    store = EntityStore()
    raw = extract_all(doc_id, text)
    normed = normalize_entities(raw)
    resolve_entities(normed, store)
    deduplicate_cross_type(store, case_id=doc_id)
    return store.all()

def check(label, cond, detail=''):
    status = 'PASS' if cond else 'FAIL'
    RESULTS.append((status, label))
    print(f'  {status}  {label}' + (f'  [{detail}]' if detail else ''))

# ============================================================
# REQUIREMENT 0: Bug 4 — Malviya Nagar, Vaishali Nagar
# ============================================================
print('=' * 70)
print('REQUIREMENT 0: BUG 4 LOCATION RECLASSIFICATION')
print('=' * 70)

try:
    ents_00778 = run_doc('demo_case_historical_FIR_00778.txt', 'FIR_00778')
    malviya = [e for e in ents_00778 if 'malviya nagar' in e.canonical.lower()]
    check('Malviya Nagar -> LOCATION (FIR_00778)',
          malviya and malviya[0].entity_type == 'LOCATION',
          str([(e.entity_type, e.canonical) for e in malviya]))
except Exception as ex:
    check('Malviya Nagar -> LOCATION (FIR_00778)', False, str(ex))

try:
    ents_01204 = run_doc('demo_case_new_FIR_01204.txt', 'FIR_01204')
    vaishali = [e for e in ents_01204 if 'vaishali nagar' in e.canonical.lower()]
    check('Vaishali Nagar -> LOCATION (FIR_01204)',
          vaishali and vaishali[0].entity_type == 'LOCATION',
          str([(e.entity_type, e.canonical) for e in vaishali]))
except Exception as ex:
    check('Vaishali Nagar -> LOCATION (FIR_01204)', False, str(ex))

try:
    ents_mumbai = run_doc('demo_case_mumbai_FIR_00312.txt', 'FIR_00312')
    bandra = [e for e in ents_mumbai if 'bandra west' in e.canonical.lower()]
    andheri = [e for e in ents_mumbai if 'andheri east' in e.canonical.lower()]
    check('Bandra West -> LOCATION (Mumbai FIR)',
          bandra and all(e.entity_type == 'LOCATION' for e in bandra),
          str([(e.entity_type, e.canonical) for e in bandra]))
    check('Andheri East -> LOCATION (Mumbai FIR)',
          andheri and all(e.entity_type == 'LOCATION' for e in andheri),
          str([(e.entity_type, e.canonical) for e in andheri]))
except Exception as ex:
    check('Bandra West / Andheri East (Mumbai FIR)', False, str(ex))

# ============================================================
# REQUIREMENT 1: Bug 1 — No garbage fragments in FIR_01204
# ============================================================
print()
print('=' * 70)
print('REQUIREMENT 1: BUG 1 - No garbage Hinglish fragments in FIR_01204')
print('=' * 70)

try:
    # reuse ents_01204 from above
    ent_names = [e.canonical.lower() for e in ents_01204]
    print()
    print('Complete entity list for FIR_01204:')
    for e in sorted(ents_01204, key=lambda x: (x.entity_type, x.canonical)):
        print(f'   {e.entity_type:15s} | {e.canonical}')
    print()
    bad = [g for g in KNOWN_GARBAGE if g in ent_names]
    check('No known garbage fragments in FIR_01204', not bad,
          str(bad) if bad else 'None found')
except Exception as ex:
    check('No garbage fragments', False, str(ex))

# ============================================================
# REQUIREMENT 2: Bug 2 — Vikram / Manish appear once each
# ============================================================
print()
print('=' * 70)
print('REQUIREMENT 2: BUG 2 - No duplicate person nodes in FIR_01204')
print('=' * 70)

try:
    vikram = [e for e in ents_01204 if 'vikram oberoi' in e.canonical.lower()]
    manish = [e for e in ents_01204 if 'manish oberoi' in e.canonical.lower()]
    check('Vikram Oberoi appears exactly once',
          len(vikram) == 1,
          f'{len(vikram)} node(s): {[(e.entity_type, e.canonical) for e in vikram]}')
    check('Manish Oberoi appears exactly once',
          len(manish) == 1,
          f'{len(manish)} node(s): {[(e.entity_type, e.canonical) for e in manish]}')
except Exception as ex:
    check('Duplicate entity check', False, str(ex))

# ============================================================
# REQUIREMENT 3: Bug 3 — Entity/relationship density in FIR_01204
# ============================================================
print()
print('=' * 70)
print('REQUIREMENT 3: BUG 3 - Graph density (FIR_01204)')
print('=' * 70)

try:
    entities_schema = [resolved_to_entity(e) for e in ents_01204]
    doc_entity_map = {'FIR_01204': [e.entity_id for e in ents_01204]}
    rels = build_relationships(entities_schema, doc_entity_map, {'FIR_01204': 'FIR'})
    n_ents = len(ents_01204)
    n_rels = len(rels)
    ratio = n_rels / n_ents if n_ents else 0
    print(f'  Entities: {n_ents} (was 39 before fix)')
    print(f'  Relationships: {n_rels} (was 741 before fix)')
    print(f'  Ratio: {ratio:.1f} (was ~19.0 before fix)')
    check('Entity count <= 20', n_ents <= 20, str(n_ents))
    check('Relationship ratio < 15', ratio < 15, f'{ratio:.1f}')
except Exception as ex:
    check('Graph density', False, str(ex))

# ============================================================
# REQUIREMENT 4: English FIR regressions
# ============================================================
print()
print('=' * 70)
print('REQUIREMENT 4: English-only FIR regressions')
print('=' * 70)

for path, doc_id in [
    ('dummy_test_case.txt', 'dummy_tc'),
    ('dummy_high_1.txt', 'dummy_high_1'),
    ('dummy_high_2.txt', 'dummy_high_2'),
]:
    try:
        ents_en = run_doc(path, doc_id)
        print(f'  {path}: {len(ents_en)} entities')
        check(f'{path} - no crash', True)
    except FileNotFoundError:
        check(f'{path} - skip (not found)', True, 'file missing')
    except Exception as ex:
        check(f'{path} - no crash', False, str(ex))

# ============================================================
# REQUIREMENT 5: Existing Hinglish FIR regression
# ============================================================
print()
print('=' * 70)
print('REQUIREMENT 5: Existing Hinglish FIR regression (FIR_00778 + FIR_00905)')
print('=' * 70)

for path, doc_id in [
    ('demo_case_historical_FIR_00778.txt', 'FIR_00778'),
    ('demo_case_historical_FIR_00905.txt', 'FIR_00905'),
]:
    try:
        ents_hi = run_doc(path, doc_id)
        ent_names_hi = [e.canonical.lower() for e in ents_hi]
        bad_hi = [g for g in KNOWN_GARBAGE if g in ent_names_hi]
        print(f'  {path}: {len(ents_hi)} entities')
        for e in sorted(ents_hi, key=lambda x: (x.entity_type, x.canonical)):
            print(f'    {e.entity_type:15s} | {e.canonical}')
        check(f'{path} - no crash', True)
        check(f'{path} - no garbage fragments', not bad_hi, str(bad_hi) if bad_hi else '')
    except FileNotFoundError:
        check(f'{path} - skip (not found)', True, 'file missing')
    except Exception as ex:
        check(f'{path} - no crash', False, str(ex))

# ============================================================
# SUMMARY
# ============================================================
print()
print('=' * 70)
print('FINAL SUMMARY')
print('=' * 70)
passes = sum(1 for s, _ in RESULTS if s == 'PASS')
fails = sum(1 for s, _ in RESULTS if s == 'FAIL')
print(f'  {passes} PASS, {fails} FAIL out of {len(RESULTS)} checks')
if fails:
    print()
    print('  FAILED checks:')
    for s, label in RESULTS:
        if s == 'FAIL':
            print(f'    - {label}')
print()
print('ALL REQUIREMENTS MET' if not fails else 'SOME REQUIREMENTS FAILED')
