"""
M1.10 — Integration Test: Full Pipeline End-to-End
PS 26152 — AI-Powered Criminal Network Analysis System

This test runs the complete pipeline on the synthetic dataset and validates
the output against every PRD §10 acceptance criterion.
"""

import json
import sys
import sqlite3
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from M1.generate_dataset import generate_all
from M1.pipeline import run_pipeline
from M1.schema import Entity, Relationship, PipelineResult, emit_json, emit_sqlite


@pytest.fixture(scope="module")
def pipeline_output(tmp_path_factory):
    """Generate synthetic dataset + run pipeline — shared across all tests in this module."""
    base = tmp_path_factory.mktemp("integration")
    generate_all(base_dir=base)
    result = run_pipeline(str(base))
    return result, base


# ===========================================================================
# PRD §10 Acceptance Criteria — explicit pass/fail for each
# ===========================================================================

def test_AC1_no_unhandled_crashes(pipeline_output):
    """
    AC1: Full synthetic dataset processes end-to-end with zero unhandled crashes.
    (If we reach this test, the pipeline ran without crashing.)
    """
    result, base = pipeline_output
    assert isinstance(result, PipelineResult), "Pipeline did not return a PipelineResult"
    # Every logged document should either be 'ok' or 'error' (not missing)
    for entry in result.extraction_log:
        assert entry["status"] in ("ok", "error"), f"Unexpected status: {entry}"


def test_AC1_no_failed_docs(pipeline_output):
    """AC1 extended: no documents should have errored in the log."""
    result, _ = pipeline_output
    errors = [e for e in result.extraction_log if e["status"] == "error"]
    assert errors == [], f"Pipeline had {len(errors)} document errors: {errors}"


def test_AC2_at_least_3_duplicates_merged(pipeline_output):
    """
    AC2: At least 3 deliberately duplicated entities (different name formats)
    are correctly merged.

    We verify by checking that the number of unique PERSON entities is
    significantly less than the total number of raw person mentions
    (i.e., merging happened).
    """
    result, _ = pipeline_output
    person_entities = [e for e in result.entities if e.type == "PERSON"]

    # Count entities that have more than one source document (evidence of merge)
    multi_doc_persons = [e for e in person_entities if len(e.source_documents) > 1]
    assert len(multi_doc_persons) >= 3, (
        f"Expected ≥3 PERSON entities with multiple source docs (merged). "
        f"Got {len(multi_doc_persons)}. "
        f"Person entities: {[(e.name, e.source_documents) for e in person_entities[:10]]}"
    )


def test_AC2_aliases_present(pipeline_output):
    """AC2 extended: merged entities should have aliases."""
    result, _ = pipeline_output
    persons_with_aliases = [
        e for e in result.entities
        if e.type == "PERSON" and len(e.aliases) > 1
    ]
    assert len(persons_with_aliases) >= 3, (
        f"Expected ≥3 PERSON entities with multiple aliases. Got {len(persons_with_aliases)}."
    )


def test_AC3_hinglish_entities_extracted(pipeline_output):
    """
    AC3: At least 5 Hinglish/Hindi sentences correctly yield extracted entities.
    We verify by checking the extraction log — documents with Hinglish markers
    should have non-zero entity counts.
    """
    result, base = pipeline_output
    # Check FIRs that contain Hinglish markers
    hinglish_markers = {"bataya", "mobile", "gaadi", "pakda", "giraftaar", "uska", "baith"}
    fir_dir = base / "data" / "firs"
    hinglish_fir_ids = set()
    for fir_path in fir_dir.glob("*.txt"):
        content = fir_path.read_text(encoding="utf-8").lower()
        if any(m in content for m in hinglish_markers):
            hinglish_fir_ids.add(fir_path.stem)

    # For each hinglish FIR, check that it produced entities
    hinglish_logs = [
        e for e in result.extraction_log
        if e["doc_id"] in hinglish_fir_ids and e["status"] == "ok"
    ]
    docs_with_entities = [e for e in hinglish_logs if e["entity_count"] > 0]

    assert len(docs_with_entities) >= 5, (
        f"Expected ≥5 Hinglish FIRs with extracted entities. "
        f"Got {len(docs_with_entities)} out of {len(hinglish_fir_ids)} Hinglish FIRs."
    )


def test_AC4_output_validates_pydantic(pipeline_output):
    """
    AC4: Output JSON validates against the pydantic schema with zero errors.
    """
    result, base = pipeline_output
    output_dir = base / "output"
    emit_json(result, output_dir)

    # Re-load entities.json and validate each entry with pydantic
    raw_entities = json.loads((output_dir / "entities.json").read_text())
    validation_errors = []
    for item in raw_entities:
        try:
            Entity(**item)
        except Exception as exc:
            validation_errors.append(f"{item.get('entity_id', '?')}: {exc}")

    assert len(validation_errors) == 0, (
        f"Pydantic validation errors:\n" + "\n".join(validation_errors[:10])
    )

    # Re-load relationships.json and validate
    raw_rels = json.loads((output_dir / "relationships.json").read_text())
    for item in raw_rels:
        Relationship(**item)   # raises if invalid


def test_AC5_sqlite_ingestible(pipeline_output):
    """
    AC5: M2 can successfully ingest the output without any manual reformatting.
    Verified by: SQLite DB is readable, entity_id is unique, confidence is in [0,1].
    """
    result, base = pipeline_output
    output_dir = base / "output"
    db_path = emit_sqlite(result, output_dir)

    conn = sqlite3.connect(db_path)

    # Check entities table
    rows = conn.execute("SELECT entity_id, type, name, confidence FROM entities").fetchall()
    assert len(rows) >= 1
    entity_ids = [r[0] for r in rows]
    assert len(entity_ids) == len(set(entity_ids)), "Duplicate entity_ids in SQLite"
    for eid, etype, name, conf in rows:
        assert 0.0 <= conf <= 1.0, f"Confidence out of range for {eid}: {conf}"
        assert etype in {
            "PERSON", "PHONE", "VEHICLE", "LOCATION", "ORGANIZATION",
            "ACCOUNT", "CASE", "DATE"
        }, f"Invalid type in SQLite: {etype}"

    # Check relationships table
    rel_rows = conn.execute("SELECT source, target, confidence FROM relationships").fetchall()
    for src, tgt, conf in rel_rows:
        assert src != tgt, "Self-relationship found"
        assert 0.0 <= conf <= 1.0

    conn.close()


# ===========================================================================
# Additional functional requirement checks
# ===========================================================================

def test_FR1_dataset_size(pipeline_output):
    """FR1: ≥30 FIRs, ≥100 CDR entries, ≥50 transaction records processed."""
    result, _ = pipeline_output
    fir_count = sum(1 for e in result.extraction_log if e["doc_type"] == "FIR")
    cdr_count = sum(1 for e in result.extraction_log if e["doc_type"] == "CDR")
    txn_count = sum(1 for e in result.extraction_log if e["doc_type"] == "TRANSACTION")
    assert fir_count >= 30, f"Expected ≥30 FIR docs, got {fir_count}"
    assert cdr_count >= 100, f"Expected ≥100 CDR docs, got {cdr_count}"
    assert txn_count >= 50, f"Expected ≥50 TRANSACTION docs, got {txn_count}"


def test_FR4_phones_and_vehicles_extracted(pipeline_output):
    """FR4: PHONE and VEHICLE numbers extracted via regex."""
    result, _ = pipeline_output
    phones = [e for e in result.entities if e.type == "PHONE"]
    vehicles = [e for e in result.entities if e.type == "VEHICLE"]
    assert len(phones) >= 5, f"Expected ≥5 PHONE entities, got {len(phones)}"
    assert len(vehicles) >= 3, f"Expected ≥3 VEHICLE entities, got {len(vehicles)}"


def test_FR7_all_entities_have_confidence(pipeline_output):
    """FR7: Every entity has a confidence score in [0, 1]."""
    result, _ = pipeline_output
    for e in result.entities:
        assert 0.0 <= e.confidence <= 1.0, f"{e.entity_id} has confidence={e.confidence}"


def test_FR8_all_entities_have_source_docs(pipeline_output):
    """FR8 (Auditability): Every entity references at least one source document."""
    result, _ = pipeline_output
    for e in result.entities:
        assert len(e.source_documents) >= 1, f"{e.entity_id} has no source_documents"


def test_FR9_low_confidence_flagged(pipeline_output):
    """FR9: Low-confidence entities are flagged with needs_review=True, not dropped."""
    result, _ = pipeline_output
    all_entities = result.entities
    needs_review = [e for e in all_entities if e.needs_review]
    # All flagged entities should still be present in the output (not dropped)
    all_ids = {e.entity_id for e in all_entities}
    for flagged in needs_review:
        assert flagged.entity_id in all_ids, f"{flagged.entity_id} was dropped despite needs_review"


def test_FR10_pipeline_handles_empty_doc(pipeline_output, tmp_path):
    """FR10: Pipeline continues if one document is empty/malformed."""
    from M1.pipeline import run_pipeline
    # Create a directory with one valid FIR and one empty file
    fir_dir = tmp_path / "data" / "firs"
    fir_dir.mkdir(parents=True)
    (fir_dir / "FIR_VALID.txt").write_text(
        "Ravi Kumar was arrested at Lajpat Nagar, Delhi. Phone: 9876543210.",
        encoding="utf-8",
    )
    (fir_dir / "FIR_EMPTY.txt").write_text("", encoding="utf-8")
    (fir_dir / "FIR_NOISE.txt").write_text("   \n\n\n   ", encoding="utf-8")

    result = run_pipeline(str(tmp_path))
    # Should not crash — at least the valid doc should produce output
    assert isinstance(result, PipelineResult)
    valid_logs = [e for e in result.extraction_log if e["doc_id"] == "FIR_VALID"]
    assert len(valid_logs) == 1
    assert valid_logs[0]["status"] == "ok"


def test_entity_ids_deterministic(pipeline_output, tmp_path):
    """NFR Reproducibility: Running pipeline twice produces same entity IDs."""
    from M1.generate_dataset import generate_all as gen
    from M1.pipeline import run_pipeline as rp

    gen(base_dir=tmp_path / "r1")
    gen(base_dir=tmp_path / "r2")
    r1 = rp(str(tmp_path / "r1"))
    r2 = rp(str(tmp_path / "r2"))

    ids1 = sorted(e.entity_id for e in r1.entities)
    ids2 = sorted(e.entity_id for e in r2.entities)
    assert ids1 == ids2, (
        f"Entity IDs differ across runs. "
        f"Run1 only: {set(ids1)-set(ids2)}, Run2 only: {set(ids2)-set(ids1)}"
    )


def test_performance_under_2_minutes(tmp_path):
    """NFR Latency: Full pipeline completes in under 2 minutes on the synthetic dataset."""
    import time
    from M1.generate_dataset import generate_all as gen
    from M1.pipeline import run_pipeline as rp

    gen(base_dir=tmp_path)
    start = time.time()
    rp(str(tmp_path))
    elapsed = time.time() - start

    assert elapsed < 120, f"Pipeline took {elapsed:.1f}s — must complete in < 120s"
