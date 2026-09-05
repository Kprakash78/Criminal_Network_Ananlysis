"""
Tests for M2.1 — Graph Loader
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from M2.loader import (
    load_entities,
    load_relationships,
    load_m1_output,
    M1Entity,
    M1Relationship,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def valid_entities_file(tmp_path):
    data = [
        {
            "entity_id": "PER_abc123",
            "type": "PERSON",
            "name": "Ravi Kumar",
            "aliases": ["R. Kumar"],
            "source_documents": ["FIR_001"],
            "confidence": 0.85,
            "needs_review": False,
        },
        {
            "entity_id": "PHN_def456",
            "type": "PHONE",
            "name": "9876543210",
            "aliases": [],
            "source_documents": ["FIR_001"],
            "confidence": 0.98,
            "needs_review": False,
        },
    ]
    path = tmp_path / "entities.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


@pytest.fixture
def valid_relationships_file(tmp_path):
    data = [
        {
            "source": "PER_abc123",
            "target": "PHN_def456",
            "relationship": "APPEARS_IN_SAME_DOCUMENT",
            "timestamp": "2025-01-01T00:00:00",
            "source_record": "FIR_001",
            "confidence": 0.85,
        }
    ]
    path = tmp_path / "relationships.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# M2.1 unit tests
# ---------------------------------------------------------------------------

def test_load_entities_valid(valid_entities_file):
    entities, skipped = load_entities(valid_entities_file)
    assert len(entities) == 2
    assert len(skipped) == 0
    assert entities[0].entity_id == "PER_abc123"
    assert entities[0].type == "PERSON"
    assert entities[1].type == "PHONE"


def test_load_relationships_valid(valid_relationships_file):
    rels, skipped = load_relationships(valid_relationships_file)
    assert len(rels) == 1
    assert len(skipped) == 0
    assert rels[0].source == "PER_abc123"
    assert rels[0].relationship == "APPEARS_IN_SAME_DOCUMENT"


def test_load_entities_invalid_type_skipped(tmp_path):
    """An entity with an unrecognised type should be skipped, not raise."""
    data = [
        {"entity_id": "BAD_001", "type": "ANIMAL", "name": "Tiger",
         "aliases": [], "source_documents": [], "confidence": 0.5, "needs_review": False},
        {"entity_id": "PER_good", "type": "PERSON", "name": "Alice",
         "aliases": [], "source_documents": [], "confidence": 0.8, "needs_review": False},
    ]
    path = tmp_path / "e.json"
    path.write_text(json.dumps(data))
    entities, skipped = load_entities(str(path))
    assert len(entities) == 1
    assert len(skipped) == 1
    assert entities[0].entity_id == "PER_good"


def test_load_entities_missing_file():
    with pytest.raises(FileNotFoundError):
        load_entities("/nonexistent/path/entities.json")


def test_load_relationships_missing_file():
    with pytest.raises(FileNotFoundError):
        load_relationships("/nonexistent/path/relationships.json")


def test_load_entities_not_a_list(tmp_path):
    path = tmp_path / "e.json"
    path.write_text(json.dumps({"not": "a list"}))
    with pytest.raises(ValueError):
        load_entities(str(path))


def test_load_m1_output_returns_both(valid_entities_file, valid_relationships_file):
    entities, rels = load_m1_output(valid_entities_file, valid_relationships_file)
    assert len(entities) == 2
    assert len(rels) == 1


def test_load_against_real_m1_output():
    """
    M2.8-style smoke test: load the real M1 output produced in this repo.
    If the file doesn't exist yet, skip gracefully.
    """
    entities_path = Path("M1/output/entities.json")
    rels_path = Path("M1/output/relationships.json")
    if not entities_path.exists() or not rels_path.exists():
        pytest.skip("Real M1 output not yet generated — run M1 pipeline first")

    entities, skipped_e = load_entities(str(entities_path))
    rels, skipped_r = load_relationships(str(rels_path))

    assert len(entities) >= 10, f"Expected ≥10 entities from real M1 output, got {len(entities)}"
    assert len(rels) >= 10, f"Expected ≥10 relationships, got {len(rels)}"
    # No record should be skipped if M1 output is clean
    assert len(skipped_e) == 0, f"Unexpected skipped entities: {skipped_e[:3]}"
    assert len(skipped_r) == 0, f"Unexpected skipped relationships: {skipped_r[:3]}"
