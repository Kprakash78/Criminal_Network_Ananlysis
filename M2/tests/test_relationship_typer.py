"""
Tests for M2.2 — Relationship Typing Rules
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from M2.loader import M1Entity, M1Relationship
from M2.relationship_typer import (
    assign_relationship_type,
    assign_weight,
    RelType,
    SOURCE_RECORD_RULES,
    ENTITY_TYPE_PAIR_RULES,
    VALID_REL_TYPES,
)


def _make_entity(eid, etype):
    return M1Entity(
        entity_id=eid, type=etype, name=f"Test {eid}",
        aliases=[], source_documents=[], confidence=0.8, needs_review=False,
    )


def _make_rel(src, tgt, source_record, confidence=0.85):
    return M1Relationship(
        source=src, target=tgt,
        relationship="APPEARS_IN_SAME_DOCUMENT",
        timestamp="2025-01-01T00:00:00",
        source_record=source_record,
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Source-record prefix rules
# ---------------------------------------------------------------------------

class TestSourceRecordRules:
    def test_cdr_maps_to_called(self):
        rel = _make_rel("A", "B", "CDR_001")
        idx = {}
        assert assign_relationship_type(rel, idx) == RelType.CALLED

    def test_cdr_case_insensitive(self):
        rel = _make_rel("A", "B", "cdr_001")
        assert assign_relationship_type(rel, {}) == RelType.CALLED

    def test_txn_maps_to_transferred_money_to(self):
        rel = _make_rel("A", "B", "TXN_042")
        assert assign_relationship_type(rel, {}) == RelType.TRANSFERRED_MONEY_TO

    def test_transaction_prefix_maps_to_transferred(self):
        rel = _make_rel("A", "B", "TRANSACTION_007")
        assert assign_relationship_type(rel, {}) == RelType.TRANSFERRED_MONEY_TO

    def test_fir_maps_to_appears_in_case(self):
        rel = _make_rel("A", "B", "FIR_001")
        assert assign_relationship_type(rel, {}) == RelType.APPEARS_IN_CASE

    def test_fir_takes_priority_over_entity_type(self):
        """Source-record prefix wins over entity-type pair matching."""
        rel = _make_rel("PER_1", "PHN_1", "FIR_001")
        idx = {
            "PER_1": _make_entity("PER_1", "PERSON"),
            "PHN_1": _make_entity("PHN_1", "PHONE"),
        }
        # FIR prefix → APPEARS_IN_CASE, not OWNS
        assert assign_relationship_type(rel, idx) == RelType.APPEARS_IN_CASE


# ---------------------------------------------------------------------------
# Entity-type pair rules (activated when no source-record prefix matches)
# ---------------------------------------------------------------------------

class TestEntityTypePairRules:
    def _rel_no_prefix_match(self, src, tgt):
        return _make_rel(src, tgt, "MISC_RECORD")  # no matching prefix

    def test_person_phone_is_owns(self):
        rel = self._rel_no_prefix_match("PER_1", "PHN_1")
        idx = {
            "PER_1": _make_entity("PER_1", "PERSON"),
            "PHN_1": _make_entity("PHN_1", "PHONE"),
        }
        assert assign_relationship_type(rel, idx) == RelType.OWNS

    def test_person_vehicle_is_owns(self):
        rel = self._rel_no_prefix_match("PER_1", "VEH_1")
        idx = {
            "PER_1": _make_entity("PER_1", "PERSON"),
            "VEH_1": _make_entity("VEH_1", "VEHICLE"),
        }
        assert assign_relationship_type(rel, idx) == RelType.OWNS

    def test_person_location_is_located_at(self):
        rel = self._rel_no_prefix_match("PER_1", "LOC_1")
        idx = {
            "PER_1": _make_entity("PER_1", "PERSON"),
            "LOC_1": _make_entity("LOC_1", "LOCATION"),
        }
        assert assign_relationship_type(rel, idx) == RelType.LOCATED_AT

    def test_person_org_is_works_for(self):
        rel = self._rel_no_prefix_match("PER_1", "ORG_1")
        idx = {
            "PER_1": _make_entity("PER_1", "PERSON"),
            "ORG_1": _make_entity("ORG_1", "ORGANIZATION"),
        }
        assert assign_relationship_type(rel, idx) == RelType.WORKS_FOR

    def test_person_person_is_associated_with(self):
        rel = self._rel_no_prefix_match("PER_1", "PER_2")
        idx = {
            "PER_1": _make_entity("PER_1", "PERSON"),
            "PER_2": _make_entity("PER_2", "PERSON"),
        }
        assert assign_relationship_type(rel, idx) == RelType.ASSOCIATED_WITH

    def test_default_fallback_when_no_match(self):
        """Unknown source record + missing entities → default ASSOCIATED_WITH."""
        rel = _make_rel("X", "Y", "UNKNOWN_999")
        assert assign_relationship_type(rel, {}) == RelType.ASSOCIATED_WITH

    def test_already_typed_rel_passes_through(self):
        """An edge already typed as CALLED should not be re-typed."""
        rel = M1Relationship(
            source="A", target="B",
            relationship="CALLED",
            timestamp="2025-01-01T00:00:00",
            source_record="FIR_001",
            confidence=0.9,
        )
        result = assign_relationship_type(rel, {})
        assert result == RelType.CALLED


# ---------------------------------------------------------------------------
# Weight tests
# ---------------------------------------------------------------------------

class TestAssignWeight:
    def test_called_gets_boosted_weight(self):
        rel = _make_rel("A", "B", "CDR_001", confidence=0.5)
        w = assign_weight(rel, RelType.CALLED)
        assert w == pytest.approx(1.0)   # min(0.5*2, 1.0)

    def test_fir_weight_equals_confidence(self):
        rel = _make_rel("A", "B", "FIR_001", confidence=0.75)
        w = assign_weight(rel, RelType.APPEARS_IN_CASE)
        assert w == pytest.approx(0.75)

    def test_weight_clamped_to_one(self):
        rel = _make_rel("A", "B", "CDR_001", confidence=0.99)
        w = assign_weight(rel, RelType.CALLED)
        assert w <= 1.0


# ---------------------------------------------------------------------------
# Schema consistency
# ---------------------------------------------------------------------------

def test_all_source_record_rule_types_are_valid():
    for _, rel_type in SOURCE_RECORD_RULES:
        assert rel_type in VALID_REL_TYPES

def test_all_entity_pair_rule_types_are_valid():
    for _, rel_type in ENTITY_TYPE_PAIR_RULES.items():
        assert rel_type in VALID_REL_TYPES
