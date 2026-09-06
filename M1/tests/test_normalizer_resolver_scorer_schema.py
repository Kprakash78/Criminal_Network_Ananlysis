"""
Tests for M1.6 (Normalizer), M1.7 (Resolver), M1.8 (Scorer), M1.9 (Schema/Emitter)
"""
import sys
import json
import sqlite3
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from M1.extractor import RawEntity
from M1.normalizer import (
    normalize_entities,
    _normalize_phone,
    _normalize_vehicle,
    _normalize_name,
    _apply_regex_precedence,
    NormalizedEntity,
)
from M1.resolver import (
    EntityStore,
    resolve_entities,
    ResolvedEntity,
    _make_entity_id,
    _fuzzy_match,
)
from M1.scorer import score_entity
from M1.schema import (
    Entity,
    Relationship,
    PipelineResult,
    resolved_to_entity,
    build_relationships,
    emit_json,
    emit_sqlite,
)


# ===========================================================================
# M1.6 — Normalizer tests
# ===========================================================================

class TestNormalizeHelpers:
    def test_normalize_phone_plain(self):
        assert _normalize_phone("9876543210") == "9876543210"

    def test_normalize_phone_prefix_91(self):
        assert _normalize_phone("+919876543210") == "9876543210"

    def test_normalize_phone_prefix_0(self):
        assert _normalize_phone("09876543210") == "9876543210"

    def test_normalize_phone_separators(self):
        assert _normalize_phone("98765-43210") == "9876543210"

    def test_normalize_vehicle_uppercase(self):
        assert _normalize_vehicle("dl01ab1234") == "DL01AB1234"

    def test_normalize_vehicle_strips_space(self):
        assert _normalize_vehicle("DL 01 AB 1234") == "DL01AB1234"

    def test_normalize_name_title_case(self):
        result = _normalize_name("ravi kumar")
        assert result == "Ravi Kumar"

    def test_normalize_name_collapses_whitespace(self):
        result = _normalize_name("  Priya   Nair  ")
        assert result == "Priya Nair"


class TestNormalizeEntities:
    def _make_raw(self, text, etype, doc_id="DOC1", method="spacy", conf=0.75, ss=-1, se=-1):
        return RawEntity(
            text=text, entity_type=etype, source_doc_id=doc_id,
            extraction_method=method, raw_confidence=conf,
            span_start=ss, span_end=se,
        )

    def test_invalid_phone_discarded(self):
        raw = [self._make_raw("12345", "PHONE", method="regex")]
        result = normalize_entities(raw)
        assert result == []

    def test_valid_phone_kept(self):
        raw = [self._make_raw("9876543210", "PHONE", method="regex")]
        result = normalize_entities(raw)
        assert len(result) == 1
        assert result[0].entity_type == "PHONE"

    def test_duplicate_entities_merged(self):
        """Two raw entities with same text+type → one NormalizedEntity."""
        raw = [
            self._make_raw("Ravi Kumar", "PERSON", doc_id="FIR_001"),
            self._make_raw("Ravi Kumar", "PERSON", doc_id="FIR_002"),
        ]
        result = normalize_entities(raw)
        assert len(result) == 1
        assert set(result[0].source_doc_ids) == {"FIR_001", "FIR_002"}

    def test_regex_precedence_over_ner(self):
        """
        If NER tags a phone number as PERSON (wrong type) and regex correctly
        tags it, the NER entity at the same span should be dropped.
        """
        raw = [
            # regex correctly identified it
            self._make_raw("9876543210", "PHONE", method="regex", ss=10, se=20),
            # spaCy wrongly tagged it as PERSON
            self._make_raw("9876543210", "PERSON", method="spacy", ss=10, se=20),
        ]
        result = normalize_entities(raw)
        # Should only have the PHONE entity, not the spurious PERSON
        types = {r.entity_type for r in result}
        assert "PHONE" in types
        assert "PERSON" not in types

    def test_alias_collection(self):
        """Raw surface forms are collected as aliases."""
        raw = [
            self._make_raw("R. Kumar", "PERSON", doc_id="D1"),
            self._make_raw("Ravi Kumar", "PERSON", doc_id="D2"),
        ]
        # These will be normalized differently, so they remain as separate normalized entities
        # But aliases should include their raw form
        result = normalize_entities(raw)
        all_aliases = {a for e in result for a in e.raw_aliases}
        assert "R. Kumar" in all_aliases or "Ravi Kumar" in all_aliases


# ===========================================================================
# M1.7 — Resolver tests
# ===========================================================================

def _make_norm(text, etype, docs=None, conf=0.75, methods=None):
    return NormalizedEntity(
        normalized_text=text,
        entity_type=etype,
        source_doc_ids=docs or ["DOC1"],
        raw_aliases={text},
        extraction_methods=methods or {"spacy"},
        raw_confidence=conf,
    )


class TestEntityStore:
    def test_add_and_retrieve(self):
        store = EntityStore()
        eid = _make_entity_id("PERSON", "Ravi Kumar")
        entity = ResolvedEntity(
            entity_id=eid, entity_type="PERSON", canonical="Ravi Kumar",
            aliases=["Ravi Kumar"], source_docs=["D1"], confidence=0.75
        )
        store.add(entity)
        assert store.get_by_id(eid) is not None

    def test_all_returns_all(self):
        store = EntityStore()
        for name in ["Alice", "Bob", "Charlie"]:
            eid = _make_entity_id("PERSON", name)
            store.add(ResolvedEntity(
                entity_id=eid, entity_type="PERSON", canonical=name,
                aliases=[name], source_docs=["D1"], confidence=0.8
            ))
        assert len(store.all()) == 3


class TestResolver:
    def test_new_entity_created(self):
        store = EntityStore()
        norms = [_make_norm("Ravi Kumar", "PERSON")]
        resolved = resolve_entities(norms, store)
        assert len(resolved) == 1
        assert resolved[0].entity_type == "PERSON"
        assert resolved[0].canonical == "Ravi Kumar"

    def test_alias_merged(self):
        """
        'Ravi Kumar' and 'R. Kumar' should be merged into one entity
        (fuzzy similarity above PERSON threshold).
        """
        store = EntityStore()
        norms1 = [_make_norm("Ravi Kumar", "PERSON", docs=["FIR_001"])]
        resolve_entities(norms1, store)

        norms2 = [_make_norm("R. Kumar", "PERSON", docs=["FIR_002"])]
        resolve_entities(norms2, store)

        persons = [e for e in store.all() if e.entity_type == "PERSON"]
        assert len(persons) == 1, f"Expected merge into 1 entity, got {[p.canonical for p in persons]}"
        assert "FIR_001" in persons[0].source_docs
        assert "FIR_002" in persons[0].source_docs

    def test_different_persons_not_merged(self):
        """'Ravi Kumar' and 'Priya Nair' should be distinct entities."""
        store = EntityStore()
        resolve_entities([_make_norm("Ravi Kumar", "PERSON")], store)
        resolve_entities([_make_norm("Priya Nair", "PERSON")], store)
        persons = [e for e in store.all() if e.entity_type == "PERSON"]
        assert len(persons) == 2

    def test_phone_exact_match_only(self):
        """Phones must match exactly — '9876543210' and '9876543211' are different."""
        store = EntityStore()
        resolve_entities([_make_norm("9876543210", "PHONE")], store)
        resolve_entities([_make_norm("9876543211", "PHONE")], store)
        phones = [e for e in store.all() if e.entity_type == "PHONE"]
        assert len(phones) == 2

    def test_deterministic_entity_id(self):
        """Same canonical text → same entity_id across two calls."""
        id1 = _make_entity_id("PERSON", "Ravi Kumar")
        id2 = _make_entity_id("PERSON", "Ravi Kumar")
        assert id1 == id2

    def test_at_least_3_duplicates_merged(self):
        """
        Acceptance criterion: ≥3 deliberately duplicated entities (different
        name formats) are correctly merged.
        """
        from M1.generate_dataset import PERSONS
        store = EntityStore()
        merge_count = 0

        for p in PERSONS:
            canonical = p[0]
            aliases = p[1]
            # First: add canonical
            resolve_entities([_make_norm(canonical, "PERSON", docs=["FIR_001"])], store)
            pre_count = len([e for e in store.all() if e.entity_type == "PERSON"])
            # Then: add each alias — should merge, not add new
            for alias in aliases[:2]:   # test first 2 aliases per person
                resolve_entities([_make_norm(alias, "PERSON", docs=["FIR_002"])], store)
                post_count = len([e for e in store.all() if e.entity_type == "PERSON"])
                if post_count == pre_count:
                    merge_count += 1
                else:
                    pre_count = post_count

        assert merge_count >= 3, f"Expected ≥3 successful merges, got {merge_count}"


# ===========================================================================
# M1.8 — Scorer tests
# ===========================================================================

def _make_resolved(conf=0.75, n_docs=1, n_aliases=1):
    return ResolvedEntity(
        entity_id="PER_test",
        entity_type="PERSON",
        canonical="Test Person",
        aliases=["Test Person"] * n_aliases,
        source_docs=[f"D{i}" for i in range(n_docs)],
        confidence=conf,
    )


class TestScorer:
    def test_regex_gets_high_score(self):
        entity = _make_resolved(conf=0.98)
        scored = score_entity(entity, {"regex"})
        assert scored.confidence >= 0.95

    def test_spacy_weighted(self):
        entity = _make_resolved(conf=0.75)
        scored = score_entity(entity, {"spacy"})
        # 0.75 * 0.85 = 0.6375, no bonuses with 1 doc/alias
        assert scored.confidence == pytest.approx(0.6375, abs=0.001)

    def test_multi_doc_bonus(self):
        entity = _make_resolved(conf=0.75, n_docs=3)
        scored = score_entity(entity, {"spacy"})
        # 0.75 * 0.85 + 0.02 * 2 = 0.6375 + 0.04 = 0.6775
        assert scored.confidence > 0.6375

    def test_low_confidence_flagged(self):
        entity = _make_resolved(conf=0.3)
        scored = score_entity(entity, {"spacy"})
        assert scored.needs_review is True

    def test_high_confidence_not_flagged(self):
        entity = _make_resolved(conf=0.98)
        scored = score_entity(entity, {"regex"})
        assert scored.needs_review is False

    def test_confidence_clamped_to_one(self):
        entity = _make_resolved(conf=1.0, n_docs=10, n_aliases=10)
        scored = score_entity(entity, {"regex"})
        assert scored.confidence <= 1.0


# ===========================================================================
# M1.9 — Schema + Emitter tests
# ===========================================================================

class TestEntitySchema:
    def test_valid_entity_passes(self):
        e = Entity(
            entity_id="PER_abc123", type="PERSON", name="Ravi Kumar",
            aliases=["R. Kumar"], source_documents=["FIR_001"], confidence=0.85
        )
        assert e.entity_id == "PER_abc123"

    def test_invalid_type_raises(self):
        with pytest.raises(Exception):
            Entity(
                entity_id="X001", type="ANIMAL", name="Tiger",
                aliases=[], source_documents=[], confidence=0.5
            )

    def test_confidence_out_of_range_raises(self):
        with pytest.raises(Exception):
            Entity(
                entity_id="X002", type="PERSON", name="Test",
                aliases=[], source_documents=[], confidence=1.5
            )

    def test_aliases_deduplicated(self):
        e = Entity(
            entity_id="P001", type="PERSON", name="Test",
            aliases=["A", "A", "B"], source_documents=[], confidence=0.7
        )
        assert e.aliases.count("A") == 1


class TestRelationshipSchema:
    def test_valid_relationship(self):
        r = Relationship(
            source="PER_001", target="PHN_002",
            relationship="APPEARS_IN_SAME_DOCUMENT",
            timestamp="2025-01-01T00:00:00",
            source_record="FIR_001",
            confidence=0.8,
        )
        assert r.source == "PER_001"

    def test_confidence_out_of_range(self):
        with pytest.raises(Exception):
            Relationship(
                source="A", target="B",
                relationship="X",
                timestamp="2025-01-01T00:00:00",
                source_record="D1",
                confidence=2.0,
            )


class TestEmitters:
    @pytest.fixture
    def sample_result(self):
        entities = [
            Entity(entity_id="PER_001", type="PERSON", name="Ravi Kumar",
                   aliases=["R. Kumar"], source_documents=["FIR_001"], confidence=0.85),
            Entity(entity_id="PHN_001", type="PHONE", name="9876543210",
                   aliases=[], source_documents=["FIR_001"], confidence=0.98),
        ]
        relationships = [
            Relationship(
                source="PER_001", target="PHN_001",
                relationship="APPEARS_IN_SAME_DOCUMENT",
                timestamp="2025-01-01T00:00:00",
                source_record="FIR_001", confidence=0.85,
            )
        ]
        log = [{"doc_id": "FIR_001", "doc_type": "FIR", "status": "ok",
                "entity_count": 2, "error_message": None}]
        return PipelineResult(
            entities=entities, relationships=relationships,
            extraction_log=log, total_documents=1,
            total_entities=2, total_relationships=1,
        )

    def test_emit_json(self, sample_result, tmp_path):
        paths = emit_json(sample_result, tmp_path)
        assert (tmp_path / "entities.json").exists()
        assert (tmp_path / "relationships.json").exists()
        assert (tmp_path / "extraction_log.json").exists()
        data = json.loads((tmp_path / "entities.json").read_text())
        assert len(data) == 2
        assert data[0]["entity_id"] == "PER_001"

    def test_emit_sqlite(self, sample_result, tmp_path):
        db_path = emit_sqlite(sample_result, tmp_path)
        assert db_path.exists()
        conn = sqlite3.connect(db_path)
        rows = conn.execute("SELECT * FROM entities").fetchall()
        assert len(rows) == 2
        rels = conn.execute("SELECT * FROM relationships").fetchall()
        assert len(rels) == 1
        conn.close()

    def test_json_validates_against_schema(self, sample_result, tmp_path):
        """Round-trip: emit to JSON, re-load, validate with pydantic."""
        paths = emit_json(sample_result, tmp_path)
        raw_data = json.loads((tmp_path / "entities.json").read_text())
        for item in raw_data:
            e = Entity(**item)   # pydantic validation
            assert 0.0 <= e.confidence <= 1.0


class TestBuildRelationships:
    def test_co_occurrence(self):
        entities = [
            Entity(entity_id="A", type="PERSON", name="Alice",
                   aliases=[], source_documents=["D1"], confidence=0.8),
            Entity(entity_id="B", type="PHONE", name="9876543210",
                   aliases=[], source_documents=["D1"], confidence=0.95),
        ]
        doc_map = {"D1": ["A", "B"]}
        rels = build_relationships(entities, doc_map)
        assert len(rels) == 1
        assert rels[0].source in ("A", "B")
        assert rels[0].target in ("A", "B")

    def test_no_self_relationships(self):
        entities = [
            Entity(entity_id="A", type="PERSON", name="Alice",
                   aliases=[], source_documents=["D1"], confidence=0.8),
        ]
        doc_map = {"D1": ["A"]}
        rels = build_relationships(entities, doc_map)
        assert rels == []

    def test_deduplication(self):
        """Same pair appearing in two docs should produce two separate relationships."""
        entities = [
            Entity(entity_id="A", type="PERSON", name="Alice",
                   aliases=[], source_documents=[], confidence=0.8),
            Entity(entity_id="B", type="PERSON", name="Bob",
                   aliases=[], source_documents=[], confidence=0.8),
        ]
        doc_map = {"D1": ["A", "B"], "D2": ["A", "B"]}
        rels = build_relationships(entities, doc_map)
        # One per document (different source_record)
        assert len(rels) == 2
