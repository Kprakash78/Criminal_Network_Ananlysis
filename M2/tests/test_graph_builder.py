"""
Tests for M2.3 — Graph Builder
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import networkx as nx
import pytest

from M2.loader import M1Entity, M1Relationship
from M2.graph_builder import build_graph, load_graph_from_m1_output, CriminalGraph
from M2.relationship_typer import RelType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entity(eid, etype="PERSON", name=None):
    return M1Entity(
        entity_id=eid, type=etype, name=name or f"Name_{eid}",
        aliases=[f"alias_{eid}"], source_documents=["FIR_001"],
        confidence=0.8, needs_review=False,
    )


def _rel(src, tgt, source_record="FIR_001", rel_type="APPEARS_IN_SAME_DOCUMENT", conf=0.85):
    return M1Relationship(
        source=src, target=tgt,
        relationship=rel_type,
        timestamp="2025-01-01T00:00:00",
        source_record=source_record,
        confidence=conf,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBuildGraph:
    def test_nodes_added(self):
        entities = [_entity("A"), _entity("B")]
        graph = build_graph(entities, [])
        assert graph.number_of_nodes() == 2
        assert "A" in graph.nodes
        assert "B" in graph.nodes

    def test_node_attributes_carried(self):
        entities = [M1Entity(
            entity_id="PER_001", type="PERSON", name="Ravi Kumar",
            aliases=["R. Kumar"], source_documents=["FIR_001"],
            confidence=0.88, needs_review=False,
        )]
        graph = build_graph(entities, [])
        node = graph.nodes["PER_001"]
        assert node["name"] == "Ravi Kumar"
        assert node["type"] == "PERSON"
        assert node["confidence"] == pytest.approx(0.88)
        assert "R. Kumar" in node["aliases"]

    def test_edges_added_with_typing(self):
        entities = [_entity("A"), _entity("B")]
        rels = [_rel("A", "B", source_record="CDR_001")]
        graph = build_graph(entities, rels)
        assert graph.number_of_edges() == 1
        edges = list(graph.edges(data=True))
        assert edges[0][2]["relationship"] == RelType.CALLED

    def test_fir_relationship_typed_as_appears_in_case(self):
        entities = [_entity("A"), _entity("B")]
        rels = [_rel("A", "B", source_record="FIR_001")]
        graph = build_graph(entities, rels)
        edge_data = list(graph.edges(data=True))[0][2]
        assert edge_data["relationship"] == RelType.APPEARS_IN_CASE

    def test_missing_source_entity_skipped(self):
        """Edge whose source is not in entity list must be skipped."""
        entities = [_entity("B")]
        rels = [_rel("MISSING", "B", source_record="FIR_001")]
        graph = build_graph(entities, rels)
        assert graph.number_of_edges() == 0
        assert graph.number_of_nodes() == 1   # B is still added

    def test_missing_target_entity_skipped(self):
        entities = [_entity("A")]
        rels = [_rel("A", "MISSING", source_record="FIR_001")]
        graph = build_graph(entities, rels)
        assert graph.number_of_edges() == 0

    def test_duplicate_edge_deduplicated(self):
        """Same (src, tgt, rel_type, source_record) submitted twice → 1 edge."""
        entities = [_entity("A"), _entity("B")]
        rels = [
            _rel("A", "B", source_record="FIR_001"),
            _rel("A", "B", source_record="FIR_001"),  # exact duplicate
        ]
        graph = build_graph(entities, rels)
        assert graph.number_of_edges() == 1

    def test_different_source_records_not_deduplicated(self):
        """Same (src, tgt) but different source_records → 2 distinct edges."""
        entities = [_entity("A"), _entity("B")]
        rels = [
            _rel("A", "B", source_record="FIR_001"),
            _rel("A", "B", source_record="FIR_002"),
        ]
        graph = build_graph(entities, rels)
        assert graph.number_of_edges() == 2

    def test_isolated_node_no_crash(self):
        """An entity with no relationships → isolated node in graph."""
        entities = [_entity("LONE")]
        graph = build_graph(entities, [])
        assert "LONE" in graph.nodes
        assert graph.number_of_edges() == 0

    def test_empty_input_returns_empty_graph(self):
        graph = build_graph([], [])
        assert graph.number_of_nodes() == 0
        assert graph.number_of_edges() == 0

    def test_multidigraph_allows_multiple_edge_types(self):
        """PERSON-PHONE can have both OWNS (entity pair) and APPEARS_IN_CASE (FIR)."""
        entities = [_entity("PER_1", "PERSON"), _entity("PHN_1", "PHONE")]
        rels = [
            _rel("PER_1", "PHN_1", source_record="FIR_001"),    # APPEARS_IN_CASE
            _rel("PER_1", "PHN_1", source_record="CDR_001"),    # CALLED
        ]
        graph = build_graph(entities, rels)
        assert graph.number_of_edges() == 2

    def test_edge_carries_source_record(self):
        """FR10: every edge must have a traceable source_record."""
        entities = [_entity("A"), _entity("B")]
        rels = [_rel("A", "B", source_record="FIR_007")]
        graph = build_graph(entities, rels)
        edge_data = list(graph.edges(data=True))[0][2]
        assert edge_data["source_record"] == "FIR_007"

    def test_graph_is_multidi(self):
        graph = build_graph([], [])
        assert isinstance(graph, nx.MultiDiGraph)


class TestLoadGraphFromM1Output:
    def test_loads_real_m1_output(self):
        e_path = Path("M1/output/entities.json")
        r_path = Path("M1/output/relationships.json")
        if not e_path.exists():
            pytest.skip("Real M1 output not yet generated")
        graph = load_graph_from_m1_output(str(e_path), str(r_path))
        assert graph.number_of_nodes() >= 10
        assert graph.number_of_edges() >= 10
        # Verify node attributes are present
        for node_id, attrs in list(graph.nodes(data=True))[:5]:
            assert "name" in attrs
            assert "type" in attrs
            assert "confidence" in attrs
        # Verify edge attributes are present
        for u, v, data in list(graph.edges(data=True))[:5]:
            assert "relationship" in data
            assert "source_record" in data
            assert "confidence" in data
            assert "weight" in data
