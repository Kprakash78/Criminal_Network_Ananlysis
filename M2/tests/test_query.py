"""
Tests for M2.5 (neighbors, shortest_path, subgraph),
M2.6 (search_by_name), and M2.7 (add_case_incrementally)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import networkx as nx
import pytest

from M2.loader import M1Entity, M1Relationship
from M2.graph_builder import build_graph, load_graph_from_m1_output, CriminalGraph
from M2.query import neighbors, shortest_path, subgraph, search_by_name, add_case_incrementally


# ---------------------------------------------------------------------------
# Shared test fixture: a small, well-known graph
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def known_graph():
    """
    Graph structure:
      PER_A (Ravi Kumar) --[FIR_001: APPEARS_IN_CASE]--> PHN_1 (9876543210)
      PER_A               --[CDR_001: CALLED]-----------> PER_B (Sunita Sharma)
      PER_B               --[FIR_001: APPEARS_IN_CASE]--> LOC_1 (Lajpat Nagar)
      PER_B               --[FIR_002: APPEARS_IN_CASE]--> PER_C (Mohammed Iqbal)
      PER_C               --[TXN_001: TRANSFERRED...]--> ACC_1 (ACC00101)
      ORG_1 (isolated)    (no edges — isolated node)
    """
    entities = [
        M1Entity(entity_id="PER_A", type="PERSON", name="Ravi Kumar",
                 aliases=["R. Kumar", "Ravi K."],
                 source_documents=["FIR_001"], confidence=0.9, needs_review=False),
        M1Entity(entity_id="PER_B", type="PERSON", name="Sunita Sharma",
                 aliases=["S. Sharma"],
                 source_documents=["FIR_001", "FIR_002"], confidence=0.85, needs_review=False),
        M1Entity(entity_id="PER_C", type="PERSON", name="Mohammed Iqbal",
                 aliases=["M. Iqbal"],
                 source_documents=["FIR_002"], confidence=0.80, needs_review=False),
        M1Entity(entity_id="PHN_1", type="PHONE", name="9876543210",
                 aliases=[], source_documents=["FIR_001"], confidence=0.98, needs_review=False),
        M1Entity(entity_id="LOC_1", type="LOCATION", name="Lajpat Nagar",
                 aliases=["Lajpat Nagar, Delhi"],
                 source_documents=["FIR_001"], confidence=0.82, needs_review=False),
        M1Entity(entity_id="ACC_1", type="ACCOUNT", name="ACC00101",
                 aliases=[], source_documents=["FIR_002"], confidence=0.95, needs_review=False),
        M1Entity(entity_id="ORG_1", type="ORGANIZATION", name="Sunrise Finance Ltd.",
                 aliases=["Sunrise Finance"],
                 source_documents=["FIR_003"], confidence=0.75, needs_review=False),
    ]
    relationships = [
        M1Relationship(source="PER_A", target="PHN_1",
                       relationship="APPEARS_IN_SAME_DOCUMENT",
                       timestamp="2025-01-01T00:00:00",
                       source_record="FIR_001", confidence=0.9),
        M1Relationship(source="PER_A", target="PER_B",
                       relationship="APPEARS_IN_SAME_DOCUMENT",
                       timestamp="2025-01-01T10:00:00",
                       source_record="CDR_001", confidence=0.95),
        M1Relationship(source="PER_B", target="LOC_1",
                       relationship="APPEARS_IN_SAME_DOCUMENT",
                       timestamp="2025-01-01T00:00:00",
                       source_record="FIR_001", confidence=0.85),
        M1Relationship(source="PER_B", target="PER_C",
                       relationship="APPEARS_IN_SAME_DOCUMENT",
                       timestamp="2025-01-02T00:00:00",
                       source_record="FIR_002", confidence=0.80),
        M1Relationship(source="PER_C", target="ACC_1",
                       relationship="APPEARS_IN_SAME_DOCUMENT",
                       timestamp="2025-01-03T00:00:00",
                       source_record="TXN_001", confidence=0.95),
    ]
    return build_graph(entities, relationships)


# ===========================================================================
# M2.5 — neighbors()
# ===========================================================================

class TestNeighbors:
    def test_returns_list(self, known_graph):
        result = neighbors(known_graph, "PER_A")
        assert isinstance(result, list)

    def test_outgoing_neighbors(self, known_graph):
        result = neighbors(known_graph, "PER_A")
        neighbor_ids = {r["entity_id"] for r in result}
        assert "PHN_1" in neighbor_ids   # PER_A → PHN_1 via FIR
        assert "PER_B" in neighbor_ids   # PER_A → PER_B via CDR

    def test_incoming_neighbors(self, known_graph):
        """PER_B receives from PER_A via CDR."""
        result = neighbors(known_graph, "PER_B")
        incoming = [r for r in result if r["direction"] == "incoming"]
        incoming_ids = {r["entity_id"] for r in incoming}
        assert "PER_A" in incoming_ids

    def test_neighbor_record_shape(self, known_graph):
        result = neighbors(known_graph, "PER_A")
        assert len(result) > 0
        rec = result[0]
        for field in ("entity_id", "name", "type", "direction", "relationship",
                      "source_record", "confidence", "weight", "timestamp"):
            assert field in rec, f"Missing field: {field}"

    def test_unknown_entity_returns_empty(self, known_graph):
        result = neighbors(known_graph, "DOES_NOT_EXIST")
        assert result == []

    def test_isolated_node_returns_empty(self, known_graph):
        result = neighbors(known_graph, "ORG_1")
        assert result == []


# ===========================================================================
# M2.5 — shortest_path()
# ===========================================================================

class TestShortestPath:
    def test_direct_path(self, known_graph):
        path = shortest_path(known_graph, "PER_A", "PER_B")
        assert path == ["PER_A", "PER_B"]

    def test_multi_hop_path(self, known_graph):
        """PER_A → PER_B → PER_C (2 hops)"""
        path = shortest_path(known_graph, "PER_A", "PER_C")
        assert path is not None
        assert path[0] == "PER_A"
        assert path[-1] == "PER_C"
        assert len(path) == 3

    def test_longer_path(self, known_graph):
        """PER_A → PER_B → PER_C → ACC_1 (3 hops)"""
        path = shortest_path(known_graph, "PER_A", "ACC_1")
        assert path is not None
        assert path[0] == "PER_A"
        assert path[-1] == "ACC_1"
        assert len(path) == 4

    def test_no_path_isolated_node(self, known_graph):
        """ORG_1 is isolated — no path from PER_A."""
        path = shortest_path(known_graph, "PER_A", "ORG_1")
        assert path is None

    def test_self_path(self, known_graph):
        path = shortest_path(known_graph, "PER_A", "PER_A")
        assert path == ["PER_A"]

    def test_missing_source_returns_none(self, known_graph):
        path = shortest_path(known_graph, "MISSING", "PER_B")
        assert path is None

    def test_missing_target_returns_none(self, known_graph):
        path = shortest_path(known_graph, "PER_A", "MISSING")
        assert path is None


# ===========================================================================
# M2.5 — subgraph()
# ===========================================================================

class TestSubgraph:
    def test_subgraph_returns_crminalgraph(self, known_graph):
        sg = subgraph(known_graph, "FIR_001")
        assert isinstance(sg, nx.MultiDiGraph)

    def test_subgraph_contains_only_case_nodes(self, known_graph):
        sg = subgraph(known_graph, "FIR_001")
        # FIR_001 contains: PER_A, PER_B, PHN_1, LOC_1 (all have FIR_001 in source_documents)
        for node_id in sg.nodes:
            attrs = sg.nodes[node_id]
            assert "FIR_001" in attrs.get("source_documents", []), (
                f"Node {node_id} should not be in FIR_001 subgraph"
            )

    def test_subgraph_no_extra_nodes(self, known_graph):
        """PER_C and ACC_1 are FIR_002-only — must not appear in FIR_001 subgraph."""
        sg = subgraph(known_graph, "FIR_001")
        assert "PER_C" not in sg.nodes
        assert "ACC_1" not in sg.nodes
        assert "ORG_1" not in sg.nodes   # FIR_003 only

    def test_subgraph_fir002(self, known_graph):
        sg = subgraph(known_graph, "FIR_002")
        assert "PER_B" in sg.nodes
        assert "PER_C" in sg.nodes
        # PER_A only in FIR_001, not FIR_002
        assert "PER_A" not in sg.nodes

    def test_unknown_case_returns_empty(self, known_graph):
        sg = subgraph(known_graph, "FIR_NONEXISTENT")
        assert sg.number_of_nodes() == 0

    def test_subgraph_does_not_modify_original(self, known_graph):
        original_nodes = known_graph.number_of_nodes()
        subgraph(known_graph, "FIR_001")
        assert known_graph.number_of_nodes() == original_nodes


# ===========================================================================
# M2.6 — search_by_name()
# ===========================================================================

class TestSearchByName:
    def test_exact_match(self, known_graph):
        results = search_by_name(known_graph, "Ravi Kumar")
        assert any(r["entity_id"] == "PER_A" for r in results)

    def test_partial_match(self, known_graph):
        """'Ravi' should match 'Ravi Kumar' (PRD §10 AC)."""
        results = search_by_name(known_graph, "Ravi")
        entity_ids = [r["entity_id"] for r in results]
        assert "PER_A" in entity_ids, f"Expected PER_A in results, got: {entity_ids}"

    def test_alias_match(self, known_graph):
        """'R. Kumar' is an alias of PER_A — should match."""
        results = search_by_name(known_graph, "R. Kumar")
        entity_ids = [r["entity_id"] for r in results]
        assert "PER_A" in entity_ids

    def test_result_shape(self, known_graph):
        results = search_by_name(known_graph, "Ravi Kumar")
        assert len(results) > 0
        rec = results[0]
        for field in ("entity_id", "name", "type", "matched_alias", "score", "confidence"):
            assert field in rec, f"Missing field: {field}"

    def test_sorted_by_score(self, known_graph):
        results = search_by_name(known_graph, "Kumar")
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_below_threshold_excluded(self, known_graph):
        """Completely unrelated query should return nothing."""
        results = search_by_name(known_graph, "ZXQWERTY_NOMATCH_9999", threshold=0.9)
        assert results == []

    def test_empty_query_returns_empty(self, known_graph):
        assert search_by_name(known_graph, "") == []
        assert search_by_name(known_graph, "   ") == []

    def test_score_in_range(self, known_graph):
        results = search_by_name(known_graph, "Ravi Kumar")
        for r in results:
            assert 0.0 <= r["score"] <= 1.0


# ===========================================================================
# M2.7 — add_case_incrementally()
# ===========================================================================

class TestAddCaseIncrementally:
    def _small_graph(self):
        entities = [
            M1Entity(entity_id="PER_A", type="PERSON", name="Ravi Kumar",
                     aliases=[], source_documents=["FIR_001"], confidence=0.9, needs_review=False),
            M1Entity(entity_id="PER_B", type="PERSON", name="Sunita Sharma",
                     aliases=[], source_documents=["FIR_001"], confidence=0.85, needs_review=False),
        ]
        rels = [
            M1Relationship(source="PER_A", target="PER_B",
                           relationship="APPEARS_IN_SAME_DOCUMENT",
                           timestamp="2025-01-01T00:00:00",
                           source_record="FIR_001", confidence=0.85),
        ]
        return build_graph(entities, rels)

    def test_new_entity_added(self):
        graph = self._small_graph()
        original_count = graph.number_of_nodes()
        new_entities = [
            M1Entity(entity_id="PER_C", type="PERSON", name="Mohammed Iqbal",
                     aliases=[], source_documents=["FIR_002"], confidence=0.8, needs_review=False)
        ]
        add_case_incrementally(graph, new_entities, [])
        assert graph.number_of_nodes() == original_count + 1
        assert "PER_C" in graph.nodes

    def test_existing_entity_not_duplicated(self):
        graph = self._small_graph()
        original_count = graph.number_of_nodes()
        # PER_A already exists — should not add a duplicate
        same_entity = [
            M1Entity(entity_id="PER_A", type="PERSON", name="Ravi Kumar",
                     aliases=[], source_documents=["FIR_002"], confidence=0.9, needs_review=False)
        ]
        add_case_incrementally(graph, same_entity, [])
        assert graph.number_of_nodes() == original_count

    def test_new_edge_added(self):
        graph = self._small_graph()
        original_edge_count = graph.number_of_edges()
        new_entities = [
            M1Entity(entity_id="PER_C", type="PERSON", name="Mohammed Iqbal",
                     aliases=[], source_documents=["FIR_002"], confidence=0.8, needs_review=False)
        ]
        new_rels = [
            M1Relationship(source="PER_B", target="PER_C",
                           relationship="APPEARS_IN_SAME_DOCUMENT",
                           timestamp="2025-01-02T00:00:00",
                           source_record="FIR_002", confidence=0.8),
        ]
        add_case_incrementally(graph, new_entities, new_rels)
        assert graph.number_of_edges() > original_edge_count

    def test_duplicate_edge_not_added(self):
        graph = self._small_graph()
        original_edge_count = graph.number_of_edges()
        # Same relationship as already in the graph
        dup_rels = [
            M1Relationship(source="PER_A", target="PER_B",
                           relationship="APPEARS_IN_SAME_DOCUMENT",
                           timestamp="2025-01-01T00:00:00",
                           source_record="FIR_001", confidence=0.85),
        ]
        add_case_incrementally(graph, [], dup_rels)
        assert graph.number_of_edges() == original_edge_count

    def test_source_docs_merged_on_existing_node(self):
        graph = self._small_graph()
        updated_entity = [
            M1Entity(entity_id="PER_A", type="PERSON", name="Ravi Kumar",
                     aliases=[], source_documents=["FIR_002"], confidence=0.9, needs_review=False)
        ]
        add_case_incrementally(graph, updated_entity, [])
        docs = graph.nodes["PER_A"]["source_documents"]
        assert "FIR_001" in docs
        assert "FIR_002" in docs

    def test_returns_same_graph_object(self):
        """add_case_incrementally modifies in place and returns the same object."""
        graph = self._small_graph()
        result = add_case_incrementally(graph, [], [])
        assert result is graph


# ===========================================================================
# Integration: query against real M1 output
# ===========================================================================

class TestQueryOnRealM1:
    @pytest.fixture(scope="class")
    def real_graph(self):
        e_path = Path("M1/output/entities.json")
        r_path = Path("M1/output/relationships.json")
        if not e_path.exists():
            pytest.skip("Real M1 output not yet generated")
        return load_graph_from_m1_output(str(e_path), str(r_path))

    def test_neighbors_on_real_graph(self, real_graph):
        nodes = list(real_graph.nodes)
        assert len(nodes) > 0
        result = neighbors(real_graph, nodes[0])
        assert isinstance(result, list)

    def test_search_ravi_finds_ravi_kumar(self, real_graph):
        """PRD §10 AC: search_by_name('Ravi') must return 'Ravi Kumar'."""
        results = search_by_name(real_graph, "Ravi", threshold=0.7)
        names = [r["name"] for r in results]
        assert any("Ravi" in name for name in names), (
            f"Expected to find 'Ravi Kumar' in results, got: {names}"
        )

    def test_subgraph_fir_001(self, real_graph):
        sg = subgraph(real_graph, "FIR_001")
        # Should have at least some nodes
        assert sg.number_of_nodes() >= 1
        # All nodes should reference FIR_001
        for node_id, attrs in sg.nodes(data=True):
            assert "FIR_001" in attrs.get("source_documents", [])

    def test_shortest_path_on_real_graph(self, real_graph):
        """Find any two connected nodes and verify a path exists."""
        edges = list(real_graph.edges())
        if not edges:
            pytest.skip("No edges in real graph")
        u, v = edges[0][0], edges[0][1]
        path = shortest_path(real_graph, u, v)
        assert path is not None
        assert path[0] == u
        assert path[-1] == v
