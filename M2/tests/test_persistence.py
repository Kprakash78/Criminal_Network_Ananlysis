"""
Tests for M2.4 — Persistence Layer (round-trip integrity)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import networkx as nx
import pytest

from M2.loader import M1Entity, M1Relationship
from M2.graph_builder import build_graph, load_graph_from_m1_output
from M2.persistence import persist_graph, load_persisted_graph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_test_graph():
    """Build a small test graph with known structure."""
    entities = [
        M1Entity(entity_id="PER_001", type="PERSON", name="Ravi Kumar",
                 aliases=["R. Kumar"], source_documents=["FIR_001"],
                 confidence=0.88, needs_review=False),
        M1Entity(entity_id="PHN_001", type="PHONE", name="9876543210",
                 aliases=[], source_documents=["FIR_001"],
                 confidence=0.98, needs_review=False),
        M1Entity(entity_id="LOC_001", type="LOCATION", name="Lajpat Nagar",
                 aliases=["Lajpat Nagar, Delhi"], source_documents=["FIR_001"],
                 confidence=0.80, needs_review=False),
    ]
    relationships = [
        M1Relationship(source="PER_001", target="PHN_001",
                       relationship="APPEARS_IN_SAME_DOCUMENT",
                       timestamp="2025-01-01T00:00:00",
                       source_record="FIR_001", confidence=0.85),
        M1Relationship(source="PER_001", target="LOC_001",
                       relationship="APPEARS_IN_SAME_DOCUMENT",
                       timestamp="2025-01-01T00:00:00",
                       source_record="FIR_001", confidence=0.80),
        M1Relationship(source="PER_001", target="PHN_001",
                       relationship="APPEARS_IN_SAME_DOCUMENT",
                       timestamp="2025-01-01T06:00:00",
                       source_record="CDR_001", confidence=0.95),
    ]
    return build_graph(entities, relationships)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_node_count_preserved(self, tmp_path):
        graph = _make_test_graph()
        db = str(tmp_path / "graph.db")
        persist_graph(graph, db)
        reloaded = load_persisted_graph(db)
        assert reloaded.number_of_nodes() == graph.number_of_nodes()

    def test_edge_count_preserved(self, tmp_path):
        graph = _make_test_graph()
        db = str(tmp_path / "graph.db")
        persist_graph(graph, db)
        reloaded = load_persisted_graph(db)
        assert reloaded.number_of_edges() == graph.number_of_edges()

    def test_node_attributes_preserved(self, tmp_path):
        graph = _make_test_graph()
        db = str(tmp_path / "graph.db")
        persist_graph(graph, db)
        reloaded = load_persisted_graph(db)
        node = reloaded.nodes["PER_001"]
        assert node["name"] == "Ravi Kumar"
        assert node["type"] == "PERSON"
        assert pytest.approx(0.88) == node["confidence"]
        assert "R. Kumar" in node["aliases"]
        assert "FIR_001" in node["source_documents"]

    def test_edge_attributes_preserved(self, tmp_path):
        graph = _make_test_graph()
        db = str(tmp_path / "graph.db")
        persist_graph(graph, db)
        reloaded = load_persisted_graph(db)
        # Find the CDR edge between PER_001 and PHN_001
        edges = [
            d for u, v, d in reloaded.edges(data=True)
            if u == "PER_001" and v == "PHN_001"
        ]
        assert len(edges) >= 1
        cdr_edges = [e for e in edges if e["source_record"] == "CDR_001"]
        assert len(cdr_edges) == 1
        assert cdr_edges[0]["relationship"] == "CALLED"
        assert "weight" in cdr_edges[0]

    def test_overwrite_on_second_persist(self, tmp_path):
        """Persisting twice should overwrite cleanly, not append."""
        graph = _make_test_graph()
        db = str(tmp_path / "graph.db")
        persist_graph(graph, db)
        persist_graph(graph, db)   # second persist
        reloaded = load_persisted_graph(db)
        assert reloaded.number_of_nodes() == graph.number_of_nodes()
        assert reloaded.number_of_edges() == graph.number_of_edges()

    def test_empty_graph_roundtrip(self, tmp_path):
        import networkx as nx
        graph = nx.MultiDiGraph()
        db = str(tmp_path / "empty.db")
        persist_graph(graph, db)
        reloaded = load_persisted_graph(db)
        assert reloaded.number_of_nodes() == 0
        assert reloaded.number_of_edges() == 0

    def test_aliases_and_source_docs_are_lists(self, tmp_path):
        """JSON deserialization must return lists, not raw strings."""
        graph = _make_test_graph()
        db = str(tmp_path / "graph.db")
        persist_graph(graph, db)
        reloaded = load_persisted_graph(db)
        node = reloaded.nodes["PER_001"]
        assert isinstance(node["aliases"], list)
        assert isinstance(node["source_documents"], list)


class TestFailureHandling:
    def test_missing_db_raises_runtime_error(self):
        with pytest.raises(RuntimeError, match="not found"):
            load_persisted_graph("/nonexistent/graph.db")

    def test_empty_file_raises_runtime_error(self, tmp_path):
        p = tmp_path / "empty.db"
        p.write_bytes(b"")
        with pytest.raises(RuntimeError, match="empty"):
            load_persisted_graph(str(p))

    def test_wrong_schema_raises_runtime_error(self, tmp_path):
        import sqlite3
        p = tmp_path / "wrong.db"
        conn = sqlite3.connect(p)
        conn.execute("CREATE TABLE foo (id INTEGER)")
        conn.close()
        with pytest.raises(RuntimeError, match="missing required tables"):
            load_persisted_graph(str(p))

    def test_parent_dir_created_automatically(self, tmp_path):
        db = str(tmp_path / "deep" / "nested" / "graph.db")
        graph = nx.MultiDiGraph()
        persist_graph(graph, db)   # should not raise
        assert Path(db).exists()


class TestRealM1RoundTrip:
    def test_round_trip_with_real_m1(self, tmp_path):
        e_path = Path("M1/output/entities.json")
        r_path = Path("M1/output/relationships.json")
        if not e_path.exists():
            pytest.skip("Real M1 output not yet generated")
        graph = load_graph_from_m1_output(str(e_path), str(r_path))
        db = str(tmp_path / "m1_graph.db")
        persist_graph(graph, db)
        reloaded = load_persisted_graph(db)
        assert reloaded.number_of_nodes() == graph.number_of_nodes()
        assert reloaded.number_of_edges() == graph.number_of_edges()
