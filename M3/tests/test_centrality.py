"""
Tests for M3.1 — Centrality Computation
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import networkx as nx
import pytest

from M3.centrality import compute_centrality


def _make_chain_graph():
    """A → B → C → D — simple chain so we know betweenness of B and C is non-zero."""
    g = nx.MultiDiGraph()
    for n in ["A", "B", "C", "D"]:
        g.add_node(n, type="PERSON", name=n, aliases=[], source_documents=[], confidence=0.9)
    g.add_edge("A", "B", relationship="CALLED", source_record="CDR_001", confidence=0.9, weight=0.9, timestamp="2025-01-01T10:00:00")
    g.add_edge("B", "C", relationship="CALLED", source_record="CDR_002", confidence=0.9, weight=0.9, timestamp="2025-01-01T10:01:00")
    g.add_edge("C", "D", relationship="CALLED", source_record="CDR_003", confidence=0.9, weight=0.9, timestamp="2025-01-01T10:02:00")
    return g


def _make_hub_graph():
    """Hub H connects to A, B, C, D — H should have highest degree."""
    g = nx.MultiDiGraph()
    for n in ["H", "A", "B", "C", "D"]:
        g.add_node(n, type="PERSON", name=n, aliases=[], source_documents=[], confidence=0.9)
    for leaf in ["A", "B", "C", "D"]:
        g.add_edge("H", leaf, relationship="CALLED", source_record="CDR_001", confidence=0.9, weight=0.9, timestamp="2025-01-01T10:00:00")
    return g


class TestComputeCentrality:
    def test_all_nodes_present(self):
        g = _make_chain_graph()
        result = compute_centrality(g)
        assert set(result.keys()) == set(g.nodes)

    def test_has_all_four_metrics(self):
        g = _make_chain_graph()
        result = compute_centrality(g)
        for node_id, metrics in result.items():
            for key in ("degree", "betweenness", "closeness", "pagerank"):
                assert key in metrics, f"Missing '{key}' for node {node_id}"

    def test_values_in_range(self):
        g = _make_chain_graph()
        result = compute_centrality(g)
        for node_id, metrics in result.items():
            for key, val in metrics.items():
                assert 0.0 <= val <= 1.0, f"{node_id}.{key} = {val} out of [0,1]"

    def test_hub_has_highest_degree(self):
        g = _make_hub_graph()
        result = compute_centrality(g)
        hub_deg = result["H"]["degree"]
        for leaf in ["A", "B", "C", "D"]:
            assert hub_deg >= result[leaf]["degree"], "Hub should have ≥ degree of leaves"

    def test_betweenness_middle_nodes_higher(self):
        """In a chain A-B-C-D, B and C are on more shortest paths than A or D."""
        g = _make_chain_graph()
        result = compute_centrality(g)
        assert result["B"]["betweenness"] > result["A"]["betweenness"]
        assert result["C"]["betweenness"] > result["D"]["betweenness"]

    def test_isolated_node_gets_zeros(self):
        g = nx.MultiDiGraph()
        g.add_node("ISOLATED", type="PERSON", name="Isolated", aliases=[], source_documents=[], confidence=0.5)
        g.add_node("A", type="PERSON", name="A", aliases=[], source_documents=[], confidence=0.5)
        g.add_node("B", type="PERSON", name="B", aliases=[], source_documents=[], confidence=0.5)
        g.add_edge("A", "B", relationship="CALLED", source_record="CDR_001", confidence=0.9, weight=0.9, timestamp="t")
        result = compute_centrality(g)
        iso = result["ISOLATED"]
        # degree, betweenness, closeness must be 0 for an isolated node.
        # PageRank distributes probability mass evenly so it is non-zero — that is correct.
        for key in ("degree", "betweenness", "closeness"):
            assert iso[key] == 0.0, f"Isolated node should have {key}=0, got {iso[key]}"
        assert 0.0 <= iso["pagerank"] <= 1.0

    def test_empty_graph_raises(self):
        with pytest.raises(ValueError, match="empty graph"):
            compute_centrality(nx.MultiDiGraph())

    def test_deterministic(self):
        g = _make_chain_graph()
        result1 = compute_centrality(g)
        result2 = compute_centrality(g)
        assert result1 == result2, "Same graph should produce identical results"

    def test_single_node_graph(self):
        g = nx.MultiDiGraph()
        g.add_node("SOLO", type="PERSON", name="Solo", aliases=[], source_documents=[], confidence=0.9)
        result = compute_centrality(g)
        assert "SOLO" in result
        # 1-node graph: degree/betweenness/closeness are 0 (no edges, no paths)
        assert result["SOLO"]["betweenness"] == 0.0
        assert result["SOLO"]["closeness"] == 0.0
        # All values in range
        for key, val in result["SOLO"].items():
            assert 0.0 <= val <= 1.0, f"SOLO.{key} = {val} out of [0,1]"

    def test_real_m2_graph(self):
        """Integration: load real M2 graph and verify all nodes have entries."""
        e_path = Path("M1/output/entities.json")
        r_path = Path("M1/output/relationships.json")
        if not e_path.exists():
            pytest.skip("Real M1/M2 output not available")
        from M2.graph_builder import load_graph_from_m1_output
        graph = load_graph_from_m1_output(str(e_path), str(r_path))
        result = compute_centrality(graph)
        assert len(result) == graph.number_of_nodes()
        for node_id, metrics in result.items():
            for key in ("degree", "betweenness", "closeness", "pagerank"):
                assert key in metrics
                assert 0.0 <= metrics[key] <= 1.0
