"""
Tests for M3.2 — Community Detection
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import networkx as nx
import pytest

from M3.community import detect_communities


def _make_two_cluster_graph():
    """
    Two clear clusters: {A, B, C} densely connected, {X, Y, Z} densely connected,
    with one bridge edge A-X.  Community detection should separate them.
    """
    g = nx.MultiDiGraph()
    for n in ["A", "B", "C", "X", "Y", "Z"]:
        g.add_node(n, type="PERSON", name=n, aliases=[], source_documents=[], confidence=0.9)

    # Cluster 1: A, B, C — all pairs connected
    for u, v in [("A", "B"), ("B", "C"), ("A", "C")]:
        g.add_edge(u, v, relationship="CALLED", source_record="CDR_001", confidence=0.9, weight=0.9, timestamp="t")

    # Cluster 2: X, Y, Z — all pairs connected
    for u, v in [("X", "Y"), ("Y", "Z"), ("X", "Z")]:
        g.add_edge(u, v, relationship="CALLED", source_record="CDR_002", confidence=0.9, weight=0.9, timestamp="t")

    # Bridge: one edge between clusters
    g.add_edge("A", "X", relationship="CALLED", source_record="CDR_003", confidence=0.9, weight=0.9, timestamp="t")

    return g


class TestDetectCommunities:
    def test_all_nodes_assigned(self):
        g = _make_two_cluster_graph()
        result = detect_communities(g)
        assert set(result.keys()) == set(g.nodes)

    def test_community_ids_are_ints(self):
        g = _make_two_cluster_graph()
        result = detect_communities(g)
        for node_id, cid in result.items():
            assert isinstance(cid, int), f"Community ID for {node_id} should be int, got {type(cid)}"

    def test_community_ids_start_at_zero(self):
        g = _make_two_cluster_graph()
        result = detect_communities(g)
        assert min(result.values()) == 0

    def test_two_clusters_found(self):
        """The two dense cliques should be separated into two distinct communities."""
        g = _make_two_cluster_graph()
        result = detect_communities(g)
        # ABC and XYZ should be in different communities
        cluster_abc = {result["A"], result["B"], result["C"]}
        cluster_xyz = {result["X"], result["Y"], result["Z"]}
        assert len(cluster_abc) == 1, "A, B, C should be in the same community"
        assert len(cluster_xyz) == 1, "X, Y, Z should be in the same community"
        assert cluster_abc != cluster_xyz, "The two dense groups should be in different communities"

    def test_same_cluster_nodes_share_id(self):
        g = _make_two_cluster_graph()
        result = detect_communities(g)
        assert result["A"] == result["B"] == result["C"]

    def test_isolated_node_gets_assigned(self):
        g = nx.MultiDiGraph()
        g.add_node("ISO", type="PERSON", name="Isolated", aliases=[], source_documents=[], confidence=0.5)
        g.add_node("A", type="PERSON", name="A", aliases=[], source_documents=[], confidence=0.9)
        g.add_node("B", type="PERSON", name="B", aliases=[], source_documents=[], confidence=0.9)
        g.add_edge("A", "B", relationship="CALLED", source_record="CDR_001", confidence=0.9, weight=0.9, timestamp="t")
        result = detect_communities(g)
        assert "ISO" in result
        assert isinstance(result["ISO"], int)

    def test_empty_graph_raises(self):
        with pytest.raises(ValueError, match="empty graph"):
            detect_communities(nx.MultiDiGraph())

    def test_deterministic(self):
        g = _make_two_cluster_graph()
        result1 = detect_communities(g)
        result2 = detect_communities(g)
        assert result1 == result2

    def test_clean_case_no_false_split(self):
        """A fully connected clique should all land in the same community (no false split)."""
        g = nx.MultiDiGraph()
        for n in ["P", "Q", "R", "S"]:
            g.add_node(n, type="PERSON", name=n, aliases=[], source_documents=[], confidence=0.9)
        for u, v in [("P", "Q"), ("P", "R"), ("P", "S"), ("Q", "R"), ("Q", "S"), ("R", "S")]:
            g.add_edge(u, v, relationship="CALLED", source_record="CDR_001", confidence=0.9, weight=0.9, timestamp="t")
        result = detect_communities(g)
        ids = set(result.values())
        assert len(ids) == 1, f"Fully connected clique should be one community, got {len(ids)}"

    def test_real_m2_graph(self):
        """Integration: every node in M2's real graph must be assigned a community."""
        e_path = Path("M1/output/entities.json")
        r_path = Path("M1/output/relationships.json")
        if not e_path.exists():
            pytest.skip("Real M1/M2 output not available")
        from M2.graph_builder import load_graph_from_m1_output
        graph = load_graph_from_m1_output(str(e_path), str(r_path))
        result = detect_communities(graph)
        assert len(result) == graph.number_of_nodes()
        assert all(isinstance(v, int) for v in result.values())
        assert min(result.values()) == 0
