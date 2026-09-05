"""
Tests for M3.7 — Priority Scorer
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import networkx as nx
import pytest

from M3.config import M3Config
from M3.models import PatternFlag, VALID_FLAG_TYPES
from M3.scorer import (
    compute_priority_score,
    detect_dense_cluster_membership,
    build_pattern_flags,
    FLAG_NORM,
)


def _make_triangle_graph():
    """Three nodes all connected to each other — high density cluster."""
    g = nx.MultiDiGraph()
    for n in ["A", "B", "C"]:
        g.add_node(n, type="PERSON", name=n, aliases=[], source_documents=[], confidence=0.9)
    for u, v in [("A", "B"), ("B", "C"), ("A", "C")]:
        g.add_edge(u, v, relationship="CALLED", source_record="CDR_001", confidence=0.9, weight=0.9, timestamp="t")
    return g


class TestComputePriorityScore:
    cfg = M3Config(
        weight_degree_centrality=0.15,
        weight_betweenness_centrality=0.20,
        weight_pagerank=0.15,
        weight_flag_count=0.50,
    )

    def test_zero_centrality_zero_flags_gives_low_score(self):
        score, evidence = compute_priority_score("E", {"degree": 0.0, "betweenness": 0.0, "pagerank": 0.0}, [], self.cfg)
        assert score == 0.0
        assert len(evidence) >= 1  # still has a default evidence string

    def test_high_centrality_boosts_score(self):
        score, evidence = compute_priority_score(
            "E",
            {"degree": 1.0, "betweenness": 1.0, "pagerank": 1.0},
            [],
            self.cfg,
        )
        expected = self.cfg.weight_degree_centrality + self.cfg.weight_betweenness_centrality + self.cfg.weight_pagerank
        assert abs(score - expected) < 0.001

    def test_flags_contribute_to_score(self):
        no_flag_score, _ = compute_priority_score("E", {"degree": 0.5, "betweenness": 0.0, "pagerank": 0.0}, [], self.cfg)
        with_flag_score, _ = compute_priority_score(
            "E", {"degree": 0.5, "betweenness": 0.0, "pagerank": 0.0},
            [("COMMUNICATION_SPIKE", "evidence text")], self.cfg
        )
        assert with_flag_score > no_flag_score

    def test_score_clamped_to_one(self):
        score, _ = compute_priority_score(
            "E",
            {"degree": 1.0, "betweenness": 1.0, "pagerank": 1.0},
            [("COMMUNICATION_SPIKE", "e"), ("HIGH_TRANSACTION_FREQUENCY", "e"), ("RAPID_FUND_MOVEMENT", "e")],
            self.cfg,
        )
        assert score <= 1.0

    def test_score_never_negative(self):
        score, _ = compute_priority_score("E", {"degree": 0.0, "betweenness": 0.0, "pagerank": 0.0}, [], self.cfg)
        assert score >= 0.0

    def test_evidence_includes_flag_types(self):
        _, evidence = compute_priority_score(
            "E", {"degree": 0.3, "betweenness": 0.0, "pagerank": 0.0},
            [("COMMUNICATION_SPIKE", "47 calls in 2 hours")], self.cfg
        )
        assert any("COMMUNICATION_SPIKE" in e for e in evidence)
        assert any("47 calls" in e for e in evidence)

    def test_deterministic(self):
        centrality = {"degree": 0.5, "betweenness": 0.3, "pagerank": 0.02}
        flags = [("COMMUNICATION_SPIKE", "5 calls in 1h")]
        s1, e1 = compute_priority_score("E", centrality, flags, self.cfg)
        s2, e2 = compute_priority_score("E", centrality, flags, self.cfg)
        assert s1 == s2
        assert e1 == e2


class TestDetectDenseClusterMembership:
    def test_dense_cluster_flagged(self):
        """All nodes in a triangle share community — all should be flagged."""
        g = _make_triangle_graph()
        community_map = {"A": 0, "B": 0, "C": 0}
        cfg = M3Config(dense_cluster_degree_ratio=0.5)
        result = detect_dense_cluster_membership(g, community_map, cfg)
        assert "A" in result or "B" in result or "C" in result
        for eid, (flag_type, evidence) in result.items():
            assert flag_type == "DENSE_CLUSTER_MEMBERSHIP"
            assert evidence != ""

    def test_non_dense_cluster_not_flagged(self):
        """Nodes whose neighbours are mostly in OTHER communities should not flag."""
        g = nx.MultiDiGraph()
        for n in ["A", "B", "C", "D"]:
            g.add_node(n, type="PERSON", name=n, aliases=[], source_documents=[], confidence=0.9)
        # A is in community 0, all its neighbors B,C,D are in different communities
        for v in ["B", "C", "D"]:
            g.add_edge("A", v, relationship="CALLED", source_record="CDR", confidence=0.9, weight=1.0, timestamp="t")
        community_map = {"A": 0, "B": 1, "C": 2, "D": 3}
        cfg = M3Config(dense_cluster_degree_ratio=0.7)
        result = detect_dense_cluster_membership(g, community_map, cfg)
        assert "A" not in result, "Node with all cross-community neighbours should not be flagged"

    def test_singleton_not_flagged(self):
        """Isolated node or node with < 2 neighbours must not be flagged."""
        g = nx.MultiDiGraph()
        g.add_node("ISO", type="PERSON", name="ISO", aliases=[], source_documents=[], confidence=0.5)
        community_map = {"ISO": 0}
        result = detect_dense_cluster_membership(g, community_map)
        assert "ISO" not in result


class TestBuildPatternFlags:
    def test_returns_pattern_flag_list(self):
        g = _make_triangle_graph()
        centrality = {n: {"degree": 0.5, "betweenness": 0.1, "closeness": 0.5, "pagerank": 0.1} for n in g.nodes}
        community = {"A": 0, "B": 0, "C": 0}
        raw_flags = {"A": [("COMMUNICATION_SPIKE", "5 calls in 1h")]}
        result = build_pattern_flags(g, centrality, community, raw_flags)
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, PatternFlag)

    def test_sorted_by_priority_descending(self):
        g = _make_triangle_graph()
        centrality = {
            "A": {"degree": 0.9, "betweenness": 0.9, "closeness": 0.9, "pagerank": 0.9},
            "B": {"degree": 0.1, "betweenness": 0.0, "closeness": 0.1, "pagerank": 0.01},
            "C": {"degree": 0.1, "betweenness": 0.0, "closeness": 0.1, "pagerank": 0.01},
        }
        community = {"A": 0, "B": 0, "C": 0}
        raw_flags = {"A": [("COMMUNICATION_SPIKE", "e1")]}
        result = build_pattern_flags(g, centrality, community, raw_flags)
        scores = [r.priority_score for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_every_flag_has_evidence(self):
        g = _make_triangle_graph()
        centrality = {n: {"degree": 0.5, "betweenness": 0.1, "closeness": 0.5, "pagerank": 0.1} for n in g.nodes}
        community = {"A": 0, "B": 0, "C": 0}
        raw_flags = {
            "A": [("COMMUNICATION_SPIKE", "5 calls"), ("HIGH_TRANSACTION_FREQUENCY", "3 txns")]
        }
        result = build_pattern_flags(g, centrality, community, raw_flags)
        for record in result:
            assert len(record.evidence) >= 1, f"No evidence for {record.entity_id}"
            for e in record.evidence:
                assert e != ""

    def test_flags_only_from_valid_set(self):
        g = _make_triangle_graph()
        centrality = {n: {"degree": 0.5, "betweenness": 0.1, "closeness": 0.5, "pagerank": 0.1} for n in g.nodes}
        community = {"A": 0, "B": 0, "C": 0}
        raw_flags = {"A": [("COMMUNICATION_SPIKE", "e")]}
        result = build_pattern_flags(g, centrality, community, raw_flags)
        for record in result:
            for ft in record.flags:
                assert ft in VALID_FLAG_TYPES

    def test_real_m2_graph(self):
        e_path = Path("M1/output/entities.json")
        r_path = Path("M1/output/relationships.json")
        if not e_path.exists():
            pytest.skip("Real M1/M2 output not available")
        from M2.graph_builder import load_graph_from_m1_output
        from M3.centrality import compute_centrality
        from M3.community import detect_communities
        graph = load_graph_from_m1_output(str(e_path), str(r_path))
        centrality = compute_centrality(graph)
        community = detect_communities(graph)
        raw_flags = {}
        result = build_pattern_flags(graph, centrality, community, raw_flags)
        assert len(result) > 0
        for record in result:
            assert 0.0 <= record.priority_score <= 1.0
            assert len(record.evidence) >= 1
