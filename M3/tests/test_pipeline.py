"""
Tests for M3.8 — Output Emitter / Full Pipeline
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import networkx as nx
import pytest

from M3.models import PatternFlag, AnalysisResult, VALID_FLAG_TYPES
from M3.pipeline import run_full_analysis, emit_outputs


def _make_simple_graph():
    g = nx.MultiDiGraph()
    for n in ["A", "B", "C"]:
        g.add_node(n, type="PERSON", name=n, aliases=[], source_documents=[], confidence=0.9)
    g.add_edge("A", "B", relationship="CALLED", source_record="CDR_001", confidence=0.9, weight=0.9, timestamp="t")
    g.add_edge("B", "C", relationship="CALLED", source_record="CDR_002", confidence=0.9, weight=0.9, timestamp="t")
    return g


class TestEmitOutputs:
    def test_writes_both_files(self):
        result = AnalysisResult(
            centrality_scores={"A": {"degree": 0.5, "betweenness": 0.1, "closeness": 0.5, "pagerank": 0.1}},
            community_map={"A": 0},
            pattern_flags=[
                PatternFlag(
                    entity_id="A",
                    degree_centrality=0.5,
                    betweenness_centrality=0.1,
                    community=0,
                    flags=["COMMUNICATION_SPIKE"],
                    priority_score=0.75,
                    evidence=["5 calls in 2 hours"],
                )
            ],
        )
        with tempfile.TemporaryDirectory() as td:
            paths = emit_outputs(result, td)
            assert Path(paths["pattern_flags"]).exists()
            assert Path(paths["centrality_scores"]).exists()

    def test_pattern_flags_schema(self):
        """JSON fields must match the shared PATTERN FLAG schema exactly."""
        result = AnalysisResult(
            centrality_scores={"A": {"degree": 0.5, "betweenness": 0.1, "closeness": 0.5, "pagerank": 0.1}},
            community_map={"A": 0},
            pattern_flags=[
                PatternFlag(
                    entity_id="A",
                    degree_centrality=0.5,
                    betweenness_centrality=0.1,
                    community=0,
                    flags=["COMMUNICATION_SPIKE"],
                    priority_score=0.75,
                    evidence=["5 calls in 2 hours"],
                )
            ],
        )
        with tempfile.TemporaryDirectory() as td:
            paths = emit_outputs(result, td)
            records = json.loads(Path(paths["pattern_flags"]).read_text())
            assert len(records) == 1
            record = records[0]
            for required_field in ("entity_id", "degree_centrality", "betweenness_centrality",
                                   "community", "flags", "priority_score", "evidence"):
                assert required_field in record, f"Missing required field: {required_field}"

    def test_centrality_scores_includes_community(self):
        """centrality_scores.json must include community field per node for M6."""
        result = AnalysisResult(
            centrality_scores={"A": {"degree": 0.5, "betweenness": 0.1, "closeness": 0.5, "pagerank": 0.1}},
            community_map={"A": 3},
            pattern_flags=[],
        )
        with tempfile.TemporaryDirectory() as td:
            paths = emit_outputs(result, td)
            data = json.loads(Path(paths["centrality_scores"]).read_text())
            assert "A" in data
            assert data["A"]["community"] == 3

    def test_empty_flags_list_ok(self):
        result = AnalysisResult(
            centrality_scores={"A": {"degree": 0.5, "betweenness": 0.1, "closeness": 0.5, "pagerank": 0.1}},
            community_map={"A": 0},
            pattern_flags=[],
        )
        with tempfile.TemporaryDirectory() as td:
            paths = emit_outputs(result, td)
            data = json.loads(Path(paths["pattern_flags"]).read_text())
            assert data == []

    def test_output_is_valid_json(self):
        result = AnalysisResult(
            centrality_scores={"A": {"degree": 0.5, "betweenness": 0.1, "closeness": 0.5, "pagerank": 0.1}},
            community_map={"A": 0},
            pattern_flags=[
                PatternFlag("A", 0.5, 0.1, 0, ["RAPID_FUND_MOVEMENT"], 0.6, ["Fund chain A→B→C"])
            ],
        )
        with tempfile.TemporaryDirectory() as td:
            paths = emit_outputs(result, td)
            for _, fp in paths.items():
                text = Path(fp).read_text()
                parsed = json.loads(text)
                assert parsed is not None


class TestRunFullAnalysis:
    def test_real_m2_end_to_end(self):
        e_path = Path("M1/output/entities.json")
        r_path = Path("M1/output/relationships.json")
        cdr_path = "M1/data/cdrs/cdr.csv"
        txn_path = "M1/data/transactions/transactions.csv"
        if not e_path.exists():
            pytest.skip("Real M1/M2 output not available")
        from M2.graph_builder import load_graph_from_m1_output
        graph = load_graph_from_m1_output(str(e_path), str(r_path))
        result = run_full_analysis(graph, cdr_path, txn_path)
        assert isinstance(result, AnalysisResult)
        assert len(result.centrality_scores) == graph.number_of_nodes()
        assert len(result.community_map) == graph.number_of_nodes()
        assert isinstance(result.pattern_flags, list)

    def test_all_scores_in_range(self):
        e_path = Path("M1/output/entities.json")
        r_path = Path("M1/output/relationships.json")
        if not e_path.exists():
            pytest.skip("Real M1/M2 output not available")
        from M2.graph_builder import load_graph_from_m1_output
        graph = load_graph_from_m1_output(str(e_path), str(r_path))
        result = run_full_analysis(graph, "M1/data/cdrs/cdr.csv", "M1/data/transactions/transactions.csv")
        for pf in result.pattern_flags:
            assert 0.0 <= pf.priority_score <= 1.0

    def test_every_flag_has_evidence(self):
        e_path = Path("M1/output/entities.json")
        r_path = Path("M1/output/relationships.json")
        if not e_path.exists():
            pytest.skip("Real M1/M2 output not available")
        from M2.graph_builder import load_graph_from_m1_output
        graph = load_graph_from_m1_output(str(e_path), str(r_path))
        result = run_full_analysis(graph, "M1/data/cdrs/cdr.csv", "M1/data/transactions/transactions.csv")
        for pf in result.pattern_flags:
            assert len(pf.evidence) >= 1
            for e in pf.evidence:
                assert e != ""

    def test_deterministic_two_runs(self):
        e_path = Path("M1/output/entities.json")
        r_path = Path("M1/output/relationships.json")
        if not e_path.exists():
            pytest.skip("Real M1/M2 output not available")
        from M2.graph_builder import load_graph_from_m1_output
        graph = load_graph_from_m1_output(str(e_path), str(r_path))
        r1 = run_full_analysis(graph, "M1/data/cdrs/cdr.csv", "M1/data/transactions/transactions.csv")
        r2 = run_full_analysis(graph, "M1/data/cdrs/cdr.csv", "M1/data/transactions/transactions.csv")
        scores1 = {pf.entity_id: pf.priority_score for pf in r1.pattern_flags}
        scores2 = {pf.entity_id: pf.priority_score for pf in r2.pattern_flags}
        assert scores1 == scores2, "Two identical runs must produce identical scores"

    def test_empty_graph_raises(self):
        with pytest.raises(ValueError):
            run_full_analysis(nx.MultiDiGraph(), "cdr.csv", "txn.csv")
