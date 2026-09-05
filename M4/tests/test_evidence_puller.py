"""
Tests for M4.5 — Evidence Puller
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import networkx as nx
import pytest

from M4.config import M4Config
from M4.evidence_puller import (
    load_pattern_flags,
    extract_entity_ids_from_text,
    pull_graph_evidence,
    pull_evidence_for_new_case,
)

cfg = M4Config()


def _make_mock_flags_file(tmp_dir: str) -> str:
    """Write a minimal mock pattern_flags.json for tests that don't need real M3 data."""
    flags = [
        {
            "entity_id": "PERSON_001",
            "degree_centrality": 0.5,
            "betweenness_centrality": 0.3,
            "community": 0,
            "flags": ["COMMUNICATION_SPIKE"],
            "priority_score": 0.72,
            "evidence": ["5 calls within 2 hours between suspect nodes"],
        },
        {
            "entity_id": "ACC_001",
            "degree_centrality": 0.4,
            "betweenness_centrality": 0.1,
            "community": 0,
            "flags": ["HIGH_TRANSACTION_FREQUENCY", "RAPID_FUND_MOVEMENT"],
            "priority_score": 0.58,
            "evidence": ["Transaction of INR 200,000 flagged", "Fund chain in 12 hours"],
        },
        {
            "entity_id": "PERSON_002",
            "degree_centrality": 0.1,
            "betweenness_centrality": 0.05,
            "community": 1,
            "flags": [],
            "priority_score": 0.12,
            "evidence": [],
        },
    ]
    path = Path(tmp_dir) / "pattern_flags.json"
    path.write_text(json.dumps(flags), encoding="utf-8")
    return str(path)


def _make_simple_graph():
    g = nx.MultiDiGraph()
    g.add_node("PERSON_001", type="PERSON", name="Ravi Kumar", aliases=["Ravi"], confidence=0.9)
    g.add_node("PERSON_002", type="PERSON", name="Sunita Sharma", aliases=[], confidence=0.9)
    g.add_node("ACC_001", type="BANK_ACCOUNT", name="ACC001", aliases=[], confidence=0.9)
    g.add_edge("PERSON_001", "ACC_001", relationship="OWNS", source_record="FIR_001", confidence=0.9, weight=0.9, timestamp="2025-01-10")
    g.add_edge("PERSON_001", "PERSON_002", relationship="CALLED", source_record="CDR_001", confidence=0.8, weight=0.8, timestamp="2025-01-11")
    return g


class TestLoadPatternFlags:
    def test_loads_real_m3_output(self):
        flags = load_pattern_flags()  # uses DEFAULT_CONFIG path
        assert isinstance(flags, dict)
        # real M3 output has 247 entities
        assert len(flags) > 0

    def test_missing_file_returns_empty(self):
        flags = load_pattern_flags("nonexistent/path.json")
        assert flags == {}

    def test_indexed_by_entity_id(self):
        with tempfile.TemporaryDirectory() as td:
            path = _make_mock_flags_file(td)
            flags = load_pattern_flags(path)
            assert "PERSON_001" in flags
            assert "ACC_001" in flags

    def test_flag_record_has_required_fields(self):
        with tempfile.TemporaryDirectory() as td:
            path = _make_mock_flags_file(td)
            flags = load_pattern_flags(path)
            for eid, record in flags.items():
                for field in ("entity_id", "flags", "evidence", "priority_score"):
                    assert field in record, f"Missing field '{field}' for {eid}"


class TestExtractEntityIds:
    def test_finds_entity_in_text(self):
        known = {"PERSON_001", "ACC_001", "PERSON_002"}
        text = "The suspect is linked to PERSON_001 and account ACC_001"
        found = extract_entity_ids_from_text(text, known)
        assert "PERSON_001" in found
        assert "ACC_001" in found

    def test_case_insensitive(self):
        known = {"PERSON_001"}
        found = extract_entity_ids_from_text("person_001 was seen", known)
        assert "PERSON_001" in found

    def test_entity_not_in_text_not_found(self):
        known = {"PERSON_999"}
        found = extract_entity_ids_from_text("no matching entity here", known)
        assert "PERSON_999" not in found

    def test_no_duplicates(self):
        known = {"PERSON_001"}
        found = extract_entity_ids_from_text("PERSON_001 PERSON_001 PERSON_001", known)
        assert found.count("PERSON_001") == 1


class TestPullGraphEvidence:
    def setup_method(self):
        self.graph = _make_simple_graph()

    def test_returns_dict_with_required_keys(self):
        with tempfile.TemporaryDirectory() as td:
            path = _make_mock_flags_file(td)
            flags = load_pattern_flags(path)
            result = pull_graph_evidence(["PERSON_001"], self.graph, flags, cfg)
            assert "entities" in result
            assert "graph_edges" in result
            assert "summary" in result

    def test_flagged_entity_included(self):
        with tempfile.TemporaryDirectory() as td:
            path = _make_mock_flags_file(td)
            flags = load_pattern_flags(path)
            result = pull_graph_evidence(["PERSON_001"], self.graph, flags, cfg)
            entity = result["entities"][0]
            assert entity["entity_id"] == "PERSON_001"
            assert entity["flags"] == ["COMMUNICATION_SPIKE"]

    def test_graph_edges_populated(self):
        with tempfile.TemporaryDirectory() as td:
            path = _make_mock_flags_file(td)
            flags = load_pattern_flags(path)
            result = pull_graph_evidence(["PERSON_001"], self.graph, flags, cfg)
            assert len(result["graph_edges"]) > 0

    def test_works_without_graph(self):
        """If graph is None, should still return M3 evidence without crashing."""
        with tempfile.TemporaryDirectory() as td:
            path = _make_mock_flags_file(td)
            flags = load_pattern_flags(path)
            result = pull_graph_evidence(["PERSON_001"], None, flags, cfg)
            assert "entities" in result

    def test_unknown_entity_handled_gracefully(self):
        with tempfile.TemporaryDirectory() as td:
            path = _make_mock_flags_file(td)
            flags = load_pattern_flags(path)
            result = pull_graph_evidence(["ENTITY_DOES_NOT_EXIST"], self.graph, flags, cfg)
            entity = result["entities"][0]
            assert entity["flags"] == []
            assert entity["graph_neighbours"] == []

    def test_empty_entity_list_returns_valid_structure(self):
        with tempfile.TemporaryDirectory() as td:
            path = _make_mock_flags_file(td)
            flags = load_pattern_flags(path)
            result = pull_graph_evidence([], self.graph, flags, cfg)
            assert result["entities"] == []
            assert result["graph_edges"] == []


class TestPullEvidenceForNewCase:
    def test_real_m3_data_integration(self):
        """Integration test with real M3 output — no graph."""
        text = "Investigation involves account ACC00102 and related transactions."
        result = pull_evidence_for_new_case(text, graph=None, config=cfg)
        assert isinstance(result, dict)
        assert "entities" in result

    def test_name_based_matching(self):
        """Entities matched by name attribute from graph."""
        graph = _make_simple_graph()
        text = "Ravi Kumar was seen making phone calls near the ATM."
        result = pull_evidence_for_new_case(text, graph=graph, config=cfg)
        matched_ids = [e["entity_id"] for e in result["entities"]]
        assert "PERSON_001" in matched_ids, (
            f"Expected PERSON_001 (Ravi Kumar) to be matched; got {matched_ids}"
        )
