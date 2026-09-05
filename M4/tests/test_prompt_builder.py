"""
Tests for M4.6 — Prompt Builder
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest

from M4.config import M4Config
from M4.models import RagResult
from M4.prompt_builder import build_prompt, build_rag_results, _truncate

cfg = M4Config()


def _mock_chunks(n=3):
    return [
        {
            "chunk_id": f"FIR_001_chunk{i}",
            "doc_id": f"FIR_001",
            "text": f"Historical case text chunk {i}: Ravi Kumar was seen in Lajpat Nagar.",
            "start": i * 100,
            "relevance_score": 0.9 - i * 0.1,
        }
        for i in range(n)
    ]


def _mock_graph_evidence(n_flagged=2):
    entities = [
        {
            "entity_id": f"PERSON_{i:03d}",
            "priority_score": 0.7 - i * 0.1,
            "flags": ["COMMUNICATION_SPIKE"] if i < n_flagged else [],
            "m3_evidence": [f"Evidence string {i}"],
            "community": 0,
            "degree_centrality": 0.5,
            "graph_neighbours": [f"PERSON_{i:03d} --[CALLED]--> PERSON_002 (Sunita)"],
            "graph_neighbour_count": 1,
        }
        for i in range(3)
    ]
    edges = [f"PERSON_000 --[CALLED]--> PERSON_001 (Deepak)", "PERSON_001 --[OWNS]--> ACC_001 (Account)"]
    edge_ids = ["GRAPH_EDGE_PERSON_000_PERSON_001", "GRAPH_EDGE_PERSON_001_ACC_001"]
    return {
        "entities": entities,
        "graph_edges": edges,
        "graph_edge_ids": edge_ids,
        "summary": "3 entities examined: 2 with pattern flags, 3 with graph connections",
    }


class TestTruncate:
    def test_no_truncation_when_short(self):
        assert _truncate("hello", 100) == "hello"

    def test_truncates_long_text(self):
        result = _truncate("A" * 200, 100)
        assert len(result) <= 115  # 100 + "…[truncated]"
        assert "truncated" in result


class TestBuildPrompt:
    def test_returns_tuple(self):
        result = build_prompt("FIR036", "New case text", _mock_chunks(), _mock_graph_evidence(), cfg)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_prompt_contains_case_id(self):
        prompt, _ = build_prompt("FIR036", "New case text", _mock_chunks(), _mock_graph_evidence(), cfg)
        assert "FIR036" in prompt

    def test_prompt_contains_case_text(self):
        prompt, _ = build_prompt("FIR036", "New case text", _mock_chunks(), _mock_graph_evidence(), cfg)
        assert "New case text" in prompt

    def test_evidence_ids_populated(self):
        _, evidence_ids = build_prompt("FIR036", "New case text", _mock_chunks(3), _mock_graph_evidence(), cfg)
        assert len(evidence_ids) > 0
        # Should have RAG_RESULT_* and PATTERN_FLAG_* and GRAPH_EDGE_* ids
        rag_ids = [e for e in evidence_ids if e.startswith("RAG_RESULT")]
        flag_ids = [e for e in evidence_ids if e.startswith("PATTERN_FLAG")]
        assert len(rag_ids) > 0, "Expected RAG_RESULT evidence IDs"
        assert len(flag_ids) > 0, "Expected PATTERN_FLAG evidence IDs"

    def test_prompt_contains_evidence_ids(self):
        prompt, evidence_ids = build_prompt("FIR036", "New case text", _mock_chunks(), _mock_graph_evidence(), cfg)
        # The prompt should reference at least one evidence ID explicitly
        found = [eid for eid in evidence_ids if f"[{eid}]" in prompt]
        assert len(found) > 0, "Evidence IDs must be cited in the prompt text"

    def test_no_evidence_case(self):
        """When no evidence exists, prompt should say so."""
        empty_evidence = {"entities": [], "graph_edges": [], "graph_edge_ids": [], "summary": ""}
        prompt, evidence_ids = build_prompt("FIR036", "New case text", [], empty_evidence, cfg)
        assert "no related historical cases" in prompt.lower()

    def test_long_case_text_truncated(self):
        """Prompt should not blow up on very long case text."""
        long_text = "X" * 5000
        prompt, _ = build_prompt("FIR036", long_text, [], {"entities": [], "graph_edges": [], "graph_edge_ids": [], "summary": ""}, cfg)
        assert len(prompt) < 10000  # bounded

    def test_verification_reminder_in_prompt(self):
        prompt, _ = build_prompt("FIR036", "text", _mock_chunks(), _mock_graph_evidence(), cfg)
        assert "investigator verification" in prompt.lower()


class TestBuildRagResults:
    def test_returns_list(self):
        results = build_rag_results("FIR036", _mock_chunks(), _mock_graph_evidence(), cfg)
        assert isinstance(results, list)

    def test_rag_results_have_schema_fields(self):
        results = build_rag_results("FIR036", _mock_chunks(), _mock_graph_evidence(), cfg)
        for r in results:
            assert isinstance(r, RagResult)
            assert r.case_id == "FIR036"
            assert isinstance(r.relevance_score, float)
            assert isinstance(r.matched_entities, list)
            assert r.source is not None

    def test_low_relevance_filtered(self):
        """Chunks below min_relevance_threshold should not produce RagResults."""
        low_chunks = [
            {"chunk_id": "c1", "doc_id": "FIR_001", "text": "text", "start": 0,
             "relevance_score": 0.01}  # well below 0.25 threshold
        ]
        results = build_rag_results("FIR036", low_chunks, {"entities": [], "graph_edges": [], "graph_edge_ids": [], "summary": ""}, cfg)
        retrieval_results = [r for r in results if r.source.startswith("doc") or r.source.startswith("FIR")]
        assert len(retrieval_results) == 0

    def test_flagged_entities_produce_results(self):
        results = build_rag_results("FIR036", [], _mock_graph_evidence(n_flagged=2), cfg)
        flag_results = [r for r in results if "M3_FLAGS" in r.source]
        assert len(flag_results) == 2
