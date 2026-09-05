"""
Tests for M4.3 + M4.4 — Vector Store and Retriever
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest

from M4.config import M4Config
from M4.vector_store import build_vector_index, retrieve_relevant_cases, VectorIndex

cfg = M4Config()

# Deliberately-planted test documents with known relationships
HISTORICAL_DOCS = [
    "Ravi Kumar was seen near Lajpat Nagar. He used phone 9876543210 and drove vehicle DL01AB1234.",
    "Sunita Sharma transferred funds from account ACC00102 to account ACC00105 at Bandra West.",
    "Mohammed Iqbal and Ravi Kumar were reportedly in contact. Tower location: Connaught Place.",
    "Financial fraud reported involving account ACC00101. Suspect used vehicle DL01AB1234.",
    "Priya Nair called Deepak Verma multiple times from Koramangala. Phone 8877665544 involved.",
]
DOC_IDS = ["FIR_001", "FIR_002", "FIR_003", "FIR_004", "FIR_005"]


class TestBuildVectorIndex:
    def test_builds_without_error(self):
        idx = build_vector_index(HISTORICAL_DOCS, DOC_IDS, cfg)
        assert idx is not None
        assert idx.size == len([c for doc in HISTORICAL_DOCS for c in [True]])  # at least one chunk per doc
        assert idx.size > 0

    def test_chunk_count_proportional_to_docs(self):
        idx = build_vector_index(HISTORICAL_DOCS, DOC_IDS, cfg)
        # 5 short docs → at least 5 chunks (one per doc)
        assert idx.size >= 5

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            build_vector_index([], cfg)

    def test_save_and_load(self):
        idx = build_vector_index(HISTORICAL_DOCS, DOC_IDS, cfg)
        with tempfile.TemporaryDirectory() as td:
            idx.save(td)
            loaded = VectorIndex.load(td, cfg)
            assert loaded.size == idx.size
            assert len(loaded.chunks) == len(idx.chunks)


class TestRetrieveRelevantCases:
    def setup_method(self):
        self.idx = build_vector_index(HISTORICAL_DOCS, DOC_IDS, cfg)

    def test_returns_results(self):
        results = retrieve_relevant_cases("Ravi Kumar phone call", self.idx, cfg)
        assert len(results) > 0

    def test_returns_max_top_k(self):
        results = retrieve_relevant_cases("case query", self.idx, cfg, top_k=3)
        assert len(results) <= 3

    def test_relevance_scores_in_range(self):
        results = retrieve_relevant_cases("Ravi Kumar Lajpat Nagar", self.idx, cfg)
        for r in results:
            assert 0.0 <= r["relevance_score"] <= 1.0

    def test_sorted_by_relevance_descending(self):
        results = retrieve_relevant_cases("Ravi Kumar phone call", self.idx, cfg)
        scores = [r["relevance_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_planted_retrieval(self):
        """
        Planted test: querying about 'Ravi Kumar Lajpat Nagar vehicle DL01AB1234'
        should surface FIR_001 and FIR_004 (they contain these exact terms).
        """
        results = retrieve_relevant_cases(
            "Ravi Kumar was seen in Lajpat Nagar driving vehicle DL01AB1234",
            self.idx, cfg, top_k=5
        )
        retrieved_docs = {r["doc_id"] for r in results}
        # At least one of the two planted matching docs should be retrieved
        assert retrieved_docs & {"FIR_001", "FIR_004"}, (
            f"Planted retrieval test failed: expected FIR_001 or FIR_004, got {retrieved_docs}"
        )

    def test_self_exclusion(self):
        """If query_doc_id is given, that document's chunks should not appear."""
        results = retrieve_relevant_cases(
            HISTORICAL_DOCS[0],  # query is FIR_001's text
            self.idx, cfg, query_doc_id="FIR_001"
        )
        for r in results:
            assert r["doc_id"] != "FIR_001", "Self-match should be excluded"

    def test_chunk_has_text(self):
        results = retrieve_relevant_cases("phone call", self.idx, cfg)
        for r in results:
            assert "text" in r
            assert r["text"] != ""

    def test_empty_index_returns_empty(self):
        from M4.embedder import embed_documents
        import faiss, numpy as np
        # Build a zero-size index
        import faiss
        empty_index = faiss.IndexFlatIP(cfg.embedding_dimension)
        idx = VectorIndex(empty_index, [], cfg)
        results = retrieve_relevant_cases("anything", idx, cfg)
        assert results == []
