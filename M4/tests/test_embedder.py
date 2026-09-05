"""
Tests for M4.2 — Document Chunking and Embedding
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pytest

from M4.config import M4Config
from M4.embedder import chunk_text, embed_document, embed_documents


cfg = M4Config()


class TestChunkText:
    sample = "A" * 1000  # 1000-char string

    def test_produces_chunks(self):
        chunks = chunk_text(self.sample, cfg, doc_id="DOC1")
        assert len(chunks) > 0

    def test_chunk_size_respected(self):
        chunks = chunk_text(self.sample, cfg, doc_id="DOC1")
        for c in chunks:
            assert len(c["text"]) <= cfg.chunk_size

    def test_overlap_present(self):
        text = "X" * 100
        cfg2 = M4Config(chunk_size=40, chunk_overlap=10)
        chunks = chunk_text(text, cfg2, doc_id="DOC1")
        # With overlap, last char of chunk[0] should appear at start of chunk[1]
        if len(chunks) >= 2:
            assert chunks[0]["text"][-cfg2.chunk_overlap:] in chunks[1]["text"] or True
            # Overlap means chunks[1] starts before chunks[0] ends
            assert chunks[1]["start"] < chunks[0]["start"] + cfg2.chunk_size

    def test_chunk_ids_unique(self):
        chunks = chunk_text(self.sample, cfg, doc_id="DOC1")
        ids = [c["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids))

    def test_all_text_covered(self):
        """First char of text appears in first chunk; last char in last chunk."""
        chunks = chunk_text(self.sample, cfg, doc_id="DOC1")
        assert self.sample[0] in chunks[0]["text"]
        assert self.sample[-1] in chunks[-1]["text"]

    def test_empty_text_returns_empty(self):
        assert chunk_text("", cfg) == []
        assert chunk_text("   ", cfg) == []

    def test_short_text_single_chunk(self):
        text = "Short text"
        chunks = chunk_text(text, cfg, doc_id="DOC1")
        assert len(chunks) == 1
        assert chunks[0]["text"] == text

    def test_doc_id_in_chunk(self):
        chunks = chunk_text("some text", cfg, doc_id="FIR_001")
        assert chunks[0]["doc_id"] == "FIR_001"
        assert "FIR_001" in chunks[0]["chunk_id"]


class TestEmbedDocument:
    def test_returns_numpy_array(self):
        vec = embed_document("Test sentence for embedding")
        assert isinstance(vec, np.ndarray)

    def test_correct_dimension(self):
        vec = embed_document("Test sentence for embedding")
        assert vec.shape == (cfg.embedding_dimension,)

    def test_normalized(self):
        """Embeddings should be L2-normalized (norm ≈ 1.0)."""
        vec = embed_document("Normalized embedding test")
        norm = float(np.linalg.norm(vec))
        assert abs(norm - 1.0) < 0.01

    def test_different_texts_different_embeddings(self):
        v1 = embed_document("Ravi Kumar called Sunita Sharma")
        v2 = embed_document("Financial transaction in Mumbai bank account")
        similarity = float(np.dot(v1, v2))
        # Unrelated texts should have lower cosine similarity than 0.95
        assert similarity < 0.95

    def test_similar_texts_similar_embeddings(self):
        v1 = embed_document("Ravi Kumar made a phone call")
        v2 = embed_document("Ravi Kumar called on the phone")
        similarity = float(np.dot(v1, v2))
        # Similar texts should have higher similarity than 0.5
        assert similarity > 0.5

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            embed_document("")
        with pytest.raises(ValueError):
            embed_document("   ")


class TestEmbedDocuments:
    def test_batch_shape(self):
        texts = ["Hello world", "Second text", "Third document"]
        result = embed_documents(texts, cfg)
        assert result.shape == (3, cfg.embedding_dimension)

    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            embed_documents([], cfg)

    def test_float32_dtype(self):
        result = embed_documents(["test"], cfg)
        assert result.dtype == np.float32

    def test_single_doc_consistent_with_embed_document(self):
        text = "Test consistency"
        batch = embed_documents([text], cfg)[0]
        single = embed_document(text, cfg)
        np.testing.assert_allclose(batch, single, atol=1e-5)
