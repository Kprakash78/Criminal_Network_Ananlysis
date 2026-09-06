"""
Tests for M6_feature/search_api.py — Case Search API
PS 26152 — AI-Powered Criminal Network Analysis System
"""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from M6_feature.search_api import (
    chunk_file,
    SearchIndex,
    search_local,
    generate_answer,
)


class TestChunkFile:
    def test_chunks_non_empty_file(self, tmp_path):
        """Chunking a file produces non-empty chunks."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Line 1: Hello world\nLine 2: Foo bar\nLine 3: Test data\n")

        chunks = chunk_file(str(test_file), chunk_size=30, stride=10)
        assert len(chunks) > 0

    def test_chunk_provenance(self, tmp_path):
        """Each chunk must have file_path, line_start, line_end."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("A" * 100 + "\n" + "B" * 100 + "\n")

        chunks = chunk_file(str(test_file), chunk_size=50, stride=10)
        for chunk in chunks:
            assert "file_path" in chunk
            assert "line_start" in chunk
            assert "line_end" in chunk
            assert isinstance(chunk["line_start"], int)
            assert chunk["line_start"] >= 1

    def test_empty_file_returns_empty(self, tmp_path):
        test_file = tmp_path / "empty.txt"
        test_file.write_text("")
        assert chunk_file(str(test_file)) == []

    def test_nonexistent_file_returns_empty(self):
        assert chunk_file("/nonexistent/path/file.txt") == []


class TestSearchIndex:
    def test_index_documents(self, tmp_path):
        """Index should accept directories and return chunk count."""
        doc_dir = tmp_path / "docs"
        doc_dir.mkdir()
        (doc_dir / "test1.txt").write_text("Ravi Kumar was seen near Park Street, Kolkata.")
        (doc_dir / "test2.txt").write_text("Transaction of Rs 50000 from ACC00101.")

        index = SearchIndex()
        count = index.index_documents([doc_dir])
        assert count > 0

    def test_search_returns_results(self, tmp_path):
        """Search should return results with scores."""
        doc_dir = tmp_path / "docs"
        doc_dir.mkdir()
        (doc_dir / "fir.txt").write_text(
            "Ravi Kumar was seen near Park Street, Kolkata on Tuesday evening. "
            "He was driving vehicle DL01AB1234."
        )

        index = SearchIndex()
        index.index_documents([doc_dir])
        results = index.search("Where was Ravi Kumar?")

        assert len(results) > 0
        assert "score" in results[0]
        assert "file_path" in results[0]
        assert "line_start" in results[0]

    def test_search_empty_index(self):
        """Searching an empty index returns empty list."""
        index = SearchIndex()
        results = index.search("test query")
        assert results == []


class TestSearchLocal:
    def test_search_against_demo_data(self):
        """search_local should work against the demo dataset."""
        data_dir = REPO_ROOT / "data"
        if not (data_dir / "firs").exists():
            pytest.skip("Demo data not available")

        result = search_local(
            "Where was the suspect on Tuesday?",
            top_k=5,
            doc_dirs=[data_dir / "firs", data_dir / "cdrs"],
        )

        assert "query" in result
        assert "answer" in result
        assert "results" in result
        assert len(result["results"]) > 0

    def test_results_have_citations(self):
        """Each result should have a citation string."""
        data_dir = REPO_ROOT / "data"
        if not (data_dir / "firs").exists():
            pytest.skip("Demo data not available")

        result = search_local(
            "Ravi Kumar",
            top_k=3,
            doc_dirs=[data_dir / "firs"],
        )

        for r in result["results"]:
            assert "citation" in r
            assert r["citation"].startswith("[source:")
            assert "line_start" in r
            assert "line_end" in r

    def test_provenance_lines_contain_expected_strings(self):
        """Returned provenance lines should contain search-relevant content."""
        data_dir = REPO_ROOT / "data"
        if not (data_dir / "firs").exists():
            pytest.skip("Demo data not available")

        result = search_local(
            "Ravi Kumar phone number",
            top_k=5,
            doc_dirs=[data_dir / "firs"],
        )

        # At least one result should mention Ravi Kumar or his phone
        texts = [r["text"].lower() for r in result["results"]]
        found = any("ravi" in t or "9876543210" in t for t in texts)
        assert found, f"Expected 'ravi' or '9876543210' in results, got: {texts[:2]}"


class TestGenerateAnswer:
    def test_empty_chunks_returns_no_info(self):
        answer = generate_answer("test", [])
        assert "No relevant information" in answer

    def test_answer_includes_citations(self):
        chunks = [
            {
                "text": "Ravi Kumar was seen near Kolkata",
                "file_path": "/data/firs/FIR_001.txt",
                "line_start": 6,
                "line_end": 8,
                "score": 0.9,
            }
        ]
        answer = generate_answer("Where was Ravi Kumar?", chunks)
        assert "[source:" in answer
        assert "Ravi Kumar" in answer
