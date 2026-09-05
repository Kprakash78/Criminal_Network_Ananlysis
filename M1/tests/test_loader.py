"""
Tests for M1.2 — Loader + Preprocessor
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from M1.generate_dataset import generate_all
from M1.loader import (
    Document,
    load_fir,
    load_fir_directory,
    load_cdrs,
    load_transactions,
    load_all,
    _clean_text,
    _split_sentences,
)


@pytest.fixture(scope="module")
def dataset_base(tmp_path_factory):
    base = tmp_path_factory.mktemp("dataset_m12")
    generate_all(base_dir=base)
    return base


# ---------------------------------------------------------------------------
# Unit tests for text cleaning helpers
# ---------------------------------------------------------------------------

def test_clean_text_strips_control_chars():
    raw = "Hello\x00World\x01."
    cleaned = _clean_text(raw)
    assert "\x00" not in cleaned
    assert "\x01" not in cleaned
    assert "Hello" in cleaned
    assert "World" in cleaned


def test_clean_text_collapses_whitespace():
    raw = "Ravi   Kumar    was   arrested."
    cleaned = _clean_text(raw)
    assert "  " not in cleaned  # no double spaces


def test_clean_text_preserves_devanagari():
    raw = "आरोपी रवि कुमार को गिरफ्तार किया गया।"
    cleaned = _clean_text(raw)
    assert "रवि" in cleaned


def test_clean_text_preserves_newlines():
    raw = "Line one.\nLine two."
    cleaned = _clean_text(raw)
    assert "\n" in cleaned


def test_split_sentences_basic():
    text = "Accused was arrested. He was taken to the station. FIR was filed."
    sentences = _split_sentences(text)
    assert len(sentences) >= 2


def test_split_sentences_single():
    text = "Just one sentence with no terminator"
    sentences = _split_sentences(text)
    assert len(sentences) == 1
    assert sentences[0] == text


# ---------------------------------------------------------------------------
# Integration tests for loaders
# ---------------------------------------------------------------------------

def test_load_fir_single(dataset_base):
    fir_dir = dataset_base / "data" / "firs"
    fir_file = sorted(fir_dir.glob("*.txt"))[0]
    doc = load_fir(fir_file)
    assert doc is not None
    assert doc.doc_type == "FIR"
    assert doc.source_path == str(fir_file)
    assert len(doc.clean_text) > 0
    assert len(doc.sentences) >= 1
    assert doc.doc_id == fir_file.stem


def test_load_fir_missing_returns_none(tmp_path):
    doc = load_fir(tmp_path / "nonexistent.txt")
    assert doc is None


def test_load_fir_directory_count(dataset_base):
    fir_dir = dataset_base / "data" / "firs"
    docs = load_fir_directory(fir_dir)
    assert len(docs) >= 30


def test_load_cdrs_count(dataset_base):
    cdr_path = dataset_base / "data" / "cdrs" / "cdr.csv"
    docs = load_cdrs(cdr_path)
    assert len(docs) >= 100


def test_load_cdrs_doc_type(dataset_base):
    cdr_path = dataset_base / "data" / "cdrs" / "cdr.csv"
    docs = load_cdrs(cdr_path)
    assert all(d.doc_type == "CDR" for d in docs)


def test_load_cdrs_metadata(dataset_base):
    cdr_path = dataset_base / "data" / "cdrs" / "cdr.csv"
    docs = load_cdrs(cdr_path)
    for doc in docs:
        assert "caller" in doc.metadata
        assert "callee" in doc.metadata
        assert "timestamp" in doc.metadata


def test_load_transactions_count(dataset_base):
    trans_path = dataset_base / "data" / "transactions" / "transactions.csv"
    docs = load_transactions(trans_path)
    assert len(docs) >= 50


def test_load_transactions_doc_type(dataset_base):
    trans_path = dataset_base / "data" / "transactions" / "transactions.csv"
    docs = load_transactions(trans_path)
    assert all(d.doc_type == "TRANSACTION" for d in docs)


def test_load_all_count(dataset_base):
    docs = load_all(dataset_base)
    fir_docs = [d for d in docs if d.doc_type == "FIR"]
    cdr_docs = [d for d in docs if d.doc_type == "CDR"]
    txn_docs = [d for d in docs if d.doc_type == "TRANSACTION"]
    assert len(fir_docs) >= 30
    assert len(cdr_docs) >= 100
    assert len(txn_docs) >= 50


def test_load_all_no_crash_missing_dir(tmp_path):
    """load_all on empty directory should return empty list, not crash."""
    docs = load_all(tmp_path)
    assert docs == []


def test_document_ids_unique(dataset_base):
    docs = load_all(dataset_base)
    ids = [d.doc_id for d in docs]
    assert len(ids) == len(set(ids)), "Document IDs are not unique"


def test_corrupted_fir_skipped(tmp_path):
    """A file that can't be decoded should be skipped, not crash."""
    bad_file = tmp_path / "bad.txt"
    bad_file.write_bytes(b"\xff\xfe bad bytes \xfe\xff")
    # This should not raise
    doc = load_fir(bad_file)
    # Either loads (if Python can partially decode) or returns None — not an exception
    # The key assertion: no unhandled exception was raised (test passes if we get here)
