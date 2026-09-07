"""
M6_feature — Case Search API with Provenance
PS 26152 — AI-Powered Criminal Network Analysis System

Provides semantic search across case documents with exact file:line provenance.
Uses sentence-transformers (all-MiniLM-L6-v2) for embeddings with TF-IDF fallback
when the model is unavailable (offline/no download).

Key functions:
    index_documents(doc_dir) → builds in-memory chunk index
    search_local(query, top_k=5) → returns ranked chunks with provenance
    generate_answer(query, chunks) → template-based answer with citations

All retrieval is logged to M6_feature/logs/retrieval_logs.jsonl.
"""

import json
import os
import re
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = Path(__file__).resolve().parent / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

CHUNK_SIZE = 250     # characters per chunk
CHUNK_STRIDE = 50    # overlap between chunks


def chunk_file(filepath: str, chunk_size: int = CHUNK_SIZE,
               stride: int = CHUNK_STRIDE) -> list[dict]:
    """
    Split a file into overlapping text chunks with line-level provenance.

    Returns list of dicts:
        {text, file_path, line_start, line_end, char_start, char_end}
    """
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
            f.seek(0)
            lines = f.readlines()
    except (OSError, IOError) as e:
        logger.warning(f"Cannot read {filepath}: {e}")
        return []

    return chunk_text(content, str(filepath), chunk_size=chunk_size, stride=stride)


def chunk_text(content: str, source_label: str = "uploaded_case.txt",
               chunk_size: int = CHUNK_SIZE, stride: int = CHUNK_STRIDE) -> list[dict]:
    """Split in-memory case text into chunks with line-level provenance."""
    if not content or not content.strip():
        return []

    # Build char->line mapping
    lines = content.splitlines(keepends=True)
    char_to_line = []
    for line_num, line in enumerate(lines, start=1):
        for _ in line:
            char_to_line.append(line_num)

    chunks = []
    pos = 0
    while pos < len(content):
        end = min(pos + chunk_size, len(content))
        chunk_text = content[pos:end].strip()

        if chunk_text:
            line_start = char_to_line[pos] if pos < len(char_to_line) else len(lines)
            line_end = char_to_line[min(end - 1, len(char_to_line) - 1)] if end > 0 and char_to_line else line_start

            chunks.append({
                "text": chunk_text,
                "file_path": source_label,
                "line_start": line_start,
                "line_end": line_end,
                "char_start": pos,
                "char_end": end,
            })

        pos += chunk_size - stride
        if pos >= len(content):
            break

    return chunks


# ---------------------------------------------------------------------------
# Embedding Engine (with fallback)
# ---------------------------------------------------------------------------

class EmbeddingEngine:
    """
    Wraps sentence-transformers or falls back to TF-IDF.

    NOTE: A bigger model (e.g., all-mpnet-base-v2 or BGE-large) would
    significantly improve search quality. MiniLM is used for offline/CPU
    feasibility on a judge's laptop.
    """

    def __init__(self):
        self._model = None
        self._tfidf = None
        self._tfidf_matrix = None
        self._use_tfidf = False

    def _load_model(self):
        """Try to load sentence-transformers, fall back to TF-IDF."""
        if self._model is not None or self._tfidf is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            logger.info("[Search] Loaded sentence-transformers/all-MiniLM-L6-v2")
        except Exception as e:
            logger.warning(
                f"[Search] sentence-transformers unavailable ({e}). "
                "Falling back to TF-IDF vectorizer."
            )
            self._use_tfidf = True

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode a list of texts into vectors."""
        self._load_model()

        if self._use_tfidf or self._model is None:
            return self._encode_tfidf(texts)

        try:
            embeddings = self._model.encode(texts, show_progress_bar=False,
                                            convert_to_numpy=True)
            return embeddings
        except Exception as e:
            logger.warning(f"[Search] Encoding failed ({e}), falling back to TF-IDF")
            self._use_tfidf = True
            return self._encode_tfidf(texts)

    def _encode_tfidf(self, texts: list[str]) -> np.ndarray:
        """TF-IDF fallback encoding."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError:
            # Ultra-fallback: simple character n-gram hashing
            logger.warning("[Search] sklearn unavailable, using hash-based vectors")
            return self._encode_hash(texts)

        if self._tfidf is None:
            self._tfidf = TfidfVectorizer(max_features=384, stop_words="english")
            self._tfidf_matrix = self._tfidf.fit_transform(texts)
            return self._tfidf_matrix.toarray().astype(np.float32)
        else:
            return self._tfidf.transform(texts).toarray().astype(np.float32)

    def _encode_hash(self, texts: list[str]) -> np.ndarray:
        """Ultra-fallback: simple hash-based vectors."""
        vectors = []
        for text in texts:
            vec = np.zeros(384, dtype=np.float32)
            for i, char in enumerate(text.lower()):
                vec[hash(char) % 384] += 1.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            vectors.append(vec)
        return np.array(vectors, dtype=np.float32)


# ---------------------------------------------------------------------------
# Search Index
# ---------------------------------------------------------------------------

class SearchIndex:
    """In-memory search index over document chunks."""

    def __init__(self):
        self.chunks: list[dict] = []
        self.embeddings: Optional[np.ndarray] = None
        self.engine = EmbeddingEngine()
        self._indexed = False

    def index_documents(self, doc_dirs: list[str | Path],
                        file_patterns: list[str] = None) -> int:
        """
        Index all documents from the given directories.

        Args:
            doc_dirs: List of directories to scan
            file_patterns: Glob patterns to match (default: *.txt, *.csv)

        Returns:
            Number of chunks indexed
        """
        if file_patterns is None:
            file_patterns = ["*.txt", "*.csv", "*.json"]

        self.chunks = []

        for doc_dir in doc_dirs:
            doc_dir = Path(doc_dir)
            if not doc_dir.exists():
                logger.warning(f"[Search] Directory not found: {doc_dir}")
                continue

            for pattern in file_patterns:
                for filepath in sorted(doc_dir.rglob(pattern)):
                    if filepath.stat().st_size > 1_000_000:  # Skip files > 1MB
                        continue
                    file_chunks = chunk_file(str(filepath))
                    self.chunks.extend(file_chunks)

        if not self.chunks:
            logger.warning("[Search] No chunks indexed!")
            return 0

        # Embed all chunks
        texts = [c["text"] for c in self.chunks]
        self.embeddings = self.engine.encode(texts)
        self._indexed = True

        logger.info(f"[Search] Indexed {len(self.chunks)} chunks from {len(doc_dirs)} directories")
        return len(self.chunks)

    def index_case_text(self, case_text: str, source_label: str = "uploaded_case.txt") -> int:
        """Build an isolated index containing only the currently uploaded case."""
        self.chunks = chunk_text(case_text, source_label)
        self.embeddings = None
        self._indexed = False
        if not self.chunks:
            return 0
        self.embeddings = self.engine.encode([c["text"] for c in self.chunks])
        self._indexed = True
        logger.info("[Search] Indexed %d chunks for uploaded case %s", len(self.chunks), source_label)
        return len(self.chunks)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Search for the most relevant chunks to the query.

        Returns list of dicts with:
            {text, file_path, line_start, line_end, score}
        """
        if not self._indexed or self.embeddings is None:
            logger.warning("[Search] Index not built. Call index_documents() first.")
            return []

        # Encode query
        query_vec = self.engine.encode([query])

        # Compute cosine similarity
        # Normalize
        query_norm = query_vec / (np.linalg.norm(query_vec, axis=1, keepdims=True) + 1e-8)
        embed_norm = self.embeddings / (np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-8)

        scores = np.dot(embed_norm, query_norm.T).flatten()

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            results.append({
                "text": chunk["text"],
                "file_path": chunk["file_path"],
                "line_start": chunk["line_start"],
                "line_end": chunk["line_end"],
                "score": float(scores[idx]),
            })

        return results


# ---------------------------------------------------------------------------
# Global index singleton
# ---------------------------------------------------------------------------

_GLOBAL_INDEX: Optional[SearchIndex] = None


def get_or_build_index(doc_dirs: list[str | Path] = None) -> SearchIndex:
    """Get or build the global search index."""
    global _GLOBAL_INDEX

    if _GLOBAL_INDEX is not None and _GLOBAL_INDEX._indexed:
        return _GLOBAL_INDEX

    if doc_dirs is None:
        doc_dirs = [
            REPO_ROOT / "data" / "firs",
            REPO_ROOT / "data" / "cdrs",
            REPO_ROOT / "data" / "transactions",
        ]

    _GLOBAL_INDEX = SearchIndex()
    _GLOBAL_INDEX.index_documents(doc_dirs)
    return _GLOBAL_INDEX


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def search_local(query: str, top_k: int = 5,
                 doc_dirs: list[str | Path] = None,
                 case_text: str = None,
                 source_label: str = "uploaded_case.txt") -> dict:
    """
    Search case documents and return results with provenance.

    Args:
        query:    Natural language query
        top_k:    Number of results to return
        doc_dirs: Directories to search (default: data/)

    Returns:
        {
            "query": str,
            "answer": str,
            "results": [
                {
                    "text": str,
                    "file_path": str,
                    "line_start": int,
                    "line_end": int,
                    "score": float,
                    "citation": str,  # e.g., "[source: data/firs/FIR_001.txt:L6-L8]"
                }
            ]
        }
    """
    # Uploaded-case searches must never use the global historical index.
    if case_text is not None:
        index = SearchIndex()
        index.index_case_text(case_text, source_label=source_label)
    else:
        index = get_or_build_index(doc_dirs)
    chunks = index.search(query, top_k=top_k)

    # Add citation strings
    for chunk in chunks:
        short_path = _short_path(chunk["file_path"])
        chunk["citation"] = f"[source: {short_path}:L{chunk['line_start']}-L{chunk['line_end']}]"

    # Generate answer
    answer = generate_answer(query, chunks, case_text=case_text)

    result = {
        "query": query,
        "answer": answer,
        "results": chunks,
    }

    # Log retrieval
    _log_retrieval(query, chunks, answer)

    return result


def generate_answer(query: str, chunks: list[dict], case_text: str = None) -> str:
    """
    Generate a template-based answer from search results.

    This is a deterministic template-based summarizer. For better quality,
    a local LLM (e.g., flan-t5-large) could be used — see M4/llm.py for
    the existing implementation.

    Returns answer text with inline citations.
    """
    if not chunks:
        return "No relevant information found in the case documents."

    # Answer location questions directly from the uploaded case context. This
    # prevents a semantic search result from becoming a long, ambiguous dump.
    if case_text and re.search(r"\bwhere\b|\blocation\b|\bplace\b", query, re.IGNORECASE):
        focused = _focused_location_answer(query, case_text)
        if focused:
            return focused

    # Build answer from top chunks
    answer_parts = []
    answer_parts.append(f"Based on the case documents, here is what was found regarding \"{query}\":\n")

    for i, chunk in enumerate(chunks[:3], 1):
        short_path = _short_path(chunk["file_path"])
        citation = f"[source: {short_path}:L{chunk['line_start']}-L{chunk['line_end']}]"

        # Clean the snippet
        snippet = chunk["text"].strip()
        if len(snippet) > 150:
            snippet = snippet[:147] + "..."

        answer_parts.append(f"• {snippet} {citation}")

    if len(chunks) > 3:
        answer_parts.append(f"\n({len(chunks) - 3} additional results available)")

    return "\n".join(answer_parts)


def _focused_location_answer(query: str, case_text: str) -> str:
    """Return a concise location answer when the case contains one."""
    query_words = [w.casefold() for w in re.findall(r"[A-Za-z][A-Za-z'-]+", query)]
    stop_words = {"where", "was", "were", "is", "are", "the", "seen", "located", "location", "place"}
    subject_words = [w for w in query_words if w not in stop_words and len(w) > 2]
    if not subject_words:
        return ""

    text_lower = case_text.casefold()
    subject_pos = next((text_lower.find(word) for word in subject_words if text_lower.find(word) >= 0), -1)
    if subject_pos < 0:
        return ""

    window_start = max(0, subject_pos - 500)
    window_end = min(len(case_text), subject_pos + 700)
    window = case_text[window_start:window_end]

    # Prefer explicit place phrases and retain a compact city/area pair such
    # as "Malad, Mumbai" from the case wording.
    place_patterns = [
        r"\b(?:in|near|at|from|around)\s+([A-Z][A-Za-z-]+(?:,\s*[A-Z][A-Za-z-]+)?)",
        r"\b([A-Z][A-Za-z-]+,\s*[A-Z][A-Za-z-]+)\b",
    ]
    for pattern in place_patterns:
        matches = re.findall(pattern, window)
        for place in matches:
            if place.casefold() not in {"the case", "the incident"}:
                return f"The uploaded case places the relevant activity in {place}."

    return ""


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _log_retrieval(query: str, chunks: list[dict], answer: str):
    """Log retrieval to JSONL for provenance tracking."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "num_results": len(chunks),
        "top_scores": [c["score"] for c in chunks[:5]],
        "top_files": [_short_path(c["file_path"]) for c in chunks[:5]],
        "answer_length": len(answer),
    }

    log_path = LOGS_DIR / "retrieval_logs.jsonl"
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except (OSError, IOError) as e:
        logger.warning(f"[Search] Failed to write retrieval log: {e}")


def _short_path(filepath: str) -> str:
    """Convert absolute path to relative path from repo root."""
    try:
        return str(Path(filepath).relative_to(REPO_ROOT))
    except ValueError:
        return Path(filepath).name
