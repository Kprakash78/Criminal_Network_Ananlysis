"""
M4.2 — Document Chunking and Embedding
PS 26152 — AI-Powered Criminal Network Analysis System

Provides two functions:
  chunk_text(text)       → list of overlapping character-window chunks
  embed_documents(texts) → numpy array of embedding vectors (shape: N×384)
  embed_document(text)   → single embedding vector (shape: 384,)

The embedding model (all-MiniLM-L6-v2) is loaded once and cached.
`local_files_only=True` enforces the no-external-network requirement.

Why character-window chunking instead of sentence splitting:
  FIR documents are short, dense paragraphs without reliable sentence
  boundaries (code-switched Hinglish, missing punctuation). A fixed-size
  character window with overlap is simpler and more reliable here.
"""

import logging
from typing import Any

import numpy as np

from M4.config import M4Config, DEFAULT_CONFIG

logger = logging.getLogger(__name__)

_embedder_cache: dict[str, Any] = {}


def _load_embedder(config: M4Config = DEFAULT_CONFIG):
    """Lazy-load the sentence-transformer embedding model (once per process)."""
    if "model" in _embedder_cache:
        return _embedder_cache["model"]

    from sentence_transformers import SentenceTransformer

    logger.info(f"[Embedder] Loading '{config.embedding_model}' (local_files_only=True)")
    model = SentenceTransformer(config.embedding_model, local_files_only=True)
    _embedder_cache["model"] = model
    logger.info("[Embedder] Embedding model loaded")
    return model


def chunk_text(
    text: str,
    config: M4Config = DEFAULT_CONFIG,
    doc_id: str = "",
) -> list[dict]:
    """
    Split a document into overlapping character-window chunks.

    Args:
        text:   the full document text
        config: M4Config (uses chunk_size and chunk_overlap)
        doc_id: optional identifier attached to each chunk for traceability

    Returns:
        list of dicts: {"chunk_id": str, "doc_id": str, "text": str, "start": int}
    """
    text = text.strip()
    if not text:
        return []

    size = config.chunk_size
    overlap = config.chunk_overlap
    step = size - overlap

    chunks = []
    pos = 0
    idx = 0
    while pos < len(text):
        chunk_text = text[pos : pos + size]
        chunks.append({
            "chunk_id": f"{doc_id}_chunk{idx}" if doc_id else f"chunk{idx}",
            "doc_id": doc_id,
            "text": chunk_text,
            "start": pos,
        })
        pos += step
        idx += 1

    return chunks


def embed_documents(
    texts: list[str],
    config: M4Config = DEFAULT_CONFIG,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Embed a list of text strings into dense vectors.

    Args:
        texts:      list of strings to embed (chunks or full documents)
        config:     M4Config
        batch_size: number of texts to encode per batch (trades speed vs. memory)

    Returns:
        numpy array of shape (len(texts), embedding_dimension).

    Raises:
        ValueError: if texts is empty.
    """
    if not texts:
        raise ValueError("Cannot embed an empty list of texts.")

    model = _load_embedder(config)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,  # normalise for cosine similarity via dot product
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def embed_document(text: str, config: M4Config = DEFAULT_CONFIG) -> np.ndarray:
    """
    Embed a single text string. Returns shape (embedding_dimension,).

    Raises:
        ValueError: if text is empty or whitespace-only.
    """
    text = text.strip()
    if not text:
        raise ValueError("Cannot embed an empty document.")
    return embed_documents([text], config)[0]
