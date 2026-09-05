"""
M4.3 + M4.4 — Vector Store (FAISS) and Retriever
PS 26152 — AI-Powered Criminal Network Analysis System

Builds a FAISS flat-L2 index (equivalent to exact cosine search on
normalized vectors) over embedded document chunks, and retrieves the
top-k most similar chunks for a query.

Why FlatL2 (not IVF/HNSW):
  At demo scale (35 FIR docs × ~5 chunks each ≈ 175 chunks), an exact
  brute-force search is fast enough (<10ms) and avoids the quantization
  error of approximate methods. Switch to HNSW when the corpus grows.

Index persistence:
  The index is saved to disk after building so it doesn't need to be
  rebuilt on every pipeline start. Load it back with load_index().
"""

import json
import logging
from pathlib import Path

import faiss
import numpy as np

from M4.config import M4Config, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


class VectorIndex:
    """
    Wraps a FAISS index with its associated metadata (chunk texts, doc IDs).
    Serialisable to disk; load with VectorIndex.load().
    """

    def __init__(
        self,
        index: faiss.IndexFlatIP,
        chunks: list[dict],
        config: M4Config = DEFAULT_CONFIG,
    ):
        self.index = index          # FAISS inner-product index (cosine on normalised vecs)
        self.chunks = chunks        # parallel list: chunks[i] ↔ index vector i
        self.config = config

    def save(self, directory: str) -> None:
        """Save the FAISS index and chunk metadata to directory."""
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(out / "faiss.index"))
        (out / "chunks.json").write_text(
            json.dumps(self.chunks, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info(f"[VectorStore] Saved index ({len(self.chunks)} chunks) to {out}")

    @classmethod
    def load(cls, directory: str, config: M4Config = DEFAULT_CONFIG) -> "VectorIndex":
        """Load a previously saved index from directory."""
        d = Path(directory)
        index = faiss.read_index(str(d / "faiss.index"))
        chunks = json.loads((d / "chunks.json").read_text(encoding="utf-8"))
        logger.info(f"[VectorStore] Loaded index ({len(chunks)} chunks) from {d}")
        return cls(index, chunks, config)

    @property
    def size(self) -> int:
        return self.index.ntotal


def build_vector_index(
    documents: list[str],
    doc_ids: list[str] | None = None,
    config: M4Config = DEFAULT_CONFIG,
) -> VectorIndex:
    """
    Chunk, embed, and index a list of document strings.

    Args:
        documents: list of raw document texts (e.g. FIR contents)
        doc_ids:   optional list of identifiers (same length as documents)
        config:    M4Config

    Returns:
        VectorIndex ready for retrieval.

    Raises:
        ValueError: if documents is empty.
    """
    if not documents:
        raise ValueError("Cannot build a vector index from an empty document list.")

    from M4.embedder import chunk_text, embed_documents

    if doc_ids is None:
        doc_ids = [f"doc{i}" for i in range(len(documents))]

    all_chunks: list[dict] = []
    for doc_text, doc_id in zip(documents, doc_ids):
        all_chunks.extend(chunk_text(doc_text, config, doc_id=doc_id))

    if not all_chunks:
        raise ValueError("All documents produced empty chunks after chunking.")

    texts = [c["text"] for c in all_chunks]
    embeddings = embed_documents(texts, config)  # shape: (N, 384)

    # IndexFlatIP = exact inner-product search; on L2-normalised vectors
    # this is equivalent to cosine similarity.
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    logger.info(
        f"[VectorStore] Built index: {len(documents)} docs, "
        f"{len(all_chunks)} chunks, dim={dim}"
    )
    return VectorIndex(index, all_chunks, config)


def retrieve_relevant_cases(
    new_case_text: str,
    index: VectorIndex,
    config: M4Config = DEFAULT_CONFIG,
    top_k: int | None = None,
    query_doc_id: str | None = None,
) -> list[dict]:
    """
    Retrieve the top-k most relevant historical-case chunks for a new case.

    Args:
        new_case_text: the full text of the new case document
        index:         the VectorIndex built from historical documents
        config:        M4Config
        top_k:         override config.top_k if provided
        query_doc_id:  if provided, exclude chunks from this same document
                       (prevents a document from retrieving itself)

    Returns:
        list of dicts, each containing:
          {
            "chunk_id": str,
            "doc_id": str,
            "text": str,
            "start": int,
            "relevance_score": float,   # cosine similarity in [0, 1]
          }
        Sorted by descending relevance_score.
        Empty list if the index is empty or no results are above threshold.
    """
    from M4.embedder import embed_document

    if index.size == 0:
        logger.warning("[VectorStore] Index is empty — returning no results")
        return []

    k = top_k if top_k is not None else config.top_k
    k = min(k, index.size)  # can't retrieve more than we have

    query_vec = embed_document(new_case_text, config).reshape(1, -1)
    scores, indices = index.index.search(query_vec, k * 3)  # over-retrieve then filter

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        chunk = index.chunks[idx]
        if query_doc_id and chunk["doc_id"] == query_doc_id:
            continue  # skip self-matches
        results.append({
            **chunk,
            "relevance_score": float(score),
        })
        if len(results) >= k:
            break

    results.sort(key=lambda x: x["relevance_score"], reverse=True)
    return results
