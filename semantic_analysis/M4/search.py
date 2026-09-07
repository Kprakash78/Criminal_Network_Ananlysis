"""
M4 Retrieval — Primary Search API
Exposes the main search interface expected by M5 (LLM/RAG):
  search(structured_query, raw_query, top_k=10) -> list[segment]
Returns ranked segment records conforming to COMMON_DATA_CONTRACT.md with populated 'scores'.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import M4Config, DEFAULT_CONFIG
from .embedder import QueryEmbedder
from .index_manager import FAISSIndexManager
from .scorer import calculate_fusion_score, object_match_score, ocr_match_score

log = logging.getLogger(__name__)


def search(
    structured_query: Dict[str, Any],
    raw_query: str,
    top_k: int = 10,
    cfg: M4Config = DEFAULT_CONFIG,
    embedder: Optional[QueryEmbedder] = None,
    index_mgr: Optional[FAISSIndexManager] = None,
    unified_segments: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Search multi-modal video corpus and return top-k ranked segments.

    Parameters:
      structured_query: parsed query dict from M5 (e.g. {'objects': [{'object': 'motorcycle', 'attributes': ['blue']}], 'context': [...]})
      raw_query: natural language text query string
      top_k: number of candidate segments to return
      cfg: M4Config instance
      embedder: optional pre-loaded QueryEmbedder
      index_mgr: optional pre-loaded FAISSIndexManager
      unified_segments: optional pre-loaded segment list (if None, reads cfg.metadata_jsonl)

    Returns:
      List of top_k segment records sorted by final_score descending, with 'scores' dict populated.
    """
    # 1. Load segments if not provided
    if unified_segments is None:
        unified_segments = _load_segments_from_metadata(cfg.metadata_jsonl)

    if not unified_segments:
        log.warning("No segment records found in metadata store")
        return []

    # 2. Lazy load dependencies if needed
    if embedder is None:
        embedder = QueryEmbedder(cfg=cfg)
    if index_mgr is None:
        index_mgr = FAISSIndexManager(cfg=cfg)

    # 3. Generate query embeddings
    vis_query_vec = embedder.embed_visual_query(raw_query)
    aud_query_vec = embedder.embed_audio_query(raw_query)

    # 4. Perform vector similarity search in FAISS indices
    vis_hits = index_mgr.search_visual(vis_query_vec, top_k=len(unified_segments))
    vis_sim_map = dict(vis_hits)

    aud_hits = index_mgr.search_audio(aud_query_vec, top_k=len(unified_segments))
    aud_sim_map = dict(aud_hits)

    # 5. Score every candidate segment
    scored_results: List[Dict[str, Any]] = []
    for seg in unified_segments:
        # Visual Similarity
        vis_info = seg.get("visual") or {}
        vis_emb_id = vis_info.get("embedding_id", "")
        vis_sim = float(vis_sim_map.get(vis_emb_id, 0.0))

        # Audio / Transcript Similarity
        aud_info = seg.get("audio") or {}
        aud_emb_id = aud_info.get("embedding_id", "")
        aud_sim = float(aud_sim_map.get(aud_emb_id, 0.0))

        # Object Match (with attribute binding)
        obj_match = object_match_score(
            structured_query.get("objects", []),
            seg.get("objects", []),
        )

        # OCR Match
        ocr_match = ocr_match_score(
            structured_query,
            seg.get("ocr", []),
        )

        # Fusion Final Score
        final = calculate_fusion_score(
            visual_sim=vis_sim,
            transcript_sim=aud_sim,
            object_match=obj_match,
            ocr_match=ocr_match,
            cfg=cfg,
        )

        # Build output record copy with populated 'scores'
        seg_out = dict(seg)
        seg_out["scores"] = {
            "visual_similarity": round(vis_sim, 3),
            "transcript_similarity": round(aud_sim, 3),
            "object_match": round(obj_match, 3),
            "ocr_match": round(ocr_match, 3),
            "final_score": round(final, 3),
        }
        scored_results.append(seg_out)

    # 6. Sort by final_score descending
    scored_results.sort(key=lambda s: s["scores"]["final_score"], reverse=True)

    log.info("search() completed: query='%s', top_score=%.3f", raw_query,
             scored_results[0]["scores"]["final_score"] if scored_results else 0.0)
    return scored_results[:top_k]


def _load_segments_from_metadata(path: Path) -> List[Dict[str, Any]]:
    """Read segment records from data/metadata.jsonl."""
    records: List[Dict[str, Any]] = []
    if not path.exists():
        log.warning("Metadata file %s does not exist", path)
        return records

    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                records.append(rec)
            except json.JSONDecodeError as exc:
                log.warning("Bad JSON line %d in %s: %s", lineno, path, exc)

    return records
