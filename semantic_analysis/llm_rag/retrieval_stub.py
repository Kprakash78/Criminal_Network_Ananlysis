"""
M5 LLM/RAG — M4 Retrieval Stub
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠  STUB — M4 is not yet integrated.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
This file provides:
  1. `retrieve_segments_stub()` — a fake retrieval function that returns
     COMMON_DATA_CONTRACT.md-shaped segment dicts for testing M5's logic
     without M4 being ready.
  2. `retrieve_segments()` — the REAL retrieval function that M5 will
     call once M4 is integrated. Currently it calls the stub. Replace the
     body of this function with the real M4 call (function import or HTTP
     request) to complete the integration.

INTEGRATION SWAP POINT:
  In `retrieve_segments()` below, replace the line:
      return retrieve_segments_stub(structured_query, raw_query, top_k=top_k)
  with a call to M4's actual retrieval function, e.g.:
      from M4.retrieval import retrieve  # or import via FastAPI call
      return retrieve(structured_query.model_dump(), raw_query, top_k=top_k)
"""
from __future__ import annotations

import logging
from typing import Any

from .schemas import StructuredQuery

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Real retrieval entry point (swap stub → real M4 call here)
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_segments(
    structured_query: StructuredQuery,
    raw_query: str,
    *,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Retrieve top-K candidate segments from M4 given a structured query.
    """
    try:
        from semantic_analysis.llm_rag.retrieval.search import search as m4_search
        sq_dict = structured_query.model_dump() if hasattr(structured_query, "model_dump") else dict(structured_query)
        results = m4_search(sq_dict, raw_query, top_k=top_k)
        if results:
            log.info("M4 real retrieval returned %d segments for query: %r", len(results), raw_query)
            return results
    except Exception as exc:
        log.warning("M4 search unavailable or failed (%s); falling back to retrieval stub", exc)

    return retrieve_segments_stub(structured_query, raw_query, top_k=top_k)


# ─────────────────────────────────────────────────────────────────────────────
# Stub — realistic fake segments for testing (COMMON_DATA_CONTRACT.md shape)
# ─────────────────────────────────────────────────────────────────────────────

# A small in-memory corpus of fake segments representative of demo video content.
# Each segment matches the full COMMON_DATA_CONTRACT.md segment-level record shape.
_STUB_CORPUS: list[dict[str, Any]] = [
    {
        "video_id": "video_001",
        "segment_id": "video_001_seg_04",
        "start_ts": 21.0,
        "end_ts": 43.5,
        "visual": {
            "description": "Person in a helmet repairing a motorcycle in a workshop",
            "embedding_id": "vis_video_001_04",
            "quality_flag": "ok",
        },
        "audio": {
            "transcript": "First remove the side panel, then loosen the bolt with a wrench...",
            "embedding_id": "aud_video_001_04",
        },
        "objects": [
            {"label": "person", "confidence": 0.96, "attributes": ["wearing helmet"]},
            {"label": "motorcycle", "confidence": 0.91, "attributes": ["blue"]},
            {"label": "wrench", "confidence": 0.83, "attributes": []},
        ],
        "ocr": ["YAMAHA"],
        "scores": {
            "visual_similarity": 0.91,
            "transcript_similarity": 0.93,
            "object_match": 1.0,
            "ocr_match": 0.7,
            "final_score": 0.92,
        },
    },
    {
        "video_id": "video_001",
        "segment_id": "video_001_seg_02",
        "start_ts": 5.0,
        "end_ts": 18.0,
        "visual": {
            "description": "Close-up of motorcycle engine with tools nearby",
            "embedding_id": "vis_video_001_02",
            "quality_flag": "ok",
        },
        "audio": {
            "transcript": "Today we're going to do a full service on this bike...",
            "embedding_id": "aud_video_001_02",
        },
        "objects": [
            {"label": "motorcycle", "confidence": 0.88, "attributes": ["blue"]},
            {"label": "person", "confidence": 0.72, "attributes": []},
        ],
        "ocr": ["YAMAHA", "600cc"],
        "scores": {
            "visual_similarity": 0.78,
            "transcript_similarity": 0.71,
            "object_match": 0.85,
            "ocr_match": 0.6,
            "final_score": 0.74,
        },
    },
    {
        "video_id": "video_003",
        "segment_id": "video_003_seg_01",
        "start_ts": 0.0,
        "end_ts": 12.0,
        "visual": {
            "description": "Chef preparing food in a professional kitchen",
            "embedding_id": "vis_video_003_01",
            "quality_flag": "ok",
        },
        "audio": {
            "transcript": "Set the air fryer to 200 degrees and place the chicken inside...",
            "embedding_id": "aud_video_003_01",
        },
        "objects": [
            {"label": "person", "confidence": 0.94, "attributes": []},
            {"label": "oven", "confidence": 0.77, "attributes": []},  # YOLO may label air fryer as oven
        ],
        "ocr": ["PHILIPS", "AIR FRYER"],
        "scores": {
            "visual_similarity": 0.65,
            "transcript_similarity": 0.88,
            "object_match": 0.6,
            "ocr_match": 0.95,
            "final_score": 0.77,
        },
    },
    {
        "video_id": "video_005",
        "segment_id": "video_005_seg_03",
        "start_ts": 45.0,
        "end_ts": 67.0,
        "visual": {
            "description": "Person in red shirt working on a blue motorcycle outdoors",
            "embedding_id": "vis_video_005_03",
            "quality_flag": "ok",
        },
        "audio": {
            "transcript": "",
            "embedding_id": "aud_video_005_03",
        },
        "objects": [
            {"label": "person", "confidence": 0.92, "attributes": ["red"]},
            {"label": "motorcycle", "confidence": 0.89, "attributes": ["blue"]},
        ],
        "ocr": [],
        "scores": {
            "visual_similarity": 0.82,
            "transcript_similarity": 0.15,
            "object_match": 0.95,
            "ocr_match": 0.0,
            "final_score": 0.73,
        },
    },
    {
        "video_id": "video_007",
        "segment_id": "video_007_seg_02",
        "start_ts": 30.0,
        "end_ts": 55.0,
        "visual": {
            "description": "Dog playing fetch in a park",
            "embedding_id": "vis_video_007_02",
            "quality_flag": "ok",
        },
        "audio": {
            "transcript": "Good boy! Go get the ball!",
            "embedding_id": "aud_video_007_02",
        },
        "objects": [
            {"label": "dog", "confidence": 0.97, "attributes": []},
            {"label": "ball", "confidence": 0.85, "attributes": []},
            {"label": "person", "confidence": 0.63, "attributes": []},
        ],
        "ocr": [],
        "scores": {
            "visual_similarity": 0.12,
            "transcript_similarity": 0.08,
            "object_match": 0.05,
            "ocr_match": 0.0,
            "final_score": 0.09,
        },
    },
]


def retrieve_segments_stub(
    structured_query: StructuredQuery,
    raw_query: str,
    *,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Stub retrieval: simple keyword + object-label scoring over the in-memory corpus.
    Returns segments sorted by final_score descending, capped to top_k.

    This is NOT real FAISS/embedding retrieval — it's a deterministic stub
    so M5's query-parse and generate logic can be tested end-to-end.
    The stub uses:
      1. Object-label overlap between the structured query and segment objects
      2. Keyword overlap between raw_query and transcripts/descriptions
    to bias which fake segments get returned, making the test realistic.
    """
    scored: list[tuple[float, dict[str, Any]]] = []

    query_labels = {o.object.lower() for o in structured_query.objects}
    query_words = set(raw_query.lower().split())

    for seg in _STUB_CORPUS:
        base_score = seg["scores"].get("final_score", 0.0) or 0.0

        # Boost by object-label overlap with the query
        seg_labels = {o["label"].lower() for o in seg.get("objects", [])}
        label_overlap = len(query_labels & seg_labels) / max(len(query_labels | seg_labels), 1)

        # Boost by transcript / description keyword overlap
        transcript = (seg.get("audio") or {}).get("transcript", "").lower()
        description = (seg.get("visual") or {}).get("description", "").lower()
        text_pool = set((transcript + " " + description).split())
        text_overlap = len(query_words & text_pool) / max(len(query_words), 1)

        # Combined score for stub ranking (final_score from "M4" dominates)
        combined = 0.5 * base_score + 0.3 * label_overlap + 0.2 * text_overlap
        scored.append((combined, seg))

    scored.sort(key=lambda x: x[0], reverse=True)
    results = [seg for _, seg in scored[:top_k]]
    log.debug("[STUB] returning %d segments for query %r", len(results), raw_query)
    return results
