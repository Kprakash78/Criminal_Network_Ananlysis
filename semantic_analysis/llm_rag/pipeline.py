"""
llm_rag/pipeline.py — Offline RAG Pipeline
===========================================
Orchestrates the full query-to-answer pipeline using only local components:
  1. parse_query()       — deterministic keyword/regex parser (no LLM)
  2. retrieve_segments() — FAISS + metadata search (local)
  3. generate_answer()   — template-based answer generator (no LLM)

No GEMINI_API_KEY or any cloud API is required.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from .schemas import M5Answer, StructuredQuery
from .query_parser import parse_query
from .retrieval_stub import retrieve_segments
from .generator import generate_answer, format_segment

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    user_query: str,
    *,
    client=None,        # Ignored — kept for backward compatibility
    model: str = "",    # Ignored — no LLM used
    top_k: int = 5,
    verbose: bool = False,
) -> M5Answer:
    """
    Full offline RAG pipeline for one user query.

    Steps
    -----
    1. parse_query()       — structured query via keyword/regex
    2. retrieve_segments() — top-K segments from FAISS retrieval
    3. generate_answer()   — grounded template answer

    Parameters
    ----------
    user_query : Natural-language search query.
    client     : Ignored (backward compatibility only).
    model      : Ignored (backward compatibility only).
    top_k      : Number of candidate segments to retrieve.
    verbose    : If True, logs structured query and segment scores.

    Returns
    -------
    M5Answer with answer_text, cited_segments, explanation_bullets, confidence.
    """
    t0 = time.perf_counter()
    log.info("=== llm_rag pipeline START  query=%r", user_query)

    # Step 1: Local query parsing
    log.info("Step 1/3: Parsing query (local, offline)")
    structured_query = parse_query(user_query)
    if verbose:
        log.info("  Structured query: %s", structured_query.summary())

    # Step 2: Retrieval
    log.info("Step 2/3: Retrieving segments (top_k=%d)", top_k)
    segments = retrieve_segments(structured_query, user_query, top_k=top_k)
    if verbose:
        for i, seg in enumerate(segments):
            score = (seg.get("scores") or {}).get("final_score", "n/a")
            log.info("  Segment %d: %s score=%s", i + 1, seg.get("segment_id"), score)

    # Step 3: Local answer generation
    log.info("Step 3/3: Generating grounded answer (local templates)")
    answer = generate_answer(user_query, segments, structured_query)

    elapsed = time.perf_counter() - t0
    log.info(
        "=== llm_rag pipeline DONE  no_match=%s  confidence=%s  time=%.2fs",
        answer.no_match, answer.confidence, elapsed,
    )
    return answer


def format_answer_for_display(answer: M5Answer) -> str:
    """Format an M5Answer as a human-readable string for CLI or frontend display."""
    lines: list[str] = []

    if answer.no_match:
        lines.append("[NO STRONG MATCH]")
        lines.append(answer.answer_text)
    else:
        if answer.cited_segments:
            seg = answer.cited_segments[0]
            score_str = f"  (score: {seg.final_score:.2f})" if seg.final_score is not None else ""
            lines.append(f"[BEST MATCH] {seg.video_id}  {seg.timestamp_range()}{score_str}")
        lines.append(f"\n{answer.answer_text}")

    if answer.explanation_bullets:
        lines.append("\nEvidence:")
        for b in answer.explanation_bullets:
            lines.append(f"  - {b}")

    lines.append(f"\nConfidence: {answer.confidence.upper()}")
    return "\n".join(lines)
