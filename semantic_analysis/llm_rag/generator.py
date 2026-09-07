"""
llm_rag/generator.py — Local Template-Based Answer Generator
=============================================================
Replaces the Gemini-based answer generator with deterministic
template logic grounded in retrieved segment evidence.

Works FULLY OFFLINE. No cloud LLM. No API key required.

Design principles:
- Never hallucinate: only use facts present in retrieved segments.
- Return timestamped evidence for every claim.
- For high-confidence temporal events, produce natural-language summaries.
- For low-confidence, say so explicitly.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from .schemas import CitedSegment, M5Answer, StructuredQuery

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

NO_MATCH_THRESHOLD: float = float(os.getenv("M5_NO_MATCH_THRESHOLD", "0.26"))
MAX_CONTEXT_SEGMENTS: int = int(os.getenv("M5_MAX_CONTEXT_SEGS", "5"))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (kept from original for compatibility)
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_ts(t: float) -> str:
    m, s = divmod(int(t), 60)
    return f"{m:02d}:{s:02d}"


def _best_score(segments: list[dict[str, Any]]) -> float:
    scores = [(s.get("scores") or {}).get("final_score") or 0.0 for s in segments]
    return max(scores, default=0.0)


def _is_no_match(segments: list[dict[str, Any]]) -> bool:
    if not segments:
        return True
    return _best_score(segments) < NO_MATCH_THRESHOLD


def format_segment(seg: dict[str, Any]) -> str:
    """Compact, structured per-segment text (kept for compatibility with pipeline.py)."""
    start = seg.get("start_ts", 0.0)
    end = seg.get("end_ts", 0.0)
    ts_range = f"{_fmt_ts(start)}-{_fmt_ts(end)}"
    visual_desc = (seg.get("visual") or {}).get("description", "n/a")
    transcript = (seg.get("audio") or {}).get("transcript", "")
    if transcript and len(transcript) > 300:
        transcript = transcript[:297] + "..."
    transcript_display = f'"{transcript}"' if transcript else "none"
    objects = seg.get("objects", [])
    obj_strs = [f"{o.get('label', '?')} [{o.get('confidence', 0.0):.2f}]" for o in objects]
    ocr = seg.get("ocr", [])
    scores = seg.get("scores", {})
    score_parts = [
        f"{lbl}={scores[key]:.2f}"
        for key, lbl in [("final_score", "final"), ("visual_similarity", "visual")]
        if scores.get(key) is not None
    ]
    return (
        f"Segment: {seg.get('video_id', '?')}, {ts_range}\n"
        f"Visual: {visual_desc}\n"
        f"Transcript: {transcript_display}\n"
        f"Objects: {', '.join(obj_strs) if obj_strs else 'none'}\n"
        f"OCR: {', '.join(ocr) if ocr else 'none'}\n"
        f"Scores: {', '.join(score_parts) if score_parts else 'not scored'}"
    )


def check_attribute_binding(
    structured_query: StructuredQuery,
    segment: dict[str, Any],
) -> list[str]:
    """Check that attributed objects in the query appear in the segment."""
    warnings: list[str] = []
    seg_objects = segment.get("objects", [])
    for qobj in structured_query.objects:
        if not qobj.attributes:
            continue
        matching = [o for o in seg_objects if o.get("label", "").lower() == qobj.object.lower()]
        if not matching:
            warnings.append(f"Query requires '{qobj.object}' but not detected in segment")
            continue
        seg_attrs = set()
        for m in matching:
            seg_attrs.update(a.lower() for a in m.get("attributes", []))
        for attr in qobj.attributes:
            if attr.lower() not in seg_attrs:
                warnings.append(
                    f"Attribute '{attr}' bound to '{qobj.object}' not found in segment "
                    f"(found: {list(seg_attrs) or 'none'})"
                )
    return warnings


# ─────────────────────────────────────────────────────────────────────────────
# Template answer builders
# ─────────────────────────────────────────────────────────────────────────────

def _detect_intent(structured_query: StructuredQuery) -> str:
    """Detect high-level user intent from the parsed query."""
    actions = {a.lower() for a in structured_query.actions}
    objects = {o.object.lower() for o in structured_query.objects}
    context = {c.lower() for c in structured_query.context}

    if "depart" in actions and ("vehicle" in objects or "car" in objects):
        return "vehicle_departure"
    if "arrive" in actions and ("vehicle" in objects or "car" in objects):
        return "vehicle_arrival"
    if "exit" in actions and "person" in {p.lower() for p in structured_query.persons}:
        return "person_exits_vehicle"
    if "exit" in actions or "depart" in actions:
        return "departure_generic"
    if "arrive" in actions or "stop" in actions:
        return "arrival_generic"
    if "enter" in actions:
        return "enter_vehicle"
    if structured_query.speech_intent:
        return "speech_query"
    if "license_plate" in objects:
        return "plate_query"
    if "timestamp_query" in context:
        return "temporal_query"
    if "walk" in actions or "move" in actions or "run" in actions:
        return "movement_query"
    return "general"


def _build_answer_from_segments(
    intent: str,
    segments: list[dict[str, Any]],
    user_query: str,
) -> tuple[str, list[str]]:
    """
    Build a grounded natural-language answer from retrieved segments.
    Returns (answer_text, explanation_bullets).
    Never fabricates information not present in segments.
    """
    if not segments:
        return ("No matching evidence found for this query.", [
            "No segments were retrieved above the confidence threshold.",
            "Try adjusting the query or uploading more video data.",
        ])

    top = segments[0]
    start = top.get("start_ts", 0.0)
    end = top.get("end_ts", 0.0)
    video_id = top.get("video_id", "unknown")
    score = (top.get("scores") or {}).get("final_score", 0.0) or 0.0

    ts_range = f"{_fmt_ts(start)}–{_fmt_ts(end)}"
    ts_secs = f"{start:.1f}s–{end:.1f}s"

    # Gather supporting evidence for bullets
    visual_desc = (top.get("visual") or {}).get("description", "")
    transcript = (top.get("audio") or {}).get("transcript", "")
    objects_found = [o.get("label", "") for o in top.get("objects", []) if o.get("label")]
    ocr_texts = top.get("ocr", [])

    bullets: list[str] = []

    if intent == "vehicle_departure":
        answer = (
            f"The vehicle appears to depart at approximately {ts_range} "
            f"(~{ts_secs}) in video '{video_id}'."
        )
        bullets.append(f"Best matching segment: {video_id} at {ts_range} (score: {score:.2f})")
        if visual_desc:
            bullets.append(f"Visual evidence: {visual_desc}")
        if "car" in objects_found or "vehicle" in objects_found:
            bullets.append("A vehicle was detected in the matched segment.")
        if transcript:
            bullets.append(f'Audio: "{transcript[:120]}"')

    elif intent == "vehicle_arrival":
        answer = (
            f"The vehicle appears to arrive at approximately {ts_range} "
            f"(~{ts_secs}) in video '{video_id}'."
        )
        bullets.append(f"Best matching segment: {video_id} at {ts_range} (score: {score:.2f})")
        if visual_desc:
            bullets.append(f"Visual evidence: {visual_desc}")

    elif intent == "person_exits_vehicle":
        answer = (
            f"A person appears to exit the vehicle at approximately {ts_range} "
            f"(~{ts_secs}) in video '{video_id}'."
        )
        bullets.append(f"Best matching segment: {video_id} at {ts_range} (score: {score:.2f})")
        if "person" in objects_found:
            bullets.append("A person was detected in the matched segment.")
        if visual_desc:
            bullets.append(f"Visual evidence: {visual_desc}")

    elif intent == "speech_query":
        if transcript:
            answer = (
                f"Speech content found at {ts_range} (~{ts_secs}) in video '{video_id}': "
                f'"{transcript[:200]}"'
            )
            bullets.append(f"Transcript at {ts_range}: \"{transcript}\"")
            bullets.append(f"Segment score: {score:.2f}")
        else:
            answer = (
                f"The closest matching segment is at {ts_range} in video '{video_id}', "
                "but no transcript was found in that interval."
            )
            bullets.append("No audio transcript available for the top-matched segment.")

    elif intent == "plate_query":
        # Look for OCR evidence of plate text
        plate_evidence = [t for t in ocr_texts if len(t) >= 5 and any(c.isdigit() for c in t)]
        if plate_evidence:
            answer = (
                f"License plate evidence found at {ts_range} in video '{video_id}': "
                f"{', '.join(plate_evidence)}"
            )
            bullets.append(f"OCR text detected: {', '.join(plate_evidence)}")
        else:
            answer = (
                f"The closest matching segment for the plate query is at {ts_range} "
                f"in video '{video_id}' (score: {score:.2f}). "
                "No plate text was clearly detected by OCR."
            )
        bullets.append(f"Segment: {video_id} at {ts_range}")

    else:
        # General fallback — describe what was found
        answer = (
            f"The most relevant segment found is at {ts_range} (~{ts_secs}) "
            f"in video '{video_id}' with a relevance score of {score:.2f}."
        )
        if visual_desc:
            bullets.append(f"Visual description: {visual_desc}")
        if transcript:
            bullets.append(f'Transcript: "{transcript[:120]}"')
        if objects_found:
            bullets.append(f"Objects detected: {', '.join(objects_found[:5])}")
        if ocr_texts:
            bullets.append(f"Text visible: {', '.join(ocr_texts[:3])}")
        bullets.append(f"Relevance score: {score:.2f}")

    # Always note additional matching segments
    if len(segments) > 1:
        others = segments[1:min(4, len(segments))]
        ranges = [f"{_fmt_ts(s.get('start_ts', 0))}–{_fmt_ts(s.get('end_ts', 0))}" for s in others]
        bullets.append(f"Additional candidate segments: {', '.join(ranges)}")

    return answer, bullets


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_answer(
    user_query: str,
    segments: list[dict[str, Any]],
    structured_query: StructuredQuery,
    *,
    # These keyword args are accepted but IGNORED for backward compatibility
    client=None,
    model: str = "",
    max_tokens: int = 1024,
) -> M5Answer:
    """
    Generate a grounded answer from retrieved segments using local templates.
    NO cloud LLM. NO API key. Works fully offline.

    Parameters
    ----------
    user_query       : Original user query string.
    segments         : Top-K segments from retrieval (COMMON_DATA_CONTRACT format).
    structured_query : Parsed query from parse_query().
    client           : Ignored (compatibility parameter).
    model            : Ignored (compatibility parameter).
    max_tokens       : Ignored (compatibility parameter).

    Returns
    -------
    M5Answer with answer_text, cited_segments, explanation_bullets, no_match, confidence.
    """
    best = _best_score(segments)
    from semantic_analysis.llm_rag.retrieval.config import DEFAULT_CONFIG as ret_cfg
    conf_thresh = getattr(ret_cfg, 'confidence_threshold', 0.26)

    if best >= 0.60:
        confidence = "HIGH"
    elif best >= conf_thresh:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    # No-match fast path
    if confidence == "LOW":
        log.info("Low confidence (best=%.2f). Returning no-strong-match answer.", best)
        cited = []
        if segments:
            top_seg = segments[0]
            cited.append(CitedSegment(
                video_id=top_seg.get("video_id", ""),
                segment_id=top_seg.get("segment_id", ""),
                start_ts=top_seg.get("start_ts", 0.0),
                end_ts=top_seg.get("end_ts", 0.0),
                final_score=(top_seg.get("scores") or {}).get("final_score"),
            ))
        return M5Answer(
            answer_text="No strong match found for this query in the available video corpus.",
            cited_segments=cited,
            explanation_bullets=[
                f"Highest retrieved score ({best:.3f}) is below the confidence threshold ({conf_thresh:.2f}).",
                "Showing closest available segments for transparency.",
            ],
            no_match=False,
            confidence="LOW",
            structured_query=structured_query,
        )

    # Attribute-binding check
    binding_warnings = check_attribute_binding(structured_query, segments[0]) if segments else []
    for w in binding_warnings:
        log.info("Attribute-binding warning: %s", w)

    # Detect intent and build template answer
    intent = _detect_intent(structured_query)
    context_segs = segments[:MAX_CONTEXT_SEGMENTS]
    answer_text, bullets = _build_answer_from_segments(intent, context_segs, user_query)

    if binding_warnings:
        bullets.extend(f"[Attribute-binding warning] {w}" for w in binding_warnings)

    # Build cited segments list
    cited: list[CitedSegment] = []
    for seg in context_segs:
        cited.append(CitedSegment(
            video_id=seg.get("video_id", ""),
            segment_id=seg.get("segment_id", ""),
            start_ts=seg.get("start_ts", 0.0),
            end_ts=seg.get("end_ts", 0.0),
            final_score=(seg.get("scores") or {}).get("final_score"),
        ))

    log.info(
        "generate_answer(%r) -> intent=%s confidence=%s bullets=%d",
        user_query, intent, confidence, len(bullets),
    )
    return M5Answer(
        answer_text=answer_text,
        cited_segments=cited,
        explanation_bullets=bullets,
        no_match=False,
        confidence=confidence,
        structured_query=structured_query,
    )
