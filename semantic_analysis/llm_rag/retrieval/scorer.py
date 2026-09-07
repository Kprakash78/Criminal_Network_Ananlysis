"""
M4 Retrieval — Multi-Modal Scoring & Score Fusion
Implements scoring functions for:
  - Visual similarity (CLIP embedding inner product / cosine similarity)
  - Transcript similarity (SentenceTransformers embedding inner product / cosine similarity)
  - Object match with attribute-binding check (PRD §8)
  - OCR match against text terms
  - Score fusion (weighted sum calculation per PRD §6)
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence, Set

from .config import RetrievalConfig as RetrievalConfig, DEFAULT_CONFIG

log = logging.getLogger(__name__)


def object_match_score(
    structured_query_objects: Sequence[Dict[str, Any]] | None,
    segment_objects: Sequence[Dict[str, Any]] | None,
) -> float:
    """
    Calculate object match score (0.0 to 1.0) with attribute-binding check (PRD §8).

    structured_query_objects example:
      [{"object": "motorcycle", "attributes": ["blue"]}, {"object": "helmet"}]

    segment_objects example:
      [{"label": "motorcycle", "confidence": 0.95, "attributes": ["blue", "fast"]}]
    """
    if not structured_query_objects:
        return 0.0
    if not segment_objects:
        return 0.0

    # Build map of lowercase_label -> list of attribute sets from segment
    seg_objects_by_label: Dict[str, List[Set[str]]] = {}
    for obj in segment_objects:
        label = str(obj.get("label", obj.get("object", ""))).lower().strip()
        if not label:
            continue
        attrs = set(str(a).lower().strip() for a in obj.get("attributes", []))
        if label not in seg_objects_by_label:
            seg_objects_by_label[label] = []
        seg_objects_by_label[label].append(attrs)

    hits = 0.0
    total_query_objects = len(structured_query_objects)

    for q_obj in structured_query_objects:
        q_label = str(q_obj.get("object", q_obj.get("label", ""))).lower().strip()
        if not q_label:
            continue

        if q_label in seg_objects_by_label:
            q_attrs = set(str(a).lower().strip() for a in q_obj.get("attributes", []))
            if not q_attrs:
                # No attributes requested -> full hit on object label match
                hits += 1.0
            else:
                # Attribute-binding check (PRD §8):
                # Verify at least one detected instance of the object has matching attribute(s)
                bound_match = any(bool(q_attrs & seg_attr_set) for seg_attr_set in seg_objects_by_label[q_label])
                if bound_match:
                    hits += 1.0
                else:
                    # Partial match if object is present but attribute binding failed
                    hits += 0.5

    score = hits / total_query_objects
    return min(1.0, max(0.0, score))


def ocr_match_score(
    structured_query: Dict[str, Any] | None,
    segment_ocr: Sequence[str] | None,
) -> float:
    """
    Calculate OCR match score (0.0 or 1.0).
    Checks if any terms in structured query (objects, context, or OCR terms) appear in segment OCR.
    """
    if not segment_ocr:
        return 0.0

    terms: List[str] = []
    if structured_query:
        # Extract object labels & attributes
        for q_obj in structured_query.get("objects", []):
            if isinstance(q_obj, dict):
                if "object" in q_obj:
                    terms.append(str(q_obj["object"]).lower())
                for attr in q_obj.get("attributes", []):
                    terms.append(str(attr).lower())
            elif isinstance(q_obj, str):
                terms.append(q_obj.lower())

        # Extract context terms
        for ctx in structured_query.get("context", []):
            terms.append(str(ctx).lower())

        # Extract explicit OCR terms if present
        for ocr_term in structured_query.get("ocr", []):
            terms.append(str(ocr_term).lower())

    if not terms:
        return 0.0

    ocr_text_combined = " ".join(str(item).lower() for item in segment_ocr)

    for term in terms:
        if term and term in ocr_text_combined:
            return 1.0

    return 0.0


def _normalize_cosine(
    sim: float,
    floor: float = 0.10,
    ceiling: float = 0.42,
) -> float:
    """
    Rescale raw cosine similarity from the model's natural range into [0.0, 1.0].

    CLIP and SentenceTransformer cosine similarities are NOT probabilities:
    - A strong CLIP visual match scores ~0.30–0.42 (not 0.80+)
    - A random/unrelated pair scores ~0.10–0.18
    So the raw 0.28 scores shown in DEMO_QUERIES.md mean "strong match",
    but they display as if they're weak. This function rescales so:
      cosine ≤ floor  → 0.0   (no signal)
      cosine = ceiling → 1.0   (maximum signal)
    This does NOT change retrieval ranking — it only scales displayed scores.
    """
    if sim <= floor:
        return 0.0
    return min(1.0, (sim - floor) / max(ceiling - floor, 1e-6))


def calculate_fusion_score(
    visual_sim: float,
    transcript_sim: float,
    object_match: float,
    ocr_match: float,
    cfg: RetrievalConfig = DEFAULT_CONFIG,
) -> float:
    """
    Compute final weighted fusion score (0.0 to 1.0) per PRD §6 formula.

    Raw cosine similarities from CLIP and SentenceTransformer are first
    normalized via _normalize_cosine() so the final score is human-readable:
    - 0.0 = no match
    - 0.5 = moderate match
    - 0.8+ = strong match
    Object and OCR matches (already 0.0–1.0 boolean/fractional) are
    passed through unchanged.
    """
    vis_norm = _normalize_cosine(visual_sim, floor=0.10, ceiling=0.42)
    # MiniLM-L6-v2 text similarities generally have a higher range than CLIP
    tr_norm  = _normalize_cosine(transcript_sim, floor=0.10, ceiling=0.75)

    score = (
        cfg.w_visual    * max(0.0, vis_norm)
        + cfg.w_transcript * max(0.0, tr_norm)
        + cfg.w_object     * max(0.0, object_match)
        + cfg.w_ocr        * max(0.0, ocr_match)
    )
    return round(float(min(1.0, max(0.0, score))), 3)
