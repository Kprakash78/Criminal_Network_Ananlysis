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


VEHICLE_LABELS = {
    "vehicle", "car", "truck", "bus", "van", "motorcycle", "motorbike",
    "bike", "bicycle", "auto", "automobile", "suv", "jeep", "cab", "taxi",
}
PERSON_LABELS = {
    "person", "people", "man", "woman", "individual", "pedestrian",
    "driver", "passenger", "occupant", "thief", "suspect", "criminal",
}


def _label_aliases(label: str) -> Set[str]:
    label = label.lower().strip()
    if label in VEHICLE_LABELS:
        return VEHICLE_LABELS
    if label in PERSON_LABELS:
        return PERSON_LABELS
    if label == "license_plate":
        return {"license_plate", "plate", "number_plate", "registration"}
    return {label}


def _labels_match(query_label: str, segment_label: str) -> bool:
    return bool(_label_aliases(query_label) & _label_aliases(segment_label))


def _segment_labels(segment: Dict[str, Any]) -> Set[str]:
    return {
        str(obj.get("label", obj.get("object", ""))).lower().strip()
        for obj in segment.get("objects", [])
        if str(obj.get("label", obj.get("object", ""))).strip()
    }


def _has_label_family(segment: Dict[str, Any], labels: Set[str]) -> bool:
    return any(_labels_match(label, wanted) for label in _segment_labels(segment) for wanted in labels)


def _segment_midpoint(segment: Dict[str, Any]) -> float:
    start = float(segment.get("start_ts", 0.0) or 0.0)
    end = float(segment.get("end_ts", start) or start)
    return (start + end) / 2.0


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

    # Build list of (lowercase_label, attributes) entries from the segment.
    # Using entries instead of a plain dict lets query labels such as "vehicle"
    # match detector labels such as "car", "truck", or "bus".
    seg_objects: List[tuple[str, Set[str]]] = []
    for obj in segment_objects:
        label = str(obj.get("label", obj.get("object", ""))).lower().strip()
        if not label:
            continue
        attrs = set(str(a).lower().strip() for a in obj.get("attributes", []))
        seg_objects.append((label, attrs))

    hits = 0.0
    total_query_objects = len(structured_query_objects)

    for q_obj in structured_query_objects:
        q_label = str(q_obj.get("object", q_obj.get("label", ""))).lower().strip()
        if not q_label:
            continue

        matching_attr_sets = [
            attrs for seg_label, attrs in seg_objects
            if _labels_match(q_label, seg_label)
        ]
        if matching_attr_sets:
            q_attrs = set(str(a).lower().strip() for a in q_obj.get("attributes", []))
            if not q_attrs:
                # No attributes requested -> full hit on object label match
                hits += 1.0
            else:
                # Attribute-binding check (PRD §8):
                # Verify at least one detected instance of the object has matching attribute(s)
                bound_match = any(bool(q_attrs & seg_attr_set) for seg_attr_set in matching_attr_sets)
                if bound_match:
                    hits += 1.0
                else:
                    # Partial match if object is present but attribute binding failed
                    hits += 0.5

    score = hits / total_query_objects
    return min(1.0, max(0.0, score))


def action_event_score(
    structured_query: Dict[str, Any] | None,
    segment: Dict[str, Any],
    all_segments: Sequence[Dict[str, Any]] | None,
) -> float:
    """
    Score temporal/event intent that object matching alone cannot express.

    Detector metadata says what is visible in each segment; it does not emit
    verbs like "arrive" or "depart". For demo-critical CCTV queries we infer:
    - arrival/stop: first segment where the vehicle appears
    - departure/leave: last segment where the vehicle is still visible
    - exit/get out: segment where people and a vehicle co-occur
    """
    if not structured_query:
        return 0.0

    actions = {
        str(action).lower().strip()
        for action in structured_query.get("actions", [])
        if str(action).strip()
    }
    if not actions:
        return 0.0

    video_id = segment.get("video_id")
    candidates = [
        seg for seg in (all_segments or [])
        if seg.get("video_id") == video_id
    ]
    candidates.sort(key=_segment_midpoint)
    if not candidates:
        candidates = [segment]

    has_vehicle = _has_label_family(segment, VEHICLE_LABELS)
    has_person = _has_label_family(segment, PERSON_LABELS)
    vehicle_segments = [seg for seg in candidates if _has_label_family(seg, VEHICLE_LABELS)]
    person_segments = [seg for seg in candidates if _has_label_family(seg, PERSON_LABELS)]

    scores: list[float] = []

    if actions & {"arrive", "stop"}:
        if has_vehicle and vehicle_segments:
            first_vehicle = vehicle_segments[0]
            scores.append(1.0 if segment is first_vehicle else 0.55)
        elif segment is candidates[0]:
            scores.append(0.25)

    if "depart" in actions:
        if has_vehicle and vehicle_segments:
            last_vehicle = vehicle_segments[-1]
            if segment is last_vehicle:
                scores.append(1.0)
            else:
                denom = max(1, len(vehicle_segments) - 1)
                idx = vehicle_segments.index(segment) if segment in vehicle_segments else 0
                scores.append(0.35 + 0.45 * (idx / denom))
        elif person_segments and segment is person_segments[-1]:
            scores.append(0.5)

    if actions & {"exit", "enter"}:
        if has_person and has_vehicle:
            scores.append(0.95)
        elif has_person:
            idx = candidates.index(segment) if segment in candidates else 0
            prev_has_vehicle = any(
                _has_label_family(prev, VEHICLE_LABELS)
                for prev in candidates[max(0, idx - 2):idx]
            )
            if prev_has_vehicle:
                scores.append(0.75)

    if actions & {"walk", "run", "move"} and has_person:
        scores.append(0.65)

    return round(max(scores, default=0.0), 3)


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

    # License-plate OCR commonly confuses O/0, I/1, and Z/2. Compare a
    # normalized alphanumeric form only for explicit plate queries so regular
    # OCR matching remains exact and conservative.
    plate_terms = []
    if structured_query:
        for q_obj in structured_query.get("objects", []):
            if isinstance(q_obj, dict) and q_obj.get("object") == "license_plate":
                plate_terms.extend(str(attr) for attr in q_obj.get("attributes", []))
    if plate_terms:
        normalize_plate = lambda value: "".join(
            {"O": "0", "Q": "0", "I": "1", "L": "1", "Z": "2"}.get(ch, ch)
            for ch in str(value).upper()
            if ch.isalnum()
        )
        normalized_ocr = {normalize_plate(item) for item in segment_ocr}
        if any(normalize_plate(term) in normalized_ocr for term in plate_terms):
            return 1.0

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
    event_match: float = 0.0,
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
        + float(getattr(cfg, "w_event", 0.25)) * max(0.0, event_match)
    )
    return round(float(min(1.0, max(0.0, score))), 3)
