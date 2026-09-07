"""
M4 Retrieval — Segment Joiner
Joins records from M1 (visual), M2 (audio), and M3 (objects/OCR) into unified
segment-level records matching COMMON_DATA_CONTRACT.md based on video_id and
time-range overlap.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from .config import RetrievalConfig as RetrievalConfig, DEFAULT_CONFIG

log = logging.getLogger(__name__)


def compute_overlap_ratio(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Compute Intersection-over-Union (IoU) overlap fraction between two time intervals."""
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return (inter / union) if union > 0 else 0.0


def join_segment_records(
    visual_records: List[Dict[str, Any]],
    audio_records: List[Dict[str, Any]],
    object_records: List[Dict[str, Any]],
    ocr_records: List[Dict[str, Any]] | None = None,
    cfg: RetrievalConfig = DEFAULT_CONFIG,
) -> List[Dict[str, Any]]:
    """
    Merge multi-modal records keyed by video_id and timestamps into unified segment records.

    Parameters:
      visual_records: list of dicts from M1 (or metadata.jsonl with 'visual' key)
      audio_records: list of dicts from M2 with 'audio' key
      object_records: list of dicts from M3 with 'objects' key
      ocr_records: optional separate list of OCR dicts from M3

    Returns:
      List of unified segment-level records adhering to COMMON_DATA_CONTRACT.md schema.
    """
    overlap_threshold = cfg.overlap_threshold
    unified_segments: List[Dict[str, Any]] = []

    # Case A: Visual records exist (standard pipeline flow)
    if visual_records:
        for idx, vis in enumerate(visual_records):
            video_id = vis["video_id"]
            start_ts = float(vis["start_ts"])
            end_ts = float(vis["end_ts"])

            seg_id = vis.get("segment_id", f"{video_id}_seg_{idx:02d}")

            unified = {
                "video_id": video_id,
                "segment_id": seg_id,
                "start_ts": start_ts,
                "end_ts": end_ts,
                "visual": vis.get("visual"),
                "audio": vis.get("audio"),
                "objects": list(vis.get("objects", [])),
                "ocr": list(vis.get("ocr", [])),
                "scores": {
                    "visual_similarity": None,
                    "transcript_similarity": None,
                    "object_match": None,
                    "ocr_match": None,
                    "final_score": None,
                },
            }

            # 1. Join Audio if not already populated
            if not unified["audio"] and audio_records:
                best_audio = None
                best_overlap = 0.0
                for aud in audio_records:
                    if aud["video_id"] == video_id:
                        a_start = float(aud.get("start_ts", aud.get("timestamp", start_ts)))
                        a_end = float(aud.get("end_ts", a_start + 1.0))
                        ratio = compute_overlap_ratio(start_ts, end_ts, a_start, a_end)
                        if ratio >= overlap_threshold and ratio > best_overlap:
                            best_overlap = ratio
                            aud_dict = dict(aud.get("audio", {})) if isinstance(aud.get("audio"), dict) else {}
                            if "_embedding_vector" in aud:
                                aud_dict["_embedding_vector"] = aud["_embedding_vector"]
                            best_audio = aud_dict
                if best_audio:
                    unified["audio"] = best_audio

            # 2. Join Objects from M3 object_records if provided separately
            if object_records:
                for obj_rec in object_records:
                    if obj_rec["video_id"] == video_id:
                        o_ts = float(obj_rec.get("timestamp", start_ts))
                        if start_ts <= o_ts <= end_ts:
                            if "objects" in obj_rec:
                                unified["objects"].extend(obj_rec["objects"])
                            if "ocr" in obj_rec:
                                unified["ocr"].extend(obj_rec["ocr"])

            # 3. Join OCR from ocr_records if provided separately
            if ocr_records:
                for ocr_rec in ocr_records:
                    if ocr_rec["video_id"] == video_id:
                        ocr_ts = float(ocr_rec.get("timestamp", start_ts))
                        if start_ts <= ocr_ts <= end_ts:
                            if "ocr" in ocr_rec:
                                unified["ocr"].extend(ocr_rec["ocr"])

            # Dedup objects and ocr strings
            unified["ocr"] = list(dict.fromkeys(unified["ocr"]))
            unified_segments.append(unified)

    # Case B: Only Audio records exist
    elif audio_records:
        for idx, aud in enumerate(audio_records):
            video_id = aud["video_id"]
            start_ts = float(aud.get("start_ts", 0.0))
            end_ts = float(aud.get("end_ts", start_ts + 2.0))
            seg_id = aud.get("segment_id", f"{video_id}_seg_{idx:02d}")

            unified = {
                "video_id": video_id,
                "segment_id": seg_id,
                "start_ts": start_ts,
                "end_ts": end_ts,
                "visual": None,
                "audio": (
                    dict(aud["audio"], _embedding_vector=aud["_embedding_vector"])
                    if isinstance(aud.get("audio"), dict) and "_embedding_vector" in aud
                    else aud.get("audio")
                ),
                "objects": [],
                "ocr": [],
                "scores": {
                    "visual_similarity": None,
                    "transcript_similarity": None,
                    "object_match": None,
                    "ocr_match": None,
                    "final_score": None,
                },
            }
            unified_segments.append(unified)

    log.info("Joined multi-modal inputs into %d unified segment records", len(unified_segments))
    return unified_segments
