"""
M1 Video / VLM — Segment Aggregator
Groups consecutive frame-level embeddings into segment-level records based on
cosine similarity thresholding (BACKEND.md §7). Prevents segment fragmentation
and duplicate embeddings while creating semantically meaningful scene boundaries.
"""
from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Dict, List

import numpy as np

from .config import VideoConfig, DEFAULT_CONFIG

log = logging.getLogger(__name__)


def aggregate_frames_to_segments(
    frame_records: List[Dict[str, Any]],
    video_id: str,
    cfg: VideoConfig = DEFAULT_CONFIG,
) -> List[Dict[str, Any]]:
    """
    Aggregate a sorted list of frame records into segment-level records.

    frame_records item structure:
      {
        'timestamp': float,
        'embedding': np.ndarray (normalized 1D),
        'description': str,
        'top_labels': list[str],
      }

    Returns segment records conforming to COMMON_DATA_CONTRACT.md's visual section:
      {
        "video_id": video_id,
        "segment_id": f"{video_id}_seg_{idx:02d}",
        "start_ts": float,
        "end_ts": float,
        "visual": {
          "description": str,
          "embedding_id": f"vis_{video_id}_{idx:02d}",
          "quality_flag": "ok"
        },
        "scores": {
          "visual_similarity": None,
          "transcript_similarity": None,
          "object_match": None,
          "ocr_match": None,
          "final_score": None
        },
        "_embedding_vector": np.ndarray (L2-normalized centroid),
      }
    """
    if not frame_records:
        log.warning("No frame records provided for aggregation of video_id=%s", video_id)
        return []

    # Sort frame records chronologically by timestamp
    frame_records = sorted(frame_records, key=lambda r: r["timestamp"])

    sim_threshold = cfg.segment_sim_threshold
    interval = cfg.frame_interval_sec
    max_seg_sec = cfg.max_segment_sec

    raw_segments: List[Dict[str, Any]] = []
    current_segment: Dict[str, Any] | None = None

    for rec in frame_records:
        vec = rec["embedding"]
        ts = rec["timestamp"]
        labels = rec.get("top_labels", [])

        if current_segment is None:
            current_segment = {
                "start_ts": ts,
                "end_ts": ts,
                "vectors": [vec],
                "timestamps": [ts],
                "labels_list": [labels],
                "descriptions": [rec["description"]],
            }
        else:
            # Duration gate: force new segment if max duration would be exceeded
            duration_so_far = ts - current_segment["start_ts"]
            over_duration = duration_so_far > max_seg_sec

            # Check similarity between new vector and current segment centroid
            centroid = np.mean(current_segment["vectors"], axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm

            sim = float(np.dot(centroid, vec))

            if sim >= sim_threshold and not over_duration:
                current_segment["end_ts"] = ts
                current_segment["vectors"].append(vec)
                current_segment["timestamps"].append(ts)
                current_segment["labels_list"].append(labels)
                current_segment["descriptions"].append(rec["description"])
            else:
                raw_segments.append(current_segment)
                current_segment = {
                    "start_ts": ts,
                    "end_ts": ts,
                    "vectors": [vec],
                    "timestamps": [ts],
                    "labels_list": [labels],
                    "descriptions": [rec["description"]],
                }

    if current_segment:
        raw_segments.append(current_segment)

    # Build schema-compliant segment records
    segment_records: List[Dict[str, Any]] = []
    for idx, seg in enumerate(raw_segments):
        start_ts = round(seg["start_ts"], 2)
        end_ts = round(seg["end_ts"], 2)
        if start_ts == end_ts:
            end_ts = round(start_ts + interval, 2)

        # Calculate average embedding and L2 normalize
        mean_vec = np.mean(seg["vectors"], axis=0)
        norm = np.linalg.norm(mean_vec)
        if norm > 0:
            centroid_vec = (mean_vec / norm).astype(np.float32)
        else:
            centroid_vec = mean_vec.astype(np.float32)

        # Determine segment description by selecting top labels weighted across frames
        label_counter: Counter = Counter()
        for frame_labels in seg["labels_list"]:
            for pos, lbl in enumerate(frame_labels):
                # Weight earlier top labels slightly higher
                label_counter[lbl] += (len(frame_labels) - pos)

        top_seg_labels = [lbl for lbl, _ in label_counter.most_common(cfg.description_top_k)]
        raw_desc = ", ".join(top_seg_labels) if top_seg_labels else seg["descriptions"][0]
        # Clean & capitalize description
        description = raw_desc[0].upper() + raw_desc[1:] if raw_desc else "Visual scene"

        seg_id = f"{video_id}_seg_{idx:02d}"
        emb_id = f"vis_{video_id}_{idx:02d}"

        rec = {
            "video_id": video_id,
            "segment_id": seg_id,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "visual": {
                "description": description,
                "embedding_id": emb_id,
                "quality_flag": "ok",
            },
            "scores": {
                "visual_similarity": None,
                "transcript_similarity": None,
                "object_match": None,
                "ocr_match": None,
                "final_score": None,
            },
            "_embedding_vector": centroid_vec,
        }
        segment_records.append(rec)

    log.info(
        "Aggregated %d frames into %d segment records for video_id=%s",
        len(frame_records),
        len(segment_records),
        video_id,
    )
    return segment_records
