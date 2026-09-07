"""
M3 Vision Intelligence — Segment Aggregator
Groups consecutive frame-level records into segment-level records per
COMMON_DATA_CONTRACT.md §Segment-level record and BACKEND.md §7.

Algorithm (precision-first, post attribute-localization fix):
  - Walk sorted frame records.
  - Start a new segment when:
      (a) The Jaccard similarity of object-label sets drops below
          cfg.segment_overlap_threshold (default 0.5), OR
      (b) Both the current frame and the ongoing segment have empty object
          sets (empty-empty Jaccard is now 0.0 — previously was 1.0, which
          caused empty frames to be merged into arbitrarily long segments), OR
      (c) The segment duration would exceed cfg.max_segment_sec (default 5s).
  - Within a segment: average confidence per label, union attributes per label,
    dedup OCR strings, report the worst quality_flag seen.

Output format matches COMMON_DATA_CONTRACT.md exactly:
{
  "video_id": str,
  "segment_id": "<video_id>_seg_<NN>",
  "start_ts": float,
  "end_ts": float,
  "visual": {"quality_flag": "ok"|"low_quality"},
  "objects": [{"label": str, "confidence": float, "attributes": [...]}],
  "ocr": [str],
  "scores": {"visual_similarity": null, "transcript_similarity": null,
             "object_match": null, "ocr_match": null, "final_score": null}
}

Note: "visual.description" and "visual.embedding_id" are intentionally left absent
(null / missing) — M1 owns VLM embeddings; M3 only sets quality_flag.
"audio" is absent — M2 owns transcripts.
"scores.*" are null — M4 populates at query time.
"""
from __future__ import annotations

import logging
from typing import Any

from .config import OcrConfig, DEFAULT_CONFIG

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_to_segments(
    frame_records: list[dict[str, Any]],
    video_id: str,
    cfg: OcrConfig = DEFAULT_CONFIG,
) -> list[dict[str, Any]]:
    """
    Convert a list of frame-level records (sorted by timestamp) into
    segment-level records per COMMON_DATA_CONTRACT.md.

    Precision improvements (attribute-localization fix):
    - Empty object sets are no longer merged (Jaccard(∅,∅) = 0.0, not 1.0).
    - Segments are capped at cfg.max_segment_sec seconds (default 5s) to keep
      timestamp ranges tight enough for precise attribute localization.
    """
    if not frame_records:
        return []

    # Ensure records are sorted by timestamp
    frame_records = sorted(frame_records, key=lambda r: r["timestamp"])

    segments: list[dict[str, Any]] = []
    current: _SegmentBuilder | None = None
    seg_idx = 0

    for rec in frame_records:
        labels = {o["label"] for o in rec.get("objects", [])}
        ts = rec["timestamp"]

        if current is None:
            current = _SegmentBuilder(video_id, seg_idx, rec, labels)
        else:
            # Duration gate: force new segment if max duration would be exceeded
            duration_so_far = ts - current.start_ts
            over_duration = duration_so_far > cfg.max_segment_sec

            overlap = _jaccard(labels, current.label_set)
            same_scene = (overlap >= cfg.segment_overlap_threshold) and not over_duration

            if same_scene:
                # Same scene and within duration budget — extend current segment
                current.extend(rec)
            else:
                # Scene changed or duration cap hit — finalise and start fresh
                segments.append(current.build())
                seg_idx += 1
                current = _SegmentBuilder(video_id, seg_idx, rec, labels)

    if current is not None:
        segments.append(current.build())

    log.info(
        "Aggregated %d frame records → %d segments for video_id=%s",
        len(frame_records), len(segments), video_id,
    )
    return segments


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two label sets.

    Returns 0.0 when both sets are empty (precision fix: empty frames should
    NOT extend the current segment indefinitely — they should start a new one).
    Previously this returned 1.0, which caused empty-object frames to be merged
    into arbitrarily long segments (14000s+) whenever the camera went dark or
    no objects were detected.
    """
    union = a | b
    if not union:
        return 0.0  # FIX: was 1.0 — empty-empty must NOT merge
    return len(a & b) / len(union)


class _SegmentBuilder:
    """Accumulates frame records that belong to one segment."""

    def __init__(
        self,
        video_id: str,
        seg_idx: int,
        first_rec: dict[str, Any],
        label_set: set[str],
    ) -> None:
        self.video_id = video_id
        self.seg_idx = seg_idx
        self.start_ts: float = first_rec["timestamp"]
        self.end_ts: float = first_rec["timestamp"]
        self.label_set: set[str] = label_set
        # label → list of (confidence, attributes) tuples for averaging
        self._obj_data: dict[str, list[tuple[float, list[str]]]] = {}
        self._ocr_seen: list[str] = []
        self._quality_flags: list[str] = []
        self._ingest(first_rec)

    def extend(self, rec: dict[str, Any]) -> None:
        """Add another frame to the current segment."""
        self.end_ts = rec["timestamp"]
        # Update label_set as a union — we capture all objects in the segment
        self.label_set |= {o["label"] for o in rec.get("objects", [])}
        self._ingest(rec)

    def build(self) -> dict[str, Any]:
        """Finalise and return a COMMON_DATA_CONTRACT.md segment record."""
        objects = self._aggregate_objects()
        ocr = self._deduped_ocr()
        quality_flag = "low_quality" if "low_quality" in self._quality_flags else "ok"

        return {
            "video_id": self.video_id,
            "segment_id": f"{self.video_id}_seg_{self.seg_idx:02d}",
            "start_ts": round(self.start_ts, 3),
            "end_ts": round(self.end_ts, 3),
            "visual": {
                # description and embedding_id are M1's responsibility
                "quality_flag": quality_flag,
            },
            "objects": objects,
            "ocr": ocr,
            "scores": {
                "visual_similarity": None,
                "transcript_similarity": None,
                "object_match": None,
                "ocr_match": None,
                "final_score": None,
            },
        }

    # ── Private ────────────────────────────────────────────────────────────

    def _ingest(self, rec: dict[str, Any]) -> None:
        for obj in rec.get("objects", []):
            label = obj["label"]
            conf = obj.get("confidence", 0.0)
            attrs = obj.get("attributes", [])
            if label not in self._obj_data:
                self._obj_data[label] = []
            self._obj_data[label].append((conf, attrs))

        for text in rec.get("ocr", []):
            if text and text not in self._ocr_seen:
                self._ocr_seen.append(text)

        flag = rec.get("quality_flag", "ok")
        self._quality_flags.append(flag)

    def _aggregate_objects(self) -> list[dict[str, Any]]:
        """
        Per label: average confidence, union of attributes seen across frames,
        sorted descending by confidence.
        """
        result = []
        for label, entries in self._obj_data.items():
            avg_conf = round(sum(c for c, _ in entries) / len(entries), 3)
            # Union all attribute lists seen in this segment
            all_attrs: list[str] = []
            seen_attrs: set[str] = set()
            for _, attrs in entries:
                for a in attrs:
                    if a not in seen_attrs:
                        all_attrs.append(a)
                        seen_attrs.add(a)

            obj: dict[str, Any] = {"label": label, "confidence": avg_conf}
            if all_attrs:
                obj["attributes"] = all_attrs
            result.append(obj)

        result.sort(key=lambda o: o["confidence"], reverse=True)
        return result

    def _deduped_ocr(self) -> list[str]:
        """Return unique OCR strings in order first seen (BACKEND.md §9)."""
        return list(dict.fromkeys(self._ocr_seen))
