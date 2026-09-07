"""
M1 Video / VLM — End-to-End Pipeline
Main entry point for processing a video file through the M1 pipeline:
  Video -> Frame Extraction -> CLIP Embedding & Description -> Aggregation -> Metadata JSONL & FAISS Index
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .aggregator import aggregate_frames_to_segments
from .config import VideoConfig, DEFAULT_CONFIG
from .embedder import CLIPEmbedder
from .sampler import sample_frames
from .validator import validate_segment_records
from .writer import write_segments_and_index

log = logging.getLogger(__name__)


def process_video(
    video_path: str | Path,
    video_id: Optional[str] = None,
    cfg: VideoConfig = DEFAULT_CONFIG,
    embedder: Optional[CLIPEmbedder] = None,
    force_redo_sampling: bool = False,
) -> List[Dict[str, Any]]:
    """
    Run end-to-end M1 pipeline on a single video file.

    Parameters:
      video_path: path to input video file (e.g. data/videos/sample_test.mp4)
      video_id: identifier for the video (defaults to stem of video_path)
      cfg: VideoConfig instance
      embedder: optional pre-loaded CLIPEmbedder instance (avoids reloading model)
      force_redo_sampling: if True, forces re-extraction of frames

    Returns:
      List of schema-valid segment-level records written to metadata.jsonl & indexed.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if not video_id:
        video_id = video_path.stem

    log.info("=== Starting M1 Video Pipeline for video_id='%s' ===", video_id)

    # STEP 1: Frame extraction / sampling
    log.info("Step 1: Frame sampling (interval=%.1fs)...", cfg.frame_interval_sec)
    frames = sample_frames(video_path, video_id, cfg=cfg, force_redo=force_redo_sampling)
    if not frames:
        raise RuntimeError(f"No frames extracted from {video_path}")

    # STEP 2: CLIP Embedding & Zero-shot description per frame
    log.info("Step 2: Generating frame embeddings and zero-shot descriptions...")
    if embedder is None:
        embedder = CLIPEmbedder(cfg=cfg)

    frame_records = embedder.process_frames(frames)

    # STEP 3: Aggregation into segment-level records
    log.info("Step 3: Aggregating frame-level embeddings into segments...")
    segment_records = aggregate_frames_to_segments(frame_records, video_id, cfg=cfg)

    # STEP 4: Schema Validation
    log.info("Step 4: Validating output segment records against data contract...")
    is_valid, errors = validate_segment_records(segment_records)
    if not is_valid:
        raise ValueError(f"Pipeline output failed schema validation: {errors}")

    # STEP 5: Write to metadata.jsonl & FAISS Index
    log.info("Step 5: Writing segments to metadata.jsonl and index...")
    records_wrote, vectors_indexed = write_segments_and_index(segment_records, cfg=cfg)

    log.info(
        "=== M1 Pipeline Completed for '%s': %d segments, %d indexed ===",
        video_id,
        records_wrote,
        vectors_indexed,
    )
    return segment_records
