"""
M6 Full Stack / Integration — Ingestion Orchestrator
Sequentially executes the upstream pipelines (M1 Visual -> M2 Audio -> M3 Vision Intel -> M4 Multi-Modal Ingestion):
  1. M1: Frame extraction, CLIP embedding, zero-shot scene description
  2. M2: Audio extraction, Whisper ASR, SentenceTransformers embedding
  3. M3: YOLOv8 object detection, EasyOCR text extraction
  4. M4: Multi-modal IoU join, FAISS vector indexing, metadata.jsonl persistence
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import M6Config, DEFAULT_CONFIG

log = logging.getLogger(__name__)


def run_full_ingestion(
    video_path: str | Path,
    video_id: Optional[str] = None,
    cfg: M6Config = DEFAULT_CONFIG,
    force_redo: bool = False,
) -> List[Dict[str, Any]]:
    """
    Run full end-to-end ingestion pipeline on a video file.

    Parameters:
      video_path: path to input video file (e.g. data/videos/sample_test.mp4)
      video_id: optional video identifier (defaults to video_path stem)
      cfg: M6Config instance
      force_redo: if True, forces re-sampling and re-extraction across modules

    Returns:
      List of unified segment records written to metadata.jsonl and indexed in FAISS.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if not video_id:
        video_id = video_path.stem

    log.info("=== Starting M6 Full Ingestion Orchestration for video_id='%s' ===", video_id)

    visual_records: List[Dict[str, Any]] = []
    audio_records: List[Dict[str, Any]] = []
    object_records: List[Dict[str, Any]] = []
    ocr_records: List[Dict[str, Any]] = []

    # ── STEP 1: M1 Visual Pipeline ───────────────────────────────────────────
    try:
        log.info("Step 1/4: Executing M1 Video/VLM pipeline...")
        from M1.pipeline import process_video as m1_process_video
        from M1.config import M1Config
        m1_cfg = M1Config(data_root=cfg.data_root)
        visual_records = m1_process_video(
            video_path=video_path,
            video_id=video_id,
            cfg=m1_cfg,
            force_redo_sampling=force_redo,
        )
        log.info("M1 produced %d visual segment records", len(visual_records))
    except Exception as exc:
        log.warning("M1 Visual pipeline encountered error or was skipped (%s)", exc)

    # ── STEP 2: M2 Audio Pipeline ────────────────────────────────────────────
    try:
        log.info("Step 2/4: Executing M2 Audio/NLP pipeline...")
        from M2.pipeline import process_video_audio as m2_process_audio
        audio_records = m2_process_audio(video_id=video_id, video_path=str(video_path))
        log.info("M2 produced %d audio transcript records", len(audio_records))
    except Exception as exc:
        log.warning("M2 Audio pipeline encountered error or was skipped (%s)", exc)

    # ── STEP 3: M3 Vision Intelligence Pipeline ──────────────────────────────
    try:
        log.info("Step 3/4: Executing M3 Vision Intelligence pipeline...")
        from M3.pipeline import process_video as m3_process_video
        from M3.config import M3Config
        m3_cfg = M3Config(data_root=cfg.data_root)
        m3_records = m3_process_video(
            video_path=video_path,
            video_id=video_id,
            cfg=m3_cfg,
            keep_frames=True,
        )
        log.info("M3 produced %d object/ocr frame records", len(m3_records))
        object_records = m3_records
    except Exception as exc:
        log.warning("M3 Vision Intelligence pipeline encountered error or was skipped (%s)", exc)

    # ── STEP 4: M4 Retrieval & FAISS Indexing ────────────────────────────────
    log.info("Step 4/4: Executing M4 Multi-Modal Join & Indexing pipeline...")
    from M4.pipeline import run_ingestion as m4_run_ingestion
    from M4.config import M4Config
    m4_cfg = M4Config(data_root=cfg.data_root)

    unified_segments = m4_run_ingestion(
        visual_records=visual_records,
        audio_records=audio_records,
        object_records=object_records,
        ocr_records=ocr_records,
        cfg=m4_cfg,
    )

    log.info(
        "=== M6 Ingestion Orchestration COMPLETE for '%s': %d unified segments indexed ===",
        video_id,
        len(unified_segments),
    )
    return unified_segments
