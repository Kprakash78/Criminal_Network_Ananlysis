"""
integration/orchestrator.py — Ingestion Orchestrator
=====================================================
Sequentially executes the upstream pipelines:
  1. video/  (CLIP visual embeddings)
  2. audio/  (Whisper ASR, transcript embeddings)
  3. ocr/    (YOLO object detection, EasyOCR)
  4. llm_rag/retrieval/  (FAISS indexing, metadata persistence)

All imports use the new semantic_analysis.* paths.
No references to old M1/M2/M3/M4 module names.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import IntegrationConfig, DEFAULT_CONFIG

log = logging.getLogger(__name__)


def run_full_ingestion(
    video_path: str | Path,
    video_id: Optional[str] = None,
    cfg: IntegrationConfig = DEFAULT_CONFIG,
    force_redo: bool = False,
) -> List[Dict[str, Any]]:
    """
    Run full end-to-end ingestion pipeline on a video file.

    Parameters
    ----------
    video_path : Path to input video file.
    video_id   : Optional identifier (defaults to video filename stem).
    cfg        : IntegrationConfig instance.
    force_redo : If True, forces re-sampling and re-extraction.

    Returns
    -------
    List of unified segment records indexed in FAISS.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if not video_id:
        video_id = video_path.stem

    log.info("=== Starting integration ingestion for video_id='%s' ===", video_id)

    visual_records: List[Dict[str, Any]] = []
    audio_records: List[Dict[str, Any]] = []
    object_records: List[Dict[str, Any]] = []
    ocr_records: List[Dict[str, Any]] = []

    # ── STEP 1: video/ — CLIP Visual Pipeline ────────────────────────────────
    try:
        log.info("Step 1/4: Running video/ CLIP pipeline...")
        from semantic_analysis.video.pipeline import process_video as video_process
        from semantic_analysis.video.config import VideoConfig
        v_cfg = VideoConfig(data_root=cfg.data_root)
        visual_records = video_process(
            video_path=video_path,
            video_id=video_id,
            cfg=v_cfg,
            force_redo_sampling=force_redo,
        )
        log.info("video/ produced %d visual segment records", len(visual_records))
    except Exception as exc:
        log.warning("video/ pipeline failed or unavailable: %s", exc)

    # ── STEP 2: audio/ — Whisper Audio Pipeline ───────────────────────────────
    try:
        log.info("Step 2/4: Running audio/ Whisper pipeline...")
        from semantic_analysis.audio.pipeline import process_video_audio
        audio_records = process_video_audio(video_id=video_id, video_path=str(video_path))
        log.info("audio/ produced %d audio transcript records", len(audio_records))
    except Exception as exc:
        log.warning("audio/ pipeline failed or unavailable: %s", exc)

    # ── STEP 3: ocr/ — YOLO + EasyOCR Pipeline ───────────────────────────────
    try:
        log.info("Step 3/4: Running ocr/ YOLO+OCR pipeline...")
        from semantic_analysis.ocr.pipeline import process_video as ocr_process
        from semantic_analysis.ocr.config import OcrConfig
        o_cfg = OcrConfig(data_root=cfg.data_root)
        object_records = ocr_process(
            video_path=video_path,
            video_id=video_id,
            cfg=o_cfg,
            keep_frames=True,
        )
        log.info("ocr/ produced %d object/ocr records", len(object_records))
    except Exception as exc:
        log.warning("ocr/ pipeline failed or unavailable: %s", exc)

    # ── STEP 4: llm_rag/retrieval/ — FAISS Indexing ──────────────────────────
    log.info("Step 4/4: Running llm_rag/retrieval/ FAISS indexing...")
    from semantic_analysis.llm_rag.retrieval.pipeline import run_ingestion
    from semantic_analysis.llm_rag.retrieval.config import RetrievalConfig
    r_cfg = RetrievalConfig(data_root=cfg.data_root)

    unified_segments = run_ingestion(
        visual_records=visual_records,
        audio_records=audio_records,
        object_records=object_records,
        ocr_records=ocr_records,
        cfg=r_cfg,
    )

    log.info(
        "=== Integration ingestion COMPLETE for '%s': %d unified segments ===",
        video_id, len(unified_segments),
    )
    return unified_segments
