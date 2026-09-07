"""
M3 Vision Intelligence — Main Pipeline
Entry point: process_video() runs the full pipeline for one video file.
Also provides a CLI for standalone use.

Usage (CLI):
    python -m M3.pipeline path/to/video.mp4 [--video-id my_video] [--interval 1.0]
                                              [--conf 0.5] [--ocr-interval 2.0]
                                              [--no-attributes]
                                              [--keep-frames]
                                              [--data-root data]

The module is designed to be runnable standalone (BACKEND.md §3, PRD §7):
it does not require M1 to have extracted frames first — it extracts its own
frames via FFmpeg (or OpenCV fallback if FFmpeg is unavailable).
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

from .config import OcrConfig
from .sampler import sample_frames, delete_frames
from .detector import process_frames_batch
from .aggregator import aggregate_to_segments
from .writer import write_segments

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def process_video(
    video_path: str | Path,
    video_id: str,
    cfg: OcrConfig | None = None,
    *,
    keep_frames: bool = False,
    extract_attributes: bool = True,
) -> list[dict[str, Any]]:
    """
    Full M3 pipeline for one video file.

    1. Sample frames at cfg.frame_interval_sec intervals (FFmpeg or OpenCV)
    2. Run YOLO + EasyOCR + quality gate on each frame
    3. Aggregate frame records into segment records (Jaccard-based dedup)
    4. Append segment records to cfg.metadata_jsonl

    Returns the list of segment dicts written to disk.
    """
    if cfg is None:
        cfg = OcrConfig()

    t0 = time.perf_counter()
    video_path = Path(video_path)
    log.info("═══ M3 pipeline START  video_id=%s  path=%s", video_id, video_path)

    # ── Step 1: Frame sampling ────────────────────────────────────────────────
    log.info("Step 1/4: Sampling frames (interval=%.1fs)", cfg.frame_interval_sec)
    frames = sample_frames(video_path, video_id, cfg)
    log.info("  → %d frames extracted", len(frames))

    if not frames:
        log.warning("No frames extracted from %s — aborting pipeline", video_path)
        return []

    # ── Step 2: Per-frame detection ───────────────────────────────────────────
    log.info(
        "Step 2/4: Running YOLO + OCR on %d frames (ocr every %.1fs)",
        len(frames), cfg.ocr_interval_sec,
    )
    frame_records = process_frames_batch(
        frames, video_id, cfg, extract_attributes=extract_attributes
    )
    log.info("  → %d frame records produced", len(frame_records))

    # ── Step 3: Aggregation ───────────────────────────────────────────────────
    log.info("Step 3/4: Aggregating to segments (overlap_threshold=%.2f)",
             cfg.segment_overlap_threshold)
    segments = aggregate_to_segments(frame_records, video_id, cfg)
    log.info("  → %d segments", len(segments))

    # ── Step 4: Write to metadata.jsonl ───────────────────────────────────────
    log.info("Step 4/4: Writing segments to %s", cfg.metadata_jsonl)
    n_written = write_segments(segments, cfg)
    log.info("  → %d segment records appended", n_written)

    # ── Cleanup temp frames ───────────────────────────────────────────────────
    if not keep_frames:
        delete_frames(video_id, cfg)

    elapsed = time.perf_counter() - t0
    log.info(
        "═══ M3 pipeline DONE  video_id=%s  segments=%d  time=%.1fs",
        video_id, len(segments), elapsed,
    )
    return segments


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="M3 Vision Intelligence — process a video into metadata.jsonl",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("video", help="Path to input video file")
    p.add_argument(
        "--video-id",
        default=None,
        help="Identifier for the video (default: video filename stem)",
    )
    p.add_argument(
        "--interval", type=float, default=1.0,
        help="Frame sampling interval in seconds",
    )
    p.add_argument(
        "--conf", type=float, default=0.5,
        help="YOLO confidence threshold",
    )
    p.add_argument(
        "--ocr-interval", type=float, default=2.0,
        help="OCR sampling interval in seconds",
    )
    p.add_argument(
        "--blur-threshold", type=float, default=60.0,
        help="Laplacian variance below this → low_quality",
    )
    p.add_argument(
        "--seg-overlap", type=float, default=0.5,
        help="Jaccard similarity threshold for segment boundary detection",
    )
    p.add_argument(
        "--data-root", default="data",
        help="Root directory for frames/ and metadata.jsonl",
    )
    p.add_argument(
        "--keep-frames", action="store_true",
        help="Do not delete extracted frames after processing",
    )
    p.add_argument(
        "--no-attributes", action="store_true",
        help="Skip optional dominant-color attribute extraction",
    )
    p.add_argument(
        "--yolo-model", default=str(Path(__file__).resolve().parent.parent / "models" / "yolo" / "yolov8n.pt"),
        help="YOLO model checkpoint (e.g. yolov8n.pt, yolov8s.pt)",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG logging",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: Video file not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    video_id = args.video_id or video_path.stem

    cfg = OcrConfig(
        frame_interval_sec=args.interval,
        yolo_conf_threshold=args.conf,
        ocr_interval_sec=args.ocr_interval,
        blur_threshold=args.blur_threshold,
        segment_overlap_threshold=args.seg_overlap,
        yolo_model=args.yolo_model,
        data_root=Path(args.data_root),
    )

    segments = process_video(
        video_path,
        video_id,
        cfg,
        keep_frames=args.keep_frames,
        extract_attributes=not args.no_attributes,
    )

    print(f"\nDone. {len(segments)} segments written to {cfg.metadata_jsonl}")


if __name__ == "__main__":
    main()
