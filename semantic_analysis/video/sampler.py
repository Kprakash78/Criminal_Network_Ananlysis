"""
M1 Video / VLM — Frame Sampler
Extracts frames from input video at fixed time intervals using FFmpeg / OpenCV.
Coordinates with M3's sampling convention (data/frames/<video_id>/frame_<ts>.jpg).
Reuses M3.sampler.sample_frames if available to ensure shared extraction pass.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple

import cv2

from .config import VideoConfig, DEFAULT_CONFIG

log = logging.getLogger(__name__)


def sample_frames(
    video_path: str | Path,
    video_id: str,
    cfg: VideoConfig = DEFAULT_CONFIG,
    *,
    force_redo: bool = False,
) -> List[Tuple[float, Path]]:
    """
    Extract frames from video_path at cfg.frame_interval_sec intervals.

    Returns sorted list of (timestamp_seconds, frame_path) tuples.
    Frames stored at cfg.frames_dir / video_id / frame_<ts>.jpg.
    If frame dir already exists and is non-empty, reuses existing frames.
    """
    # Attempt to reuse M3's sampler if available to share extraction pass
    try:
        from semantic_analysis.ocr.sampler import sample_frames as m3_sample_frames
        from semantic_analysis.ocr.config import OcrConfig as M3Config
        m3_cfg = OcrConfig(
            frame_interval_sec=cfg.frame_interval_sec,
            data_root=cfg.data_root,
        )
        log.info("Reusing M3 frame sampler for video_id=%s", video_id)
        return m3_sample_frames(video_path, video_id, cfg=m3_cfg, force_redo=force_redo)
    except Exception as exc:
        log.debug("M3 sampler not available or failed (%s); using standalone M1 sampler", exc)

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    frame_dir = cfg.frames_dir / video_id
    frame_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(frame_dir.glob("frame_*.jpg"))
    if existing and not force_redo:
        log.info("Frame dir %s already populated (%d frames); skipping extraction", frame_dir, len(existing))
        return _frames_from_dir(frame_dir)

    # Remove stale frames and temporary files so we start clean
    for f in frame_dir.glob("*"):
        if f.is_file():
            f.unlink()

    if shutil.which("ffmpeg"):
        _extract_with_ffmpeg(video_path, frame_dir, cfg.frame_interval_sec)
    else:
        log.warning("ffmpeg not on PATH; falling back to OpenCV frame extraction")
        _extract_with_opencv(video_path, frame_dir, cfg.frame_interval_sec)

    frames = _frames_from_dir(frame_dir)
    log.info("Extracted %d frames for video_id=%s", len(frames), video_id)
    return frames


def _extract_with_ffmpeg(video_path: Path, frame_dir: Path, interval_sec: float) -> None:
    fps_expr = f"1/{interval_sec}"
    tmp_pattern = str(frame_dir / "tmp_%05d.jpg")

    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vf", f"fps={fps_expr}",
        "-q:v", "2",
        tmp_pattern,
        "-y",
        "-loglevel", "error",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed (rc={result.returncode}): {result.stderr}")

    # Rename tmp_00001.jpg -> frame_<seconds>.jpg
    for tmp_path in sorted(frame_dir.glob("tmp_*.jpg")):
        idx_str = tmp_path.stem.replace("tmp_", "")
        idx = int(idx_str) - 1  # 1-indexed to 0-indexed frame count
        ts = round(idx * interval_sec, 3)
        canonical = frame_dir / f"frame_{ts:.3f}.jpg"
        if tmp_path != canonical:
            tmp_path.rename(canonical)


def _extract_with_opencv(video_path: Path, frame_dir: Path, interval_sec: float) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration_sec = total_frames / fps

    ts = 0.0
    while ts <= duration_sec:
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
        ret, frame = cap.read()
        if not ret:
            break
        out_path = frame_dir / f"frame_{ts:.3f}.jpg"
        cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        ts = round(ts + interval_sec, 3)

    cap.release()


def _frames_from_dir(frame_dir: Path) -> List[Tuple[float, Path]]:
    result: List[Tuple[float, Path]] = []
    for p in sorted(frame_dir.glob("frame_*.jpg")):
        ts_str = p.stem.replace("frame_", "")
        try:
            ts = float(ts_str)
        except ValueError:
            log.warning("Unexpected frame filename: %s; skipping", p.name)
            continue
        result.append((ts, p))
    result.sort(key=lambda x: x[0])
    return result
