"""
M3 Vision Intelligence — Frame Sampler
Extracts frames from a video at a fixed time interval using FFmpeg (preferred,
BACKEND.md §3) with a fallback to OpenCV when FFmpeg is not on PATH.

Output convention: data/frames/<video_id>/frame_<seconds>.jpg
  e.g. frame_0.0.jpg, frame_1.0.jpg, frame_2.0.jpg …
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterator

import cv2

from .config import M3Config, DEFAULT_CONFIG

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def sample_frames(
    video_path: str | Path,
    video_id: str,
    cfg: M3Config = DEFAULT_CONFIG,
    *,
    force_redo: bool = False,
) -> list[tuple[float, Path]]:
    """
    Extract frames from *video_path* at cfg.frame_interval_sec intervals.

    Returns a sorted list of (timestamp_seconds, frame_path) tuples.
    Frames are stored under cfg.frames_dir / video_id / frame_<ts>.jpg.
    If the frame directory already exists and is non-empty, skip extraction
    unless *force_redo* is True.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    frame_dir = cfg.frames_dir / video_id
    frame_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(frame_dir.glob("frame_*.jpg"))
    if existing and not force_redo:
        log.info("Frame dir %s already populated (%d frames); skipping extraction",
                 frame_dir, len(existing))
        return _frames_from_dir(frame_dir)

    # Remove stale frames so we start clean
    for f in existing:
        f.unlink()

    # Try FFmpeg first; fall back to OpenCV
    if shutil.which("ffmpeg"):
        _extract_with_ffmpeg(video_path, frame_dir, cfg.frame_interval_sec)
    else:
        log.warning(
            "ffmpeg not found on PATH — falling back to OpenCV frame extraction. "
            "Install FFmpeg for more reliable seeking."
        )
        _extract_with_opencv(video_path, frame_dir, cfg.frame_interval_sec)

    frames = _frames_from_dir(frame_dir)
    log.info("Extracted %d frames for video_id=%s", len(frames), video_id)
    return frames


def delete_frames(video_id: str, cfg: M3Config = DEFAULT_CONFIG) -> None:
    """
    Delete the temporary frame directory for *video_id* after processing.
    Called by the pipeline after metadata.jsonl has been written (BACKEND.md §3).
    """
    frame_dir = cfg.frames_dir / video_id
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
        log.info("Deleted temp frames for video_id=%s", video_id)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_with_ffmpeg(
    video_path: Path, frame_dir: Path, interval_sec: float
) -> None:
    """
    Shell out to ffmpeg:
        ffmpeg -i video.mp4 -vf fps=1/<interval> frame_%07.3f.jpg
    Frames are named by their timestamp so bookkeeping is trivial (BACKEND.md §4).
    """
    fps_expr = f"1/{interval_sec}"
    # %07.3f gives filenames like frame_0023.500.jpg which sort correctly
    # We rename afterwards to the canonical frame_<ts>.jpg name
    tmp_pattern = str(frame_dir / "tmp_%07.3f.jpg")

    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vf", f"fps={fps_expr}",
        "-q:v", "2",        # high quality JPEG
        "-vsync", "vfr",
        tmp_pattern,
        "-y",
        "-loglevel", "error",
    ]
    log.debug("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (rc={result.returncode}):\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    # Rename tmp_XXXX.XXX.jpg → frame_<seconds>.jpg
    #   ffmpeg names with 1-indexed frame index when using the sequence pattern;
    #   but with fps filter and %07.3f it names by time offset in seconds.
    for tmp_path in sorted(frame_dir.glob("tmp_*.jpg")):
        name = tmp_path.stem.replace("tmp_", "")
        ts = float(name)
        canonical = frame_dir / f"frame_{ts:.3f}.jpg"
        tmp_path.rename(canonical)


def _extract_with_opencv(
    video_path: Path, frame_dir: Path, interval_sec: float
) -> None:
    """
    OpenCV-based fallback.  Seeks to each target timestamp and reads the frame.
    Less accurate than FFmpeg for B-frame heavy codecs but works without a
    system install.
    """
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


def _frames_from_dir(frame_dir: Path) -> list[tuple[float, Path]]:
    """Return sorted (timestamp, path) pairs from frame_*.jpg files."""
    result: list[tuple[float, Path]] = []
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
