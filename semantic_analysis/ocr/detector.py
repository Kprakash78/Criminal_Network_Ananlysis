"""
M3 Vision Intelligence — Frame Detector
Runs YOLOv8 object detection + EasyOCR text extraction on each sampled frame,
plus Laplacian-variance blur quality gate.

Produces frame-level records matching COMMON_DATA_CONTRACT.md §Frame-level record:
{
  "video_id": str,
  "timestamp": float,   # always seconds
  "objects": [{"label": str, "confidence": float}],
  "ocr": [str],
  "quality_flag": "ok" | "low_quality"
}

Optionally enriches objects with a dominant-color attribute for attribute binding
(optional tier 1, PRD §3.1 / §5).
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import OcrConfig, DEFAULT_CONFIG

log = logging.getLogger(__name__)

VEHICLE_LABELS = {"car", "truck", "bus", "van", "motorcycle", "bicycle"}
PLATE_TEXT_RE = re.compile(r"[A-Z0-9][A-Z0-9\-\s]{4,12}[A-Z0-9]", re.I)

# ── Lazy singletons so models load once per process ───────────────────────────
_yolo_instance = None
_ocr_instance = None


def _get_yolo(cfg: OcrConfig):
    global _yolo_instance
    if _yolo_instance is None:
        from ultralytics import YOLO  # deferred import — slow first time
        log.info("Loading YOLO model: %s", cfg.yolo_model)
        _yolo_instance = YOLO(cfg.yolo_model)
    return _yolo_instance


def _get_ocr(cfg: OcrConfig):
    global _ocr_instance
    if _ocr_instance is None:
        import easyocr  # deferred import
        log.info("Initialising EasyOCR (gpu=%s)", cfg.ocr_gpu)
        _ocr_instance = easyocr.Reader(["en"], gpu=cfg.ocr_gpu, verbose=False)
    return _ocr_instance


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def process_frame(
    frame_path: Path,
    video_id: str,
    timestamp: float,
    run_ocr: bool,
    cfg: OcrConfig = DEFAULT_CONFIG,
    *,
    extract_attributes: bool = True,
) -> dict[str, Any]:
    """
    Produce a frame-level record for one JPEG frame.

    Parameters
    ----------
    frame_path : Path to the JPEG frame.
    video_id   : Identifier for the parent video.
    timestamp  : Seconds from video start (float).
    run_ocr    : Whether to run EasyOCR on this frame (lower-freq than YOLO).
    extract_attributes : If True, attempt dominant-color extraction per detected
                         box (optional tier 1, PRD §3.1).
    """
    img = cv2.imread(str(frame_path))
    if img is None:
        log.warning("Could not read frame: %s — skipping", frame_path)
        return _empty_record(video_id, timestamp)

    # ── Quality gate ──────────────────────────────────────────────────────────
    quality_flag = _quality_gate(img, cfg.blur_threshold)

    # ── YOLO detection ───────────────────────────────────────────────────────
    yolo = _get_yolo(cfg)
    objects = _run_yolo(yolo, img, cfg.yolo_conf_threshold, extract_attributes)

    # ── OCR ───────────────────────────────────────────────────────────────────
    ocr_texts: list[str] = []
    if run_ocr:
        # Skip OCR on frames we've flagged as low quality — likely garbage output
        if quality_flag == "ok":
            ocr_texts = _run_ocr(img, cfg)
            ocr_texts.extend(_run_vehicle_plate_ocr(img, objects, cfg))
            ocr_texts = list(dict.fromkeys(ocr_texts))
        else:
            log.debug("Skipping OCR on low-quality frame at t=%.3f", timestamp)

    return {
        "video_id": video_id,
        "timestamp": round(timestamp, 3),
        "objects": objects,
        "ocr": ocr_texts,
        "quality_flag": quality_flag,
    }


def process_frames_batch(
    frames: list[tuple[float, Path]],
    video_id: str,
    cfg: OcrConfig = DEFAULT_CONFIG,
    *,
    extract_attributes: bool = True,
) -> list[dict[str, Any]]:
    """
    Process all sampled frames for a video.

    OCR runs at a lower frequency than YOLO: every time the elapsed time since
    the last OCR run exceeds cfg.ocr_interval_sec (BACKEND.md §6/9).
    """
    records: list[dict[str, Any]] = []
    last_ocr_ts: float = -cfg.ocr_interval_sec  # force OCR on first frame

    total = len(frames)
    for idx, (ts, path) in enumerate(frames):
        run_ocr = (ts - last_ocr_ts) >= cfg.ocr_interval_sec
        if run_ocr:
            last_ocr_ts = ts

        if idx % 10 == 0:
            log.info(
                "Processing frame %d/%d  t=%.3fs  video_id=%s",
                idx + 1, total, ts, video_id,
            )

        rec = process_frame(
            path, video_id, ts, run_ocr=run_ocr, cfg=cfg,
            extract_attributes=extract_attributes,
        )
        records.append(rec)

    return records


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _quality_gate(img: np.ndarray, blur_threshold: float) -> str:
    """
    Compute Laplacian variance of the greyscale image.
    Values below *blur_threshold* → 'low_quality' (BACKEND.md §5).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return "low_quality" if lap_var < blur_threshold else "ok"


def _run_yolo(
    yolo,
    img: np.ndarray,
    conf_threshold: float,
    extract_attributes: bool,
) -> list[dict[str, Any]]:
    """Run YOLO inference and return filtered object list."""
    results = yolo(img, verbose=False)[0]
    objects: list[dict[str, Any]] = []

    for box in results.boxes:
        conf = float(box.conf[0])
        if conf < conf_threshold:
            continue
        label = yolo.names[int(box.cls[0])]
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
        obj: dict[str, Any] = {
            "label": label,
            "confidence": round(conf, 3),
            "bbox": [x1, y1, x2, y2],
        }
        if extract_attributes:
            attrs = _extract_color_attributes(img, box)
            if attrs:
                obj["attributes"] = attrs

        objects.append(obj)

    return objects


def _extract_color_attributes(img: np.ndarray, box) -> list[str]:
    """
    Optional tier 1 (PRD §3.1): crop the detected bounding box, compute the
    dominant HSV color via histogram, map to a human-readable color name.
    Returns e.g. ["red"] or [] if the crop is too small or fails.
    """
    try:
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
        crop = img[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            return []
        color_name = _dominant_color(crop)
        return [color_name] if color_name else []
    except Exception as exc:
        log.debug("Color attribute extraction failed: %s", exc)
        return []


_COLOR_RANGES_HSV = [
    # name, (h_lo, s_lo, v_lo), (h_hi, s_hi, v_hi)
    ("red",    (  0, 100, 70), ( 10, 255, 255)),
    ("red",    (170, 100, 70), (180, 255, 255)),  # wrapped hue
    ("orange", ( 11,  80, 70), ( 25, 255, 255)),
    ("yellow", ( 26, 80,  70), ( 35, 255, 255)),
    ("green",  ( 36, 50,  40), ( 85, 255, 255)),
    ("cyan",   ( 86, 50,  40), ( 99, 255, 255)),
    ("blue",   (100, 80,  40), (130, 255, 255)),
    ("purple", (131, 50,  40), (160, 255, 255)),
    ("pink",   (161, 40,  70), (169, 255, 255)),
    ("white",  (  0,  0, 200), (180,  30, 255)),
    ("black",  (  0,  0,   0), (180, 255,  50)),
    ("gray",   (  0,  0,  51), (180,  30, 199)),
]


def _dominant_color(crop_bgr: np.ndarray) -> str | None:
    """
    Convert crop to HSV, find the most common named color via per-range pixel
    counting.  Returns None for ambiguous/neutral cases.
    """
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    total = hsv.shape[0] * hsv.shape[1]
    best_name, best_count = None, 0

    color_counts: dict[str, int] = {}
    for name, lo, hi in _COLOR_RANGES_HSV:
        lo_arr = np.array(lo, dtype=np.uint8)
        hi_arr = np.array(hi, dtype=np.uint8)
        mask = cv2.inRange(hsv, lo_arr, hi_arr)
        count = int(mask.sum() // 255)
        color_counts[name] = color_counts.get(name, 0) + count

    for name, count in color_counts.items():
        if count > best_count:
            best_count = count
            best_name = name

    # Only report if the dominant colour covers at least 20% of the crop
    if best_count / total < 0.20:
        return None
    return best_name


def _run_ocr(img: np.ndarray, cfg: OcrConfig) -> list[str]:
    """Run EasyOCR and return deduplicated non-trivial text strings."""
    reader = _get_ocr(cfg)
    raw: list[str] = reader.readtext(img, detail=0)  # detail=0 → text only
    # Filter: minimum length (BACKEND.md §8), strip whitespace
    filtered = list(dict.fromkeys(
        t.strip() for t in raw
        if len(t.strip()) >= cfg.ocr_min_len
    ))
    return filtered


def _run_vehicle_plate_ocr(
    img: np.ndarray,
    objects: list[dict[str, Any]],
    cfg: OcrConfig,
) -> list[str]:
    """Run OCR on enlarged vehicle crops to improve small plate reads."""
    texts: list[str] = []
    for obj in objects:
        if str(obj.get("label", "")).lower() not in VEHICLE_LABELS:
            continue
        bbox = obj.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        crop = _vehicle_plate_region(img, bbox)
        if crop is None:
            continue
        for text in _run_ocr(crop, cfg):
            cleaned = _normalize_plate_text(text)
            if cleaned and cleaned not in texts:
                texts.append(cleaned)
    return texts


def _vehicle_plate_region(img: np.ndarray, bbox: list[int]) -> np.ndarray | None:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None

    box_w = x2 - x1
    box_h = y2 - y1
    if box_w < 20 or box_h < 20:
        return None

    pad_x = int(box_w * 0.12)
    crop_y1 = y1 + int(box_h * 0.45)
    crop_y2 = y2
    crop_x1 = max(0, x1 - pad_x)
    crop_x2 = min(w, x2 + pad_x)
    crop = img[crop_y1:crop_y2, crop_x1:crop_x2]
    if crop.size == 0:
        return None

    scale = 2 if min(crop.shape[:2]) >= 40 else 3
    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _normalize_plate_text(text: str) -> str | None:
    value = " ".join(text.upper().replace("-", " ").split())
    if not PLATE_TEXT_RE.search(value):
        return None
    compact = "".join(ch for ch in value if ch.isalnum())
    if len(compact) < 5 or not any(ch.isdigit() for ch in compact):
        return None
    return compact


def _empty_record(video_id: str, timestamp: float) -> dict[str, Any]:
    return {
        "video_id": video_id,
        "timestamp": round(timestamp, 3),
        "objects": [],
        "ocr": [],
        "quality_flag": "low_quality",
    }
