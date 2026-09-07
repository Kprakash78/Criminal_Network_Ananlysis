"""
M3 Vision Intelligence — Configuration
All tunable parameters in one place. Override via environment variables or
pass as kwargs to the pipeline functions.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class OcrConfig:
    # ── Frame sampling ────────────────────────────────────────────────────────
    # Interval in seconds between sampled frames (BACKEND.md §3)
    frame_interval_sec: float = float(os.getenv("M3_FRAME_INTERVAL", "1.0"))

    # ── YOLO ──────────────────────────────────────────────────────────────────
    # Pretrained checkpoint; yolov8n.pt auto-downloads on first use
    yolo_model: str = os.getenv("M3_YOLO_MODEL", str(Path(__file__).resolve().parent.parent / "models" / "yolo" / "yolov8n.pt"))
    # Confidence threshold (BACKEND.md §8)
    yolo_conf_threshold: float = float(os.getenv("M3_YOLO_CONF", "0.5"))

    # ── OCR ───────────────────────────────────────────────────────────────────
    # Run OCR every N seconds (lower freq than YOLO per BACKEND.md §6/9)
    ocr_interval_sec: float = float(os.getenv("M3_OCR_INTERVAL", "2.0"))
    # Minimum character length to keep an OCR result (BACKEND.md §8)
    ocr_min_len: int = int(os.getenv("M3_OCR_MIN_LEN", "2"))
    # GPU flag for EasyOCR
    ocr_gpu: bool = os.getenv("M3_OCR_GPU", "false").lower() == "true"

    # ── Quality gate ─────────────────────────────────────────────────────────
    # Laplacian variance below this → low_quality (BACKEND.md §5)
    blur_threshold: float = float(os.getenv("M3_BLUR_THRESHOLD", "60.0"))

    # ── Aggregation ───────────────────────────────────────────────────────────
    # Jaccard similarity threshold: below this → new segment (BACKEND.md §7)
    segment_overlap_threshold: float = float(os.getenv("M3_SEG_OVERLAP", "0.5"))
    # Maximum segment duration in seconds (precision fix: caps runaway segments
    # caused by long runs of visually-similar or empty-object frames)
    max_segment_sec: float = float(os.getenv("M3_MAX_SEGMENT_SEC", "5.0"))

    # ── Paths ─────────────────────────────────────────────────────────────────
    # Root data directory (mirrors COMMON_DATA_CONTRACT.md §Storage layout)
    data_root: Path = field(
        default_factory=lambda: Path(os.getenv("M3_DATA_ROOT", "data"))
    )

    @property
    def videos_dir(self) -> Path:
        return self.data_root / "videos"

    @property
    def frames_dir(self) -> Path:
        return self.data_root / "frames"

    @property
    def metadata_jsonl(self) -> Path:
        return self.data_root / "metadata.jsonl"


# Module-level default instance
DEFAULT_CONFIG = OcrConfig()
