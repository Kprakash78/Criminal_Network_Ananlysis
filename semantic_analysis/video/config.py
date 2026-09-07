"""
M1 Video / VLM — Configuration
Tunable parameters for frame sampling, CLIP embedding, zero-shot scene description,
and segment aggregation. Override via environment variables or pass as kwargs.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_CANDIDATE_LABELS = [
    "a person",
    "a chef cooking",
    "a person repairing a vehicle or machine",
    "a motorcycle or bicycle",
    "an outdoor nature scene",
    "an indoor room or workshop",
    "a kitchen or food preparation area",
    "a close-up of hands or tools",
    "a vehicle on a road",
    "text or document display",
    "a person wearing a helmet or protective gear",
    "an aerial or wide view of a landscape",
    "people talking or presenting",
    "a sports or action activity",
    "animals or pets",
    "an air fryer or countertop cooking appliance",
]


@dataclass
class VideoConfig:
    # ── Frame sampling ────────────────────────────────────────────────────────
    # Interval in seconds between sampled frames (matches M3 convention)
    frame_interval_sec: float = float(os.getenv("M1_FRAME_INTERVAL", "1.0"))

    # ── CLIP Model ───────────────────────────────────────────────────────────
    clip_model_name: str = os.getenv("M1_CLIP_MODEL", "ViT-B-32")
    clip_pretrained: str = os.getenv("M1_CLIP_PRETRAINED", "openai")
    batch_size: int = int(os.getenv("M1_BATCH_SIZE", "16"))
    device: str = os.getenv("M1_DEVICE", "cpu")

    # ── Zero-shot Description ─────────────────────────────────────────────────
    candidate_labels: list[str] = field(
        default_factory=lambda: list(DEFAULT_CANDIDATE_LABELS)
    )
    description_top_k: int = int(os.getenv("M1_TOP_K_LABELS", "3"))

    # ── Aggregation ───────────────────────────────────────────────────────────
    # Cosine similarity threshold: consecutive frames >= threshold → same segment
    segment_sim_threshold: float = float(
        os.getenv("M1_SEGMENT_SIM_THRESHOLD", "0.85")
    )
    # Maximum segment duration in seconds — hard ceiling so that even visually-
    # homogeneous content (e.g. a cooking video where every frame looks alike)
    # is split into small, locatable segments (precision-over-coverage fix)
    max_segment_sec: float = float(os.getenv("M1_MAX_SEGMENT_SEC", "5.0"))

    # ── Paths ─────────────────────────────────────────────────────────────────
    data_root: Path = field(
        default_factory=lambda: Path(os.getenv("M1_DATA_ROOT", "data"))
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

    @property
    def faiss_visual_index(self) -> Path:
        return self.data_root / "faiss_visual.index"

    @property
    def id_map_json(self) -> Path:
        return self.data_root / "id_map.json"


DEFAULT_CONFIG = VideoConfig()
