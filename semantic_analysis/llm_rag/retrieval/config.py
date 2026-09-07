"""
M4 Retrieval Module — Configuration
Defines FAISS index dimensions, weight parameters, overlap threshold,
embedding model checkpoints, and persistent file paths.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RetrievalConfig:
    # ── Dimensions & Model Checkpoints ───────────────────────────────────────
    # Must match M1 checkpoint (ViT-B-32 = 512)
    visual_dim: int = int(os.getenv("M4_VISUAL_DIM", "512"))
    clip_model_name: str = os.getenv("M4_CLIP_MODEL", "ViT-B-32")
    clip_pretrained: str = os.getenv("M4_CLIP_PRETRAINED", "openai")

    # Must match M2 checkpoint (all-MiniLM-L6-v2 = 384)
    audio_dim: int = int(os.getenv("M4_AUDIO_DIM", "384"))
    sentence_model_name: str = os.getenv("M4_SENTENCE_MODEL", "all-MiniLM-L6-v2")

    # ── Ingestion Join Parameters ─────────────────────────────────────────────
    # Minimum time-range overlap fraction (intersection / union) to join records
    overlap_threshold: float = float(os.getenv("M4_OVERLAP_THRESHOLD", "0.3"))

    # ── Scoring Fusion Weights (PRD §6 starting weights) ─────────────────────
    w_visual: float = float(os.getenv("M4_W_VISUAL", "0.35"))
    w_transcript: float = float(os.getenv("M4_W_TRANSCRIPT", "0.35"))
    w_object: float = float(os.getenv("M4_W_OBJECT", "0.20"))
    w_ocr: float = float(os.getenv("M4_W_OCR", "0.10"))
    # Deterministic event-position signal for arrival/departure/exit queries.
    w_event: float = float(os.getenv("M4_W_EVENT", "0.25"))

    # Device for query embedding ('cpu' or 'cuda')
    device: str = os.getenv("M4_DEVICE", "cpu")

    # ── API Response Confidence Thresholds ────────────────────────────────────
    confidence_threshold: float = float(os.getenv("M4_CONFIDENCE_THRESHOLD", "0.45"))

    # ── Paths ─────────────────────────────────────────────────────────────────
    data_root: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "M4_DATA_ROOT",
                str(Path(__file__).resolve().parents[3] / "data" / "video"),
            )
        )
    )

    @property
    def metadata_jsonl(self) -> Path:
        return self.data_root / "metadata.jsonl"

    @property
    def faiss_visual_index(self) -> Path:
        return self.data_root / "faiss_visual.index"

    @property
    def faiss_audio_index(self) -> Path:
        return self.data_root / "faiss_audio.index"

    @property
    def id_map_json(self) -> Path:
        return self.data_root / "id_map.json"


DEFAULT_CONFIG = RetrievalConfig()
