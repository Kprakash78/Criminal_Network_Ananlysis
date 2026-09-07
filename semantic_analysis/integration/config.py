"""
integration/config.py — Integration Module Configuration
=========================================================
Configuration for the FastAPI integration layer (was M6/config.py).
Data root points to Criminal_detection/data/video by default.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Default data root: Criminal_detection/data/video/
# Can be overridden via SA_DATA_ROOT environment variable.
_DEFAULT_DATA_ROOT = str(
    Path(__file__).resolve().parent.parent.parent / "data" / "video"
)
_DEFAULT_STATIC_DIR = str(
    Path(__file__).resolve().parent / "static"
)


@dataclass
class IntegrationConfig:
    host: str = os.getenv("SA_HOST", "0.0.0.0")
    port: int = int(os.getenv("SA_PORT", "8001"))

    # Root data directory for video evidence
    data_root: Path = field(
        default_factory=lambda: Path(os.getenv("SA_DATA_ROOT", _DEFAULT_DATA_ROOT))
    )

    # Static assets directory for frontend
    static_dir: Path = field(
        default_factory=lambda: Path(os.getenv("SA_STATIC_DIR", _DEFAULT_STATIC_DIR))
    )

    @property
    def videos_dir(self) -> Path:
        return self.data_root / "videos"

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


DEFAULT_CONFIG = IntegrationConfig()
