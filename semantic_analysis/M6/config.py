"""
M6 Full Stack / Integration — Configuration
Defines server host/port, video storage directory, static asset directory,
and default API execution parameters.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class M6Config:
    host: str = os.getenv("M6_HOST", "0.0.0.0")
    port: int = int(os.getenv("M6_PORT", "8000"))

    # Root data directory
    data_root: Path = field(
        default_factory=lambda: Path(os.getenv("M6_DATA_ROOT", "data"))
    )

    # Static assets directory for frontend
    static_dir: Path = field(
        default_factory=lambda: Path(os.getenv("M6_STATIC_DIR", "M6/static"))
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


DEFAULT_CONFIG = M6Config()
