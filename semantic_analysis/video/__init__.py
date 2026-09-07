# video/ — Video/VLM module (was M1)
from .config import VideoConfig, DEFAULT_CONFIG
from .embedder import CLIPEmbedder
from .pipeline import process_video
from .validator import validate_segment_records

__all__ = [
    "VideoConfig",
    "DEFAULT_CONFIG",
    "CLIPEmbedder",
    "process_video",
    "validate_segment_records",
]
