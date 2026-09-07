# M1 — Video / VLM Module
# Visual embedding & zero-shot scene description pipeline for semantic search
from .config import M1Config, DEFAULT_CONFIG
from .embedder import CLIPEmbedder
from .pipeline import process_video
from .validator import validate_segment_records

__all__ = [
    "M1Config",
    "DEFAULT_CONFIG",
    "CLIPEmbedder",
    "process_video",
    "validate_segment_records",
]
