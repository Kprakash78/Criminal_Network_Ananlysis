# ocr/ — Vision Intelligence module (was M3)
from .config import OcrConfig, DEFAULT_CONFIG
from .pipeline import process_video

__all__ = ["OcrConfig", "DEFAULT_CONFIG", "process_video"]
