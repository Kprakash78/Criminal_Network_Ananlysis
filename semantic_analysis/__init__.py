"""
semantic_analysis — Video Evidence Analysis Subsystem
=====================================================
Provides video ingestion, CLIP visual embeddings, Whisper ASR,
YOLO object detection, EasyOCR, FAISS retrieval, and offline
query/answer generation for the Criminal Analysis platform.

Module mapping (old → new):
  M1 → video/        (CLIP embeddings, frame sampling)
  M2 → audio/        (Whisper ASR, transcript embeddings)
  M3 → ocr/          (YOLO detection, EasyOCR, tracking)
  M4 → llm_rag/retrieval/  (FAISS index, multimodal search)
  M5 → llm_rag/      (local query parser, template answer generator)
  M6 → integration/  (FastAPI app, ingestion orchestrator)

ALL inference runs LOCALLY. No cloud APIs, no GEMINI_API_KEY required.
"""
from __future__ import annotations

from pathlib import Path

__version__ = "1.0.0"

# Centralized model directory
MODELS_DIR = Path(__file__).resolve().parent / "models"


def check_models() -> dict[str, str]:
    """
    Validate that local model files exist.
    Returns a dict of {model_name: "OK" | "MISSING" | "WARN"}.
    Does NOT download anything.
    """
    status: dict[str, str] = {}

    yolo_path = MODELS_DIR / "yolo" / "yolov8n.pt"
    status["YOLO"] = "OK" if yolo_path.exists() else "MISSING"

    # CLIP is loaded on demand via transformers/open_clip — just report
    status["CLIP"] = "OK (loaded on demand)"
    # Whisper is loaded on demand
    status["Whisper"] = "OK (loaded on demand)"
    # Plate detector is optional
    plate_candidates = list((MODELS_DIR / "plate").glob("*.pt")) + list((MODELS_DIR / "plate").glob("*.onnx"))
    status["PlateDetector"] = "OK" if plate_candidates else "WARN — not installed (plate OCR disabled)"

    return status


def print_model_status() -> None:
    """Print a startup readiness report to stdout."""
    statuses = check_models()
    print("\n── semantic_analysis model readiness ──")
    for name, s in statuses.items():
        tag = "[OK]" if s.startswith("OK") else ("[WARN]" if "WARN" in s else "[MISSING]")
        print(f"  {tag:10s} {name}: {s}")
    print()
