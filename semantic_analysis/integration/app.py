"""
integration/app.py — FastAPI Web Server for Video Evidence Analysis
====================================================================
Exposes REST endpoints for video upload, semantic search, and streaming.

Key changes from old M6/app.py:
  - ALL imports use semantic_analysis.* paths (no M1/M2/M3/M4/M5 references)
  - GEMINI_API_KEY check REMOVED — local query parser always used
  - Data root defaults to Criminal_detection/data/video/
  - Runs on port 8001 (Criminal_detection backend is on 8000)

Endpoints:
  POST /upload       — Upload video + run full ingestion pipeline
  POST /search       — Semantic search (local parse + FAISS + template answer)
  GET  /video/{id}   — Stream MP4 video files
  GET  /videos       — List available video corpus
  GET  /            — Serve frontend index.html
"""
from __future__ import annotations

import json
import logging
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import IntegrationConfig, DEFAULT_CONFIG
from .orchestrator import run_full_ingestion

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("semantic_analysis.integration.app")

app = FastAPI(
    title="Criminal Detection — Video Evidence Analysis API",
    description=(
        "Offline-only multimodal video retrieval. "
        "CLIP visual embeddings, Whisper ASR, YOLO object detection, FAISS retrieval."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

cfg = DEFAULT_CONFIG

cfg.videos_dir.mkdir(parents=True, exist_ok=True)
cfg.static_dir.mkdir(parents=True, exist_ok=True)

# In-memory search result cache
_SEARCH_CACHE: dict[str, Any] = {}


# ── Pydantic models ───────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str = Field(..., description="Natural-language search query")
    top_k: int = Field(default=5, ge=1, le=20, description="Max segments to retrieve")


class SegmentScores(BaseModel):
    final_score: float
    visual_similarity: Optional[float] = None
    transcript_similarity: Optional[float] = None
    object_match: Optional[float] = None
    ocr_match: Optional[float] = None


class SegmentResult(BaseModel):
    video_id: str
    segment_id: str
    start_ts: float
    end_ts: float
    visual_description: str = ""
    audio_transcript: str = ""
    objects: List[Dict[str, Any]] = []
    ocr: List[str] = []
    scores: SegmentScores


class SearchResponse(BaseModel):
    query: str
    answer: str
    explanation_bullets: List[str] = []
    confidence: str = "HIGH"
    no_match: bool = False
    results: List[SegmentResult] = []


class UploadResponse(BaseModel):
    video_id: str
    filename: str
    status: str
    segments_count: int


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_video(file: UploadFile = File(...)):
    """Upload a video file and execute the full ingestion pipeline."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file has no filename")

    filename = Path(file.filename).name
    stem = Path(filename).stem.replace(" ", "_")
    video_id = stem if stem else f"video_{uuid.uuid4().hex[:8]}"
    target_path = cfg.videos_dir / f"{video_id}.mp4"

    try:
        with open(target_path, "wb") as fh:
            shutil.copyfileobj(file.file, fh)
        log.info("Saved uploaded file to %s", target_path)

        segments = run_full_ingestion(video_path=target_path, video_id=video_id, cfg=cfg)
        _SEARCH_CACHE.clear()
        log.info("Search cache cleared after new upload.")

        return UploadResponse(
            video_id=video_id,
            filename=filename,
            status="ingested",
            segments_count=len(segments),
        )
    except Exception as exc:
        log.error("Upload/ingestion failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")


@app.post("/search", response_model=SearchResponse)
async def search_endpoint(req: SearchRequest):
    """
    Execute offline semantic search:
      1. Local keyword query parser (no LLM)
      2. FAISS + metadata retrieval
      3. Template-based answer generator (no LLM)
    """
    query_str = req.query.strip()
    if not query_str:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    log.info("Search: query=%r top_k=%d", query_str, req.top_k)

    cache_key = f"{query_str.lower()}_{req.top_k}"
    if cache_key in _SEARCH_CACHE:
        log.info("Cache hit for: %r", query_str)
        return _SEARCH_CACHE[cache_key]

    try:
        # Always use local pipeline — no cloud API key needed
        from semantic_analysis.llm_rag.pipeline import run_pipeline
        m5_ans = run_pipeline(user_query=query_str, top_k=req.top_k)

        answer_text = m5_ans.answer_text
        bullets = m5_ans.explanation_bullets
        confidence = m5_ans.confidence.upper()
        no_match = m5_ans.no_match

        # Build formatted segment results
        # Try to use M4-style search to get full segment dicts for display
        formatted_results: List[SegmentResult] = []

        try:
            from semantic_analysis.llm_rag.retrieval.search import search as ret_search
            from semantic_analysis.llm_rag.query_parser import parse_query
            sq = parse_query(query_str)
            sq_dict = {
                "objects": [{"object": o.object, "attributes": o.attributes} for o in sq.objects],
                "actions": sq.actions,
                "persons": sq.persons,
            }
            raw_list = ret_search(structured_query=sq_dict, raw_query=query_str, top_k=req.top_k)
        except Exception:
            raw_list = []

        for seg in raw_list:
            if isinstance(seg, dict):
                v_id = seg.get("video_id", "unknown")
                s_id = seg.get("segment_id", f"{v_id}_seg_00")
                start_ts = float(seg.get("start_ts", 0.0))
                end_ts = float(seg.get("end_ts", start_ts + 2.0))
                vis = seg.get("visual") or {}
                aud = seg.get("audio") or {}
                sc = seg.get("scores") or {}

                formatted_results.append(SegmentResult(
                    video_id=v_id,
                    segment_id=s_id,
                    start_ts=start_ts,
                    end_ts=end_ts,
                    visual_description=vis.get("description", "") if isinstance(vis, dict) else "",
                    audio_transcript=aud.get("transcript", "") if isinstance(aud, dict) else "",
                    objects=seg.get("objects", []),
                    ocr=seg.get("ocr", []),
                    scores=SegmentScores(
                        final_score=float(sc.get("final_score", 0.0) or 0.0),
                        visual_similarity=sc.get("visual_similarity"),
                        transcript_similarity=sc.get("transcript_similarity"),
                        object_match=sc.get("object_match"),
                        ocr_match=sc.get("ocr_match"),
                    ),
                ))

        final_resp = SearchResponse(
            query=query_str,
            answer=answer_text,
            explanation_bullets=bullets,
            confidence=confidence,
            no_match=no_match,
            results=formatted_results[:req.top_k],
        )

        if len(_SEARCH_CACHE) > 500:
            _SEARCH_CACHE.clear()
        _SEARCH_CACHE[cache_key] = final_resp
        return final_resp

    except Exception as exc:
        log.error("Search failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}")


@app.get("/video/{video_id}")
async def stream_video(video_id: str):
    """Stream or serve an MP4 video file by video_id."""
    clean_id = Path(video_id).stem
    candidates = [
        cfg.videos_dir / f"{clean_id}.mp4",
        cfg.videos_dir / video_id,
    ]
    for path in candidates:
        if path.exists() and path.is_file():
            return FileResponse(path, media_type="video/mp4")
    for f in cfg.videos_dir.glob("*.mp4"):
        if f.stem == clean_id:
            return FileResponse(f, media_type="video/mp4")
    raise HTTPException(status_code=404, detail=f"Video '{video_id}' not found")


@app.get("/videos")
async def list_videos():
    """List all ingested video files in the corpus."""
    videos = []
    if cfg.videos_dir.exists():
        for file in sorted(cfg.videos_dir.glob("*.mp4")):
            videos.append({
                "video_id": file.stem,
                "filename": file.name,
                "size_mb": round(file.stat().st_size / (1024 * 1024), 2),
            })
    return {"count": len(videos), "videos": videos}


# Mount static files
app.mount("/static", StaticFiles(directory=str(cfg.static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve the frontend index.html."""
    index_file = cfg.static_dir / "index.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Video Analysis API Running</h1><p>Frontend index.html not found.</p>")
