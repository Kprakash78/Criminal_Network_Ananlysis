"""
M6 Full Stack / Integration — FastAPI Web Server
Exposes REST endpoints for video upload, multimodal search, video streaming, and video corpus discovery:
  - POST /upload       : Upload video file & execute end-to-end ingestion
  - POST /search       : Natural language semantic search (M5 Query Parsing -> M4 Retrieval -> M5 Answer Generation)
  - GET  /video/{id}   : Serve/stream MP4 video files
  - GET  /videos       : List ingested video corpus
  - Static file mount  : Serves frontend UI (HTML/CSS/JS)
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import M6Config, DEFAULT_CONFIG
from .orchestrator import run_full_ingestion

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("M6.app")

app = FastAPI(
    title="PS2614 Multimodal Short-Video Semantic Search API",
    description="Full-stack integration for multimodal short-video retrieval, visual embedding, speech, object detection, and LLM RAG answer generation.",
    version="1.0.0",
)

# Enable CORS for local dev flexiblity
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

cfg = DEFAULT_CONFIG

# Ensure data and static directories exist
cfg.videos_dir.mkdir(parents=True, exist_ok=True)
cfg.static_dir.mkdir(parents=True, exist_ok=True)

# Global in-memory cache for search results
GLOBAL_SEARCH_CACHE = {}

# ── Request / Response Pydantic Models ───────────────────────────────────────

class SearchRequest(BaseModel):
    query: str = Field(..., description="Natural language search query string")
    top_k: int = Field(default=5, ge=1, le=20, description="Maximum candidate segments to retrieve")


class SegmentScoreModel(BaseModel):
    final_score: float
    visual_similarity: Optional[float] = None
    transcript_similarity: Optional[float] = None
    object_match: Optional[float] = None
    ocr_match: Optional[float] = None


class SegmentResultModel(BaseModel):
    video_id: str
    segment_id: str
    start_ts: float
    end_ts: float
    visual_description: str = ""
    audio_transcript: str = ""
    objects: List[Dict[str, Any]] = []
    ocr: List[str] = []
    scores: SegmentScoreModel


class SearchResponseModel(BaseModel):
    query: str
    answer: str
    explanation_bullets: List[str] = []
    confidence: str = "HIGH"
    no_match: bool = False
    results: List[SegmentResultModel] = []


class UploadResponseModel(BaseModel):
    video_id: str
    filename: str
    status: str
    segments_count: int


# ── REST API Endpoints ────────────────────────────────────────────────────────

@app.post("/upload", response_model=UploadResponseModel, status_code=status.HTTP_201_CREATED)
async def upload_video(file: UploadFile = File(...)):
    """
    Upload a video file (.mp4) and execute full ingestion pipeline (M1 -> M2 -> M3 -> M4).
    """
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

        # Run ingestion orchestrator
        segments = run_full_ingestion(video_path=target_path, video_id=video_id, cfg=cfg)

        # Clear the global cache to fix "Stale Data" when new videos are added
        GLOBAL_SEARCH_CACHE.clear()
        log.info("Global search cache cleared due to new video upload.")

        return UploadResponseModel(
            video_id=video_id,
            filename=filename,
            status="ingested",
            segments_count=len(segments),
        )
    except Exception as exc:
        log.error("Upload/ingestion failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")


@app.post("/search", response_model=SearchResponseModel)
async def search_endpoint(req: SearchRequest):
    """
    Execute semantic query search:
      1. M5 query parser converts query to structured form
      2. M4 dense & sparse retrieval ranks candidate segments
      3. M5 answer generator produces grounded response with explainability bullets
    """
    query_str = req.query.strip()
    if not query_str:
        raise HTTPException(status_code=400, detail="Query string cannot be empty")

    log.info("Received search request: query=%r top_k=%d", query_str, req.top_k)

    # 1. Check Global Cache
    cache_key = f"{query_str.lower()}_{req.top_k}"
    if cache_key in GLOBAL_SEARCH_CACHE:
        log.info("Cache hit! Returning instantly for: %r", query_str)
        return GLOBAL_SEARCH_CACHE[cache_key]

    try:
        # Check if GEMINI_API_KEY is available for M5 LLM calls
        api_key = os.getenv("GEMINI_API_KEY")

        if api_key:
            from M5_llm_rag import run_pipeline as m5_run_pipeline
            m5_ans = m5_run_pipeline(user_query=query_str, top_k=req.top_k)

            answer_text = m5_ans.answer_text
            bullets = m5_ans.explanation_bullets
            confidence = m5_ans.confidence.upper()
            no_match = m5_ans.no_match
            cited_segs = m5_ans.cited_segments
        else:
            # Fallback path if GEMINI_API_KEY is not set: direct M4 retrieval
            log.warning("GEMINI_API_KEY not set; using direct M4 search fallback")
            from M4.search import search as m4_search

            terms = [t for t in query_str.lower().split() if len(t) > 2]
            struct_q = {
                "visual_concepts": [query_str],
                "objects": [{"object": t, "attributes": []} for t in terms],
                "ocr": terms,
                "context": [query_str],
            }
            m4_results = m4_search(structured_query=struct_q, raw_query=query_str, top_k=req.top_k)

            from M4.config import DEFAULT_CONFIG as m4_cfg
            conf_thresh = m4_cfg.confidence_threshold
            best_score = m4_results[0]["scores"]["final_score"] if m4_results else 0.0

            if best_score >= 0.60:
                confidence = "HIGH"
            elif best_score >= conf_thresh:
                confidence = "MEDIUM"
            else:
                confidence = "LOW"
            
            no_match = False # Never hide segments

            if confidence == "LOW":
                answer_text = "No strong match found for this query in the available video corpus."
                bullets = [
                    f"Highest retrieved fusion score ({best_score:.3f}) is below the confident match threshold ({conf_thresh:.2f})",
                    "Showing the closest available segments for transparency, but they are not confident matches."
                ]
            else:
                answer_text = (
                    f"Top relevant segment found at timestamp [{m4_results[0]['start_ts']:.1f}s - {m4_results[0]['end_ts']:.1f}s] "
                    f"in video '{m4_results[0]['video_id']}' with fusion score {best_score:.3f}."
                )
                bullets = [
                    f"Visual signal: {m4_results[0].get('visual', {}).get('description', 'visual concepts evaluated')}" if m4_results and m4_results[0].get('visual') else "Visual signal evaluated",
                    f"Audio transcript: \"{m4_results[0].get('audio', {}).get('transcript', 'speech transcript evaluated')}\"" if m4_results and m4_results[0].get('audio') and m4_results[0].get('audio', {}).get('transcript') else "Audio transcript evaluated",
                    f"Highest fusion score: {best_score:.3f}" if m4_results else "No candidate matches",
                ]
            cited_segs = m4_results

        # Format output segment objects
        formatted_results: List[SegmentResultModel] = []
        raw_list = cited_segs if not api_key else [
            # Map M5 CitedSegment objects or segment dicts
            getattr(s, "raw_segment", s) if hasattr(s, "raw_segment") else dict(s.__dict__) if hasattr(s, "__dict__") else s
            for s in (m5_ans.cited_segments or [])
        ]

        # If empty or M5 returned synthetic segment objects, fallback to direct M4 search results
        if not raw_list:
            from M4.search import search as m4_search
            _terms = [t for t in query_str.lower().split() if len(t) > 2]
            _stopwords = {"the", "and", "for", "are", "that", "this", "with", "where", "videos", "using"}
            _content_terms = [t for t in _terms if t not in _stopwords]
            _fallback_sq = {"objects": [{"object": t, "attributes": []} for t in _content_terms], "ocr": _content_terms}
            raw_list = m4_search(structured_query=_fallback_sq, raw_query=query_str, top_k=req.top_k)

        for seg in raw_list:
            if isinstance(seg, dict):
                v_id = seg.get("video_id", "unknown")
                s_id = seg.get("segment_id", f"{v_id}_seg_00")
                start_ts = float(seg.get("start_ts", 0.0))
                end_ts = float(seg.get("end_ts", start_ts + 2.0))

                vis = seg.get("visual") or {}
                aud = seg.get("audio") or {}
                sc = seg.get("scores") or {}

                formatted_results.append(
                    SegmentResultModel(
                        video_id=v_id,
                        segment_id=s_id,
                        start_ts=start_ts,
                        end_ts=end_ts,
                        visual_description=vis.get("description", "") if isinstance(vis, dict) else "",
                        audio_transcript=aud.get("transcript", "") if isinstance(aud, dict) else "",
                        objects=seg.get("objects", []),
                        ocr=seg.get("ocr", []),
                        scores=SegmentScoreModel(
                            final_score=float(sc.get("final_score", 0.0) or 0.0),
                            visual_similarity=sc.get("visual_similarity"),
                            transcript_similarity=sc.get("transcript_similarity"),
                            object_match=sc.get("object_match"),
                            ocr_match=sc.get("ocr_match"),
                        ),
                    )
                )

        # 2. Build final response
        final_response = SearchResponseModel(
            query=query_str,
            answer=answer_text,
            explanation_bullets=bullets,
            confidence=confidence,
            no_match=no_match,
            results=formatted_results[:req.top_k],
        )

        # 3. Save to Global Cache (and fix RAM explosion by limiting size)
        if len(GLOBAL_SEARCH_CACHE) > 500:
            GLOBAL_SEARCH_CACHE.clear()
            log.info("Global cache exceeded 500 items and was cleared to save RAM.")
            
        GLOBAL_SEARCH_CACHE[cache_key] = final_response

        return final_response
    except Exception as exc:
        log.error("Search execution failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}")


@app.get("/video/{video_id}")
async def stream_video(video_id: str):
    """
    Stream or serve MP4 video file by video_id.
    """
    # Sanitize video_id filename
    clean_id = Path(video_id).stem
    candidates = [
        cfg.videos_dir / f"{clean_id}.mp4",
        cfg.videos_dir / video_id,
    ]

    for path in candidates:
        if path.exists() and path.is_file():
            return FileResponse(path, media_type="video/mp4")

    # Search for matching file stem in videos directory
    for file in cfg.videos_dir.glob("*.mp4"):
        if file.stem == clean_id:
            return FileResponse(file, media_type="video/mp4")

    raise HTTPException(status_code=404, detail=f"Video '{video_id}' not found")


@app.get("/videos")
async def list_videos():
    """
    List available ingested video files in the corpus.
    """
    videos = []
    if cfg.videos_dir.exists():
        for file in sorted(cfg.videos_dir.glob("*.mp4")):
            videos.append({
                "video_id": file.stem,
                "filename": file.name,
                "size_mb": round(file.stat().st_size / (1024 * 1024), 2),
            })
    return {"count": len(videos), "videos": videos}


# Mount static assets directory
app.mount("/static", StaticFiles(directory=str(cfg.static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve main frontend web application HTML page."""
    index_file = cfg.static_dir / "index.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>PS2614 Semantic Search API Running</h1><p>Frontend index.html not found in static dir.</p>")
