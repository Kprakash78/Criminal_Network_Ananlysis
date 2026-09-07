# M6 Full Stack / Integration — System-Wide Integration Notes

## 1. System Integration Overview
Module **M6** connects all five underlying pipelines into a unified full-stack web application:

```
[ User Input / Web UI ]
       │
       ▼
 [ POST /search ] ──────────► [ M5 Query Parser ]
       │                             │
       ▼                             ▼
 [ M4 Retrieval Engine ] ◄─── [ Structured Query ]
       │ (FAISS Visual + Audio Index + Metadata)
       ▼
 [ Grounded Candidates ] ───► [ M5 Answer Generator (Gemini RAG) ]
       │                             │
       ▼                             ▼
 [ Ranked Segments ] ◄─────── [ AI Answer & Evidence Bullets ]
       │
       ▼
 [ Web UI Player ] ───► HTML5 Video <video> timestamp seek (`currentTime = start_ts`)
```

---

## 2. Pipeline Integration Matrix

| Module | Purpose | Status | Integration Wiring |
| :--- | :--- | :--- | :--- |
| **M1** | Video / VLM | **REAL (Wired)** | `from M1.pipeline import process_video` — CLIP `ViT-B-32` frame embeddings & zero-shot descriptions. |
| **M2** | Audio / NLP | **REAL (Wired)** | `from M2.pipeline import process_video_audio` — Whisper ASR speech transcription & SentenceTransformers embeddings (`all-MiniLM-L6-v2`). |
| **M3** | Vision Intelligence | **REAL (Wired)** | `from M3.pipeline import process_video` — YOLOv8 object detection & EasyOCR text extraction. |
| **M4** | Retrieval / FAISS | **REAL (Wired)** | `from M4.pipeline import run_ingestion` & `from M4.search import search` — IoU multi-modal join, FAISS indices (`faiss_visual.index`, `faiss_audio.index`), and score fusion. |
| **M5** | LLM / RAG | **REAL (Wired)** | `from M5.pipeline import run_pipeline` — Gemini 2-call RAG pipeline (structured query parsing + answer generation with explainability bullets). Fallback to direct M4 retrieval if `GEMINI_API_KEY` is unset. |
| **M6** | Full Stack / UI | **REAL (Wired)** | FastAPI backend (`M6/app.py`), batch ingestion (`ingest_corpus.py`), and interactive dark-mode HTML5 web UI (`M6/static/`). |

---

## 3. How to Run the Demo

### Step 1: Pre-Ingest Video Corpus (Recommended for Stage Demos)
Run the batch ingestion script to process videos in `data/videos/` ahead of time:
```bash
python ingest_corpus.py
```

### Step 2: Launch Web Server
```bash
python -m uvicorn M6.app:app --host 0.0.0.0 --port 8000
```

### Step 3: Open Web UI
Navigate to `http://localhost:8000` in your web browser.

---

## 4. Judge Demo Walkthrough & Capability Summary

### What a Judge CAN Do:
1. **Interactive Semantic Search**: Type natural language queries (e.g. *"chef cooking food in air fryer"*, *"person repairing motorcycle wearing helmet"*, or *"highway scenic view"*).
2. **AI Answer & Evidence**: View generated natural language answers with confidence badges (`HIGH`, `MEDIUM`, `LOW`, `NO_MATCH`) and evidence bullet points citing specific signals (visual, speech, objects, OCR).
3. **Instant Video Timestamp Seek**: Clicking any returned candidate segment automatically updates the video player and jumps (`player.currentTime = start_ts`) directly to the cited video moment.
4. **Live Video Upload**: Upload new `.mp4` video files via the UI to trigger live ingestion.

### Known Gaps / Future Improvements:
- High-concurrency async job queues (Celery/Redis) for video ingestion — currently runs synchronously in background.
- BLIP-2 fine-grained captioning upgrade for M1 (currently using fast CLIP zero-shot classification).
