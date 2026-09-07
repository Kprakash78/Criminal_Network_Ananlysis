# M3 Vision Intelligence — Integration Notes

## What M3 currently provides

M3 is a standalone vision intelligence module for the PS2614 multimodal video search
system. It produces `data/metadata.jsonl` — append-only, one JSON segment record per
line — conforming exactly to `COMMON_DATA_CONTRACT.md`.

### Pipeline (what runs today, end-to-end)
```
video file (.mp4 / .avi / etc.)
  ↓ FFmpeg (or OpenCV fallback) frame extraction at fixed interval (default 1s)
  ↓ YOLOv8n object detection per frame  (confidence ≥ 0.5, configurable)
  ↓ EasyOCR text extraction             (every 2s, deduped per segment)
  ↓ Laplacian-variance blur quality gate (→ quality_flag: ok / low_quality)
  ↓ Optional: dominant-color attribute extraction per detected box
  ↓ Jaccard-similarity segment aggregation (dedup consecutive same-scene frames)
  ↓ data/metadata.jsonl  (append-only, idempotent re-run)
```

### Files produced
- `data/metadata.jsonl`  — the primary output, consumed by M4
- `data/frames/<video_id>/frame_<ts>.jpg` — temp, deleted after processing by default

---

## What is NOT yet integrated

### M1 (Frame Extraction / VLM Embeddings)
- **Not integrated**. M3 extracts its own frames via FFmpeg/OpenCV — this is
  intentional (PRD §7: "do not block on M1").
- When M1 is ready: M3 can accept a pre-populated `data/frames/<video_id>/` directory
  and skip its own extraction. Call `sample_frames(..., force_redo=False)` — it will
  skip if frames already exist.
- **Gap**: `visual.embedding_id` and `visual.description` in segment records are
  currently absent (not populated). M1 must fill these via its VLM embedding step
  and then update the matching `segment_id` records in `metadata.jsonl`, OR M4 must
  join M1 and M3 outputs at query time.

### M2 (Whisper Audio / Transcript)
- **Not integrated**. `audio.transcript` and `audio.embedding_id` fields in segment
  records are absent — M3 does not write them (correct — M2 owns those fields).
- Integration point: M2 writes transcript records indexed by `video_id` and timestamp
  range. M4 must align them with M3's segment boundaries at query time.

### M4 (FAISS Retrieval / Scoring)
- **Not integrated**. M3 writes `metadata.jsonl` but M4 has not yet been hooked up
  to read it.
- M3's `scores.*` fields are all `null` as required by the contract — M4 populates
  them at query time.
- Integration point: M4 reads `metadata.jsonl`, loads visual/audio FAISS indices
  (built by M1/M2), and for each query computes `visual_similarity`,
  `transcript_similarity`, `object_match`, `ocr_match`, `final_score`.

### M5 (LLM / RAG)
- **Not integrated**. M5 has its own standalone pipeline.
- M5 depends on M4's retrieval output (segments with `scores` populated).
- `objects[].attributes` fields M3 supplies (dominant color) are ready for M5's
  attribute-binding logic once M5 implements that optional tier.

### M6 (FastAPI + Frontend)
- **Not integrated**. M3 has no HTTP endpoint — it is a batch processing function.
- Integration: M6 should call `M3.pipeline.process_video(video_path, video_id)`
  or shell out to `python -m M3 <video>` for new video uploads.

---

## Known gaps vs full architecture

| Gap | Severity | Notes |
|-----|----------|-------|
| `visual.description` not populated | Medium | Needs M1's VLM (CLIP/BLIP) step |
| `visual.embedding_id` not populated | Medium | Needs M1's FAISS write step |
| `audio.*` not populated | Medium | Needs M2's Whisper step |
| Segment boundaries don't account for M2 transcript timing | Low | Jaccard dedup is frame-visual only; speech boundaries may differ |
| `yolov8n.pt` COCO classes only | Low | No custom classes; "motorcycle" may be labelled "bicycle" — accepted limitation per BACKEND.md §11 |
| No real object tracking (ByteTrack) | Low | Optional PRD §3.3 item; naive Jaccard dedup is sufficient for MVP |
| OCR language: English only | Low | EasyOCR configured `["en"]`; add more languages if needed |
| FFmpeg path | Low | FFmpeg installed via winget; may need `$env:PATH` refresh in new shells |

---

## Running M3 standalone

```powershell
# Generate a synthetic test video and run full verification
python -m M3.verify_pipeline --generate-sample

# Process a real video
python -m M3 path/to/video.mp4 --video-id my_video_001

# Validate the output
python -m M3.validator data/metadata.jsonl

# Full options
python -m M3 --help
```

## Module layout
```
M3/
  __init__.py          — package marker
  __main__.py          — python -m M3 entry point
  config.py            — all tunable parameters
  sampler.py           — FFmpeg / OpenCV frame extraction
  detector.py          — YOLO + EasyOCR + quality gate + color attributes
  aggregator.py        — Jaccard-based segment aggregation
  writer.py            — append-only JSONL writer
  pipeline.py          — orchestrator + CLI
  validator.py         — schema validation against COMMON_DATA_CONTRACT.md
  verify_pipeline.py   — end-to-end verification (STEP 2)
  make_sample_video.py — synthetic test video generator
  requirements.txt
  INTEGRATION_NOTES.md (this file)
```
