# M1 Video / VLM — Integration Notes

## 1. Frame Extraction & Sampling Convention
- **Shared Sampling Pass**: M1 frame extraction uses `M1/sampler.py` which dynamically imports and reuses `M3.sampler.sample_frames` when available. This avoids duplicate frame extraction when both M1 (visual/VLM) and M3 (objects/OCR) run on the same video.
- **Sampling Interval**: Standardized at **1.0 second** (configurable via `M1_FRAME_INTERVAL` or `M1Config.frame_interval_sec`).
- **Timestamp & Storage Convention**: Frames are named `data/frames/<video_id>/frame_<ts>.jpg` where `<ts>` represents timestamp in seconds (e.g. `frame_0.000.jpg`, `frame_1.000.jpg`).

## 2. FAISS Index & Metadata Ownership Decision
- **Visual Index Ownership**: Per PRD.md §7 open decision, M1 **writes directly to the shared FAISS visual index** (`data/faiss_visual.index`) and updates `data/id_map.json` under `id_map["visual"][embedding_id] = index_row`.
- **JSONL Ingestion**: Segment-level records are merged into `data/metadata.jsonl`. If `M3` or another module already generated a segment record for a given `segment_id`, `M1` merges its `visual` block (`description`, `embedding_id`, `quality_flag`) into the record without overwriting existing `objects` or `ocr` keys.

## 3. Assumptions about Downstream Modules
- **M4 (Retrieval / FAISS)**: Assumes M4 reads `data/faiss_visual.index`, `data/id_map.json`, and `data/metadata.jsonl`. When querying, M4 embeds text queries with CLIP (`open_clip` `ViT-B-32`, L2-normalized) and performs inner-product / cosine similarity search against `faiss_visual.index`.
- **M5 (LLM / RAG)**: Assumes M5 uses `visual.description` in prompt context for natural language reasoning and query answer generation.
- **M2 & M3 (Audio & Objects/OCR)**: Assumes segment start/end timestamps line up across modalities using shared float seconds (`start_ts`, `end_ts`).

## 4. What is NOT yet integrated / Known Gaps
- **BLIP-2 / Heavy VLM Captioning**: Currently using the fast MVP zero-shot CLIP label classification fallback for frame descriptions. Can be upgraded to BLIP-2 if extra GPU compute is allocated.
- **Shot-Boundary Detection**: Segments are aggregated based on frame embedding cosine similarity thresholding (>= 0.85). Heuristic shot-boundary thresholding can be added as an optional refinement.
- **Compositional Attribute Binding**: CLIP embeddings focus on global scene semantics. Compositional attribute binding (e.g., distinguishing "red car and blue house" vs "blue car and red house") relies on M3 object attribute detection & M5 query handling.
