# M4 Retrieval Module — Integration Notes

## 1. Multi-Modal Ingestion & FAISS Ownership
- **FAISS Write Ownership**: Per PRD.md §3/§7 recommendation, **M4 owns all FAISS vector index writes**. Upstream modules (M1 visual, M2 audio) pass raw L2-normalized float32 vectors (`_embedding_vector`) to M4, which adds them to `data/faiss_visual.index` (512-dim) and `data/faiss_audio.index` (384-dim), maintaining `data/id_map.json`.
- **Join Strategy & Overlap Threshold**:
  - `overlap_threshold` = **0.30** (Intersection-over-Union time overlap ratio).
  - Multi-modal segment joining matches M1 visual segment anchors with overlapping M2 audio transcripts and M3 object/OCR detections by `video_id` and timestamp range `[start_ts, end_ts]`.

## 2. Score Fusion & Default Weights
Scoring formula per PRD.md §6:
```
final_score = 0.35 * visual_similarity
            + 0.35 * transcript_similarity
            + 0.20 * object_match
            + 0.10 * ocr_match
```
- **Visual Similarity**: Computed via CLIP `ViT-B-32` (`openai`) text query embedding inner-product against visual FAISS index.
- **Transcript Similarity**: Computed via SentenceTransformers `all-MiniLM-L6-v2` text query embedding inner-product against audio FAISS index.
- **Object Match**: Includes attribute-binding check (PRD §8) against `objects[].attributes`.
- **OCR Match**: Binary term match against `ocr` text tokens.

## 3. Upstream Checkpoints & Contract Assumptions
- **Visual Embedding**: 512-dim L2-normalized vector from `open_clip` (`ViT-B-32`, `openai`).
- **Audio Embedding**: 384-dim L2-normalized vector from `sentence-transformers` (`all-MiniLM-L6-v2`).
- **Schema Compliance**: Ingestion reads and outputs segment records conforming strictly to `COMMON_DATA_CONTRACT.md`.

## 4. Current Integration State & Known Gaps
- **Downstream Consumer (M5 LLM/RAG)**: `M4` provides `search(structured_query, raw_query, top_k)` returning ranked segment dictionaries with populated `scores`. M5 can directly import `from M4.search import search`.
- **Audio Module Integration**: M2 code is pending in the repo; `M4` safely handles missing `audio` fields (`audio: None`, `transcript_similarity: 0.0`) without errors.
