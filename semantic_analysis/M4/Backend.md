# M4 — Retrieval — Backend / Implementation Guide

## 1. Concepts you need (in order, don't over-study)
1. **FAISS basics**: a library for fast nearest-neighbor search over vectors. You need
   exactly two operations — `index.add(vectors)` to ingest, `index.search(query_vector,
   k)` to retrieve nearest neighbors. That's genuinely most of what you need.
2. **Cosine similarity via inner product**: if vectors are L2-normalized before adding,
   FAISS's `IndexFlatIP` (inner product) index gives you cosine similarity directly —
   simpler than computing cosine similarity manually after retrieval.
3. **ID mapping**: FAISS indexes return integer row positions, not your own IDs — you
   need a small mapping (`row_index -> embedding_id -> segment record`) maintained
   alongside the index.
4. **Score fusion**: combining multiple independent 0–1 signals into one ranking score
   via a weighted sum. Nothing more sophisticated is needed at this scale — do not
   reach for a learned reranker.

You do NOT need to learn: FAISS's approximate-search index types (IVF, HNSW, PQ) —
those matter at millions-of-vectors scale, not a hackathon corpus. `IndexFlatIP` (exact
brute-force search) is fast enough for the corpus size you'll actually have, and it's
far simpler to reason about and debug.

## 2. Libraries and installs
```bash
pip install faiss-cpu       # faiss-gpu only if you have real scale needs, you won't
pip install numpy
pip install open_clip_torch  # to embed query text into the same space as M1's visual embeddings
pip install sentence-transformers  # to embed query text into the same space as M2's transcript embeddings
```
Important: query-time text embedding for visual search must use the **same CLIP
checkpoint** M1 used for images (so text and image embeddings are comparable), and
query-time text embedding for transcript search must use the **same
Sentence-Transformers model** M2 used for transcript chunks. Confirm the exact
checkpoint names with M1/M2 — a mismatch here silently produces meaningless similarity
scores, not an error.

## 3. FAISS index setup
```python
import faiss
import numpy as np

VISUAL_DIM = 512   # matches CLIP ViT-B-32 output dim -- confirm against M1's actual checkpoint
AUDIO_DIM = 384     # matches all-MiniLM-L6-v2 output dim -- confirm against M2's actual checkpoint

visual_index = faiss.IndexFlatIP(VISUAL_DIM)
audio_index = faiss.IndexFlatIP(AUDIO_DIM)

# id_map[i] = embedding_id string for row i in the index
visual_id_map = []
audio_id_map = []

def add_visual(embedding_id: str, vector: np.ndarray):
    vec = vector / np.linalg.norm(vector)   # normalize for cosine-via-inner-product
    visual_index.add(vec.reshape(1, -1).astype("float32"))
    visual_id_map.append(embedding_id)

def add_audio(embedding_id: str, vector: np.ndarray):
    vec = vector / np.linalg.norm(vector)
    audio_index.add(vec.reshape(1, -1).astype("float32"))
    audio_id_map.append(embedding_id)
```
Save/load: `faiss.write_index(visual_index, "faiss_visual.index")` /
`faiss.read_index(...)`. Persist `visual_id_map`/`audio_id_map` as plain JSON alongside
the index files — this is your `id_map.json` from `COMMON_DATA_CONTRACT.md`.

## 4. Ingestion: joining M1/M2/M3 output into unified segment records
```python
def join_segments(visual_records, audio_records, object_records, overlap_threshold=0.3):
    """
    Each *_records list is keyed by video_id + (start_ts, end_ts).
    Merge records from different modules whose time ranges overlap on the same video
    into one unified segment matching COMMON_DATA_CONTRACT.md.
    """
    def overlaps(a_start, a_end, b_start, b_end):
        inter = max(0, min(a_end, b_end) - max(a_start, b_start))
        union = max(a_end, b_end) - min(a_start, b_start)
        return (inter / union) if union > 0 else 0

    unified = []
    for v in visual_records:
        seg = {
            "video_id": v["video_id"],
            "start_ts": v["start_ts"], "end_ts": v["end_ts"],
            "visual": v["visual"], "audio": None, "objects": [], "ocr": [],
        }
        for a in audio_records:
            if a["video_id"] == v["video_id"] and overlaps(
                v["start_ts"], v["end_ts"], a["start_ts"], a["end_ts"]
            ) >= overlap_threshold:
                seg["audio"] = a["audio"]
                break
        for o in object_records:
            if o["video_id"] == v["video_id"] and overlaps(
                v["start_ts"], v["end_ts"], o["timestamp"], o["timestamp"]
            ) >= 0:  # a single timestamp always "overlaps" if inside the range
                if v["start_ts"] <= o["timestamp"] <= v["end_ts"]:
                    seg["objects"].extend(o["objects"])
                    seg["ocr"].extend(o["ocr"])
        unified.append(seg)
    return unified
```
This is intentionally a simple, readable O(n*m) join — fine at hackathon corpus scale.
Don't optimize this into an indexed join unless ingestion is actually slow in testing.
Document the `overlap_threshold` you land on so M1/M2/M3 know what's expected.

## 5. Retrieval + scoring function (the core M4 deliverable)
```python
def object_match_score(structured_query_objects, segment_objects):
    if not structured_query_objects:
        return 0.0
    seg_labels = {o["label"].lower(): o for o in segment_objects}
    hits = 0
    for q_obj in structured_query_objects:
        label = q_obj["object"].lower()
        if label in seg_labels:
            # attribute-binding check (optional tier, see PRD §8) -- only counts as a
            # full hit if the query's attributes for THIS object actually match the
            # detected object's own attributes, not just present anywhere in the segment
            q_attrs = set(a.lower() for a in q_obj.get("attributes", []))
            seg_attrs = set(a.lower() for a in seg_labels[label].get("attributes", []))
            if not q_attrs or q_attrs & seg_attrs:
                hits += 1
    return hits / len(structured_query_objects)

def ocr_match_score(structured_query, segment_ocr):
    if not segment_ocr:
        return 0.0
    terms = [o["object"] for o in structured_query.get("objects", [])] + structured_query.get("context", [])
    text = " ".join(segment_ocr).lower()
    return 1.0 if any(t.lower() in text for t in terms) else 0.0

def search(structured_query: dict, raw_query: str, unified_segments: list, top_k=10):
    query_visual_vec = embed_query_visual(raw_query)   # CLIP text encoder, same checkpoint as M1
    query_audio_vec = embed_query_audio(raw_query)      # Sentence-Transformers, same model as M2

    results = []
    for seg in unified_segments:
        visual_sim = cosine(query_visual_vec, seg["visual"]["_vector"]) if seg.get("visual") else 0.0
        transcript_sim = cosine(query_audio_vec, seg["audio"]["_vector"]) if seg.get("audio") else 0.0
        obj_match = object_match_score(structured_query.get("objects", []), seg.get("objects", []))
        ocr_match = ocr_match_score(structured_query, seg.get("ocr", []))

        final = 0.35 * visual_sim + 0.35 * transcript_sim + 0.20 * obj_match + 0.10 * ocr_match

        seg_out = dict(seg)
        seg_out["scores"] = {
            "visual_similarity": round(visual_sim, 3),
            "transcript_similarity": round(transcript_sim, 3),
            "object_match": round(obj_match, 3),
            "ocr_match": round(ocr_match, 3),
            "final_score": round(final, 3),
        }
        results.append(seg_out)

    results.sort(key=lambda s: s["scores"]["final_score"], reverse=True)
    return results[:top_k]
```
In practice, prefer computing `visual_sim`/`transcript_sim` via `faiss_index.search()`
against the query vector rather than looping cosine-by-hand over every segment once
your corpus is more than trivially small — the loop above is for clarity; swap in
actual FAISS search calls (`index.search(query_vec, k)`) once ingestion is done, using
the `id_map` to translate FAISS row hits back to segment records.

## 6. Testing methodology
1. **Ingestion check**: after running the join step, spot-check 3–5 unified segments —
   do they have plausible `visual`, `audio`, `objects`, `ocr` populated where the
   source video actually has that content, and empty/null where it doesn't.
2. **Score distribution check**: run your 5+ demo queries, print the full `scores`
   breakdown for the top 5 results each, confirm scores actually differentiate (not
   all clustered near 0.5) — this is what tells you whether weight tuning (PRD §6) is
   needed.
3. **Embedding-space sanity check**: before trusting `search()`, directly verify
   query-text embeddings and stored embeddings are in the same space — embed a text
   query that's an exact restatement of a segment's own `visual.description` or
   `audio.transcript`, confirm it scores very highly against that specific segment
   (near 1.0 cosine sim). If it doesn't, you have a checkpoint-mismatch bug (§2), not a
   ranking-quality problem — debug that first.
4. **No-match behavior**: run a query about content absent from the corpus, confirm
   `final_score` for the "best" result is clearly low rather than misleadingly high.
5. **End-to-end integration test**: run all 5+ demo queries through the full
   `search()` path and manually verify top-1 result and its timestamp are correct
   against the actual source video — this is the test to run the day before the
   deadline, not for the first time live.

## 7. Common failure modes + debugging checklist
| Symptom | Likely cause | Fix |
|---|---|---|
| All scores look similar/undifferentiated | Embedding checkpoint mismatch between query-time and ingestion-time (§2), or vectors not normalized | Run the embedding-space sanity check (§6.3) first, always |
| `search()` crashes on missing modality | Segment has `audio: None` or empty `objects` and code assumes it's always present | Guard every score computation for missing fields, per §5's `if seg.get(...)` pattern |
| Ingestion join produces duplicate or missing segments | `overlap_threshold` too strict/loose, or timestamp units mismatch across M1/M2/M3 | Print join statistics (segments in, segments out, join hit rate) during ingestion; confirm all three modules use seconds consistently |
| Object match score always 0 even for correct matches | Label casing mismatch ("Person" vs "person"), or structured query field name mismatch with M5's actual schema | Lowercase-normalize both sides; confirm the exact field names in M5's `StructuredQuery` match what M4's scoring code expects |
| Weights feel arbitrary / final ranking doesn't match intuition | Never actually tuned past the starting-point weights | Run PRD §6's tuning pass against your real demo queries before the deadline |
| Slow query response | Looping cosine-by-hand over every segment instead of using FAISS's actual search | Swap to real `index.search()` calls per §5's closing note |

## 8. Integration contract recap
See PRD §7. M4 owns FAISS writes and the join across M1/M2/M3's outputs, exposes
`search(structured_query, raw_query, top_k)` to M5, and returns segments matching
`COMMON_DATA_CONTRACT.md` with `scores` populated.