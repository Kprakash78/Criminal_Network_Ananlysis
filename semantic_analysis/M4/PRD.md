# M4 — Retrieval — PRD

## 1. What M4 owns
The convergence point of the whole system: own the FAISS vector stores (visual +
audio), own the metadata store (`metadata.jsonl` / equivalent), and combine four
independent signals — visual similarity, transcript similarity, object match, OCR
match — into one ranked list of candidate segments per query. M4 does NOT parse
natural language queries (M5 does that and hands M4 a structured query) and does NOT
generate the final answer (M5 does that with M4's results as evidence). M4's job is
retrieval and scoring only.

## 2. Why this exists in the architecture
M1, M2, and M3 each produce one useful but individually incomplete signal — visual
embeddings alone miss precise object claims, object detection alone misses semantic
"how it looks" context, transcript alone misses anything unspoken. M4 is what turns
three-plus independent, individually noisy signals into one reliable ranking, and it's
the only module that has both the raw embeddings (via FAISS) and the structured
metadata (objects/OCR/transcript) in one place at query time — which is exactly what's
needed to compute a combined score and hand back evidence-rich results M5 can use to
build a grounded, explainable answer.

## 3. MVP scope (must ship by 24 Aug) vs optional
**MVP:**
- Own two FAISS indexes: one for visual embeddings (from M1), one for transcript
  embeddings (from M2). Decide explicitly with M1 and M2 who performs the actual
  `index.add()` write — recommended default: **M4 owns all FAISS writes**, M1/M2/M3
  just hand M4 vectors + metadata via a shared ingestion function/file, so there's one
  writer and no race conditions. State this decision back to M1/M2 clearly (their docs
  flag it as open).
- An ingestion step that reads M1's visual records, M2's audio records, and M3's
  object/OCR records (each keyed by `video_id` + `start_ts`/`end_ts`), writes
  embeddings into FAISS, and merges everything into one metadata store matching
  `COMMON_DATA_CONTRACT.md`'s full segment-level shape (joining records that overlap
  in time into a single segment record with `visual`, `audio`, `objects`, and `ocr`
  all populated where available).
- A retrieval function with the exact signature M5 expects:
  `search(structured_query, raw_query, top_k=10) -> list[segment]`, computing:
  - `visual_similarity`: cosine similarity between CLIP-embedded raw_query text and
    stored visual embeddings
  - `transcript_similarity`: cosine similarity between embedded raw_query and stored
    transcript embeddings
  - `object_match`: fraction/count of structured_query's objects (and, if available,
    attributes) found in the segment's `objects` list
  - `ocr_match`: whether any structured_query term appears in the segment's `ocr` list
  - `final_score`: a simple weighted sum of the above (start with equal weights,
    tune empirically per §7)
- Results returned sorted by `final_score`, each populated with the `scores` object
  per the common contract, ready for M5 to format into context.

**Optional / only if time remains, in priority order:**
1. Score-weight tuning based on actual demo-query performance (see §7) — cheap,
   high-value, do this before anything else optional.
2. Attribute-binding-aware `object_match` scoring: if M3 supplies `objects[].attributes`
   and M5's structured query supplies attribute-bound objects, verify the *same*
   object carries the matching attribute rather than just checking both terms appear
   somewhere in the segment (see §8 — this is the concrete fix for the CLIP
   bag-of-words problem, and M4 is where it actually gets computed).
2b. A minimal caching layer (e.g. cache query embeddings within a session) — only if
   response latency is visibly annoying during demo rehearsal, not pre-emptively.
3. A basic score-explanation object (which signals fired and by how much) beyond the
   raw numeric `scores` fields, if M5 wants richer input for its explainability bullets
   than just numbers.

## 4. Non-goals
- No distributed vector database (Milvus/Qdrant/Pinecone) — flat FAISS is enough at
  hackathon corpus scale. Don't add infrastructure complexity that buys nothing at
  this scale.
- No learned re-ranking model — if reranking is needed at all, it's LLM-based and
  lives in M5 (optional tier there), not a trained model in M4.
- No real-time index updates during a live demo query — build the index once from the
  demo video corpus ahead of time; ingestion and query-time search are separate paths.

## 5. Success criteria for the demo
- Given the team's demo video corpus (ingested ahead of time), `search()` returns the
  correct top-1 segment for at least 5 hand-picked demo queries spanning: pure visual
  query, object-specific query, transcript/speech-intent query, and one combined query.
- `final_score` clearly differentiates a strong match from a weak one (not all scores
  clustered near the same value) — verified by printing the score distribution across
  a test query set during development, not assumed.
- No crashes/empty results on a query with a genuinely weak or absent match — returns
  a low-scoring result (or empty list) cleanly, which M5 uses to trigger its
  "no strong match" path.
- End-to-end smoke test: a query typed by a judge returns a segment whose cited
  timestamp, when scrubbed to in the actual video, visibly matches the query.

## 6. Score combination — starting point and tuning approach
Start simple and empirical, not theoretical:
```
final_score = 0.35 * visual_similarity
            + 0.35 * transcript_similarity
            + 0.20 * object_match
            + 0.10 * ocr_match
```
These starting weights are a reasonable prior (visual and transcript carry the most
semantic signal; object/OCR are precise but narrower), not a tuned result. Actually
tune them: run your 5+ demo queries against a few different weight combinations, see
which produces the best top-1 accuracy on judgment by eye, and lock in whatever works
best for your actual demo corpus before the deadline — don't ship the untuned prior.
If a query's structured form has no objects/actions (pure visual/semantic query), fall
back to `object_match = 0` and `ocr_match = 0` rather than skipping those terms
awkwardly — the weighted sum handles missing signal gracefully as long as absent
components are just zero, not `null`.

## 7. Integration contract (what M4 promises M1/M2/M3/M5)
- Input from M1: visual segment records (`visual.description`, `visual.embedding_id`,
  raw embedding vectors) keyed by `video_id` + time range.
- Input from M2: audio segment records (`audio.transcript`, `audio.embedding_id`, raw
  embedding vectors) keyed by `video_id` + time range.
- Input from M3: `metadata.jsonl` segment records (`objects`, `ocr`) keyed by
  `video_id` + time range.
- M4 is the join point: segments from M1/M2/M3 that overlap in time on the same
  `video_id` get merged into one unified segment record before/during ingestion. If
  time ranges don't align perfectly (different sampling intervals across modules),
  join on best-overlap rather than requiring exact match — document whatever overlap
  rule you actually use so M1/M2/M3 know what's expected of their timestamp output.
- Output to M5: `search(structured_query, raw_query, top_k) -> list[segment]`, each
  segment matching `COMMON_DATA_CONTRACT.md` exactly with `scores` populated.
- M4 owns all FAISS index writes (recommended default per §3) — M1/M2 hand off vectors
  + metadata, they don't call `index.add()` themselves, avoiding two modules racing to
  write the same index.

## 8. Niche/research-backed addition worth demoing (optional, ties M3+M4+M5 together)
This is the retrieval-side half of the attribute-binding fix flagged in M1/M3/M5's
docs: CLIP-style cosine-similarity matching is documented to lose track of which
attribute belongs to which object when a scene has multiple objects (e.g. confusing
"person in red, blue motorcycle" with "person in blue, red motorcycle"). The pure
embedding-similarity score (`visual_similarity`) can't fix this on its own — but
`object_match`, computed here in M4 against M3's per-object `attributes` field, can:
check that the *specific object* the query attributes to has that attribute in the
segment's detection, not just that both terms appear somewhere in the segment. This
is a real, concrete, defensible answer to "how does your system handle compositional
queries" if a judge asks — worth implementing if M3 shipped attribute extraction.