# M1 — Video / VLM — PRD

## 1. What M1 owns
The upstream visual pipeline for the whole system: extract frames from uploaded
video, sample them sensibly, run a pretrained vision-language model (CLIP/SigLIP or a
captioning VLM) to produce (a) visual embeddings for semantic similarity search and
(b) short natural-language scene descriptions, all timestamp-aligned. M1 does NOT do
object detection/OCR (M3) and does NOT do speech (M2) — it owns the "what does this
scene generally look like / mean" signal, as opposed to M3's "what specific objects
and text are present" signal. The two are deliberately complementary, not overlapping.

## 2. Why this exists in the architecture
CLIP/SigLIP-style embeddings are what let the system match a query like "chef using an
air fryer" against a video frame even if no OCR text or exact object label says
"air fryer" — it's fuzzy semantic match, which is the backbone of the whole retrieval
system. M3's structured object/OCR metadata and M2's transcript are the precise,
citable evidence layered on top; M1's embeddings are the thing that actually finds
candidates in the first place. Without M1, M4 has no visual search signal at all —
this module is on the critical path, not an optional enhancement.

## 3. MVP scope (must ship by 24 Aug) vs optional
**MVP:**
- FFmpeg-based frame extraction at a fixed time interval (shared sampling convention
  with M3 — coordinate on the exact interval so M1's and M3's frame timestamps line up
  and downstream aggregation doesn't have to reconcile two different sampling grids)
- CLIP (or SigLIP) pretrained model producing a visual embedding per sampled frame
- A short scene description per sampled frame or per aggregated segment — either via
  a lightweight captioning VLM (e.g. BLIP-2 base, if time/compute allow) or, as a
  cheaper MVP fallback, zero-shot CLIP classification against a fixed candidate-label
  set to produce a templated description ("person, motorcycle, indoor" → "Person with
  a motorcycle indoors")
- Aggregation of frame-level embeddings into segment-level records (avoid one
  embedding per near-duplicate frame — see BACKEND.md §7)
- Output written to `metadata.jsonl`'s `visual` field per the common data contract,
  embeddings stored in FAISS (coordinate with M4 on who owns the FAISS write, same
  open question as M2 has for the audio index — pick an owner explicitly)

**Optional / only if time remains, in priority order:**
1. Real captioning VLM (BLIP-2 or similar) instead of the templated zero-shot fallback,
   if compute/time allow — meaningfully better scene descriptions for the demo.
2. Shot/scene-boundary detection (e.g. simple frame-difference thresholding) to drive
   smarter segment boundaries than fixed-interval sampling, instead of relying purely
   on embedding-similarity-based aggregation.
3. Frame-quality gating (blur/dark/noisy frame detection) — same rationale as M3's
   optional tier (documented UGC video degradation); if M3 already builds this, M1 can
   reuse the same flag/check rather than duplicating it — coordinate with M3 first.

## 4. Non-goals
- No training/fine-tuning of CLIP, SigLIP, or any captioning model.
- No building a custom scene-classification taxonomy — use whatever labels are useful
  for the demo's actual content, not an exhaustive general-purpose ontology.
- No real shot-detection research pipeline — the optional tier above is a cheap
  heuristic, not a robust production feature.

## 5. Success criteria for the demo
- Given a demo video, M1 produces a visual embedding + scene description for each
  segment, written to `metadata.jsonl` matching the common contract.
- A quick standalone similarity-search test (text query → CLIP text embedding → cosine
  similarity against stored visual embeddings) returns the intuitively correct segment
  for at least 3–5 hand-picked demo queries, before M4's full pipeline is even wired up.
- Scene descriptions are recognizably accurate by eye for the demo video set (doesn't
  need to be perfect prose, just correct at the level of "person repairing a
  motorcycle" vs. wildly wrong).
- No duplicate near-identical embeddings flooding one continuous shot (aggregation is
  working, not just raw per-frame dump).

## 6. Niche/research-relevant note (keep in mind, low build priority)
CLIP-style embeddings are documented to struggle with **compositional/attribute-binding
queries** — correctly telling "a black dog and white cat" from "a white dog and black
cat" — because attribute-object binding gets lost in cross-modal cosine-similarity
alignment even though the individual modalities encode the right information. M1
itself doesn't need to fix this (that's handled downstream via M3's optional
attribute-extraction tier feeding M5's attribute-aware scoring — see M3/M5 docs). What
M1 should do is simply be aware of it: don't oversell CLIP similarity alone as
precise for multi-object/multi-attribute queries in judge conversations, and make sure
segment descriptions are specific enough (not just "a person and a motorcycle" with no
other detail) to give the rest of the system something to work with.

## 7. Integration contract (what M1 promises M2/M3/M4/M5)
- Input: raw video file. M1 owns primary frame extraction; if M1's extraction isn't
  ready when M2/M3 need to start dev, they extract their own via FFmpeg independently
  (per their own docs) rather than blocking — do not assume M1 finishes first.
- Output: `metadata.jsonl` segment records populated in the `visual` field
  (`description` + `embedding_id`) matching `COMMON_DATA_CONTRACT.md` exactly, keyed
  by `video_id` and `start_ts`/`end_ts` in seconds — same timestamp convention as
  M2/M3 so all three can be joined by overlapping time range.
- Embeddings: M1 produces the vectors; confirm with M4 whether M1 writes directly to
  the shared FAISS visual index or hands vectors + metadata to M4 to write — same class
  of open question flagged in M2's PRD for the audio index, pick one owner per index
  explicitly on day 1.
- Frame sampling interval: coordinate explicitly with M3 (which also samples frames)
  so both modules either share the exact same sampled-frame set (more efficient — one
  extraction pass, both M1 and M3 consume it) or, if run independently, use the same
  interval so timestamps still line up cleanly for aggregation. Decide this on day 1,
  don't let it drift.