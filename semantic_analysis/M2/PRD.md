# M2 — Audio / NLP — PRD

## 1. What M2 owns
Turn a video's audio track into timestamped text and searchable transcript embeddings:
extract audio, run speech-to-text (Whisper), chunk the transcript into segments with
timestamps, and embed each chunk. M2 does NOT do visual understanding (M1/M3) and does
NOT do retrieval/ranking (M4) — its job ends at producing clean, timestamped,
embedded transcript chunks that go into the common data contract's `audio` field.

## 2. Why this exists in the architecture
Spoken audio is often the most semantically dense signal in a short-form video — "First
remove the side panel, then loosen the bolt" tells you exactly what's happening in a
way no object detector can. Queries with speech-intent ("explaining the process",
"chef explaining a recipe") can only be answered if transcript text exists and is
timestamp-aligned with the visual/object evidence from M1/M3. M2 is what makes those
queries answerable at all.

## 3. MVP scope (must ship by 24 Aug) vs optional
**MVP:**
- Audio extraction from uploaded video via FFmpeg (mono, 16kHz — Whisper's expected
  input format)
- Whisper (pretrained, `openai-whisper` or `faster-whisper`) transcription with
  word/segment-level timestamps
- Transcript chunking into fixed-ish time windows (aligned to Whisper's natural segment
  boundaries, not mid-sentence) that match the segment granularity M1/M3 use, so all
  modalities can be joined on `video_id` + overlapping time range
- Sentence-embedding each chunk (Sentence-Transformers, e.g. `all-MiniLM-L6-v2`)
- Output written to the common data contract's `audio.transcript` /
  `audio.embedding_id` fields, embeddings stored in FAISS (or handed to M4 to store —
  clarify with M4 who owns the FAISS write, see §7)

**Optional / only if time remains, in priority order:**
1. Language/silence detection — skip Whisper calls entirely on silent segments (music
   with no speech, ambient shots) rather than transcribing noise. Cheap (RMS energy
   check before calling Whisper), avoids wasted compute and avoids Whisper
   hallucinating text on silence (a documented Whisper failure mode).
2. Speaker-count hinting (not full diarization — just "one speaker" vs "multiple/
   crosstalk" as a flag) if a quick VAD-based heuristic is available; skip real speaker
   diarization entirely, it's not worth the time under this deadline.
3. Basic profanity/filler-word cleanup on transcript text before embedding, if it's
   visibly hurting embedding quality in testing.

## 4. Non-goals
- No training/fine-tuning of Whisper or the embedding model.
- No real speaker diarization (pyannote-style) — out of scope for the deadline.
- No translation — assume single-language demo videos unless the team's demo set
  specifically needs multilingual support (check with the team before adding this).
- No building a custom ASR model — Whisper pretrained is the whole story here.

## 5. Success criteria for the demo
- Given a demo video with clear speech, M2 produces a transcript that's recognizably
  correct (spot-checked by ear against 2–3 demo videos).
- Timestamps on transcript chunks are close enough to the actual speech (within ~1–2s)
  that a judge scrubbing to the cited timestamp hears the relevant line.
- Transcript embeddings are queryable — a semantically related text query returns the
  correct chunk in a quick standalone similarity-search test (before even touching M4's
  full pipeline).
- No transcript text is silently blank on a video that clearly has spoken audio.

## 6. Niche/research-relevant note (keep in mind, low build priority)
Whisper is well known to hallucinate text during silence or non-speech audio (music,
noise) rather than outputting nothing — this is a real, documented failure mode, not
just a theoretical concern. The optional silence-detection gate in §3.1 is the direct,
cheap mitigation. If time doesn't allow building it, at minimum flag this as a known
limitation in judge Q&A rather than being caught off guard by a hallucinated transcript
line during a live demo.

## 7. Integration contract (what M2 promises M1/M3/M4/M5)
- Input: raw video file (or audio already extracted by M1, if M1's pipeline produces
  it first — don't assume, extract independently so M2 isn't blocked).
- Output: transcript chunks matching `COMMON_DATA_CONTRACT.md`'s `audio` field shape
  (`transcript` string + `embedding_id` reference), keyed by `video_id` and a time
  range (`start_ts`/`end_ts`) so M4 can join M2's chunks against M1/M3's segments by
  overlapping timestamp ranges.
- Embeddings: M2 produces the vectors; confirm with M4 whether M2 writes directly to
  the shared FAISS audio index or hands vectors + metadata to M4 to write — pick one
  owner explicitly on day 1 to avoid two modules racing to write the same index.
- Timestamps always in seconds (float), matching M1/M3's convention, never Whisper's
  raw frame/token indices.s