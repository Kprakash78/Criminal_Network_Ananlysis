# M2 — Audio / NLP — Backend / Implementation Guide

## 1. Concepts you need (in order, don't over-study)
1. **Audio extraction**: pulling the audio track out of a video container into a
   plain waveform file (WAV), at the sample rate/format Whisper expects.
2. **ASR (automatic speech recognition) basics**: audio in, text + timestamps out.
   Treat Whisper as a black box — you don't need to understand its architecture.
3. **Segment-level timestamps**: Whisper naturally outputs text broken into segments
   with start/end times already — you're mostly consuming this, not building it.
4. **Sentence embeddings**: turning a text chunk into a fixed-size vector for semantic
   similarity search. Same idea as CLIP text embeddings, different model.

You do NOT need to learn: Whisper's transformer internals, beam search decoding
details, or how to train/fine-tune an ASR model.

## 2. Libraries and installs
```bash
pip install openai-whisper         # simplest, good enough for MVP
# OR, if speed matters and GPU is scarce:
pip install faster-whisper         # CTranslate2-based, notably faster on CPU
pip install sentence-transformers
pip install ffmpeg-python           # or shell out to ffmpeg binary directly
```
Recommended pretrained checkpoint: Whisper `base` or `small` for the MVP — good
accuracy/speed tradeoff for short-form video on limited hardware. Only go to
`medium`/`large` if `base`/`small` is visibly mistranscribing your actual demo videos
and you have GPU headroom to spare.

Recommended embedding model: `all-MiniLM-L6-v2` (Sentence-Transformers) — small, fast,
good enough semantic quality for a hackathon corpus, and it's the same
family/dimension class that's easy to reason about alongside M1's visual embeddings.

## 3. Audio extraction
```bash
ffmpeg -i input.mp4 -ac 1 -ar 16000 -vn output.wav
```
`-ac 1` (mono), `-ar 16000` (16kHz), `-vn` (no video) — this is exactly what Whisper
expects, so there's no resampling surprise later. Do this once per video, cache the
WAV, don't re-extract on every dev iteration.

## 4. Minimal code example (extraction → transcription → chunking → embedding)
```python
import whisper
from sentence_transformers import SentenceTransformer
import subprocess, os, uuid

asr_model = whisper.load_model("base")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_audio(video_path, out_path):
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path,
        "-ac", "1", "-ar", "16000", "-vn", out_path
    ], check=True, capture_output=True)

def transcribe(audio_path):
    # Whisper returns dict with 'segments', each with start/end/text already
    result = asr_model.transcribe(audio_path, verbose=False)
    return result["segments"]  # list of {start, end, text, ...}

def process_video_audio(video_id, video_path):
    audio_path = f"/data/audio/{video_id}.wav"
    os.makedirs(os.path.dirname(audio_path), exist_ok=True)
    extract_audio(video_path, audio_path)

    segments = transcribe(audio_path)

    chunks = []
    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue  # skip empty segments rather than embedding nothing
        embedding = embed_model.encode(text)
        embedding_id = f"aud_{video_id}_{uuid.uuid4().hex[:8]}"
        chunks.append({
            "video_id": video_id,
            "start_ts": round(seg["start"], 2),
            "end_ts": round(seg["end"], 2),
            "audio": {
                "transcript": text,
                "embedding_id": embedding_id,
            },
            "_embedding_vector": embedding,  # handed to FAISS writer, not stored in JSON
        })
    return chunks
```

## 5. Chunking strategy
- Default to Whisper's own segment boundaries (typically sentence- or phrase-length,
  a few seconds each) rather than inventing your own fixed-window chunker — Whisper
  already avoids cutting mid-word, and re-chunking risks breaking that.
- If Whisper's segments come out too fine-grained (many very short segments), merge
  adjacent segments up to a max window (e.g. 8–10s) before embedding, so each embedded
  chunk carries enough context to be semantically meaningful on its own.
- Keep the same `start_ts`/`end_ts` per-chunk fields M1/M3 use, in seconds — this is
  what lets M4 join across modalities by overlapping time range later.

## 6. Silence / non-speech handling (optional tier, cheap and worth it)
```python
import numpy as np
import soundfile as sf

def has_speech_energy(audio_path, threshold_db=-40):
    data, sr = sf.read(audio_path)
    rms = np.sqrt(np.mean(data**2))
    db = 20 * np.log10(rms + 1e-9)
    return db > threshold_db
```
Run this before calling Whisper on a chunk of audio (or on the whole file, coarsely)
to skip transcribing pure silence/music-only stretches. This directly mitigates
Whisper's documented tendency to hallucinate plausible-sounding text on non-speech
audio instead of returning nothing.

## 7. Batching / GPU-CPU considerations
- `openai-whisper`'s `base` model runs acceptably on CPU for short-form video length
  (15–90s clips) — test actual per-video processing time on day 1 so you know your
  real throughput budget across the whole demo corpus.
- `faster-whisper` is a drop-in-ish alternative worth switching to if CPU-only
  transcription is too slow in testing — meaningfully faster with the same accuracy
  class, minimal code change (different import, similar API).
- No need for batched inference across multiple videos simultaneously for a hackathon
  corpus size — process videos one at a time, sequentially, and don't build a job queue.

## 8. Testing methodology
1. **Transcription spot-check**: pick 3 demo videos, listen to a 20–30s clip, compare
   by ear against Whisper's output — check for major mistranscriptions, not perfection.
2. **Timestamp alignment check**: pick a few transcript lines, scrub to the cited
   timestamp in the source video, confirm the words are actually being said around
   there (within ~1–2s tolerance).
3. **Embedding sanity check**: embed a handful of transcript chunks plus a
   hand-written query text, compute cosine similarity directly (no FAISS needed for
   this quick check), and confirm the semantically closest chunk is intuitively
   correct — this catches embedding-pipeline bugs before M4 is even involved.
4. **Silence test** (if §6 is implemented): run against a video with a music-only or
   silent stretch, confirm no hallucinated transcript text appears for that stretch.
5. **Schema validation**: confirm output chunks match `COMMON_DATA_CONTRACT.md`'s
   `audio` field shape exactly, and that `start_ts`/`end_ts` are in seconds and
   monotonic within a video.

## 9. Common failure modes + debugging checklist
| Symptom | Likely cause | Fix |
|---|---|---|
| Empty/garbled transcript | Wrong sample rate/mono conversion, or genuinely silent audio | Re-check FFmpeg extraction command (§3); verify with §6's energy check |
| Hallucinated text during silence/music | Known Whisper behavior on non-speech audio | Implement §6 silence gate, or filter obviously-repetitive/looping hallucinated phrases |
| Timestamps off by a noticeable amount | Video/audio stream desync in source file, or timestamp units mismatch | Confirm FFmpeg extraction isn't dropping frames; confirm you're using Whisper's segment `start`/`end` directly, not recomputing from word count |
| Very slow processing | Using `medium`/`large` Whisper model unnecessarily, or no GPU and large model | Drop to `base`/`small`, or switch to `faster-whisper` |
| Embeddings don't retrieve semantically sensible chunks | Chunk text too short/fragmented to carry meaning | Merge adjacent short segments per §5 before embedding |
| Duplicate embedding_ids across videos | UUID/ID generation not actually unique, or copy-paste bug | Always include `video_id` in the ID string, verify uniqueness in a quick test loop |

## 10. Integration contract recap
See PRD §7. M2 extracts its own audio if needed (don't block on M1), outputs
timestamped transcript chunks matching the common contract's `audio` field, and
coordinates with M4 on who owns the actual FAISS write for the audio index.