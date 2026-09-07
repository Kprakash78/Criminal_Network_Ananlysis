import logging
import os
import pathlib
import tempfile
import uuid
import subprocess
import numpy as np
import soundfile as sf
# pyrefly: ignore [missing-import]
import whisper
from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)

MAX_CHUNK_DURATION = 10.0
SILENCE_THRESHOLD_DB = -40

# Cross-platform audio extraction directory — works on Windows and Linux.
# Uses a sub-directory inside the system temp folder instead of hard-coded /tmp.
_AUDIO_TMP_DIR = pathlib.Path(tempfile.gettempdir()) / "m2_audio_extract"

asr_model = None
embed_model = None


def init_models():
    """Lazy initialize the models so importing the module doesn't block."""
    global asr_model, embed_model
    if asr_model is None:
        print("Loading Whisper base model...")
        asr_model = whisper.load_model("base")
    if embed_model is None:
        print("Loading MiniLM embedding model...")
        embed_model = SentenceTransformer("all-MiniLM-L6-v2")


def has_speech_energy(audio_path, threshold_db=SILENCE_THRESHOLD_DB):
    """Energy check to skip Whisper for purely silent or very quiet segments.

    Returns:
        (passed: bool, rms_db: float) — passed=True means energy is above threshold.
    """
    try:
        data, sr = sf.read(audio_path)
        if len(data) == 0:
            return False, -90.0
        rms = np.sqrt(np.mean(data**2))
        db = 20 * np.log10(rms + 1e-9)
        return db > threshold_db, db
    except Exception as e:
        log.warning("Energy check failed for %s: %s", audio_path, e)
        return True, 0.0  # Fallback: pass so we don't accidentally skip


# pyrefly: ignore [missing-import]
import imageio_ffmpeg


def extract_audio(video_path, out_path):
    """Extract audio via FFmpeg to mono, 16000 Hz WAV.

    Raises RuntimeError if the source video has no audio stream (the most common
    BUG-01 failure mode: videos downloaded video-only without audio).
    Raises subprocess.CalledProcessError for other ffmpeg failures.
    """
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    result = subprocess.run(
        [
            ffmpeg_exe, "-y", "-i", video_path,
            "-ac", "1", "-ar", "16000", "-vn", out_path,
        ],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        stderr_text = result.stderr.decode("utf-8", errors="replace")
        # Specific diagnostic for the video-only (no audio stream) case.
        # ffmpeg exits with rc=-22 (EINVAL) and this message when -vn is requested
        # but the source has no audio track at all.
        if "Output file does not contain any stream" in stderr_text:
            raise RuntimeError(
                f"Video '{video_path}' has NO audio stream — was downloaded video-only. "
                "Re-download with audio using redownload_audio.py."
            )
        raise subprocess.CalledProcessError(
            result.returncode, result.args, result.stdout, result.stderr
        )


def transcribe(audio_path):
    """Run Whisper to transcribe the audio with fast greedy decoding (beam_size=1)."""
    result = asr_model.transcribe(audio_path, verbose=False, beam_size=1, best_of=1)
    return result.get("segments", [])


def merge_segments(segments, max_window=MAX_CHUNK_DURATION):
    """
    Merge Whisper segments up to the max_window length in seconds
    so they have enough context for meaningful embeddings.
    """
    if not segments:
        return []

    chunks = []
    current_chunk = None

    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue

        if current_chunk is None:
            current_chunk = {
                "start": seg["start"],
                "end": seg["end"],
                "text": text,
            }
        else:
            duration = seg["end"] - current_chunk["start"]
            # Join if it fits within the max window
            if duration <= max_window:
                current_chunk["end"] = seg["end"]
                current_chunk["text"] += " " + text
            else:
                chunks.append(current_chunk)
                current_chunk = {
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": text,
                }

    if current_chunk is not None:
        chunks.append(current_chunk)

    return chunks


def process_video_audio(video_id, video_path):
    """
    End-to-end pipeline for M2.
    Takes a video ID and video file path, extracts audio, transcribes,
    chunks, embeds, and returns a list of dicts matching COMMON_DATA_CONTRACT.md.

    Returns empty list [] when:
    - Video has no audio stream (most common failure: see redownload_audio.py)
    - Audio is below the silence/energy threshold
    - Whisper produces no segments
    Logs the specific reason at WARNING level so callers can see why (fixes the
    silent-failure mode where the orchestrator's bare except swallowed errors).
    """
    log.info("M2 pipeline START for video_id=%r", video_id)
    init_models()

    # Cross-platform temp directory (replaces hard-coded /tmp/m2_audio_extract)
    _AUDIO_TMP_DIR.mkdir(parents=True, exist_ok=True)
    audio_path = str(_AUDIO_TMP_DIR / f"{video_id}.wav")

    try:
        extract_audio(video_path, audio_path)
    except RuntimeError as e:
        # Friendly message for the no-audio-stream case
        log.warning("M2 audio extraction: %s", e)
        return []
    except subprocess.CalledProcessError as e:
        stderr_text = e.stderr.decode("utf-8", errors="replace") if e.stderr else str(e)
        log.warning("M2 ffmpeg audio extraction failed for %r: %s", video_id, stderr_text[-300:])
        return []
    except Exception as e:
        log.warning("M2 audio extraction unexpected error for %r: %s", video_id, e)
        return []

    passed_energy, actual_db = has_speech_energy(audio_path)
    if not passed_energy:
        log.warning(
            "M2 audio below energy threshold (%.1f dB < %d dB threshold) — skipping "
            "transcription for %r",
            actual_db, SILENCE_THRESHOLD_DB, video_id,
        )
        return []

    # Transcribe and merge short Whisper segments
    segments = transcribe(audio_path)
    log.info("M2 Whisper returned %d raw segments for %r", len(segments), video_id)
    merged_chunks = merge_segments(segments, max_window=MAX_CHUNK_DURATION)
    log.info("M2 merged into %d chunks for %r", len(merged_chunks), video_id)

    output_records = []
    for chunk in merged_chunks:
        text = chunk["text"].strip()
        if not text:
            continue

        embedding = embed_model.encode(text)
        embedding_id = f"aud_{video_id}_{uuid.uuid4().hex[:8]}"

        start_ts = round(chunk["start"], 2)
        end_ts = round(chunk["end"], 2)

        if start_ts >= end_ts:
            end_ts = start_ts + 0.1  # Ensure monotonically increasing time range

        output_records.append({
            "video_id": video_id,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "audio": {
                "transcript": text,
                "embedding_id": embedding_id,
            },
            # Raw embedding vector handed to M4 for FAISS write; not stored in JSON.
            "_embedding_vector": embedding.tolist(),
        })

    log.info(
        "M2 pipeline DONE for %r: produced %d audio records",
        video_id, len(output_records),
    )
    return output_records


if __name__ == "__main__":
    # A quick integration test on execution
    import sys
    if len(sys.argv) > 2:
        vid_id = sys.argv[1]
        vid_path = sys.argv[2]
        print(f"Testing pipeline for {vid_id} at {vid_path}")
        results = process_video_audio(vid_id, vid_path)

        import json
        for r in results:
            r_copy = dict(r)
            del r_copy["_embedding_vector"]
            print(json.dumps(r_copy, indent=2))
