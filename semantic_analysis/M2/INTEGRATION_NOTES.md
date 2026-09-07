# M2 Integration Notes

## Current Capabilities
- **Audio Extraction**: M2 extracts audio via `imageio-ffmpeg` to ensure it isn't blocked waiting for M1 to do extraction.
- **Silence Check**: Implements a rough RMS energy threshold logic (-40dB default) to skip audio tracks with low perceived energy (silence, no speech). This successfully prevents Whisper from hallucinating text during dead air.
- **Transcription**: Uses Whisper (`base` model by default) to transcribe speech into small segments.
- **Chunking**: Fuses short transcription segments up to a 10s maximum window to ensure semantic search queries have enough context per chunk. Embeddings are generated using SentenceTransformers (`all-MiniLM-L6-v2`).
- **Data Shape**: Outputs standard dict objects strictly complying with `COMMON_DATA_CONTRACT.md` (`video_id`, `start_ts`, `end_ts`, and `audio` holding `transcript` and `embedding_id`).

## Known Gaps & Remaining Steps
- M2 module generates embeddings as vectors and stores them securely inside an internal `_embedding_vector` field on each chunk.
- **Index Write Ownership**: M2 **DOES NOT** write directly to the `faiss_audio.index` or `id_map.json`. Instead, M2 produces vectors and structured JSON schema so that M4 (Retrieval) handles the unified writes.
- We process clips sequentially and don't yet feature large batch extraction parallelism.
- Speaker diartization & language detection defaults to Whisper's internal capabilities and not separated out. 
- M2 explicitly needs M1/M3 bounds formatted correctly in float seconds (`start_ts`, `end_ts`), avoiding native token frames for seamless integration logic.
