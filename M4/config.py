"""
M4 Configuration
PS 26152 — AI-Powered Criminal Network Analysis System

Single source of truth for all model names, paths, thresholds, and
generation parameters. Change values here; never hardcode them in module code.

Privacy guarantee:
  All model identifiers point to local HuggingFace Hub cache paths.
  No model is downloaded at inference time — if a model isn't cached,
  the system fails loudly rather than making a network call.
  Set HF_DATASETS_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 in production
  to enforce this at the OS level (see M4.1 test).
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class M4Config:
    # ---- Local LLM ----
    # Priority order: use the first model that loads without OOM.
    # All must already be in the local HuggingFace cache.
    # "google/flan-t5-large" is the always-viable CPU fallback (~770 MB).
    llm_model_candidates: tuple[str, ...] = (
        str(Path.home() / ".cache" / "huggingface" / "hub" / "models--google--flan-t5-large" / "snapshots" / "direct_download"),
        "google/flan-t5-large",
        "google/flan-t5-base",
    )
    # Generation parameters — low temperature for reproducibility (PRD NFR)
    llm_max_new_tokens: int = 512
    llm_temperature: float = 0.1
    llm_do_sample: bool = False          # greedy decoding unless temp > 0

    # ---- Embedding model ----
    # Must already be in the local HuggingFace cache.
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384       # MiniLM-L6-v2 output dimension

    # ---- Chunking ----
    chunk_size: int = 400                # characters per chunk
    chunk_overlap: int = 80             # overlap to preserve context

    # ---- Retrieval ----
    top_k: int = 5                       # how many chunks to retrieve per query

    # ---- Confidence scoring ----
    # When the best retrieval score is below this, report "no strong evidence"
    min_relevance_threshold: float = 0.25

    # ---- Input / output paths ----
    # Paths to M3's real output files
    m3_pattern_flags_path: str = "M3/output/pattern_flags.json"
    m3_centrality_scores_path: str = "M3/output/centrality_scores.json"

    # Historical FIR documents M1 produced (embedded into the vector store)
    fir_documents_dir: str = "M1/data/firs"

    # M4 output directory
    output_dir: str = "M4/output"

    # ---- Language enforcement ----
    # These words must NEVER appear in generated summaries.
    # Checked as a post-generation safety pass (belt-and-suspenders).
    banned_output_words: tuple[str, ...] = (
        "guilty", "convicted", "criminal", "confirmed criminal",
        "proven", "definitely", "certainly committed", "is the perpetrator",
        "is guilty", "is responsible for the crime",
    )


# Default singleton — import this everywhere in M4
DEFAULT_CONFIG = M4Config()
