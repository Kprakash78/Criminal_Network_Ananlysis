"""
M4.1 — Local LLM Setup and Inference
PS 26152 — AI-Powered Criminal Network Analysis System

Wraps a locally-cached HuggingFace seq2seq model (flan-t5-large by default)
for fully-local inference with zero external network calls.

DATA PRIVACY GUARANTEE:
  These three environment variables are set at import time. They instruct the
  HuggingFace library to refuse any network connection — the library will raise
  an error rather than send data outside. This means:
    - Model weights are loaded from the local disk cache ONLY
    - Your case text / FIR data is never transmitted anywhere
    - No telemetry, no version-check, no download attempt during inference
  The one-time model download (done once before deployment) is the ONLY moment
  the library touches the internet, and only to download the model — not your data.

Design decisions:
  - Seq2seq (encoder-decoder) models are used rather than causal LMs because
    they fit in CPU RAM at a useful quality tier (~770 MB for flan-t5-large),
    handle instruction-following prompts well, and generate in ~3s on CPU.
  - The model is loaded lazily (on first call) to avoid startup cost when
    M4 is imported but not yet exercised.
  - A model priority list is tried in order; the first one that loads without
    OOM is used. This makes the code portable to better hardware without
    config changes.
  - `local_files_only=True` on every load call enforces the "no network"
    requirement at the library level as a second layer of protection.

Language enforcement:
  - A system instruction baked into every prompt tells the model it MUST
    phrase all findings as investigative leads requiring verification.
  - Post-generation, a banned-word scan provides belt-and-suspenders safety.
"""

import logging
import os
from typing import Any

# ---------------------------------------------------------------------------
# PRIVACY: Force HuggingFace library into fully offline mode.
# These must be set BEFORE any HuggingFace import to take effect.
# With these set, the library will raise OfflineModeIsEnabled rather than
# attempt any network call — ensuring case data stays on the local machine.
# ---------------------------------------------------------------------------
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from M4.config import M4Config, DEFAULT_CONFIG

logger = logging.getLogger(__name__)

# Module-level lazy cache so the model loads once per process
_llm_cache: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# System prompt — baked into every call to the LLM
# ---------------------------------------------------------------------------

SYSTEM_INSTRUCTION = (
    "You are an investigative analysis assistant helping law enforcement officers "
    "review case connections. Your role is to summarize factual links between cases "
    "based ONLY on the evidence provided to you. "
    "\n\n"
    "STRICT RULES you must follow without exception:\n"
    "1. Only state facts that are directly supported by the evidence provided. "
    "Do not infer, assume, or add information not present in the evidence.\n"
    "2. Never state or imply that any person is guilty, criminal, or responsible "
    "for any crime. Use only: 'potential connection', 'requires investigator "
    "verification', 'pattern flagged for review', 'elevated priority'.\n"
    "3. If the evidence is weak or absent, say explicitly: "
    "'No strong evidence found linking this case to prior records.'\n"
    "4. End every summary with: 'All findings require investigator verification "
    "before any action is taken.'\n"
    "5. Keep summaries factual, concise, and under 300 words."
)


def _load_model(config: M4Config = DEFAULT_CONFIG) -> tuple[Any, Any]:
    """
    Load the highest-priority locally-cached LLM that fits in memory.

    Returns:
        (model, tokenizer) tuple.

    Raises:
        RuntimeError: if no model in the candidate list can be loaded locally.
                      Does NOT silently fall back to a network call.
    """
    if "model" in _llm_cache:
        return _llm_cache["model"], _llm_cache["tokenizer"]

    from transformers import T5ForConditionalGeneration, AutoTokenizer

    # --- RAM guard: flan-t5-large needs ~3.1GB. Skip if insufficient RAM. ---
    try:
        import psutil
        avail_gb = psutil.virtual_memory().available / (1024 ** 3)
        if avail_gb < 4.0:
            raise RuntimeError(
                f"[LLM] Insufficient RAM to load model safely "
                f"({avail_gb:.1f} GB free, need ≥ 4 GB). "
                f"Dashboard will use M5 built-in mocks for this session."
            )
        logger.info(f"[LLM] RAM available: {avail_gb:.1f} GB — proceeding with model load")
    except ImportError:
        # psutil not installed — skip check, attempt load anyway
        logger.warning("[LLM] psutil not available — skipping RAM check")

    last_error: Exception | None = None

    for model_id in config.llm_model_candidates:
        try:
            logger.info(f"[LLM] Attempting to load '{model_id}' (local_files_only=True)")
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                local_files_only=True,
            )
            model = T5ForConditionalGeneration.from_pretrained(
                model_id,
                local_files_only=True,
                low_cpu_mem_usage=True,
            )
            model.eval()
            _llm_cache["model"] = model
            _llm_cache["tokenizer"] = tokenizer
            _llm_cache["model_id"] = model_id
            logger.info(f"[LLM] Loaded '{model_id}' successfully")
            return model, tokenizer
        except MemoryError as e:
            logger.warning(f"[LLM] OOM loading '{model_id}': {e}")
            last_error = e
            continue
        except Exception as e:
            logger.warning(f"[LLM] Could not load '{model_id}': {e}")
            last_error = e
            continue

    raise RuntimeError(
        f"[LLM] No local model could be loaded from the candidate list "
        f"{config.llm_model_candidates}. "
        f"Last error: {last_error}. "
        f"Ensure at least one model is cached locally (huggingface-cli download <model_id>). "
        f"Do NOT add external API credentials — this system must run fully locally."
    )


def get_loaded_model_id() -> str | None:
    """Return the model_id that is currently loaded, or None if not yet loaded."""
    return _llm_cache.get("model_id")


def generate_text(
    prompt: str,
    config: M4Config = DEFAULT_CONFIG,
    extra_context: str = "",
) -> str:
    """
    Generate text from a prompt using the local LLM.

    The system instruction is prepended to every prompt to enforce
    language and grounding constraints at the model level.

    Args:
        prompt:        the query / instruction portion of the prompt
        config:        M4Config
        extra_context: additional grounding context (evidence, retrieved chunks)
                       pre-joined and passed separately to keep the interface clean

    Returns:
        Generated text string (stripped of whitespace).

    Raises:
        RuntimeError: if the model cannot be loaded locally.
    """
    model, tokenizer = _load_model(config)

    # flan-t5 format: the full instruction goes in the input sequence
    full_input = f"{SYSTEM_INSTRUCTION}\n\n{prompt}"
    if extra_context:
        full_input = f"{SYSTEM_INSTRUCTION}\n\nContext:\n{extra_context}\n\n{prompt}"

    import torch
    inputs = tokenizer(
        full_input,
        return_tensors="pt",
        max_length=1024,
        truncation=True,
    )

    with torch.no_grad():
        if config.llm_do_sample and config.llm_temperature > 0:
            output_ids = model.generate(
                **inputs,
                max_new_tokens=config.llm_max_new_tokens,
                do_sample=True,
                temperature=config.llm_temperature,
            )
        else:
            output_ids = model.generate(
                **inputs,
                max_new_tokens=config.llm_max_new_tokens,
                do_sample=False,
            )

    result = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return result.strip()


def check_banned_words(text: str, config: M4Config = DEFAULT_CONFIG) -> list[str]:
    """
    Return a list of any banned words found in the generated text.
    Empty list means the text is clean.
    Used as post-generation safety check (belt-and-suspenders).
    """
    found = []
    text_lower = text.lower()
    for word in config.banned_output_words:
        if word.lower() in text_lower:
            found.append(word)
    return found
