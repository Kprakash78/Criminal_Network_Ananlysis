"""
Tests for M4.1 — Local LLM (flan-t5-large)

Note: these tests require flan-t5-large to be cached locally.
They will be skipped if the model isn't available.
The M4.1 model-download script must have run first.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest

from M4.config import M4Config
from M4.llm import generate_text, check_banned_words, get_loaded_model_id, SYSTEM_INSTRUCTION

cfg = M4Config()


def _model_available():
    try:
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained(
            cfg.llm_model_candidates[0],
            local_files_only=True,
        )
        return True
    except Exception:
        return False


skip_if_no_model = pytest.mark.skipif(
    not _model_available(),
    reason="flan-t5-large not in local HF cache — run M4.1 setup first"
)


class TestCheckBannedWords:
    """These tests don't require the model."""

    def test_finds_banned_word(self):
        found = check_banned_words("The suspect is guilty of the crime.", cfg)
        assert "guilty" in found

    def test_clean_text_returns_empty(self):
        found = check_banned_words("This requires investigator verification.", cfg)
        assert found == []

    def test_case_insensitive(self):
        found = check_banned_words("The person is GUILTY.", cfg)
        assert "guilty" in found

    def test_multiple_banned_words(self):
        text = "He is guilty and convicted."
        found = check_banned_words(text, cfg)
        assert len(found) >= 2


class TestGenerateText:
    @skip_if_no_model
    def test_generates_non_empty_output(self):
        result = generate_text(
            "Summarize this case connection in one sentence: "
            "Person A and Person B used the same phone number.",
            cfg
        )
        assert isinstance(result, str)
        assert len(result.strip()) > 0

    @skip_if_no_model
    def test_runs_locally_no_network(self):
        """Confirm local_files_only=True doesn't throw (model IS cached)."""
        result = generate_text(
            "What is the capital of India?",
            cfg
        )
        assert isinstance(result, str)
        # Don't assert content — just that it ran without hitting network

    @skip_if_no_model
    def test_model_id_recorded(self):
        generate_text("test", cfg)
        model_id = get_loaded_model_id()
        assert model_id is not None
        assert "flan-t5" in model_id.lower()

    @skip_if_no_model
    def test_output_is_string(self):
        result = generate_text("List three colours.", cfg)
        assert isinstance(result, str)

    @skip_if_no_model
    def test_deterministic_with_do_sample_false(self):
        """With do_sample=False, two identical prompts should produce identical output."""
        prompt = "Summarize: entity X called entity Y 10 times in 2 hours."
        cfg_det = M4Config(llm_do_sample=False)
        r1 = generate_text(prompt, cfg_det)
        r2 = generate_text(prompt, cfg_det)
        assert r1 == r2, "Greedy decoding must be deterministic"

    @skip_if_no_model
    def test_no_banned_words_in_investigation_prompt(self):
        """Model should respect system instruction and avoid guilt language."""
        prompt = (
            "Based on the following evidence, write a summary: "
            "Entity P001 has 5 pattern flags and high centrality. "
            "It is connected to Entity P002 via financial transactions."
        )
        result = generate_text(prompt, cfg)
        banned = check_banned_words(result, cfg)
        assert banned == [], (
            f"Model produced banned words: {banned}\nOutput: {result[:300]}"
        )
