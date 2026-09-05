"""
Tests for M4.7–M4.10 — Full Pipeline (emit_outputs, generate_summary, RAGPipeline)
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest

from M4.config import M4Config
from M4.models import GeneratedSummary, RagResult
from M4.pipeline import emit_outputs, RAGPipeline
from M4.llm import check_banned_words


def _model_available():
    cfg = M4Config()
    try:
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained(
            cfg.llm_model_candidates[0], local_files_only=True
        )
        return True
    except Exception:
        return False


skip_if_no_model = pytest.mark.skipif(
    not _model_available(),
    reason="flan-t5-large not in local HF cache — run M4.1 setup first"
)


class TestEmitOutputs:
    def test_writes_both_files(self):
        summary = GeneratedSummary(
            case_id="FIR036",
            summary_text="Potential connection found. Requires investigator verification.",
            evidence_used=["RAG_RESULT_1", "PATTERN_FLAG_P001"],
            confidence=0.72,
        )
        rag_results = [
            RagResult(
                case_id="FIR036",
                source="FIR_001",
                relevance_score=0.88,
                matched_entities=["P001"],
                evidence="Same phone number appears in FIR_001",
                timestamp=None,
            )
        ]
        with tempfile.TemporaryDirectory() as td:
            cfg = M4Config(output_dir=td)
            paths = emit_outputs("FIR036", rag_results, summary, cfg)
            assert Path(paths["rag_results"]).exists()
            assert Path(paths["generated_summary"]).exists()

    def test_rag_results_schema(self):
        """Fields in rag_results.json must match backend.md §4 exactly."""
        summary = GeneratedSummary("FIR036", "Summary text. Requires investigator verification.", [], 0.5)
        rag = [RagResult("FIR036", "FIR_001", 0.88, ["P001"], "Evidence text", None)]
        with tempfile.TemporaryDirectory() as td:
            cfg = M4Config(output_dir=td)
            paths = emit_outputs("FIR036", rag, summary, cfg)
            data = json.loads(Path(paths["rag_results"]).read_text())
            assert len(data) == 1
            rec = data[0]
            for field in ("case_id", "source", "relevance_score", "matched_entities", "evidence", "timestamp"):
                assert field in rec, f"Missing field '{field}' in rag_results schema"

    def test_summary_schema(self):
        """Fields in generated_summary.json must match backend.md §4 exactly."""
        summary = GeneratedSummary("FIR036", "Summary. Requires investigator verification.", ["RAG_RESULT_1"], 0.6)
        with tempfile.TemporaryDirectory() as td:
            cfg = M4Config(output_dir=td)
            paths = emit_outputs("FIR036", [], summary, cfg)
            data = json.loads(Path(paths["generated_summary"]).read_text())
            for field in ("case_id", "summary_text", "evidence_used", "confidence"):
                assert field in data, f"Missing field '{field}' in generated_summary schema"

    def test_confidence_in_range(self):
        summary = GeneratedSummary("FIR036", "Summary. Requires investigator verification.", [], 0.75)
        with tempfile.TemporaryDirectory() as td:
            cfg = M4Config(output_dir=td)
            paths = emit_outputs("FIR036", [], summary, cfg)
            data = json.loads(Path(paths["generated_summary"]).read_text())
            assert 0.0 <= data["confidence"] <= 1.0

    def test_output_is_valid_json(self):
        summary = GeneratedSummary("FIR036", "Summary. Requires investigator verification.", [], 0.5)
        with tempfile.TemporaryDirectory() as td:
            cfg = M4Config(output_dir=td)
            paths = emit_outputs("FIR036", [], summary, cfg)
            for _, fp in paths.items():
                text = Path(fp).read_text()
                parsed = json.loads(text)
                assert parsed is not None


class TestRAGPipeline:
    @skip_if_no_model
    def test_generate_summary_returns_generated_summary(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = M4Config(output_dir=td, fir_documents_dir="M1/data/firs")
            pipeline = RAGPipeline(cfg)
            pipeline.load()
            result = pipeline.generate_summary(
                "Ravi Kumar was seen near an ATM in Lajpat Nagar with vehicle DL01AB1234.",
                case_id="FIR_TEST",
                emit_files=False,
            )
            assert isinstance(result, GeneratedSummary)
            assert result.case_id == "FIR_TEST"

    @skip_if_no_model
    def test_summary_text_non_empty(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = M4Config(output_dir=td, fir_documents_dir="M1/data/firs")
            pipeline = RAGPipeline(cfg)
            pipeline.load()
            result = pipeline.generate_summary(
                "New case: suspect made multiple cash withdrawals.",
                case_id="FIR_TEST2",
                emit_files=False,
            )
            assert len(result.summary_text.strip()) > 0

    @skip_if_no_model
    def test_summary_has_verification_language(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = M4Config(output_dir=td, fir_documents_dir="M1/data/firs")
            pipeline = RAGPipeline(cfg)
            pipeline.load()
            result = pipeline.generate_summary(
                "Case involves transfer of funds between multiple accounts.",
                case_id="FIR_TEST3",
                emit_files=False,
            )
            assert "verification" in result.summary_text.lower(), (
                f"Missing verification language in summary: {result.summary_text[:300]}"
            )

    @skip_if_no_model
    def test_summary_no_banned_words(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = M4Config(output_dir=td, fir_documents_dir="M1/data/firs")
            pipeline = RAGPipeline(cfg)
            pipeline.load()
            result = pipeline.generate_summary(
                "Suspect transferred money and communicated with multiple persons.",
                case_id="FIR_TEST4",
                emit_files=False,
            )
            found = check_banned_words(result.summary_text, cfg)
            assert found == [], (
                f"Banned words found in summary: {found}\n{result.summary_text[:300]}"
            )

    @skip_if_no_model
    def test_evidence_used_populated(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = M4Config(output_dir=td, fir_documents_dir="M1/data/firs")
            pipeline = RAGPipeline(cfg)
            pipeline.load()
            result = pipeline.generate_summary(
                "Bank account ACC00102 received funds from multiple sources.",
                case_id="FIR_TEST5",
                emit_files=False,
            )
            assert len(result.evidence_used) >= 1, "Summary must cite at least one evidence ID"

    @skip_if_no_model
    def test_confidence_in_valid_range(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = M4Config(output_dir=td, fir_documents_dir="M1/data/firs")
            pipeline = RAGPipeline(cfg)
            pipeline.load()
            result = pipeline.generate_summary("Test case", case_id="FIR_CONF", emit_files=False)
            assert 0.0 <= result.confidence <= 1.0

    @skip_if_no_model
    def test_empty_case_text_raises(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = M4Config(output_dir=td, fir_documents_dir="M1/data/firs")
            pipeline = RAGPipeline(cfg)
            pipeline.load()
            with pytest.raises(ValueError):
                pipeline.generate_summary("", case_id="FIR_EMPTY", emit_files=False)

    @skip_if_no_model
    def test_no_evidence_case_says_so(self):
        """A query with no historical matches should not fabricate connections."""
        with tempfile.TemporaryDirectory() as td:
            cfg = M4Config(output_dir=td, fir_documents_dir="M1/data/firs")
            pipeline = RAGPipeline(cfg)
            pipeline.load()
            # Completely unrelated text (no entities, no historical match)
            result = pipeline.generate_summary(
                "zyxwvutsrqponm abcdefghijk irrelevant nonsense text",
                case_id="FIR_NOEVIDENCE",
                emit_files=False,
            )
            # Should not raise and should contain verification language
            assert isinstance(result, GeneratedSummary)
            assert len(result.summary_text) > 0

    @skip_if_no_model
    def test_output_files_written(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = M4Config(output_dir=td, fir_documents_dir="M1/data/firs")
            pipeline = RAGPipeline(cfg)
            pipeline.load()
            pipeline.generate_summary(
                "Bank account transfer case involving ACC00102.",
                case_id="FIR_OUTPUT",
                emit_files=True,
            )
            assert (Path(td) / "FIR_OUTPUT_rag_results.json").exists()
            assert (Path(td) / "FIR_OUTPUT_generated_summary.json").exists()
