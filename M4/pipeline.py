"""
M4.7–M4.10 — Full RAG Pipeline + Output Emitter
PS 26152 — AI-Powered Criminal Network Analysis System

Public API:
    generate_summary(new_case_text, case_id, ...)  → GeneratedSummary
    RAGPipeline class — for pre-loading the index once, then reusing it

This is what M5 calls. M5 should instantiate RAGPipeline once at startup
(which loads the model and index), then call pipeline.generate_summary()
per case query to avoid reloading on every call.

Output files (M4.9):
    M4/output/rag_results.json        — list of RagResult records
    M4/output/generated_summary.json  — the GeneratedSummary record

Both files match the shared schemas in backend.md §4 exactly.
"""

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import networkx as nx

from M4.config import M4Config, DEFAULT_CONFIG
from M4.models import GeneratedSummary, RagResult
from M4.embedder import embed_documents, chunk_text
from M4.vector_store import VectorIndex, build_vector_index, retrieve_relevant_cases
from M4.evidence_puller import (
    load_pattern_flags,
    pull_graph_evidence,
    pull_evidence_for_new_case,
)
from M4.prompt_builder import build_prompt, build_rag_results
from M4.llm import generate_text, check_banned_words, SYSTEM_INSTRUCTION

logger = logging.getLogger(__name__)

_INDEX_SUBDIR = "vector_index"


# ---------------------------------------------------------------------------
# Confidence calculation
# ---------------------------------------------------------------------------

def _compute_confidence(
    rag_results: list[RagResult],
    graph_evidence: dict,
    banned_words_found: list[str],
) -> float:
    """
    Derive a confidence score for the generated summary from signal strength.

    Formula:
      - Start at 0
      - Best retrieval similarity contributes up to 0.4
      - Flagged entity with priority_score contributes up to 0.4
      - Penalty of 0.2 per banned word found (capped at 0.4 penalty)
    Clamped to [0, 1].
    """
    score = 0.0

    # Retrieval quality
    retrieval_scores = [r.relevance_score for r in rag_results if not r.source.startswith("M3_FLAGS_")]
    if retrieval_scores:
        score += min(max(retrieval_scores), 1.0) * 0.4

    # M3 flag quality
    flag_scores = [
        r.relevance_score for r in rag_results
        if r.source.startswith("M3_FLAGS_")
    ]
    if flag_scores:
        score += min(max(flag_scores), 1.0) * 0.4

    # Penalty for banned words in output
    penalty = min(len(banned_words_found) * 0.2, 0.4)
    score -= penalty

    return round(max(0.0, min(score, 1.0)), 4)


# ---------------------------------------------------------------------------
# RAGPipeline class — persistent across calls
# ---------------------------------------------------------------------------

class RAGPipeline:
    """
    Stateful RAG pipeline that loads resources once and handles multiple queries.

    Usage (from M5):
        pipeline = RAGPipeline()
        pipeline.load()            # loads LLM, index; call once at startup
        summary = pipeline.generate_summary(text, case_id="FIR036")
    """

    def __init__(self, config: M4Config = DEFAULT_CONFIG):
        self.config = config
        self._index: VectorIndex | None = None
        self._flags_by_entity: dict | None = None
        self._graph: nx.MultiDiGraph | None = None
        self._historical_docs: list[tuple[str, str]] = []  # (doc_id, text)

    def load(
        self,
        graph: nx.MultiDiGraph | None = None,
        rebuild_index: bool = False,
    ) -> None:
        """
        Load the LLM, embedding model, vector index, and M3 flags.

        Args:
            graph:         M2's graph (optional; enhances evidence quality)
            rebuild_index: if True, rebuild the FAISS index even if a saved
                           one exists. Use after adding new historical docs.
        """
        # Pre-load the LLM (validates it's available locally)
        from M4.llm import _load_model
        _load_model(self.config)

        # Load M3 flags
        self._flags_by_entity = load_pattern_flags(self.config.m3_pattern_flags_path)
        logger.info(f"[Pipeline] Loaded {len(self._flags_by_entity)} M3 flag records")

        # Store graph reference
        self._graph = graph

        # Load or build the FAISS index
        index_dir = str(Path(self.config.output_dir) / _INDEX_SUBDIR)
        index_path = Path(index_dir) / "faiss.index"

        if index_path.exists() and not rebuild_index:
            self._index = VectorIndex.load(index_dir, self.config)
            logger.info(f"[Pipeline] Loaded existing vector index ({self._index.size} chunks)")
        else:
            self._index = self._build_index_from_fir_docs()

        logger.info("[Pipeline] Ready")

    def _build_index_from_fir_docs(self) -> VectorIndex:
        """Read all FIR .txt files and build the FAISS index."""
        import re

        fir_dir = Path(self.config.fir_documents_dir)
        doc_texts: list[str] = []
        doc_ids: list[str] = []

        if fir_dir.exists():
            for fir_file in sorted(fir_dir.glob("*.txt")):
                text = fir_file.read_text(encoding="utf-8", errors="replace").strip()
                if text:
                    # Extract the true case number from the file content
                    # (filename is FIR_001.txt but content says "Case No.: FIR109")
                    match = re.search(r"Case No\.:\s*(FIR\d+)", text)
                    doc_id = match.group(1) if match else fir_file.stem
                    doc_texts.append(text)
                    doc_ids.append(doc_id)
            logger.info(f"[Pipeline] Found {len(doc_texts)} FIR documents to index")
        else:
            logger.warning(f"[Pipeline] FIR directory not found: {fir_dir}")

        if not doc_texts:
            # Build a minimal single-doc index so the pipeline doesn't crash
            doc_texts = ["No historical case documents available."]
            doc_ids = ["empty"]

        self._historical_docs = list(zip(doc_ids, doc_texts))

        index = build_vector_index(doc_texts, doc_ids, self.config)

        # Persist for future runs
        index_dir = str(Path(self.config.output_dir) / _INDEX_SUBDIR)
        index.save(index_dir)

        return index

    def generate_summary(
        self,
        new_case_text: str,
        case_id: str = "UNKNOWN",
        entity_ids: list[str] | None = None,
        emit_files: bool = True,
    ) -> GeneratedSummary:
        """
        Run the full RAG pipeline for a new case document.

        Args:
            new_case_text: raw text of the new case (e.g. a new FIR)
            case_id:       identifier for this case (e.g. "FIR036")
            entity_ids:    explicit entity IDs to pull evidence for
                           (if None, extracted automatically from the text)
            emit_files:    if True, write rag_results.json and
                           generated_summary.json to M4/output/

        Returns:
            GeneratedSummary matching the shared schema in backend.md §4.

        Raises:
            ValueError: if new_case_text is empty or whitespace-only.
            RuntimeError: if the LLM cannot be loaded locally.
        """
        if not new_case_text or not new_case_text.strip():
            raise ValueError(
                f"[Pipeline] Cannot generate summary for {case_id}: "
                "new_case_text is empty. Provide the full case document text."
            )

        if self._index is None:
            raise RuntimeError(
                "[Pipeline] Call pipeline.load() before generate_summary()."
            )

        t_start = time.perf_counter()
        logger.info(f"[Pipeline] Generating summary for {case_id}")

        # M4.4 — Retrieve relevant historical cases
        retrieved_chunks = retrieve_relevant_cases(
            new_case_text,
            self._index,
            self.config,
            query_doc_id=case_id,  # avoid self-match if new case is already indexed
        )
        logger.info(f"[Pipeline] Retrieved {len(retrieved_chunks)} chunks")

        # M4.5 — Pull graph + M3 evidence
        if entity_ids is not None:
            graph_evidence = pull_graph_evidence(
                entity_ids,
                graph=self._graph,
                flags_by_entity=self._flags_by_entity,
                config=self.config,
            )
        else:
            graph_evidence = pull_evidence_for_new_case(
                new_case_text,
                graph=self._graph,
                config=self.config,
            )

        # M4.6 — Build prompt and collect evidence IDs
        prompt, evidence_ids = build_prompt(
            case_id, new_case_text, retrieved_chunks, graph_evidence, self.config
        )

        # Build RagResult objects (for output schema)
        rag_results = build_rag_results(
            case_id, retrieved_chunks, graph_evidence, self.config
        )

        # M4.7 — LLM summary generation
        t_gen = time.perf_counter()
        raw_summary = generate_text(prompt, self.config)
        gen_elapsed = time.perf_counter() - t_gen
        logger.info(f"[Pipeline] LLM generation: {gen_elapsed:.1f}s")

        # M4.8 — Post-generation language enforcement
        banned_found = check_banned_words(raw_summary, self.config)
        if banned_found:
            logger.warning(
                f"[Pipeline] BANNED WORDS in generated summary for {case_id}: "
                f"{banned_found}. Appending override disclaimer."
            )
            raw_summary += (
                "\n\n[SYSTEM NOTE: Some language in the above text may require "
                "review. All findings require investigator verification before "
                "any action is taken.]"
            )

        # Ensure the mandatory closing line is always present
        if "requires investigator verification" not in raw_summary.lower():
            raw_summary += (
                "\n\nAll findings require investigator verification before "
                "any action is taken."
            )

        # Confidence score
        confidence = _compute_confidence(rag_results, graph_evidence, banned_found)

        # M4.9 — Assemble GeneratedSummary (matching schema exactly)
        summary = GeneratedSummary(
            case_id=case_id,
            summary_text=raw_summary,
            evidence_used=evidence_ids,
            confidence=confidence,
        )

        total_elapsed = time.perf_counter() - t_start
        logger.info(
            f"[Pipeline] Done for {case_id}: "
            f"confidence={confidence:.3f}, "
            f"total_elapsed={total_elapsed:.1f}s, "
            f"evidence_count={len(evidence_ids)}"
        )

        # Emit output files
        if emit_files:
            emit_outputs(case_id, rag_results, summary, self.config)

        return summary


# ---------------------------------------------------------------------------
# M4.9 — Output emitter
# ---------------------------------------------------------------------------

def emit_outputs(
    case_id: str,
    rag_results: list[RagResult],
    summary: GeneratedSummary,
    config: M4Config = DEFAULT_CONFIG,
) -> dict[str, str]:
    """
    Write rag_results.json and generated_summary.json to the output directory.

    Returns: dict mapping logical name → absolute file path.
    """
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # rag_results.json
    rag_path = out / f"{case_id}_rag_results.json"
    rag_data = [
        {
            "case_id": r.case_id,
            "source": r.source,
            "relevance_score": r.relevance_score,
            "matched_entities": r.matched_entities,
            "evidence": r.evidence,
            "timestamp": r.timestamp,
        }
        for r in rag_results
    ]
    rag_path.write_text(
        json.dumps(rag_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # generated_summary.json
    summary_path = out / f"{case_id}_generated_summary.json"
    summary_data = {
        "case_id": summary.case_id,
        "summary_text": summary.summary_text,
        "evidence_used": summary.evidence_used,
        "confidence": summary.confidence,
    }
    summary_path.write_text(
        json.dumps(summary_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    logger.info(
        f"[Pipeline] Wrote output files: {rag_path.name}, {summary_path.name}"
    )
    return {
        "rag_results": str(rag_path.resolve()),
        "generated_summary": str(summary_path.resolve()),
    }


# ---------------------------------------------------------------------------
# Module-level convenience function (for M5 to call directly)
# ---------------------------------------------------------------------------

# Module-level singleton — loaded lazily on first call to generate_summary()
_default_pipeline: RAGPipeline | None = None


def generate_summary(
    new_case_text: str,
    case_id: str = "UNKNOWN",
    graph: nx.MultiDiGraph | None = None,
    entity_ids: list[str] | None = None,
    config: M4Config = DEFAULT_CONFIG,
    emit_files: bool = True,
) -> GeneratedSummary:
    """
    Convenience function: load the pipeline lazily and generate a summary.

    M5 may call this directly without managing a RAGPipeline instance.
    The first call takes longer (model + index load); subsequent calls reuse
    cached state.

    Args:
        new_case_text: raw text of the new case document
        case_id:       identifier (e.g. "FIR036")
        graph:         M2's graph (optional; improves evidence quality)
        entity_ids:    explicit entity IDs (optional; auto-extracted if None)
        config:        M4Config
        emit_files:    write rag_results.json + generated_summary.json

    Returns:
        GeneratedSummary matching backend.md §4 schema.
    """
    global _default_pipeline

    if _default_pipeline is None:
        _default_pipeline = RAGPipeline(config)
        _default_pipeline.load(graph=graph)
    elif graph is not None and _default_pipeline._graph is None:
        # Graph was provided after first call — update the reference
        _default_pipeline._graph = graph

    return _default_pipeline.generate_summary(
        new_case_text,
        case_id=case_id,
        entity_ids=entity_ids,
        emit_files=emit_files,
    )
