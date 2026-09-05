"""
M4.6 — Prompt Builder
PS 26152 — AI-Powered Criminal Network Analysis System

Assembles a structured, grounded prompt from:
  - the new case text (truncated to fit token budget)
  - retrieved historical case chunks (from M4.4)
  - graph + M3 analytics evidence (from M4.5)

The prompt is designed for flan-t5's seq2seq format:
  "Summarize the following ... Given the evidence below, ..."

Every piece of evidence included in the prompt gets a unique citation ID
so the summary generator can reference it explicitly.
"""

import logging

from M4.config import M4Config, DEFAULT_CONFIG
from M4.models import RagResult

logger = logging.getLogger(__name__)

# Maximum characters of new-case text to include in the prompt.
# flan-t5-large has a 512-token encoder limit; ~1000 chars is safe.
MAX_CASE_TEXT_CHARS = 800

# Maximum evidence string length per item (prevent one item consuming budget)
MAX_EVIDENCE_ITEM_CHARS = 300


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "…[truncated]"


def build_prompt(
    case_id: str,
    new_case_text: str,
    retrieved_chunks: list[dict],
    graph_evidence: dict,
    config: M4Config = DEFAULT_CONFIG,
) -> tuple[str, list[str]]:
    """
    Build the full grounded prompt for the LLM.

    Args:
        case_id:          identifier for the new case (e.g. "FIR036")
        new_case_text:    raw text of the new case document
        retrieved_chunks: output of retrieve_relevant_cases() — list of chunk dicts
        graph_evidence:   output of pull_graph_evidence() — evidence bundle dict
        config:           M4Config

    Returns:
        (prompt_text: str, evidence_ids: list[str])
        evidence_ids is a list of citation keys included in the prompt.
    """
    evidence_ids: list[str] = []
    evidence_blocks: list[str] = []

    # ---- Section 1: retrieved historical case chunks ----
    if retrieved_chunks:
        evidence_blocks.append("=== RETRIEVED HISTORICAL CASE EVIDENCE ===")
        for i, chunk in enumerate(retrieved_chunks[:config.top_k]):
            eid = f"RAG_RESULT_{i+1}"
            evidence_ids.append(eid)
            score = chunk.get("relevance_score", 0.0)
            doc_id = chunk.get("doc_id", "unknown")
            text = _truncate(chunk.get("text", ""), MAX_EVIDENCE_ITEM_CHARS)
            evidence_blocks.append(
                f"[{eid}] Source: {doc_id} (relevance: {score:.2f})\n{text}"
            )
    else:
        evidence_blocks.append(
            "=== RETRIEVED HISTORICAL CASE EVIDENCE ===\n"
            "No related historical cases found in the database."
        )

    # ---- Section 2: M3 pattern flags ----
    entity_list = graph_evidence.get("entities", [])
    flagged_entities = [e for e in entity_list if e.get("flags")]

    if flagged_entities:
        evidence_blocks.append("=== ANALYTICS PATTERN FLAGS (M3) ===")
        for entity in flagged_entities:
            eid_str = entity["entity_id"]
            flag_eid = f"PATTERN_FLAG_{eid_str}"
            evidence_ids.append(flag_eid)
            flags = ", ".join(entity["flags"])
            score = entity.get("priority_score", "N/A")
            m3_ev = "; ".join(
                _truncate(ev, MAX_EVIDENCE_ITEM_CHARS)
                for ev in entity.get("m3_evidence", [])[:3]
            )
            evidence_blocks.append(
                f"[{flag_eid}] Entity: {eid_str} | "
                f"Flags: {flags} | Priority: {score}\n"
                f"Evidence: {m3_ev}"
            )

    # ---- Section 3: graph edges ----
    graph_edges = graph_evidence.get("graph_edges", [])
    graph_edge_ids = graph_evidence.get("graph_edge_ids", [])
    if graph_edges:
        evidence_blocks.append("=== GRAPH CONNECTIONS (M2) ===")
        for edge_str, edge_id in zip(graph_edges[:10], graph_edge_ids[:10]):
            evidence_ids.append(edge_id)
            evidence_blocks.append(f"[{edge_id}] {edge_str}")

    # ---- Assemble final prompt ----
    case_snippet = _truncate(new_case_text, MAX_CASE_TEXT_CHARS)
    evidence_text = "\n\n".join(evidence_blocks)

    prompt = (
        f"Case ID: {case_id}\n\n"
        f"NEW CASE DOCUMENT:\n{case_snippet}\n\n"
        f"{evidence_text}\n\n"
        f"TASK: Based ONLY on the evidence above, write a concise investigative "
        f"summary for Case {case_id}. Identify any potential connections to "
        f"historical cases or entities with elevated pattern flags. "
        f"Cite the evidence IDs (e.g. [RAG_RESULT_1], [PATTERN_FLAG_X]) "
        f"when making specific claims. "
        f"Do not state conclusions of guilt or criminality. "
        f"End with: 'All findings require investigator verification before any action is taken.'"
    )

    logger.info(
        f"[PromptBuilder] Built prompt for {case_id}: "
        f"{len(retrieved_chunks)} retrieved chunks, "
        f"{len(flagged_entities)} flagged entities, "
        f"{len(graph_edges)} graph edges, "
        f"prompt length: {len(prompt)} chars"
    )

    return prompt, evidence_ids


def build_rag_results(
    case_id: str,
    retrieved_chunks: list[dict],
    graph_evidence: dict,
    config: M4Config = DEFAULT_CONFIG,
) -> list[RagResult]:
    """
    Convert retrieved chunks and graph evidence into RagResult objects
    matching the shared schema (backend.md §4).
    """
    results: list[RagResult] = []

    # One RagResult per retrieved chunk
    for chunk in retrieved_chunks:
        score = chunk.get("relevance_score", 0.0)
        if score < config.min_relevance_threshold:
            continue  # skip very low-relevance results

        results.append(RagResult(
            case_id=case_id,
            source=chunk.get("doc_id", "unknown"),
            relevance_score=round(score, 4),
            matched_entities=[],  # entity matching is in evidence_puller
            evidence=_truncate(chunk.get("text", ""), 200),
            timestamp=None,
        ))

    # One RagResult per flagged entity (from M3)
    for entity in graph_evidence.get("entities", []):
        if not entity.get("flags"):
            continue
        m3_ev = entity.get("m3_evidence", [])
        evidence_str = _truncate(
            " | ".join(m3_ev[:2]) if m3_ev else "Pattern flags detected",
            200,
        )
        results.append(RagResult(
            case_id=case_id,
            source=f"M3_FLAGS_{entity['entity_id']}",
            relevance_score=float(entity.get("priority_score") or 0.0),
            matched_entities=[entity["entity_id"]],
            evidence=evidence_str,
            timestamp=None,
        ))

    return results
