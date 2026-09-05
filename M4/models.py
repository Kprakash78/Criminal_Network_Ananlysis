"""
M4 — Shared Data Structures
PS 26152 — AI-Powered Criminal Network Analysis System

RagResult and GeneratedSummary match the project-wide schema in backend.md §4
EXACTLY. Do not change field names without team sign-off — M5 and M6 depend on
these shapes.
"""

from dataclasses import dataclass, field


@dataclass
class RagResult:
    """
    One retrieved historical-case evidence record.
    Schema matches backend.md §4 `RAG RESULT` contract exactly.
    """
    case_id: str           # ID of the NEW case being analysed
    source: str            # source document/chunk ID this was retrieved from
    relevance_score: float # cosine similarity in [0, 1]
    matched_entities: list[str]  # entity IDs that appear in both documents
    evidence: str          # human-readable description of what matched
    timestamp: str | None = None  # optional timestamp from source document


@dataclass
class GeneratedSummary:
    """
    M4's final output.
    Schema matches backend.md §4 `GENERATED SUMMARY` contract exactly.
    """
    case_id: str
    summary_text: str          # investigator-facing natural-language summary
    evidence_used: list[str]   # list of evidence IDs cited in the summary
    confidence: float          # [0, 1] — how grounded/supported this summary is
