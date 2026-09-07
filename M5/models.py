"""
M5 — Core Data Models
PS 26152 — AI-Powered Criminal Network Analysis System

Defines the SESSION STATE and FINAL RESPONSE schemas from backend.md §4.
These shapes are fixed — M6 depends on FinalResponse being exact.
Do NOT change field names without team sign-off.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Request types — what M6 sends in
# ---------------------------------------------------------------------------

@dataclass
class NewCaseUpload:
    """Investigator uploaded a new case document for analysis."""
    case_id: str           # e.g. "FIR103"
    case_text: str         # raw FIR text content
    source_dir: str = ""   # optional: path to full source dir for M1.run_pipeline()


@dataclass
class FollowUpQuestion:
    """Investigator asked a follow-up question in an existing session."""
    question: str          # free-text investigator question


# ---------------------------------------------------------------------------
# SESSION STATE — internal LangGraph state, persisted across turns
# Matches backend.md §4 SESSION STATE schema exactly.
# ---------------------------------------------------------------------------

@dataclass
class SessionState:
    session_id: str
    current_case_id: str = ""
    current_case_text: str = ""
    conversation_history: list[dict] = field(default_factory=list)
    entities_in_context: list[str] = field(default_factory=list)  # entity_ids
    last_evidence: list[str] = field(default_factory=list)         # evidence IDs

    # Internal workflow carry-forward (not in the backend.md schema, but needed
    # to pass intermediate results between graph nodes without extra storage).
    _m1_entities: list[dict] = field(default_factory=list)
    _m2_neighbors: list[dict] = field(default_factory=list)
    _m3_pattern_flags: list[dict] = field(default_factory=list)
    _m4_summary: dict = field(default_factory=dict)
    _confidence: float = 0.0
    _requires_human_review: bool = False
    _error_message: str = ""

    # Guardrails — count of tool calls so far this request
    _tool_call_count: int = 0


# ---------------------------------------------------------------------------
# FINAL RESPONSE — what M5 returns to M6
# Matches backend.md §4 FINAL RESPONSE schema exactly.
# ---------------------------------------------------------------------------

@dataclass
class FinalResponse:
    session_id: str
    response_text: str
    evidence: list[str]
    confidence: float
    requires_human_review: bool
    # Entities extracted from the uploaded document (Bug 2 fix — populated from M1 extraction)
    # Each dict: {entity_id, name, type, confidence}
    extracted_entities: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = {
            "session_id": self.session_id,
            "response_text": self.response_text,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "requires_human_review": self.requires_human_review,
        }
        if self.extracted_entities:
            data["extracted_entities"] = self.extracted_entities
        return data


# ---------------------------------------------------------------------------
# Routing decision (used internally by the confidence router node)
# ---------------------------------------------------------------------------

@dataclass
class RoutingDecision:
    route: str              # "continue" | "human_review"
    reason: str
    confidence: float
    threshold_used: float
