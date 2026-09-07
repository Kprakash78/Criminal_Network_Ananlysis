"""
M5 LLM/RAG — Schema definitions
Pydantic models for structured query parsing output (PRD §4) and the
final answer returned to M6/frontend (PRD §11).

These are the ONLY schemas M5 defines — segment records come in via
COMMON_DATA_CONTRACT.md shape from M4 and are read as plain dicts.
"""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Query-side schema  (PRD §4 / BACKEND.md §3)
# ─────────────────────────────────────────────────────────────────────────────

class AttributedObject(BaseModel):
    """An object with optional color/attribute bindings (e.g. 'blue motorcycle')."""
    object: str = Field(description="The object class name (e.g. 'motorcycle', 'person')")
    attributes: list[str] = Field(
        default_factory=list,
        description="Visual attributes bound to THIS object (e.g. ['blue', 'large'])"
    )

    def __str__(self) -> str:
        if self.attributes:
            return f"{self.object}({', '.join(self.attributes)})"
        return self.object


class StructuredQuery(BaseModel):
    """
    Structured representation of a natural-language video search query.
    All lists are empty when nothing is stated or clearly implied.
    Matches BACKEND.md §3 exactly.
    """
    objects: list[AttributedObject] = Field(
        default_factory=list,
        description="Physical objects in the query, each with bound attributes"
    )
    actions: list[str] = Field(
        default_factory=list,
        description="Verbs / activities described (e.g. 'repairing', 'cooking')"
    )
    persons: list[str] = Field(
        default_factory=list,
        description="Person descriptors mentioned (e.g. 'chef', 'person', 'woman')"
    )
    speech_intent: list[str] = Field(
        default_factory=list,
        description="What is being said / explained on-screen (e.g. 'explaining repair steps')"
    )
    context: list[str] = Field(
        default_factory=list,
        description="Scene/setting context (e.g. 'workshop', 'kitchen')"
    )
    raw_query: str = Field(
        default="",
        description="The original unmodified query string"
    )

    def is_empty(self) -> bool:
        """True when no meaningful structure was extracted."""
        return (
            not self.objects
            and not self.actions
            and not self.persons
            and not self.speech_intent
            and not self.context
        )

    def summary(self) -> str:
        """Human-readable one-liner of the parsed query."""
        parts: list[str] = []
        if self.objects:
            parts.append("objects=" + str([str(o) for o in self.objects]))
        if self.actions:
            parts.append("actions=" + str(self.actions))
        if self.persons:
            parts.append("persons=" + str(self.persons))
        if self.speech_intent:
            parts.append("speech=" + str(self.speech_intent))
        if self.context:
            parts.append("context=" + str(self.context))
        return "StructuredQuery(" + ", ".join(parts) + ")"


# ─────────────────────────────────────────────────────────────────────────────
# Answer-side schema  (PRD §11 — output to M6/frontend)
# ─────────────────────────────────────────────────────────────────────────────

class CitedSegment(BaseModel):
    """A retrieved segment cited in the answer, with its key metadata."""
    video_id: str
    segment_id: str
    start_ts: float
    end_ts: float
    final_score: float | None = None   # M4-populated score, None if not available

    def timestamp_range(self) -> str:
        """Returns 'MM:SS-MM:SS' formatted timestamp range."""
        def fmt(t: float) -> str:
            m, s = divmod(int(t), 60)
            return f"{m:02d}:{s:02d}"
        return f"{fmt(self.start_ts)}-{fmt(self.end_ts)}"


class M5Answer(BaseModel):
    """
    Structured answer returned by M5 to M6/frontend (PRD §11).
    `answer_text` is the full human-readable response.
    `cited_segments` are the segments the answer is grounded in.
    `explanation_bullets` are the per-signal evidence citations.
    `no_match` is True when no segment exceeded the relevance threshold.
    """
    answer_text: str
    cited_segments: list[CitedSegment] = Field(default_factory=list)
    explanation_bullets: list[str] = Field(default_factory=list)
    no_match: bool = False
    confidence: str = "high"   # "high" | "low" | "none"
    structured_query: StructuredQuery | None = None   # for debugging/M6 display
