"""
M3 — Shared Data Structures
PS 26152 — AI-Powered Criminal Network Analysis System

Defines the PatternFlag dataclass (matches the shared PATTERN FLAG schema
from backend.md §4 exactly) and the AnalysisResult container.
These types must not be changed without team sign-off — M4, M5, M6 depend on them.
"""

from dataclasses import dataclass, field


# Authoritative flag-type list (backend.md §4).
# Do not add to this set without updating backend.md and notifying the team.
VALID_FLAG_TYPES = frozenset({
    "COMMUNICATION_SPIKE",
    "HIGH_TRANSACTION_FREQUENCY",
    "MULTI_SUSPECT_SHARED_ACCOUNT",
    "RAPID_FUND_MOVEMENT",
    "INCIDENT_TIMING_CLUSTER",
    "DENSE_CLUSTER_MEMBERSHIP",
})


@dataclass
class PatternFlag:
    """
    One investigation-lead record per flagged entity.
    Matches the PATTERN FLAG schema in backend.md §4 exactly.

    Language note: flags indicate elevated priority for investigator review,
    never guilt or criminal determination.
    """
    entity_id: str
    degree_centrality: float
    betweenness_centrality: float
    community: int
    flags: list[str]          # subset of VALID_FLAG_TYPES
    priority_score: float     # [0, 1] — higher = higher investigator priority
    evidence: list[str]       # ≥1 human-readable string per flag, always populated


@dataclass
class AnalysisResult:
    """
    Full output of run_full_analysis().
    Contains both the complete centrality table (M6 needs all nodes)
    and the focused flag list (M4/M5 need only flagged entities).
    """
    # Every node — used by M6 dashboard for ranking view
    centrality_scores: dict[str, dict[str, float]]  # entity_id → centrality dict
    community_map: dict[str, int]                    # entity_id → community_id

    # Only flagged entities — used by M4 (RAG) and M5 (routing)
    pattern_flags: list[PatternFlag]
