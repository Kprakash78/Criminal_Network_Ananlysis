"""
M6 — Investigator Dashboard Mocks
PS 26152 — AI-Powered Criminal Network Analysis System

Schema-compliant mock implementations of M5.handle_investigator_request(),
M2's query functions, and M3's centrality/priority scores.

These are used by default when USE_REAL_MODULES=False (the default).
They return the same fictional FIR103 / Ravi Kumar case data used in M5's mocks,
so the full dashboard flow can be tested and demoed without the real backend.

DO NOT import from this file in production code paths.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import random


# ---------------------------------------------------------------------------
# Mock FINAL RESPONSE (mimics M5's FinalResponse dataclass exactly)
# ---------------------------------------------------------------------------

@dataclass
class MockFinalResponse:
    session_id: str
    response_text: str
    evidence: list
    confidence: float
    requires_human_review: bool

    def to_dict(self):
        return {
            "session_id": self.session_id,
            "response_text": self.response_text,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "requires_human_review": self.requires_human_review,
        }


_SESSION_STORE: dict[str, dict] = {}
_SESSION_COUNTER = [0]


def mock_handle_investigator_request(session_id, request, **kwargs):
    """
    Mock M5.handle_investigator_request().
    Accepts NewCaseUpload or FollowUpQuestion (by duck-typing, not import).
    Returns a MockFinalResponse.
    """
    # Determine if new case or follow-up
    is_followup = hasattr(request, "question")
    is_new_case = hasattr(request, "case_id")

    if is_new_case and not session_id:
        _SESSION_COUNTER[0] += 1
        session_id = f"SESS_MOCK_{_SESSION_COUNTER[0]:04d}"

    if session_id not in _SESSION_STORE:
        _SESSION_STORE[session_id] = {
            "case_id": getattr(request, "case_id", "UNKNOWN"),
            "history": [],
        }

    store = _SESSION_STORE[session_id]

    if is_followup:
        question = request.question
        store["history"].append({"role": "investigator", "content": question})
        summary_text = (
            f"Based on the active session for case {store['case_id']}, "
            f"regarding your question: \"{question[:80]}\" — "
            "entity P001 (Ravi Kumar) shows connections to account ACC00102 "
            "via phone 9876543210. Transaction patterns indicate elevated activity "
            "within a 2-hour window on the incident date. "
            "All findings require investigator verification before any action is taken."
        )
        confidence = 0.68
    else:
        store["case_id"] = getattr(request, "case_id", "FIR103")
        store["history"].append({"role": "investigator", "content": store["case_id"]})
        summary_text = (
            f"Analysis of case {store['case_id']} identifies Ravi Kumar (P001) "
            "as a potential person of interest based on communication spike patterns "
            "and network centrality. Entity P001 was involved in 14 calls within a "
            "2-hour window and shows connections to account ACC00102. "
            "These are potential investigative leads requiring investigator verification "
            "before any action is taken. "
            "All findings require investigator verification before any action is taken."
        )
        confidence = 0.72

    store["history"].append({"role": "system", "content": summary_text})

    return MockFinalResponse(
        session_id=session_id,
        response_text=summary_text,
        evidence=["RAG_RESULT_1", "PATTERN_FLAG_P001", "GRAPH_EDGE_P001_PH001"],
        confidence=confidence,
        requires_human_review=confidence < 0.5,
    )


def mock_handle_investigator_request_error(session_id, request, **kwargs):
    """Simulates a backend failure — for error-state testing."""
    raise RuntimeError("Simulated backend failure: M5 pipeline could not complete.")


# ---------------------------------------------------------------------------
# Mock M2 query functions
# ---------------------------------------------------------------------------

MOCK_NODES = [
    {"id": "P001", "label": "Ravi Kumar", "type": "PERSON", "confidence": 0.91},
    {"id": "P002", "label": "Meena Rao", "type": "PERSON", "confidence": 0.85},
    {"id": "PH001", "label": "9876543210", "type": "PHONE", "confidence": 0.98},
    {"id": "ACC001", "label": "ACC00102", "type": "ACCOUNT", "confidence": 0.97},
    {"id": "LOC001", "label": "Lajpat Nagar", "type": "LOCATION", "confidence": 0.80},
    {"id": "P003", "label": "Suresh Patel", "type": "PERSON", "confidence": 0.75},
    {"id": "PH002", "label": "9123456789", "type": "PHONE", "confidence": 0.92},
]

MOCK_EDGES = [
    {"source": "P001", "target": "PH001", "type": "USED_PHONE", "confidence": 0.88, "weight": 1.5},
    {"source": "P001", "target": "P002", "type": "APPEARS_WITH", "confidence": 0.85, "weight": 1.0},
    {"source": "P001", "target": "ACC001", "type": "LINKED_ACCOUNT", "confidence": 0.90, "weight": 2.0},
    {"source": "P001", "target": "LOC001", "type": "SEEN_AT", "confidence": 0.78, "weight": 1.0},
    {"source": "P002", "target": "ACC001", "type": "LINKED_ACCOUNT", "confidence": 0.82, "weight": 1.5},
    {"source": "P003", "target": "PH002", "type": "USED_PHONE", "confidence": 0.88, "weight": 1.0},
    {"source": "P001", "target": "P003", "type": "APPEARS_WITH", "confidence": 0.72, "weight": 0.8},
    {"source": "PH001", "target": "PH002", "type": "CALL_LINK", "confidence": 0.85, "weight": 3.0},
]

# Synthetic date range for relationships
_BASE_DATE = datetime(2024, 1, 15)
for i, edge in enumerate(MOCK_EDGES):
    edge["timestamp"] = (_BASE_DATE + timedelta(hours=i * 2)).strftime("%Y-%m-%d %H:%M")


def mock_subgraph(graph, case_id: str) -> dict:
    """Returns full mock graph for any case_id."""
    return {"nodes": MOCK_NODES, "edges": MOCK_EDGES}


def mock_neighbors(graph, entity_id: str) -> list[dict]:
    """Returns neighbours of the given entity_id from mock data."""
    neighbors = []
    for edge in MOCK_EDGES:
        if edge["source"] == entity_id:
            target = next((n for n in MOCK_NODES if n["id"] == edge["target"]), None)
            if target:
                neighbors.append({
                    "entity_id": target["id"],
                    "name": target["label"],
                    "type": target["type"],
                    "direction": "outgoing",
                    "relationship": edge["type"],
                    "confidence": edge["confidence"],
                    "weight": edge["weight"],
                    "timestamp": edge.get("timestamp", ""),
                    "source_record": "FIR103",
                })
        elif edge["target"] == entity_id:
            source = next((n for n in MOCK_NODES if n["id"] == edge["source"]), None)
            if source:
                neighbors.append({
                    "entity_id": source["id"],
                    "name": source["label"],
                    "type": source["type"],
                    "direction": "incoming",
                    "relationship": edge["type"],
                    "confidence": edge["confidence"],
                    "weight": edge["weight"],
                    "timestamp": edge.get("timestamp", ""),
                    "source_record": "FIR103",
                })
    return neighbors


def mock_search_by_name(graph, query: str, threshold: float = 60.0) -> list[dict]:
    """Fuzzy-matches query against mock node labels."""
    query_lower = query.lower()
    results = []
    for node in MOCK_NODES:
        label_lower = node["label"].lower()
        if query_lower in label_lower or label_lower in query_lower:
            results.append({
                "entity_id": node["id"],
                "name": node["label"],
                "type": node["type"],
                "score": 90.0,
                "confidence": node["confidence"],
            })
    return results


# ---------------------------------------------------------------------------
# Mock M3 centrality / priority scores
# ---------------------------------------------------------------------------

MOCK_CENTRALITY_SCORES = {
    "P001": {"degree": 0.67, "betweenness": 0.45, "closeness": 0.60, "pagerank": 0.15, "community": 0},
    "P002": {"degree": 0.33, "betweenness": 0.10, "closeness": 0.40, "pagerank": 0.08, "community": 0},
    "PH001": {"degree": 0.22, "betweenness": 0.05, "closeness": 0.30, "pagerank": 0.05, "community": 0},
    "ACC001": {"degree": 0.33, "betweenness": 0.20, "closeness": 0.45, "pagerank": 0.10, "community": 0},
    "LOC001": {"degree": 0.11, "betweenness": 0.02, "closeness": 0.20, "pagerank": 0.03, "community": 1},
    "P003": {"degree": 0.22, "betweenness": 0.08, "closeness": 0.35, "pagerank": 0.06, "community": 1},
    "PH002": {"degree": 0.22, "betweenness": 0.12, "closeness": 0.38, "pagerank": 0.07, "community": 1},
}

MOCK_PATTERN_FLAGS = [
    {
        "entity_id": "P001",
        "name": "Ravi Kumar",
        "flags": ["COMMUNICATION_SPIKE", "DENSE_CLUSTER_MEMBERSHIP"],
        "priority_score": 0.82,
        "evidence": [
            "Degree centrality 0.670 — entity connects to 67.0% of network",
            "[COMMUNICATION_SPIKE] 14 calls to PH002 in 2 hours on 2024-01-15",
        ],
    },
    {
        "entity_id": "ACC001",
        "name": "ACC00102",
        "flags": ["HIGH_TRANSACTION_FREQUENCY", "RAPID_FUND_MOVEMENT"],
        "priority_score": 0.65,
        "evidence": [
            "[HIGH_TRANSACTION_FREQUENCY] 8 transactions in 3 hours",
            "[RAPID_FUND_MOVEMENT] INR 200,000 transferred within 30 minutes",
        ],
    },
    {
        "entity_id": "PH001",
        "name": "9876543210",
        "flags": ["COMMUNICATION_SPIKE"],
        "priority_score": 0.55,
        "evidence": ["[COMMUNICATION_SPIKE] 14 outgoing calls in 2-hour window"],
    },
    {
        "entity_id": "P003",
        "name": "Suresh Patel",
        "flags": ["DENSE_CLUSTER_MEMBERSHIP"],
        "priority_score": 0.42,
        "evidence": ["3/4 connections belong to the same cluster (cluster 1)"],
    },
]


def mock_get_key_players():
    """Returns sorted pattern flags for the key-players view."""
    return sorted(MOCK_PATTERN_FLAGS, key=lambda x: x["priority_score"], reverse=True)
