"""
M5 — Mock Tool Implementations
PS 26152 — AI-Powered Criminal Network Analysis System

Schema-compliant fake implementations of every M1–M4 tool function.
These return realistic-looking data that matches the exact output shapes
documented in each module's backend.md.

These mocks are used ONLY in tests (via use_mocks=True in build_workflow_graph()).
Production paths NEVER import from this file.

Mock data represents a fictional investigation with:
  - Suspect "Ravi Kumar" (P001) and associate "Meena Rao" (P002)
  - Phone 9876543210 (PH001), Account ACC00102 (ACC001)
  - Case FIR103
"""

import time
from dataclasses import asdict


# ---------------------------------------------------------------------------
# Mock M1 — run_pipeline()
# ---------------------------------------------------------------------------

MOCK_M1_ENTITIES = [
    {
        "entity_id": "P001",
        "type": "PERSON",
        "name": "Ravi Kumar",
        "aliases": ["R. Kumar", "Ravi"],
        "source_documents": ["FIR103"],
        "confidence": 0.91,
        "needs_review": False,
    },
    {
        "entity_id": "P002",
        "type": "PERSON",
        "name": "Meena Rao",
        "aliases": ["M. Rao"],
        "source_documents": ["FIR103"],
        "confidence": 0.85,
        "needs_review": False,
    },
    {
        "entity_id": "PH001",
        "type": "PHONE",
        "name": "9876543210",
        "aliases": [],
        "source_documents": ["FIR103"],
        "confidence": 0.98,
        "needs_review": False,
    },
    {
        "entity_id": "ACC001",
        "type": "ACCOUNT",
        "name": "ACC00102",
        "aliases": [],
        "source_documents": ["FIR103"],
        "confidence": 0.97,
        "needs_review": False,
    },
]

MOCK_M1_RELATIONSHIPS = [
    {
        "source": "P001",
        "target": "PH001",
        "relationship": "USED_PHONE",
        "timestamp": "2024-01-15T10:00:00",
        "source_record": "FIR103",
        "confidence": 0.88,
    },
    {
        "source": "P001",
        "target": "P002",
        "relationship": "APPEARS_IN_SAME_DOCUMENT",
        "timestamp": "2024-01-15T10:00:00",
        "source_record": "FIR103",
        "confidence": 0.86,
    },
]


def mock_extract_entities(state, case_id: str, case_text: str) -> dict:
    """Mock M1: returns schema-compliant fake entity extraction."""
    time.sleep(0.05)
    return {
        "entities": MOCK_M1_ENTITIES,
        "relationships": MOCK_M1_RELATIONSHIPS,
        "total_entities": len(MOCK_M1_ENTITIES),
        "total_relationships": len(MOCK_M1_RELATIONSHIPS),
    }


# ---------------------------------------------------------------------------
# Mock M2 — neighbors(), search_by_name()
# ---------------------------------------------------------------------------

MOCK_M2_NEIGHBORS = {
    "P001": [
        {
            "entity_id": "PH001",
            "name": "9876543210",
            "type": "PHONE",
            "direction": "outgoing",
            "relationship": "USED_PHONE",
            "source_record": "FIR103",
            "confidence": 0.88,
            "weight": 1.5,
            "timestamp": "2024-01-15T10:00:00",
        },
        {
            "entity_id": "P002",
            "name": "Meena Rao",
            "type": "PERSON",
            "direction": "outgoing",
            "relationship": "APPEARS_IN_SAME_DOCUMENT",
            "source_record": "FIR103",
            "confidence": 0.85,
            "weight": 1.0,
            "timestamp": "2024-01-15T10:00:00",
        },
    ],
    "default": [],
}

MOCK_M2_SEARCH = [
    {
        "entity_id": "P001",
        "name": "Ravi Kumar",
        "type": "PERSON",
        "score": 95.0,
    }
]


def mock_graph_neighbors(state, graph, entity_id: str) -> dict:
    """Mock M2.neighbors(): returns neighbors for P001, empty for others."""
    time.sleep(0.02)
    return {"neighbors": MOCK_M2_NEIGHBORS.get(entity_id, MOCK_M2_NEIGHBORS["default"])}


def mock_graph_search(state, graph, query: str, threshold: float = 80.0) -> dict:
    """Mock M2.search_by_name(): returns Ravi Kumar for any query."""
    time.sleep(0.02)
    return {"matches": MOCK_M2_SEARCH}


# ---------------------------------------------------------------------------
# Mock M3 — run_full_analysis()
# ---------------------------------------------------------------------------

MOCK_M3_PATTERN_FLAGS = [
    {
        "entity_id": "P001",
        "degree_centrality": 0.67,
        "betweenness_centrality": 0.45,
        "community": 0,
        "flags": ["COMMUNICATION_SPIKE", "DENSE_CLUSTER_MEMBERSHIP"],
        "priority_score": 0.82,
        "evidence": [
            "Degree centrality 0.670 — entity connects to 67.0% of network",
            "[COMMUNICATION_SPIKE] 14 calls to PH002 in 2 hours on 2024-01-15",
        ],
    },
    {
        "entity_id": "ACC001",
        "degree_centrality": 0.33,
        "betweenness_centrality": 0.20,
        "community": 0,
        "flags": ["HIGH_TRANSACTION_FREQUENCY", "RAPID_FUND_MOVEMENT"],
        "priority_score": 0.65,
        "evidence": [
            "[HIGH_TRANSACTION_FREQUENCY] 8 transactions in 3 hours",
            "[RAPID_FUND_MOVEMENT] INR 200,000 transferred within 30 minutes",
        ],
    },
]

MOCK_M3_CENTRALITY = {
    "P001": {"degree": 0.67, "betweenness": 0.45, "closeness": 0.60, "pagerank": 0.15},
    "P002": {"degree": 0.33, "betweenness": 0.10, "closeness": 0.40, "pagerank": 0.08},
    "PH001": {"degree": 0.22, "betweenness": 0.05, "closeness": 0.30, "pagerank": 0.05},
    "ACC001": {"degree": 0.33, "betweenness": 0.20, "closeness": 0.45, "pagerank": 0.10},
}


def mock_run_analysis(state, graph, cdr_path: str, txn_path: str) -> dict:
    """Mock M3.run_full_analysis(): returns realistic pattern flags."""
    time.sleep(0.05)
    return {
        "pattern_flags": MOCK_M3_PATTERN_FLAGS,
        "centrality_scores": MOCK_M3_CENTRALITY,
    }


# ---------------------------------------------------------------------------
# Mock M4 — generate_summary()
# ---------------------------------------------------------------------------

MOCK_M4_SUMMARY = {
    "case_id": "FIR103",
    "summary_text": (
        "Analysis of case FIR103 identifies Ravi Kumar (P001) as a potential "
        "person of interest based on communication spike patterns and network "
        "centrality. Entity P001 was involved in 14 calls within a 2-hour "
        "window and shows connections to account ACC00102. These are potential "
        "investigative leads requiring investigator verification before any "
        "action is taken. All findings require investigator verification before "
        "any action is taken."
    ),
    "evidence_used": ["RAG_RESULT_1", "PATTERN_FLAG_P001", "GRAPH_EDGE_P001_PH001"],
    "confidence": 0.72,
}


def mock_generate_summary(state, pipeline, case_text: str, case_id: str,
                          entity_ids: list | None = None) -> dict:
    """Mock M4.generate_summary(): returns realistic RAG summary."""
    time.sleep(0.1)
    result = dict(MOCK_M4_SUMMARY)
    result["case_id"] = case_id
    return result
