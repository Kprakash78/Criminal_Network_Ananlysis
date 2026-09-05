"""
M5 — Agentic AI / LangGraph Engineer
PS 26152 — AI-Powered Criminal Network Analysis System

Public API:
    handle_investigator_request(session_id, request) -> FinalResponse
    build_workflow_graph() -> CompiledStateGraph
    get_session_state(session_id) -> SessionState | None
    route_on_confidence(confidence, threshold) -> RoutingDecision

M6 should call handle_investigator_request() for all interactions.
M5 should NOT be called directly for entity extraction, graph queries,
analytics, or RAG — those always go through the tool wrappers in tools.py.
"""

from M5.models import SessionState, FinalResponse, NewCaseUpload, FollowUpQuestion
from M5.session import get_session_state, clear_session
from M5.entry import handle_investigator_request
from M5.routing import route_on_confidence
from M5.graph import build_workflow_graph

__all__ = [
    "handle_investigator_request",
    "build_workflow_graph",
    "get_session_state",
    "clear_session",
    "route_on_confidence",
    "SessionState",
    "FinalResponse",
    "NewCaseUpload",
    "FollowUpQuestion",
]
