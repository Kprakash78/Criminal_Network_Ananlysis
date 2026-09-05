"""
M5.6 — Top-Level Entry Point
PS 26152 — AI-Powered Criminal Network Analysis System

handle_investigator_request() is the SINGLE function M6 calls for all
interactions — both new case uploads and follow-up questions.

M6 usage:
    from M5 import handle_investigator_request
    from M5.models import NewCaseUpload, FollowUpQuestion

    # New case
    response = handle_investigator_request("SESS_001", NewCaseUpload(
        case_id="FIR103",
        case_text="...",
    ))

    # Follow-up
    response = handle_investigator_request("SESS_001", FollowUpQuestion(
        question="How is Ravi Kumar connected to the financial transactions?",
    ))

Both calls return a FinalResponse matching backend.md §4 FINAL RESPONSE schema.
"""

import logging
from typing import Union

from M5.models import (
    FinalResponse, NewCaseUpload, FollowUpQuestion, SessionState
)
from M5.session import get_or_create_session, save_session_state
from M5.graph import build_workflow_graph, WorkflowState

logger = logging.getLogger(__name__)

# Compile the graph once at module-level (avoids recompiling per request)
_COMPILED_GRAPH = None


def _get_graph():
    global _COMPILED_GRAPH
    if _COMPILED_GRAPH is None:
        _COMPILED_GRAPH = build_workflow_graph()
    return _COMPILED_GRAPH


def handle_investigator_request(
    session_id: str | None,
    request: Union[NewCaseUpload, FollowUpQuestion],
    *,
    use_mocks: bool = False,
    graph=None,          # M2's CriminalGraph — pass None to use mocks
    m4_pipeline=None,    # M4's RAGPipeline — pass None to use mocks
    cdr_path: str = "",
    txn_path: str = "",
    confidence_threshold: float = 0.5,
) -> FinalResponse:
    """
    Top-level entry point for M6. Handles both new-case uploads and follow-up questions.

    Args:
        session_id:   existing session ID (or None to start a new session)
        request:      NewCaseUpload or FollowUpQuestion
        use_mocks:    if True, use mock M1–M4 tools (for testing)
        graph:        M2's CriminalGraph (pass None when use_mocks=True)
        m4_pipeline:  M4's RAGPipeline, already loaded (pass None when use_mocks=True)
        cdr_path:     path to CDR CSV (for M3 real call)
        txn_path:     path to transactions CSV (for M3 real call)
        confidence_threshold: override the default routing threshold

    Returns:
        FinalResponse matching backend.md §4 schema exactly.
    """
    # --- Session setup ---
    state = get_or_create_session(session_id)

    # --- Reset per-request state (guardrail counters, error flags) ---
    state._tool_call_count = 0
    state._requires_human_review = False
    state._error_message = ""
    state._m1_entities = []
    state._m2_neighbors = []
    state._m3_pattern_flags = []
    state._m4_summary = {}
    state._confidence = 0.0

    # --- Populate state from request ---
    question = ""
    if isinstance(request, NewCaseUpload):
        state.current_case_id = request.case_id
        state.current_case_text = request.case_text
        # Clear entities for a brand-new case
        state.entities_in_context = []
        state.last_evidence = []
        logger.info(
            f"[Entry] New case upload — case={request.case_id}, session={state.session_id}"
        )
    elif isinstance(request, FollowUpQuestion):
        question = request.question
        if not state.current_case_id:
            # No active case — treat as a new case with the question as context
            state.current_case_id = "UNKNOWN"
            state.current_case_text = request.question
        logger.info(
            f"[Entry] Follow-up question — session={state.session_id}: {question[:80]}"
        )
    else:
        raise TypeError(f"request must be NewCaseUpload or FollowUpQuestion, got {type(request)}")

    # --- Build the workflow state dict ---
    ws: WorkflowState = {
        "state": state,
        "use_mocks": use_mocks,
        "graph": graph,
        "m4_pipeline": m4_pipeline,
        "cdr_path": cdr_path,
        "txn_path": txn_path,
        "question": question,
    }

    # --- Run the graph ---
    compiled = _get_graph()
    try:
        result_ws = compiled.invoke(ws)
        final_state: SessionState = result_ws["state"]
    except Exception as exc:
        logger.error(f"[Entry] Graph execution failed: {exc}", exc_info=True)
        # Catastrophic failure — build a human-review response without crashing
        final_state = state
        final_state._requires_human_review = True
        final_state._error_message = f"Internal workflow error: {exc}"
        final_state._confidence = 0.0

    # --- Persist updated session state ---
    save_session_state(final_state)

    # --- Build and return FINAL RESPONSE ---
    if final_state._requires_human_review:
        response_text = (
            final_state._error_message
            if final_state._error_message
            else "Automated analysis could not be completed. Please review manually."
        )
    else:
        summary = final_state._m4_summary
        response_text = summary.get(
            "summary_text",
            "Analysis complete. Evidence is listed in the evidence field."
        )

    # Build extracted_entities list for M6 entity display (Bug 2 fix)
    # This ensures the entity table shows entities FROM the uploaded doc, not the static graph.
    extracted_entities: list[dict] = []
    for e in (final_state._m1_entities or []):
        if isinstance(e, dict):
            extracted_entities.append({
                "entity_id": e.get("entity_id", ""),
                "name": e.get("name", e.get("entity_id", "")),
                "type": e.get("type", "UNKNOWN"),
                "confidence": e.get("confidence", 0.75),
            })

    return FinalResponse(
        session_id=final_state.session_id,
        response_text=response_text,
        evidence=final_state.last_evidence,
        confidence=round(final_state._confidence, 4),
        requires_human_review=final_state._requires_human_review,
        extracted_entities=extracted_entities,
    )
