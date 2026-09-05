"""
M5.2–M5.7 — LangGraph Workflow Definition
PS 26152 — AI-Powered Criminal Network Analysis System

Defines the fixed investigation workflow as a LangGraph StateGraph.

Step order (from backend.md §2):
  1. extract_entities_node   (calls M1 — SKIP for follow-up questions)
  2. graph_lookup_node       (calls M2.neighbors + M2.search_by_name)
  3. rag_retrieval_node      (calls M4.generate_summary as first pass — see NOTE below)
  4. graph_analysis_node     (calls M3.run_full_analysis for centrality/flags)
  5. generate_summary_node   (calls M4.generate_summary with full context)
  6. confidence_check_node   (routes to human_review or emit)
  7. emit_response_node      (M5.8 — assembles the FINAL RESPONSE)
  8. human_review_node       (sets requires_human_review=True and assembles response)

NOTE on workflow vs backend.md step order:
  backend.md lists: extract → graph lookup → RAG retrieval → graph analysis →
  pattern analysis → generate summary → confidence check.
  M4's RAGPipeline combines "RAG retrieval" and "generate summary" in a single
  generate_summary() call (it retrieves then generates internally). We therefore
  call M4 once in generate_summary_node with the M3 flags already in state.
  The graph_analysis_node handles both "graph analysis" and "pattern analysis"
  by calling M3.run_full_analysis() which runs centrality + pattern rules.

M5.5 — Follow-up question handling:
  If state.entities_in_context is non-empty (session already has context),
  we skip extract_entities_node and re-enter at graph_lookup_node.

M5.7 — Guardrails:
  Tool-call cap (MAX_TOOL_CALLS) and per-call timeout (TOOL_TIMEOUT_SECONDS)
  are enforced in tools._call_tool(). If the cap is exceeded mid-workflow,
  the tool returns {"error": ...} and the node routes to human_review.
"""

import logging
from typing import Annotated, TypedDict, Literal

from langgraph.graph import StateGraph, END, START

from M5.models import SessionState
from M5.routing import route_on_confidence, HUMAN_REVIEW_THRESHOLD

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LangGraph state dict — thin wrapper that holds the SessionState object.
# LangGraph requires its state to be a TypedDict; we use a single "state" key
# holding our SessionState dataclass. This avoids the overhead of flattening
# all SessionState fields into TypedDict keys.
# ---------------------------------------------------------------------------

class WorkflowState(TypedDict):
    state: SessionState
    use_mocks: bool
    graph: object          # M2's CriminalGraph (may be None for mock runs)
    m4_pipeline: object    # M4's RAGPipeline (may be None for mock runs)
    cdr_path: str
    txn_path: str
    question: str          # follow-up question text (empty for new-case uploads)


# ---------------------------------------------------------------------------
# Tool dispatch helper — picks real or mock tool based on use_mocks flag
# ---------------------------------------------------------------------------

def _pick(real_fn, mock_fn, use_mocks: bool):
    return mock_fn if use_mocks else real_fn


# ---------------------------------------------------------------------------
# Node: extract_entities
# M5.2, M5.5 — called only for NEW case uploads, skipped for follow-ups
# ---------------------------------------------------------------------------

def extract_entities_node(ws: WorkflowState) -> WorkflowState:
    from M5.tools import tool_extract_entities
    from M5.mocks.mock_tools import mock_extract_entities

    state = ws["state"]
    use_mocks = ws["use_mocks"]
    logger.info(f"[Node] extract_entities — session {state.session_id}")

    fn = _pick(tool_extract_entities, mock_extract_entities, use_mocks)
    result = fn(state, state.current_case_id, state.current_case_text)

    if "error" in result:
        state._error_message = result["error"]
        state._requires_human_review = True
    else:
        entities = result.get("entities", [])
        state._m1_entities = entities
        # Populate entities_in_context with extracted entity IDs
        state.entities_in_context = [e["entity_id"] for e in entities]
        logger.info(f"[Node] Extracted {len(entities)} entities")

    ws["state"] = state
    return ws


# ---------------------------------------------------------------------------
# Node: graph_lookup
# M5.2 — look up neighbours for all entities currently in context
# ---------------------------------------------------------------------------

def graph_lookup_node(ws: WorkflowState) -> WorkflowState:
    from M5.tools import tool_graph_neighbors, tool_graph_search
    from M5.mocks.mock_tools import mock_graph_neighbors, mock_graph_search

    state = ws["state"]
    use_mocks = ws["use_mocks"]
    graph = ws["graph"]
    logger.info(f"[Node] graph_lookup — session {state.session_id}")

    if state._requires_human_review:
        return ws  # already failed upstream; skip

    neighbors_fn = _pick(tool_graph_neighbors, mock_graph_neighbors, use_mocks)
    search_fn = _pick(tool_graph_search, mock_graph_search, use_mocks)

    all_neighbors: list[dict] = []

    # For follow-up questions, also search by keywords from the question text
    if ws.get("question"):
        search_result = search_fn(state, graph, ws["question"])
        if "error" not in search_result:
            matched_ids = [m["entity_id"] for m in search_result.get("matches", [])]
            # Add newly matched IDs that aren't already in context
            for eid in matched_ids:
                if eid not in state.entities_in_context:
                    state.entities_in_context.append(eid)

    # Pull neighbours for each entity in context
    for entity_id in state.entities_in_context[:5]:  # cap to 5 to stay within tool budget
        result = neighbors_fn(state, graph, entity_id)
        if "error" not in result:
            all_neighbors.extend(result.get("neighbors", []))

    state._m2_neighbors = all_neighbors
    logger.info(f"[Node] Found {len(all_neighbors)} neighbour edges")
    ws["state"] = state
    return ws


# ---------------------------------------------------------------------------
# Node: graph_analysis
# M5.2 — run M3 analytics (centrality + pattern flags)
# ---------------------------------------------------------------------------

def graph_analysis_node(ws: WorkflowState) -> WorkflowState:
    from M5.tools import tool_run_analysis
    from M5.mocks.mock_tools import mock_run_analysis

    state = ws["state"]
    use_mocks = ws["use_mocks"]
    graph = ws["graph"]
    logger.info(f"[Node] graph_analysis — session {state.session_id}")

    if state._requires_human_review:
        return ws

    fn = _pick(tool_run_analysis, mock_run_analysis, use_mocks)
    result = fn(state, graph, ws["cdr_path"], ws["txn_path"])

    if "error" in result:
        state._error_message = result["error"]
        state._requires_human_review = True
        logger.warning(f"[Node] graph_analysis failed: {result['error']}")
    else:
        state._m3_pattern_flags = result.get("pattern_flags", [])
        logger.info(f"[Node] Got {len(state._m3_pattern_flags)} pattern flags")

    ws["state"] = state
    return ws


# ---------------------------------------------------------------------------
# Node: generate_summary
# M5.2 — call M4 to produce the final RAG-grounded summary
# ---------------------------------------------------------------------------

def generate_summary_node(ws: WorkflowState) -> WorkflowState:
    from M5.tools import tool_generate_summary
    from M5.mocks.mock_tools import mock_generate_summary

    state = ws["state"]
    use_mocks = ws["use_mocks"]
    pipeline = ws["m4_pipeline"]
    logger.info(f"[Node] generate_summary — session {state.session_id}")

    if state._requires_human_review:
        return ws

    fn = _pick(tool_generate_summary, mock_generate_summary, use_mocks)

    # For follow-up questions, build the query text from the question + conversation history
    if ws.get("question"):
        case_text = (
            f"Follow-up question: {ws['question']}\n\n"
            f"Original case: {state.current_case_text[:1000]}"
        )
    else:
        case_text = state.current_case_text

    result = fn(
        state, pipeline, case_text, state.current_case_id,
        entity_ids=state.entities_in_context or None,
    )

    if "error" in result:
        state._error_message = result["error"]
        state._requires_human_review = True
        logger.warning(f"[Node] generate_summary failed: {result['error']}")
    else:
        state._m4_summary = result
        state._confidence = result.get("confidence", 0.0)
        state.last_evidence = result.get("evidence_used", [])
        logger.info(
            f"[Node] Summary generated, confidence={state._confidence:.3f}"
        )

    ws["state"] = state
    return ws


# ---------------------------------------------------------------------------
# Node: confidence_check
# M5.3 — route on confidence threshold
# ---------------------------------------------------------------------------

def confidence_check_node(ws: WorkflowState) -> WorkflowState:
    state = ws["state"]

    if state._requires_human_review:
        return ws  # already routing to human review

    decision = route_on_confidence(state._confidence)
    # BUG B fix: do not short-circuit to human review for follow-up questions
    if decision.route == "human_review" and not ws.get("question"):
        state._requires_human_review = True
        state._error_message = decision.reason

    ws["state"] = state
    return ws


# ---------------------------------------------------------------------------
# Node: emit_response  (M5.8 — validated FINAL RESPONSE output)
# ---------------------------------------------------------------------------

def emit_response_node(ws: WorkflowState) -> WorkflowState:
    state = ws["state"]
    logger.info(f"[Node] emit_response — session {state.session_id}")

    summary = state._m4_summary
    response_text = summary.get(
        "summary_text",
        "Analysis complete. Please review the evidence list."
    )

    # Append conversation turn
    state.conversation_history.append(
        {"role": "investigator", "content": ws.get("question") or state.current_case_id}
    )
    state.conversation_history.append(
        {"role": "system", "content": response_text}
    )

    ws["state"] = state
    return ws


# ---------------------------------------------------------------------------
# Node: human_review
# M5.3, M5.8 — handles low-confidence or error cases
# ---------------------------------------------------------------------------

def human_review_node(ws: WorkflowState) -> WorkflowState:
    state = ws["state"]
    logger.info(f"[Node] human_review — session {state.session_id}")

    error_msg = state._error_message or "Automated analysis could not be completed."
    response_text = (
        f"[REQUIRES HUMAN REVIEW] {error_msg} "
        f"An investigator should manually review case {state.current_case_id}."
    )

    state.conversation_history.append(
        {"role": "investigator", "content": ws.get("question") or state.current_case_id}
    )
    state.conversation_history.append(
        {"role": "system", "content": response_text}
    )

    ws["state"] = state
    return ws


# ---------------------------------------------------------------------------
# Conditional edges
# ---------------------------------------------------------------------------

def _route_after_confidence(ws: WorkflowState) -> Literal["emit_response", "human_review"]:
    if ws["state"]._requires_human_review:
        return "human_review"
    return "emit_response"


def _should_skip_extraction(ws: WorkflowState) -> Literal["extract_entities", "graph_lookup"]:
    """Skip entity extraction for follow-up questions (M5.5)."""
    state = ws["state"]
    is_followup = bool(ws.get("question")) and bool(state.entities_in_context)
    if is_followup:
        logger.info(f"[Graph] Follow-up detected — skipping extraction, re-entering at graph_lookup")
        return "graph_lookup"
    return "extract_entities"


# ---------------------------------------------------------------------------
# M5.2 — Build the workflow graph
# ---------------------------------------------------------------------------

def build_workflow_graph() -> object:
    """
    Construct and compile the LangGraph workflow.

    Returns a compiled StateGraph ready to be invoked.
    Call this once at startup and reuse the compiled graph.
    """
    builder = StateGraph(WorkflowState)

    # Register nodes
    builder.add_node("extract_entities", extract_entities_node)
    builder.add_node("graph_lookup", graph_lookup_node)
    builder.add_node("graph_analysis", graph_analysis_node)
    builder.add_node("generate_summary", generate_summary_node)
    builder.add_node("confidence_check", confidence_check_node)
    builder.add_node("emit_response", emit_response_node)
    builder.add_node("human_review", human_review_node)

    # Entry: conditional — new case runs extraction, follow-up skips it
    builder.add_conditional_edges(
        START,
        _should_skip_extraction,
        {"extract_entities": "extract_entities", "graph_lookup": "graph_lookup"},
    )

    # Linear flow
    builder.add_edge("extract_entities", "graph_lookup")
    builder.add_edge("graph_lookup", "graph_analysis")
    builder.add_edge("graph_analysis", "generate_summary")
    builder.add_edge("generate_summary", "confidence_check")

    # Confidence routing
    builder.add_conditional_edges(
        "confidence_check",
        _route_after_confidence,
        {"emit_response": "emit_response", "human_review": "human_review"},
    )

    # Both terminal nodes end
    builder.add_edge("emit_response", END)
    builder.add_edge("human_review", END)

    compiled = builder.compile()
    logger.info("[Graph] Workflow compiled successfully")
    return compiled
