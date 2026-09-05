"""
M5.1 — Tool Wrappers
PS 26152 — AI-Powered Criminal Network Analysis System

Wraps M1–M4's real functions as callable tools for the LangGraph workflow.
Each tool function:
  - has a clear docstring (what it does, what it returns)
  - catches all exceptions from the upstream module and converts them to
    a dict with an "error" key — the graph node reads this and routes to
    human review rather than crashing the session
  - enforces the tool-call budget (M5.7 guardrails) via the shared counter
    in SessionState

PRODUCTION WIRING (default):
  By default, this module imports and calls the REAL M1–M4 implementations.
  The graph.py and entry.py modules pass `use_mocks=False` by default.

MOCK WIRING (for testing):
  Pass `use_mocks=True` to build_workflow_graph() (or call the graph nodes
  with state that has _use_mocks=True) to use mocks/mock_tools.py instead.
  Mocks live in M5/mocks/ and are NOT imported here — no test code leaks
  into the production path.

Guardrail: each call to _call_tool() increments state._tool_call_count.
If the count exceeds MAX_TOOL_CALLS, a guardrail error is returned without
calling the upstream module.

Timeout: each call runs with a configurable timeout (TOOL_TIMEOUT_SECONDS).
If the upstream call exceeds the timeout, the exception is caught and the
error dict is returned.
"""

import logging
import signal
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

# M5.7 Guardrails — configured here, enforced in _call_tool()
MAX_TOOL_CALLS = 10        # max tool calls per single handle_investigator_request()
TOOL_TIMEOUT_SECONDS = 60  # per-tool timeout


def _run_with_timeout(fn, args, kwargs, timeout_seconds: int) -> Any:
    """
    Run fn(*args, **kwargs) in a thread. If it doesn't finish within
    timeout_seconds, raise TimeoutError.
    """
    result_container: list = []
    error_container: list = []

    def target():
        try:
            result_container.append(fn(*args, **kwargs))
        except Exception as exc:
            error_container.append(exc)

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout=timeout_seconds)

    if t.is_alive():
        raise TimeoutError(
            f"Tool call to {fn.__name__} exceeded {timeout_seconds}s timeout."
        )
    if error_container:
        raise error_container[0]
    return result_container[0]


def _call_tool(fn, args: tuple, kwargs: dict, state, tool_name: str) -> Any:
    """
    Central dispatcher: checks guardrail budget, runs fn with timeout,
    catches all exceptions.

    Returns the tool's return value, OR a dict {"error": str} on failure.
    Increments state._tool_call_count regardless of outcome.
    """
    state._tool_call_count += 1

    if state._tool_call_count > MAX_TOOL_CALLS:
        msg = (
            f"[Guardrail] Tool-call cap ({MAX_TOOL_CALLS}) exceeded at call #{state._tool_call_count}. "
            f"Aborting {tool_name} — returning partial results."
        )
        logger.warning(msg)
        return {"error": msg}

    logger.info(
        f"[Tool] Calling {tool_name} (call #{state._tool_call_count}/{MAX_TOOL_CALLS})"
    )
    t0 = time.perf_counter()
    try:
        result = _run_with_timeout(fn, args, kwargs, TOOL_TIMEOUT_SECONDS)
        elapsed = time.perf_counter() - t0
        logger.info(f"[Tool] {tool_name} completed in {elapsed:.1f}s")
        return result
    except TimeoutError as exc:
        logger.error(f"[Tool] {tool_name} timed out: {exc}")
        return {"error": str(exc)}
    except Exception as exc:
        logger.error(f"[Tool] {tool_name} failed: {exc}", exc_info=True)
        return {"error": f"{tool_name} raised {type(exc).__name__}: {exc}"}


# ---------------------------------------------------------------------------
# Real tool implementations — these call the actual M1–M4 modules
# ---------------------------------------------------------------------------

def tool_extract_entities(state, case_id: str, case_text: str) -> dict:
    """
    Call M1.extract_entities() to extract entities from a new case text.
    Writes the text to a temporary file to interface with M1's file-based API.

    Returns:
        dict with key "entities" (list[dict]), or {"error": str} on failure.
    """
    import tempfile
    import os
    from M1.pipeline import extract_entities

    # Write case text to a temporary file
    fd, temp_path = tempfile.mkstemp(suffix=".txt", prefix=f"{case_id}_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(case_text)
        
        result = _call_tool(
            extract_entities, (temp_path, "FIR"), {}, state, "M1.extract_entities"
        )
        if isinstance(result, dict) and "error" in result:
            return result

        # result is a list of Entity pydantic models; convert to dicts
        return {
            "entities": [e.model_dump() for e in result]
        }
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


def tool_graph_neighbors(state, graph, entity_id: str) -> dict:
    """
    Call M2.query.neighbors() to get all direct connections of an entity.

    Returns:
        dict with key "neighbors" (list[dict]),
        or {"error": str} on failure.
    """
    from M2.query import neighbors

    result = _call_tool(
        neighbors, (graph, entity_id), {}, state, "M2.neighbors"
    )
    if isinstance(result, dict) and "error" in result:
        return result
    return {"neighbors": result}


def tool_graph_search(state, graph, query: str, threshold: float = 80.0) -> dict:
    """
    Call M2.query.search_by_name() to find entities by fuzzy name match.

    Returns:
        dict with key "matches" (list[dict]),
        or {"error": str} on failure.
    """
    from M2.query import search_by_name

    result = _call_tool(
        search_by_name, (graph, query), {"threshold": threshold}, state, "M2.search_by_name"
    )
    if isinstance(result, dict) and "error" in result:
        return result
    return {"matches": result}


def tool_run_analysis(state, graph, cdr_path: str, txn_path: str) -> dict:
    """
    Call M3.pipeline.run_full_analysis() to compute centrality and pattern flags.

    Returns:
        dict with keys "pattern_flags" (list[dict]) and "centrality_scores" (dict),
        or {"error": str} on failure.
    """
    from M3.pipeline import run_full_analysis
    from dataclasses import asdict

    result = _call_tool(
        run_full_analysis,
        (graph, cdr_path, txn_path),
        {},
        state,
        "M3.run_full_analysis",
    )
    if isinstance(result, dict) and "error" in result:
        return result

    return {
        "pattern_flags": [asdict(pf) for pf in result.pattern_flags],
        "centrality_scores": result.centrality_scores,
    }


def tool_generate_summary(state, pipeline, case_text: str, case_id: str,
                          entity_ids: list[str] | None = None) -> dict:
    """
    Call M4.RAGPipeline.generate_summary() to produce the RAG-grounded summary.

    Args:
        pipeline: an already-loaded M4.RAGPipeline instance.
        case_text: raw FIR text for the new case.
        case_id: identifier for this case (e.g. "FIR103").
        entity_ids: optional list of entity IDs to pull evidence for.

    Returns:
        dict with keys "summary_text", "evidence_used", "confidence",
        or {"error": str} on failure.
    """
    from dataclasses import asdict

    result = _call_tool(
        pipeline.generate_summary,
        (),
        {"new_case_text": case_text, "case_id": case_id,
         "entity_ids": entity_ids, "emit_files": True},
        state,
        "M4.generate_summary",
    )
    if isinstance(result, dict) and "error" in result:
        return result

    return {
        "summary_text": result.summary_text,
        "evidence_used": result.evidence_used,
        "confidence": result.confidence,
        "case_id": result.case_id,
    }
