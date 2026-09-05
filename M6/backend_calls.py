"""
M6 — Backend Call Layer
PS 26152 — AI-Powered Criminal Network Analysis System

This module is the ONLY place in M6 that imports from M5/M2/M3/M4.
All other dashboard code calls through these functions.

Mode selection:
  USE_REAL_MODULES = False (default) → schema-compliant mocks (fast, no deps)
  USE_REAL_MODULES = True            → real M1/M2/M3/M4/M5 backends

Set via environment variable:
    $env:CRIMINAL_USE_REAL_MODULES="1"   # PowerShell
    export CRIMINAL_USE_REAL_MODULES=1   # bash

Real-module startup (done once at import time, cached):
  - Loads CriminalGraph from M1/output/entities.json and relationships.json (M2.graph_builder)
  - Instantiates M4.RAGPipeline and calls .load(graph=...) — loads LLM + FAISS index
  Both are singletons. Subsequent calls reuse the loaded objects.

Data paths (all relative to repo root):
  Graph (Entities)   : M1/output/entities.json
  Graph (Edges)      : M1/output/relationships.json
  CDR data  : M1/data/cdrs/cdr.csv
  TXN data  : M1/data/transactions/transactions.csv
  M3 flags  : M3/output/pattern_flags.json
  M3 scores : M3/output/centrality_scores.json
"""

import logging
import os
import json
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature flag
# ---------------------------------------------------------------------------
USE_REAL_MODULES = os.getenv("CRIMINAL_USE_REAL_MODULES", "0") == "1"

# ---------------------------------------------------------------------------
# Paths (all relative to repo root — app.py adds repo root to sys.path)
# ---------------------------------------------------------------------------
_ENTITIES_PATH      = "M1/output/entities.json"
_RELATIONSHIPS_PATH = "M1/output/relationships.json"
_CDR_PATH           = "M1/data/cdrs/cdr.csv"
_TXN_PATH           = "M1/data/transactions/transactions.csv"
_M3_FLAGS_PATH      = Path("M3/output/pattern_flags.json")
_M3_CENTRALITY_PATH = Path("M3/output/centrality_scores.json")

# ---------------------------------------------------------------------------
# Module-level singletons (loaded once, reused on every request)
# ---------------------------------------------------------------------------
_CRIMINAL_GRAPH = None      # M2 CriminalGraph object
_M4_PIPELINE    = None      # M4 RAGPipeline object
_INIT_ERROR     = None      # Set if startup loading fails — surfaced to UI


def _load_real_backends():
    """
    Load the graph and M4 pipeline once at startup.
    Sets module-level singletons. Logs any failures but does NOT raise —
    individual call functions check the singletons and fall back to mocks
    if loading failed.
    """
    global _CRIMINAL_GRAPH, _M4_PIPELINE, _INIT_ERROR

    if not USE_REAL_MODULES:
        return

    # --- Load graph from M1 output JSONs ---
    if _CRIMINAL_GRAPH is None:
        try:
            from M2.graph_builder import load_graph_from_m1_output
            _CRIMINAL_GRAPH = load_graph_from_m1_output(_ENTITIES_PATH, _RELATIONSHIPS_PATH)
            logger.info(
                f"[Backend] Graph loaded: "
                f"{_CRIMINAL_GRAPH.number_of_nodes()} nodes, "
                f"{_CRIMINAL_GRAPH.number_of_edges()} edges"
            )
        except Exception as e:
            _INIT_ERROR = f"Graph load failed: {e}"
            logger.error(f"[Backend] {_INIT_ERROR}", exc_info=True)

    # --- Load M4 RAGPipeline ---
    if _M4_PIPELINE is None:
        try:
            from M4.pipeline import RAGPipeline
            from M4.config import DEFAULT_CONFIG
            pipeline = RAGPipeline(DEFAULT_CONFIG)
            # Pass the real graph so M4 can pull graph evidence
            pipeline.load(graph=_CRIMINAL_GRAPH)
            _M4_PIPELINE = pipeline
            logger.info("[Backend] M4 RAGPipeline loaded successfully")
        except Exception as e:
            _INIT_ERROR = (_INIT_ERROR or "") + f" | M4 pipeline load failed: {e}"
            logger.error(f"[Backend] M4 pipeline load failed: {e}", exc_info=True)


# Trigger loading at import time when real modules are enabled
_load_real_backends()


# ---------------------------------------------------------------------------
# M5 — handle_investigator_request
# ---------------------------------------------------------------------------

def call_m5(session_id, request, **kwargs):
    """
    Call M5.handle_investigator_request() with real or mock backends.

    In real-module mode, passes:
      use_mocks=False
      graph=<loaded CriminalGraph>
      m4_pipeline=<loaded RAGPipeline>
      cdr_path=<M1/data/cdrs/cdr.csv>
      txn_path=<M1/data/transactions/transactions.csv>

    Falls back to mock on import error or startup failure.
    Raises on analysis failure — callers must wrap in try/except.
    """
    if USE_REAL_MODULES:
        try:
            from M5 import handle_investigator_request

            # Use real graph and pipeline if they loaded; fall back to mocks if not
            graph      = _CRIMINAL_GRAPH
            m4_pipeline = _M4_PIPELINE
            use_mocks  = (graph is None or m4_pipeline is None)

            if use_mocks:
                logger.warning(
                    "[Backend] Real backends not fully loaded — using M5 built-in mocks. "
                    f"Init error: {_INIT_ERROR}"
                )

            return handle_investigator_request(
                session_id,
                request,
                use_mocks=use_mocks,
                graph=graph,
                m4_pipeline=m4_pipeline,
                cdr_path=_CDR_PATH,
                txn_path=_TXN_PATH,
                **kwargs,
            )
        except ImportError as e:
            logger.warning(f"[Backend] M5 import failed, falling back to mock: {e}")
        except Exception as e:
            logger.error(f"[Backend] M5 call failed: {e}", exc_info=True)
            raise  # Re-raise so the UI shows the error state

    from M6.mocks.mock_backends import mock_handle_investigator_request
    return mock_handle_investigator_request(session_id, request, **kwargs)


# ---------------------------------------------------------------------------
# M2 — graph query functions
# ---------------------------------------------------------------------------

def _get_graph():
    """Returns real graph if loaded, else None (callers fall back to mocks)."""
    if USE_REAL_MODULES and _CRIMINAL_GRAPH is not None:
        return _CRIMINAL_GRAPH
    return None


def call_m2_subgraph(graph, case_id: str) -> dict:
    """
    Returns dict with "nodes" and "edges" lists for the given case.
    Each node: {id, label, type, confidence}
    Each edge: {source, target, type, confidence, weight, timestamp}

    Uses the loaded singleton graph (ignores the `graph` argument when
    USE_REAL_MODULES=True — the singleton is the authoritative source).
    """
    real_graph = _get_graph()
    if real_graph is not None:
        try:
            from M2.query import subgraph
            g = subgraph(real_graph, case_id)

            # If subgraph is empty, fall back to full graph for display
            if g.number_of_nodes() == 0:
                logger.info(
                    f"[Backend] subgraph('{case_id}') returned empty — "
                    "displaying full graph for visualization"
                )
                g = real_graph

            nodes = [
                {
                    "id": n,
                    "label": data.get("name", n),
                    "type": data.get("type", "UNKNOWN"),
                    "confidence": data.get("confidence", 1.0),
                }
                for n, data in g.nodes(data=True)
            ]
            edges = [
                {
                    "source": u,
                    "target": v,
                    "type": data.get("relationship", ""),
                    "confidence": data.get("confidence", 1.0),
                    "weight": data.get("weight", 1.0),
                    "timestamp": data.get("timestamp", ""),
                    "source_record": data.get("source_record", "Unknown"),
                }
                for u, v, data in g.edges(data=True)
            ]
            return {"nodes": nodes, "edges": edges}
        except Exception as e:
            logger.warning(f"[Backend] M2 subgraph failed, falling back to mock: {e}")

    from M6.mocks.mock_backends import mock_subgraph
    return mock_subgraph(graph, case_id)


def call_m2_neighbors(graph, entity_id: str) -> list[dict]:
    """Returns list of neighbour dicts for entity_id."""
    real_graph = _get_graph()
    if real_graph is not None:
        try:
            from M2.query import neighbors
            return neighbors(real_graph, entity_id)
        except Exception as e:
            logger.warning(f"[Backend] M2 neighbors failed, falling back to mock: {e}")

    from M6.mocks.mock_backends import mock_neighbors
    return mock_neighbors(graph, entity_id)


def call_m2_search(graph, query: str, threshold: float = 60.0) -> list[dict]:
    """
    Returns fuzzy-matched entity dicts for query string.

    Note: real M2.search_by_name() uses threshold in [0.0, 1.0].
    The UI passes threshold as a 0–100 percentage (legacy mock convention);
    this function normalises it automatically.
    """
    real_graph = _get_graph()
    if real_graph is not None:
        try:
            from M2.query import search_by_name
            # Normalise threshold: if caller passes >1, assume it's 0-100 scale
            normalised_threshold = threshold / 100.0 if threshold > 1.0 else threshold
            return search_by_name(real_graph, query, threshold=normalised_threshold)
        except Exception as e:
            logger.warning(f"[Backend] M2 search failed, falling back to mock: {e}")

    from M6.mocks.mock_backends import mock_search_by_name
    return mock_search_by_name(graph, query, threshold)


# ---------------------------------------------------------------------------
# M3 — key players / centrality (reads static JSON files M3 produced)
# ---------------------------------------------------------------------------

def call_m3_key_players() -> list[dict]:
    """
    Returns sorted list of pattern flag dicts for the key-players view.
    Each item: {entity_id, name, flags, priority_score, evidence}
    Sorted descending by priority_score.

    Reads M3/output/pattern_flags.json directly — M3 writes this file
    as part of its pipeline run and it is stable between requests.
    """
    if USE_REAL_MODULES:
        try:
            if _M3_FLAGS_PATH.exists():
                flags = json.loads(_M3_FLAGS_PATH.read_text(encoding="utf-8"))

                # Optionally enrich with entity names from centrality file
                if _M3_CENTRALITY_PATH.exists():
                    centrality = json.loads(
                        _M3_CENTRALITY_PATH.read_text(encoding="utf-8")
                    )
                    # centrality is a dict: {entity_id: {degree, betweenness, ...}}
                    # pattern_flags already has entity_id and usually has name;
                    # fill in name from centrality if missing.
                    name_map = {
                        eid: info.get("name", eid)
                        for eid, info in centrality.items()
                        if isinstance(info, dict)
                    }
                    for flag in flags:
                        if not flag.get("name") and flag.get("entity_id") in name_map:
                            flag["name"] = name_map[flag["entity_id"]]

                return sorted(
                    flags,
                    key=lambda x: x.get("priority_score", 0),
                    reverse=True,
                )
            else:
                logger.warning(
                    f"[Backend] M3 pattern_flags.json not found at {_M3_FLAGS_PATH} "
                    "— falling back to mock"
                )
        except Exception as e:
            logger.warning(f"[Backend] M3 key_players failed, falling back to mock: {e}")

    from M6.mocks.mock_backends import mock_get_key_players
    return mock_get_key_players()


# ---------------------------------------------------------------------------
# Health check — used by the sidebar to show status
# ---------------------------------------------------------------------------

def get_backend_status() -> dict:
    """Returns a status dict for the sidebar health display."""
    return {
        "use_real_modules": USE_REAL_MODULES,
        "graph_loaded": _CRIMINAL_GRAPH is not None,
        "graph_nodes": _CRIMINAL_GRAPH.number_of_nodes() if _CRIMINAL_GRAPH else 0,
        "graph_edges": _CRIMINAL_GRAPH.number_of_edges() if _CRIMINAL_GRAPH else 0,
        "pipeline_loaded": _M4_PIPELINE is not None,
        "m3_flags_available": _M3_FLAGS_PATH.exists(),
        "init_error": _INIT_ERROR,
    }
