"""
M3.8 — Output Emitter + Full Analysis Pipeline
PS 26152 — AI-Powered Criminal Network Analysis System

Two public functions:
  run_full_analysis()  — runs the complete M3 pipeline end-to-end
  emit_outputs()       — writes pattern_flags.json and centrality_scores.json

The JSON schemas produced here are consumed by M4 (RAG), M5 (routing),
and M6 (dashboard) without reformatting — do not change field names without
coordinating with those modules.
"""

import json
import logging
from dataclasses import asdict
from pathlib import Path

import networkx as nx

from M3.centrality import compute_centrality
from M3.community import detect_communities
from M3.config import M3Config, DEFAULT_CONFIG
from M3.models import AnalysisResult, PatternFlag
from M3.pattern_rules import run_pattern_rules
from M3.scorer import build_pattern_flags

logger = logging.getLogger(__name__)


def run_full_analysis(
    graph: nx.MultiDiGraph,
    cdr_path: str,
    txn_path: str,
    incident_dates: list[str] | None = None,
    config: M3Config = DEFAULT_CONFIG,
) -> AnalysisResult:
    """
    Run the complete M3 analytics pipeline.

    Order of operations:
      1. compute_centrality  (M3.1)
      2. detect_communities  (M3.2)
      3. run_pattern_rules   (M3.3–M3.6)
      4. build_pattern_flags (M3.7 — scorer + DENSE_CLUSTER flag)

    Args:
        graph:          M2's CriminalGraph (NetworkX MultiDiGraph)
        cdr_path:       absolute path to CDR CSV
        txn_path:       absolute path to transactions CSV
        incident_dates: optional list of YYYY-MM-DD incident date strings
        config:         M3Config (defaults to DEFAULT_CONFIG)

    Returns:
        AnalysisResult with centrality_scores, community_map, and pattern_flags.

    Raises:
        ValueError: if the graph is empty.
    """
    if graph.number_of_nodes() == 0:
        raise ValueError(
            "M3.run_full_analysis received an empty graph. "
            "Verify M2 handed off a populated graph before calling M3."
        )

    logger.info(
        f"[M3] Starting full analysis: "
        f"{graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
    )

    # M3.1 — Centrality
    centrality_scores = compute_centrality(graph)

    # M3.2 — Community detection
    community_map = detect_communities(graph)

    # M3.3–M3.6 — Pattern rules
    raw_flags = run_pattern_rules(
        graph, cdr_path, txn_path, incident_dates, config
    )

    # M3.7 — Priority scoring + DENSE_CLUSTER flag
    pattern_flags = build_pattern_flags(
        graph, centrality_scores, community_map, raw_flags, config
    )

    logger.info(
        f"[M3] Analysis complete: "
        f"{len(pattern_flags)} entities with elevated priority scores, "
        f"{sum(1 for r in pattern_flags if r.flags)} with pattern flags"
    )

    return AnalysisResult(
        centrality_scores=centrality_scores,
        community_map=community_map,
        pattern_flags=pattern_flags,
    )


def emit_outputs(
    result: AnalysisResult,
    output_dir: str,
    pretty: bool = True,
) -> dict[str, str]:
    """
    Write analysis results to JSON files in output_dir.

    Files produced:
      pattern_flags.json     — list of PatternFlag records (flagged entities only)
      centrality_scores.json — dict entity_id → {degree, betweenness, closeness,
                               pagerank, community} for EVERY node (M6 needs all)

    Args:
        result:     AnalysisResult from run_full_analysis()
        output_dir: directory path; created if it doesn't exist
        pretty:     if True, indent JSON for human readability

    Returns:
        dict mapping logical name → absolute file path written.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    indent = 2 if pretty else None

    # ---- pattern_flags.json ----
    flags_data = [
        {
            "entity_id": pf.entity_id,
            "degree_centrality": pf.degree_centrality,
            "betweenness_centrality": pf.betweenness_centrality,
            "community": pf.community,
            "flags": pf.flags,
            "priority_score": pf.priority_score,
            "evidence": pf.evidence,
        }
        for pf in result.pattern_flags
    ]
    flags_path = out_path / "pattern_flags.json"
    flags_path.write_text(json.dumps(flags_data, indent=indent, ensure_ascii=False), encoding="utf-8")
    logger.info(f"[M3] Wrote {len(flags_data)} records to {flags_path}")

    # ---- centrality_scores.json ----
    # Merge community_map into centrality scores so M6 gets everything in one file.
    centrality_data = {}
    for entity_id, metrics in result.centrality_scores.items():
        centrality_data[entity_id] = {
            **metrics,
            "community": result.community_map.get(entity_id, -1),
        }
    centrality_path = out_path / "centrality_scores.json"
    centrality_path.write_text(
        json.dumps(centrality_data, indent=indent, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(f"[M3] Wrote centrality scores for {len(centrality_data)} nodes to {centrality_path}")

    return {
        "pattern_flags": str(flags_path.resolve()),
        "centrality_scores": str(centrality_path.resolve()),
    }
