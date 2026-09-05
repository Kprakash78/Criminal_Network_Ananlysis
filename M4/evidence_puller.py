"""
M4.5 — Graph / Analytics Evidence Puller
PS 26152 — AI-Powered Criminal Network Analysis System

Given a list of entity IDs mentioned in a new case document, pulls:
  1. M3 pattern flags and priority scores for those entities
  2. M2 graph neighbour relationships for those entities
  3. A formatted evidence bundle ready for the prompt builder

This module never recomputes centrality or patterns — it reads M3's
already-computed JSON output and M2's graph object (if available).

If M2's graph is unavailable, it falls back to M3's evidence strings
alone — the pipeline degrades gracefully.
"""

import json
import logging
import re
from pathlib import Path

import networkx as nx

from M4.config import M4Config, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


def load_pattern_flags(
    flags_path: str = DEFAULT_CONFIG.m3_pattern_flags_path,
) -> dict[str, dict]:
    """
    Load M3's pattern_flags.json and index by entity_id.

    Returns:
        dict: entity_id → pattern flag record dict
    """
    p = Path(flags_path)
    if not p.exists():
        logger.warning(f"[EvidencePuller] pattern_flags.json not found at {flags_path}")
        return {}
    records = json.loads(p.read_text(encoding="utf-8"))
    return {r["entity_id"]: r for r in records}


def extract_entity_ids_from_text(
    text: str,
    known_entity_ids: set[str],
) -> list[str]:
    """
    Find which of the known entity IDs appear verbatim in the text.

    This is a simple substring scan — M4 does not re-run NER.
    The graph's entity IDs (hashed IDs like 'ACC_b2210d1b') are unlikely
    to appear in raw FIR text, but entity names (stored in node attributes)
    can be found. This function handles both:
      - Direct entity_id matches (for programmatic calls from M5)
      - Substring matching (for text-based queries)

    Args:
        text:             raw case document text
        known_entity_ids: set of entity IDs from M3's output

    Returns:
        list of matched entity IDs (preserving order, deduplicated)
    """
    found = []
    seen = set()
    text_lower = text.lower()

    for eid in known_entity_ids:
        if eid.lower() in text_lower and eid not in seen:
            found.append(eid)
            seen.add(eid)

    return found


def pull_graph_evidence(
    entity_ids: list[str],
    graph: nx.MultiDiGraph | None = None,
    flags_by_entity: dict[str, dict] | None = None,
    config: M4Config = DEFAULT_CONFIG,
) -> dict:
    """
    Assemble the full evidence bundle for a set of entity IDs.

    Args:
        entity_ids:       IDs to pull evidence for
        graph:            M2's NetworkX MultiDiGraph (optional — degrades gracefully)
        flags_by_entity:  pre-loaded M3 flags dict (if None, loaded from disk)
        config:           M4Config

    Returns:
        dict with keys:
          "entities":     list of per-entity evidence dicts
          "graph_edges":  list of edge evidence strings (from M2, if available)
          "summary":      one-line human-readable count of what was found
    """
    if flags_by_entity is None:
        flags_by_entity = load_pattern_flags(config.m3_pattern_flags_path)

    entity_evidence = []
    graph_edges = []
    graph_edge_ids = []

    for eid in entity_ids:
        flag_record = flags_by_entity.get(eid, {})

        # ---- M3 analytics evidence ----
        entity_info = {
            "entity_id": eid,
            "priority_score": flag_record.get("priority_score", None),
            "flags": flag_record.get("flags", []),
            "m3_evidence": flag_record.get("evidence", []),
            "community": flag_record.get("community", None),
            "degree_centrality": flag_record.get("degree_centrality", None),
        }

        # ---- M2 graph neighbour evidence ----
        if graph is not None and eid in graph:
            neighbours = []
            for _, nbr, edge_data in graph.out_edges(eid, data=True):
                rel = edge_data.get("relationship", "CONNECTED_TO")
                nbr_name = graph.nodes[nbr].get("name", nbr)
                edge_str = f"{eid} --[{rel}]--> {nbr} ({nbr_name})"
                neighbours.append(edge_str)
                edge_id = f"GRAPH_EDGE_{eid}_{nbr}"
                if edge_id not in graph_edge_ids:
                    graph_edges.append(edge_str)
                    graph_edge_ids.append(edge_id)

            # cap at 5 neighbours to keep the prompt manageable
            entity_info["graph_neighbours"] = neighbours[:5]
            entity_info["graph_neighbour_count"] = len(neighbours)
        else:
            entity_info["graph_neighbours"] = []
            entity_info["graph_neighbour_count"] = 0

        entity_evidence.append(entity_info)

    n_flagged = sum(1 for e in entity_evidence if e["flags"])
    n_graph = sum(1 for e in entity_evidence if e["graph_neighbour_count"] > 0)

    summary = (
        f"{len(entity_ids)} entities examined: "
        f"{n_flagged} with pattern flags, "
        f"{n_graph} with graph connections"
    )

    return {
        "entities": entity_evidence,
        "graph_edges": graph_edges[:20],   # cap total edges to keep prompt bounded
        "graph_edge_ids": graph_edge_ids[:20],
        "summary": summary,
    }


def pull_evidence_for_new_case(
    new_case_text: str,
    graph: nx.MultiDiGraph | None = None,
    config: M4Config = DEFAULT_CONFIG,
) -> dict:
    """
    Convenience function: given raw new-case text, automatically extract
    entity IDs and pull graph + M3 evidence.

    Entity extraction here is limited to finding graph node names that
    appear as substrings in the text. This is intentionally simple —
    M1/M2 own the NER; M4 just reads what's in the graph.
    """
    flags_by_entity = load_pattern_flags(config.m3_pattern_flags_path)

    # Extract entity names from the graph and match against the case text
    matched_entity_ids = []

    if graph is not None:
        for node_id, attrs in graph.nodes(data=True):
            name = attrs.get("name", "")
            if name and len(name) > 3 and name.lower() in new_case_text.lower():
                matched_entity_ids.append(node_id)
            # Also check aliases
            for alias in attrs.get("aliases", []):
                if alias and len(alias) > 3 and alias.lower() in new_case_text.lower():
                    if node_id not in matched_entity_ids:
                        matched_entity_ids.append(node_id)

    # Also try direct entity_id substring match (for programmatic calls)
    for eid in flags_by_entity:
        if eid.lower() in new_case_text.lower() and eid not in matched_entity_ids:
            matched_entity_ids.append(eid)

    if not matched_entity_ids:
        logger.info("[EvidencePuller] No entity IDs matched in new case text")

    return pull_graph_evidence(
        matched_entity_ids,
        graph=graph,
        flags_by_entity=flags_by_entity,
        config=config,
    )
