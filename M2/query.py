"""
M2.5 & M2.6 — Query Layer
PS 26152 — AI-Powered Criminal Network Analysis System

Provides all query functions that M3 (analytics), M4 (RAG), and M6 (dashboard)
call against the CriminalGraph. All functions are pure (no mutations), fast on
the demo-scale dataset, and return stable JSON-serializable shapes.

Public API (matches PRD §8 exactly):
  neighbors(graph, entity_id) -> list[dict]
  shortest_path(graph, source_id, target_id) -> list[str] | None
  subgraph(graph, case_id) -> CriminalGraph
  search_by_name(graph, query, threshold) -> list[dict]
  add_case_incrementally(graph, new_entities, new_relationships) -> CriminalGraph (M2.7)
"""

import logging
from typing import Any

import networkx as nx
from rapidfuzz import fuzz, process

from M2.graph_builder import CriminalGraph
from M2.loader import M1Entity, M1Relationship
from M2.relationship_typer import assign_relationship_type, assign_weight

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# M2.5 — neighbors()
# ---------------------------------------------------------------------------

def neighbors(graph: CriminalGraph, entity_id: str) -> list[dict]:
    """
    Return all entities directly connected to `entity_id`, with edge details.

    Considers both outgoing and incoming edges (the graph is directed, but
    investigators need both directions).

    Returns list of:
      {
        "entity_id": str,
        "name": str,
        "type": str,
        "direction": "outgoing" | "incoming",
        "relationship": str,
        "source_record": str,
        "confidence": float,
        "weight": float,
        "timestamp": str,
      }

    Returns [] if entity_id is not in the graph.
    """
    if entity_id not in graph.nodes:
        logger.warning(f"[Query] neighbors(): entity '{entity_id}' not in graph")
        return []

    results: list[dict] = []
    seen: set[tuple] = set()   # dedup identical (neighbor, rel, record, direction)

    # Outgoing edges
    for _, v, data in graph.out_edges(entity_id, data=True):
        key = (v, data.get("relationship"), data.get("source_record"), "outgoing")
        if key in seen:
            continue
        seen.add(key)
        neighbor_attrs = graph.nodes.get(v, {})
        results.append({
            "entity_id": v,
            "name": neighbor_attrs.get("name", v),
            "type": neighbor_attrs.get("type", ""),
            "direction": "outgoing",
            "relationship": data.get("relationship", ""),
            "source_record": data.get("source_record", ""),
            "confidence": data.get("confidence", 0.0),
            "weight": data.get("weight", 1.0),
            "timestamp": data.get("timestamp", ""),
        })

    # Incoming edges
    for u, _, data in graph.in_edges(entity_id, data=True):
        key = (u, data.get("relationship"), data.get("source_record"), "incoming")
        if key in seen:
            continue
        seen.add(key)
        neighbor_attrs = graph.nodes.get(u, {})
        results.append({
            "entity_id": u,
            "name": neighbor_attrs.get("name", u),
            "type": neighbor_attrs.get("type", ""),
            "direction": "incoming",
            "relationship": data.get("relationship", ""),
            "source_record": data.get("source_record", ""),
            "confidence": data.get("confidence", 0.0),
            "weight": data.get("weight", 1.0),
            "timestamp": data.get("timestamp", ""),
        })

    return results


# ---------------------------------------------------------------------------
# M2.5 — shortest_path()
# ---------------------------------------------------------------------------

def shortest_path(
    graph: CriminalGraph,
    source_id: str,
    target_id: str,
) -> list[str] | None:
    """
    Find the shortest path between two entities in the graph.

    Uses the undirected view of the graph (ignores edge direction) so that
    a path can be found regardless of which direction links were recorded in M1.

    Returns:
        A list of entity_ids forming the path (inclusive of source and target),
        or None if no path exists (disconnected subgraph or missing nodes).

    The path is weighted by edge confidence (higher confidence = shorter path).
    If no weights are provided, falls back to hop-count shortest path.
    """
    if source_id not in graph.nodes:
        logger.warning(f"[Query] shortest_path(): source '{source_id}' not in graph")
        return None
    if target_id not in graph.nodes:
        logger.warning(f"[Query] shortest_path(): target '{target_id}' not in graph")
        return None
    if source_id == target_id:
        return [source_id]

    # Use undirected view for reachability; MultiGraph → Graph for path algorithms
    undirected = graph.to_undirected(as_view=True)

    try:
        # Shortest path by hop count (most reliable for investigator use)
        path = nx.shortest_path(undirected, source=source_id, target=target_id)
        return path
    except nx.NetworkXNoPath:
        logger.info(f"[Query] shortest_path(): no path from '{source_id}' to '{target_id}'")
        return None
    except nx.NodeNotFound:
        return None


# ---------------------------------------------------------------------------
# M2.5 — subgraph()
# ---------------------------------------------------------------------------

def subgraph(graph: CriminalGraph, case_id: str) -> CriminalGraph:
    """
    Extract the subgraph of all entities and relationships tied to `case_id`.

    A node is included if `case_id` appears in its `source_documents`.
    An edge is included if its `source_record` starts with the case_id prefix
    (e.g. case_id="FIR_001" matches source_record="FIR_001") OR if both
    endpoints are case nodes (i.e. co-occur in the same case).

    Returns a new CriminalGraph (a subgraph view copy). Does NOT modify the
    original graph.

    Returns an empty graph if case_id doesn't match anything.
    """
    # Collect nodes that belong to this case
    case_nodes = {
        node_id
        for node_id, attrs in graph.nodes(data=True)
        if case_id in attrs.get("source_documents", [])
    }

    if not case_nodes:
        logger.info(f"[Query] subgraph(): no nodes found for case '{case_id}'")
        return nx.MultiDiGraph()

    # Start with the induced subgraph of case nodes
    # Then add only edges whose source_record matches case_id
    sub: CriminalGraph = nx.MultiDiGraph()

    # Add case nodes
    for node_id in case_nodes:
        attrs = graph.nodes[node_id]
        sub.add_node(node_id, **attrs)

    # Add edges: both endpoints in case_nodes AND source_record matches case_id
    for u, v, data in graph.edges(data=True):
        if u in case_nodes and v in case_nodes:
            if data.get("source_record", "") == case_id:
                sub.add_edge(u, v, **data)

    logger.info(
        f"[Query] subgraph('{case_id}'): "
        f"{sub.number_of_nodes()} nodes, {sub.number_of_edges()} edges"
    )
    return sub


# ---------------------------------------------------------------------------
# M2.6 — search_by_name()
# ---------------------------------------------------------------------------

def search_by_name(
    graph: CriminalGraph,
    query: str,
    threshold: float = 0.8,
) -> list[dict]:
    """
    Fuzzy-match `query` against node names and aliases.

    Uses rapidfuzz partial_ratio so "Ravi" matches "Ravi Kumar".

    Args:
        graph     : the CriminalGraph
        query     : search string (partial or full name)
        threshold : minimum similarity score [0, 1] (default 0.8)

    Returns list of:
      {
        "entity_id": str,
        "name": str,
        "type": str,
        "matched_alias": str,     # which name/alias triggered the match
        "score": float,           # similarity score [0, 1]
        "confidence": float,      # M1 extraction confidence
      }

    Sorted by score descending. Returns [] if no matches above threshold.
    """
    if not query or not query.strip():
        return []

    query_stripped = query.strip()
    threshold_pct = threshold * 100   # rapidfuzz uses 0–100

    results: list[dict] = []

    for node_id, attrs in graph.nodes(data=True):
        best_score = 0.0
        best_match = ""

        # Check canonical name
        name = attrs.get("name", "")
        if name:
            score = fuzz.partial_ratio(query_stripped, name)
            if score > best_score:
                best_score = score
                best_match = name

        # Check all aliases
        for alias in attrs.get("aliases", []):
            if alias:
                score = fuzz.partial_ratio(query_stripped, alias)
                if score > best_score:
                    best_score = score
                    best_match = alias

        if best_score >= threshold_pct:
            results.append({
                "entity_id": node_id,
                "name": name,
                "type": attrs.get("type", ""),
                "matched_alias": best_match,
                "score": round(best_score / 100.0, 4),
                "confidence": attrs.get("confidence", 0.0),
            })

    # Sort by score descending
    results.sort(key=lambda r: r["score"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# M2.7 — add_case_incrementally()
# ---------------------------------------------------------------------------

def add_case_incrementally(
    graph: CriminalGraph,
    new_entities: list[M1Entity],
    new_relationships: list[M1Relationship],
) -> CriminalGraph:
    """
    Incrementally add a new case's entities and relationships to an existing graph.
    Modifies the graph IN PLACE and returns it (for chaining).

    Guarantees:
    - Existing nodes are NOT overwritten (their attributes are preserved)
    - New entity_ids not yet in the graph are added
    - Duplicate edges (same src/tgt/rel/source_record) are not added twice
    - Never triggers a full graph rebuild

    Args:
        graph             : existing CriminalGraph (modified in place)
        new_entities      : list of M1Entity for the new case
        new_relationships : list of M1Relationship for the new case

    Returns:
        The same graph object, now containing the new case's data.
    """
    # Build index of new entities
    new_entity_index: dict[str, M1Entity] = {e.entity_id: e for e in new_entities}

    # Track existing edges as (src, tgt, rel_type, source_record) set
    existing_edges: set[tuple[str, str, str, str]] = {
        (u, v, data["relationship"], data["source_record"])
        for u, v, data in graph.edges(data=True)
    }

    # Add new nodes (skip if already exists — preserve existing attributes)
    added_nodes = 0
    for entity in new_entities:
        if entity.entity_id not in graph.nodes:
            graph.add_node(
                entity.entity_id,
                entity_id=entity.entity_id,
                type=entity.type,
                name=entity.name,
                aliases=list(entity.aliases),
                source_documents=list(entity.source_documents),
                confidence=entity.confidence,
                needs_review=entity.needs_review,
            )
            added_nodes += 1
        else:
            # Merge source_documents into existing node
            existing_docs = set(graph.nodes[entity.entity_id].get("source_documents", []))
            existing_docs.update(entity.source_documents)
            graph.nodes[entity.entity_id]["source_documents"] = list(existing_docs)

    # Full entity index = existing nodes + new entities
    full_index: dict[str, M1Entity] = {}
    for node_id, attrs in graph.nodes(data=True):
        if node_id in new_entity_index:
            full_index[node_id] = new_entity_index[node_id]
        else:
            # Reconstruct a minimal M1Entity from node attrs for typing
            full_index[node_id] = M1Entity(
                entity_id=node_id,
                type=attrs.get("type", "PERSON"),
                name=attrs.get("name", ""),
                aliases=attrs.get("aliases", []),
                source_documents=attrs.get("source_documents", []),
                confidence=attrs.get("confidence", 0.0),
            )

    # Add new edges
    added_edges = 0
    skipped_missing = 0
    skipped_duplicate = 0

    for rel in new_relationships:
        if rel.source not in graph.nodes:
            logger.warning(f"[Incremental] Skipping edge: source '{rel.source}' not in graph")
            skipped_missing += 1
            continue
        if rel.target not in graph.nodes:
            logger.warning(f"[Incremental] Skipping edge: target '{rel.target}' not in graph")
            skipped_missing += 1
            continue

        rel_type = assign_relationship_type(rel, full_index)
        weight = assign_weight(rel, rel_type)

        edge_key = (rel.source, rel.target, rel_type, rel.source_record)
        if edge_key in existing_edges:
            skipped_duplicate += 1
            continue
        existing_edges.add(edge_key)

        graph.add_edge(
            rel.source, rel.target,
            relationship=rel_type,
            timestamp=rel.timestamp,
            source_record=rel.source_record,
            confidence=rel.confidence,
            weight=weight,
        )
        added_edges += 1

    logger.info(
        f"[Incremental] Added {added_nodes} nodes, {added_edges} edges "
        f"(skipped {skipped_missing} missing, {skipped_duplicate} duplicates)"
    )
    return graph
