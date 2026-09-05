"""
M2.3 — Graph Builder
PS 26152 — AI-Powered Criminal Network Analysis System

Constructs a NetworkX MultiDiGraph from M1 entities and relationships.

Each node = one entity, carrying all M1 attributes.
Each edge = one typed relationship, carrying evidence attributes.

Failure handling:
  - Relationship referencing a missing entity → logged warning, edge skipped
  - Duplicate (src, tgt, rel_type, source_record) → first one wins, no parallel dupes

The graph object is the primary handoff to M3, M4, and M6.
They should NOT modify it — use `copy.deepcopy(graph)` if mutation is needed.
"""

import logging
from typing import Any

import networkx as nx

from M2.loader import M1Entity, M1Relationship, load_m1_output
from M2.relationship_typer import assign_relationship_type, assign_weight

logger = logging.getLogger(__name__)

# Type alias for clarity
CriminalGraph = nx.MultiDiGraph


def build_graph(
    entities: list[M1Entity],
    relationships: list[M1Relationship],
) -> CriminalGraph:
    """
    Build and return a typed criminal network graph.

    Nodes   : one per entity_id, carrying type/name/aliases/confidence/source_documents
    Edges   : one per (source, target, relationship_type, source_record) unique tuple.
              If the same tuple appears twice, the first occurrence is kept.

    Args:
        entities      : validated M1Entity list (from M2 loader)
        relationships : validated M1Relationship list (from M2 loader)

    Returns:
        A networkx.MultiDiGraph populated with nodes and typed edges.
    """
    graph: CriminalGraph = nx.MultiDiGraph()

    # ---- Build entity index and add nodes ----
    entity_index: dict[str, M1Entity] = {}
    for entity in entities:
        entity_index[entity.entity_id] = entity
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

    logger.info(f"[GraphBuilder] Added {graph.number_of_nodes()} nodes")

    # ---- Add typed edges, deduplicating identical (src, tgt, rel_type, source_record) ----
    seen_edges: set[tuple[str, str, str, str]] = set()
    skipped_missing = 0
    skipped_duplicate = 0

    for rel in relationships:
        # Validate both endpoints exist
        if rel.source not in entity_index:
            logger.warning(
                f"[GraphBuilder] Skipping edge: source '{rel.source}' not in entity list"
            )
            skipped_missing += 1
            continue
        if rel.target not in entity_index:
            logger.warning(
                f"[GraphBuilder] Skipping edge: target '{rel.target}' not in entity list"
            )
            skipped_missing += 1
            continue

        rel_type = assign_relationship_type(rel, entity_index)
        weight = assign_weight(rel, rel_type)

        # Deduplicate identical edges
        edge_key = (rel.source, rel.target, rel_type, rel.source_record)
        if edge_key in seen_edges:
            skipped_duplicate += 1
            continue
        seen_edges.add(edge_key)

        graph.add_edge(
            rel.source,
            rel.target,
            relationship=rel_type,
            timestamp=rel.timestamp,
            source_record=rel.source_record,
            confidence=rel.confidence,
            weight=weight,
        )

    logger.info(
        f"[GraphBuilder] Added {graph.number_of_edges()} edges "
        f"(skipped {skipped_missing} missing-entity, {skipped_duplicate} duplicate)"
    )

    return graph


def load_graph_from_m1_output(
    entities_path: str,
    relationships_path: str,
) -> CriminalGraph:
    """
    Full pipeline shortcut: load M1 files → build typed graph.
    This is the primary public API entry point for other modules.

    Returns a ready-to-use CriminalGraph (networkx.MultiDiGraph).
    """
    entities, relationships = load_m1_output(entities_path, relationships_path)
    return build_graph(entities, relationships)
