"""
M3.7 — Priority Scorer
PS 26152 — AI-Powered Criminal Network Analysis System

Combines centrality scores + pattern-flag signals into a single interpretable
investigation-priority score per entity, in [0, 1].

Formula (all weights defined in M3Config):
  priority_score = (
      w_degree       × degree_centrality
    + w_betweenness  × betweenness_centrality
    + w_pagerank     × pagerank
    + w_flags        × clamp(flag_count / FLAG_NORM)
  )

FLAG_NORM: the denominator used to normalise the flag count.
  Default = 3 (so 3+ flags gives full weight on the flag component).
  Clamped to [0, 1] so the whole score stays in [0, 1].

Every score above zero gets an evidence string listing the specific
centrality values and flag types that contributed to it.

Language: priority scores indicate investigator-review priority only.
No score implies guilt, involvement, or criminal determination.
"""

import logging
from typing import Any

import networkx as nx

from M3.config import M3Config, DEFAULT_CONFIG
from M3.models import PatternFlag, VALID_FLAG_TYPES

logger = logging.getLogger(__name__)

# Normalisation constant for the flag component.
# An entity with FLAG_NORM or more distinct flags gets the maximum flag contribution.
FLAG_NORM = 3


def compute_priority_score(
    entity_id: str,
    centrality: dict[str, float],
    flag_pairs: list[tuple[str, str]],
    config: M3Config = DEFAULT_CONFIG,
) -> tuple[float, list[str]]:
    """
    Compute the priority score and collect all evidence strings for one entity.

    Args:
        entity_id:    the entity being scored
        centrality:   its centrality dict from M3.1 (keys: degree, betweenness,
                      closeness, pagerank)
        flag_pairs:   list of (flag_type, evidence_string) from M3.3–M3.6
        config:       M3Config holding scoring weights

    Returns:
        (priority_score: float in [0,1], evidence: list[str])
    """
    degree = centrality.get("degree", 0.0)
    betweenness = centrality.get("betweenness", 0.0)
    pagerank = centrality.get("pagerank", 0.0)

    flag_count = len(flag_pairs)
    flag_component = min(flag_count / FLAG_NORM, 1.0)

    score = (
        config.weight_degree_centrality * degree
        + config.weight_betweenness_centrality * betweenness
        + config.weight_pagerank * pagerank
        + config.weight_flag_count * flag_component
    )
    score = round(min(max(score, 0.0), 1.0), 4)

    evidence: list[str] = []

    # Centrality contributions
    if degree > 0:
        evidence.append(
            f"Degree centrality {degree:.3f} — entity connects to "
            f"{degree * 100:.1f}% of the network relative to maximum possible"
        )
    if betweenness > 0:
        evidence.append(
            f"Betweenness centrality {betweenness:.3f} — entity lies on "
            f"{betweenness * 100:.1f}% of shortest paths between other entities"
        )
    if pagerank > 0:
        evidence.append(
            f"PageRank {pagerank:.4f} — entity receives elevated link-weight "
            f"from well-connected neighbours"
        )

    # Flag contributions
    for flag_type, flag_evidence in flag_pairs:
        evidence.append(f"[{flag_type}] {flag_evidence}")

    if not evidence:
        evidence.append(
            f"Priority score {score:.4f} based on centrality metrics "
            f"(no pattern flags detected)"
        )

    return score, evidence


def detect_dense_cluster_membership(
    graph: nx.MultiDiGraph,
    community_map: dict[str, int],
    config: M3Config = DEFAULT_CONFIG,
) -> dict[str, tuple[str, str]]:
    """
    Flag nodes that belong to a dense internal cluster.

    A node is flagged if the fraction of its simple-undirected neighbours
    that are IN THE SAME COMMUNITY exceeds dense_cluster_degree_ratio.
    Requires the node to have at least 2 neighbours (trivially dense
    singletons/pairs are excluded).

    Returns: entity_id → (flag_type, evidence_string)
    """
    simple = nx.Graph(graph.to_undirected(as_view=True))
    flags: dict[str, tuple[str, str]] = {}

    for node_id in simple.nodes:
        community_id = community_map.get(node_id, -1)
        neighbours = list(simple.neighbors(node_id))

        if len(neighbours) < 2:
            continue  # Too small to assess density

        same_community = [n for n in neighbours if community_map.get(n) == community_id]
        ratio = len(same_community) / len(neighbours)

        if ratio >= config.dense_cluster_degree_ratio:
            evidence = (
                f"{len(same_community)}/{len(neighbours)} of this entity's direct "
                f"connections belong to the same cluster (cluster {community_id}), "
                f"ratio {ratio:.2f} ≥ threshold {config.dense_cluster_degree_ratio:.2f}"
            )
            flags[node_id] = ("DENSE_CLUSTER_MEMBERSHIP", evidence)

    logger.info(f"[Scorer] DENSE_CLUSTER_MEMBERSHIP: {len(flags)} entities flagged")
    return flags


def build_pattern_flags(
    graph: nx.MultiDiGraph,
    centrality_scores: dict[str, dict[str, float]],
    community_map: dict[str, int],
    raw_flags: dict[str, list[tuple[str, str]]],
    config: M3Config = DEFAULT_CONFIG,
) -> list[PatternFlag]:
    """
    Assemble the final list of PatternFlag records.

    Only entities with at least one flag OR a priority score above the bare
    centrality baseline are included in the output (M4 doesn't need every node).

    Args:
        graph:             M2's CriminalGraph
        centrality_scores: output of M3.1 compute_centrality()
        community_map:     output of M3.2 detect_communities()
        raw_flags:         output of run_pattern_rules() — entity_id → [(type, evidence)]
        config:            M3Config

    Returns:
        Sorted list of PatternFlag (descending priority_score).
    """
    # Add DENSE_CLUSTER_MEMBERSHIP flags
    dense_flags = detect_dense_cluster_membership(graph, community_map, config)
    all_flags: dict[str, list[tuple[str, str]]] = {}
    for eid, pairs in raw_flags.items():
        all_flags[eid] = list(pairs)
    for eid, flag_pair in dense_flags.items():
        all_flags.setdefault(eid, []).append(flag_pair)

    results: list[PatternFlag] = []

    for entity_id in graph.nodes:
        centrality = centrality_scores.get(entity_id, {
            "degree": 0.0, "betweenness": 0.0, "closeness": 0.0, "pagerank": 0.0
        })
        flag_pairs = all_flags.get(entity_id, [])

        score, evidence = compute_priority_score(entity_id, centrality, flag_pairs, config)

        # Only emit records for entities that have a flag OR meaningful centrality
        has_flags = len(flag_pairs) > 0
        has_meaningful_centrality = centrality.get("degree", 0.0) > 0

        if not (has_flags or has_meaningful_centrality):
            continue

        # Deduplicate flag types (keep all evidence strings)
        seen_types: set[str] = set()
        unique_flags: list[str] = []
        for ft, _ in flag_pairs:
            if ft not in seen_types and ft in VALID_FLAG_TYPES:
                unique_flags.append(ft)
                seen_types.add(ft)

        results.append(PatternFlag(
            entity_id=entity_id,
            degree_centrality=centrality.get("degree", 0.0),
            betweenness_centrality=centrality.get("betweenness", 0.0),
            community=community_map.get(entity_id, -1),
            flags=unique_flags,
            priority_score=score,
            evidence=evidence,
        ))

    results.sort(key=lambda r: r.priority_score, reverse=True)
    logger.info(f"[Scorer] Built {len(results)} PatternFlag records")
    return results
