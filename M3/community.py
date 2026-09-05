"""
M3.2 — Community Detection
PS 26152 — AI-Powered Criminal Network Analysis System

Assigns each node in M2's graph a community/cluster ID using greedy modularity
optimisation (a standard, deterministic community-detection algorithm available
in NetworkX without additional dependencies).

Guarantees:
  - Every node gets a community ID (integer ≥ 0), including isolated nodes
  - Empty graph raises ValueError immediately
  - Deterministic — same graph produces the same community assignments
  - The simple undirected graph is used (parallel edges collapsed), consistent
    with the centrality module's treatment of the MultiDiGraph
"""

import logging

import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities

logger = logging.getLogger(__name__)


def detect_communities(graph: nx.MultiDiGraph) -> dict[str, int]:
    """
    Detect communities/clusters in M2's graph.

    Uses greedy modularity optimisation (Clauset-Newman-Moore), which is:
      - Deterministic (no random seeds needed)
      - O(n log² n) — fast enough at demo scale
      - Available in NetworkX without extra dependencies

    Args:
        graph: a NetworkX MultiDiGraph (M2's CriminalGraph).

    Returns:
        dict mapping entity_id → community_id (int ≥ 0).
        Community IDs are assigned in order of community size, largest first,
        so community 0 is always the biggest cluster.

    Raises:
        ValueError: if the graph is empty (no nodes).
    """
    if graph.number_of_nodes() == 0:
        raise ValueError(
            "M3 received an empty graph. Cannot run community detection on zero nodes."
        )

    # Collapse to a simple undirected graph — same reasoning as centrality module.
    simple = nx.Graph(graph.to_undirected(as_view=True))

    n = simple.number_of_nodes()
    logger.info(f"[Community] Detecting communities in {n}-node graph")

    # greedy_modularity_communities returns a list of frozensets, each being one
    # community, ordered by decreasing size.
    communities: list[frozenset] = list(greedy_modularity_communities(simple))

    # Build entity_id → community_id mapping.
    community_map: dict[str, int] = {}
    for community_id, community_set in enumerate(communities):
        for node_id in community_set:
            community_map[node_id] = community_id

    # Sanity: every node must have been assigned.
    for node_id in graph.nodes:
        if node_id not in community_map:
            # Should not happen, but handle defensively.
            logger.warning(
                f"[Community] Node '{node_id}' was not assigned a community — "
                "assigning to a singleton group."
            )
            community_map[node_id] = len(communities)

    logger.info(
        f"[Community] Found {len(communities)} communities "
        f"(largest: {len(communities[0])} nodes)"
    )
    return community_map
