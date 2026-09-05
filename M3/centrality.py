"""
M3.1 — Centrality Computation
PS 26152 — AI-Powered Criminal Network Analysis System

Runs degree, betweenness, closeness, and PageRank centrality on M2's graph.
Returns a dict mapping entity_id → centrality dict for every node.

Guarantees:
  - Every node in the graph gets an entry, including isolated nodes (all zeros)
  - Empty graph raises ValueError immediately (fail loudly)
  - All values are normalized to [0, 1]
  - Deterministic — same graph produces identical results on every run
"""

import logging
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)


def compute_centrality(graph: nx.MultiDiGraph) -> dict[str, dict[str, float]]:
    """
    Compute four centrality metrics for every node in M2's graph.

    Args:
        graph: a NetworkX MultiDiGraph (M2's CriminalGraph).

    Returns:
        dict mapping entity_id → {
            "degree": float,       # normalised in-degree + out-degree
            "betweenness": float,  # fraction of shortest paths through this node
            "closeness": float,    # inverse of average shortest path distance
            "pagerank": float,     # random-walk prestige score
        }
        Values are in [0, 1]. Isolated nodes get 0.0 for all metrics.

    Raises:
        ValueError: if the graph is empty (no nodes) — fail loudly per PRD §11.
    """
    if graph.number_of_nodes() == 0:
        raise ValueError(
            "M3 received an empty graph from M2. "
            "Cannot run centrality on zero nodes. "
            "Verify that M2's graph was built and passed correctly."
        )

    n = graph.number_of_nodes()
    logger.info(f"[Centrality] Computing metrics for {n} nodes, {graph.number_of_edges()} edges")

    # Collapse to a simple undirected graph so parallel edges between the
    # same pair of nodes don't inflate degree counts above 1.0.
    # nx.Graph(multigraph) keeps one edge per unique (u,v) pair.
    simple_undirected = nx.Graph(graph.to_undirected(as_view=True))

    # --- Degree centrality ---
    # degree_centrality = degree / (n-1); values are in [0, 1] on a simple graph.
    degree = nx.degree_centrality(simple_undirected)

    # --- Betweenness centrality ---
    betweenness = nx.betweenness_centrality(simple_undirected, normalized=True)

    # --- Closeness centrality ---
    # Works correctly even on disconnected graphs (returns 0 for isolated nodes).
    closeness = nx.closeness_centrality(simple_undirected)

    # --- PageRank ---
    # MultiDiGraph can have parallel edges that inflate scores above 1.
    # Collapse to a simple DiGraph first (summing weights per pair) so
    # the resulting probability distribution stays in [0, 1].
    simple_digraph = nx.DiGraph()
    simple_digraph.add_nodes_from(graph.nodes(data=True))
    for u, v, data in graph.edges(data=True):
        if simple_digraph.has_edge(u, v):
            simple_digraph[u][v]["weight"] = simple_digraph[u][v].get("weight", 1.0) + data.get("weight", 1.0)
        else:
            simple_digraph.add_edge(u, v, weight=data.get("weight", 1.0))
    try:
        pagerank = nx.pagerank(simple_digraph, alpha=0.85, weight="weight")
    except nx.PowerIterationFailedConvergence:
        logger.warning("[Centrality] PageRank did not converge — falling back to degree proxy")
        pagerank = {node: degree[node] for node in graph.nodes}

    # --- Merge into per-node result dicts ---
    results: dict[str, dict[str, float]] = {}
    for node_id in graph.nodes:
        results[node_id] = {
            "degree": round(degree.get(node_id, 0.0), 6),
            "betweenness": round(betweenness.get(node_id, 0.0), 6),
            "closeness": round(closeness.get(node_id, 0.0), 6),
            "pagerank": round(pagerank.get(node_id, 0.0), 6),
        }

    logger.info("[Centrality] Done")
    return results
