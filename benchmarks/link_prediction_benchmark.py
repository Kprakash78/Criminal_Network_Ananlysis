#!/usr/bin/env python3
"""
benchmarks/link_prediction_benchmark.py
PS 26152 — Criminal Network Analysis System
--------------------------------------------
Evaluates link-prediction quality on synthetic transaction networks.

Method (held-out evaluation on FULL edge set):
  - Extract ALL directed edges from transaction CSV (not just known_edges).
  - Hold out 40% of edges as test set (deterministic via sorted+stride selection).
  - Build a RESIDUAL graph from remaining 60% of edges.
  - Score held-out positive pairs + random negative pairs against the residual.
  - Heuristic: common_neighbours + degree + reverse_edge_signal.

AUC > 0.5 means the heuristic successfully discriminates held-out positive
pairs (which share common neighbours via the residual) from random negatives
(which have no graph context).

Metrics:
  - AUC-ROC: computed manually (no sklearn required).
  - Precision@5: held-out positives in top-5 scored unobserved pairs.

Fully offline. Deterministic with SEED=42.

Usage:
    python benchmarks/link_prediction_benchmark.py
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT   = Path(__file__).resolve().parent.parent
DEMO_DIR    = REPO_ROOT / "demo_dataset"
RESULTS_DIR = REPO_ROOT / "results"
SEED = 42
np.random.seed(SEED)


def load_txn(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_graph(rows: list[dict]) -> tuple[dict, dict]:
    """
    Build undirected adjacency list and directed edge-weight map.
    adj[node] = set of ALL connected nodes (undirected for common-neighbour).
    ew[(src, dst)] = total_amount for directed scoring.
    """
    adj: dict = defaultdict(set)
    ew:  dict = defaultdict(float)
    for row in rows:
        src = row.get("sender_account", "").strip()
        dst = row.get("receiver_account", "").strip()
        amt = float(row.get("amount_inr", 0))
        if src and dst and src != dst:
            adj[src].add(dst)
            adj[dst].add(src)   # undirected for common-neighbour
            ew[(src, dst)] += amt
    return dict(adj), dict(ew)


def score_edge(src: str, dst: str, adj: dict, ew: dict) -> float:
    """
    Heuristic score for a candidate edge (src, dst) on the RESIDUAL graph.
    High score = heuristic believes this edge is likely to exist.

    Components:
      1. common_neighbours (Jaccard-scaled)
      2. reverse edge signal (dst→src weight in residual)
      3. presence bonus (both endpoints are active in residual graph)
    """
    nbrs_src = adj.get(src, set())
    nbrs_dst = adj.get(dst, set())

    union = len(nbrs_src | nbrs_dst)
    jaccard = len(nbrs_src & nbrs_dst) / union if union > 0 else 0.0

    rev_w = ew.get((dst, src), 0.0)
    fwd_w = ew.get((src, dst), 0.0)
    edge_signal = float(np.log1p((rev_w + fwd_w) / 50_000))

    presence = 1.0 if (src in adj and dst in adj) else 0.0

    return round(jaccard * 0.5 + edge_signal * 0.3 + presence * 0.2, 6)


def manual_auc(scores_pos: list[float], scores_neg: list[float]) -> float:
    """AUC-ROC via brute-force pairwise comparison. Ties count 0.5."""
    total = len(scores_pos) * len(scores_neg)
    if total == 0:
        return 0.5
    wins = sum(
        1.0 if p > n else (0.5 if p == n else 0.0)
        for p in scores_pos
        for n in scores_neg
    )
    return round(wins / total, 4)


def evaluate_case(case_id: str, gt: dict) -> dict:
    txn_file = REPO_ROOT / gt["txn_file"]
    txn_rows = load_txn(txn_file)
    if not txn_rows:
        return {"case_id": case_id, "error": "no_txn_data"}

    # All unique directed edges (deduplicated by account pair)
    seen_pairs: dict = {}
    for row in txn_rows:
        src = row.get("sender_account", "").strip()
        dst = row.get("receiver_account", "").strip()
        if src and dst and src != dst:
            key = (src, dst)
            seen_pairs[key] = seen_pairs.get(key, 0) + 1

    all_edges = sorted(seen_pairs.keys())  # deterministic order
    if len(all_edges) < 2:
        return {"case_id": case_id, "error": "insufficient_edges"}

    # Collect all nodes
    all_nodes = set()
    for s, d in all_edges:
        all_nodes.add(s)
        all_nodes.add(d)
    nodes_list = sorted(all_nodes)

    # Held-out split: 40% hold-out (every 5th edge starting at 0)
    held_out_indices = list(range(0, len(all_edges), int(max(1, len(all_edges) // max(1, int(len(all_edges)*0.4))))))
    held_out_pos = [all_edges[i] for i in held_out_indices if i < len(all_edges)]
    held_out_set = set(held_out_pos)
    train_edges  = [e for e in all_edges if e not in held_out_set]

    # Residual graph (train set only)
    residual_rows = [
        row for row in txn_rows
        if (row.get("sender_account", "").strip(),
            row.get("receiver_account", "").strip()) not in held_out_set
    ]
    adj_res, ew_res = build_graph(residual_rows)

    all_observed = set(all_edges)  # All observed edges (for negative sampling)

    # Negative sampling — pairs not in ANY observed edge
    rng = np.random.default_rng(SEED + abs(hash(case_id)) % 997)
    negative_edges = []
    target_negs = min(len(held_out_pos) * 8, 80)
    tries = 0
    while len(negative_edges) < target_negs and tries < 1000:
        tries += 1
        i = int(rng.integers(0, len(nodes_list)))
        j = int(rng.integers(0, len(nodes_list)))
        pair = (nodes_list[i], nodes_list[j])
        if pair[0] != pair[1] and pair not in all_observed:
            negative_edges.append(pair)

    # Score positive held-out edges and negative edges on RESIDUAL
    scores_pos = [score_edge(s, d, adj_res, ew_res) for s, d in held_out_pos]
    scores_neg = [score_edge(s, d, adj_res, ew_res) for s, d in negative_edges]

    auc = manual_auc(scores_pos, scores_neg)

    # Precision@5: top-5 unobserved pairs by heuristic score
    res_observed = set(ew_res.keys())
    candidates = []
    for src in nodes_list:
        for dst in nodes_list:
            pair = (src, dst)
            if src != dst and pair not in res_observed:
                sc = score_edge(src, dst, adj_res, ew_res)
                candidates.append((pair, sc))
    candidates.sort(key=lambda x: x[1], reverse=True)
    top5 = [pair for pair, _ in candidates[:5]]
    p_at_5 = round(sum(1 for e in top5 if e in held_out_set) / 5, 4)

    return {
        "case_id": case_id,
        "n_total_edges": len(all_edges),
        "n_held_out_positives": len(held_out_pos),
        "n_train_edges": len(train_edges),
        "n_negative_edges": len(negative_edges),
        "auc_roc": auc,
        "precision_at_5": p_at_5,
        "top5_predicted": [list(e) for e in top5],
    }


def run_link_prediction_benchmark() -> dict:
    gt_path = DEMO_DIR / "ground_truth.json"
    if not gt_path.exists():
        print("ERROR: demo_dataset/ground_truth.json not found.")
        sys.exit(1)

    gt_all = json.loads(gt_path.read_text())
    case_results = []

    for case_id, gt in gt_all.items():
        result = evaluate_case(case_id, gt)
        case_results.append(result)
        if "error" in result:
            print(f"  {case_id}: {result['error']}")
        else:
            print(f"  {case_id}: AUC={result['auc_roc']:.3f}  "
                  f"P@5={result['precision_at_5']:.3f}  "
                  f"({result['n_held_out_positives']} held-out / "
                  f"{result['n_total_edges']} total edges)")

    valid = [r for r in case_results if "error" not in r]
    if valid:
        avg_auc   = round(sum(r["auc_roc"]         for r in valid) / len(valid), 4)
        avg_p_at5 = round(sum(r["precision_at_5"]  for r in valid) / len(valid), 4)
    else:
        avg_auc = avg_p_at5 = 0.0

    return {
        "benchmark": "link_prediction",
        "n_cases": len(case_results),
        "per_case": case_results,
        "aggregate": {"mean_auc_roc": avg_auc, "mean_precision_at_5": avg_p_at5},
    }


if __name__ == "__main__":
    print("=== Link-Prediction Benchmark ===")
    result = run_link_prediction_benchmark()
    print(f"\n--- Aggregate ---")
    print(f"  Mean AUC-ROC:     {result['aggregate']['mean_auc_roc']:.3f}")
    print(f"  Mean Precision@5: {result['aggregate']['mean_precision_at_5']:.3f}")
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "link_prediction_results.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"\n✓ Results saved: {out}")
