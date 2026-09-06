"""
M3_feature/ghost_node.py
PS 26152 — AI-Powered Criminal Network Analysis System
======================================================
Ghost-Node Finder — Unseen Intermediary Detection.

Runs cheap graph heuristics to suggest possible unobserved intermediaries
between network clusters. Uses three link-score functions:
  - Common Neighbours (CN)
  - Adamic-Adar (AA)  — log-weighted common neighbours
  - Jaccard similarity

For pairs that score above a threshold, emits a human-readable suggestion
with evidence pointers back to source CSV files.

OUTPUT: M3_feature/ghost_suggestions.json
        demo_cache/feature_outputs/ghost_suggestions.json

LANGUAGE: Suggestions are framed as "possible unobserved intermediary"
requiring investigator verification. No guilt determination is made.

Usage (CLI):
    python -m M3_feature.ghost_node suggest case_A
    python -m M3_feature.ghost_node suggest_all

API:
    from M3_feature.ghost_node import suggest_ghosts
    suggestions = suggest_ghosts("case_A", top_k=10)
"""

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT   = Path(__file__).resolve().parent.parent
DEMO_DIR    = REPO_ROOT / "demo_dataset"
OUTPUT_DIR  = Path(__file__).resolve().parent
FEAT_OUT    = REPO_ROOT / "demo_cache" / "feature_outputs"

SEED = 42
np.random.seed(SEED)

SCORE_THRESHOLD = 0.05   # minimum composite score to report a suggestion

PERSONS = [
    {"name": "Ravi Kumar",     "phone": "9876543210", "account": "ACC00101"},
    {"name": "Sunita Sharma",  "phone": "9123456789", "account": "ACC00102"},
    {"name": "Mohammed Iqbal", "phone": "9988776655", "account": "ACC00103"},
    {"name": "Priya Nair",     "phone": "8877665544", "account": "ACC00104"},
    {"name": "Deepak Verma",   "phone": "7766554433", "account": "ACC00105"},
    {"name": "Anjali Singh",   "phone": "9654321098", "account": "ACC00106"},
    {"name": "Rakesh Yadav",   "phone": "9543210987", "account": "ACC00107"},
    {"name": "Fatima Begum",   "phone": "9432109876", "account": "ACC00108"},
    {"name": "Suresh Patil",   "phone": "9321098765", "account": "ACC00109"},
    {"name": "Kavya Reddy",    "phone": "9210987654", "account": "ACC00110"},
]
PHONE_TO_NAME   = {p["phone"]:   p["name"] for p in PERSONS}
ACCOUNT_TO_PHONE = {p["account"]: p["phone"] for p in PERSONS}


# ── Graph construction ────────────────────────────────────────────────────────

def _build_adj(cdr_path: Path, txn_path: Path
               ) -> tuple[dict[str, set], list[str]]:
    """
    Build an undirected adjacency structure from CDR + transactions.
    adj[node] = set of all connected nodes.
    """
    adj: dict = defaultdict(set)

    if cdr_path.exists():
        with open(cdr_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                a = row.get("caller", "").strip()
                b = row.get("receiver", "").strip()
                if a and b and a != b:
                    adj[a].add(b)
                    adj[b].add(a)

    if txn_path.exists():
        with open(txn_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                src = row.get("sender_account", "").strip()
                dst = row.get("receiver_account", "").strip()
                a = ACCOUNT_TO_PHONE.get(src, src)
                b = ACCOUNT_TO_PHONE.get(dst, dst)
                if a and b and a != b:
                    adj[a].add(b)
                    adj[b].add(a)

    nodes = sorted(adj.keys())
    return dict(adj), nodes


# ── Link-score functions ──────────────────────────────────────────────────────

def _common_neighbours(u: str, v: str, adj: dict) -> int:
    return len(adj.get(u, set()) & adj.get(v, set()))


def _adamic_adar(u: str, v: str, adj: dict) -> float:
    score = 0.0
    for w in adj.get(u, set()) & adj.get(v, set()):
        deg_w = len(adj.get(w, set()))
        if deg_w > 1:
            score += 1.0 / math.log(deg_w)
    return round(score, 6)


def _jaccard(u: str, v: str, adj: dict) -> float:
    nu = adj.get(u, set())
    nv = adj.get(v, set())
    union = len(nu | nv)
    if union == 0:
        return 0.0
    return round(len(nu & nv) / union, 6)


# ── Cluster detection (connected components) ──────────────────────────────────

def _connected_components(adj: dict) -> dict[str, int]:
    """Return a mapping node → cluster_id."""
    visited: dict = {}
    cluster_id = 0
    for start in sorted(adj.keys()):
        if start in visited:
            continue
        stack = [start]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited[node] = cluster_id
            stack.extend(adj.get(node, set()) - visited.keys())
        cluster_id += 1
    return visited


# ── Ghost suggestion ──────────────────────────────────────────────────────────

def suggest_ghosts(case_id: str, top_k: int = 10) -> list[dict]:
    """
    Suggest possible unobserved intermediary nodes for `case_id`.

    For every pair (u, v) of nodes in different clusters that is NOT
    already directly connected, compute CN, AA, Jaccard composite score.
    If score > threshold, emit a suggestion.

    Returns list of suggestion dicts (JSON-serialisable), sorted by score.
    """
    np.random.seed(SEED)

    gt_path = DEMO_DIR / "ground_truth.json"
    if not gt_path.exists():
        raise FileNotFoundError("ground_truth.json missing")
    gt = json.loads(gt_path.read_text())[case_id]

    cdr_path = REPO_ROOT / gt["cdr_file"]
    txn_path = REPO_ROOT / gt["txn_file"]

    adj, nodes = _build_adj(cdr_path, txn_path)
    cluster_map = _connected_components(adj)
    observed_edges = {(u, v) for u, ns in adj.items() for v in ns}

    suggestions = []
    for i, u in enumerate(nodes):
        for j, v in enumerate(nodes):
            if j <= i:
                continue
            if (u, v) in observed_edges:
                continue

            cn = _common_neighbours(u, v, adj)
            aa = _adamic_adar(u, v, adj)
            jac = _jaccard(u, v, adj)

            # Composite: 40% CN (normalised by max possible), 40% AA, 20% Jaccard
            max_cn = max(len(adj.get(u, set())), len(adj.get(v, set())), 1)
            composite = round(0.40 * (cn / max_cn) + 0.40 * min(aa, 1.0) + 0.20 * jac, 6)

            if composite < SCORE_THRESHOLD:
                continue

            u_name = PHONE_TO_NAME.get(u, u)
            v_name = PHONE_TO_NAME.get(v, v)
            cl_u   = cluster_map.get(u, -1)
            cl_v   = cluster_map.get(v, -1)

            cross_cluster = cl_u != cl_v
            explanation   = (
                f"Possible unobserved intermediary between "
                f"{'different clusters' if cross_cluster else 'the same cluster'}: "
                f"'{u_name}' and '{v_name}' share {cn} mutual contact(s). "
                f"Adamic-Adar={aa:.3f}, Jaccard={jac:.3f}. "
                "Requires investigator verification."
            )

            suggestions.append({
                "pair": [u_name, v_name],
                "raw_ids": [u, v],
                "cluster_u": cl_u,
                "cluster_v": cl_v,
                "cross_cluster": cross_cluster,
                "scores": {"common_neighbours": cn, "adamic_adar": aa,
                           "jaccard": jac, "composite": composite},
                "explanation": explanation,
                "evidence": {
                    "cdr_file": str(cdr_path.relative_to(REPO_ROOT)),
                    "txn_file": str(txn_path.relative_to(REPO_ROOT)),
                },
            })

    suggestions.sort(key=lambda s: s["scores"]["composite"], reverse=True)
    return suggestions[:top_k]


def suggest_all(top_k: int = 10) -> dict[str, list[dict]]:
    """Run ghost-node finder for every case in ground_truth.json."""
    gt = json.loads((DEMO_DIR / "ground_truth.json").read_text())
    return {case_id: suggest_ghosts(case_id, top_k=top_k) for case_id in gt}


def save_suggestions(suggestions_by_case: dict) -> Path:
    """Write ghost_suggestions.json to M3_feature/ and feature_outputs/."""
    data = {"seed": SEED, "threshold": SCORE_THRESHOLD,
            "suggestions": suggestions_by_case}
    out1 = OUTPUT_DIR / "ghost_suggestions.json"
    FEAT_OUT.mkdir(parents=True, exist_ok=True)
    out2 = FEAT_OUT / "ghost_suggestions.json"
    for path in (out1, out2):
        path.write_text(json.dumps(data, indent=2))
    return out1


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli():
    args = sys.argv[1:]
    if not args:
        print("Usage: python -m M3_feature.ghost_node suggest <case_id> [top_k]")
        print("       python -m M3_feature.ghost_node suggest_all")
        sys.exit(1)

    cmd = args[0]
    if cmd == "suggest":
        case_id = args[1] if len(args) > 1 else "case_A"
        top_k   = int(args[2]) if len(args) > 2 else 10
        print(f"Running ghost-node finder for {case_id} (top_k={top_k}) ...")
        suggestions = suggest_ghosts(case_id, top_k=top_k)
        save_suggestions({case_id: suggestions})
        print(f"  {len(suggestions)} suggestion(s).")
        for s in suggestions[:3]:
            print(f"  {s['pair']}: composite={s['scores']['composite']:.4f}")
    elif cmd == "suggest_all":
        print("Running ghost-node finder for all cases ...")
        all_s = suggest_all()
        save_suggestions(all_s)
        total = sum(len(v) for v in all_s.values())
        print(f"  {total} total suggestions across {len(all_s)} cases.")
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    _cli()
