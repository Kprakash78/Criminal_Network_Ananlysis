"""
M3_feature/event_causality.py
PS 26152 — AI-Powered Criminal Network Analysis System
======================================================
Event-Causality Chain Builder.

Reads existing parsed outputs (CDR CSV + transaction CSV) and links events
into ordered chains using a sliding time-window heuristic. Marks "pivot
events" — moments where the action type shifts (call → transfer → report)
— and scores them by temporal proximity and actor centrality.

OUTPUT: M3_feature/event_chains.json (also demo_cache/feature_outputs/)
FORMAT per chain:
  {
    "chain_id": "case_A_c0",
    "case_id":  "case_A",
    "steps": [
      { "t": "2024-03-01T19:50:00",
        "actor": "9123456789",
        "action": "call",
        "target": "9654321098",
        "duration_sec": 122,
        "source": "demo_dataset/cdr_files/case_A_cdr.csv",
        "line": 2 }
    ],
    "pivot": {"step_index": 2, "score": 0.82},
    "n_steps": 5
  }

LANGUAGE: All explanations are phrased as observations requiring
investigator verification. No guilt determinations are made.

Usage (CLI):
    python -m M3_feature.event_causality build case_A
    python -m M3_feature.event_causality build_all

API:
    from M3_feature.event_causality import build_chains
    chains = build_chains("case_A")
"""

import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parent.parent
DEMO_DIR    = REPO_ROOT / "demo_dataset"
OUTPUT_DIR  = Path(__file__).resolve().parent   # M3_feature/
FEAT_OUT    = REPO_ROOT / "demo_cache" / "feature_outputs"

SEED = 42
np.random.seed(SEED)

# Window within which events are considered causally related
WINDOW_MINUTES = 30


# ── Person/phone lookup (shared across modules) ───────────────────────────────
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
ACCOUNT_TO_NAME = {p["account"]: p["name"] for p in PERSONS}
ACCOUNT_TO_PHONE = {p["account"]: p["phone"] for p in PERSONS}


# ── Event loading ─────────────────────────────────────────────────────────────

def _load_cdr(path: Path) -> list[dict]:
    """Load CDR CSV → list of normalised event dicts."""
    events = []
    if not path.exists():
        return events
    with open(path, newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f), start=2):
            try:
                ts = datetime.fromisoformat(row["timestamp"].strip())
            except ValueError:
                continue
            events.append({
                "t": ts,
                "actor": row.get("caller", "").strip(),
                "action": "call",
                "target": row.get("receiver", "").strip(),
                "duration_sec": int(row.get("duration_sec", 0)),
                "tower": row.get("tower_id", ""),
                "source": str(path.relative_to(REPO_ROOT)),
                "line": i,
            })
    return events


def _load_txn(path: Path) -> list[dict]:
    """Load transaction CSV → list of normalised event dicts."""
    events = []
    if not path.exists():
        return events
    with open(path, newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f), start=2):
            try:
                ts = datetime.fromisoformat(row["timestamp"].strip())
            except ValueError:
                continue
            # Map account → phone for consistent actor ID
            src_acct = row.get("sender_account", "").strip()
            dst_acct = row.get("receiver_account", "").strip()
            events.append({
                "t": ts,
                "actor": ACCOUNT_TO_PHONE.get(src_acct, src_acct),
                "action": "transfer",
                "target": ACCOUNT_TO_PHONE.get(dst_acct, dst_acct),
                "amount_inr": float(row.get("amount_inr", 0)),
                "flagged": row.get("flagged", "0") == "1",
                "source": str(path.relative_to(REPO_ROOT)),
                "line": i,
            })
    return events


# ── Chain building ─────────────────────────────────────────────────────────────

def _events_to_records(events: list[dict]) -> list[dict]:
    """Normalise events to JSON-serialisable records."""
    out = []
    for e in events:
        rec = {k: v for k, v in e.items() if k != "t"}
        rec["t"] = e["t"].isoformat()
        out.append(rec)
    return out


def _actor_degree(events: list[dict]) -> dict[str, int]:
    """Count how many unique partners each actor has (simple degree)."""
    partners: dict = defaultdict(set)
    for e in events:
        partners[e["actor"]].add(e["target"])
        partners[e["target"]].add(e["actor"])
    return {actor: len(p) for actor, p in partners.items()}


def _pivot_score(step_idx: int, steps: list[dict],
                 degree: dict[str, int]) -> float:
    """
    Score pivotness for a step inside a chain:
    - +0.4 if action type changes from previous step
    - +0.3 if temporal gap to next step < 10 min
    - +0.3 proportional to actor degree (normalised)
    Returns float in [0, 1].
    """
    s = steps[step_idx]
    score = 0.0

    if step_idx > 0 and steps[step_idx - 1]["action"] != s["action"]:
        score += 0.4

    if step_idx < len(steps) - 1:
        t_curr = datetime.fromisoformat(s["t"])
        t_next = datetime.fromisoformat(steps[step_idx + 1]["t"])
        gap_min = abs((t_next - t_curr).total_seconds()) / 60
        score += 0.3 * max(0.0, 1.0 - gap_min / 10)

    max_deg = max(degree.values(), default=1)
    actor_deg = degree.get(s["actor"], 0)
    score += 0.3 * (actor_deg / max_deg)

    return round(min(score, 1.0), 4)


def build_chains(case_id: str,
                 window_minutes: int = WINDOW_MINUTES) -> list[dict]:
    """
    Build ordered event-causality chains for one case.

    Algorithm:
      1. Load all CDR + transaction events and sort by timestamp.
      2. Slide a window of `window_minutes` forward.
      3. Within each window, link events that share an actor or target.
      4. Merge overlapping groups into chains.
      5. For each chain, identify the step with the highest pivot score.

    Returns a list of chain dicts (JSON-serialisable).
    """
    np.random.seed(SEED)

    gt_path = DEMO_DIR / "ground_truth.json"
    if not gt_path.exists():
        raise FileNotFoundError("ground_truth.json missing — run generate_demo_data.py")
    gt = json.loads(gt_path.read_text())[case_id]

    events: list[dict] = []
    events += _load_cdr(REPO_ROOT / gt["cdr_file"])
    events += _load_txn(REPO_ROOT / gt["txn_file"])
    events.sort(key=lambda e: e["t"])

    if not events:
        return []

    degree = _actor_degree(events)
    window = timedelta(minutes=window_minutes)

    # ── Sliding window grouping ──────────────────────────────────────────────
    # Union-find to merge events that share actor/target within window
    parent = list(range(len(events)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    for i in range(len(events)):
        for j in range(i + 1, len(events)):
            if events[j]["t"] - events[i]["t"] > window:
                break
            shared = ({events[i]["actor"], events[i]["target"]} &
                      {events[j]["actor"], events[j]["target"]})
            if shared:
                union(i, j)

    # ── Group by root ────────────────────────────────────────────────────────
    groups: dict = defaultdict(list)
    for i in range(len(events)):
        groups[find(i)].append(i)

    # ── Build chain dicts ────────────────────────────────────────────────────
    chains = []
    for chain_idx, (root, indices) in enumerate(sorted(groups.items())):
        steps_raw = [events[i] for i in sorted(indices)]
        steps = _events_to_records(steps_raw)

        # Pivot = step with highest pivot score
        scores = [_pivot_score(k, steps, degree) for k in range(len(steps))]
        best_k = int(np.argmax(scores))

        chains.append({
            "chain_id": f"{case_id}_c{chain_idx}",
            "case_id":  case_id,
            "n_steps":  len(steps),
            "steps":    steps,
            "pivot":    {"step_index": best_k, "score": scores[best_k]},
            "note":     ("Ordered sequence of observed events within a "
                         f"{window_minutes}-minute window. "
                         "Requires investigator verification."),
        })

    # Sort chains by size descending for relevance
    chains.sort(key=lambda c: c["n_steps"], reverse=True)
    return chains


def build_all_chains() -> dict[str, list[dict]]:
    """Build chains for all cases in ground_truth.json."""
    gt_path = DEMO_DIR / "ground_truth.json"
    if not gt_path.exists():
        raise FileNotFoundError("ground_truth.json missing")
    gt = json.loads(gt_path.read_text())
    return {case_id: build_chains(case_id) for case_id in gt}


def save_chains(chains_by_case: dict) -> Path:
    """Write event_chains.json to M3_feature/ and demo_cache/feature_outputs/."""
    flat = [chain for chains in chains_by_case.values() for chain in chains]
    data = {"seed": SEED, "window_minutes": WINDOW_MINUTES, "chains": flat}
    out1 = OUTPUT_DIR / "event_chains.json"
    FEAT_OUT.mkdir(parents=True, exist_ok=True)
    out2 = FEAT_OUT / "event_chains.json"
    for path in (out1, out2):
        path.write_text(json.dumps(data, indent=2))
    return out1


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli():
    args = sys.argv[1:]
    if not args:
        print("Usage: python -m M3_feature.event_causality build <case_id>")
        print("       python -m M3_feature.event_causality build_all")
        sys.exit(1)

    cmd = args[0]
    if cmd == "build" and len(args) >= 2:
        case_id = args[1]
        print(f"Building chains for {case_id} ...")
        chains = build_chains(case_id)
        save_chains({case_id: chains})
        print(f"  {len(chains)} chains built.")
        for c in chains[:3]:
            print(f"  {c['chain_id']}: {c['n_steps']} steps, "
                  f"pivot@{c['pivot']['step_index']} score={c['pivot']['score']}")
    elif cmd == "build_all":
        print("Building chains for all cases ...")
        all_chains = build_all_chains()
        save_chains(all_chains)
        total = sum(len(v) for v in all_chains.values())
        print(f"  {total} chains across {len(all_chains)} cases saved.")
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    _cli()
