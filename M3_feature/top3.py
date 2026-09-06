"""
M3_feature — Top-3 Suspect Ranking with Explainability
PS 26152 — AI-Powered Criminal Network Analysis System

Identifies the top 3 highest-risk entities from the criminal network graph
using a transparent, weighted risk scoring formula. For each suspect, produces
plain-English explanations backed by exact evidence line references.

Risk Score Formula (all weights configurable):
    risk_score = normalize(
        W_DEGREE      * degree_centrality       # network connectivity
      + W_TEMPORAL    * temporal_anomaly_score   # unusual timing patterns
      + W_COLOCATION  * colocation_score         # same-location convergence
      + W_CALL        * call_pattern_score       # suspicious call patterns
      + W_TRANSACTION * transaction_anomaly_score # financial red flags
    )

Language discipline: All scores indicate investigation-priority only.
No score implies guilt or criminal determination.

Usage:
    python -m M3_feature.top3
    # or
    from M3_feature.top3 import run_top3_analysis
    result = run_top3_analysis()
"""

import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Scoring weight constants (must sum to 1.0)
# These can be tuned to adjust sensitivity of each factor.
# ---------------------------------------------------------------------------
W_DEGREE = 0.20       # How connected the entity is in the network
W_TEMPORAL = 0.20     # Unusual timing patterns (late-night, bursts)
W_COLOCATION = 0.15   # Co-location with other flagged entities at same tower
W_CALL = 0.20         # Frequency and pattern of calls to flagged numbers
W_TRANSACTION = 0.25  # Financial anomalies (large amounts, high frequency)

assert abs(W_DEGREE + W_TEMPORAL + W_COLOCATION + W_CALL + W_TRANSACTION - 1.0) < 1e-6, \
    "Scoring weights must sum to 1.0"

# Phone -> person name lookup (from known entity pool)
PERSONS = [
    {"name": "Ravi Kumar",     "aliases": ["Ravi K.", "R. Kumar", "Ravi"],     "phone": "9876543210", "vehicle": "DL01AB1234", "account": "ACC00101"},
    {"name": "Sunita Sharma",  "aliases": ["S. Sharma", "Sunita S."],          "phone": "9123456789", "vehicle": "MH12CD5678", "account": "ACC00102"},
    {"name": "Mohammed Iqbal", "aliases": ["M. Iqbal", "Mohd Iqbal", "Iqbal"],"phone": "9988776655", "vehicle": "UP32EF9012", "account": "ACC00103"},
    {"name": "Priya Nair",     "aliases": ["P. Nair", "Priya N."],            "phone": "8877665544", "vehicle": "KA05GH3456", "account": "ACC00104"},
    {"name": "Deepak Verma",   "aliases": ["D. Verma", "Deepak V.", "Deepak"],"phone": "7766554433", "vehicle": "RJ14IJ7890", "account": "ACC00105"},
    {"name": "Anjali Singh",   "aliases": ["A. Singh", "Anjali S."],           "phone": "9654321098", "vehicle": "GJ01KL2345", "account": "ACC00106"},
    {"name": "Rakesh Yadav",   "aliases": ["R. Yadav", "Rakesh Y."],           "phone": "9543210987", "vehicle": "HR26MN6789", "account": "ACC00107"},
    {"name": "Fatima Begum",   "aliases": ["F. Begum", "Fatima B."],           "phone": "9432109876", "vehicle": "TN09OP0123", "account": "ACC00108"},
    {"name": "Suresh Patil",   "aliases": ["S. Patil", "Suresh P."],           "phone": "9321098765", "vehicle": "MP07QR4567", "account": "ACC00109"},
    {"name": "Kavya Reddy",    "aliases": ["K. Reddy", "Kavya R."],            "phone": "9210987654", "vehicle": "AP28ST8901", "account": "ACC00110"},
]

PHONE_TO_PERSON = {p["phone"]: p for p in PERSONS}
ACCOUNT_TO_PERSON = {p["account"]: p for p in PERSONS}
NAME_TO_PERSON = {p["name"]: p for p in PERSONS}


# ===========================================================================
# Individual Score Components — Each returns a float in [0, 1]
# ===========================================================================

def compute_degree_centrality(person: dict, cdr_rows: list[dict],
                              txn_rows: list[dict]) -> float:
    """
    Degree centrality: how many distinct entities this person communicates with.

    Counts unique phones called/called-by in CDRs plus unique accounts
    transferred-to/received-from in transactions. Normalized by total
    distinct counterparties in the dataset.

    Returns: float in [0, 1]
    """
    phone = person["phone"]
    account = person["account"]

    counterparties = set()

    # CDR counterparties
    for row in cdr_rows:
        caller = row.get("caller", "").strip()
        callee = row.get("callee", "").strip()
        if caller == phone:
            counterparties.add(callee)
        elif callee == phone:
            counterparties.add(caller)

    # Transaction counterparties
    for row in txn_rows:
        src = row.get("source_account", "").strip()
        tgt = row.get("target_account", "").strip()
        if src == account:
            counterparties.add(tgt)
        elif tgt == account:
            counterparties.add(src)

    # Normalize by total distinct entities in the dataset
    all_phones = set()
    for row in cdr_rows:
        all_phones.add(row.get("caller", "").strip())
        all_phones.add(row.get("callee", "").strip())
    all_accounts = set()
    for row in txn_rows:
        all_accounts.add(row.get("source_account", "").strip())
        all_accounts.add(row.get("target_account", "").strip())

    max_possible = len(all_phones | all_accounts) - 1  # exclude self
    if max_possible <= 0:
        return 0.0

    return min(len(counterparties) / max_possible, 1.0)


def compute_temporal_anomaly(person: dict, cdr_rows: list[dict]) -> float:
    """
    Temporal anomaly: proportion of calls made during unusual hours (00:00–06:00).

    Late-night and early-morning calls are weighted higher as they may indicate
    covert communication. Also considers burst patterns (multiple calls within
    a short window).

    Returns: float in [0, 1]
    """
    phone = person["phone"]
    call_times = []

    for row in cdr_rows:
        caller = row.get("caller", "").strip()
        callee = row.get("callee", "").strip()
        if caller == phone or callee == phone:
            ts_str = row.get("timestamp", "").strip()
            try:
                ts = datetime.fromisoformat(ts_str)
                call_times.append(ts)
            except (ValueError, TypeError):
                continue

    if not call_times:
        return 0.0

    # Late-night ratio (00:00 - 06:00)
    late_night = sum(1 for t in call_times if 0 <= t.hour < 6)
    late_night_ratio = late_night / len(call_times)

    # Burst detection: calls within 2-hour windows
    call_times.sort()
    max_burst = 0
    for i, t in enumerate(call_times):
        burst = sum(1 for t2 in call_times[i:] if (t2 - t).total_seconds() <= 7200)
        max_burst = max(max_burst, burst)

    # Normalize burst: 5+ calls in 2 hours = max score
    burst_score = min(max_burst / 5.0, 1.0)

    # Combined: 60% late-night ratio + 40% burst score
    return min(0.6 * late_night_ratio + 0.4 * burst_score, 1.0)


def compute_colocation_score(person: dict, cdr_rows: list[dict]) -> float:
    """
    Co-location score: how often this person appears at the same tower/location
    as other persons within a short time window.

    Higher score = more co-location events with other known entities.

    Returns: float in [0, 1]
    """
    phone = person["phone"]
    other_phones = {p["phone"] for p in PERSONS if p["phone"] != phone}

    # Build location-time index for this person
    person_events = []
    for row in cdr_rows:
        caller = row.get("caller", "").strip()
        callee = row.get("callee", "").strip()
        if caller == phone or callee == phone:
            ts_str = row.get("timestamp", "").strip()
            location = row.get("tower_location", "").strip()
            try:
                ts = datetime.fromisoformat(ts_str)
                person_events.append((ts, location))
            except (ValueError, TypeError):
                continue

    if not person_events:
        return 0.0

    # Build location-time index for all other persons
    other_events = []
    for row in cdr_rows:
        caller = row.get("caller", "").strip()
        callee = row.get("callee", "").strip()
        if caller in other_phones or callee in other_phones:
            ts_str = row.get("timestamp", "").strip()
            location = row.get("tower_location", "").strip()
            try:
                ts = datetime.fromisoformat(ts_str)
                other_events.append((ts, location))
            except (ValueError, TypeError):
                continue

    # Count co-location events (same location, within 24 hours)
    colocations = 0
    for p_ts, p_loc in person_events:
        for o_ts, o_loc in other_events:
            if p_loc == o_loc and abs((p_ts - o_ts).total_seconds()) <= 86400:
                colocations += 1

    # Normalize: 10+ co-locations = max score
    return min(colocations / 10.0, 1.0)


def compute_call_pattern_score(person: dict, cdr_rows: list[dict]) -> float:
    """
    Call pattern score: frequency and regularity of calls.

    Considers:
    - Total call volume (relative to other entities)
    - Repeated calls to same number (>3 calls to same number is suspicious)
    - Very short calls (< 30s) which may indicate signaling

    Returns: float in [0, 1]
    """
    phone = person["phone"]
    call_counts = Counter()  # callee -> count
    short_calls = 0
    total_calls = 0

    for row in cdr_rows:
        caller = row.get("caller", "").strip()
        callee = row.get("callee", "").strip()
        duration = int(row.get("duration_seconds", "0").strip() or "0")

        if caller == phone:
            call_counts[callee] += 1
            total_calls += 1
            if duration < 30:
                short_calls += 1
        elif callee == phone:
            total_calls += 1

    if total_calls == 0:
        return 0.0

    # Volume component: normalize by total CDR rows
    volume_score = min(total_calls / 30.0, 1.0)

    # Repeat-call component: any number called 3+ times
    max_repeats = max(call_counts.values()) if call_counts else 0
    repeat_score = min(max_repeats / 5.0, 1.0)

    # Short-call component (potential signaling)
    short_ratio = short_calls / total_calls if total_calls > 0 else 0.0

    # Combined: 40% volume + 40% repeats + 20% short-call
    return min(0.4 * volume_score + 0.4 * repeat_score + 0.2 * short_ratio, 1.0)


def compute_transaction_anomaly(person: dict, txn_rows: list[dict]) -> float:
    """
    Transaction anomaly: identifies unusual financial patterns.

    Considers:
    - Total transaction volume (number of transactions)
    - Large single transactions (> ₹100,000)
    - Average transaction amount relative to network average
    - Number of distinct counterparty accounts

    Returns: float in [0, 1]
    """
    account = person["account"]
    amounts = []
    counterparties = set()

    for row in txn_rows:
        src = row.get("source_account", "").strip()
        tgt = row.get("target_account", "").strip()
        try:
            amount = float(row.get("amount_inr", "0").strip() or "0")
        except (ValueError, TypeError):
            amount = 0.0

        if src == account:
            amounts.append(amount)
            counterparties.add(tgt)
        elif tgt == account:
            amounts.append(amount)
            counterparties.add(src)

    if not amounts:
        return 0.0

    # Volume component
    volume_score = min(len(amounts) / 15.0, 1.0)

    # Large transaction component (> ₹100,000)
    large_txns = sum(1 for a in amounts if a > 100000)
    large_score = min(large_txns / 3.0, 1.0)

    # Total amount component
    total = sum(amounts)
    total_score = min(total / 500000.0, 1.0)

    # Counterparty diversity
    diversity_score = min(len(counterparties) / 8.0, 1.0)

    # Combined: 25% each
    return min(
        0.25 * volume_score +
        0.30 * large_score +
        0.25 * total_score +
        0.20 * diversity_score,
        1.0
    )


# ===========================================================================
# Evidence Collection — Finds exact file:line references
# ===========================================================================

def find_evidence_in_file(filepath: str, search_terms: list[str],
                          max_snippets: int = 5) -> list[dict]:
    """
    Search a file for occurrences of any search term.
    Returns list of {file, line_start, line_end, text_snippet}.
    """
    evidence = []
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            line_lower = line.lower()
            for term in search_terms:
                if term.lower() in line_lower:
                    snippet = line.strip()[:200]
                    evidence.append({
                        "file": str(filepath),
                        "line_start": i + 1,
                        "line_end": i + 1,
                        "text_snippet": snippet,
                    })
                    break  # One match per line

            if len(evidence) >= max_snippets:
                break

    except (OSError, IOError):
        pass

    return evidence


def collect_evidence(person: dict, data_dir: Path) -> list[dict]:
    """
    Collect all evidence references for a person across FIRs, CDRs, and transactions.
    Returns list of evidence dicts with file:line provenance.
    """
    search_terms = [person["name"], person["phone"], person["account"]]
    search_terms.extend(person["aliases"])
    if person.get("vehicle"):
        search_terms.append(person["vehicle"])

    all_evidence = []

    # Search FIR files
    fir_dir = data_dir / "firs"
    if fir_dir.exists():
        for fir_file in sorted(fir_dir.glob("*.txt")):
            evidence = find_evidence_in_file(str(fir_file), search_terms, max_snippets=3)
            all_evidence.extend(evidence)

    # Search CDR
    cdr_path = data_dir / "cdrs" / "cdr.csv"
    if cdr_path.exists():
        evidence = find_evidence_in_file(str(cdr_path), search_terms, max_snippets=5)
        all_evidence.extend(evidence)

    # Search transactions
    txn_path = data_dir / "transactions" / "transactions.csv"
    if txn_path.exists():
        evidence = find_evidence_in_file(str(txn_path), search_terms, max_snippets=5)
        all_evidence.extend(evidence)

    return all_evidence[:15]  # Cap at 15 evidence items per suspect


# ===========================================================================
# Explanation Generator
# ===========================================================================

def generate_explanation(person: dict, scores: dict, evidence: list[dict],
                         cdr_rows: list[dict], txn_rows: list[dict]) -> str:
    """
    Generate a plain-English explanation with 3 bullet points,
    each referencing specific evidence.
    """
    phone = person["phone"]
    account = person["account"]
    name = person["name"]
    bullets = []

    # Bullet 1: Communication patterns
    call_count = sum(1 for r in cdr_rows
                     if r.get("caller", "").strip() == phone
                     or r.get("callee", "").strip() == phone)
    late_night = sum(1 for r in cdr_rows
                     if (r.get("caller", "").strip() == phone
                         or r.get("callee", "").strip() == phone)
                     and _is_late_night(r.get("timestamp", "")))

    cdr_evidence = [e for e in evidence if "cdr" in e["file"].lower()]
    cdr_ref = ""
    if cdr_evidence:
        e = cdr_evidence[0]
        cdr_ref = f" (see {_short_path(e['file'])}:L{e['line_start']})"

    if late_night > 0:
        bullets.append(
            f"Made {call_count} calls total, {late_night} during late-night hours "
            f"(00:00-06:00), suggesting covert communication patterns{cdr_ref}."
        )
    else:
        bullets.append(
            f"Connected to {call_count} calls across the CDR records, "
            f"indicating an active communication role in the network{cdr_ref}."
        )

    # Bullet 2: Financial patterns
    txn_count = 0
    total_amount = 0.0
    large_txns = 0
    for r in txn_rows:
        src = r.get("source_account", "").strip()
        tgt = r.get("target_account", "").strip()
        try:
            amt = float(r.get("amount_inr", "0") or "0")
        except (ValueError, TypeError):
            amt = 0.0
        if src == account or tgt == account:
            txn_count += 1
            total_amount += amt
            if amt > 100000:
                large_txns += 1

    txn_evidence = [e for e in evidence if "transaction" in e["file"].lower()]
    txn_ref = ""
    if txn_evidence:
        e = txn_evidence[0]
        txn_ref = f" (see {_short_path(e['file'])}:L{e['line_start']})"

    if large_txns > 0:
        bullets.append(
            f"Involved in {txn_count} financial transactions totalling "
            f"₹{total_amount:,.0f}, including {large_txns} large transactions "
            f"exceeding ₹1,00,000{txn_ref}."
        )
    elif txn_count > 0:
        bullets.append(
            f"Participated in {txn_count} financial transactions totalling "
            f"₹{total_amount:,.0f}{txn_ref}."
        )
    else:
        bullets.append(f"No direct financial transactions found for account {account}.")

    # Bullet 3: FIR mentions and network position
    fir_evidence = [e for e in evidence
                    if "fir" in e["file"].lower() and "cdr" not in e["file"].lower()]
    fir_ref = ""
    if fir_evidence:
        e = fir_evidence[0]
        fir_ref = f" (see {_short_path(e['file'])}:L{e['line_start']})"
        fir_count = len(set(e["file"] for e in fir_evidence))
        bullets.append(
            f"Named or referenced in {fir_count} FIR document(s), with a network "
            f"connectivity score of {scores['degree']:.0%}{fir_ref}."
        )
    else:
        bullets.append(
            f"Network analysis shows elevated connectivity "
            f"(degree: {scores['degree']:.0%}, temporal anomaly: {scores['temporal']:.0%})."
        )

    return " ".join(f"{i+1}) {b}" for i, b in enumerate(bullets))


def _is_late_night(ts_str: str) -> bool:
    """Check if a timestamp falls in 00:00-06:00."""
    try:
        ts = datetime.fromisoformat(ts_str.strip())
        return 0 <= ts.hour < 6
    except (ValueError, TypeError):
        return False


def _short_path(filepath: str) -> str:
    """Convert absolute path to relative path from repo root."""
    try:
        return str(Path(filepath).relative_to(REPO_ROOT))
    except ValueError:
        return Path(filepath).name


# ===========================================================================
# Main Analysis Pipeline
# ===========================================================================

def load_cdr_rows(cdr_path: Path) -> list[dict]:
    """Load CDR CSV into list of dicts."""
    rows = []
    with open(cdr_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def load_txn_rows(txn_path: Path) -> list[dict]:
    """Load transaction CSV into list of dicts."""
    rows = []
    with open(txn_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def run_top3_analysis(
    data_dir: Path | None = None,
    output_path: Path | None = None,
    top_n: int = 3,
) -> dict:
    """
    Run the complete top-3 suspect analysis.

    Args:
        data_dir:    Directory containing data/firs/, data/cdrs/, data/transactions/
                     Defaults to REPO_ROOT / "data"
        output_path: Where to write top3_results.json
                     Defaults to REPO_ROOT / "M3_feature" / "top3_results.json"
        top_n:       Number of top suspects to return (default 3)

    Returns:
        Dict matching the top3_results.json schema.
    """
    if data_dir is None:
        data_dir = REPO_ROOT / "data"
    if output_path is None:
        output_path = REPO_ROOT / "M3_feature" / "top3_results.json"

    print(f"[Top3] Loading data from {data_dir}...")

    # Load raw data
    cdr_path = data_dir / "cdrs" / "cdr.csv"
    txn_path = data_dir / "transactions" / "transactions.csv"

    cdr_rows = load_cdr_rows(cdr_path) if cdr_path.exists() else []
    txn_rows = load_txn_rows(txn_path) if txn_path.exists() else []

    print(f"[Top3] Loaded {len(cdr_rows)} CDR rows, {len(txn_rows)} transaction rows.")

    # Score each person
    scored_persons = []

    for person in PERSONS:
        # Compute individual score components
        degree = compute_degree_centrality(person, cdr_rows, txn_rows)
        temporal = compute_temporal_anomaly(person, cdr_rows)
        colocation = compute_colocation_score(person, cdr_rows)
        call_pattern = compute_call_pattern_score(person, cdr_rows)
        txn_anomaly = compute_transaction_anomaly(person, txn_rows)

        # Weighted combination
        raw_score = (
            W_DEGREE * degree +
            W_TEMPORAL * temporal +
            W_COLOCATION * colocation +
            W_CALL * call_pattern +
            W_TRANSACTION * txn_anomaly
        )

        # Normalize to 0-100 scale
        risk_score = round(raw_score * 100, 1)

        scores = {
            "degree": degree,
            "temporal": temporal,
            "colocation": colocation,
            "call_pattern": call_pattern,
            "transaction": txn_anomaly,
        }

        scored_persons.append({
            "person": person,
            "risk_score": risk_score,
            "scores": scores,
        })

    # Sort by risk score descending
    scored_persons.sort(key=lambda x: x["risk_score"], reverse=True)

    # Build top-N output
    suspects = []
    for entry in scored_persons[:top_n]:
        person = entry["person"]
        scores = entry["scores"]

        # Collect evidence
        evidence = collect_evidence(person, data_dir)

        # Generate explanation
        explanation = generate_explanation(
            person, scores, evidence, cdr_rows, txn_rows
        )

        # Determine display ID
        entity_id = f"person_{re.sub(r'[^a-zA-Z0-9]', '_', person['name']).lower()}"

        # Build aliases string
        alias_str = person["aliases"][0] if person["aliases"] else ""
        display_name = f"{person['name']} (alias: {alias_str})" if alias_str else person["name"]

        suspects.append({
            "id": entity_id,
            "name": display_name,
            "risk_score": entry["risk_score"],
            "evidence": evidence,
            "explanation": explanation,
        })

    result = {"suspects": suspects}

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"[Top3] Top {top_n} suspects written to {output_path}")
    for i, s in enumerate(suspects, 1):
        print(f"  {i}. {s['name']} — risk_score: {s['risk_score']}")

    return result


# ===========================================================================
# CLI entry point
# ===========================================================================

if __name__ == "__main__":
    run_top3_analysis()
