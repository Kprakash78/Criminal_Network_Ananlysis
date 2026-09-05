"""
M3.3–M3.6 — Pattern Rules Engine
PS 26152 — AI-Powered Criminal Network Analysis System

Five rule-based detectors that operate over the graph structure and the
underlying timestamped CDR / transaction records.  Each detector is a
standalone function returning a dict[entity_id → (flags, evidence_strings)].

Rules are deliberately rule-based (not ML-based) for explainability.
All thresholds come from M3Config — change them there, not here.

Language: flags indicate elevated investigator priority only.
No flag implies guilt, involvement, or criminal determination.
"""

import csv
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import networkx as nx

from M3.config import M3Config, DEFAULT_CONFIG

logger = logging.getLogger(__name__)

# Type alias: entity_id → list of (flag_type, evidence_string) pairs
_FlagMap = dict[str, list[tuple[str, str]]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_ts(ts_str: str) -> datetime | None:
    """Parse an ISO-8601-like timestamp string; return None on failure."""
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(ts_str.strip(), fmt)
        except ValueError:
            continue
    logger.warning(f"[PatternRules] Unrecognised timestamp format: '{ts_str}' — skipping record")
    return None


def _account_to_entity(graph: nx.MultiDiGraph) -> dict[str, str]:
    """
    Build a reverse-lookup: account_number → entity_id.
    BANK_ACCOUNT-type nodes have their account ID as the name.
    """
    mapping: dict[str, str] = {}
    for node_id, attrs in graph.nodes(data=True):
        node_type = attrs.get("type", "")
        name = attrs.get("name", "")
        aliases = attrs.get("aliases", [])
        if node_type in ("BANK_ACCOUNT", "ACCOUNT"):
            mapping[name] = node_id
            for alias in aliases:
                mapping[alias] = node_id
    return mapping


def _load_cdr(cdr_path: str) -> list[dict]:
    """
    Load CDR CSV into a list of row dicts.  Skips rows with missing/malformed
    timestamps with a warning — does NOT crash the analysis pass.
    """
    rows = []
    try:
        with open(cdr_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ts = _parse_ts(row.get("timestamp", ""))
                if ts is None:
                    continue
                rows.append({**row, "_ts": ts})
    except FileNotFoundError:
        logger.warning(f"[PatternRules] CDR file not found: {cdr_path}")
    return rows


def _load_transactions(txn_path: str) -> list[dict]:
    """
    Load transaction CSV into a list of row dicts.  Skips malformed rows.
    """
    rows = []
    try:
        with open(txn_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ts = _parse_ts(row.get("timestamp", ""))
                if ts is None:
                    continue
                try:
                    amount = float(row.get("amount_inr", 0))
                except ValueError:
                    amount = 0.0
                rows.append({**row, "_ts": ts, "_amount": amount})
    except FileNotFoundError:
        logger.warning(f"[PatternRules] Transaction file not found: {txn_path}")
    return rows


# ---------------------------------------------------------------------------
# M3.3 — Communication Spike Rule
# ---------------------------------------------------------------------------

def detect_communication_spikes(
    graph: nx.MultiDiGraph,
    cdr_rows: list[dict],
    config: M3Config = DEFAULT_CONFIG,
) -> _FlagMap:
    """
    Flag entities with ≥ spike_call_count_threshold calls involving their phone
    number within any spike_window_hours rolling window.

    Uses a sliding-window approach: for each entity's call list, count calls
    within every [call_i, call_i + window] interval.

    Returns: entity_id → [(flag_type, evidence_string), ...]
    """
    flags: _FlagMap = defaultdict(list)

    # Collect call timestamps per phone number from the CDR rows.
    phone_calls: dict[str, list[datetime]] = defaultdict(list)
    for row in cdr_rows:
        caller = row.get("caller", "").strip()
        callee = row.get("callee", "").strip()
        ts: datetime = row["_ts"]
        phone_calls[caller].append(ts)
        phone_calls[callee].append(ts)

    # Map phone string → entity_id from PHONE-type nodes, whose name IS the
    # phone number (canonical M1 form) with aliases as fallbacks. The flag is
    # placed on the PHONE node itself: that is the entity the CDR evidence
    # actually names — inferring ownership from CALLED edges would misattribute
    # the spike to whichever co-mentioned PERSON node happens to be iterated last.
    phone_entity_direct: dict[str, str] = {}
    for node_id, attrs in graph.nodes(data=True):
        if attrs.get("type") == "PHONE":
            name = attrs.get("name", "").strip()
            if name:
                phone_entity_direct[name] = node_id
            for alias in attrs.get("aliases", []):
                if alias.strip():
                    phone_entity_direct[alias.strip()] = node_id

    window = timedelta(hours=config.spike_window_hours)
    threshold = config.spike_call_count_threshold

    # For each phone number in CDR, flag its PHONE entity if any window spikes.
    for phone, timestamps in phone_calls.items():
        entity_id = phone_entity_direct.get(phone)
        if entity_id is None:
            # Phone not in graph — skip (CDR may contain unregistered numbers)
            continue

        # Sliding window: sort timestamps, find max calls in any window
        sorted_ts = sorted(timestamps)
        max_in_window = 0
        for i, t_start in enumerate(sorted_ts):
            count = sum(1 for t in sorted_ts[i:] if t <= t_start + window)
            max_in_window = max(max_in_window, count)

        if max_in_window >= threshold:
            evidence = (
                f"{max_in_window} call events involving phone {phone} "
                f"found within a {config.spike_window_hours:.0f}-hour window "
                f"(threshold: {threshold})"
            )
            flags[entity_id].append(("COMMUNICATION_SPIKE", evidence))
            logger.info(f"[PatternRules] COMMUNICATION_SPIKE flagged for {entity_id}: {evidence}")

    return dict(flags)


# ---------------------------------------------------------------------------
# M3.4 — Transaction-Frequency / Large-Amount Anomaly Rule
# ---------------------------------------------------------------------------

def detect_transaction_anomalies(
    graph: nx.MultiDiGraph,
    txn_rows: list[dict],
    config: M3Config = DEFAULT_CONFIG,
) -> _FlagMap:
    """
    Flag entities showing unusual transaction frequency (relative to their own
    baseline) or individual large transactions.

    Two sub-checks:
      1. HIGH_TRANSACTION_FREQUENCY: any window of spike_window_hours contains
         ≥ txn_min_window_count transactions AND that count is ≥
         txn_frequency_multiplier × the entity's mean window count.
      2. HIGH_TRANSACTION_FREQUENCY (large amount): any single transaction
         exceeds txn_large_amount_inr.

    Both use the same output flag type; evidence strings distinguish them.
    """
    flags: _FlagMap = defaultdict(list)
    account_to_entity = _account_to_entity(graph)

    # Collect transactions per account
    account_txns: dict[str, list[tuple[datetime, float]]] = defaultdict(list)
    for row in txn_rows:
        for acct_field in ("source_account", "target_account"):
            acct = row.get(acct_field, "").strip()
            if acct:
                account_txns[acct].append((row["_ts"], row["_amount"]))

    window = timedelta(hours=config.spike_window_hours)
    freq_mult = config.txn_frequency_multiplier
    min_count = config.txn_min_window_count
    large_threshold = config.txn_large_amount_inr

    already_flagged_large: set[str] = set()

    for acct, txns in account_txns.items():
        entity_id = account_to_entity.get(acct)
        if entity_id is None:
            continue

        sorted_txns = sorted(txns, key=lambda x: x[0])
        timestamps = [t for t, _ in sorted_txns]
        amounts = [a for _, a in sorted_txns]

        # ---- Large single transaction ----
        for amt in amounts:
            if amt >= large_threshold and entity_id not in already_flagged_large:
                evidence = (
                    f"Single transaction of INR {amt:,.0f} on account {acct}, "
                    f"exceeding the large-transaction threshold of "
                    f"INR {large_threshold:,.0f}"
                )
                flags[entity_id].append(("HIGH_TRANSACTION_FREQUENCY", evidence))
                already_flagged_large.add(entity_id)
                logger.info(f"[PatternRules] HIGH_TRANSACTION_FREQUENCY (large amount) for {entity_id}")
                break

        # ---- Frequency spike ----
        if len(timestamps) < 2:
            continue

        # Count transactions in each window starting at each transaction
        window_counts = []
        for i, t_start in enumerate(timestamps):
            count = sum(1 for t in timestamps[i:] if t <= t_start + window)
            window_counts.append(count)

        max_window_count = max(window_counts)
        mean_window_count = sum(window_counts) / len(window_counts)

        if (
            max_window_count >= min_count
            and mean_window_count > 0
            and max_window_count >= freq_mult * mean_window_count
        ):
            # Don't double-flag if large amount already flagged this entity
            evidence = (
                f"Transaction frequency spike on account {acct}: "
                f"peak of {max_window_count} transactions in "
                f"{config.spike_window_hours:.0f} hours "
                f"vs. mean of {mean_window_count:.1f} "
                f"({freq_mult}× threshold)"
            )
            flags[entity_id].append(("HIGH_TRANSACTION_FREQUENCY", evidence))
            logger.info(f"[PatternRules] HIGH_TRANSACTION_FREQUENCY (frequency) for {entity_id}")

    return dict(flags)


# ---------------------------------------------------------------------------
# M3.5 — Shared-Account / Rapid Fund-Movement Rule
# ---------------------------------------------------------------------------

def detect_shared_account_and_rapid_movement(
    graph: nx.MultiDiGraph,
    txn_rows: list[dict],
    config: M3Config = DEFAULT_CONFIG,
) -> _FlagMap:
    """
    Two sub-rules sharing one function (they both operate on the transaction graph):

    MULTI_SUSPECT_SHARED_ACCOUNT:
      An account receives funds from ≥ multi_suspect_min_senders distinct source
      accounts.  Flags the receiving entity (the account entity and/or its owner).

    RAPID_FUND_MOVEMENT:
      A chain of ≥ rapid_fund_min_hops fund transfers completes within
      rapid_fund_hours.  Flags every entity in the chain.
    """
    flags: _FlagMap = defaultdict(list)
    account_to_entity = _account_to_entity(graph)

    # Build a directed transaction graph: source_account → target_account edges
    # with timestamps so we can detect rapid chains.
    txn_graph = nx.MultiDiGraph()
    for row in txn_rows:
        src = row.get("source_account", "").strip()
        tgt = row.get("target_account", "").strip()
        if src and tgt:
            txn_graph.add_edge(src, tgt, ts=row["_ts"], amount=row["_amount"])

    # ---- MULTI_SUSPECT_SHARED_ACCOUNT ----
    for acct in txn_graph.nodes:
        incoming_sources = set()
        for src, _, _ in txn_graph.in_edges(acct, data=True):
            incoming_sources.add(src)

        if len(incoming_sources) >= config.multi_suspect_min_senders:
            entity_id = account_to_entity.get(acct)
            if entity_id:
                evidence = (
                    f"Account {acct} received funds from {len(incoming_sources)} "
                    f"distinct source accounts "
                    f"(threshold: {config.multi_suspect_min_senders}) — "
                    f"requires investigator verification"
                )
                flags[entity_id].append(("MULTI_SUSPECT_SHARED_ACCOUNT", evidence))
                logger.info(f"[PatternRules] MULTI_SUSPECT_SHARED_ACCOUNT for {entity_id}")

    # ---- RAPID_FUND_MOVEMENT ----
    # For each edge in the transaction graph, follow chains of length ≥
    # rapid_fund_min_hops where the total elapsed time fits in rapid_fund_hours.
    rapid_window = timedelta(hours=config.rapid_fund_hours)
    min_hops = config.rapid_fund_min_hops
    flagged_chains: set[frozenset] = set()

    for start_node in txn_graph.nodes:
        # DFS through the transaction graph looking for chains
        stack: list[tuple[str, list[str], datetime | None, datetime | None]] = [
            (start_node, [start_node], None, None)
        ]
        while stack:
            current, path, earliest_ts, latest_ts = stack.pop()
            for _, nxt, edge_data in txn_graph.out_edges(current, data=True):
                edge_ts: datetime = edge_data["ts"]
                new_earliest = min(earliest_ts, edge_ts) if earliest_ts else edge_ts
                new_latest = max(latest_ts, edge_ts) if latest_ts else edge_ts
                new_path = path + [nxt]

                hops = len(new_path) - 1
                elapsed = new_latest - new_earliest

                if hops >= min_hops and elapsed <= rapid_window:
                    chain_key = frozenset(new_path)
                    if chain_key not in flagged_chains:
                        flagged_chains.add(chain_key)
                        # Flag every entity in the chain
                        for acct_in_chain in new_path:
                            entity_id = account_to_entity.get(acct_in_chain)
                            if entity_id:
                                evidence = (
                                    f"Fund movement chain of {hops} hop(s) "
                                    f"through {' → '.join(new_path)} "
                                    f"completed in {elapsed.total_seconds()/3600:.1f} hours "
                                    f"(threshold: {config.rapid_fund_hours:.0f} hours) — "
                                    f"requires investigator verification"
                                )
                                flags[entity_id].append(("RAPID_FUND_MOVEMENT", evidence))

                # Continue DFS if chain can still be extended within window
                if earliest_ts is None or (new_latest - new_earliest) < rapid_window:
                    if len(new_path) < 6:  # cap depth to avoid exponential blowup
                        stack.append((nxt, new_path, new_earliest, new_latest))

    if flagged_chains:
        logger.info(f"[PatternRules] RAPID_FUND_MOVEMENT: {len(flagged_chains)} chain(s) found")

    return dict(flags)


# ---------------------------------------------------------------------------
# M3.6 — Incident Timing Cluster Rule
# ---------------------------------------------------------------------------

def detect_incident_timing_clusters(
    graph: nx.MultiDiGraph,
    cdr_rows: list[dict],
    txn_rows: list[dict],
    incident_dates: list[str],
    config: M3Config = DEFAULT_CONFIG,
) -> _FlagMap:
    """
    Flag entities whose call/transaction activity clusters around known incident dates.

    For each incident date: count how many CDR events and transaction events
    involving each entity fall within ± incident_window_days days.
    Entities with ≥ incident_min_activity events near the incident date are flagged.

    Args:
        incident_dates: list of date strings in YYYY-MM-DD format.
                        If empty, this rule produces no flags (graceful skip).
    """
    flags: _FlagMap = defaultdict(list)

    if not incident_dates:
        logger.info("[PatternRules] No incident dates provided — skipping timing-cluster rule")
        return {}

    phone_entity_map: dict[str, str] = {}
    for node_id, attrs in graph.nodes(data=True):
        if attrs.get("type") == "PHONE":
            phone_entity_map[attrs.get("name", "")] = node_id
    account_to_entity = _account_to_entity(graph)

    window_days = timedelta(days=config.incident_window_days)

    for inc_date_str in incident_dates:
        try:
            inc_date = datetime.strptime(inc_date_str.strip(), "%Y-%m-%d")
        except ValueError:
            logger.warning(f"[PatternRules] Unrecognised incident date format: '{inc_date_str}' — skipping")
            continue

        win_start = inc_date - window_days
        win_end = inc_date + window_days

        # Count activity per entity near this incident
        activity_count: dict[str, int] = defaultdict(int)

        for row in cdr_rows:
            ts: datetime = row["_ts"]
            if win_start <= ts <= win_end:
                for phone_field in ("caller", "callee"):
                    phone = row.get(phone_field, "").strip()
                    entity_id = phone_entity_map.get(phone)
                    if entity_id:
                        activity_count[entity_id] += 1

        for row in txn_rows:
            ts: datetime = row["_ts"]
            if win_start <= ts <= win_end:
                for acct_field in ("source_account", "target_account"):
                    acct = row.get(acct_field, "").strip()
                    entity_id = account_to_entity.get(acct)
                    if entity_id:
                        activity_count[entity_id] += 1

        for entity_id, count in activity_count.items():
            if count >= config.incident_min_activity:
                evidence = (
                    f"{count} call/transaction event(s) found within "
                    f"{config.incident_window_days} days of incident date {inc_date_str} — "
                    f"requires investigator verification"
                )
                flags[entity_id].append(("INCIDENT_TIMING_CLUSTER", evidence))
                logger.info(
                    f"[PatternRules] INCIDENT_TIMING_CLUSTER for {entity_id}: "
                    f"{count} events near {inc_date_str}"
                )

    return dict(flags)


# ---------------------------------------------------------------------------
# Public runner — merge all rule outputs
# ---------------------------------------------------------------------------

def run_pattern_rules(
    graph: nx.MultiDiGraph,
    cdr_path: str,
    txn_path: str,
    incident_dates: list[str] | None = None,
    config: M3Config = DEFAULT_CONFIG,
) -> dict[str, list[tuple[str, str]]]:
    """
    Run all pattern rules and return a merged dict:
      entity_id → [(flag_type, evidence_string), ...]

    Args:
        graph:          M2's CriminalGraph
        cdr_path:       path to the CDR CSV file
        txn_path:       path to the transaction CSV file
        incident_dates: optional list of YYYY-MM-DD date strings
        config:         M3Config (defaults to DEFAULT_CONFIG)

    Returns:
        Merged flag map across all rules for all entities.
    """
    if graph.number_of_nodes() == 0:
        raise ValueError("Cannot run pattern rules on an empty graph.")

    cdr_rows = _load_cdr(cdr_path)
    txn_rows = _load_transactions(txn_path)

    logger.info(
        f"[PatternRules] Loaded {len(cdr_rows)} CDR rows, "
        f"{len(txn_rows)} transaction rows"
    )

    merged: dict[str, list[tuple[str, str]]] = defaultdict(list)

    for rule_name, rule_result in [
        ("COMMUNICATION_SPIKE",          detect_communication_spikes(graph, cdr_rows, config)),
        ("HIGH_TRANSACTION_FREQUENCY",   detect_transaction_anomalies(graph, txn_rows, config)),
        ("SHARED_ACCOUNT/RAPID_FUNDS",   detect_shared_account_and_rapid_movement(graph, txn_rows, config)),
        ("INCIDENT_TIMING",              detect_incident_timing_clusters(
                                             graph, cdr_rows, txn_rows,
                                             incident_dates or [], config)),
    ]:
        for entity_id, flag_pairs in rule_result.items():
            merged[entity_id].extend(flag_pairs)

    logger.info(f"[PatternRules] Total entities with flags: {len(merged)}")
    return dict(merged)
