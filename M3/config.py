"""
M3 — Graph Intelligence Module
PS 26152 — AI-Powered Criminal Network Analysis System

All configurable thresholds and scoring weights live here.
Change a value here to tune detection sensitivity without touching rule code.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class M3Config:
    # ---- M3.3: communication-spike rule ----
    # Minimum number of distinct calls within spike_window_hours to trigger a flag
    spike_call_count_threshold: int = 5
    # Time window (hours) over which calls are counted for spike detection
    spike_window_hours: float = 24.0

    # ---- M3.4: transaction-frequency anomaly rule ----
    # An entity's transaction count must exceed this multiple of its own mean
    # (per spike_window_hours window) to be flagged
    txn_frequency_multiplier: float = 2.5
    # Minimum absolute transaction count in one window to avoid flagging trivially
    txn_min_window_count: int = 3
    # Large single transaction threshold (INR)
    txn_large_amount_inr: float = 150_000.0

    # ---- M3.5: shared-account / rapid fund-movement rule ----
    # Minimum number of distinct callers/senders to an account to flag shared-account
    multi_suspect_min_senders: int = 3
    # Maximum hours for a fund chain to qualify as "rapid"
    rapid_fund_hours: float = 48.0
    # Minimum hops in a rapid fund chain
    rapid_fund_min_hops: int = 2

    # ---- M3.6: incident-timing cluster rule ----
    # Activity within this many days of a known incident date is flagged
    incident_window_days: int = 3
    # Minimum calls/transactions near incident date to be flagged
    incident_min_activity: int = 2

    # ---- M3.7: priority scorer weights ----
    # All weights must sum to 1.0
    weight_degree_centrality: float = 0.15
    weight_betweenness_centrality: float = 0.20
    weight_pagerank: float = 0.15
    weight_flag_count: float = 0.50

    # ---- M3.7: dense-cluster flag threshold ----
    # Minimum in-cluster degree ratio to flag DENSE_CLUSTER_MEMBERSHIP
    dense_cluster_degree_ratio: float = 0.9


# Default singleton used everywhere in the module.
# Override by constructing a new M3Config and passing it explicitly.
DEFAULT_CONFIG = M3Config()
