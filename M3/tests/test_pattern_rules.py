"""
Tests for M3.3–M3.6 — Pattern Rules Engine
Each rule has:
  - at least one "planted" case that MUST flag
  - at least one "clean" case that must NOT flag (no false positive)
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import networkx as nx
import pytest

from M3.config import M3Config
from M3.pattern_rules import (
    detect_communication_spikes,
    detect_incident_timing_clusters,
    detect_shared_account_and_rapid_movement,
    detect_transaction_anomalies,
    _load_cdr,
    _load_transactions,
    run_pattern_rules,
)

# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

def _make_phone_graph(phone_entity_pairs: list[tuple[str, str]]) -> nx.MultiDiGraph:
    """Build a graph where each pair (phone, entity_id) is a PHONE node."""
    g = nx.MultiDiGraph()
    for phone, eid in phone_entity_pairs:
        g.add_node(eid, type="PHONE", name=phone, aliases=[], source_documents=[], confidence=0.9)
    return g


def _make_account_graph(account_entity_pairs: list[tuple[str, str]]) -> nx.MultiDiGraph:
    """Build a graph where each pair (account, entity_id) is a BANK_ACCOUNT node."""
    g = nx.MultiDiGraph()
    for acct, eid in account_entity_pairs:
        g.add_node(eid, type="BANK_ACCOUNT", name=acct, aliases=[], source_documents=[], confidence=0.9)
    return g


def _ts(base: datetime, delta_hours: float = 0.0) -> dict:
    """Make a CDR row dict with a parsed timestamp."""
    ts = base + timedelta(hours=delta_hours)
    return {
        "caller": "",
        "callee": "",
        "timestamp": ts.isoformat(),
        "_ts": ts,
    }


def _txn_row(src: str, tgt: str, base: datetime, delta_hours: float = 0.0, amount: float = 1000.0) -> dict:
    ts = base + timedelta(hours=delta_hours)
    return {
        "source_account": src,
        "target_account": tgt,
        "amount_inr": str(amount),
        "timestamp": ts.isoformat(),
        "remarks": "Transfer",
        "_ts": ts,
        "_amount": amount,
    }


# ---------------------------------------------------------------------------
# M3.3 — Communication Spike tests
# ---------------------------------------------------------------------------

class TestCommunicationSpike:
    cfg = M3Config(spike_call_count_threshold=3, spike_window_hours=2.0)
    base = datetime(2025, 6, 1, 12, 0, 0)
    phone_a = "9000000001"
    phone_b = "9000000002"
    entity_a = "PHONE_A"
    entity_b = "PHONE_B"

    def _graph(self):
        return _make_phone_graph([
            (self.phone_a, self.entity_a),
            (self.phone_b, self.entity_b),
        ])

    def _cdr(self, phone: str, count: int, spread_hours: float = 0.3) -> list[dict]:
        """Generate `count` CDR rows for `phone` as caller within spread_hours total."""
        rows = []
        for i in range(count):
            ts = self.base + timedelta(hours=i * spread_hours / max(count - 1, 1))
            rows.append({
                "caller": phone,
                "callee": self.phone_b,
                "timestamp": ts.isoformat(),
                "_ts": ts,
            })
        return rows

    def test_planted_spike_is_flagged(self):
        """phone_a makes 5 calls in 2 hours — should flag."""
        rows = self._cdr(self.phone_a, count=5, spread_hours=1.5)
        result = detect_communication_spikes(self._graph(), rows, self.cfg)
        assert self.entity_a in result, "Entity with spike should be flagged"
        flag_types = [f for f, _ in result[self.entity_a]]
        assert "COMMUNICATION_SPIKE" in flag_types

    def test_evidence_string_populated(self):
        rows = self._cdr(self.phone_a, count=5, spread_hours=1.5)
        result = detect_communication_spikes(self._graph(), rows, self.cfg)
        evidence_strings = [e for _, e in result[self.entity_a]]
        assert any(evidence_strings), "At least one evidence string required"
        assert any(str(self.cfg.spike_call_count_threshold) in e for e in evidence_strings)

    def test_clean_case_not_flagged(self):
        """phone_a makes only 2 calls — below threshold=3, must NOT flag."""
        rows = self._cdr(self.phone_a, count=2, spread_hours=1.0)
        result = detect_communication_spikes(self._graph(), rows, self.cfg)
        assert self.entity_a not in result, "Low-call-count entity must not be flagged"

    def test_spread_across_window_not_flagged(self):
        """5 calls spread over 10 hours — none occur in a 2-hour window in groups ≥ 3."""
        rows = self._cdr(self.phone_a, count=5, spread_hours=10.0)
        result = detect_communication_spikes(self._graph(), rows, self.cfg)
        # With 5 calls spread over 10h and window=2h, max in any 2h window depends
        # on exact spacing — with threshold=3, 5 calls evenly spread over 10h
        # gives ~1 call per 2 hours, so this should NOT trigger.
        if self.entity_a in result:
            flag_types = [f for f, _ in result[self.entity_a]]
            assert "COMMUNICATION_SPIKE" not in flag_types

    def test_unknown_phone_does_not_crash(self):
        """CDR rows with phones not in graph must be skipped gracefully."""
        rows = [{
            "caller": "0000000000",
            "callee": "1111111111",
            "timestamp": self.base.isoformat(),
            "_ts": self.base,
        }]
        result = detect_communication_spikes(self._graph(), rows, self.cfg)
        # Should produce an empty or clean result, no exception
        assert isinstance(result, dict)

    def test_empty_cdr_produces_no_flags(self):
        result = detect_communication_spikes(self._graph(), [], self.cfg)
        assert result == {} or all(len(v) == 0 for v in result.values())


# ---------------------------------------------------------------------------
# M3.4 — Transaction Anomaly tests
# ---------------------------------------------------------------------------

class TestTransactionAnomalies:
    cfg = M3Config(
        txn_frequency_multiplier=2.0,
        txn_min_window_count=3,
        txn_large_amount_inr=100_000.0,
        spike_window_hours=24.0,
    )
    base = datetime(2025, 6, 1, 12, 0, 0)
    entity_a = "ACCT_A"
    acct_a = "ACC001"

    def _graph(self):
        return _make_account_graph([(self.acct_a, self.entity_a)])

    def test_large_transaction_flagged(self):
        rows = [_txn_row(self.acct_a, "ACC_OTHER", self.base, amount=200_000.0)]
        result = detect_transaction_anomalies(self._graph(), rows, self.cfg)
        assert self.entity_a in result
        flag_types = [f for f, _ in result[self.entity_a]]
        assert "HIGH_TRANSACTION_FREQUENCY" in flag_types

    def test_large_transaction_evidence_has_amount(self):
        rows = [_txn_row(self.acct_a, "ACC_OTHER", self.base, amount=200_000.0)]
        result = detect_transaction_anomalies(self._graph(), rows, self.cfg)
        evidence = [e for _, e in result[self.entity_a]]
        assert any("200,000" in e or "200000" in e for e in evidence)

    def test_small_transaction_not_flagged(self):
        """Transactions below threshold must not flag."""
        rows = [_txn_row(self.acct_a, "ACC_OTHER", self.base, amount=5_000.0)]
        result = detect_transaction_anomalies(self._graph(), rows, self.cfg)
        if self.entity_a in result:
            flag_types = [f for f, _ in result[self.entity_a]]
            assert "HIGH_TRANSACTION_FREQUENCY" not in flag_types

    def test_empty_transactions_produces_no_flags(self):
        result = detect_transaction_anomalies(self._graph(), [], self.cfg)
        assert result == {} or all(len(v) == 0 for v in result.values())

    def test_unknown_account_does_not_crash(self):
        rows = [_txn_row("UNKNOWN_ACC", "ALSO_UNKNOWN", self.base, amount=500_000.0)]
        result = detect_transaction_anomalies(self._graph(), rows, self.cfg)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# M3.5 — Shared Account / Rapid Fund Movement tests
# ---------------------------------------------------------------------------

class TestSharedAccountAndRapidMovement:
    cfg = M3Config(
        multi_suspect_min_senders=2,
        rapid_fund_hours=6.0,
        rapid_fund_min_hops=2,
    )
    base = datetime(2025, 6, 1, 12, 0, 0)

    def _graph_with_accounts(self, acct_entity_pairs):
        return _make_account_graph(acct_entity_pairs)

    def test_shared_account_flagged(self):
        """Account C receives from A AND B — should flag MULTI_SUSPECT_SHARED_ACCOUNT."""
        g = self._graph_with_accounts([
            ("ACC_A", "ENT_A"), ("ACC_B", "ENT_B"), ("ACC_C", "ENT_C"),
        ])
        rows = [
            _txn_row("ACC_A", "ACC_C", self.base, 0.0),
            _txn_row("ACC_B", "ACC_C", self.base, 0.5),
        ]
        result = detect_shared_account_and_rapid_movement(g, rows, self.cfg)
        assert "ENT_C" in result
        flag_types = [f for f, _ in result["ENT_C"]]
        assert "MULTI_SUSPECT_SHARED_ACCOUNT" in flag_types

    def test_single_sender_not_flagged(self):
        """Account C only receives from A — min_senders=2, so should NOT flag."""
        g = self._graph_with_accounts([
            ("ACC_A", "ENT_A"), ("ACC_C", "ENT_C"),
        ])
        rows = [
            _txn_row("ACC_A", "ACC_C", self.base, 0.0),
        ]
        result = detect_shared_account_and_rapid_movement(g, rows, self.cfg)
        if "ENT_C" in result:
            flag_types = [f for f, _ in result["ENT_C"]]
            assert "MULTI_SUSPECT_SHARED_ACCOUNT" not in flag_types

    def test_rapid_fund_movement_flagged(self):
        """A→B→C all within 4h, min_hops=2, window=6h — should flag RAPID_FUND_MOVEMENT."""
        g = self._graph_with_accounts([
            ("ACC_A", "ENT_A"), ("ACC_B", "ENT_B"), ("ACC_C", "ENT_C"),
        ])
        rows = [
            _txn_row("ACC_A", "ACC_B", self.base, 0.0),
            _txn_row("ACC_B", "ACC_C", self.base, 2.0),   # 2h later
        ]
        result = detect_shared_account_and_rapid_movement(g, rows, self.cfg)
        flagged_rapid = [eid for eid, pairs in result.items()
                         if any(f == "RAPID_FUND_MOVEMENT" for f, _ in pairs)]
        assert len(flagged_rapid) > 0, "Rapid chain should be flagged"

    def test_slow_fund_movement_not_flagged(self):
        """A→B→C over 10h, window=6h — should NOT flag RAPID_FUND_MOVEMENT."""
        g = self._graph_with_accounts([
            ("ACC_A", "ENT_A"), ("ACC_B", "ENT_B"), ("ACC_C", "ENT_C"),
        ])
        rows = [
            _txn_row("ACC_A", "ACC_B", self.base, 0.0),
            _txn_row("ACC_B", "ACC_C", self.base, 10.0),  # 10h later — outside window
        ]
        result = detect_shared_account_and_rapid_movement(g, rows, self.cfg)
        rapid_flags = [eid for eid, pairs in result.items()
                       if any(f == "RAPID_FUND_MOVEMENT" for f, _ in pairs)]
        assert len(rapid_flags) == 0, "Slow chain should not be flagged as rapid"

    def test_shared_account_evidence_populated(self):
        g = self._graph_with_accounts([
            ("ACC_A", "ENT_A"), ("ACC_B", "ENT_B"), ("ACC_C", "ENT_C"),
        ])
        rows = [
            _txn_row("ACC_A", "ACC_C", self.base, 0.0),
            _txn_row("ACC_B", "ACC_C", self.base, 0.5),
        ]
        result = detect_shared_account_and_rapid_movement(g, rows, self.cfg)
        for _, e in result.get("ENT_C", []):
            assert e != "", "Evidence string must not be empty"


# ---------------------------------------------------------------------------
# M3.6 — Incident Timing Cluster tests
# ---------------------------------------------------------------------------

class TestIncidentTimingCluster:
    cfg = M3Config(incident_window_days=2, incident_min_activity=2)
    incident_date = "2025-06-15"
    inc_dt = datetime(2025, 6, 15)
    phone_e = "9999999999"
    entity_e = "PHONE_ENT"

    def _graph(self):
        return _make_phone_graph([(self.phone_e, self.entity_e)])

    def _near_cdr(self, count: int) -> list[dict]:
        return [
            {
                "caller": self.phone_e,
                "callee": "0000000000",
                "timestamp": (self.inc_dt + timedelta(hours=i)).isoformat(),
                "_ts": self.inc_dt + timedelta(hours=i),
            }
            for i in range(count)
        ]

    def test_planted_activity_near_incident_flagged(self):
        """3 calls on incident date should flag."""
        cdr = self._near_cdr(3)
        result = detect_incident_timing_clusters(
            self._graph(), cdr, [], [self.incident_date], self.cfg
        )
        assert self.entity_e in result
        flag_types = [f for f, _ in result[self.entity_e]]
        assert "INCIDENT_TIMING_CLUSTER" in flag_types

    def test_evidence_mentions_incident_date(self):
        cdr = self._near_cdr(3)
        result = detect_incident_timing_clusters(
            self._graph(), cdr, [], [self.incident_date], self.cfg
        )
        evidence_strs = [e for _, e in result[self.entity_e]]
        assert any(self.incident_date in e for e in evidence_strs)

    def test_no_activity_near_incident_not_flagged(self):
        """Activity from 30 days before incident should NOT flag."""
        far_ts = self.inc_dt - timedelta(days=30)
        cdr = [{
            "caller": self.phone_e,
            "callee": "0000",
            "timestamp": far_ts.isoformat(),
            "_ts": far_ts,
        }]
        result = detect_incident_timing_clusters(
            self._graph(), cdr, [], [self.incident_date], self.cfg
        )
        if self.entity_e in result:
            flag_types = [f for f, _ in result[self.entity_e]]
            assert "INCIDENT_TIMING_CLUSTER" not in flag_types

    def test_no_incident_dates_produces_no_flags(self):
        cdr = self._near_cdr(5)
        result = detect_incident_timing_clusters(self._graph(), cdr, [], [], self.cfg)
        assert result == {}

    def test_malformed_incident_date_graceful_skip(self):
        cdr = self._near_cdr(3)
        result = detect_incident_timing_clusters(
            self._graph(), cdr, [], ["not-a-date"], self.cfg
        )
        # Should return empty without crashing
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# run_pattern_rules() — end-to-end with real M2 data
# ---------------------------------------------------------------------------

class TestRunPatternRulesIntegration:
    def test_real_data_no_crash(self):
        e_path = Path("M1/output/entities.json")
        r_path = Path("M1/output/relationships.json")
        cdr_path = "M1/data/cdrs/cdr.csv"
        txn_path = "M1/data/transactions/transactions.csv"
        if not e_path.exists():
            pytest.skip("Real M1/M2 output not available")
        from M2.graph_builder import load_graph_from_m1_output
        graph = load_graph_from_m1_output(str(e_path), str(r_path))
        result = run_pattern_rules(graph, cdr_path, txn_path)
        assert isinstance(result, dict)
        # All flag types must be from the valid set
        from M3.models import VALID_FLAG_TYPES
        for entity_id, flag_pairs in result.items():
            for flag_type, evidence in flag_pairs:
                assert flag_type in VALID_FLAG_TYPES, f"Unknown flag type: {flag_type}"
                assert evidence != "", f"Evidence string must not be empty for {entity_id}"

    def test_empty_graph_raises(self):
        with pytest.raises(ValueError):
            run_pattern_rules(nx.MultiDiGraph(), "no_cdr.csv", "no_txn.csv")
