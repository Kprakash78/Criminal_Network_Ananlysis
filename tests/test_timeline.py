"""
Tests for M6_feature/timeline_player.py — Timeline Event Player
PS 26152 — AI-Powered Criminal Network Analysis System
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from M6_feature.timeline_player import build_timeline


class TestBuildTimeline:
    def test_returns_list(self):
        """build_timeline should return a list."""
        data_dir = REPO_ROOT / "data"
        if not (data_dir / "cdrs" / "cdr.csv").exists():
            pytest.skip("Demo data not available")

        events = build_timeline()
        assert isinstance(events, list)
        assert len(events) > 0

    def test_events_sorted_chronologically(self):
        """Events must be sorted by timestamp ascending."""
        data_dir = REPO_ROOT / "data"
        if not (data_dir / "cdrs" / "cdr.csv").exists():
            pytest.skip("Demo data not available")

        events = build_timeline()
        timestamps = [e["t"] for e in events]
        assert timestamps == sorted(timestamps), "Events are not sorted chronologically"

    def test_event_has_required_fields(self):
        """Each event must have 't', 'type', 'file', 'line', 'caption'."""
        data_dir = REPO_ROOT / "data"
        if not (data_dir / "cdrs" / "cdr.csv").exists():
            pytest.skip("Demo data not available")

        events = build_timeline()
        for event in events[:20]:  # Check first 20
            assert "t" in event, f"Missing 't' in event: {event}"
            assert "type" in event, f"Missing 'type' in event: {event}"
            assert "file" in event, f"Missing 'file' in event: {event}"
            assert "line" in event, f"Missing 'line' in event: {event}"
            assert "caption" in event, f"Missing 'caption' in event: {event}"

    def test_event_types_are_valid(self):
        """Event types must be one of: call, transaction, fir_filing."""
        data_dir = REPO_ROOT / "data"
        if not (data_dir / "cdrs" / "cdr.csv").exists():
            pytest.skip("Demo data not available")

        events = build_timeline()
        valid_types = {"call", "transaction", "fir_filing"}
        for event in events:
            assert event["type"] in valid_types, \
                f"Invalid event type: {event['type']}"

    def test_captions_are_non_empty(self):
        """Every event caption must be a non-empty string."""
        data_dir = REPO_ROOT / "data"
        if not (data_dir / "cdrs" / "cdr.csv").exists():
            pytest.skip("Demo data not available")

        events = build_timeline()
        for event in events:
            assert isinstance(event["caption"], str)
            assert len(event["caption"]) > 0

    def test_file_provenance_is_relative(self):
        """File paths should be relative (not absolute)."""
        data_dir = REPO_ROOT / "data"
        if not (data_dir / "cdrs" / "cdr.csv").exists():
            pytest.skip("Demo data not available")

        events = build_timeline()
        for event in events:
            assert not event["file"].startswith("/"), \
                f"File path should be relative, got: {event['file']}"

    def test_line_numbers_are_positive(self):
        """Line numbers must be positive integers."""
        data_dir = REPO_ROOT / "data"
        if not (data_dir / "cdrs" / "cdr.csv").exists():
            pytest.skip("Demo data not available")

        events = build_timeline()
        for event in events:
            assert isinstance(event["line"], int)
            assert event["line"] > 0

    def test_call_events_have_from_and_to(self):
        """Call events must have 'from' and 'to' fields."""
        data_dir = REPO_ROOT / "data"
        if not (data_dir / "cdrs" / "cdr.csv").exists():
            pytest.skip("Demo data not available")

        events = build_timeline()
        call_events = [e for e in events if e["type"] == "call"]
        assert len(call_events) > 0, "Expected at least one call event"

        for event in call_events:
            assert "from" in event
            assert "to" in event
            assert len(event["from"]) > 0
            assert len(event["to"]) > 0

    def test_transaction_events_have_amount(self):
        """Transaction events must have an 'amount' field."""
        data_dir = REPO_ROOT / "data"
        if not (data_dir / "transactions" / "transactions.csv").exists():
            pytest.skip("Demo data not available")

        events = build_timeline()
        txn_events = [e for e in events if e["type"] == "transaction"]
        assert len(txn_events) > 0, "Expected at least one transaction event"

        for event in txn_events:
            assert "amount" in event
            assert isinstance(event["amount"], (int, float))
            assert event["amount"] > 0

    def test_includes_all_event_types(self):
        """Timeline should include calls, transactions, and FIR filings."""
        data_dir = REPO_ROOT / "data"
        if not (data_dir / "cdrs" / "cdr.csv").exists():
            pytest.skip("Demo data not available")

        events = build_timeline()
        types = {e["type"] for e in events}
        assert "call" in types, "Missing call events"
        assert "transaction" in types, "Missing transaction events"
        assert "fir_filing" in types, "Missing FIR filing events"
