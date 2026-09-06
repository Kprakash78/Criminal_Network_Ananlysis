"""
Tests for M3_feature/top3.py — Top-3 Suspect Ranking
PS 26152 — AI-Powered Criminal Network Analysis System
"""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from M3_feature.top3 import (
    compute_degree_centrality,
    compute_temporal_anomaly,
    compute_colocation_score,
    compute_call_pattern_score,
    compute_transaction_anomaly,
    run_top3_analysis,
    PERSONS,
)


@pytest.fixture
def sample_cdr_rows():
    """Minimal CDR rows for testing."""
    return [
        {"caller": "9876543210", "callee": "9123456789", "timestamp": "2024-01-15T02:26:05",
         "duration_seconds": "240", "tower_location": "Bandra West, Mumbai"},
        {"caller": "9876543210", "callee": "9988776655", "timestamp": "2024-01-15T03:30:00",
         "duration_seconds": "15", "tower_location": "Bandra West, Mumbai"},
        {"caller": "9123456789", "callee": "9876543210", "timestamp": "2024-01-15T14:00:00",
         "duration_seconds": "300", "tower_location": "Park Street, Kolkata"},
        {"caller": "9876543210", "callee": "9654321098", "timestamp": "2024-01-16T04:10:00",
         "duration_seconds": "120", "tower_location": "Bandra West, Mumbai"},
    ]


@pytest.fixture
def sample_txn_rows():
    """Minimal transaction rows for testing."""
    return [
        {"source_account": "ACC00101", "target_account": "ACC00106",
         "amount_inr": "150000", "timestamp": "2024-01-15T10:00:00", "remarks": "Payment"},
        {"source_account": "ACC00106", "target_account": "ACC00101",
         "amount_inr": "75000", "timestamp": "2024-01-16T11:00:00", "remarks": "Loan"},
        {"source_account": "ACC00101", "target_account": "ACC00103",
         "amount_inr": "50000", "timestamp": "2024-01-17T09:00:00", "remarks": "Transfer"},
    ]


class TestDegreeentrality:
    def test_returns_float_in_range(self, sample_cdr_rows, sample_txn_rows):
        person = PERSONS[0]  # Ravi Kumar
        score = compute_degree_centrality(person, sample_cdr_rows, sample_txn_rows)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_person_with_connections_scores_higher(self, sample_cdr_rows, sample_txn_rows):
        ravi = PERSONS[0]  # Has calls + transactions
        priya = PERSONS[3]  # No connections in sample data
        ravi_score = compute_degree_centrality(ravi, sample_cdr_rows, sample_txn_rows)
        priya_score = compute_degree_centrality(priya, sample_cdr_rows, sample_txn_rows)
        assert ravi_score > priya_score

    def test_empty_data_returns_zero(self):
        score = compute_degree_centrality(PERSONS[0], [], [])
        assert score == 0.0


class TestTemporalAnomaly:
    def test_returns_float_in_range(self, sample_cdr_rows):
        score = compute_temporal_anomaly(PERSONS[0], sample_cdr_rows)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_late_night_calls_increase_score(self, sample_cdr_rows):
        # Ravi has late-night calls (02:26, 03:30, 04:10)
        score = compute_temporal_anomaly(PERSONS[0], sample_cdr_rows)
        assert score > 0.0

    def test_empty_data_returns_zero(self):
        score = compute_temporal_anomaly(PERSONS[0], [])
        assert score == 0.0


class TestColocationScore:
    def test_returns_float_in_range(self, sample_cdr_rows):
        score = compute_colocation_score(PERSONS[0], sample_cdr_rows)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_same_location_increases_score(self, sample_cdr_rows):
        # Ravi has multiple events at Bandra West
        score = compute_colocation_score(PERSONS[0], sample_cdr_rows)
        assert score > 0.0


class TestCallPatternScore:
    def test_returns_float_in_range(self, sample_cdr_rows):
        score = compute_call_pattern_score(PERSONS[0], sample_cdr_rows)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_active_caller_scores_higher(self, sample_cdr_rows):
        ravi = compute_call_pattern_score(PERSONS[0], sample_cdr_rows)  # 3 outgoing calls
        kavya = compute_call_pattern_score(PERSONS[9], sample_cdr_rows)  # no calls
        assert ravi > kavya


class TestTransactionAnomaly:
    def test_returns_float_in_range(self, sample_txn_rows):
        score = compute_transaction_anomaly(PERSONS[0], sample_txn_rows)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_large_txn_increases_score(self, sample_txn_rows):
        # Ravi has a ₹150,000 transaction
        ravi_score = compute_transaction_anomaly(PERSONS[0], sample_txn_rows)
        assert ravi_score > 0.0

    def test_no_transactions_returns_zero(self):
        score = compute_transaction_anomaly(PERSONS[3], [])  # Priya
        assert score == 0.0


class TestRunTop3Analysis:
    def test_produces_valid_output(self, tmp_path):
        """Run full analysis against demo dataset and validate schema."""
        data_dir = REPO_ROOT / "data"
        if not data_dir.exists():
            pytest.skip("Demo data not available")

        output_path = tmp_path / "top3_results.json"
        result = run_top3_analysis(data_dir=data_dir, output_path=output_path)

        # Schema validation
        assert "suspects" in result
        assert len(result["suspects"]) == 3

        for suspect in result["suspects"]:
            assert "id" in suspect
            assert "name" in suspect
            assert "risk_score" in suspect
            assert isinstance(suspect["risk_score"], (int, float))
            assert 0 <= suspect["risk_score"] <= 100
            assert "evidence" in suspect
            assert "explanation" in suspect
            assert len(suspect["explanation"]) > 0

        # Verify evidence items have provenance
        for suspect in result["suspects"]:
            for ev in suspect["evidence"]:
                assert "file" in ev
                assert "line_start" in ev
                assert "line_end" in ev
                assert "text_snippet" in ev

    def test_suspects_are_sorted_by_score(self, tmp_path):
        """Verify suspects are sorted descending by risk_score."""
        data_dir = REPO_ROOT / "data"
        if not data_dir.exists():
            pytest.skip("Demo data not available")

        output_path = tmp_path / "top3_results.json"
        result = run_top3_analysis(data_dir=data_dir, output_path=output_path)

        scores = [s["risk_score"] for s in result["suspects"]]
        assert scores == sorted(scores, reverse=True)

    def test_explanation_has_three_bullets(self, tmp_path):
        """Each explanation should contain 3 numbered bullets."""
        data_dir = REPO_ROOT / "data"
        if not data_dir.exists():
            pytest.skip("Demo data not available")

        output_path = tmp_path / "top3_results.json"
        result = run_top3_analysis(data_dir=data_dir, output_path=output_path)

        for suspect in result["suspects"]:
            explanation = suspect["explanation"]
            # Should contain "1)", "2)", "3)"
            assert "1)" in explanation
            assert "2)" in explanation
            assert "3)" in explanation
