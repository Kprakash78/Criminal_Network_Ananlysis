"""
tests/test_event_chains.py
PS 26152 — Criminal Network Analysis System
------------------------------------------
Tests for M3_feature/event_causality.py.

Verifies:
  - Chains are returned as a list of dicts with required keys.
  - Steps within each chain are chronologically ordered.
  - Pivot detection is within bounds.
  - build_chains() works for each demo case.
  - save_chains() writes JSON to both output paths.
"""

import json
from datetime import datetime
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def all_chains():
    """Build chains for all cases once per test module."""
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from M3_feature.event_causality import build_all_chains, save_chains
    chains = build_all_chains()
    save_chains(chains)
    return chains


class TestChainStructure:
    def test_returns_dict_of_lists(self, all_chains):
        assert isinstance(all_chains, dict)
        for case_id, chains in all_chains.items():
            assert isinstance(chains, list), f"{case_id} chains not a list"

    def test_each_chain_has_required_keys(self, all_chains):
        required = {"chain_id", "case_id", "n_steps", "steps", "pivot", "note"}
        for case_id, chains in all_chains.items():
            for c in chains:
                missing = required - c.keys()
                assert not missing, f"{c['chain_id']} missing keys: {missing}"

    def test_chain_id_matches_case(self, all_chains):
        for case_id, chains in all_chains.items():
            for c in chains:
                assert c["case_id"] == case_id

    def test_n_steps_matches_actual(self, all_chains):
        for case_id, chains in all_chains.items():
            for c in chains:
                assert c["n_steps"] == len(c["steps"])

    def test_steps_are_non_empty(self, all_chains):
        for case_id, chains in all_chains.items():
            for c in chains:
                assert len(c["steps"]) > 0, f"{c['chain_id']} has no steps"


class TestChainOrdering:
    def test_steps_chronologically_ordered(self, all_chains):
        """Steps within each chain must be in non-decreasing timestamp order."""
        for case_id, chains in all_chains.items():
            for c in chains:
                times = [datetime.fromisoformat(s["t"]) for s in c["steps"]]
                assert times == sorted(times), (
                    f"{c['chain_id']}: steps not in chronological order"
                )

    def test_step_has_provenance(self, all_chains):
        """Each step must include source file and line number."""
        for case_id, chains in all_chains.items():
            for c in chains:
                for step in c["steps"]:
                    assert "source" in step, f"Step missing 'source'"
                    assert "line"   in step, f"Step missing 'line'"


class TestPivotDetection:
    def test_pivot_index_in_bounds(self, all_chains):
        for case_id, chains in all_chains.items():
            for c in chains:
                pivot_idx = c["pivot"]["step_index"]
                assert 0 <= pivot_idx < len(c["steps"]), (
                    f"{c['chain_id']}: pivot index {pivot_idx} out of range"
                )

    def test_pivot_score_in_range(self, all_chains):
        for case_id, chains in all_chains.items():
            for c in chains:
                score = c["pivot"]["score"]
                assert 0.0 <= score <= 1.0, (
                    f"{c['chain_id']}: pivot score {score} out of [0,1]"
                )


class TestSaveChains:
    def test_output_files_exist(self):
        feat_out  = REPO_ROOT / "demo_cache" / "feature_outputs" / "event_chains.json"
        m3_out    = REPO_ROOT / "M3_feature" / "event_chains.json"
        assert feat_out.exists(), f"Missing: {feat_out}"
        assert m3_out.exists(),   f"Missing: {m3_out}"

    def test_output_json_valid(self):
        p = REPO_ROOT / "demo_cache" / "feature_outputs" / "event_chains.json"
        data = json.loads(p.read_text())
        assert "chains" in data
        assert "seed"   in data

    def test_all_cases_present(self):
        p = REPO_ROOT / "demo_cache" / "feature_outputs" / "event_chains.json"
        data = json.loads(p.read_text())
        case_ids = {c["case_id"] for c in data["chains"]}
        for expected in ("case_A", "case_B", "case_C"):
            assert expected in case_ids, f"case {expected} missing from chains"


class TestSafeLanguage:
    def test_note_field_mentions_verification(self, all_chains):
        """All chain notes must mention 'investigator verification'."""
        for case_id, chains in all_chains.items():
            for c in chains:
                note = c.get("note", "")
                assert "verification" in note.lower(), (
                    f"{c['chain_id']} note missing 'verification'"
                )

    def test_no_guilt_language(self, all_chains):
        forbidden = {"guilty", "criminal", "perpetrator", "accused"}
        for case_id, chains in all_chains.items():
            for c in chains:
                text = json.dumps(c).lower()
                found = [w for w in forbidden if w in text]
                assert not found, (
                    f"{c['chain_id']} contains forbidden language: {found}"
                )
