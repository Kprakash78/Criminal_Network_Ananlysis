"""
tests/test_ghost_node.py
PS 26152 — Criminal Network Analysis System
------------------------------------------
Tests for M3_feature/ghost_node.py.

Verifies:
  - suggest_ghosts() returns a list with required fields.
  - Scores are valid floats in [0, 1] (composite, jaccard) or >= 0 (AA).
  - Explanation strings contain safe language.
  - Evidence pointers (cdr_file, txn_file) are present.
  - Output files are written correctly.
  - Cross-cluster flag is a boolean.
"""

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def all_suggestions():
    """Run ghost-node finder for all cases once."""
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from M3_feature.ghost_node import suggest_all, save_suggestions
    suggs = suggest_all(top_k=10)
    save_suggestions(suggs)
    return suggs


class TestSuggestionStructure:
    def test_returns_dict_of_lists(self, all_suggestions):
        assert isinstance(all_suggestions, dict)
        for case_id, suggs in all_suggestions.items():
            assert isinstance(suggs, list), f"{case_id} not a list"

    def test_each_suggestion_has_required_keys(self, all_suggestions):
        required = {"pair", "raw_ids", "scores", "explanation",
                    "evidence", "cross_cluster"}
        for case_id, suggs in all_suggestions.items():
            for s in suggs:
                missing = required - s.keys()
                assert not missing, f"Suggestion missing keys: {missing}"

    def test_pair_has_two_elements(self, all_suggestions):
        for case_id, suggs in all_suggestions.items():
            for s in suggs:
                assert len(s["pair"]) == 2, "pair must have exactly 2 elements"

    def test_cross_cluster_is_bool(self, all_suggestions):
        for case_id, suggs in all_suggestions.items():
            for s in suggs:
                assert isinstance(s["cross_cluster"], bool)


class TestScores:
    def test_composite_non_negative(self, all_suggestions):
        for case_id, suggs in all_suggestions.items():
            for s in suggs:
                comp = s["scores"]["composite"]
                assert comp >= 0, f"Negative composite score: {comp}"

    def test_jaccard_in_range(self, all_suggestions):
        for case_id, suggs in all_suggestions.items():
            for s in suggs:
                jac = s["scores"]["jaccard"]
                assert 0.0 <= jac <= 1.0, f"Jaccard {jac} out of [0,1]"

    def test_adamic_adar_non_negative(self, all_suggestions):
        for case_id, suggs in all_suggestions.items():
            for s in suggs:
                aa = s["scores"]["adamic_adar"]
                assert aa >= 0, f"Negative Adamic-Adar: {aa}"

    def test_common_neighbours_non_negative(self, all_suggestions):
        for case_id, suggs in all_suggestions.items():
            for s in suggs:
                cn = s["scores"]["common_neighbours"]
                assert cn >= 0

    def test_suggestions_sorted_by_composite_desc(self, all_suggestions):
        for case_id, suggs in all_suggestions.items():
            composites = [s["scores"]["composite"] for s in suggs]
            assert composites == sorted(composites, reverse=True), (
                f"{case_id}: suggestions not sorted by composite score"
            )


class TestEvidence:
    def test_evidence_has_cdr_and_txn(self, all_suggestions):
        for case_id, suggs in all_suggestions.items():
            for s in suggs:
                ev = s["evidence"]
                assert "cdr_file" in ev, "evidence missing cdr_file"
                assert "txn_file" in ev, "evidence missing txn_file"

    def test_evidence_files_point_to_real_paths(self, all_suggestions):
        for case_id, suggs in all_suggestions.items():
            for s in suggs[:2]:  # check first 2 per case
                ev = s["evidence"]
                cdr = REPO_ROOT / ev["cdr_file"]
                txn = REPO_ROOT / ev["txn_file"]
                assert cdr.exists(), f"CDR file not found: {cdr}"
                assert txn.exists(), f"TXN file not found: {txn}"


class TestSafeLanguage:
    def test_explanation_mentions_verification(self, all_suggestions):
        for case_id, suggs in all_suggestions.items():
            for s in suggs:
                expl = s["explanation"].lower()
                assert "verification" in expl, (
                    f"Suggestion missing 'verification' in explanation"
                )

    def test_explanation_mentions_possible(self, all_suggestions):
        """Suggestions must be framed as 'possible', not assertions."""
        for case_id, suggs in all_suggestions.items():
            for s in suggs:
                expl = s["explanation"].lower()
                assert "possible" in expl or "potential" in expl or "suggest" in expl

    def test_no_guilt_language(self, all_suggestions):
        forbidden = {"guilty", "perpetrator", "criminal act", "accused"}
        for case_id, suggs in all_suggestions.items():
            for s in suggs:
                text = json.dumps(s).lower()
                found = [w for w in forbidden if w in text]
                assert not found, f"Forbidden language found: {found}"


class TestOutputFiles:
    def test_ghost_json_exists(self):
        p = REPO_ROOT / "demo_cache" / "feature_outputs" / "ghost_suggestions.json"
        assert p.exists(), f"Missing: {p}"

    def test_ghost_json_valid(self):
        p = REPO_ROOT / "demo_cache" / "feature_outputs" / "ghost_suggestions.json"
        data = json.loads(p.read_text())
        assert "suggestions" in data
        assert "threshold" in data

    def test_m3_output_exists(self):
        p = REPO_ROOT / "M3_feature" / "ghost_suggestions.json"
        assert p.exists(), f"Missing: {p}"
