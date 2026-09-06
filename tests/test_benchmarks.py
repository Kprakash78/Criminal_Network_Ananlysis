"""
tests/test_benchmarks.py
PS 26152 — Criminal Network Analysis System
--------------------------------------------
Pytest wrapper that validates the benchmark pipeline end-to-end.

Tests:
  1. Demo data exists (generate_demo_data.py ran successfully).
  2. Demo cache exists (prepare_demo_cache.py ran successfully).
  3. scorecard.json exists with all required top-level keys.
  4. Key metrics are present and within expected types.
  5. Determinism test passes (n_passed == n_files_checked).
  6. Extraction metrics have valid F1 range [0, 1].
  7. Link-prediction AUC is in range [0, 1].
  8. Candidate coverage is a float in [0, 1].
  9. Latency values are positive numbers.
 10. Scorecard markdown exists and contains required sections.

Usage:
    pytest -q tests/test_benchmarks.py
    # or after run_benchmarks.sh:
    pytest -q
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR  = REPO_ROOT / "demo_dataset"
CACHE_DIR = REPO_ROOT / "demo_cache"
RESULTS   = REPO_ROOT / "results"


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def run_pipeline():
    """Run the full pipeline once per session using SKIP_VENV=1."""
    import os, subprocess
    env = os.environ.copy()
    env["SKIP_VENV"] = "1"
    env["PYTHONPATH"] = str(REPO_ROOT)
    result = subprocess.run(
        ["bash", "scripts/run_benchmarks.sh"],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    return result


@pytest.fixture(scope="session")
def scorecard(run_pipeline):
    path = RESULTS / "scorecard.json"
    assert path.exists(), (
        f"scorecard.json not found at {path}\n"
        f"STDOUT: {run_pipeline.stdout[-2000:]}\n"
        f"STDERR: {run_pipeline.stderr[-2000:]}"
    )
    return json.loads(path.read_text())


# ─── Tests ───────────────────────────────────────────────────────────────────

class TestPipelineRan:
    def test_pipeline_exit_code(self, run_pipeline):
        """run_benchmarks.sh should complete without fatal error."""
        # We allow non-zero only if determinism was the sole failure
        # (exit code from determinism sub-process is captured separately in sh)
        assert run_pipeline.returncode == 0, (
            f"run_benchmarks.sh exited {run_pipeline.returncode}\n"
            f"STDERR: {run_pipeline.stderr[-3000:]}"
        )

    def test_demo_data_exists(self, run_pipeline):
        gt = DEMO_DIR / "ground_truth.json"
        assert gt.exists(), f"ground_truth.json missing: {gt}"
        data = json.loads(gt.read_text())
        assert len(data) >= 3, "Expected at least 3 demo cases"

    def test_fir_files_exist(self, run_pipeline):
        firs = list((DEMO_DIR / "fir_files").glob("*.txt"))
        assert len(firs) >= 3, f"Expected ≥3 FIR files, found {len(firs)}"

    def test_cdr_files_exist(self, run_pipeline):
        cdrs = list((DEMO_DIR / "cdr_files").glob("*.csv"))
        assert len(cdrs) >= 3

    def test_txn_files_exist(self, run_pipeline):
        txns = list((DEMO_DIR / "transaction_files").glob("*.csv"))
        assert len(txns) >= 3


class TestDemoCache:
    def test_cache_manifest_exists(self, run_pipeline):
        manifest = CACHE_DIR / "cache_manifest.json"
        assert manifest.exists(), f"cache_manifest.json missing: {manifest}"

    def test_cache_entities_exist(self, run_pipeline):
        for case in ["case_A", "case_B", "case_C"]:
            p = CACHE_DIR / f"{case}_entities.json"
            assert p.exists(), f"Missing cache file: {p}"

    def test_cache_top3_exist(self, run_pipeline):
        for case in ["case_A", "case_B", "case_C"]:
            p = CACHE_DIR / f"{case}_top3.json"
            assert p.exists(), f"Missing cache file: {p}"

    def test_search_index_exists(self, run_pipeline):
        assert (CACHE_DIR / "search_index.json").exists()


class TestScorecardStructure:
    def test_scorecard_json_exists(self, run_pipeline):
        assert (RESULTS / "scorecard.json").exists()

    def test_scorecard_md_exists(self, run_pipeline):
        assert (RESULTS / "scorecard.md").exists()

    def test_top_level_keys(self, scorecard):
        for key in ("meta", "metrics", "thresholds", "acceptance_checks", "overall_pass"):
            assert key in scorecard, f"Missing key '{key}' in scorecard"

    def test_metrics_keys(self, scorecard):
        m = scorecard["metrics"]
        for key in ("extraction", "link_prediction", "candidate_coverage",
                    "latency_ms", "determinism"):
            assert key in m, f"Missing metrics key: {key}"


class TestExtractionMetrics:
    def test_extraction_entity_types_present(self, scorecard):
        ext = scorecard["metrics"]["extraction"]
        for etype in ("names", "phones", "dates", "amounts"):
            assert etype in ext, f"Missing extraction type: {etype}"

    def test_extraction_f1_in_range(self, scorecard):
        ext = scorecard["metrics"]["extraction"]
        for etype, m in ext.items():
            f1 = m.get("macro_f1")
            assert f1 is not None, f"macro_f1 missing for {etype}"
            assert 0.0 <= f1 <= 1.0, f"{etype} F1={f1} out of range"

    def test_extraction_names_f1_meets_threshold(self, scorecard):
        f1 = scorecard["metrics"]["extraction"]["names"].get("macro_f1", 0)
        assert f1 >= 0.80, f"Names F1={f1:.3f} below threshold 0.80"

    def test_extraction_phones_f1_meets_threshold(self, scorecard):
        f1 = scorecard["metrics"]["extraction"]["phones"].get("macro_f1", 0)
        assert f1 >= 0.90, f"Phones F1={f1:.3f} below threshold 0.90"


class TestLinkPrediction:
    def test_auc_present(self, scorecard):
        auc = scorecard["metrics"]["link_prediction"]["mean_auc_roc"]
        assert auc is not None, "mean_auc_roc is None"

    def test_auc_in_range(self, scorecard):
        auc = scorecard["metrics"]["link_prediction"]["mean_auc_roc"]
        assert 0.0 <= auc <= 1.0, f"AUC={auc} out of range"

    def test_auc_meets_threshold(self, scorecard):
        auc = scorecard["metrics"]["link_prediction"]["mean_auc_roc"]
        # Threshold is 0.55 — realistic for a common-neighbour heuristic
        # on a small synthetic graph with only ~8 unique edges per case.
        assert auc >= 0.55, f"AUC={auc:.3f} below threshold 0.55"

    def test_precision_at_5_present(self, scorecard):
        p5 = scorecard["metrics"]["link_prediction"]["mean_precision_at_5"]
        assert p5 is not None


class TestCandidateCoverage:
    def test_coverage_present(self, scorecard):
        cov = scorecard["metrics"]["candidate_coverage"]["value"]
        assert cov is not None, "candidate_coverage.value is None"

    def test_coverage_in_range(self, scorecard):
        cov = scorecard["metrics"]["candidate_coverage"]["value"]
        assert 0.0 <= cov <= 1.0, f"Coverage={cov} out of range"

    def test_coverage_meets_threshold(self, scorecard):
        cov = scorecard["metrics"]["candidate_coverage"]["value"]
        assert cov >= 0.67, f"Candidate coverage={cov:.3f} below threshold 0.67"


class TestLatency:
    @pytest.mark.parametrize("module,max_ms", [
        ("parse",       50),
        ("graph_build", 100),
        ("search",      20),
        ("top3",        100),
    ])
    def test_latency_positive(self, scorecard, module, max_ms):
        val = scorecard["metrics"]["latency_ms"][module].get("median_ms")
        assert val is not None, f"median_ms missing for {module}"
        # Search can legitimately be 0.0ms on fast hardware (TF-IDF lookup)
        assert val >= 0, f"{module} median_ms={val} should be non-negative"

    @pytest.mark.parametrize("module,max_ms", [
        ("parse",       50),
        ("graph_build", 100),
        ("search",      20),
        ("top3",        100),
    ])
    def test_latency_within_threshold(self, scorecard, module, max_ms):
        val = scorecard["metrics"]["latency_ms"][module].get("median_ms")
        assert val is not None
        assert val <= max_ms, (
            f"{module} median latency {val:.1f}ms exceeds threshold {max_ms}ms"
        )


class TestDeterminism:
    def test_determinism_key_present(self, scorecard):
        assert "determinism" in scorecard["metrics"]

    def test_determinism_pass(self, scorecard):
        d = scorecard["metrics"]["determinism"]
        n_checked = d.get("n_files_checked", 0)
        n_passed  = d.get("n_passed", 0)
        assert n_checked > 0, "No files checked in determinism test"
        assert n_passed == n_checked, (
            f"Determinism: only {n_passed}/{n_checked} files match cached outputs"
        )


class TestScorecardMarkdown:
    def test_md_sections(self, run_pipeline):
        md_path = RESULTS / "scorecard.md"
        if not md_path.exists():
            pytest.skip("scorecard.md not generated yet")
        text = md_path.read_text()
        required_sections = [
            "What we measured",
            "Key Numbers",
            "How to Reproduce",
            "Caveats",
            "Candidate coverage",       # capital C as generated by scorecard
            "investigator verification",
        ]
        for section in required_sections:
            assert section in text, f"Missing section/phrase in scorecard.md: '{section}'"
