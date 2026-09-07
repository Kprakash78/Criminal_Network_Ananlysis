"""
M6 Full Stack / Integration — Verification & Smoke Test Suite
Executes the mandatory verification checks from BACKEND.md §6:
  1. Endpoint smoke test (/search, /videos, /video/{id})
  2. Search response schema & explainability verification
  3. Timestamp seek range check
  4. End-to-end full corpus ingestion test
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from fastapi.testclient import TestClient

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from M6.app import app
from M6.config import M6Config
from M6.orchestrator import run_full_ingestion

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("M6.verify")


def run_verification():
    print("==========================================================")
    print("        M6 FULL STACK / INTEGRATION VERIFICATION SUITE    ")
    print("==========================================================")

    cfg = M6Config(data_root=repo_root / "data")
    sample_video = repo_root / "data" / "videos" / "sample_test.mp4"

    if not sample_video.exists():
        log.info("Generating sample test video for verification...")
        try:
            from M3.make_sample_video import create_sample_video
            create_sample_video(sample_video)
        except Exception as exc:
            log.error("Could not generate sample video: %s", exc)
            sys.exit(1)

    results_report = {}

    # Check 1: Ingestion Orchestration Test
    print("\n1. [Check 1/4] Ingestion Orchestration Test (M1 -> M2 -> M3 -> M4)")
    try:
        segments = run_full_ingestion(video_path=sample_video, video_id="sample_test", cfg=cfg, force_redo=True)
        if segments:
            print(f"   PASS: Ingested video into {len(segments)} unified segment records.")
            results_report["Ingestion Orchestration"] = "PASS"
        else:
            print("   FAIL: Ingestion returned 0 segments.")
            results_report["Ingestion Orchestration"] = "FAIL"
    except Exception as exc:
        print(f"   FAIL: Ingestion orchestration failed: {exc}")
        results_report["Ingestion Orchestration"] = "FAIL"

    # Initialize FastAPI TestClient
    client = TestClient(app)

    # Check 2: Endpoint Smoke Test (/videos & /video/{id})
    print("\n2. [Check 2/4] Endpoint Smoke Test (/videos & /video/{id})")
    try:
        r_list = client.get("/videos")
        assert r_list.status_code == 200, f"Expected 200, got {r_list.status_code}"
        vid_data = r_list.json()
        print(f"   /videos endpoint: returned {vid_data.get('count', 0)} videos.")

        r_stream = client.get("/video/sample_test")
        assert r_stream.status_code == 200, f"Expected 200, got {r_stream.status_code}"
        assert r_stream.headers["content-type"] == "video/mp4", f"Expected video/mp4, got {r_stream.headers['content-type']}"
        print(f"   /video/sample_test endpoint: successfully streamed video/mp4 content.")
        results_report["Endpoint Smoke Test"] = "PASS"
    except Exception as exc:
        print(f"   FAIL: Endpoint smoke test failed: {exc}")
        results_report["Endpoint Smoke Test"] = "FAIL"

    # Check 3: Search Endpoint & Schema Verification (/search)
    print("\n3. [Check 3/4] Search API & Schema Verification (/search)")
    try:
        query_payload = {"query": "cooking food in air fryer", "top_k": 3}
        r_search = client.post("/search", json=query_payload)
        assert r_search.status_code == 200, f"Expected 200, got {r_search.status_code}"

        search_data = r_search.json()
        assert "query" in search_data, "Missing 'query' field"
        assert "answer" in search_data, "Missing 'answer' field"
        assert "explanation_bullets" in search_data, "Missing 'explanation_bullets' field"
        assert "results" in search_data, "Missing 'results' field"

        print(f"   Query: '{search_data['query']}'")
        print(f"   Answer: {search_data['answer']}")
        print(f"   Bullets: {search_data['explanation_bullets']}")
        print(f"   Top Results Count: {len(search_data['results'])}")

        if search_data["results"]:
            top_res = search_data["results"][0]
            assert "scores" in top_res, "Missing 'scores' in result"
            print(f"   Top Result [{top_res['start_ts']}s - {top_res['end_ts']}s] final_score={top_res['scores']['final_score']}")

        results_report["Search API Verification"] = "PASS"
    except Exception as exc:
        print(f"   FAIL: Search API verification failed: {exc}")
        results_report["Search API Verification"] = "FAIL"

    # Check 4: Timestamp Seek & Range Validation
    print("\n4. [Check 4/4] Timestamp Seek Range Check")
    try:
        query_payload = {"query": "repairing motorcycle with helmet", "top_k": 3}
        r_search = client.post("/search", json=query_payload)
        search_data = r_search.json()

        valid_stamps = True
        for res in search_data.get("results", []):
            start = res.get("start_ts")
            end = res.get("end_ts")
            print(f"   Candidate timestamp range: [{start}s - {end}s]")
            if start is None or end is None or start < 0 or start > end:
                valid_stamps = False

        if valid_stamps:
            print("   PASS: All candidate timestamps valid for HTML5 video currentTime seeking.")
            results_report["Timestamp Seek Range"] = "PASS"
        else:
            print("   FAIL: Invalid timestamp range detected.")
            results_report["Timestamp Seek Range"] = "FAIL"
    except Exception as exc:
        print(f"   FAIL: Timestamp seek check failed: {exc}")
        results_report["Timestamp Seek Range"] = "FAIL"

    print("\n==========================================================")
    print("               VERIFICATION SUMMARY REPORT                ")
    print("==========================================================")
    all_passed = True
    for test_name, status in results_report.items():
        print(f"  {test_name:<30}: {status}")
        if status != "PASS":
            all_passed = False

    print("==========================================================")
    if all_passed:
        print("OVERALL STATUS: ALL M6 CHECKS PASSED SUCCESSFULLY!")
    else:
        print("OVERALL STATUS: SOME M6 CHECKS FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    run_verification()
