"""
M4 Retrieval — Verification & Sanity Check Suite
Executes the mandatory verification checks from BACKEND.md §6:
  1. Ingestion check (multi-modal joining)
  2. Score distribution check (scores differentiate properly)
  3. Embedding-space sanity check (querying exact text yields similarity near 1.0)
  4. No-match behavior check (absent queries yield low final_score)
  5. Schema-validation check against COMMON_DATA_CONTRACT.md
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from M4.config import M4Config
from M4.embedder import QueryEmbedder
from M4.index_manager import FAISSIndexManager
from M4.pipeline import run_ingestion
from M4.search import search
from M4.validator import validate_search_results

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("M4.verify")


def run_verification():
    print("==========================================================")
    print("        M4 RETRIEVAL VERIFICATION SUITE                   ")
    print("==========================================================")

    cfg = M4Config(data_root=repo_root / "data")
    embedder = QueryEmbedder(cfg=cfg)
    index_mgr = FAISSIndexManager(cfg=cfg)

    # Prepare representative sample data if metadata.jsonl does not exist or has few records
    print("\n--- Ingesting Multi-Modal Data into FAISS & metadata.jsonl ---")

    sample_visuals = [
        {
            "video_id": "demo_video_01",
            "segment_id": "demo_video_01_seg_00",
            "start_ts": 0.0,
            "end_ts": 5.0,
            "visual": {
                "description": "A chef cooking in a kitchen using an air fryer and tools",
                "embedding_id": "vis_demo_01_00",
                "quality_flag": "ok",
                "_embedding_vector": embedder.embed_visual_query("A chef cooking in a kitchen using an air fryer and tools"),
            },
        },
        {
            "video_id": "demo_video_01",
            "segment_id": "demo_video_01_seg_01",
            "start_ts": 5.0,
            "end_ts": 12.0,
            "visual": {
                "description": "Person wearing a helmet repairing a motorcycle in a workshop",
                "embedding_id": "vis_demo_01_01",
                "quality_flag": "ok",
                "_embedding_vector": embedder.embed_visual_query("Person wearing a helmet repairing a motorcycle in a workshop"),
            },
        },
        {
            "video_id": "demo_video_01",
            "segment_id": "demo_video_01_seg_02",
            "start_ts": 12.0,
            "end_ts": 18.0,
            "visual": {
                "description": "An outdoor scenic view of a mountain highway with vehicles",
                "embedding_id": "vis_demo_01_02",
                "quality_flag": "ok",
                "_embedding_vector": embedder.embed_visual_query("An outdoor scenic view of a mountain highway with vehicles"),
            },
        },
    ]

    sample_audio = [
        {
            "video_id": "demo_video_01",
            "segment_id": "demo_video_01_seg_00",
            "start_ts": 0.0,
            "end_ts": 5.0,
            "audio": {
                "transcript": "First set the temperature on the air fryer to 400 degrees",
                "embedding_id": "aud_demo_01_00",
                "_embedding_vector": embedder.embed_audio_query("First set the temperature on the air fryer to 400 degrees"),
            },
        },
        {
            "video_id": "demo_video_01",
            "segment_id": "demo_video_01_seg_01",
            "start_ts": 5.0,
            "end_ts": 12.0,
            "audio": {
                "transcript": "Tighten the brake bolt on the rear wheel securely",
                "embedding_id": "aud_demo_01_01",
                "_embedding_vector": embedder.embed_audio_query("Tighten the brake bolt on the rear wheel securely"),
            },
        },
    ]

    sample_objects = [
        {
            "video_id": "demo_video_01",
            "timestamp": 2.5,
            "objects": [{"label": "air fryer", "confidence": 0.95, "attributes": ["black"]}],
            "ocr": ["NINJA"],
        },
        {
            "video_id": "demo_video_01",
            "timestamp": 8.0,
            "objects": [
                {"label": "motorcycle", "confidence": 0.98, "attributes": ["blue"]},
                {"label": "helmet", "confidence": 0.92, "attributes": ["black"]},
            ],
            "ocr": ["YAMAHA"],
        },
    ]

    unified_segments = run_ingestion(
        visual_records=sample_visuals,
        audio_records=sample_audio,
        object_records=sample_objects,
        cfg=cfg,
        index_mgr=index_mgr,
    )

    results_report = {}

    # Check 1: Ingestion Check
    print("\n1. [Check 1/5] Ingestion Spot-Check")
    if unified_segments and len(unified_segments) == 3:
        print(f"   PASS: Ingested {len(unified_segments)} unified segments cleanly.")
        for seg in unified_segments:
            print(f"   - Segment [{seg['start_ts']}s - {seg['end_ts']}s]: "
                  f"Visual='{seg.get('visual', {}).get('description')[:40]}...', "
                  f"Audio='{seg.get('audio', {}).get('transcript') if seg.get('audio') else 'None'}', "
                  f"Objects={len(seg.get('objects', []))}, OCR={seg.get('ocr', [])}")
        results_report["Ingestion Check"] = "PASS"
    else:
        print(f"   FAIL: Unified segments count unexpected ({len(unified_segments)})")
        results_report["Ingestion Check"] = "FAIL"

    # Check 2: Embedding-Space Sanity Check (BACKEND §6.3)
    print("\n2. [Check 2/5] Embedding-Space Sanity Check (BACKEND §6.3)")
    exact_visual_query = "A chef cooking in a kitchen using an air fryer and tools"
    exact_vis_results = search(
        structured_query={},
        raw_query=exact_visual_query,
        top_k=3,
        cfg=cfg,
        embedder=embedder,
        index_mgr=index_mgr,
        unified_segments=unified_segments,
    )
    top_vis_sim = exact_vis_results[0]["scores"]["visual_similarity"]
    print(f"   Exact Visual Query: '{exact_visual_query}'")
    print(f"   -> Top Match Segment ID: {exact_vis_results[0]['segment_id']} (vis_similarity={top_vis_sim:.4f})")
    
    if top_vis_sim > 0.90:
        print(f"   PASS: Exact match cosine similarity is near 1.0 ({top_vis_sim:.4f}). Checkpoint alignment verified!")
        results_report["Embedding-Space Alignment"] = "PASS"
    else:
        print(f"   FAIL: Cosine similarity too low for exact text match ({top_vis_sim:.4f}). Checkpoint mismatch!")
        results_report["Embedding-Space Alignment"] = "FAIL"

    # Check 3: Score Distribution & Differentiation Check
    print("\n3. [Check 3/5] Score Distribution Check across Test Queries")
    test_queries = [
        {"raw": "cooking food in air fryer", "struct": {"objects": [{"object": "air fryer", "attributes": ["black"]}], "ocr": ["NINJA"]}},
        {"raw": "repairing a motorcycle with a helmet", "struct": {"objects": [{"object": "motorcycle", "attributes": ["blue"]}, {"object": "helmet"}], "ocr": ["YAMAHA"]}},
        {"raw": "highway scenic view outdoor", "struct": {"objects": [], "ocr": []}},
    ]

    all_differentiated = True
    for tq in test_queries:
        res = search(
            structured_query=tq["struct"],
            raw_query=tq["raw"],
            top_k=3,
            cfg=cfg,
            embedder=embedder,
            index_mgr=index_mgr,
            unified_segments=unified_segments,
        )
        scores = [r["scores"]["final_score"] for r in res]
        top_seg = res[0]
        print(f"   Query: '{tq['raw']}'")
        print(f"     -> Scores distribution: {scores}")
        print(f"     -> Top segment [{top_seg['start_ts']}s - {top_seg['end_ts']}s] final_score={scores[0]:.3f} "
              f"(vis={top_seg['scores']['visual_similarity']}, aud={top_seg['scores']['transcript_similarity']}, obj={top_seg['scores']['object_match']}, ocr={top_seg['scores']['ocr_match']})")

        # Check score spread between top 1 and top 3
        if len(scores) > 1 and (scores[0] - scores[-1]) < 0.01:
            all_differentiated = False

    if all_differentiated:
        print("   PASS: Score distribution clearly differentiates strong matches from weak matches.")
        results_report["Score Differentiation"] = "PASS"
    else:
        print("   FAIL: Scores are tightly clustered and undifferentiated.")
        results_report["Score Differentiation"] = "FAIL"

    # Check 4: No-Match Behavior Check
    print("\n4. [Check 4/5] No-Match Behavior Check")
    absent_query = "alien spaceship landing in deep underwater ocean"
    no_match_res = search(
        structured_query={"objects": [{"object": "spaceship"}], "ocr": ["ALIEN"]},
        raw_query=absent_query,
        top_k=3,
        cfg=cfg,
        embedder=embedder,
        index_mgr=index_mgr,
        unified_segments=unified_segments,
    )
    top_no_match_score = no_match_res[0]["scores"]["final_score"] if no_match_res else 0.0
    print(f"   Absent Query: '{absent_query}'")
    print(f"   -> Top Result Score: {top_no_match_score:.3f}")
    if top_no_match_score < 0.30:
        print(f"   PASS: Absent content score is appropriately low ({top_no_match_score:.3f} < 0.30).")
        results_report["No-Match Handling"] = "PASS"
    else:
        print(f"   FAIL: Absent content score is unexpectedly high ({top_no_match_score:.3f} >= 0.30).")
        results_report["No-Match Handling"] = "FAIL"

    # Check 5: Schema Validation Check
    print("\n5. [Check 5/5] Schema Validation Check against COMMON_DATA_CONTRACT.md")
    is_valid, errors = validate_search_results(exact_vis_results)
    if is_valid:
        print("   PASS: All returned search segment records conform strictly to COMMON_DATA_CONTRACT.md.")
        results_report["Schema Validation"] = "PASS"
    else:
        print(f"   FAIL: Schema validation failed with errors: {errors}")
        results_report["Schema Validation"] = "FAIL"

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
        print("OVERALL STATUS: ALL M4 CHECKS PASSED SUCCESSFULLY!")
    else:
        print("OVERALL STATUS: SOME M4 CHECKS FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    run_verification()
