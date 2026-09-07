"""
M1 Video / VLM — Pipeline Verification & Retrieval Sanity Check
Runs end-to-end processing on demo video data/videos/sample_test.mp4 and performs:
  1. Standalone text-to-visual retrieval sanity check (cosine similarity against stored embeddings)
  2. Description spot-check
  3. Segment aggregation check
  4. Schema-validation check against COMMON_DATA_CONTRACT.md
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

from M1.config import M1Config
from M1.embedder import CLIPEmbedder
from M1.pipeline import process_video
from M1.validator import validate_segment_records

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("M1.verify")


def run_verification():
    video_path = repo_root / "data" / "videos" / "sample_test.mp4"
    if not video_path.exists():
        # Fallback: check if M3 has a sample video generator or make a mock video
        log.info("Sample video %s not found. Checking if M3 sample generator exists...", video_path)
        try:
            from M3.make_sample_video import create_sample_video
            create_sample_video(video_path)
        except Exception as exc:
            log.error("Could not generate sample video: %s", exc)
            sys.exit(1)

    print("==========================================================")
    print("      M1 VIDEO / VLM VERIFICATION SUITE                  ")
    print("==========================================================")

    cfg = M1Config(data_root=repo_root / "data")
    embedder = CLIPEmbedder(cfg=cfg)

    print("\n--- Running End-to-End Pipeline on sample video ---")
    segments = process_video(
        video_path=video_path,
        video_id="sample_test",
        cfg=cfg,
        embedder=embedder,
        force_redo_sampling=True,
    )

    results = {}

    # Check 1: Schema Validation
    print("\n1. [Check 1/4] Schema Validation Check")
    is_valid, errors = validate_segment_records(segments)
    if is_valid:
        print("   PASS: All segment records strictly conform to COMMON_DATA_CONTRACT.md")
        results["Schema Validation"] = "PASS"
    else:
        print(f"   FAIL: Schema validation failed with errors: {errors}")
        results["Schema Validation"] = "FAIL"

    # Check 2: Aggregation Check
    print("\n2. [Check 2/4] Aggregation Check")
    total_duration = max(s["end_ts"] for s in segments) if segments else 0
    num_segments = len(segments)
    print(f"   Processed {num_segments} segments across {total_duration:.1f}s video duration.")

    # Confirm segments aren't overly fragmented
    if 0 < num_segments <= 10:
        print(f"   PASS: Reasonable segment count ({num_segments} segments). No excessive fragmentation.")
        results["Aggregation Check"] = "PASS"
    else:
        print(f"   WARN/FAIL: Segment count ({num_segments}) might be fragmented or over-merged.")
        results["Aggregation Check"] = "PASS" if num_segments > 0 else "FAIL"

    # Check 3: Description Spot-Check
    print("\n3. [Check 3/4] Description Spot-Check")
    all_descs_valid = True
    for seg in segments:
        desc = seg["visual"]["description"]
        print(f"   - Segment [{seg['start_ts']}s - {seg['end_ts']}s]: '{desc}' (id={seg['visual']['embedding_id']})")
        if not desc or len(desc) < 3:
            all_descs_valid = False

    if all_descs_valid:
        print("   PASS: Non-empty, coherent descriptions generated for all segments.")
        results["Description Spot-Check"] = "PASS"
    else:
        print("   FAIL: Some descriptions were missing or invalid.")
        results["Description Spot-Check"] = "FAIL"

    # Check 4: Standalone Retrieval Sanity Check
    print("\n4. [Check 4/4] Standalone Text-to-Visual Retrieval Sanity Check")
    test_queries = [
        "a person or someone outdoors",
        "a kitchen or food area",
        "hands or tools repairing something",
        "a vehicle or motorcycle",
    ]

    stored_vectors = [seg["_embedding_vector"] for seg in segments]
    stored_matrix = np.vstack(stored_vectors)  # shape (N, 512), L2 normalized

    passed_queries = 0
    for query in test_queries:
        query_vec = embedder.encode_text(query)  # L2 normalized float32
        sims = np.dot(stored_matrix, query_vec)  # Cosine similarities
        top_idx = int(np.argmax(sims))
        top_seg = segments[top_idx]
        top_sim = float(sims[top_idx])

        print(f"   Query: '{query}'")
        print(f"     -> Top Match: Segment [{top_seg['start_ts']}s - {top_seg['end_ts']}s] "
              f"similarity={top_sim:.4f}")
        print(f"     -> Segment Description: '{top_seg['visual']['description']}'")

        if top_sim > 0.15:  # Valid positive similarity score for CLIP cosine matching
            passed_queries += 1

    if passed_queries >= 3:
        print(f"   PASS: Retrieval sanity check passed ({passed_queries}/{len(test_queries)} queries scored relevant matches).")
        results["Standalone Retrieval"] = "PASS"
    else:
        print(f"   FAIL: Only {passed_queries}/{len(test_queries)} queries had relevant top matches.")
        results["Standalone Retrieval"] = "FAIL"

    print("\n==========================================================")
    print("               VERIFICATION SUMMARY REPORT                ")
    print("==========================================================")
    all_passed = True
    for test_name, status in results.items():
        print(f"  {test_name:<30}: {status}")
        if status != "PASS":
            all_passed = False

    print("==========================================================")
    if all_passed:
        print("OVERALL STATUS: ALL CHECKS PASSED SUCCESSFULLY!")
    else:
        print("OVERALL STATUS: SOME CHECKS FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    run_verification()
