"""
CLI entry point for M4 Retrieval module.
Usage: python -m M4 "person repairing a motorcycle"
"""
import argparse
import json
import logging
import sys

from .config import M4Config
from .search import search


def main():
    parser = argparse.ArgumentParser(description="M4 Multi-Modal Retrieval CLI")
    parser.add_argument("query", type=str, help="Natural language text query")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top candidates to return")
    parser.add_argument("--objects", type=str, default=None, help="JSON string for structured query objects")
    parser.add_argument("--ocr", type=str, default=None, help="Comma-separated OCR terms")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    structured_query = {"objects": [], "context": [], "ocr": []}
    if args.objects:
        try:
            structured_query["objects"] = json.loads(args.objects)
        except Exception as exc:
            print(f"Error parsing --objects JSON: {exc}", file=sys.stderr)
            sys.exit(1)

    if args.ocr:
        structured_query["ocr"] = [term.strip() for term in args.ocr.split(",")]

    cfg = M4Config()
    try:
        results = search(structured_query=structured_query, raw_query=args.query, top_k=args.top_k, cfg=cfg)
        print(f"\n==========================================================")
        print(f" M4 RETRIEVAL RESULTS for: '{args.query}'")
        print(f"==========================================================")
        if not results:
            print("No matching segments found.")
            return

        for idx, seg in enumerate(results, 1):
            scores = seg["scores"]
            vis_desc = seg.get("visual", {}).get("description", "N/A") if seg.get("visual") else "N/A"
            print(f"\n[{idx}] Segment ID: {seg['segment_id']} (video: {seg['video_id']}, ts: {seg['start_ts']}s - {seg['end_ts']}s)")
            print(f"    Final Score : {scores['final_score']:.3f} (Vis: {scores['visual_similarity']:.3f}, "
                  f"Aud: {scores['transcript_similarity']:.3f}, Obj: {scores['object_match']:.3f}, OCR: {scores['ocr_match']:.3f})")
            print(f"    Visual Desc : {vis_desc}")
            if seg.get("ocr"):
                print(f"    OCR Text    : {seg['ocr']}")
            if seg.get("objects"):
                print(f"    Objects     : {seg['objects']}")
    except Exception as exc:
        print(f"\nERROR: Search failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
