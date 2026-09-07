"""
CLI entry point for M1 Video / VLM module.
Usage: python -m M1 <path_to_video.mp4>
"""
import argparse
import logging
import sys
from pathlib import Path

from .config import M1Config
from .pipeline import process_video


def main():
    parser = argparse.ArgumentParser(description="M1 Video / VLM Processing Pipeline")
    parser.add_argument("video_path", type=str, help="Path to input video file")
    parser.add_argument("--video-id", type=str, default=None, help="Video ID override")
    parser.add_argument("--interval", type=float, default=1.0, help="Frame sampling interval in seconds")
    parser.add_argument("--sim-threshold", type=float, default=0.85, help="Segment similarity threshold")
    parser.add_argument("--force-redo", action="store_true", help="Force re-sampling of frames")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose debug logging")

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    cfg = M1Config(
        frame_interval_sec=args.interval,
        segment_sim_threshold=args.sim_threshold,
    )

    try:
        segments = process_video(
            video_path=args.video_path,
            video_id=args.video_id,
            cfg=cfg,
            force_redo_sampling=args.force_redo,
        )
        print(f"\nSUCCESS: Processed {len(segments)} segment records.")
        for seg in segments:
            print(f"  - [{seg['start_ts']}s - {seg['end_ts']}s] {seg['segment_id']}: {seg['visual']['description']}")
    except Exception as exc:
        print(f"\nERROR: Pipeline execution failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
