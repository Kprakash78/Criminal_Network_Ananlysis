"""
M3 Vision Intelligence — Sample Video Generator
Creates a short synthetic MP4 test video using OpenCV + NumPy.

This replaces the need for a real video file during CI/unit testing, while
still exercising the full pipeline.  The video contains:
  - A plain colored background that changes every 5 seconds (tests aggregation
    of consecutive similar frames AND detection of scene change)
  - Text rendered on each scene (tests OCR)
  - A brief dark/blurred scene (tests quality_flag = low_quality)

Usage:
    python -m M3.make_sample_video [output_path] [duration_sec]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def make_sample_video(
    output_path: str | Path = "data/videos/sample_test.mp4",
    duration_sec: int = 20,
    fps: int = 10,
) -> Path:
    """
    Generate a synthetic test video at *output_path* and return its path.

    Scenes:
      0–5s  : Blue background, text "YAMAHA", simple shapes (simulates branded object)
      5–10s : Green background, text "HONDA", different shapes
      10–13s: Nearly black (very dark → quality_flag=low_quality)
      13–20s: Red background, text "SUZUKI", shapes
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    W, H = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (W, H))

    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path}")

    total_frames = duration_sec * fps
    for frame_idx in range(total_frames):
        t = frame_idx / fps  # seconds

        if t < 5:
            # Scene 1 — blue, YAMAHA text
            img = np.full((H, W, 3), (180, 50, 30), dtype=np.uint8)  # BGR blue-ish
            cv2.putText(img, "YAMAHA", (180, 240), cv2.FONT_HERSHEY_SIMPLEX,
                        3.0, (255, 255, 255), 4)
            # Draw a rectangle to give YOLO something to see
            cv2.rectangle(img, (50, 100), (200, 350), (255, 255, 0), -1)
            cv2.circle(img, (450, 200), 80, (0, 200, 255), -1)
        elif t < 10:
            # Scene 2 — green, HONDA text
            img = np.full((H, W, 3), (40, 160, 40), dtype=np.uint8)
            cv2.putText(img, "HONDA", (190, 240), cv2.FONT_HERSHEY_SIMPLEX,
                        3.0, (255, 255, 255), 4)
            cv2.rectangle(img, (100, 80), (300, 280), (0, 0, 200), -1)
            cv2.circle(img, (480, 350), 60, (200, 200, 0), -1)
        elif t < 13:
            # Scene 3 — very dark (tests quality_flag=low_quality)
            img = np.full((H, W, 3), (8, 8, 8), dtype=np.uint8)
            # Add a tiny amount of noise so it isn't pure solid black
            noise = np.random.randint(0, 15, (H, W, 3), dtype=np.uint8)
            img = cv2.add(img, noise)
        else:
            # Scene 4 — red/orange, SUZUKI text
            img = np.full((H, W, 3), (30, 60, 200), dtype=np.uint8)  # BGR orange-red
            cv2.putText(img, "SUZUKI", (150, 240), cv2.FONT_HERSHEY_SIMPLEX,
                        3.0, (255, 255, 255), 4)
            cv2.rectangle(img, (60, 60), (250, 300), (0, 255, 0), -1)
            cv2.ellipse(img, (450, 300), (100, 60), 0, 0, 360, (255, 0, 100), -1)

        writer.write(img)

    writer.release()
    print(f"Sample video written: {output_path}  ({duration_sec}s @ {fps}fps)")
    return output_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic test video for M3 pipeline verification"
    )
    parser.add_argument(
        "output", nargs="?",
        default="data/videos/sample_test.mp4",
        help="Output path for the test video",
    )
    parser.add_argument(
        "--duration", type=int, default=20,
        help="Duration in seconds",
    )
    args = parser.parse_args(argv)
    make_sample_video(args.output, args.duration)


if __name__ == "__main__":
    main()
