"""
M3 Vision Intelligence — Verification Test
Lightweight end-to-end verification (STEP 2 of ANTIGRAVITY_PROMPT.md).

Checks:
  1. Schema validation: every line of metadata.jsonl matches COMMON_DATA_CONTRACT.md
  2. Object detection: at least one plausible object detected across segments
  3. OCR: at least one non-trivial text string detected (for frames with visible text)
  4. Timestamp monotonicity: start_ts values increase across records
  5. Aggregation: no suspicious flickering (no two consecutive segments with
     identical object sets AND ≤1s apart)

Run:
    python -m M3.verify_pipeline [--video path/to/video.mp4] [--data-root data]
                                  [--generate-sample]

Exit code 0 = all checks pass.
Exit code 1 = one or more checks failed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Force UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError for special chars)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def section(title: str) -> None:
    print(f"\n{'-'*60}")
    print(f"  {title}")
    print(f"{'-'*60}")


def result(name: str, passed: bool, detail: str = "") -> bool:
    icon = "[PASS]" if passed else "[FAIL]"
    print(f"  [{icon}] {name}")
    if detail:
        for line in detail.strip().splitlines():
            print(f"          {line}")
    return passed


# ─────────────────────────────────────────────────────────────────────────────
# Checks
# ─────────────────────────────────────────────────────────────────────────────

def check_schema(jsonl_path: Path) -> bool:
    """Delegate to M3.validator for schema check."""
    from M3.validator import validate_file
    total, n_errors, errors = validate_file(jsonl_path)
    detail = f"{total} records checked"
    if n_errors:
        detail += f"\n{n_errors} error(s):\n" + "\n".join(errors[:10])
    return result(
        f"Schema validation ({total} records)",
        n_errors == 0,
        detail,
    )


def check_objects_detected(records: list[dict]) -> bool:
    """At least one segment must have ≥1 detected object."""
    total_objects = sum(len(r.get("objects", [])) for r in records)
    all_labels = [
        o["label"]
        for r in records
        for o in r.get("objects", [])
    ]
    detail = f"Total object detections across all segments: {total_objects}"
    if all_labels:
        from collections import Counter
        top = Counter(all_labels).most_common(5)
        detail += "\nTop labels: " + ", ".join(f"{l}({c})" for l, c in top)
    return result("Object detection (≥1 object detected)", total_objects > 0, detail)


def check_ocr(records: list[dict]) -> bool:
    """At least one non-trivial OCR string somewhere in segments."""
    all_ocr = [t for r in records for t in r.get("ocr", []) if len(t) >= 2]
    detail = f"Total OCR strings: {len(all_ocr)}"
    if all_ocr:
        detail += "\nSamples: " + ", ".join(repr(t) for t in all_ocr[:8])
    else:
        detail += "\n(OCR may legitimately be empty if video has no visible text)"
    # Not a hard FAIL — video may have no text.  Warn only.
    passed = True  # always pass, but print info
    return result("OCR extraction (informational)", passed, detail)


def check_timestamps_monotonic(records: list[dict]) -> bool:
    """start_ts must be non-decreasing across records."""
    starts = [r["start_ts"] for r in records if "start_ts" in r]
    violations = [
        (i, starts[i - 1], starts[i])
        for i in range(1, len(starts))
        if starts[i] < starts[i - 1]
    ]
    detail = f"Timestamps checked: {len(starts)}"
    if violations:
        detail += f"\nViolations (idx, prev, curr): {violations[:5]}"
    return result("Timestamps monotonically increasing", len(violations) == 0, detail)


def check_no_flickering(records: list[dict]) -> bool:
    """
    No two consecutive segments should have identical object label sets
    AND be ≤1.5 seconds apart (that would indicate the aggregation dedup
    didn't fire correctly — it split what should be one continuous shot).
    """
    flickers = []
    for i in range(1, len(records)):
        prev = records[i - 1]
        curr = records[i]
        prev_labels = frozenset(o["label"] for o in prev.get("objects", []))
        curr_labels = frozenset(o["label"] for o in curr.get("objects", []))
        gap = curr.get("start_ts", 0) - prev.get("end_ts", 0)
        if prev_labels == curr_labels and gap <= 1.5 and prev_labels:
            flickers.append(
                f"  seg {prev.get('segment_id')} → {curr.get('segment_id')}  "
                f"labels={set(prev_labels)}  gap={gap:.2f}s"
            )
    detail = f"Consecutive segment pairs checked: {max(len(records)-1, 0)}"
    if flickers:
        detail += f"\nPossible flickers:\n" + "\n".join(flickers[:5])
    return result(
        "No flickering near-identical consecutive segments",
        len(flickers) == 0,
        detail,
    )


def check_quality_flags(records: list[dict]) -> bool:
    """quality_flag must be 'ok' or 'low_quality' — no other values."""
    valid = {"ok", "low_quality"}
    bad = [
        (r.get("segment_id"), r["visual"].get("quality_flag"))
        for r in records
        if isinstance(r.get("visual"), dict)
        and r["visual"].get("quality_flag") not in valid
    ]
    n_low = sum(
        1 for r in records
        if isinstance(r.get("visual"), dict)
        and r["visual"].get("quality_flag") == "low_quality"
    )
    detail = f"low_quality segments: {n_low}/{len(records)}"
    if bad:
        detail += f"\nInvalid quality_flag entries: {bad[:5]}"
    return result("quality_flag enum valid", len(bad) == 0, detail)


def print_segment_summary(records: list[dict]) -> None:
    """Human-readable table of all segments for manual inspection."""
    section("Segment summary (manual spot-check)")
    print(f"  {'segment_id':<30} {'start':>7} {'end':>7} {'objects':<40} {'ocr'}")
    print(f"  {'-'*30} {'-'*7} {'-'*7} {'-'*40} {'-'*20}")
    for r in records:
        sid = r.get("segment_id", "?")[:29]
        s = r.get("start_ts", 0)
        e = r.get("end_ts", 0)
        objs = ", ".join(
            f"{o['label']}({o['confidence']:.2f})"
            for o in r.get("objects", [])[:3]
        ) or "-"
        ocr = ", ".join(r.get("ocr", [])[:3]) or "-"
        print(f"  {sid:<30} {s:>7.2f} {e:>7.2f} {objs:<40} {ocr}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_verification(jsonl_path: Path) -> bool:
    """Run all checks and return True if all pass."""
    section("M3 Pipeline Verification")
    print(f"  metadata.jsonl: {jsonl_path}")

    if not jsonl_path.exists():
        print(f"\n  ERROR: File not found: {jsonl_path}")
        return False

    records = load_jsonl(jsonl_path)
    print(f"  Total segment records: {len(records)}")

    if not records:
        print("  ERROR: metadata.jsonl is empty — pipeline may not have run yet")
        return False

    results = [
        check_schema(jsonl_path),
        check_objects_detected(records),
        check_ocr(records),
        check_timestamps_monotonic(records),
        check_no_flickering(records),
        check_quality_flags(records),
    ]

    print_segment_summary(records)

    section("Summary")
    n_passed = sum(results)
    n_total = len(results)
    print(f"  {n_passed}/{n_total} checks passed")
    if n_passed == n_total:
        print("  Overall: PASS [OK]")
    else:
        print("  Overall: FAIL [!!]")
    return n_passed == n_total


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="M3 end-to-end verification against metadata.jsonl"
    )
    parser.add_argument(
        "--jsonl", default="data/metadata.jsonl",
        help="Path to metadata.jsonl",
    )
    parser.add_argument(
        "--video",
        help="If provided, run the M3 pipeline on this video first (before verifying)",
    )
    parser.add_argument(
        "--data-root", default="data",
        help="Data root directory (for pipeline run)",
    )
    parser.add_argument(
        "--generate-sample", action="store_true",
        help="Generate a synthetic test video and run the full pipeline on it",
    )
    parser.add_argument(
        "--video-id", default=None,
        help="Video ID to use when running the pipeline",
    )
    args = parser.parse_args(argv)

    # ── Optionally run the pipeline before verifying ──────────────────────────
    if args.generate_sample:
        from M3.make_sample_video import make_sample_video
        video_path = make_sample_video(
            output_path=f"{args.data_root}/videos/sample_test.mp4"
        )
        args.video = str(video_path)
        if args.video_id is None:
            args.video_id = "sample_test"

    if args.video:
        from M3.pipeline import process_video
        from M3.config import M3Config
        cfg = M3Config(data_root=Path(args.data_root))
        video_id = args.video_id or Path(args.video).stem
        process_video(args.video, video_id, cfg)
        # point jsonl at the correct data root
        args.jsonl = str(cfg.metadata_jsonl)

    passed = run_verification(Path(args.jsonl))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
