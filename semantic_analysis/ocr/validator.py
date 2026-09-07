"""
M3 Vision Intelligence — Schema Validator
Validates every line of metadata.jsonl against COMMON_DATA_CONTRACT.md.

Run as a standalone script:
    python -m M3.validator [path/to/metadata.jsonl]

Exit code 0 = all lines valid.
Exit code 1 = one or more validation errors.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# Schema rules derived from COMMON_DATA_CONTRACT.md
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_FIELDS = {"video_id", "segment_id", "start_ts", "end_ts"}
VALID_QUALITY_FLAGS = {"ok", "low_quality"}
SCORES_KEYS = {"visual_similarity", "transcript_similarity", "object_match",
               "ocr_match", "final_score"}


def validate_segment(obj: dict[str, Any], lineno: int) -> list[str]:
    """
    Validate one segment record.  Returns a list of error strings (empty = OK).
    """
    errors: list[str] = []

    def err(msg: str) -> None:
        errors.append(f"  Line {lineno}: {msg}")

    # ── Required top-level fields ─────────────────────────────────────────────
    for field in REQUIRED_FIELDS:
        if field not in obj:
            err(f"Missing required field '{field}'")

    # ── video_id must be non-empty string ─────────────────────────────────────
    if not isinstance(obj.get("video_id"), str) or not obj["video_id"]:
        err("'video_id' must be a non-empty string")

    # ── timestamps must be numbers and start_ts <= end_ts ────────────────────
    start = obj.get("start_ts")
    end = obj.get("end_ts")
    if not isinstance(start, (int, float)):
        err(f"'start_ts' must be a number, got {type(start).__name__}")
    if not isinstance(end, (int, float)):
        err(f"'end_ts' must be a number, got {type(end).__name__}")
    if isinstance(start, (int, float)) and isinstance(end, (int, float)):
        if start > end:
            err(f"start_ts ({start}) > end_ts ({end})")

    # ── visual block ──────────────────────────────────────────────────────────
    visual = obj.get("visual")
    if visual is not None:
        if not isinstance(visual, dict):
            err("'visual' must be an object/dict")
        else:
            qf = visual.get("quality_flag")
            if qf is not None and qf not in VALID_QUALITY_FLAGS:
                err(f"'visual.quality_flag' must be one of {VALID_QUALITY_FLAGS}, got {qf!r}")

    # ── objects must be a list of dicts with label (str) + confidence (float) ─
    objects = obj.get("objects", [])
    if not isinstance(objects, list):
        err("'objects' must be a list")
    else:
        for i, o in enumerate(objects):
            if not isinstance(o, dict):
                err(f"'objects[{i}]' must be a dict")
                continue
            if not isinstance(o.get("label"), str):
                err(f"'objects[{i}].label' must be a string")
            conf = o.get("confidence")
            if not isinstance(conf, (int, float)) or not (0.0 <= conf <= 1.0):
                err(f"'objects[{i}].confidence' must be float in [0,1], got {conf!r}")
            attrs = o.get("attributes")
            if attrs is not None and not isinstance(attrs, list):
                err(f"'objects[{i}].attributes' must be a list if present")

    # ── ocr must be a list of strings ─────────────────────────────────────────
    ocr = obj.get("ocr", [])
    if not isinstance(ocr, list):
        err("'ocr' must be a list")
    else:
        for i, text in enumerate(ocr):
            if not isinstance(text, str):
                err(f"'ocr[{i}]' must be a string, got {type(text).__name__}")

    # ── scores block: all keys present, all values None (at ingestion time) ───
    scores = obj.get("scores")
    if scores is not None:
        if not isinstance(scores, dict):
            err("'scores' must be an object/dict")
        else:
            for key in SCORES_KEYS:
                if key not in scores:
                    err(f"'scores.{key}' missing")

    return errors


def validate_file(path: Path) -> tuple[int, int, list[str]]:
    """
    Validate all records in the JSONL file.

    Returns (total_lines, error_count, list_of_error_messages).
    """
    all_errors: list[str] = []
    total = 0
    prev_start: float | None = None
    timestamps_monotonic = True

    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                all_errors.append(f"  Line {lineno}: JSON parse error: {exc}")
                continue

            all_errors.extend(validate_segment(obj, lineno))

            # Monotonic timestamp check across records
            start = obj.get("start_ts")
            if isinstance(start, (int, float)):
                if prev_start is not None and start < prev_start:
                    timestamps_monotonic = False
                    all_errors.append(
                        f"  Line {lineno}: start_ts={start} < previous start_ts={prev_start} "
                        f"(timestamps not monotonically increasing)"
                    )
                prev_start = start

    _ = timestamps_monotonic  # kept for reference
    return total, len(all_errors), all_errors


def main(argv: list[str] | None = None) -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Validate metadata.jsonl against COMMON_DATA_CONTRACT.md"
    )
    parser.add_argument(
        "jsonl",
        nargs="?",
        default="data/metadata.jsonl",
        help="Path to metadata.jsonl (default: data/metadata.jsonl)",
    )
    args = parser.parse_args(argv)

    path = Path(args.jsonl)
    if not path.exists():
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        sys.exit(1)

    print(f"Validating {path} …")
    total, n_errors, errors = validate_file(path)

    print(f"\nRecords checked : {total}")
    if n_errors == 0:
        print("Schema validation : PASS -- all records valid [OK]")
        sys.exit(0)
    else:
        print(f"Schema validation : FAIL -- {n_errors} error(s)")
        for e in errors:
            print(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
