"""
M3 Vision Intelligence — Metadata Writer
Appends segment-level records to data/metadata.jsonl in append-only fashion,
one JSON object per line, per COMMON_DATA_CONTRACT.md §Storage layout.

Design decisions:
- Append-only: safe for multi-video runs and re-runs that add new videos.
- Each record is newline-terminated valid JSON (JSONL).
- Idempotency flag: skip writing if a segment_id already exists in the file
  (prevents duplicates on re-run of the same video).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .config import M3Config, DEFAULT_CONFIG

log = logging.getLogger(__name__)


def write_segments(
    segments: list[dict[str, Any]],
    cfg: M3Config = DEFAULT_CONFIG,
    *,
    skip_duplicates: bool = True,
) -> int:
    """
    Append segment records to cfg.metadata_jsonl.

    Returns the number of records actually written.
    If *skip_duplicates* is True (default), any segment_id already present in
    the file is silently skipped — safe to re-run the same video.
    """
    out_path = cfg.metadata_jsonl
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing_ids: set[str] = set()
    if skip_duplicates and out_path.exists():
        existing_ids = _load_existing_ids(out_path)
        if existing_ids:
            log.info(
                "metadata.jsonl already has %d records; will skip duplicates",
                len(existing_ids),
            )

    written = 0
    with open(out_path, "a", encoding="utf-8") as fh:
        for seg in segments:
            seg_id = seg.get("segment_id", "")
            if seg_id and seg_id in existing_ids:
                log.debug("Skipping duplicate segment_id=%s", seg_id)
                continue
            fh.write(json.dumps(seg, ensure_ascii=False) + "\n")
            written += 1

    log.info(
        "Wrote %d new segment records to %s", written, out_path
    )
    return written


def _load_existing_ids(path: Path) -> set[str]:
    """Read all segment_ids already in the JSONL file."""
    ids: set[str] = set()
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                sid = obj.get("segment_id", "")
                if sid:
                    ids.add(sid)
            except json.JSONDecodeError as exc:
                log.warning("Bad JSON on line %d of %s: %s", lineno, path, exc)
    return ids
