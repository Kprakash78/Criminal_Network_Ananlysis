"""
M1 Video / VLM — Schema Validator
Validates segment-level records against COMMON_DATA_CONTRACT.md's visual section.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

log = logging.getLogger(__name__)


def validate_segment_records(
    records: List[Dict[str, Any]]
) -> Tuple[bool, List[str]]:
    """
    Validate list of segment records against COMMON_DATA_CONTRACT.md schema.

    Returns (is_valid, list_of_error_strings).
    """
    errors: List[str] = []

    if not records:
        errors.append("Record list is empty")
        return False, errors

    for idx, rec in enumerate(records):
        prefix = f"Record [{idx}] (segment_id={rec.get('segment_id', 'MISSING')})"

        # Required top-level keys
        for key in ["video_id", "segment_id", "start_ts", "end_ts"]:
            if key not in rec:
                errors.append(f"{prefix}: missing top-level field '{key}'")

        if "start_ts" in rec and "end_ts" in rec:
            try:
                start = float(rec["start_ts"])
                end = float(rec["end_ts"])
                if start > end:
                    errors.append(f"{prefix}: start_ts ({start}) > end_ts ({end})")
            except (ValueError, TypeError):
                errors.append(f"{prefix}: start_ts or end_ts is not numeric")

        # Visual sub-schema check
        if "visual" not in rec or not isinstance(rec["visual"], dict):
            errors.append(f"{prefix}: missing or invalid 'visual' dictionary")
        else:
            vis = rec["visual"]
            if "description" not in vis or not isinstance(vis["description"], str):
                errors.append(f"{prefix}: visual.description missing or not string")
            elif not vis["description"].strip():
                errors.append(f"{prefix}: visual.description is empty string")

            if "embedding_id" not in vis or not isinstance(vis["embedding_id"], str):
                errors.append(f"{prefix}: visual.embedding_id missing or not string")

            if "quality_flag" in vis and vis["quality_flag"] not in ("ok", "low_quality"):
                errors.append(f"{prefix}: visual.quality_flag invalid: '{vis['quality_flag']}'")

    is_valid = len(errors) == 0
    if is_valid:
        log.info("Schema validation PASSED for %d records", len(records))
    else:
        log.error("Schema validation FAILED with %d errors:\n  %s", len(errors), "\n  ".join(errors))

    return is_valid, errors
