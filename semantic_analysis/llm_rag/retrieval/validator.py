"""
M4 Retrieval — Schema Validator
Validates unified segment records and search results against COMMON_DATA_CONTRACT.md.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

log = logging.getLogger(__name__)


def validate_search_results(
    results: List[Dict[str, Any]]
) -> Tuple[bool, List[str]]:
    """
    Validate list of search segment records against COMMON_DATA_CONTRACT.md schema.

    Returns (is_valid, list_of_error_strings).
    """
    errors: List[str] = []

    if not results:
        log.info("Search result list is empty (valid for no-match queries)")
        return True, errors

    for idx, seg in enumerate(results):
        prefix = f"Segment [{idx}] (segment_id={seg.get('segment_id', 'MISSING')})"

        # 1. Required top-level keys
        for key in ["video_id", "segment_id", "start_ts", "end_ts"]:
            if key not in seg:
                errors.append(f"{prefix}: missing required key '{key}'")

        if "start_ts" in seg and "end_ts" in seg:
            try:
                start = float(seg["start_ts"])
                end = float(seg["end_ts"])
                if start > end:
                    errors.append(f"{prefix}: start_ts ({start}) > end_ts ({end})")
            except (ValueError, TypeError):
                errors.append(f"{prefix}: start_ts or end_ts non-numeric")

        # 2. Scores dictionary validation
        if "scores" not in seg or not isinstance(seg["scores"], dict):
            errors.append(f"{prefix}: missing or invalid 'scores' dict")
        else:
            scores = seg["scores"]
            for score_key in ["visual_similarity", "transcript_similarity", "object_match", "ocr_match", "final_score"]:
                if score_key not in scores:
                    errors.append(f"{prefix}: missing score field '{score_key}'")
                elif scores[score_key] is not None and not isinstance(scores[score_key], (int, float)):
                    errors.append(f"{prefix}: score field '{score_key}' is not float/None")

            if "final_score" in scores and isinstance(scores["final_score"], (int, float)):
                fs = scores["final_score"]
                if not (0.0 <= fs <= 1.0):
                    errors.append(f"{prefix}: final_score out of range [0.0, 1.0]: {fs}")

        # 3. Optional visual section check
        if "visual" in seg and seg["visual"] is not None:
            vis = seg["visual"]
            if not isinstance(vis, dict):
                errors.append(f"{prefix}: 'visual' is not dict")

        # 4. Optional audio section check
        if "audio" in seg and seg["audio"] is not None:
            aud = seg["audio"]
            if not isinstance(aud, dict):
                errors.append(f"{prefix}: 'audio' is not dict")

    is_valid = len(errors) == 0
    if is_valid:
        log.info("Schema validation PASSED for %d search segment records", len(results))
    else:
        log.error("Schema validation FAILED with %d errors:\n  %s", len(errors), "\n  ".join(errors))

    return is_valid, errors
