"""
M5 LLM/RAG — Shared retry helper for Gemini API calls.

Handles 503 UNAVAILABLE and 429 RESOURCE_EXHAUSTED with exponential backoff.
Used by query_parser.py and generator.py.
"""
from __future__ import annotations

import logging
import time
from typing import Callable, TypeVar

log = logging.getLogger(__name__)

T = TypeVar("T")

# HTTP status codes that are safe to retry
_RETRYABLE_CODES = {429, 503}

# Default retry schedule: delays in seconds before each retry attempt
_DEFAULT_DELAYS = [2, 5, 15, 30]


def call_with_retry(
    fn: Callable[[], T],
    delays: list[int] | None = None,
    label: str = "Gemini API call",
) -> T:
    """
    Call fn() and retry on transient Gemini API errors (429/503).

    Parameters
    ----------
    fn     : Zero-argument callable that makes the API call.
    delays : List of sleep durations (seconds) before each retry.
             Default: [2, 5, 15, 30] — up to 4 retries.
    label  : Human-readable label for log messages.

    Returns
    -------
    The return value of fn() on success.

    Raises
    ------
    The last exception if all retries are exhausted.
    """
    if delays is None:
        delays = _DEFAULT_DELAYS

    last_exc: Exception | None = None
    for attempt, delay in enumerate([0] + delays):
        if delay > 0:
            log.info(
                "%s: retrying in %ds (attempt %d/%d)...",
                label, delay, attempt, len(delays),
            )
            time.sleep(delay)
        try:
            return fn()
        except Exception as exc:
            code = _extract_http_code(exc)
            if code in _RETRYABLE_CODES:
                log.warning("%s: got %s %s — will retry", label, code, type(exc).__name__)
                last_exc = exc
                continue
            raise   # non-retryable — propagate immediately

    # All retries exhausted
    raise last_exc  # type: ignore[misc]


def _extract_http_code(exc: Exception) -> int | None:
    """Extract HTTP status code from a google.genai API exception, if any."""
    # google.genai raises exceptions with a .code attribute (int) or embedded in the message
    if hasattr(exc, "code"):
        try:
            return int(exc.code)
        except (TypeError, ValueError):
            pass
    # Fallback: check string representation
    msg = str(exc)
    for code in _RETRYABLE_CODES:
        if str(code) in msg:
            return code
    return None
