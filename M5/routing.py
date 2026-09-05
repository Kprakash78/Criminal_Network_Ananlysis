"""
M5 — Confidence-Based Routing
PS 26152 — AI-Powered Criminal Network Analysis System

M5.3 — Confidence-based routing: simple threshold check on the composite
confidence score produced by M4's generate_summary(). Routing is intentionally
kept simple and explainable — no learned router, just a threshold comparison.

The default threshold (0.5) is listed in RUN_REPORT.md so the team can review
and tune it together before the demo.
"""

import logging

from M5.models import RoutingDecision

logger = logging.getLogger(__name__)

# Default confidence threshold (backend.md §5).
# If confidence < HUMAN_REVIEW_THRESHOLD, the result is flagged for human review.
# Review and tune this value with the team before the demo.
HUMAN_REVIEW_THRESHOLD = 0.5


def route_on_confidence(
    confidence: float,
    threshold: float = HUMAN_REVIEW_THRESHOLD,
) -> RoutingDecision:
    """
    Decide whether the current result needs human review.

    Returns a RoutingDecision with route = "continue" or "human_review".
    """
    if confidence < threshold:
        reason = (
            f"Confidence {confidence:.3f} is below the human-review threshold "
            f"({threshold:.2f}). Result requires investigator verification."
        )
        decision = RoutingDecision(
            route="human_review",
            reason=reason,
            confidence=confidence,
            threshold_used=threshold,
        )
        logger.info(f"[Routing] → human_review (confidence={confidence:.3f})")
    else:
        reason = (
            f"Confidence {confidence:.3f} meets or exceeds threshold ({threshold:.2f}). "
            f"Proceeding with automated result."
        )
        decision = RoutingDecision(
            route="continue",
            reason=reason,
            confidence=confidence,
            threshold_used=threshold,
        )
        logger.info(f"[Routing] → continue (confidence={confidence:.3f})")

    return decision
