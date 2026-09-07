"""
llm_rag/query_parser.py — Local Offline Query Parser
======================================================
Replaces the Gemini-based query parser with a deterministic
keyword/regex/rule-based parser.

Works FULLY OFFLINE. No API key required. No internet connection needed.

Handles queries like:
  "When did the car arrive?"
  "When did the car leave?"
  "Find vehicle MH12AB4587."
  "Show me the red car."
  "What did the people say?"
  "Show people near the vehicle."
  "Person exits vehicle"

Returns a StructuredQuery following the COMMON_DATA_CONTRACT schema.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from .schemas import AttributedObject, StructuredQuery

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Vocabulary tables
# ─────────────────────────────────────────────────────────────────────────────

# Vehicle synonyms
_VEHICLES = {"car", "vehicle", "truck", "van", "bus", "auto", "automobile", "suv", "jeep", "cab", "taxi"}

# Person synonyms
_PERSONS = {
    "person", "people", "man", "woman", "individual", "individuals",
    "pedestrian", "pedestrians", "officer", "occupant", "occupants",
    "driver", "passenger", "passengers", "thief", "thieves", "suspect",
    "suspects", "criminal", "criminals",
}

# Action / event keywords mapped to normalized action tokens
_ACTION_MAP = {
    # Departure synonyms
    r"\bleav(e|es|ing|ed)\b": "depart",
    r"\bdepartur(e|es)\b": "depart",
    r"\bdeparted?\b": "depart",
    r"\bdrive away\b": "depart",
    r"\bmov(e|es|ing|ed) away\b": "depart",
    r"\bgoes? away\b": "depart",
    # Arrival synonyms
    r"\barriv(e|es|ing|ed)\b": "arrive",
    r"\barrival\b": "arrive",
    r"\bpark(s|ed|ing)?\b": "arrive",
    r"\bstops?\b": "stop",
    r"\bstopped\b": "stop",
    # Exit synonyms
    r"\bexit(s|ing|ed)?\b": "exit",
    r"\bgets? out\b": "exit",
    r"\bgot out\b": "exit",
    r"\bget(s)? out of\b": "exit",
    r"\bstep(s|ped|ping)? out\b": "exit",
    r"\bclimb(s|ed|ing)? out\b": "exit",
    # Enter synonyms
    r"\benter(s|ing|ed)?\b": "enter",
    r"\bget(s)? in(to)?\b": "enter",
    r"\bgot in(to)?\b": "enter",
    r"\bboard(s|ed|ing)?\b": "enter",
    r"\bclimb(s|ed|ing)? in\b": "enter",
    # Walk / move synonyms
    r"\bwalk(s|ing|ed)?\b": "walk",
    r"\brun(s|ning)?\b": "run",
    r"\bran\b": "run",
    r"\bmov(e|es|ing|ed)\b": "move",
    # Speech
    r"\bsay(s|ing)?\b": "speech",
    r"\bsaid\b": "speech",
    r"\bspeak(s|ing)?\b": "speech",
    r"\bspoke\b": "speech",
    r"\btalk(s|ing|ed)?\b": "speech",
    r"\bshout(s|ing|ed)?\b": "speech",
    r"\bcall(s|ing|ed)?\b": "speech",
}

# License plate pattern: 2+ uppercase letters + digits, like MH12AB4587
_PLATE_PATTERN = re.compile(
    r'\b([A-Z]{2}\s?\d{1,2}\s?[A-Z]{1,3}\s?\d{3,4})\b'
    r'|\b([A-Z0-9]{5,10})\b'  # fallback for non-standard formats
)

# Color adjectives
_COLORS = {"red", "blue", "green", "black", "white", "silver", "grey", "gray", "yellow",
           "orange", "brown", "purple", "maroon", "gold", "dark", "light"}

# Context / setting keywords
_CONTEXT_WORDS = {"cctv", "footage", "camera", "scene", "road", "street", "parking", "area",
                  "nearby", "near", "around", "gate", "entrance", "exit", "building"}

# Temporal intent keywords
_WHEN_WORDS = {"when", "what time", "at what", "moment", "timestamp", "time"}


# ─────────────────────────────────────────────────────────────────────────────
# Parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_query(
    user_query: str,
    *,
    # These keyword args are accepted but IGNORED — keeps signature compatible
    # with the old Gemini-based parser so callers don't need to change.
    client=None,
    model: str = "",
    max_tokens: int = 1024,
) -> StructuredQuery:
    """
    Parse a natural-language query into a StructuredQuery using deterministic
    keyword/regex logic.  No LLM, no network, no API key required.

    Parameters
    ----------
    user_query : Raw user search string.
    client     : Ignored (compatibility parameter).
    model      : Ignored (compatibility parameter).
    max_tokens : Ignored (compatibility parameter).

    Returns
    -------
    StructuredQuery
    """
    q = user_query.lower().strip()
    log.debug("local parse_query: %r", user_query)

    objects: list[AttributedObject] = []
    actions: list[str] = []
    persons: list[str] = []
    speech_intent: list[str] = []
    context: list[str] = []

    # ── 1. License plate detection ──────────────────────────────────────────
    # Check original query (upper case) for plate-like strings
    for m in _PLATE_PATTERN.finditer(user_query):
        plate = m.group(0).replace(" ", "").upper()
        if len(plate) >= 5 and any(c.isdigit() for c in plate):
            objects.append(AttributedObject(object="license_plate", attributes=[plate]))

    # ── 2. Vehicle detection ────────────────────────────────────────────────
    vehicle_found = False
    for v in _VEHICLES:
        if re.search(r'\b' + re.escape(v) + r'\b', q):
            vehicle_found = True
            # Detect color attributes
            attrs = [c for c in _COLORS if re.search(r'\b' + c + r'\b', q)]
            # Only add if not already added (avoid duplicates)
            if not any(o.object == "vehicle" for o in objects):
                objects.append(AttributedObject(object="vehicle", attributes=attrs))
            break

    # ── 3. Person detection ─────────────────────────────────────────────────
    for p in _PERSONS:
        if re.search(r'\b' + re.escape(p) + r'\b', q):
            label = "people" if p in ("people", "individuals", "pedestrians", "passengers", "occupants") else "person"
            if label not in persons:
                persons.append(label)

    # ── 4. Actions / event detection ───────────────────────────────────────
    for pattern, action in _ACTION_MAP.items():
        if re.search(pattern, q) and action not in actions:
            actions.append(action)

    # ── 5. Speech intent ────────────────────────────────────────────────────
    if "speech" in actions:
        actions.remove("speech")
        # Determine what speech topic might be
        speech_hints = []
        if re.search(r'\bsay(s|ing)?\b|\bsaid\b', q):
            speech_hints.append("what was said")
        if re.search(r'\btalk(s|ing|ed)?\b', q):
            speech_hints.append("conversation")
        if re.search(r'\bshout(s|ing|ed)?\b', q):
            speech_hints.append("shouting")
        speech_intent.extend(speech_hints or ["speech content"])

    # ── 6. Context keywords ─────────────────────────────────────────────────
    for cw in _CONTEXT_WORDS:
        if re.search(r'\b' + re.escape(cw) + r'\b', q):
            context.append(cw)

    # ── 7. Temporal intent ─────────────────────────────────────────────────
    is_temporal = any(re.search(r'\b' + re.escape(w) + r'\b', q) for w in _WHEN_WORDS)
    if is_temporal and "timestamp" not in context:
        context.append("timestamp_query")

    # ── 8. Fallback: if query appears to be about a specific plate ─────────
    # e.g. "Find MH12AB4587" or "vehicle MH12AB4587"
    if not objects and not persons and not actions:
        # Generic fallback — capture key nouns as objects
        nouns = re.findall(r'\b([a-z]{4,})\b', q)
        for noun in nouns:
            if noun not in {"find", "show", "when", "what", "where", "does", "did", "the",
                            "and", "with", "from", "that", "this", "them", "they"}:
                objects.append(AttributedObject(object=noun, attributes=[]))
                if len(objects) >= 3:
                    break

    sq = StructuredQuery(
        objects=objects,
        actions=actions,
        persons=persons,
        speech_intent=speech_intent,
        context=context,
        raw_query=user_query,
    )
    log.info("local parse_query(%r) -> %s", user_query, sq.summary())
    return sq
