"""
M1.6 — Entity Normalizer
PS 26152 — AI-Powered Criminal Network Analysis System

Takes a list of RawEntity objects and normalises them:
  - Casing: PERSON/LOCATION/ORGANIZATION → title case (with exceptions)
  - PHONE → canonical 10-digit form (strip +91/0 prefix)
  - VEHICLE → uppercase
  - ACCOUNT → uppercase
  - Whitespace → stripped, collapsed
  - Alias collection: collects raw surface forms as aliases
  - Regex type precedence: for spans also matched by NER as wrong type,
    regex-extracted structured fields win (per PRD §11)
  - Returns list[NormalizedEntity]
"""

import re
from dataclasses import dataclass, field
from typing import Optional
from M1.extractor import RawEntity


# ---------------------------------------------------------------------------
# Normalised entity (output of this stage, input to resolver)
# ---------------------------------------------------------------------------

@dataclass
class NormalizedEntity:
    """
    A cleaned, type-validated entity ready for resolution and ID assignment.
    The `aliases` set accumulates every surface form seen across documents.
    """
    normalized_text: str
    entity_type: str
    source_doc_ids: list[str]           # all documents this text was seen in
    raw_aliases: set[str]               # all raw surface forms collected so far
    extraction_methods: set[str]        # which methods contributed
    raw_confidence: float               # best raw confidence seen so far


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

_PHONE_CLEANUP_RE = re.compile(r"[\s\-.]")   # strip separators inside number
_MULTI_SPACE_RE = re.compile(r"\s+")

# Common title abbreviations that should stay uppercased
_TITLE_ABBREV = {"Lt.", "Insp.", "SI", "ASI", "DCP", "SP", "DSP", "SHO", "IO"}


def _normalize_name(text: str) -> str:
    """
    Title-case a person/org/location name.
    Strips leading/trailing whitespace and collapses internal runs.
    """
    text = _MULTI_SPACE_RE.sub(" ", text.strip())
    # Title-case each word, but preserve known abbreviations
    words = text.split()
    result = []
    for w in words:
        if w.upper() in {a.upper() for a in _TITLE_ABBREV}:
            result.append(w.upper())
        else:
            result.append(w.capitalize())
    return " ".join(result)


def _normalize_phone(text: str) -> str:
    """Strip any prefix and separators; return bare 10-digit number."""
    digits = re.sub(r"\D", "", text)
    if digits.startswith("91") and len(digits) == 12:
        digits = digits[2:]
    elif digits.startswith("0") and len(digits) == 11:
        digits = digits[1:]
    return digits


def _normalize_vehicle(text: str) -> str:
    """Uppercase and remove internal spaces/hyphens."""
    return re.sub(r"[\s\-]", "", text.upper())


def _normalize_account(text: str) -> str:
    """Uppercase account numbers, except preserve lowercase for ACC_hex IDs."""
    text = text.strip()
    if text.upper().startswith("ACC_"):
        return "ACC_" + text[4:].lower()
    return text.upper()


def _normalize_text(entity_type: str, raw_text: str) -> Optional[str]:
    """
    Apply type-appropriate normalisation.
    Returns None if the entity should be discarded (e.g. too short, known noise).
    """
    text = raw_text.strip()
    if not text:
        return None

    if entity_type == "PHONE":
        norm = _normalize_phone(text)
        if len(norm) != 10 or not norm[0].isdigit() or int(norm[0]) < 6:
            return None   # failed validation — discard
        return norm

    if entity_type == "VEHICLE":
        norm = _normalize_vehicle(text)
        # Must match Indian format after normalisation
        if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}", norm):
            return None
        return norm

    if entity_type == "ACCOUNT":
        return _normalize_account(text)

    if entity_type in ("PERSON", "LOCATION", "ORGANIZATION"):
        norm = _normalize_name(text)
        # Discard single-character or very short non-meaningful tokens
        if len(norm) < 2:
            return None
        # Discard pure-digit "names"
        if re.fullmatch(r"\d+", norm):
            return None
        return norm

    if entity_type == "DATE":
        return text.strip()

    return text.strip()


# ---------------------------------------------------------------------------
# Regex takes precedence: mark spans occupied by regex entities
# so spaCy doesn't get to claim the same span as a wrong type.
# ---------------------------------------------------------------------------

def _apply_regex_precedence(entities: list[RawEntity]) -> list[RawEntity]:
    """
    Per PRD §11: regex-extracted structured fields (PHONE/VEHICLE/ACCOUNT)
    always win over NER for the same span.

    We identify regex spans and remove NER entities that overlap them.
    Returns the pruned list.
    """
    regex_spans: list[tuple[int, int]] = [
        (e.span_start, e.span_end)
        for e in entities
        if e.extraction_method == "regex" and e.span_start >= 0
    ]

    if not regex_spans:
        return entities

    def _overlaps(start: int, end: int) -> bool:
        for rs, re_ in regex_spans:
            if start < re_ and end > rs:
                return True
        return False

    result = []
    for e in entities:
        if e.extraction_method != "regex" and e.span_start >= 0:
            if _overlaps(e.span_start, e.span_end):
                continue   # drop this NER entity; regex covers the span
        result.append(e)

    return result


# ---------------------------------------------------------------------------
# Main normaliser
# ---------------------------------------------------------------------------

def normalize_entities(raw_entities: list[RawEntity]) -> list[NormalizedEntity]:
    """
    Normalise a list of RawEntity objects extracted from one or more documents.

    Steps:
      1. Apply regex-precedence pruning
      2. Normalise each entity's text for its type
      3. Group identical (normalized_text, entity_type) pairs, accumulating aliases
         and source_doc_ids

    Returns a list of NormalizedEntity objects (one per unique normalised text+type).
    """
    pruned = _apply_regex_precedence(raw_entities)

    # bucket: (normalized_text, entity_type) → NormalizedEntity
    bucket: dict[tuple[str, str], NormalizedEntity] = {}

    for raw in pruned:
        norm_text = _normalize_text(raw.entity_type, raw.text)
        if norm_text is None:
            continue   # discard invalid

        key = (norm_text, raw.entity_type)
        if key in bucket:
            existing = bucket[key]
            existing.raw_aliases.add(raw.text)
            if raw.source_doc_id not in existing.source_doc_ids:
                existing.source_doc_ids.append(raw.source_doc_id)
            existing.extraction_methods.add(raw.extraction_method)
            existing.raw_confidence = max(existing.raw_confidence, raw.raw_confidence)
        else:
            bucket[key] = NormalizedEntity(
                normalized_text=norm_text,
                entity_type=raw.entity_type,
                source_doc_ids=[raw.source_doc_id],
                raw_aliases={raw.text},
                extraction_methods={raw.extraction_method},
                raw_confidence=raw.raw_confidence,
            )

    return list(bucket.values())
