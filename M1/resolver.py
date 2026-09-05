"""
M1.7 — Entity Resolver
PS 26152 — AI-Powered Criminal Network Analysis System

Fuzzy-matches incoming NormalizedEntity objects against an existing entity
store, merging entities that refer to the same real-world entity.

Uses rapidfuzz for fast string similarity. Matching is type-scoped: only
entities of the same type are compared (no PERSON–LOCATION confusion).

Produces ResolvedEntity objects with persistent, deterministic entity_id
values based on type prefix + SHA-256 hash of the canonical text.
"""

import hashlib
from dataclasses import dataclass, field
from typing import Optional

from rapidfuzz import fuzz, process
from M1.normalizer import NormalizedEntity


# ---------------------------------------------------------------------------
# Resolved entity — output of the resolver, input to scorer and emitter
# ---------------------------------------------------------------------------

@dataclass
class ResolvedEntity:
    """
    A fully resolved entity with a stable entity_id.

    entity_id    : deterministic ID (e.g. "P001", "PHN001" …)
    entity_type  : PERSON | PHONE | VEHICLE | LOCATION | ORGANIZATION |
                   ACCOUNT | DATE | CASE
    canonical    : the best/most-frequent normalised name for this entity
    aliases      : all surface forms ever seen (including canonical)
    source_docs  : union of all source_doc_ids from merged entities
    confidence   : raw_confidence (further adjusted by scorer in M1.8)
    needs_review : True when confidence is low — entity is retained, not dropped
    """
    entity_id: str
    entity_type: str
    canonical: str
    aliases: list[str]
    source_docs: list[str]
    confidence: float
    needs_review: bool = False


# ---------------------------------------------------------------------------
# Type-prefix table for entity_id generation
# ---------------------------------------------------------------------------

_TYPE_PREFIX = {
    "PERSON":       "PER",
    "PHONE":        "PHN",
    "VEHICLE":      "VEH",
    "LOCATION":     "LOC",
    "ORGANIZATION": "ORG",
    "ACCOUNT":      "ACC",
    "DATE":         "DAT",
    "CASE":         "CAS",
}

_NEEDS_REVIEW_THRESHOLD = 0.45   # confidence below this triggers needs_review

# Fuzzy-matching thresholds per type (structured fields are exact, free text is fuzzy)
_MATCH_THRESHOLDS = {
    "PERSON":       80,    # partial_ratio handles "R. Kumar" vs "Ravi Kumar"
    "LOCATION":     85,
    "ORGANIZATION": 80,
    "PHONE":        100,   # exact match only
    "VEHICLE":      100,   # exact match only
    "ACCOUNT":      100,   # exact match only
    "DATE":         70,
    "CASE":         95,
}


def _make_entity_id(entity_type: str, canonical: str, case_prefix: str = "") -> str:
    """
    Deterministic entity ID: type-prefix + first 8 hex chars of SHA-256(canonical).
    Satisfies the NFR: same canonical → same ID across runs.
    """
    prefix = _TYPE_PREFIX.get(entity_type, "ENT")
    
    # If the canonical string is already a formatted ID (like ACC_dac31588), use it directly!
    import re
    if re.fullmatch(rf"{prefix}_[a-fA-F0-9]{{8}}", canonical):
        return canonical
        
    hash_hex = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]
    if case_prefix:
        return f"{case_prefix}_{prefix}_{hash_hex}"
    return f"{prefix}_{hash_hex}"


def _fuzzy_match(
    candidate: str,
    existing_canonicals: list[str],
    threshold: int,
    entity_type: str,
) -> Optional[str]:
    """
    Try to match `candidate` against `existing_canonicals`.
    Uses token_sort_ratio for PERSON (handles "Ravi Kumar" vs "R. Kumar"),
    plain ratio for others.
    Returns the best-matching canonical if score ≥ threshold, else None.
    """
    if not existing_canonicals:
        return None

    if entity_type == "PERSON":
        scorer = fuzz.partial_ratio    # handles initials: "R. Kumar" ~ "Ravi Kumar"
    elif entity_type == "ORGANIZATION":
        scorer = fuzz.partial_ratio
    else:
        scorer = fuzz.ratio

    # rapidfuzz.process.extractOne returns (match, score, index)
    result = process.extractOne(
        candidate,
        existing_canonicals,
        scorer=scorer,
        score_cutoff=threshold,
    )
    if result is None:
        return None
    return result[0]   # matched canonical string


# ---------------------------------------------------------------------------
# Entity store (in-memory, keyed by entity_id)
# ---------------------------------------------------------------------------

class EntityStore:
    """
    In-memory store of resolved entities.
    Provides fast lookup by canonical text (per type) for fuzzy matching.
    """

    def __init__(self) -> None:
        self._entities: dict[str, ResolvedEntity] = {}   # entity_id → entity
        # Index: entity_type → list of (canonical, entity_id)
        self._index: dict[str, list[tuple[str, str]]] = {}

    def all(self) -> list[ResolvedEntity]:
        return list(self._entities.values())

    def get_by_id(self, entity_id: str) -> Optional[ResolvedEntity]:
        return self._entities.get(entity_id)

    def _index_canonicals(self, entity_type: str) -> list[str]:
        return [c for c, _ in self._index.get(entity_type, [])]

    def _index_id_for(self, entity_type: str, canonical: str) -> Optional[str]:
        for c, eid in self._index.get(entity_type, []):
            if c == canonical:
                return eid
        return None

    def add(self, entity: ResolvedEntity) -> None:
        self._entities[entity.entity_id] = entity
        bucket = self._index.setdefault(entity.entity_type, [])
        bucket.append((entity.canonical, entity.entity_id))

    def merge_into(self, entity_id: str, new_entity: NormalizedEntity) -> None:
        """Merge new_entity's data into an existing ResolvedEntity."""
        existing = self._entities[entity_id]
        for doc in new_entity.source_doc_ids:
            if doc not in existing.source_docs:
                existing.source_docs.append(doc)
        for alias in new_entity.raw_aliases:
            if alias not in existing.aliases:
                existing.aliases.append(alias)
        if new_entity.normalized_text not in existing.aliases:
            existing.aliases.append(new_entity.normalized_text)
        existing.confidence = max(existing.confidence, new_entity.raw_confidence)
        existing.needs_review = existing.confidence < _NEEDS_REVIEW_THRESHOLD


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------

def resolve_entities(
    new_entities: list[NormalizedEntity],
    store: EntityStore,
) -> list[ResolvedEntity]:
    """
    Match each NormalizedEntity against the existing store.
    - If a match is found (fuzzy similarity ≥ threshold), merge into existing.
    - Otherwise, create a new ResolvedEntity and add to store.

    Returns the list of ResolvedEntity objects created or updated in this call.
    """
    touched: list[ResolvedEntity] = []

    for norm in new_entities:
        threshold = _MATCH_THRESHOLDS.get(norm.entity_type, 85)
        
        # Isolate by case if possible (doc_id format CAS001_FIR01 -> CAS001)
        case_prefix = ""
        if norm.source_doc_ids:
            first_doc = norm.source_doc_ids[0]
            case_prefix = first_doc.split("_")[0]
            
        # Only fuzzy match against entities that belong to the same case prefix
        existing_in_case = [
            c for c, eid in store._index.get(norm.entity_type, [])
            if not case_prefix or eid.startswith(f"{case_prefix}_")
        ]

        matched_canonical = _fuzzy_match(
            norm.normalized_text,
            existing_in_case,
            threshold,
            norm.entity_type,
        )

        if matched_canonical is not None:
            # Merge into existing
            eid = None
            for c, candidate_eid in store._index.get(norm.entity_type, []):
                if c == matched_canonical and (not case_prefix or candidate_eid.startswith(f"{case_prefix}_")):
                    eid = candidate_eid
                    break
            if eid:
                store.merge_into(eid, norm)
                touched.append(store.get_by_id(eid))
        else:
            # Create new resolved entity
            canonical = norm.normalized_text
            eid = _make_entity_id(norm.entity_type, canonical, case_prefix)

            # Handle (unlikely) hash collision: append a counter
            if eid in store._entities:
                eid = eid + "_x"

            resolved = ResolvedEntity(
                entity_id=eid,
                entity_type=norm.entity_type,
                canonical=canonical,
                aliases=list(norm.raw_aliases | {canonical}),
                source_docs=list(norm.source_doc_ids),
                confidence=norm.raw_confidence,
                needs_review=norm.raw_confidence < _NEEDS_REVIEW_THRESHOLD,
            )
            store.add(resolved)
            touched.append(resolved)

    return touched
