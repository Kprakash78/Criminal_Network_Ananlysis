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
    Deterministic entity ID: type-prefix + first 8 hex chars of SHA-256(type + canonical).
    Satisfies the NFR: same canonical and type → same ID across runs.
    """
    prefix = _TYPE_PREFIX.get(entity_type, "ENT")
    
    # If the canonical string is already a formatted ID (like ACC_dac31588), use it directly!
    import re
    if re.fullmatch(rf"{prefix}_[a-fA-F0-9]{{8}}", canonical):
        return canonical
        
    hash_source = f"{entity_type}:{canonical}"
    hash_hex = hashlib.sha256(hash_source.encode("utf-8")).hexdigest()[:8]
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
    Uses custom strict token_sort_ratio for PERSON to avoid false-positive substring merges,
    while still handling initials (e.g. "R. Kumar" ~ "Ravi Kumar").
    Returns the best-matching canonical if score >= threshold, else None.
    """
    if not existing_canonicals:
        return None

    if entity_type == "PERSON":
        best_match = None
        best_score = 0
        for existing in existing_canonicals:
            # Using token_sort_ratio prevents substring overlap from scoring 100
            # e.g., token_sort_ratio("Reshma Solapur", "Solapur") = 66
            score = fuzz.token_sort_ratio(candidate, existing)

            if score < threshold:
                # Extended initials/abbreviation logic:
                # Handles cases like:
                #   "Ravi K."    ↔ "Ravi Kumar"        (abbreviated last name)
                #   "R. Kumar"   ↔ "Ravi Kumar"        (abbreviated first name)
                #   "S. Verma"   ↔ "Suresh Verma"      (abbreviated first name, 2-word)
                #   "S. Verma"   ↔ "Suresh K. Verma"   (abbreviated first, middle initial)
                c_parts = [p.rstrip('.') for p in candidate.split()]
                e_parts = [p.rstrip('.') for p in existing.split()]

                def _initial_matches(short_name: str, full_name: str) -> bool:
                    """
                    Check if short_name could be an abbreviated form of full_name.
                    'short_name' is 2 words, 'full_name' is 2 or 3 words.
                    An initial is a 1-char string (after stripping dots).
                    """
                    s = [p.rstrip('.') for p in short_name.split()]
                    f = [p.rstrip('.') for p in full_name.split()]

                    if len(s) != 2:
                        return False

                    # Case: last names match, first name is initial of full first name
                    # e.g. "R. Kumar" ↔ "Ravi Kumar": s=["R","Kumar"], f=["Ravi","Kumar"]
                    if len(f) == 2 and s[-1].lower() == f[-1].lower():
                        if (len(s[0]) == 1 and f[0].lower().startswith(s[0].lower())) or \
                           (len(f[0]) == 1 and s[0].lower().startswith(f[0].lower())):
                            return True

                    # Case: last names match, first name is partial first name
                    # e.g. "Ravi K." ↔ "Ravi Kumar": s=["Ravi","K"], f=["Ravi","Kumar"]
                    if len(f) == 2 and s[0].lower() == f[0].lower():
                        if (len(s[1]) == 1 and f[1].lower().startswith(s[1].lower())) or \
                           (len(f[1]) == 1 and s[1].lower().startswith(f[1].lower())):
                            return True

                    # Case: 3-word full name; short name is abbreviated form
                    # e.g. "S. Verma" ↔ "Suresh K. Verma" or "Suresh Verma"
                    if len(f) == 3:
                        # Try matching short against (f[0], f[2]) — drop middle
                        reduced_full = f"{f[0]} {f[2]}"
                        if _initial_matches(short_name, reduced_full):
                            return True
                        # Try matching short against (f[0][0], f[2]) — abbrev first
                        abbrev_full = f"{f[0][0]} {f[2]}"
                        if s[-1].lower() == f[-1].lower() and \
                           len(s[0]) == 1 and f[0].lower().startswith(s[0].lower()):
                            return True

                    return False

                candidate_str = candidate
                existing_str = existing

                if _initial_matches(candidate_str, existing_str) or \
                   _initial_matches(existing_str, candidate_str):
                    score = max(score, 85)

            if score > best_score:
                best_score = score
                best_match = existing

        if best_score >= threshold:
            return best_match
        return None

        
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
    return result[0]


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
        
        # Isolate by case/document. The full doc_id serves as the case identifier.
        case_id = ""
        if norm.source_doc_ids:
            case_id = norm.source_doc_ids[0]
            
        # Only fuzzy match against entities that belong to the same case_id
        existing_in_case = [
            c for c, eid in store._index.get(norm.entity_type, [])
            if not case_id or eid.startswith(f"{case_id}_")
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
                if c == matched_canonical and (not case_id or candidate_eid.startswith(f"{case_id}_")):
                    eid = candidate_eid
                    break
            if eid:
                store.merge_into(eid, norm)
                touched.append(store.get_by_id(eid))
        else:
            # Create new resolved entity
            canonical = norm.normalized_text
            eid = _make_entity_id(norm.entity_type, canonical, case_id)

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
