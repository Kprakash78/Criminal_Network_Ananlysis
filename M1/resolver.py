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
    case_id: str = "",
) -> list[ResolvedEntity]:
    """
    Match each NormalizedEntity against the existing store.
    - If a match is found (fuzzy similarity >= threshold), merge into existing.
    - Otherwise, create a new ResolvedEntity and add to store.

    Isolation model:
      The `case_id` parameter controls the fuzzy-match scope.
      Only entities whose entity_id starts with `case_id + '_'` are candidates
      for fuzzy matching. This ensures entities from different cases (different
      FIR investigations) are never incorrectly merged, even when they share the
      same EntityStore instance within a single pipeline run.

      When `case_id` is supplied by the caller, it is used for BOTH match-scoping
      AND entity ID generation. This allows multiple documents from the SAME
      logical case (e.g., a primary FIR + a supplementary statement with the same
      case number) to share a case_id and merge correctly.

      When `case_id` is NOT supplied, it falls back to the entity's own
      source_doc_id, which gives per-document isolation (the original behaviour).

    Examples:
      # Single document (default): entities are isolated per doc
      resolve_entities(normed, store)

      # Multi-document same case: explicitly group under one case_id
      resolve_entities(normed, store, case_id="FIR_2026_00931")

    Returns the list of ResolvedEntity objects created or updated in this call.
    """
    touched: list[ResolvedEntity] = []

    for norm in new_entities:
        threshold = _MATCH_THRESHOLDS.get(norm.entity_type, 85)

        # Determine the effective case scope for this entity.
        # Explicit case_id argument takes priority; falls back to source_doc_id.
        effective_case = case_id
        if not effective_case and norm.source_doc_ids:
            effective_case = norm.source_doc_ids[0]

        # Fuzzy-match ONLY against entities in the same case.
        # This is the isolation guarantee: different cases never merge.
        existing_in_case = [
            c for c, eid in store._index.get(norm.entity_type, [])
            if not effective_case or eid.startswith(f"{effective_case}_")
        ]

        matched_canonical = _fuzzy_match(
            norm.normalized_text,
            existing_in_case,
            threshold,
            norm.entity_type,
        )

        if matched_canonical is not None:
            # Merge into existing entity within this case
            eid = None
            for c, candidate_eid in store._index.get(norm.entity_type, []):
                if c == matched_canonical and (
                    not effective_case or candidate_eid.startswith(f"{effective_case}_")
                ):
                    eid = candidate_eid
                    break
            if eid:
                store.merge_into(eid, norm)
                touched.append(store.get_by_id(eid))
        else:
            # Create new resolved entity under the effective case
            canonical = norm.normalized_text
            eid = _make_entity_id(norm.entity_type, canonical, effective_case)

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



# ---------------------------------------------------------------------------
# Cross-type duplicate merger (Bug 2 fix)
# ---------------------------------------------------------------------------

# Type priority for human names: when two entities collide with different types,
# the one with the higher priority wins and absorbs the lower one.
_TYPE_PRIORITY = {
    "PERSON":       5,
    "ORGANIZATION": 4,
    "LOCATION":     3,
    "ACCOUNT":      2,
    "VEHICLE":      2,
    "PHONE":        2,
    "DATE":         1,
    "CASE":         1,
}


def deduplicate_cross_type(
    store: EntityStore,
    case_id: str = "",
    threshold: int = 92,
) -> list[str]:
    """
    Cross-type duplicate detection and merge.

    After entity resolution, scan all entities for pairs that:
      (a) share the same case_id prefix (same document)
      (b) have different entity_types
      (c) have canonical text that fuzzy-matches at >= threshold

    When found, merge the lower-priority-type entity into the higher-priority
    one (e.g., 'Vikram Oberoi' LOCATION is absorbed by 'Vikram Oberoi' PERSON).

    Returns a list of entity_ids that were removed (merged away).

    This is the fix for Bug 2: spaCy sometimes tags human names as FAC/GPE
    (which maps to LOCATION). The type-strict resolver then creates two separate
    nodes for the same person. This pass collapses them.
    """
    all_entities = [
        e for e in store.all()
        if not case_id or e.entity_id.startswith(f"{case_id}_")
    ]

    removed_ids: list[str] = []

    # Build (entity_id, canonical, type) triples for comparison
    candidates = [(e.entity_id, e.canonical.lower(), e.entity_type) for e in all_entities]

    for i, (eid_a, canon_a, type_a) in enumerate(candidates):
        if eid_a in removed_ids:
            continue
        for j, (eid_b, canon_b, type_b) in enumerate(candidates):
            if i >= j:
                continue
            if eid_b in removed_ids:
                continue
            if type_a == type_b:
                continue  # Same type — normal resolver already handles this

            # Only consider PERSON/LOCATION/ORGANIZATION collisions
            # (structured types like PHONE/VEHICLE are exact-match only and
            # should never be cross-merged)
            if type_a not in ("PERSON", "LOCATION", "ORGANIZATION") or \
               type_b not in ("PERSON", "LOCATION", "ORGANIZATION"):
                continue

            # Fuzzy match the canonicals
            score = fuzz.token_sort_ratio(canon_a, canon_b)
            if score < threshold:
                continue

            # They match — decide which one wins
            priority_a = _TYPE_PRIORITY.get(type_a, 0)
            priority_b = _TYPE_PRIORITY.get(type_b, 0)

            entity_a = store.get_by_id(eid_a)
            entity_b = store.get_by_id(eid_b)
            if entity_a is None or entity_b is None:
                continue

            if priority_a >= priority_b:
                winner, loser = entity_a, entity_b
            else:
                winner, loser = entity_b, entity_a

            # Merge loser into winner
            for doc in loser.source_docs:
                if doc not in winner.source_docs:
                    winner.source_docs.append(doc)
            for alias in loser.aliases:
                if alias not in winner.aliases:
                    winner.aliases.append(alias)
            winner.confidence = max(winner.confidence, loser.confidence)
            winner.needs_review = winner.confidence < _NEEDS_REVIEW_THRESHOLD

            # Remove loser from store
            store._entities.pop(loser.entity_id, None)
            # Remove loser from index
            for etype in list(store._index.keys()):
                store._index[etype] = [
                    (c, eid) for c, eid in store._index[etype]
                    if eid != loser.entity_id
                ]

            removed_ids.append(loser.entity_id)

    return removed_ids
