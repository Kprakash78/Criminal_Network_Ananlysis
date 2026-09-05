"""
M2.2 — Relationship Typing Rules
PS 26152 — AI-Powered Criminal Network Analysis System

Maps M1's raw co-occurrence relationships to semantically meaningful
typed edges based on source-record type and entity-type pair.

All typing rules are defined in RELATIONSHIP_TYPING_RULES (one place).
To add a new relationship type, add an entry there — no other code changes needed.

Typing priority (highest to lowest):
  1. Source-record prefix (CDR → CALLED, TXN → TRANSFERRED_MONEY_TO)
  2. Entity-type pair (PERSON → PHONE: OWNS, PERSON → VEHICLE: OWNS,
                       PERSON → LOCATION: LOCATED_AT, etc.)
  3. Default fallback: ASSOCIATED_WITH (generic)

Never infer guilt — these are evidence-link labels only.
"""

from M2.loader import M1Entity, M1Relationship

# ---------------------------------------------------------------------------
# Relationship type constants (must match backend.md §4 exactly)
# ---------------------------------------------------------------------------

class RelType:
    CALLED                = "CALLED"
    TRANSFERRED_MONEY_TO  = "TRANSFERRED_MONEY_TO"
    APPEARS_IN_CASE       = "APPEARS_IN_CASE"
    OWNS                  = "OWNS"
    LOCATED_AT            = "LOCATED_AT"
    ASSOCIATED_WITH       = "ASSOCIATED_WITH"
    WORKS_FOR             = "WORKS_FOR"
    CONNECTED_TO          = "CONNECTED_TO"
    # M1 passthrough (not used in typed graph — only present during loading)
    APPEARS_IN_SAME_DOCUMENT = "APPEARS_IN_SAME_DOCUMENT"


VALID_REL_TYPES = {
    RelType.CALLED, RelType.TRANSFERRED_MONEY_TO, RelType.APPEARS_IN_CASE,
    RelType.OWNS, RelType.LOCATED_AT, RelType.ASSOCIATED_WITH,
    RelType.WORKS_FOR, RelType.CONNECTED_TO, RelType.APPEARS_IN_SAME_DOCUMENT,
}

# ---------------------------------------------------------------------------
# Rule tables — single source of truth for all relationship-typing decisions
# ---------------------------------------------------------------------------

# 1. Source-record prefix → relationship type
#    Matched against source_record.upper() with startswith()
SOURCE_RECORD_RULES: list[tuple[str, str]] = [
    # CDR (Call Detail Records): phone-to-phone calls
    ("CDR",         RelType.CALLED),
    # Transaction records
    ("TXN",         RelType.TRANSFERRED_MONEY_TO),
    ("TRANSACTION", RelType.TRANSFERRED_MONEY_TO),
    # FIR (First Information Report) as case reference
    ("FIR",         RelType.APPEARS_IN_CASE),
]

# 2. Entity-type pair → relationship type
#    Keys: (source_entity_type, target_entity_type)
#    Used when source_record prefix doesn't yield a typed rule.
ENTITY_TYPE_PAIR_RULES: dict[tuple[str, str], str] = {
    # A person owning a phone number
    ("PERSON",       "PHONE"):        RelType.OWNS,
    ("PHONE",        "PERSON"):       RelType.OWNS,
    # A person owning a vehicle
    ("PERSON",       "VEHICLE"):      RelType.OWNS,
    ("VEHICLE",      "PERSON"):       RelType.OWNS,
    # A person owning an account
    ("PERSON",       "ACCOUNT"):      RelType.OWNS,
    ("ACCOUNT",      "PERSON"):       RelType.OWNS,
    # A person at a location
    ("PERSON",       "LOCATION"):     RelType.LOCATED_AT,
    ("LOCATION",     "PERSON"):       RelType.LOCATED_AT,
    # A person working for an organisation
    ("PERSON",       "ORGANIZATION"): RelType.WORKS_FOR,
    ("ORGANIZATION", "PERSON"):       RelType.WORKS_FOR,
    # Two accounts linked (money flow, no CDR/TXN prefix)
    ("ACCOUNT",      "ACCOUNT"):      RelType.TRANSFERRED_MONEY_TO,
    # Phone-to-phone (no CDR prefix but co-occurring)
    ("PHONE",        "PHONE"):        RelType.CONNECTED_TO,
    # Person-to-person (same FIR, no stronger signal)
    ("PERSON",       "PERSON"):       RelType.ASSOCIATED_WITH,
}

# 3. Default fallback when no rule matches
DEFAULT_REL_TYPE = RelType.ASSOCIATED_WITH


# ---------------------------------------------------------------------------
# Main typing function
# ---------------------------------------------------------------------------

def assign_relationship_type(
    rel: M1Relationship,
    entity_index: dict[str, M1Entity],
) -> str:
    """
    Determine the typed relationship label for a given M1 co-occurrence.

    Decision order:
      1. Source-record prefix (CDR, TXN, FIR …)
      2. Entity-type pair (PERSON→PHONE = OWNS, etc.)
      3. Default: ASSOCIATED_WITH

    If rel.relationship is already a typed M2 type (not APPEARS_IN_SAME_DOCUMENT),
    it is returned unchanged so already-typed edges survive round-trips.
    """
    # Pass through edges that are already typed
    if rel.relationship != RelType.APPEARS_IN_SAME_DOCUMENT:
        return rel.relationship

    # 1. Source-record prefix matching
    sr_upper = rel.source_record.upper()
    for prefix, rel_type in SOURCE_RECORD_RULES:
        if sr_upper.startswith(prefix):
            return rel_type

    # 2. Entity-type pair matching
    src_entity = entity_index.get(rel.source)
    tgt_entity = entity_index.get(rel.target)
    if src_entity and tgt_entity:
        pair = (src_entity.type, tgt_entity.type)
        if pair in ENTITY_TYPE_PAIR_RULES:
            return ENTITY_TYPE_PAIR_RULES[pair]

    # 3. Default fallback
    return DEFAULT_REL_TYPE


def assign_weight(rel: M1Relationship, rel_type: str) -> float:
    """
    Assign an edge weight. Higher = stronger evidence signal.
    Used by shortest-path and analytics downstream.

    Weights are evidence-strength, not criminality scores.
    """
    # Direct communication/financial links carry more weight
    if rel_type in (RelType.CALLED, RelType.TRANSFERRED_MONEY_TO):
        return min(rel.confidence * 2.0, 1.0)   # boost direct links
    return rel.confidence
