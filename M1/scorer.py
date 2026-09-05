"""
M1.8 — Confidence Scorer
PS 26152 — AI-Powered Criminal Network Analysis System

Combines extraction-method confidence and match-quality signals into a
final confidence score per entity.

Scoring formula:
  base_confidence = raw_confidence from extractor
  method_weight:
    regex     → 1.0 (deterministic, high trust)
    spacy     → 0.85
    indicbert → 0.80 (strong but domain-shifted)
  source_count_bonus: +0.02 per additional source doc (capped at +0.10)
  alias_diversity_bonus: +0.01 per unique alias (capped at +0.05)

  final = clip(base * method_weight + bonuses, 0.0, 1.0)

Entities with final < NEEDS_REVIEW_THRESHOLD get needs_review = True.
"""

from M1.resolver import ResolvedEntity

_NEEDS_REVIEW_THRESHOLD = 0.45

_METHOD_WEIGHTS = {
    "regex":      1.00,
    "spacy":      0.85,
    "indicbert":  0.80,
}
_DEFAULT_METHOD_WEIGHT = 0.75


def score_entity(entity: ResolvedEntity, extraction_methods: set[str]) -> ResolvedEntity:
    """
    Compute final confidence for a ResolvedEntity.

    `extraction_methods` is the set of extractors that contributed to this
    entity (carried through from NormalizedEntity.extraction_methods).

    Mutates entity.confidence and entity.needs_review in place.
    Returns the entity for chaining.
    """
    # Best method weight (pick highest trust among contributing methods)
    method_weight = max(
        (_METHOD_WEIGHTS.get(m, _DEFAULT_METHOD_WEIGHT) for m in extraction_methods),
        default=_DEFAULT_METHOD_WEIGHT,
    )

    base = entity.confidence * method_weight

    # Source count bonus: more docs mentioning this entity → more confidence
    source_bonus = min(0.02 * (len(entity.source_docs) - 1), 0.10)

    # Alias diversity bonus: more distinct surface forms → more confidence
    alias_bonus = min(0.01 * max(0, len(set(entity.aliases)) - 1), 0.05)

    final = min(base + source_bonus + alias_bonus, 1.0)
    final = max(final, 0.0)

    entity.confidence = round(final, 4)
    entity.needs_review = final < _NEEDS_REVIEW_THRESHOLD

    return entity
