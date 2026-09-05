"""
M2.1 — Graph Loader
PS 26152 — AI-Powered Criminal Network Analysis System

Reads M1's entity/relationship JSON output, validates each record against
the shared pydantic schema, and returns clean Python objects ready for
graph construction.

Failure handling (per backend.md §7):
  - Missing or unreadable file → FileNotFoundError (explicit, not silent)
  - Individual entity/relationship records that fail validation → logged, skipped
  - A relationship referencing a missing entity → logged, skipped (in graph builder)
"""

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared schema: must match M1's output exactly (backend.md §4)
# ---------------------------------------------------------------------------

VALID_ENTITY_TYPES = {
    "PERSON", "PHONE", "VEHICLE", "LOCATION",
    "ORGANIZATION", "ACCOUNT", "CASE", "DATE",
}

VALID_RELATIONSHIP_TYPES = {
    # M1 emits this; M2 upgrades to typed relationships
    "APPEARS_IN_SAME_DOCUMENT",
    # M2 typed relationships
    "CALLED", "TRANSFERRED_MONEY_TO", "APPEARS_IN_CASE",
    "OWNS", "LOCATED_AT", "ASSOCIATED_WITH", "WORKS_FOR", "CONNECTED_TO",
}


class M1Entity(BaseModel):
    """M1 ENTITY schema — mirrors M1/schema.py Entity exactly."""
    entity_id: str
    type: str
    name: str
    aliases: list[str] = Field(default_factory=list)
    source_documents: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    needs_review: bool = False

    @field_validator("type")
    @classmethod
    def type_must_be_valid(cls, v: str) -> str:
        if v not in VALID_ENTITY_TYPES:
            raise ValueError(f"Invalid entity type '{v}'")
        return v


class M1Relationship(BaseModel):
    """M1 RELATIONSHIP schema — mirrors M1/schema.py Relationship exactly."""
    source: str
    target: str
    relationship: str = "APPEARS_IN_SAME_DOCUMENT"
    timestamp: str
    source_record: str
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("relationship")
    @classmethod
    def rel_must_be_valid(cls, v: str) -> str:
        if v not in VALID_RELATIONSHIP_TYPES:
            raise ValueError(f"Invalid relationship type '{v}'")
        return v


# ---------------------------------------------------------------------------
# Loader functions
# ---------------------------------------------------------------------------

def _load_json_file(path: Path) -> list[dict]:
    """Read a JSON file containing a list. Raises FileNotFoundError or ValueError."""
    if not path.exists():
        raise FileNotFoundError(f"M1 output file not found: {path}")
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}, got {type(data).__name__}")
    return data


def load_entities(entities_path: str) -> tuple[list[M1Entity], list[dict]]:
    """
    Load and validate M1 entity records.

    Returns:
        (valid_entities, skipped_records)
        skipped_records is a list of {'record': ..., 'error': ...} dicts.
    """
    path = Path(entities_path)
    raw_list = _load_json_file(path)

    valid: list[M1Entity] = []
    skipped: list[dict] = []

    for i, record in enumerate(raw_list):
        try:
            entity = M1Entity.model_validate(record)
            valid.append(entity)
        except Exception as exc:
            logger.warning(f"[Loader] Skipping entity record {i}: {exc} — record: {record!r}")
            skipped.append({"record": record, "error": str(exc)})

    logger.info(f"[Loader] Loaded {len(valid)} entities, skipped {len(skipped)}")
    return valid, skipped


def load_relationships(relationships_path: str) -> tuple[list[M1Relationship], list[dict]]:
    """
    Load and validate M1 relationship records.

    Returns:
        (valid_relationships, skipped_records)
    """
    path = Path(relationships_path)
    raw_list = _load_json_file(path)

    valid: list[M1Relationship] = []
    skipped: list[dict] = []

    for i, record in enumerate(raw_list):
        try:
            rel = M1Relationship.model_validate(record)
            valid.append(rel)
        except Exception as exc:
            logger.warning(f"[Loader] Skipping relationship record {i}: {exc}")
            skipped.append({"record": record, "error": str(exc)})

    logger.info(f"[Loader] Loaded {len(valid)} relationships, skipped {len(skipped)}")
    return valid, skipped


def load_m1_output(
    entities_path: str,
    relationships_path: str,
) -> tuple[list[M1Entity], list[M1Relationship]]:
    """
    Convenience wrapper: load both files and return only the valid records.
    Any per-record failures are logged but do not abort the load.
    """
    entities, _ = load_entities(entities_path)
    relationships, _ = load_relationships(relationships_path)
    return entities, relationships
