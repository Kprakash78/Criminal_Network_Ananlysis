"""
M1.9 — Output Schema + Emitter
PS 26152 — AI-Powered Criminal Network Analysis System

Defines the shared pydantic ENTITY and RELATIONSHIP schemas (exact match
with backend.md §4) and emits validated JSON files and an SQLite database.

IMPORTANT: This schema is the project-wide contract. Do NOT alter field
names or types without notifying M2/M4 teams — their code imports these models.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from M1.resolver import ResolvedEntity


# ---------------------------------------------------------------------------
# Shared pydantic schema (MUST match backend.md §4 exactly)
# ---------------------------------------------------------------------------

VALID_ENTITY_TYPES = {
    "PERSON", "PHONE", "VEHICLE", "LOCATION",
    "ORGANIZATION", "ACCOUNT", "CASE", "DATE",
}


class Entity(BaseModel):
    """Shared ENTITY schema — consumed by M2 (graph) and M4 (RAG)."""
    entity_id: str = Field(..., description="Stable unique identifier, e.g. PER_a1b2c3d4")
    type: str = Field(..., description="Entity type from the fixed set")
    name: str = Field(..., description="Canonical name/value for this entity")
    aliases: list[str] = Field(default_factory=list, description="All known surface forms")
    source_documents: list[str] = Field(default_factory=list, description="Source doc IDs")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Final confidence score")
    needs_review: bool = Field(default=False, description="True if confidence is low")

    @field_validator("type")
    @classmethod
    def type_must_be_valid(cls, v: str) -> str:
        if v not in VALID_ENTITY_TYPES:
            raise ValueError(f"Invalid entity type '{v}'. Must be one of {VALID_ENTITY_TYPES}")
        return v

    @field_validator("aliases")
    @classmethod
    def deduplicate_aliases(cls, v: list[str]) -> list[str]:
        return list(dict.fromkeys(v))   # preserve order, remove dupes


class Relationship(BaseModel):
    """
    Shared RELATIONSHIP schema — basic co-occurrence produced by M1.
    Deeper relationship typing is M2's job.
    """
    source: str = Field(..., description="entity_id of source entity")
    target: str = Field(..., description="entity_id of target entity")
    relationship: str = Field(default="APPEARS_IN_SAME_DOCUMENT")
    timestamp: str = Field(..., description="ISO 8601 timestamp of the source record")
    source_record: str = Field(..., description="doc_id of the document establishing this link")
    confidence: float = Field(..., ge=0.0, le=1.0)


class PipelineResult(BaseModel):
    """Top-level container returned by run_pipeline()."""
    entities: list[Entity]
    relationships: list[Relationship]
    extraction_log: list[dict]
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    total_documents: int
    total_entities: int
    total_relationships: int


# ---------------------------------------------------------------------------
# Converter: ResolvedEntity → Entity (validated pydantic object)
# ---------------------------------------------------------------------------

def resolved_to_entity(resolved: ResolvedEntity) -> Entity:
    """Convert a ResolvedEntity to the validated pydantic Entity schema."""
    return Entity(
        entity_id=resolved.entity_id,
        type=resolved.entity_type,
        name=resolved.canonical,
        aliases=list(set(resolved.aliases)),
        source_documents=list(set(resolved.source_docs)),
        confidence=resolved.confidence,
        needs_review=resolved.needs_review,
    )


# ---------------------------------------------------------------------------
# Co-occurrence relationship builder
# ---------------------------------------------------------------------------

def build_relationships(
    entities: list[Entity],
    doc_entity_map: dict[str, list[str]],   # doc_id → [entity_id, ...]
    doc_type_map: dict[str, str] = None,    # doc_id → doc_type
) -> list[Relationship]:
    """
    Build co-occurrence relationships for every pair of entities
    that co-occur in the same source document.
    If the document is an FIR, grant an APPEARS_IN_CASE edge.
    Otherwise, default to APPEARS_IN_SAME_DOCUMENT.
    """
    if doc_type_map is None:
        doc_type_map = {}
        
    relationships: list[Relationship] = []
    seen: set[tuple[str, str, str]] = set()   # (src_id, tgt_id, doc_id) dedup

    for doc_id, ent_ids in doc_entity_map.items():
        # Deterministic timestamp: use midnight of today as a placeholder
        # (real timestamp comes from document metadata where available)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        
        doc_type = doc_type_map.get(doc_id, "UNKNOWN").upper()
        # If it's an FIR, emit APPEARS_IN_CASE. Else fallback.
        rel_type = "APPEARS_IN_CASE" if doc_type == "FIR" else "APPEARS_IN_SAME_DOCUMENT"

        unique_ids = list(dict.fromkeys(ent_ids))   # preserve order, dedup
        for i, src_id in enumerate(unique_ids):
            for tgt_id in unique_ids[i + 1:]:
                key = (src_id, tgt_id, doc_id)
                if key in seen:
                    continue
                seen.add(key)

                # Confidence is min of the two entity confidences
                src_conf = next((e.confidence for e in entities if e.entity_id == src_id), 0.5)
                tgt_conf = next((e.confidence for e in entities if e.entity_id == tgt_id), 0.5)

                relationships.append(Relationship(
                    source=src_id,
                    target=tgt_id,
                    relationship=rel_type,
                    timestamp=ts,
                    source_record=doc_id,
                    confidence=round(min(src_conf, tgt_conf), 4),
                ))

    return relationships


# ---------------------------------------------------------------------------
# JSON emitter
# ---------------------------------------------------------------------------

def emit_json(result: PipelineResult, output_dir: Path) -> dict[str, Path]:
    """Write entities.json, relationships.json, and extraction_log.json."""
    output_dir.mkdir(parents=True, exist_ok=True)

    entities_path = output_dir / "entities.json"
    relationships_path = output_dir / "relationships.json"
    log_path = output_dir / "extraction_log.json"

    entities_path.write_text(
        json.dumps([e.model_dump() for e in result.entities], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    relationships_path.write_text(
        json.dumps([r.model_dump() for r in result.relationships], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log_path.write_text(
        json.dumps(result.extraction_log, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "entities": entities_path,
        "relationships": relationships_path,
        "extraction_log": log_path,
    }


# ---------------------------------------------------------------------------
# SQLite emitter
# ---------------------------------------------------------------------------

def emit_sqlite(result: PipelineResult, output_dir: Path) -> Path:
    """Write entities and relationships to an SQLite database."""
    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = output_dir / "criminal_network.db"

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.executescript("""
        CREATE TABLE IF NOT EXISTS entities (
            entity_id        TEXT PRIMARY KEY,
            type             TEXT NOT NULL,
            name             TEXT NOT NULL,
            aliases          TEXT,          -- JSON array
            source_documents TEXT,          -- JSON array
            confidence       REAL NOT NULL,
            needs_review     INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS relationships (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            source           TEXT NOT NULL,
            target           TEXT NOT NULL,
            relationship     TEXT NOT NULL,
            timestamp        TEXT,
            source_record    TEXT,
            confidence       REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS extraction_log (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id           TEXT,
            doc_type         TEXT,
            status           TEXT,
            entity_count     INTEGER,
            error_message    TEXT
        );
    """)

    cur.executemany(
        """
        INSERT OR REPLACE INTO entities
            (entity_id, type, name, aliases, source_documents, confidence, needs_review)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                e.entity_id, e.type, e.name,
                json.dumps(e.aliases, ensure_ascii=False),
                json.dumps(e.source_documents, ensure_ascii=False),
                e.confidence,
                int(e.needs_review),
            )
            for e in result.entities
        ],
    )

    cur.executemany(
        """
        INSERT INTO relationships (source, target, relationship, timestamp, source_record, confidence)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (r.source, r.target, r.relationship, r.timestamp, r.source_record, r.confidence)
            for r in result.relationships
        ],
    )

    cur.executemany(
        """
        INSERT INTO extraction_log (doc_id, doc_type, status, entity_count, error_message)
        VALUES (:doc_id, :doc_type, :status, :entity_count, :error_message)
        """,
        result.extraction_log,
    )

    conn.commit()
    conn.close()
    return db_path
