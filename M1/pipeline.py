"""
M1 Pipeline — Main entry point
PS 26152 — AI-Powered Criminal Network Analysis System

Implements the public interface defined in PRD §8:
  extract_entities(document_path, doc_type) -> list[Entity]
  resolve_entities(new_entities, existing_store)  -> list[Entity]
  run_pipeline(source_dir) -> PipelineResult

Usage:
  from M1.pipeline import run_pipeline
  result = run_pipeline("path/to/source_dir")
"""

import logging
from pathlib import Path
from typing import Optional

from M1.generate_dataset import generate_all
from M1.loader import Document, load_all, load_fir, load_cdrs, load_transactions
from M1.extractor import extract_all, RawEntity
from M1.normalizer import normalize_entities, NormalizedEntity
from M1.resolver import EntityStore, resolve_entities as _resolve, ResolvedEntity, deduplicate_cross_type
from M1.scorer import score_entity
from M1.schema import (
    Entity,
    Relationship,
    PipelineResult,
    resolved_to_entity,
    build_relationships,
    emit_json,
    emit_sqlite,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API (per PRD §8)
# ---------------------------------------------------------------------------

def extract_entities(document_path: str, doc_type: str) -> list[Entity]:
    """
    Extract entities from a single document file.

    document_path : path to a .txt (FIR) or .csv (CDR/TRANSACTION) file
    doc_type      : 'FIR' | 'CDR' | 'TRANSACTION'

    Returns a list of validated Entity objects.
    """
    path = Path(document_path)
    doc_type_upper = doc_type.upper()

    if doc_type_upper == "FIR":
        doc = load_fir(path)
        docs = [doc] if doc else []
    elif doc_type_upper == "CDR":
        docs = load_cdrs(path)
    elif doc_type_upper == "TRANSACTION":
        docs = load_transactions(path)
    else:
        raise ValueError(f"Unknown doc_type: {doc_type!r}. Use 'FIR', 'CDR', or 'TRANSACTION'.")

    store = EntityStore()
    for doc in docs:
        raw = extract_all(doc.doc_id, doc.clean_text)
        normed = normalize_entities(raw)
        _resolve(normed, store)

    # Bug 2 fix: cross-type deduplication (e.g., Vikram Oberoi PERSON + LOCATION → PERSON)
    for doc in docs:
        deduplicate_cross_type(store, case_id=doc.doc_id)

    # Score and convert
    entities = []
    for resolved in store.all():
        # Use a generic set for extraction_methods (single-doc call; we don't carry them here)
        score_entity(resolved, {"spacy"})
        entities.append(resolved_to_entity(resolved))

    return entities


def resolve_entities(new_entities: list[Entity], existing_store: EntityStore) -> list[Entity]:
    """
    Resolve a batch of already-validated Entity objects against an existing store.
    This is the public interface; internally maps Entity → NormalizedEntity for resolver.
    """
    normed = []
    for e in new_entities:
        normed.append(NormalizedEntity(
            normalized_text=e.name,
            entity_type=e.type,
            source_doc_ids=list(e.source_documents),
            raw_aliases=set(e.aliases),
            extraction_methods={"spacy"},
            raw_confidence=e.confidence,
        ))
    resolved = _resolve(normed, existing_store)
    return [resolved_to_entity(r) for r in resolved]


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_pipeline(source_dir: str) -> PipelineResult:
    """
    Run the complete M1 pipeline on a source directory.

    Expected layout (produced by generate_dataset.py):
      source_dir/data/firs/*.txt
      source_dir/data/cdrs/cdr.csv
      source_dir/data/transactions/transactions.csv

    Returns a PipelineResult with entities, relationships, and extraction log.
    """
    base = Path(source_dir)
    store = EntityStore()
    extraction_log = []

    # doc_id → list of entity_ids that appeared in that doc (for relationship building)
    doc_entity_map: dict[str, list[str]] = {}
    doc_type_map: dict[str, str] = {}

    # track extraction_methods per entity_id for scoring
    entity_methods: dict[str, set[str]] = {}

    # ---- Load all documents ----
    documents: list[Document] = load_all(base)
    logger.info(f"[Pipeline] Loaded {len(documents)} documents.")

    # ---- Per-document extraction, normalisation, resolution ----
    for doc in documents:
        doc_type_map[doc.doc_id] = doc.doc_type
        
        log_entry = {
            "doc_id": doc.doc_id,
            "doc_type": doc.doc_type,
            "status": "ok",
            "entity_count": 0,
            "error_message": None,
        }

        try:
            raw_entities: list[RawEntity] = extract_all(doc.doc_id, doc.clean_text)
            normed: list[NormalizedEntity] = normalize_entities(raw_entities)
            # NOTE: no case_id is passed here, so each document is its own isolation
            # bucket. Aliases for the same person across two files of the same case
            # will NOT be merged. This is safe (no cross-case merging) but means
            # cross-document alias resolution within a case does not occur in batch
            # mode. See M1/RUN_REPORT.md §5 Known Limitation 5 for full details and
            # the future fix path (Document.case_id + loader convention).
            resolved_batch: list[ResolvedEntity] = _resolve(normed, store)

            # For CDR/TRANSACTION, also extract from structured metadata fields
            # (phones and accounts appear there directly)
            if doc.doc_type in ("CDR", "TRANSACTION"):
                meta_text = " ".join(str(v) for v in doc.metadata.values())
                meta_raw = extract_all(doc.doc_id, meta_text)
                meta_normed = normalize_entities(meta_raw)
                # Same isolation note as above: per-document, no case_id passed.
                meta_resolved = _resolve(meta_normed, store)
                resolved_batch.extend(meta_resolved)

            # Track entity→doc mapping
            seen_ids: set[str] = set()
            for r in resolved_batch:
                if r.entity_id not in seen_ids:
                    doc_entity_map.setdefault(doc.doc_id, []).append(r.entity_id)
                    seen_ids.add(r.entity_id)

                # Accumulate extraction methods per entity
                for m in (store.get_by_id(r.entity_id).aliases or []):
                    pass  # just ensuring the entity exists
                methods = entity_methods.setdefault(r.entity_id, set())
                # Pick methods from the raw entities that contributed
                for n in normed:
                    if n.normalized_text == r.canonical or n.normalized_text in r.aliases:
                        methods.update(n.extraction_methods)

            log_entry["entity_count"] = len(seen_ids)

        except Exception as exc:
            logger.error(f"[Pipeline] Error processing {doc.doc_id}: {exc}")
            log_entry["status"] = "error"
            log_entry["error_message"] = str(exc)

        extraction_log.append(log_entry)

    # ---- Score all entities ----
    # Bug 2 fix: run cross-type deduplication before scoring.
    # This collapses pairs like (Vikram Oberoi LOCATION) + (Vikram Oberoi PERSON)
    # into a single PERSON node, which also reduces the edge count (Bug 3).
    for doc in documents:
        removed = deduplicate_cross_type(store, case_id=doc.doc_id)
        if removed:
            logger.info(f"[Pipeline] Cross-type dedup removed {len(removed)} duplicate(s) for {doc.doc_id}: {removed}")
            # Clean up doc_entity_map so removed entities don't create ghost edges
            for doc_id_key in list(doc_entity_map.keys()):
                doc_entity_map[doc_id_key] = [
                    eid for eid in doc_entity_map[doc_id_key] if eid not in removed
                ]

    all_resolved = store.all()
    for resolved in all_resolved:
        methods = entity_methods.get(resolved.entity_id, {"spacy"})
        score_entity(resolved, methods)

    # ---- Convert to validated schema ----
    entities: list[Entity] = [resolved_to_entity(r) for r in all_resolved]

    # ---- Build co-occurrence relationships ----
    relationships: list[Relationship] = build_relationships(entities, doc_entity_map, doc_type_map)

    return PipelineResult(
        entities=entities,
        relationships=relationships,
        extraction_log=extraction_log,
        total_documents=len(documents),
        total_entities=len(entities),
        total_relationships=len(relationships),
    )
