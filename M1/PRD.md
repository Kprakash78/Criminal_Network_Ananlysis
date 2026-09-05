# M1 — PRD: NLP & Data Extraction Module
### PS 26152 — AI-Powered Criminal Network Analysis System

---

## 1. Problem
Investigators receive fragmented, unstructured crime data (FIRs, CDRs, transaction logs) with no automated way to extract structured entities (people, phones, vehicles, locations, organizations) or reconcile the same person/entity appearing under different names/formats across documents.

## 2. Objective
Build a pipeline that ingests simulated FIR/CDR/transaction data and outputs a clean, deduplicated, confidence-scored set of structured entities and basic co-occurrence relationships, ready for graph construction (M2) and retrieval (M4).

## 3. Scope

**In scope:**
- Synthetic data generation (FIR text, CDR CSV, transaction CSV)
- Text preprocessing (English + Hindi/Hinglish code-mixed)
- Named Entity Recognition for: PERSON, PHONE, VEHICLE, LOCATION, ORGANIZATION, ACCOUNT, DATE
- Regex-based structured field extraction (phone numbers, vehicle plates, account numbers)
- Entity normalization (casing, alias collection)
- Entity resolution (deduplication via fuzzy matching)
- Confidence scoring per entity
- Structured JSON/SQLite output matching the shared data contract

**Out of scope:**
- Relationship typing beyond basic co-occurrence (owned by M2)
- Graph analytics, centrality, pattern detection (M3)
- LLM-based summarization (M4)
- Any UI (M6)
- Real police data — simulated/dummy data only, for the entire prototype

## 4. Functional Requirements

| ID | Requirement |
|---|---|
| FR1 | System shall generate a synthetic dataset of ≥30 FIRs, ≥100 CDR entries, ≥50 transaction records, with intentional entity overlap across documents |
| FR2 | System shall extract PERSON, LOCATION, ORGANIZATION entities from English text with ≥80% precision on the synthetic test set |
| FR3 | System shall extract entities from Hinglish/Hindi code-mixed text using an Indic NLP model |
| FR4 | System shall extract PHONE and VEHICLE numbers via regex with format validation (Indian formats) |
| FR5 | System shall normalize entity names (casing, whitespace, common abbreviation expansion) |
| FR6 | System shall detect and merge duplicate entities referring to the same real-world entity (e.g. "Ravi Kumar" / "R. Kumar") using fuzzy string matching |
| FR7 | System shall assign each entity a confidence score between 0 and 1 |
| FR8 | System shall emit output strictly matching the shared `ENTITY` and `RELATIONSHIP` JSON schema |
| FR9 | System shall tag low-confidence entities with `"needs_review": true` rather than dropping them |
| FR10 | System shall continue processing remaining documents if one document fails to parse (no pipeline crash) |

## 5. Non-Functional Requirements

- **Reliability:** pipeline must run end-to-end on the full synthetic dataset without manual intervention
- **Latency:** processing the full synthetic dataset should complete in under 2 minutes on a standard laptop (no GPU dependency for this module specifically — NER models should run on CPU acceptably at this data scale)
- **Reproducibility:** same input must always produce the same entity IDs (deterministic ID assignment, not random)
- **Auditability:** every entity must retain a reference to its source document(s)

## 6. Inputs
- Raw FIR text files (`.txt`)
- CDR records (`.csv`: caller, callee, timestamp, duration)
- Transaction records (`.csv`: source account, target account, amount, timestamp)

## 7. Outputs
- `entities.json` / `entities` SQLite table — validated against the shared ENTITY schema
- `relationships.json` / `relationships` SQLite table — basic co-occurrence relationships
- `extraction_log.json` — per-document processing log (success/failure, entities found, errors)

## 8. Interfaces / APIs

Expose a simple Python function-level interface for other modules to call directly (no need for a network API at MVP stage, since everything runs in one local process):

```python
def extract_entities(document_path: str, doc_type: str) -> list[Entity]: ...
def resolve_entities(new_entities: list[Entity], existing_store: EntityStore) -> list[Entity]: ...
def run_pipeline(source_dir: str) -> PipelineResult: ...
```

## 9. Data Structures
See shared contract in `backend.md` Section 4 — ENTITY and RELATIONSHIP schemas are fixed and must not be altered without team sign-off.

## 10. Acceptance Criteria
- [ ] Full synthetic dataset processes end-to-end with zero unhandled crashes
- [ ] At least 3 deliberately duplicated entities (different name formats) are correctly merged
- [ ] At least 5 Hinglish/Hindi sentences correctly yield extracted entities
- [ ] Output JSON validates against the pydantic schema with zero validation errors
- [ ] M2 can successfully ingest the output without any manual reformatting

## 11. Failure Handling
- Malformed source document → log error, skip document, continue pipeline
- NER model returns nothing for a document → still emit an empty-but-valid entity list for that document, flagged in the log
- Regex extraction conflicts with NER extraction (e.g., NER tags a phone number as PERSON) → regex-validated structured fields (phone/vehicle/account) always take precedence over NER for those types

## 12. Performance Requirements
- CPU-only execution acceptable (no GPU required for this module)
- Should scale to at least 200 documents without redesign (even though demo dataset is smaller)

## 13. Testing Requirements
- Unit tests: regex extractors (valid/invalid phone/vehicle formats), normalizer (casing/alias merge logic)
- Integration test: full pipeline run on synthetic dataset, output validated against schema
- Edge cases: empty document, document with no valid entities, document with only Hindi text, duplicate entity with slightly different spelling
- Manual review: spot-check 10 random extracted entities against source text for correctness