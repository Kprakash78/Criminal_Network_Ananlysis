# M1 — NLP & Data Engineer — Backend Architecture
### PS 26152 — AI-Powered Criminal Network Analysis System

---

## 1. Role Summary

M1 owns the **entry point of the entire pipeline**: turning raw, messy, simulated crime data (FIRs, CDRs, transaction records) into clean, structured, confidence-scored entities that every other module (M2 graph, M3 analytics, M4 RAG) depends on. If M1's output is wrong or inconsistent, everything downstream is wrong. This module has zero tolerance for silent schema drift — every other teammate's code depends on your output shape staying exact.

**You own:** synthetic dataset generation, text preprocessing, NER, entity normalization, entity resolution, confidence scoring, structured output emission.

**You do NOT own:** graph construction (M2), centrality/pattern detection (M3), LLM/RAG (M4), agent orchestration (M5), dashboard (M6).

---

## 2. Internal Architecture

```
RAW SOURCES (simulated)
  FIR text files
  CDR (call detail record) CSVs
  Transaction record CSVs
        |
        v
  +----------------+
  | 2a. LOADER     |  reads raw files, normalizes encoding, basic cleaning
  +----------------+
        |
        v
  +----------------+
  | 2b. PREPROCESS |  sentence splitting, Hinglish/Hindi normalization,
  |                |  noise removal, tokenization
  +----------------+
        |
        v
  +----------------+
  | 2c. NER ENGINE |  spaCy pipeline + IndicBERT/HF model
  |                |  extracts: PERSON, PHONE, VEHICLE, LOCATION,
  |                |  ORGANIZATION, ACCOUNT, DATE
  +----------------+
        |
        v
  +----------------+
  | 2d. NORMALIZER |  regex validation (phone/vehicle formats),
  |                |  string cleanup, casing, alias collection
  +----------------+
        |
        v
  +----------------+
  | 2e. RESOLVER   |  fuzzy-matches new entities against existing
  |                |  entity store (dedupe "Ravi Kumar" vs "R. Kumar")
  +----------------+
        |
        v
  +----------------+
  | 2f. SCORER     |  assigns confidence score per entity based on
  |                |  extraction method + match quality
  +----------------+
        |
        v
   ENTITY STORE (JSON/SQLite) ---> consumed by M2 (graph) and M4 (RAG)
```

---

## 3. Data Flow — Upstream / Downstream

**Upstream (what you receive):** nothing — you are the pipeline's entry point. You generate or ingest raw simulated data yourself.

**Downstream (what you produce, and who consumes it):**
- **M2 (Knowledge Graph Engineer)** consumes your `ENTITY` and `RELATIONSHIP` records to build graph nodes/edges
- **M4 (Local LLM + RAG Engineer)** consumes your structured entities + source document text for embedding/retrieval
- **M6 (Dashboard)** indirectly consumes your entity metadata for search/filter UI

---

## 4. Core Data Structures (must match project-wide contract exactly)

```json
// ENTITY
{
  "entity_id": "P001",
  "type": "PERSON",
  "name": "Ravi Kumar",
  "aliases": ["Ravi K.", "R. Kumar"],
  "source_documents": ["FIR102", "FIR87"],
  "confidence": 0.94
}

// RELATIONSHIP (basic co-occurrence relationships M1 can also emit;
// deeper relationship typing is refined by M2)
{
  "source": "P001",
  "target": "P002",
  "relationship": "APPEARS_IN_SAME_DOCUMENT",
  "timestamp": "2026-08-20T13:20:00",
  "source_record": "FIR102",
  "confidence": 0.9
}
```

Entity types to support: `PERSON, PHONE, VEHICLE, LOCATION, ORGANIZATION, ACCOUNT, CASE, DATE`

---

## 5. Tech Stack (fixed — do not deviate without team discussion)

| Component | Choice | Why |
|---|---|---|
| Language | Python 3.11+ | team-wide standard |
| Data handling | Pandas | fast, standard, everyone can read it |
| Base NER | spaCy (`en_core_web_trf` or `en_core_web_sm` for speed) | mature, fast, good out-of-box entity types |
| Hinglish/Hindi NER | IndicBERT / Indic NLP models via Hugging Face Transformers | required — Indian FIRs are code-mixed, generic English NER will silently fail on Hindi terms |
| Regex validation | Python `re` | phone numbers, vehicle plate formats (India: `XX00XX0000`) |
| Fuzzy matching (entity resolution) | `rapidfuzz` (faster than `fuzzywuzzy`) | dedupe aliases/near-duplicate names |
| Data validation | `pydantic` | enforce the ENTITY/RELATIONSHIP schema so downstream never gets malformed JSON |
| Storage | JSON files during dev, SQLite for integration | simple, no server needed |

---

## 6. Milestones (build in this order — do not skip ahead)

1. **M1.1 — Synthetic dataset generator**: script producing realistic dummy FIR text, CDR CSV, transaction CSV, with deliberately overlapping entities across documents (so resolution/matching has something real to do)
2. **M1.2 — Loader + preprocessing**: read all three source types into a unified internal text representation
3. **M1.3 — NER extraction (English)**: get spaCy pipeline extracting PERSON/LOCATION/ORG reliably on your synthetic English FIRs
4. **M1.4 — NER extraction (Hinglish/Hindi)**: add IndicBERT model for code-mixed text, validate on a Hinglish subset of your dataset
5. **M1.5 — Regex extractors**: phone numbers, vehicle numbers, account numbers (rule-based, not ML — more reliable for structured formats)
6. **M1.6 — Normalizer**: standardize casing, formats, collect aliases per entity
7. **M1.7 — Resolver**: fuzzy-match against existing entity store, merge duplicates, assign persistent `entity_id`
8. **M1.8 — Confidence scorer**: combine extraction-method confidence + match-quality confidence into final score
9. **M1.9 — Output emitter**: validated JSON/SQLite output matching the exact shared schema
10. **M1.10 — Integration test with M2**: hand off a real batch of entities, confirm M2's graph builder can ingest without errors

---

## 7. Failure Handling Rules

- Never silently drop an entity — if confidence is very low, still emit it, but flag `"needs_review": true`
- Never crash the pipeline on one bad document — catch per-document errors, log them, continue processing the rest
- Never change the output schema without notifying M2/M4 — this breaks their code immediately

---

## 8. What Is Explicitly NOT M1's Job (avoid scope creep)

- Deciding whether two entities are "the same criminal network" — that's graph/analytics (M2/M3)
- Generating investigator-facing summaries — that's M4/M5
- Any UI — that's M6

Stay in your lane; a clean, reliable entity extraction pipeline is worth far more to the team's demo than a half-built graph feature bolted onto M1's code.