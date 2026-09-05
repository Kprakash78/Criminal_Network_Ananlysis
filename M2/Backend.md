# M2 — Knowledge Graph Engineer — Backend Architecture
### PS 26152 — AI-Powered Criminal Network Analysis System

---

## 1. Role Summary

M2 takes M1's extracted entities and turns them into a **queryable relationship graph** — the actual "network" in "Criminal Network Analysis." You decide how entities connect, how strongly, and with what evidence. M3 (graph analytics) and M4 (RAG) both query your graph directly, so your graph's correctness and query performance gate everything downstream of you.

**You own:** entity linking (deciding which co-occurring entities actually relate), relationship typing, graph construction, graph persistence, graph traversal/query functions.

**You do NOT own:** raw entity extraction from text (M1), centrality/pattern scoring (M3), LLM summarization (M4), agent orchestration (M5), UI (M6).

---

## 2. Internal Architecture

```
M1 OUTPUT
  entities.json / entities table
  relationships.json (basic co-occurrence)
        |
        v
  +--------------------+
  | 2a. GRAPH LOADER    |  reads M1 output, validates against schema
  +--------------------+
        |
        v
  +--------------------+
  | 2b. RELATIONSHIP    |  upgrades raw co-occurrence into typed edges:
  |     TYPER            |  CALLED, TRANSFERRED_MONEY_TO, APPEARS_IN_CASE,
  |                      |  OWNS, LOCATED_AT, ASSOCIATED_WITH, WORKS_FOR,
  |                      |  CONNECTED_TO — based on source record type
  |                      |  (CDR -> CALLED, transaction -> TRANSFERRED_MONEY_TO,
  |                      |  same-FIR co-occurrence -> ASSOCIATED_WITH, etc.)
  +--------------------+
        |
        v
  +--------------------+
  | 2c. GRAPH BUILDER   |  NetworkX MultiDiGraph: nodes = entities,
  |                      |  edges = typed relationships with weight/evidence
  +--------------------+
        |
        v
  +--------------------+
  | 2d. PERSISTENCE     |  serialize graph to SQLite (or Neo4j if time
  |                      |  permits) so it survives across pipeline runs
  +--------------------+
        |
        v
  +--------------------+
  | 2e. QUERY LAYER     |  functions: neighbors(entity), shortest_path(A,B),
  |                      |  subgraph(case_id), search_by_name(query)
  +--------------------+
        |
        v
   GRAPH STORE  ---> consumed by M3 (analytics), M4 (RAG context), M6 (viz)
```

---

## 3. Data Flow — Upstream / Downstream

**Upstream (what you receive):**
- From **M1**: `ENTITY` records and basic `RELATIONSHIP` (co-occurrence) records, in the fixed shared schema

**Downstream (what you produce, and who consumes it):**
- **M3 (Graph Intelligence Engineer)** queries your graph object directly to compute centrality, communities, and patterns
- **M4 (Local LLM + RAG Engineer)** calls your query layer (`neighbors()`, `subgraph()`) to pull graph evidence into RAG context for a given case/entity
- **M6 (Dashboard)** calls your query layer to render the visual graph and support search

---

## 4. Core Data Structures (must match project-wide contract exactly)

```json
// GRAPH EDGE (typed relationship — what M2 adds on top of M1's raw co-occurrence)
{
  "source": "P001",
  "target": "P002",
  "relationship": "CALLED",
  "timestamp": "2026-08-20T13:20:00",
  "source_record": "CDR_889",
  "confidence": 0.98,
  "weight": 1.0
}
```

Relationship types: `CALLED, TRANSFERRED_MONEY_TO, APPEARS_IN_CASE, OWNS, LOCATED_AT, ASSOCIATED_WITH, WORKS_FOR, CONNECTED_TO`

Node attributes carried over from M1's ENTITY schema: `entity_id, type, name, aliases, source_documents, confidence`

---

## 5. Tech Stack (fixed)

| Component | Choice | Why |
|---|---|---|
| Graph library | NetworkX (`MultiDiGraph`) | supports multiple typed edges between same nodes (e.g., a person can both CALL and TRANSFER_MONEY_TO another), pure Python, no server setup |
| Persistence | SQLite (adjacency-list tables: nodes, edges) | simple, no infra; re-hydrate into NetworkX graph on load |
| Optional upgrade | Neo4j (only if time permits and team wants native graph queries) | not required for MVP — NetworkX + SQLite is sufficient and lower-risk |
| Query interface | Plain Python functions, not a REST API (MVP runs in one process) | avoids unnecessary infra for a hackathon timeline |

---

## 6. Milestones (build in this order)

1. **M2.1 — Graph loader**: read M1's entity/relationship output, validate schema, load into memory
2. **M2.2 — Relationship typing rules**: map source-record type (CDR/transaction/FIR co-occurrence) to relationship type
3. **M2.3 — Graph construction**: build the NetworkX MultiDiGraph with typed, weighted, evidence-carrying edges
4. **M2.4 — Persistence layer**: serialize graph to SQLite, and reliably reload it (round-trip test)
5. **M2.5 — Basic query functions**: `neighbors(entity_id)`, `shortest_path(A, B)`, `subgraph(case_id)`
6. **M2.6 — Search function**: `search_by_name(query_string)` using fuzzy matching against node names/aliases
7. **M2.7 — Incremental update**: support adding a NEW case's entities/relationships into the existing graph without rebuilding from scratch (needed for the "new case file comes in" workflow)
8. **M2.8 — Integration test with M1**: run M1's real output through your loader, confirm zero schema errors
9. **M2.9 — Integration handoff to M3**: expose the graph object/query functions in the exact shape M3 expects, confirm M3 can call `neighbors()`/traverse without modification
10. **M2.10 — Integration handoff to M4/M6**: confirm `subgraph(case_id)` and `search_by_name()` return data M4 and M6 can consume directly

---

## 7. Failure Handling Rules

- If M1 emits an entity referencing a `source_document` that doesn't otherwise appear — still add the node, just with sparse metadata (don't fail the whole load)
- If a relationship references an entity_id not present in the entity list — log a warning, skip that single edge, continue building the rest of the graph
- Never let one malformed record from M1 crash the entire graph build

---

## 8. What Is Explicitly NOT M2's Job (avoid scope creep)

- Deciding who the "key players" are — that's centrality/analytics (M3)
- Generating natural-language explanations of the graph — that's M4
- Rendering the graph visually — that's M6 (you provide the data/query layer, not the visualization)

Stay focused on a correct, queryable, incrementally-updatable graph — that's the single most load-bearing piece of the whole system.