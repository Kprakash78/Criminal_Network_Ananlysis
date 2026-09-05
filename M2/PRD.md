# M2 — PRD: Knowledge Graph Module
### PS 26152 — AI-Powered Criminal Network Analysis System

---

## 1. Problem
Extracted entities from M1 are just isolated data points — investigators need to see how entities relate to each other (calls, money transfers, shared cases) as a connected structure, and the system needs a queryable representation of those relationships for downstream analytics and retrieval.

## 2. Objective
Build a typed, persistent, incrementally-updatable relationship graph from M1's entity/relationship output, with a query layer that M3, M4, and M6 can all rely on.

## 3. Scope

**In scope:**
- Loading and validating M1's entity/relationship output
- Typing relationships based on source-record type (CALLED, TRANSFERRED_MONEY_TO, APPEARS_IN_CASE, OWNS, LOCATED_AT, ASSOCIATED_WITH, WORKS_FOR, CONNECTED_TO)
- Graph construction (NetworkX MultiDiGraph)
- Persistence to SQLite with reliable reload
- Query functions: neighbor lookup, shortest path, subgraph extraction, name search
- Incremental graph updates when a new case is added

**Out of scope:**
- Centrality, community detection, anomaly scoring (M3)
- Any LLM-based reasoning or summarization (M4)
- Graph visualization/rendering (M6 — you provide data, not pixels)
- Determining if a connection implies guilt (never — evidence-linking only)

## 4. Functional Requirements

| ID | Requirement |
|---|---|
| FR1 | System shall load and validate M1's entity/relationship JSON output without schema errors |
| FR2 | System shall assign a relationship type to every edge based on documented source-record-type rules |
| FR3 | System shall construct a NetworkX MultiDiGraph supporting multiple distinct typed edges between the same node pair |
| FR4 | System shall persist the graph to SQLite and reload it identically (round-trip integrity) |
| FR5 | System shall provide `neighbors(entity_id)` returning all directly connected entities with relationship type and evidence |
| FR6 | System shall provide `shortest_path(entity_a, entity_b)` returning the connecting path, or a clear "no path found" result |
| FR7 | System shall provide `subgraph(case_id)` returning all entities/relationships tied to a given case |
| FR8 | System shall provide `search_by_name(query)` using fuzzy matching over node names and aliases |
| FR9 | System shall support incremental insertion of a new case's entities/relationships into an existing graph without a full rebuild |
| FR10 | System shall retain source-record evidence (which FIR/CDR/transaction) on every edge for auditability |

## 5. Non-Functional Requirements

- **Reliability:** graph load/persist round-trip must be lossless
- **Performance:** query functions should return in well under 1 second on the demo-scale dataset (tens to low hundreds of nodes)
- **Auditability:** every edge must be traceable to its source document/record
- **Extensibility:** relationship-typing rules should be defined in one clearly separated place (e.g., a config/mapping dict), not scattered across the codebase, so new relationship types can be added easily

## 6. Inputs
- `entities.json` / `entities` table (from M1)
- `relationships.json` (basic co-occurrence, from M1)

## 7. Outputs
- Persisted graph (SQLite: `nodes` table, `edges` table)
- In-memory NetworkX graph object for other modules to query within the same process
- Query results in a documented, stable JSON shape for M3/M4/M6 to consume

## 8. Interfaces / APIs

```python
def load_graph_from_m1_output(entities_path: str, relationships_path: str) -> Graph: ...
def persist_graph(graph: Graph, db_path: str) -> None: ...
def load_persisted_graph(db_path: str) -> Graph: ...
def neighbors(graph: Graph, entity_id: str) -> list[dict]: ...
def shortest_path(graph: Graph, source_id: str, target_id: str) -> list[str] | None: ...
def subgraph(graph: Graph, case_id: str) -> Graph: ...
def search_by_name(graph: Graph, query: str, threshold: float = 0.8) -> list[dict]: ...
def add_case_incrementally(graph: Graph, new_entities: list, new_relationships: list) -> Graph: ...
```

## 9. Data Structures
See shared contract in `backend.md` Section 4 — GRAPH EDGE schema, relationship type list, and node attribute set are fixed and must not be altered without team sign-off.

## 10. Acceptance Criteria
- [ ] Full graph builds from M1's real output with zero unhandled errors
- [ ] Graph persists to SQLite and reloads with identical node/edge count and attributes
- [ ] `shortest_path()` correctly finds a known multi-hop connection in the synthetic dataset
- [ ] `subgraph(case_id)` returns exactly the entities/relationships tied to that case, nothing extra
- [ ] `search_by_name("Ravi")` correctly returns "Ravi Kumar" even with a partial/fuzzy query
- [ ] Incremental update correctly adds a new case's data without duplicating existing nodes
- [ ] M3 can call `neighbors()` and traverse the graph object without any modification to M2's code

## 11. Failure Handling
- Entity referenced in a relationship but missing from entity list → log warning, skip that edge, continue
- Duplicate relationship (same source/target/type/source_record) submitted twice → deduplicate, don't create parallel identical edges
- Corrupt/missing SQLite file on load → clear error message, do not silently return an empty graph without warning

## 12. Performance Requirements
- Must handle at least 500 nodes / 2000 edges without redesign, even though the demo dataset will be smaller
- Query functions should not require a full graph rebuild on every call

## 13. Testing Requirements
- Unit tests: relationship-typing rules (correct type assigned for each source-record type), fuzzy search matching
- Integration test: load real M1 output → build graph → persist → reload → verify identical structure
- Edge cases: entity with no relationships (isolated node), relationship referencing a missing entity, duplicate relationship records, empty input file
- Manual review: visually sanity-check one small subgraph against the source FIRs to confirm the connections make sense