# M6 — PRD: Investigator Dashboard Module
### PS 26152 — AI-Powered Criminal Network Analysis System

---

## 1. Problem
Everything the other five modules compute — entities, graph structure, key players, pattern flags, grounded summaries — is inaccessible to an investigator (or a judge) unless it's presented through a usable, visual interface with search, upload, and export.

## 2. Objective
Build a working investigator-facing dashboard that lets a user upload a case, see it analyzed (summary + evidence + confidence), explore the relationship graph visually, search/filter entities, view key-player rankings, ask follow-up questions, and export results — all by calling M5's (and directly, M2's/M3's) existing functions.

## 3. Scope

**In scope:**
- Case upload UI
- Rendering of M5's `FINAL RESPONSE` (summary, evidence, confidence, human-review flag)
- Interactive graph visualization (subgraph/neighbors from M2)
- Entity/case search and date-range filtering
- Key-players ranking view (from M3)
- Follow-up question / conversational UI within a session
- CSV, JSON, and PDF export

**Out of scope:**
- Any computation — entity extraction, graph building, analytics, or summary generation (all delegated to M1–M5)
- User authentication/access control (explicitly out of scope for a hackathon MVP unless the team decides otherwise — note as future scope)
- Any language implying guilt/certainty anywhere in the UI

## 4. Functional Requirements

| ID | Requirement |
|---|---|
| FR1 | System shall provide a file upload interface for new case documents |
| FR2 | System shall call M5's `handle_investigator_request()` and render the returned summary, evidence, confidence score, and human-review flag |
| FR3 | System shall render an interactive graph visualization for a given case or entity, using M2's query layer |
| FR4 | System shall provide entity/case search with fuzzy matching (via M2's `search_by_name()`) |
| FR5 | System shall provide date-range filtering on displayed cases/relationships |
| FR6 | System shall display a ranked "key players" list using M3's centrality/priority scores |
| FR7 | System shall support follow-up questions within an active session, calling M5 again and displaying conversation history |
| FR8 | System shall export the currently displayed case/results as CSV, JSON, and PDF |
| FR9 | System shall display clear, non-technical error states when any backend call fails, never a raw stack trace |
| FR10 | System shall use only non-accusatory language in all UI labels, headers, and messages ("flagged," "potential connection," "requires verification" — never "criminal," "guilty," "confirmed") |

## 5. Non-Functional Requirements

- **Usability:** a first-time user (a judge, in a demo context) should be able to understand what they're looking at without narration — clear labels, sensible layout
- **Resilience:** the UI must never show a blank screen or raw error/stack trace to the user under any backend failure condition
- **Demo-readiness:** the full upload → analyze → explore → export flow must work reliably and repeatably, since this is what will actually be demonstrated live
- **Independence from backend timing:** the UI must be buildable and testable against mocks, not blocked on M1–M5's completion status

## 6. Inputs
- Case document uploads (from the investigator/user)
- Search/filter/follow-up question input (from the investigator/user)
- M5's `handle_investigator_request()` function
- M2's query functions (`subgraph`, `neighbors`, `search_by_name`)
- M3's centrality/priority score output

## 7. Outputs
- Rendered dashboard views (upload, summary, graph, search results, key-players ranking, conversation)
- Exported files: `.csv`, `.json`, `.pdf`

## 8. Interfaces / APIs

```python
def render_upload_view() -> None: ...
def render_result_view(final_response: FinalResponse) -> None: ...
def render_graph_view(subgraph_data) -> None: ...
def render_search_view(query: str) -> None: ...
def render_key_players_view(centrality_scores: dict) -> None: ...
def export_results(current_case_id: str, format: Literal["csv", "json", "pdf"]) -> str: ...
```

## 9. Data Structures
M6 consumes but does not define new shared schemas — see `FINAL RESPONSE` (M5's backend.md), graph structure (M2's backend.md), and `centrality_scores.json` (M3's backend.md) as the authoritative shapes this module renders.

## 10. Acceptance Criteria
- [ ] A full case upload → summary display → graph view → export flow works end-to-end with zero manual intervention
- [ ] Graph visualization correctly renders a real subgraph from M2 with distinguishable node/edge types
- [ ] Search returns and displays correct fuzzy-matched results
- [ ] Key-players view correctly reflects M3's actual ranking, not a placeholder
- [ ] A follow-up question in the same session displays with correct conversation history, not a reset state
- [ ] Export produces valid, openable CSV/JSON/PDF files with the correct current case's data
- [ ] A simulated backend failure (M5 call throws an error) shows a clear non-technical error message, not a crash or blank screen
- [ ] No UI text anywhere uses guilt/certainty language — spot-checked against a banned-word list

## 11. Failure Handling
- Backend call fails/times out → show calm error message, offer retry, never show a stack trace
- Empty search/graph result → explicit "no results found" state, not a blank panel
- Export fails → clear error message, no corrupt/empty file silently produced
- Upload of an unsupported/malformed file → clear validation message before attempting to process

## 12. Performance Requirements
- Dashboard should remain responsive during backend processing — show a loading state for any call expected to take more than ~1 second (particularly M4's LLM generation via M5)

## 13. Testing Requirements
- Unit tests: search/filter logic, export file generation (valid CSV/JSON/PDF structure)
- Integration test: full flow against mocked M5/M2/M3, then again against real modules once available
- Edge cases: empty upload, malformed file, search with no matches, backend timeout, very large case with many entities (does the graph view remain usable)
- Manual review: run the exact intended demo script (M6.9 in backend.md) live, end-to-end, at least twice before the actual presentation, ideally in front of a teammate playing "judge" and asking unexpected questions