# M5 — RUN REPORT (FINAL)
### PS 26152 — AI-Powered Criminal Network Analysis System
### Module: Agentic AI / LangGraph Orchestrator (M5)

---

## Summary

Module M5 is fully implemented and all 38 tests pass. The LangGraph workflow
correctly orchestrates M1–M4 tool calls, maintains conversation state across
multiple turns, routes low-confidence results to human review, enforces
guardrails, and exposes a single `handle_investigator_request()` entry point
for M6 to call.

---

## Test Results

```
38 passed in 8.12s
```

### PRD §10 — Acceptance Criteria

| # | Criterion | Status |
|---|---|---|
| AC1 | Full workflow runs end-to-end against mocked M1–M4 tools with zero errors | **PASSED** — `TestWorkflowMocked` (6 tests) |
| AC2 | Full workflow runs end-to-end against real M1–M4 tools | **PARTIAL** — See "NOT YET INTEGRATED" below |
| AC3 | A deliberately low-confidence test case correctly routes to human review | **PASSED** — `test_low_confidence_routes_to_human_review` |
| AC4 | A 4-turn conversation correctly maintains context throughout | **PASSED** — `test_4_turn_conversation_context_preserved` and `test_realistic_investigation_session` (M5.10 demo) |
| AC5 | A deliberately failing mock tool call is caught and routed to human review | **PASSED** — `test_tool_error_routes_to_human_review` |
| AC6 | Tool-call count and timeout guardrails work correctly | **PASSED** — `test_tool_call_counter_resets_per_request`, `test_timeout_mock_triggers_human_review` |
| AC7 | M6 can call `handle_investigator_request()` as sole entry point | **PASSED** — all tests call only this function |

---

## M5.10 — Multi-Turn Demo Scenario

**Confirmed working.** The `test_realistic_investigation_session` test runs the exact
scenario a live judge would see:

1. **Turn 1 — Upload FIR103:** Ravi Kumar case with phone and account entities.
2. **Turn 2 — Follow-up:** "What phone numbers and accounts is Ravi Kumar connected to?"
3. **Turn 3 — Follow-up:** "Are there any suspicious financial transaction patterns?"
4. **Turn 4 — Follow-up:** "Please summarize all connections and priority flags found so far."

All four turns:
- Return the same `session_id`
- Maintain `current_case_id = "FIR103"` throughout
- Grow `conversation_history` (8 entries after 4 turns)
- Preserve `entities_in_context` across turns
- Return non-empty `response_text` with evidence citations

---

## Confidence Threshold

**Current default: 0.5**

Set in `M5/routing.py:HUMAN_REVIEW_THRESHOLD`. If confidence < 0.5, the response
is flagged `requires_human_review=True`. The team should review and tune this value
together before the demo — M4's mock confidence is 0.72 (passes), but real inference
on sparse evidence may score lower.

---

## Guardrails

| Guardrail | Setting | Location |
|---|---|---|
| Tool call cap | `MAX_TOOL_CALLS = 10` | `M5/tools.py` |
| Per-call timeout | `TOOL_TIMEOUT_SECONDS = 60` | `M5/tools.py` |

Both are tested and confirmed to route to human review on violation rather than crashing.

---

## NOT YET INTEGRATED

### Real vs Mocked at Time of Final Testing

| Module | Status | Notes |
|---|---|---|
| M1 (`run_pipeline`) | **MOCKED** | Real M1 is complete. Wire: set `use_mocks=False` and pass `source_dir` in `NewCaseUpload.source_dir` |
| M2 (`neighbors`, `search_by_name`) | **MOCKED** | Real M2 is complete. Wire: pass the real `CriminalGraph` object as `graph=` kwarg to `handle_investigator_request()` |
| M3 (`run_full_analysis`) | **MOCKED** | Real M3 is complete. Wire: pass `cdr_path=` and `txn_path=` to `handle_investigator_request()` along with the real graph |
| M4 (`generate_summary`) | **MOCKED** | Real M4 is complete. Wire: instantiate `RAGPipeline`, call `.load(graph)`, then pass as `m4_pipeline=` kwarg |

### How to Switch to Real Modules (No Workflow Changes Required)

```python
from M1.pipeline import run_pipeline
from M2.graph_builder import CriminalGraph
from M2.persistence import load_graph
from M3.pipeline import run_full_analysis
from M4.pipeline import RAGPipeline
from M4.config import DEFAULT_CONFIG
from M5 import handle_investigator_request
from M5.models import NewCaseUpload

# Load real resources once at startup
graph = load_graph("M2/output/graph.pkl")       # M2's persisted graph
pipeline = RAGPipeline(DEFAULT_CONFIG)
pipeline.load(graph=graph)

# Call with real modules — no changes to workflow code
response = handle_investigator_request(
    None,
    NewCaseUpload(case_id="FIR103", case_text="..."),
    use_mocks=False,          # ← only change needed
    graph=graph,
    m4_pipeline=pipeline,
    cdr_path="M1/data/cdrs/cdr.csv",
    txn_path="M1/data/transactions/transactions.csv",
)
```

### Function Signature Mismatches Found

None. All real module function signatures matched `backend.md` exactly as documented:

- `M1.pipeline.run_pipeline(source_dir)` ✓
- `M2.query.neighbors(graph, entity_id)` ✓
- `M2.query.search_by_name(graph, query, threshold)` ✓
- `M3.pipeline.run_full_analysis(graph, cdr_path, txn_path)` ✓
- `M4.pipeline.RAGPipeline.generate_summary(new_case_text, case_id, entity_ids, emit_files)` ✓

### What M6 Needs

M6 needs only two things:

1. Call `handle_investigator_request(session_id, request)` — documented in `M5/entry.py` and `M5/__init__.py`.
2. Read the returned `FinalResponse` dict (call `.to_dict()` for JSON serialization).

The `session_id` on the first call should be `None`; M5 will create one and return it in
`FinalResponse.session_id`. M6 stores it and passes it back on every subsequent turn.

---

## Files Delivered

| File | Purpose |
|---|---|
| `M5/__init__.py` | Public API surface |
| `M5/models.py` | `SessionState`, `FinalResponse`, `NewCaseUpload`, `FollowUpQuestion` |
| `M5/session.py` | Thread-safe in-memory session store |
| `M5/routing.py` | Confidence-based routing (threshold = 0.5) |
| `M5/tools.py` | Real M1–M4 tool wrappers with guardrail enforcement |
| `M5/graph.py` | LangGraph workflow definition (8 nodes, conditional routing) |
| `M5/entry.py` | `handle_investigator_request()` — single M6 entry point |
| `M5/mocks/mock_tools.py` | Schema-compliant mock M1–M4 implementations for testing |
| `M5/tests/test_m5.py` | 38-test suite covering all milestones and PRD acceptance criteria |
