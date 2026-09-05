# M5 — PRD: Agentic Workflow / LangGraph Module
### PS 26152 — AI-Powered Criminal Network Analysis System

---

## 1. Problem
The system's individual capabilities (entity extraction, graph queries, analytics, RAG summaries) need to be orchestrated into one coherent investigation workflow that an investigator can actually interact with — including asking follow-up questions — rather than being four disconnected tools they'd have to run manually in sequence.

## 2. Objective
Build a LangGraph-based orchestration layer that runs the fixed investigation workflow, maintains session context for follow-up questions, and routes low-confidence results to human review — calling M1–M4 as tools, without reimplementing their logic.

## 3. Scope

**In scope:**
- LangGraph workflow definition matching the fixed step order (extract → graph lookup → RAG → analysis → pattern → summary → confidence check)
- Tool wrapping of M1–M4's existing functions
- Session-scoped conversation state and follow-up question handling
- Confidence-based routing to a "human review" flag
- Guardrails against uncontrolled/looping agent behavior
- A single top-level entry point for M6 to call

**Out of scope:**
- Any actual entity extraction, graph construction, analytics computation, or summary generation logic — always delegated to M1–M4
- Persistent cross-session memory (database-backed) — in-memory, single-session state only, per backend.md
- Any UI

## 4. Functional Requirements

| ID | Requirement |
|---|---|
| FR1 | System shall execute the fixed workflow (extract → graph lookup → RAG → analysis → pattern → summary → confidence check) for a new case upload |
| FR2 | System shall wrap M1, M2, M3, and M4's functions as callable tools without duplicating their internal logic |
| FR3 | System shall maintain session state (conversation history, entities in context) across multiple turns of a single investigation session |
| FR4 | System shall detect follow-up questions within an active session and reuse existing context rather than re-running the full pipeline from scratch |
| FR5 | System shall route any response below a configurable confidence threshold to a "requires human review" flag, rather than presenting it as a confident answer |
| FR6 | System shall cap the number of tool calls per request and enforce a timeout, preventing uncontrolled looping |
| FR7 | System shall catch and gracefully handle any failure in a called tool (M1–M4), routing to human review with a clear message rather than crashing |
| FR8 | System shall expose a single top-level function for M6 to call for both new-case uploads and follow-up questions |
| FR9 | System shall be testable against mock M1–M4 implementations independently of whether real implementations are finished |
| FR10 | System shall produce output matching the shared `FINAL RESPONSE` schema |

## 5. Non-Functional Requirements

- **Resilience:** the system must degrade gracefully (route to human review) rather than fail hard when any upstream module errors
- **Statefulness:** session context must correctly persist across at least 4 conversational turns without loss or corruption
- **Predictability:** the workflow's step order must be deterministic for a given input — no unpredictable agent wandering between arbitrary tools
- **Testability:** the entire workflow must be runnable and testable against mocked M1–M4 tools, independent of other teams' completion status

## 6. Inputs
- New case document (from M6, via investigator upload)
- Follow-up question text (from M6, via investigator chat)
- M1, M2, M3, M4's function interfaces (real or mocked)

## 7. Outputs
- `final_response.json` — matching the shared `FINAL RESPONSE` schema, returned to M6

## 8. Interfaces / APIs

```python
def handle_investigator_request(session_id: str, request: NewCaseUpload | FollowUpQuestion) -> FinalResponse: ...
def build_workflow_graph() -> LangGraphWorkflow: ...
def get_session_state(session_id: str) -> SessionState: ...
def route_on_confidence(result, threshold: float = 0.5) -> RoutingDecision: ...
```

## 9. Data Structures
See shared contract in `backend.md` Section 4 — `SESSION STATE` and `FINAL RESPONSE` schemas are fixed and must not be altered without team sign-off.

## 10. Acceptance Criteria
- [ ] Full workflow runs end-to-end against mocked M1–M4 tools with zero errors
- [ ] Full workflow runs end-to-end against real M1–M4 tools (as they become available) with zero errors
- [ ] A deliberately low-confidence test case correctly routes to human review
- [ ] A 4-turn conversation (upload + 3 follow-ups) correctly maintains context throughout, verified by checking that follow-up answers reference entities from earlier in the session
- [ ] A deliberately failing mock tool call (simulated error) is caught and routed to human review instead of crashing the session
- [ ] Tool-call count and timeout guardrails correctly stop an artificially looping test case
- [ ] M6 can call `handle_investigator_request()` as the sole entry point for all interaction types

## 11. Failure Handling
- Any tool (M1–M4) throws an exception → catch, log, route to human review with a clear message
- Tool call exceeds timeout → abort that step, route to human review, do not hang the session
- Session ID not found (e.g., session expired/lost) → start a new session cleanly rather than erroring out unrecoverably

## 12. Performance Requirements
- Orchestration overhead itself (excluding the actual M1–M4 processing time) should be negligible — this module should not be the bottleneck; if M4's LLM generation is slow, that's M4's latency budget, not something M5 needs to optimize around beyond setting a sane timeout

## 13. Testing Requirements
- Unit tests: confidence routing logic (above/below threshold), tool-call guardrail enforcement, session state read/write
- Integration test: full workflow against mocked M1–M4, then again against real M1–M4 once available
- Edge cases: tool failure mid-workflow, follow-up question referencing an entity not in current session context, session state loss mid-conversation, deliberately triggering the tool-call cap
- Manual review: run a realistic multi-turn demo scenario manually and confirm the conversation feels coherent and context-aware, not just technically passing automated tests