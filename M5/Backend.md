# M5 — Agentic AI / LangGraph Engineer — Backend Architecture
### PS 26152 — AI-Powered Criminal Network Analysis System

---

## 1. Role Summary

M5 is the **conductor**, not a performer — you don't extract entities, build graphs, compute scores, or generate summaries yourself. You orchestrate the existing modules (M1–M4) into one coherent investigation workflow, manage session state so investigators can ask follow-up questions, and decide when the system should defer to a human instead of answering confidently. Because you sit on top of everything else, you're also the module most exposed to other teams' delays — plan for that explicitly (see Section 6).

**You own:** LangGraph workflow definition, state management, tool-calling orchestration, conversation/session memory, confidence-based routing (including routing to human review).

**You do NOT own:** any of the actual extraction/graph/analytics/generation logic — you call M1–M4's functions as tools, you don't reimplement them.

---

## 2. Internal Architecture

```
INVESTIGATOR ACTION (via M6: upload case / ask question)
        |
        v
  +----------------------+
  |  LANGGRAPH STATE       |  holds: current case, conversation history,
  |  (session-scoped)      |  entities discussed, last retrieved evidence
  +----------------------+
        |
        v
   [START] --> ENTITY EXTRACTION (calls M1)
        |
        v
   GRAPH LOOKUP (calls M2)
        |
        v
   RAG RETRIEVAL (calls M4)
        |
        v
   GRAPH ANALYSIS (calls M3)
        |
        v
   PATTERN ANALYSIS (calls M3)
        |
        v
   GENERATE SUMMARY (calls M4)
        |
        v
   CONFIDENCE CHECK
        |
        +---- low confidence ---> HUMAN REVIEW FLAG (surfaced to M6, not blocked)
        |
        v
   [END] --> FINAL RESPONSE (returned to M6, added to session state)

FOLLOW-UP QUESTION (same session):
  re-enters the graph at GRAPH LOOKUP / RAG RETRIEVAL using existing
  session state (already-extracted entities, prior context) instead of
  re-running entity extraction from scratch every time
```

---

## 3. Data Flow — Upstream / Downstream

**Upstream (what you receive, all as callable tools/functions, not reimplemented):**
- M1's `run_pipeline()` — entity extraction from a new case
- M2's `neighbors()`, `subgraph()`, `search_by_name()` — graph queries
- M3's `run_full_analysis()` — pattern flags / priority scores
- M4's `retrieve_relevant_cases()`, `generate_summary()` — RAG + LLM summary

**Downstream (what you produce, and who consumes it):**
- **M6 (Dashboard)** calls your top-level `handle_investigator_request()` entry point for both new-case uploads and follow-up questions, and displays whatever you return (summary, evidence, human-review flag)

---

## 4. Core Data Structures

```json
// SESSION STATE (LangGraph state object — internal to M5, but shape matters for debugging/logging)
{
  "session_id": "SESS_001",
  "current_case_id": "FIR103",
  "conversation_history": [
    {"role": "investigator", "content": "How is Ravi Kumar connected to Case 103?"},
    {"role": "system", "content": "<generated summary text>"}
  ],
  "entities_in_context": ["P001", "P002"],
  "last_evidence": ["RAG_RESULT_1", "PATTERN_FLAG_P001"]
}

// FINAL RESPONSE (what M5 returns to M6)
{
  "session_id": "SESS_001",
  "response_text": "<investigator-facing text>",
  "evidence": [ /* from M4's GENERATED SUMMARY */ ],
  "confidence": 0.78,
  "requires_human_review": false
}
```

---

## 5. Tech Stack (fixed)

| Component | Choice | Why |
|---|---|---|
| Workflow orchestration | LangGraph | purpose-built for exactly this — explicit stateful graph of steps with conditional routing, better than a hand-rolled if/else chain for a workflow this shape |
| Tool wrapping | LangChain tool interface (wrapping M1–M4's existing functions) | standard pattern, keeps tool definitions clean and swappable |
| Session memory | LangGraph's built-in state persistence (in-memory dict for MVP; do not build a database-backed session store unless there's spare time) | in-memory is sufficient for a single demo session, avoids unnecessary infra |
| Confidence routing | Simple threshold check on M4's/M3's confidence scores (e.g., `if confidence < 0.5: flag_for_human_review`) | must stay simple and explainable — no need for a learned routing model |

---

## 6. Milestones (build in this order)

1. **M5.1 — Tool wrappers**: wrap M1–M4's functions as LangChain-compatible tools (start with MOCK versions of M1–M4 if their real code isn't ready yet — see Section 7)
2. **M5.2 — Basic linear graph**: build the LangGraph workflow with the fixed step order (extract → graph lookup → RAG → analysis → pattern → summary → confidence check), no conditional logic yet
3. **M5.3 — Confidence-based routing**: add the conditional edge that routes low-confidence results to a "human review" flag instead of silently returning a confident-sounding answer
4. **M5.4 — Session state**: implement state that persists across a single investigation session (entities discussed, conversation history)
5. **M5.5 — Follow-up question handling**: detect a follow-up question in the same session and re-enter the graph at the appropriate step, reusing existing context instead of re-extracting from scratch
6. **M5.6 — Top-level entry point**: build `handle_investigator_request()` — the single function M6 calls for both new-case uploads and follow-up questions
7. **M5.7 — Guardrails**: prevent uncontrolled agent behavior — cap the number of tool calls per request, add a timeout, ensure the agent can't loop indefinitely between steps
8. **M5.8 — Output emitter**: validated JSON output matching the `FINAL RESPONSE` schema
9. **M5.9 — Integration test with real M1–M4**: swap mock tool implementations for real ones as they become available, re-test the full workflow each time
10. **M5.10 — Multi-turn demo scenario test**: run a realistic 3-4 turn investigator conversation end-to-end (upload case → ask question → ask follow-up → ask a second follow-up), confirm context carries correctly throughout

---

## 7. Handling Cross-Team Dependency Risk (important — this is real for your role specifically)

You depend on M1, M2, M3, and M4 all having working, schema-compliant output — more than any other module, you're exposed to their delays. **Do not block your own progress waiting for them:**

- Build MOCK versions of every tool (M1's extraction, M2's graph queries, M3's analysis, M4's summary) that return small, realistic, schema-compliant fake data
- Build and test your entire LangGraph workflow against these mocks first
- Swap in each real module's function as it becomes available, one at a time, re-running your integration tests each time
- This means your workflow and orchestration logic is fully validated independent of whether teammates finish on time — a genuinely useful risk-management approach for this specific role

---

## 8. Failure Handling Rules

- If any tool call (to M1–M4) throws an error or times out — catch it, route to human review with a clear "system could not complete automated analysis" message, never let one failed tool call crash the whole session
- If the agent would need more than a fixed maximum number of tool calls to answer (guardrail from M5.7) — stop and return partial results with a note, rather than looping indefinitely
- If session state is somehow lost/corrupted mid-conversation — start a fresh session rather than returning a broken/partial state to the investigator

---

## 9. What Is Explicitly NOT M5's Job (avoid scope creep)

- Reimplementing any of M1–M4's logic "because it'd be easier to have it all in one place" — resist this, it breaks the team's division of labor and creates duplicate, drifting logic
- Deciding the actual content of pattern-detection rules or summary wording — you call M3/M4, you don't second-guess their output content
- Any UI — that's M6, you just provide the single entry point it calls