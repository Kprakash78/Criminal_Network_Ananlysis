# M6 — Full Stack / Investigator Dashboard Engineer — Backend Architecture
### PS 26152 — AI-Powered Criminal Network Analysis System

---

## 1. Role Summary

M6 is the **only module a judge or investigator actually sees.** Every other module's correctness is invisible unless you surface it clearly — a perfect graph algorithm that nobody can see or query is worthless in a demo. Your job is to make the whole system legible: upload a case, search entities, see the network visually, read the AI summary with its evidence, and export results.

**You own:** the dashboard UI, graph visualization, search/filter, file upload, result display, calling M5's entry point, export functionality, final end-to-end integration.

**You do NOT own:** any of the underlying extraction/graph/analytics/RAG/orchestration logic — you call M5's single entry point and render what comes back.

---

## 2. Internal Architecture

```
INVESTIGATOR (browser)
        |
        v
  +----------------------+
  | 6a. UPLOAD/QUERY UI     |  new case upload, entity/case search box,
  |                        |  date-range filter, follow-up question input
  +----------------------+
        |
        v
  +----------------------+
  | 6b. API/CALL LAYER      |  calls M5.handle_investigator_request()
  |                        |  (in-process function call for MVP, or a
  |                        |  thin FastAPI wrapper if using React)
  +----------------------+
        |
        v
  +----------------------+
  | 6c. RESULT RENDERER     |  displays FINAL RESPONSE: summary text,
  |                        |  evidence list, confidence, human-review flag
  +----------------------+
        |
        v
  +----------------------+
  | 6d. GRAPH VISUALIZER    |  calls M2's query layer directly (subgraph,
  |                        |  neighbors) to render the network graph
  |                        |  visually (nodes/edges, color-coded)
  +----------------------+
        |
        v
  +----------------------+
  | 6e. EXPORT MODULE       |  CSV / JSON / PDF report generation from
  |                        |  currently displayed results
  +----------------------+
```

---

## 3. Data Flow — Upstream / Downstream

**Upstream (what you receive, all as callable functions):**
- **M5**: `handle_investigator_request()` — the single entry point for uploads and questions
- **M2**: query functions (`subgraph`, `neighbors`, `search_by_name`) — called directly for visualization, since graph rendering needs the raw graph structure, not just M5's text summary
- **M3**: centrality/priority scores — for a "key players" ranking view on the dashboard

**Downstream:** nothing — you are the system's final output surface, consumed directly by the investigator/judge.

---

## 4. Core Data Structures

You primarily consume, not define, data structures — but your rendering layer should expect exactly:
- `FINAL RESPONSE` (from M5, Section 4 of M5's backend.md)
- Raw graph structure (nodes/edges, from M2)
- `centrality_scores.json` (from M3)

No new shared schema is introduced by M6 — this deliberately keeps you from becoming a second source of truth for data shapes other modules also need.

---

## 5. Tech Stack (fixed — MVP default, upgrade only if time allows)

| Component | Choice | Why |
|---|---|---|
| Dashboard framework | Streamlit (MVP default) | fastest to build a working, presentable dashboard with zero frontend boilerplate — ideal for hackathon timeframe |
| Graph visualization | PyVis (built on top of NetworkX, renders interactive HTML graphs) or `streamlit-agraph` | integrates directly with NetworkX objects from M2, minimal glue code |
| Advanced upgrade (only if ahead of schedule) | React + FastAPI + D3.js | more polished, but meaningfully more work — do not start this unless the Streamlit MVP is fully working and demo-ready first |
| Export | `pandas.to_csv()`, `json.dump()`, and a simple PDF via `reportlab` or `fpdf2` for the report format | standard, low-effort, satisfies the PS's explicit export requirement |

**Priority order, stated plainly: a working, demo-ready Streamlit MVP beats an unfinished React upgrade. Do not start the React version until Streamlit is fully functional and integrated.**

---

## 6. Milestones (build in this order)

1. **M6.1 — Basic Streamlit shell**: upload widget + a text output area, wired to a MOCK `handle_investigator_request()` (if M5 isn't ready yet)
2. **M6.2 — Result rendering**: display summary text, evidence list, confidence score, and human-review flag from `FINAL RESPONSE`
3. **M6.3 — Graph visualization**: render a subgraph (via M2's query layer, mocked if needed) as an interactive PyVis graph
4. **M6.4 — Search & filter**: entity/case search box, date-range filter, wired to M2's `search_by_name()`
5. **M6.5 — Key players view**: ranked list from M3's centrality/priority scores
6. **M6.6 — Follow-up question UI**: a simple chat-style input that re-calls M5 within the same session, displaying conversation history
7. **M6.7 — Export**: CSV, JSON, and PDF report export of the currently displayed case/results
8. **M6.8 — Integration test with real M5/M2/M3**: swap mocks for real modules as they become available, re-test each time
9. **M6.9 — Full demo walkthrough test**: run the entire intended demo script (upload → view graph → ask questions → export) end-to-end without manual intervention
10. **M6.10 — Polish pass**: layout, labeling, color-coding for confidence/priority (do this LAST, after everything works — a good-looking broken dashboard is worse than a plain working one)

---

## 7. Mock-First Strategy (same reasoning as M5 — you depend on everyone)

Build against small mock versions of M5's `handle_investigator_request()`, M2's query functions, and M3's scores first, so your UI work isn't blocked by other teammates' progress. Swap in real calls as they become available. This is especially important for you because **the dashboard is what the demo actually is** — it must be working and polished regardless of which backend pieces finish first, so budget your own time independent of others' schedules.

---

## 8. Failure Handling Rules

- If M5's call fails/times out — show a clear, calm error state in the UI ("Analysis could not complete — please try again or escalate for manual review"), never a raw stack trace or blank screen
- If the graph is empty for a search/case — show an explicit "no results found" state, not a blank visualization panel that looks broken
- If export generation fails — show an error, don't silently produce a corrupt/empty file

---

## 9. What Is Explicitly NOT M6's Job (avoid scope creep)

- Computing anything — you render what M2/M3/M5 give you, you don't calculate centrality, generate summaries, or run analysis yourself
- Deciding routing/confidence logic — that's M5's job, you just display the flag it returns
- **Language discipline applies here too**: never let UI labels/headers use words like "criminal," "guilty," or "confirmed" — use "flagged entity," "potential connection," "requires verification," consistent with every other module's language rules, since this is the one place investigators and judges will actually read.