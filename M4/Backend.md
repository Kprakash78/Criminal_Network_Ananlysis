# M4 — Local LLM + RAG Engineer — Backend Architecture
### PS 26152 — AI-Powered Criminal Network Analysis System

---

## 1. Role Summary

M4 is where the system stops being "a bunch of scores and a graph" and starts being able to **explain itself in plain language, grounded in real evidence from M2's graph and M3's analytics**. This is also the module carrying the "local, self-hosted, no data leaves the system" requirement — a genuine, correct design decision for a law-enforcement-facing tool, not just a technical preference.

**You own:** local LLM hosting/inference, document embedding, vector retrieval, RAG pipeline (new case in → grounded context out → LLM summary), prompt design, evidence grounding.

**You do NOT own:** entity extraction (M1), graph construction (M2), centrality/pattern scoring (M3), agent workflow orchestration/memory (M5 — you generate a summary when asked, M5 decides when/how to ask and manages the conversation), UI (M6).

---

## 2. Internal Architecture

```
NEW CASE DOCUMENT (from investigator, via M6 upload)
        |
        v
  +----------------------+
  | 4a. CHUNKER            |  splits new case text into overlapping chunks
  +----------------------+
        |
        v
  +----------------------+
  | 4b. EMBEDDER           |  sentence-transformer model -> vector embeddings
  |                        |  (fully local, no external API)
  +----------------------+
        |
        v
  +----------------------+
  | 4c. VECTOR STORE       |  FAISS (or Chroma) index of embeddings for
  |                        |  all past case documents + new case chunks
  +----------------------+
        |
        v
  +----------------------+
  | 4d. RETRIEVER          |  finds top-k most relevant past case chunks
  |                        |  for the new case's content
  +----------------------+
        |
        v
  +----------------------+
  | 4e. GRAPH/ANALYTICS    |  pulls relevant M2 subgraph + M3 pattern
  |     EVIDENCE PULLER    |  flags for entities mentioned in the new case
  +----------------------+
        |
        v
  +----------------------+
  | 4f. PROMPT BUILDER     |  assembles: new case text + retrieved past
  |                        |  case evidence + graph/analytics evidence
  |                        |  into a structured, grounded prompt
  +----------------------+
        |
        v
  +----------------------+
  | 4g. LOCAL LLM           |  self-hosted 7-8B model, generates an
  |     INFERENCE           |  investigator-facing summary, citing evidence
  +----------------------+
        |
        v
   GROUNDED SUMMARY  ---> consumed by M5 (agent workflow), M6 (display)
```

---

## 3. Data Flow — Upstream / Downstream

**Upstream (what you receive):**
- From **M2**: graph query functions (`neighbors`, `subgraph`) for pulling relationship evidence
- From **M3**: pattern flags and priority scores for entities in the new case
- From **M6** (indirectly, via M5): the new case document itself, when an investigator uploads one

**Downstream (what you produce, and who consumes it):**
- **M5 (LangGraph Agent)** calls your RAG pipeline as a tool within its workflow, and may ask follow-up questions that also route through you
- **M6 (Dashboard)** displays your generated summary + the evidence citations you attach to it

---

## 4. Core Data Structures (must match project-wide contract exactly)

```json
// RAG RESULT
{
  "case_id": "FIR103",
  "source": "FIR102",
  "relevance_score": 0.91,
  "matched_entities": ["P001", "PHONE_X"],
  "evidence": "Same phone number appears in FIR102",
  "timestamp": null
}
```

```json
// GENERATED SUMMARY (M4's final output shape)
{
  "case_id": "FIR103",
  "summary_text": "Ravi Kumar appears in Case 103 through phone number X and vehicle Y. The graph also shows a communication relationship with Suresh Kumar, connected to two other cases. These are potential investigative links requiring verification.",
  "evidence_used": ["RAG_RESULT_1", "PATTERN_FLAG_P001", "GRAPH_EDGE_P001_P004"],
  "confidence": 0.78
}
```

---

## 5. Tech Stack (fixed — priority order matters)

| Component | Choice | Why |
|---|---|---|
| Local LLM | Llama 3 8B or Mistral 7B (whichever runs acceptably on available team hardware — confirm GPU access before committing) | open-weight, runs fully local, no data leaves the system — this is the core privacy requirement for a police-facing tool |
| Embeddings | `sentence-transformers` (e.g. `all-MiniLM-L6-v2`), fully local | no external API call needed for embedding either — full privacy chain |
| Vector store | FAISS | fast, local, no server needed; Chroma as a fallback if FAISS setup is troublesome |
| RAG orchestration | LangChain | handles chunking, embedding calls, retrieval — don't hand-roll this, LangChain's abstractions are well-tested for exactly this pipeline |
| Fine-tuning | **Not in MVP scope.** LoRA/QLoRA only as an explicit stretch goal, attempted only after the full RAG pipeline works end-to-end and only if GPU time genuinely remains | see Section 6 — RAG-first is a deliberate, non-negotiable priority order |

**Priority order, stated plainly: RAG working end-to-end > fine-tuning. Do not start fine-tuning until RAG is fully functional and demo-tested.**

---

## 6. Milestones (build in this order)

1. **M4.1 — Local model setup**: get chosen local LLM running inference on a simple test prompt, confirm hardware can handle it (this de-risks the biggest unknown first)
2. **M4.2 — Embedding pipeline**: chunk + embed a small set of sample documents, confirm embeddings generate correctly
3. **M4.3 — Vector store**: build FAISS index from embedded documents, confirm similarity search returns sensible results
4. **M4.4 — Retriever**: given a new case's text, retrieve top-k relevant past-case chunks
5. **M4.5 — Graph/analytics evidence puller**: given entities mentioned in a new case, pull relevant M2 subgraph + M3 pattern flags
6. **M4.6 — Prompt builder**: assemble retrieved evidence + graph/analytics evidence into a structured prompt template
7. **M4.7 — LLM summary generation**: feed the built prompt to the local LLM, generate an investigator-facing summary
8. **M4.8 — Evidence citation**: ensure every summary explicitly references which retrieved/graph evidence it's grounded in (no ungrounded claims)
9. **M4.9 — Output emitter**: validated JSON output matching the `GENERATED SUMMARY` schema
10. **M4.10 — Integration test with M2/M3**: run against real graph/analytics output; **M4.11 (stretch, only if time remains) — LoRA fine-tuning experiment**, clearly separated from the core deliverable

---

## 7. Failure Handling Rules

- If the local LLM produces an ungrounded claim (something not traceable to retrieved/graph evidence) — this is a real risk with any LLM; mitigate with strict prompt instructions ("only state what's supported by the evidence provided, say 'no strong evidence found' if uncertain") and treat any hallucination found during testing as a bug to fix in the prompt, not something to paper over
- If retrieval finds nothing relevant — say so explicitly in the summary ("no related historical cases found"), don't force a summary out of nothing
- If the local LLM is too slow/unavailable — fail gracefully with a clear error, don't hang the whole system

---

## 8. What Is Explicitly NOT M4's Job (avoid scope creep)

- Computing centrality or detecting patterns yourself — pull M3's existing output, don't recompute it
- Managing multi-turn conversation state/memory — that's M5's job; you generate one grounded summary per call, M5 handles the "conversation" layer around you
- Any UI — that's M6
- **Never** let the LLM state a conclusion of guilt — every summary must end with language like "requires investigator verification," and this instruction belongs in your system prompt itself, not just your own head.