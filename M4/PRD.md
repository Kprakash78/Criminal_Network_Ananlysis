# M4 — PRD: Local LLM + RAG Module
### PS 26152 — AI-Powered Criminal Network Analysis System

---

## 1. Problem
Raw graph structure and analytics scores (from M2/M3) are not directly readable by an investigator under time pressure — they need a plain-language, evidence-grounded explanation of how a new case connects to existing data, generated without sending sensitive case data to any external service.

## 2. Objective
Build a fully local RAG pipeline: given a new case document, retrieve relevant historical case evidence and graph/analytics evidence, and generate a grounded, evidence-cited investigator summary using a self-hosted LLM — no external API calls anywhere in the pipeline.

## 3. Scope

**In scope:**
- Local LLM hosting and inference
- Document chunking and local embedding generation
- Local vector store (FAISS) and retrieval
- Pulling relevant graph/analytics evidence from M2/M3 for entities in a new case
- Prompt construction that grounds the LLM strictly in retrieved evidence
- Grounded summary generation with explicit evidence citations
- Output matching the shared `RAG RESULT` and `GENERATED SUMMARY` schemas

**Out of scope:**
- Fine-tuning the LLM (stretch goal only, after RAG is fully working — see backend.md Section 6 priority order)
- Multi-turn conversation memory / session state (M5's job)
- Any recomputation of centrality/pattern flags (consume M3's output as-is)
- Any external API calls of any kind — the entire pipeline must run on local infrastructure, this is a hard requirement given the target users are law enforcement

## 4. Functional Requirements

| ID | Requirement |
|---|---|
| FR1 | System shall run LLM inference fully locally, with zero network calls to external LLM providers |
| FR2 | System shall run embedding generation fully locally, with zero calls to external embedding APIs |
| FR3 | System shall chunk and embed new case documents on upload |
| FR4 | System shall retrieve the top-k most relevant historical case chunks for a new case via vector similarity search |
| FR5 | System shall pull relevant M2 graph evidence (relationships) and M3 pattern flags for entities mentioned in the new case |
| FR6 | System shall generate a natural-language summary grounded strictly in retrieved and graph/analytics evidence |
| FR7 | System shall attach explicit evidence references to every generated summary |
| FR8 | System shall explicitly state "no strong evidence found" rather than fabricating a connection when retrieval/graph evidence is weak or absent |
| FR9 | System shall never generate language stating or implying guilt, criminal determination, or certainty of identity — only "potential connection," "requires verification," equivalent |
| FR10 | System shall produce output matching the shared `GENERATED SUMMARY` schema exactly |

## 5. Non-Functional Requirements

- **Privacy:** the entire pipeline (embedding + retrieval + generation) must be runnable with zero internet access, since this is the module's core justification
- **Groundedness:** summaries must be checkable against their cited evidence — a human should be able to verify every claim in the summary against the evidence list
- **Latency:** a single case summary generation should complete in a reasonable demo timeframe (target: under 30 seconds on available hardware — confirm this is realistic on your team's actual hardware early, per M4.1)
- **Determinism (as much as possible):** use a low temperature setting for generation to reduce variability between runs of the same input, since reproducibility matters for a demo and for trust in an investigative tool

## 6. Inputs
- New case document text (from M6 upload, via M5)
- Historical case documents (from M1's synthetic dataset, embedded ahead of time)
- M2 graph query functions
- M3 pattern flags / priority scores

## 7. Outputs
- `rag_results.json` — retrieved evidence matches, matching the shared `RAG RESULT` schema
- `generated_summary.json` — the final grounded summary, matching the shared `GENERATED SUMMARY` schema

## 8. Interfaces / APIs

```python
def embed_document(text: str) -> list[float]: ...
def build_vector_index(documents: list[str]) -> VectorIndex: ...
def retrieve_relevant_cases(new_case_text: str, index: VectorIndex, top_k: int = 5) -> list[RagResult]: ...
def pull_graph_evidence(entity_ids: list[str], graph) -> dict: ...
def generate_summary(new_case_text: str, rag_results: list[RagResult], graph_evidence: dict, pattern_flags: list) -> GeneratedSummary: ...
```

## 9. Data Structures
See shared contract in `backend.md` Section 4 — `RAG RESULT` and `GENERATED SUMMARY` schemas are fixed and must not be altered without team sign-off.

## 10. Acceptance Criteria
- [ ] LLM inference and embedding generation both run with network access disabled, and still work (proves the "fully local" requirement is actually met, not just assumed)
- [ ] Retrieval correctly surfaces the deliberately-planted related historical case for at least 3 test new-case documents
- [ ] Every generated summary includes at least one explicit evidence citation
- [ ] When no relevant evidence exists, the summary explicitly says so rather than inventing a connection
- [ ] No generated summary anywhere uses guilt/certainty language — spot-checked against a banned-word list
- [ ] Generation completes within the target latency on the team's actual hardware
- [ ] M5 can call `generate_summary()` as a tool and receive output matching the exact schema

## 11. Failure Handling
- Local LLM fails to load / out of memory → clear error message, do not silently fall back to a smaller model without flagging it
- Retrieval returns zero results → summary explicitly states this, does not fabricate
- Malformed new case document → return a clear error, don't attempt to generate a summary from unparseable input

## 12. Performance Requirements
- Must run on whatever GPU/CPU hardware the team actually has confirmed access to — **this must be validated in milestone M4.1 before any other work proceeds**, since it's the single biggest risk to this entire module

## 13. Testing Requirements
- Unit tests: chunking logic, embedding shape/dimension consistency, evidence-citation formatting
- Integration test: full pipeline run — new case in, summary out — against real M2/M3 output
- Edge cases: new case with no matching historical data, new case mentioning an entity not in the graph at all, empty/malformed document
- Manual review: read 5 generated summaries end-to-end and manually verify every claim in each is actually supported by its cited evidence — this is the real correctness check for a RAG system, automated tests alone won't catch hallucination