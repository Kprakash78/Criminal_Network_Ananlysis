# M3 — Graph Intelligence Engineer — Backend Architecture
### PS 26152 — AI-Powered Criminal Network Analysis System

---

## 1. Role Summary

M3 turns M2's raw graph into **investigative intelligence** — this is where "who's the key player" and "what looks suspicious" actually get answered. Your output is the closest thing the system has to "insight," which means it's also the module most likely to be misread as making accusations if you're not careful with language and framing. Everything you output must be evidence-backed and phrased as a lead for investigator review, never a conclusion.

**You own:** centrality computation, community detection, rule-based suspicious-pattern detection, priority/anomaly scoring, evidence generation for every flag you raise.

**You do NOT own:** graph construction itself (M2), natural-language summarization (M4), agent orchestration (M5), UI (M6).

---

## 2. Internal Architecture

```
M2 GRAPH (NetworkX object / query layer)
        |
        v
  +----------------------+
  | 3a. CENTRALITY        |  degree, betweenness, closeness, PageRank
  |     ENGINE             |  per node — "how connected/influential"
  +----------------------+
        |
        v
  +----------------------+
  | 3b. COMMUNITY         |  connected components / community detection
  |     DETECTOR           |  (e.g. Louvain) — groups nodes into clusters
  +----------------------+
        |
        v
  +----------------------+
  | 3c. PATTERN RULES      |  rule-based checks over graph + underlying
  |     ENGINE              |  CDR/transaction timestamps:
  |                        |  - communication spikes
  |                        |  - unusual transaction frequency
  |                        |  - multiple suspects -> one account
  |                        |  - rapid money movement
  |                        |  - timing clustering around an incident
  +----------------------+
        |
        v
  +----------------------+
  | 3d. PRIORITY SCORER    |  combines centrality + pattern flags into
  |                        |  one interpretable investigation-priority
  |                        |  score per entity, with supporting evidence
  +----------------------+
        |
        v
   ANALYTICS STORE  ---> consumed by M4 (RAG context), M5 (confidence
                          routing), M6 (dashboard display)
```

---

## 3. Data Flow — Upstream / Downstream

**Upstream (what you receive):**
- From **M2**: the graph object (NetworkX) and its query functions (`neighbors`, `subgraph`, etc.)

**Downstream (what you produce, and who consumes it):**
- **M4 (RAG Engineer)** pulls your pattern flags/priority scores as grounding evidence when generating investigator summaries
- **M5 (LangGraph Agent)** uses your confidence/priority scores to decide whether to route a case to "human review" (low confidence) vs. present directly
- **M6 (Dashboard)** displays your centrality rankings, community clusters, and pattern flags visually

---

## 4. Core Data Structures (must match project-wide contract exactly)

```json
// PATTERN FLAG
{
  "entity_id": "P001",
  "degree_centrality": 0.72,
  "betweenness_centrality": 0.81,
  "community": 2,
  "flags": [
    "COMMUNICATION_SPIKE",
    "HIGH_TRANSACTION_FREQUENCY"
  ],
  "priority_score": 0.84,
  "evidence": [
    "47 calls within 2 hours",
    "Transaction activity increased 320%"
  ]
}
```

Pattern flag types (extendable, but this is the MVP set): `COMMUNICATION_SPIKE, HIGH_TRANSACTION_FREQUENCY, MULTI_SUSPECT_SHARED_ACCOUNT, RAPID_FUND_MOVEMENT, INCIDENT_TIMING_CLUSTER, DENSE_CLUSTER_MEMBERSHIP`

---

## 5. Tech Stack (fixed)

| Component | Choice | Why |
|---|---|---|
| Centrality algorithms | NetworkX built-ins (`degree_centrality`, `betweenness_centrality`, `closeness_centrality`, `pagerank`) | already available on M2's graph object, no extra dependency |
| Community detection | `networkx.algorithms.community` (e.g. `greedy_modularity_communities`) or `python-louvain` if finer control needed | standard, well-documented, fast enough at demo scale |
| Pattern rules | Plain Python + Pandas over timestamped CDR/transaction data joined with graph structure | rule-based is more explainable and more reliable than ML anomaly detection at this data scale — do NOT attempt to train an anomaly-detection model, there isn't enough data or time |
| Scoring | Simple weighted combination (e.g., `priority_score = w1*centrality + w2*pattern_flag_count`, weights tuned by eyeballing results on synthetic data) | must stay interpretable — a judge will ask "why is this score 0.84" and you need a plain-English answer |

---

## 6. Milestones (build in this order)

1. **M3.1 — Centrality computation**: run degree/betweenness/closeness/PageRank on M2's graph, store per-node results
2. **M3.2 — Community detection**: assign each node a community/cluster ID
3. **M3.3 — Pattern rule: communication spike**: flag entities with unusually high call frequency in a short window
4. **M3.4 — Pattern rule: transaction anomalies**: flag unusual transaction frequency/amount changes
5. **M3.5 — Pattern rule: shared account / rapid fund movement**: flag accounts linked to multiple suspects, or money moved through multiple hops quickly
6. **M3.6 — Pattern rule: incident timing clustering**: flag activity spikes around a known event/incident date (if the synthetic dataset includes incident dates)
7. **M3.7 — Priority scorer**: combine centrality + flags into one interpretable score per entity, always attached to evidence strings
8. **M3.8 — Output emitter**: validated JSON output matching the exact `PATTERN FLAG` schema
9. **M3.9 — Integration test with M2**: run against M2's real graph output, confirm no errors, sanity-check flagged entities against the synthetic dataset's intentional "planted" suspicious patterns
10. **M3.10 — Integration handoff to M4/M5/M6**: confirm output shape matches what those modules expect

---

## 7. Failure Handling Rules

- If the graph has isolated nodes (no edges) — still compute what's computable (centrality = 0), don't crash
- If timestamp data is missing/malformed for a pattern rule — skip that specific rule for that entity, don't fail the whole scoring pass
- Never output a flag without at least one evidence string — an unexplained flag is worse than no flag

---

## 8. What Is Explicitly NOT M3's Job (avoid scope creep)

- Building or updating the graph structure itself — that's M2
- Turning your scores/flags into natural-language investigator summaries — that's M4
- Deciding final human-review routing logic — that's M5 (you provide the confidence number, M5 decides what to do with it)
- **Never** label an entity "criminal," "guilty," or "involved in [crime]" — use "elevated priority," "pattern flagged," "requires investigator verification" language only, everywhere, including in code comments and variable names, since this habit prevents accidental overclaiming in the demo/dashboard later.