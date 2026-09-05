# M3 — PRD: Graph Intelligence Module
### PS 26152 — AI-Powered Criminal Network Analysis System

---

## 1. Problem
A raw relationship graph (from M2) shows connections but doesn't tell an investigator who the key/influential individuals are, which clusters exist, or what activity looks statistically unusual — that interpretation currently happens manually and slowly.

## 2. Objective
Compute centrality, detect communities, run rule-based suspicious-pattern detection, and produce an interpretable, evidence-backed priority score per entity — all clearly framed as investigation leads, never conclusions.

## 3. Scope

**In scope:**
- Centrality computation (degree, betweenness, closeness, PageRank)
- Community/cluster detection
- Rule-based pattern detection: communication spikes, transaction anomalies, shared-account flags, rapid fund movement, incident-timing clustering
- Priority scoring combining centrality + pattern flags, always with evidence
- Output matching the shared `PATTERN FLAG` schema

**Out of scope:**
- Graph construction (M2)
- Any ML-based anomaly detection model (insufficient data/time — rule-based only, explicitly)
- Natural-language summary generation (M4)
- Deciding what action to take on a flagged entity (M5's routing logic)
- Any language implying guilt, criminality, or accusation

## 4. Functional Requirements

| ID | Requirement |
|---|---|
| FR1 | System shall compute degree, betweenness, closeness centrality and PageRank for every node in M2's graph |
| FR2 | System shall assign each node a community/cluster ID via a standard community-detection algorithm |
| FR3 | System shall detect communication-spike patterns based on call frequency within a configurable time window |
| FR4 | System shall detect transaction-frequency/amount anomalies relative to an entity's own baseline activity |
| FR5 | System shall detect accounts/entities connected to multiple distinct suspects |
| FR6 | System shall detect rapid multi-hop fund movement patterns within a configurable time window |
| FR7 | System shall detect activity clustering around a known incident date, where incident dates are available in the dataset |
| FR8 | System shall compute a single interpretable `priority_score` (0–1) per entity combining centrality and pattern-flag signals |
| FR9 | System shall attach at least one human-readable evidence string to every flag and every priority score |
| FR10 | System shall never use language implying guilt or criminal determination anywhere in output, code, or logs |

## 5. Non-Functional Requirements

- **Explainability:** every score/flag must be traceable to a specific, statable reason — no opaque ML black-box scoring
- **Performance:** full analytics pass should complete in well under 1 minute on the demo-scale dataset
- **Configurability:** thresholds for "spike," "unusual," "rapid" should be defined in one config location, not hardcoded inline, so they can be tuned during testing without code changes
- **Consistency:** rerunning analytics on the same unchanged graph must produce identical scores (deterministic, no randomness in scoring)

## 6. Inputs
- M2's graph object (NetworkX) and query functions
- Underlying timestamped CDR/transaction data (via M2's graph or passed through from M1, whichever the team agrees is cleaner — confirm with M1/M2)

## 7. Outputs
- `pattern_flags.json` — one `PATTERN FLAG` record per flagged entity, matching the shared schema exactly
- `centrality_scores.json` — full centrality/community results for every node (not just flagged ones — M6 needs this for the dashboard's ranking view)

## 8. Interfaces / APIs

```python
def compute_centrality(graph) -> dict[str, dict]: ...
def detect_communities(graph) -> dict[str, int]: ...
def run_pattern_rules(graph, cdr_data, transaction_data, incident_dates=None) -> list[PatternFlag]: ...
def compute_priority_score(entity_id, centrality_scores, flags) -> float: ...
def run_full_analysis(graph, cdr_data, transaction_data) -> AnalysisResult: ...
```

## 9. Data Structures
See shared contract in `backend.md` Section 4 — `PATTERN FLAG` schema and the fixed flag-type list are authoritative and must not be altered without team sign-off.

## 10. Acceptance Criteria
- [ ] Centrality scores compute successfully for every node in a real M2 graph, zero errors
- [ ] Community detection assigns every node to a cluster
- [ ] At least the "planted" suspicious patterns in the synthetic dataset (deliberately engineered by M1) are correctly flagged
- [ ] No flag exists without at least one evidence string
- [ ] No output anywhere uses guilt/criminal-implying language — spot-checked against a banned-word list (e.g., "criminal," "guilty," "perpetrator")
- [ ] Rerunning analysis on the same graph twice produces identical scores
- [ ] M4 can consume `pattern_flags.json` directly as RAG grounding context without reformatting

## 11. Failure Handling
- Node with zero edges → centrality = 0, no crash, no flags (nothing to flag)
- Missing/malformed timestamp on a call or transaction record → skip that record for time-based pattern rules, log a warning, continue
- Empty graph (e.g., M2 handoff failed) → fail loudly with a clear error, do not silently emit an empty-but-"successful" result

## 12. Performance Requirements
- Must scale to at least 500 nodes without algorithmic redesign (NetworkX centrality functions handle this fine at this scale)

## 13. Testing Requirements
- Unit tests: each pattern rule tested against a small synthetic graph with a known planted pattern and a known "clean" case (no false positive)
- Integration test: full analysis run against M2's real graph output
- Edge cases: isolated node, node with only one edge, entity with no timestamped records, empty graph
- Manual review: confirm the entities M3 flags as high-priority actually correspond to the suspicious patterns M1 deliberately planted in the synthetic dataset (this is your real correctness check, since there's no ground truth beyond what you designed in)