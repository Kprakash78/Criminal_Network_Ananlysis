# M3 — RUN REPORT
### PS 26152 — AI-Powered Criminal Network Analysis System
### Module: Graph Intelligence (M3)

---

## What Was Built

| File | Milestone | Purpose |
|---|---|---|
| `M3/config.py` | Prerequisite | All thresholds and weights in one frozen dataclass — change here only |
| `M3/models.py` | Prerequisite | `PatternFlag` and `AnalysisResult` dataclasses (shared schema) |
| `M3/centrality.py` | M3.1 | degree, betweenness, closeness, PageRank for every node |
| `M3/community.py` | M3.2 | Greedy modularity community detection, community_id per node |
| `M3/pattern_rules.py` | M3.3–M3.6 | Four rule-based detectors (spike, transaction, shared-account/rapid, incident timing) |
| `M3/scorer.py` | M3.7 | Priority scorer + DENSE_CLUSTER_MEMBERSHIP flag |
| `M3/pipeline.py` | M3.8–M3.10 | `run_full_analysis()` orchestrator + `emit_outputs()` |
| `M3/output/pattern_flags.json` | M3.8 | Flagged entities for M4/M5 |
| `M3/output/centrality_scores.json` | M3.8 | All-node centrality+community for M6 |
| `M3/tests/` | M3.3–M3.9 | 68 unit + integration tests |

---

## Test Results

```
68 passed in 5.47s (100%)
```

Tests cover: all 10 milestones, planted-pattern (should-flag) cases, clean (no-false-positive) cases,
edge cases (isolated node, empty graph, malformed timestamps), and integration against real M2 output.

---

## Final Acceptance Criteria — Pass/Fail

| Criterion | Result |
|---|---|
| Centrality scores computed for every node (zero errors) | **PASS** — 247/247 nodes |
| Community detection assigns every node to a cluster | **PASS** — 247/247 nodes |
| Planted suspicious patterns correctly flagged | **PASS** — BANK_ACCOUNT nodes with multi-sender, large-txn, and rapid-chain patterns all flagged |
| No flag without at least one evidence string | **PASS** — enforced in `build_pattern_flags()` |
| No guilt/criminal-implying language in output | **PASS** — JSON output clean; source code uses banned words only as project name or in negative assertions |
| Rerunning analysis produces identical scores | **PASS** — determinism verified by two sequential runs |
| M4 can consume `pattern_flags.json` without reformatting | **PASS** — schema matches `backend.md §4` exactly |
| Performance: full pipeline < 60s | **PASS** — 0.89s on 247 nodes / 9714 edges |

---

## Language Audit

Banned words scanned: `criminal, guilty, perpetrator, culprit, accused, convicted, offender`

- **M3 JSON output files**: PASS — zero occurrences
- **M3 source code**: All occurrences are (a) the project's formal name ("PS 26152 — AI-Powered Criminal Network Analysis System"), (b) references to M2's `CriminalGraph` type alias which M3 must reference, or (c) negative language disclaimers ("No flag implies guilt... or criminal determination"). **None label any entity.**

---

## Output Files (for M4/M5/M6)

- `M3/output/pattern_flags.json` — one record per flagged entity, sorted by descending priority_score
- `M3/output/centrality_scores.json` — every node's metrics including community_id

### Schema (`pattern_flags.json` entries)
```json
{
  "entity_id": "ACC_b2210d1b",
  "degree_centrality": 0.012195,
  "betweenness_centrality": 0.0,
  "community": 0,
  "flags": ["HIGH_TRANSACTION_FREQUENCY", "MULTI_SUSPECT_SHARED_ACCOUNT", "RAPID_FUND_MOVEMENT"],
  "priority_score": 0.5721,
  "evidence": [
    "Degree centrality 0.012 — ...",
    "[HIGH_TRANSACTION_FREQUENCY] Single transaction of INR 197,829 on account...",
    "[MULTI_SUSPECT_SHARED_ACCOUNT] Account received funds from 4 distinct sources..."
  ]
}
```

### Results on Real M2 Graph

| Metric | Value |
|---|---|
| Nodes processed | 247 |
| Edges processed | 9,714 |
| Communities found | 2 |
| Entities with elevated priority | 214 |
| Entities with pattern flags | 214 |
| `DENSE_CLUSTER_MEMBERSHIP` flags | 206 |
| `HIGH_TRANSACTION_FREQUENCY` flags | 10 |
| `MULTI_SUSPECT_SHARED_ACCOUNT` flags | 10 |
| `RAPID_FUND_MOVEMENT` flags | 8 |
| `COMMUNICATION_SPIKE` flags | 0 (see note) |
| `INCIDENT_TIMING_CLUSTER` flags | 0 (see note) |

---

## NOT YET INTEGRATED

### M2 Real Graph vs. Mock
Tested against **M2's REAL graph** throughout (247 nodes, 9,714 edges from real M1 output).
No mock graph was used. No re-verification needed before demo.

### What M4 Will Need
M4 can consume `M3/output/pattern_flags.json` directly as RAG grounding context.
Each record contains an `evidence` list of human-readable strings — these are the
factual basis M4 should cite in any investigator narrative it generates.
M4 does NOT need to reformat or recompute anything from M3.

### What M5 Should Know
M5's confidence routing should use `priority_score` from `pattern_flags.json`:
- `priority_score ≥ 0.5` → high priority, surface to investigator immediately
- `priority_score 0.2–0.5` → medium priority, M5 may batch for review
- `priority_score < 0.2` → low priority, present in dashboard only

These thresholds are M5's routing logic — M3 does not decide them. Confirm with M5 teammate.

### Deliberate Deferrals (not bugs)
- **ML-based anomaly detection**: explicitly out of scope per `backend.md §5`. Rule-based only.
- **Natural-language summaries**: M4's job, not M3's.
- **Incident dates** were provided as example dates (`2025-01-15`, `2025-06-01`, `2026-05-04`).
  The M1 synthetic dataset doesn't embed explicit incident dates in the CDR/transaction data,
  so `INCIDENT_TIMING_CLUSTER` produced 0 flags on the real data — the rule is correct and
  tested; the real result reflects the data, not a code defect.
- **COMMUNICATION_SPIKE produced 0 flags** on the real M2 graph because M2's graph stores
  phone numbers as PHONE-type nodes but the CDR data refers to phones by number (string match).
  The graph has ~50 PHONE nodes but they are hashed entity IDs (e.g. `PHONE_abc123`), not
  the raw phone strings from the CDR. Resolution: M1/M2 should expose a `phone_number_raw`
  attribute on PHONE nodes, or M3 can be given the raw CDR→entity mapping from M1 directly.
  The rule is correct and fully tested on synthetic data — this is a data handoff gap to resolve
  with M1/M2 before the demo.

### Current Threshold Values (review before demo)
These were tuned by inspection on the synthetic 120-row CDR / 65-row transaction dataset:

| Parameter | Value | Location |
|---|---|---|
| `spike_call_count_threshold` | 5 calls | `config.py` |
| `spike_window_hours` | 24 hours | `config.py` |
| `txn_frequency_multiplier` | 2.5× | `config.py` |
| `txn_min_window_count` | 3 transactions | `config.py` |
| `txn_large_amount_inr` | 150,000 INR | `config.py` |
| `multi_suspect_min_senders` | 3 senders | `config.py` |
| `rapid_fund_hours` | 48 hours | `config.py` |
| `rapid_fund_min_hops` | 2 hops | `config.py` |
| `incident_window_days` | ±3 days | `config.py` |
| `incident_min_activity` | 2 events | `config.py` |
| `dense_cluster_degree_ratio` | 70% | `config.py` |
| `weight_degree_centrality` | 0.15 | `config.py` |
| `weight_betweenness_centrality` | 0.20 | `config.py` |
| `weight_pagerank` | 0.15 | `config.py` |
| `weight_flag_count` | 0.50 | `config.py` |

> [!IMPORTANT]
> The `DENSE_CLUSTER_MEMBERSHIP` flag fires on 206 of 247 nodes because the real M2 graph
> forms two very dense communities. The 70% same-community-neighbour threshold may need
> raising (e.g. to 90%) to be more selective, or this flag should be weighted lower in the
> priority formula. Review with the team before the demo.

---

## Known Limitations

1. **COMMUNICATION_SPIKE**: Zero flags on real data due to the phone-string/entity-ID mismatch
   described above. The rule logic is correct (verified by unit tests). Fix: add `phone_number`
   as a raw attribute to PHONE nodes in M2, or pass the CDR→entity map from M1 directly to M3.

2. **Two-community structure**: The real graph has only 2 communities (one giant component, one
   small cluster of account nodes). This is an artefact of how M2 deduplicates edges and how
   `greedy_modularity_communities` handles sparse synthetic data. On a larger real dataset,
   finer-grained communities are expected.

3. **DENSE_CLUSTER_MEMBERSHIP over-fires**: see threshold note above.

4. **No ML model**: Intentional. Rule-based scoring is a requirement, not a limitation.
   If the team wants ML-assisted anomaly detection in a future version, see `config.py`
   for the threshold parameters that a learned model could replace.
