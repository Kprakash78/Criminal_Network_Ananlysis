# README_FEATURES.md
# PS 26152 — AI-Powered Criminal Network Analysis System

# New Analytical Features (v3.0)

This document describes four new offline analytical features added to the
Criminal Network Analysis System. All features are CPU-only, fully offline,
and use safe investigative language — no guilt determinations are made.

---

## Quick Start — Three Commands

```bash
# 1. Clone & switch to the feature branch
git clone https://github.com/keshavmittal1907-alt/Criminal_Network_Ananlysis.git
cd Criminal_Network_Ananlysis
git checkout feature/event-causality-auto-brief-ghostnode

# 2. Run all four new features (creates demo data + cache if missing)
bash scripts/run_features.sh

# 3. Run the new pytest suite
pytest -q tests/test_event_chains.py tests/test_auto_brief.py tests/test_ghost_node.py
```

Key outputs appear in `demo_cache/feature_outputs/`:

| File | Description |
|------|-------------|
| `event_chains.json` | Ordered event-causality chains per case |
| `ghost_suggestions.json` | Possible unobserved intermediary suggestions |
| `motif_flags.json` | Temporal anomaly flags per phone |
| `brief_case_A.pdf` | One-page tactical investigation brief |
| `manifest_case_A.json` | SHA-256 + HMAC signature for the brief |

---

## Feature 1 — Event-Causality Chain Builder

**File:** [`M3_feature/event_causality.py`](M3_feature/event_causality.py)

Reads CDR and transaction CSV files and links events into ordered chains
using a sliding 30-minute time-window heuristic. Events are connected when
they share an actor or target within the window. Marks "pivot events" where
the action type shifts (call → transfer) and scores them by:
- Temporal proximity to next event
- Actor degree centrality
- Action-type change

```bash
python -m M3_feature.event_causality build case_A
python -m M3_feature.event_causality build_all
```

**Output format (per chain):**
```json
{
  "chain_id": "case_A_c0",
  "n_steps": 7,
  "steps": [{"t": "2024-03-01T19:50:00", "actor": "9123456789",
              "action": "call", "target": "9654321098",
              "source": "demo_dataset/cdr_files/case_A_cdr.csv", "line": 2}],
  "pivot": {"step_index": 2, "score": 0.82},
  "note": "Requires investigator verification."
}
```

---

## Feature 2 — Auto-Brief Generator

**File:** [`M6_feature/auto_brief.py`](M6_feature/auto_brief.py)

Generates a one-page PDF tactical brief using a minimal pure-Python PDF
writer (no reportlab or fpdf required). The brief includes:
- **Top-N elevated-priority candidates** with risk scores and provenance
- **Ordered event timeline** (up to 8 bullets) from event chains
- **Temporal anomaly highlights** from the motif flagger
- **Template-based investigator next steps**
- Mandatory disclaimer (investigator verification required)
- HMAC-SHA256 signed `manifest_{case_id}.json`

```bash
python -m M6_feature.auto_brief export case_A
python -m M6_feature.auto_brief export case_A out/brief_case_A.pdf
```

**Safe language examples:**
- ✅ "elevated-priority candidate — requires investigator verification"
- ✅ "This pattern requires investigator verification."
- ❌ Never: "guilty", "criminal", "perpetrator"

---

## Feature 3 — Ghost-Node Finder

**File:** [`M3_feature/ghost_node.py`](M3_feature/ghost_node.py)

Applies three graph heuristics to suggest possible unobserved intermediaries
between network clusters:

| Heuristic | Formula | What it captures |
|-----------|---------|-----------------|
| Common Neighbours (CN) | `|N(u) ∩ N(v)|` | Shared mutual contacts |
| Adamic-Adar (AA) | `Σ 1/log(deg(w))` for w ∈ CN | Rare common contacts score higher |
| Jaccard | `|N(u) ∩ N(v)| / |N(u) ∪ N(v)|` | Relative neighbourhood overlap |

Composite score = `0.40 × CN_norm + 0.40 × AA_norm + 0.20 × Jaccard`.
Suggestions above threshold (0.05) are emitted with cross-cluster flags
and evidence pointers (CDR / transaction file + line).

```bash
python -m M3_feature.ghost_node suggest case_A
python -m M3_feature.ghost_node suggest_all
```

---

## Feature 4 — Temporal Motif Flagger

**File:** [`M3_feature/motif_flagger.py`](M3_feature/motif_flagger.py)

Rule-based detection of anomalous off-hour call patterns per phone number.

- **Off-hour window:** 23:00–04:59 (configurable)
- **Baseline:** average calls-per-hour during 05:00–22:59
- **Flag if:** off-hour rate > 1.5× baseline (configurable)

```bash
python -m M3_feature.motif_flagger flag case_A
python -m M3_feature.motif_flagger flag_all
```

**Output per flag:**
```json
{
  "phone": "9432109876", "name": "Fatima Begum",
  "off_hour_calls": 5, "multiplier_observed": 3.2,
  "explanation": "… requires investigator verification.",
  "evidence": {"cdr_file": "demo_dataset/.../case_A_cdr.csv", "flagged_rows": [4, 7]}
}
```

---

## Streamlit UI Cards

**File:** [`M6_feature/ui_cards.py`](M6_feature/ui_cards.py)

Three UI cards for the existing Streamlit dashboard:

```python
from M6_feature.ui_cards import (
    render_chains_card,
    render_ghost_card,
    render_motif_card,
    render_brief_card,
)
render_chains_card("case_A")  # Show event chains
render_ghost_card("case_A")   # Show ghost suggestions
render_brief_card("case_A")   # Download PDF brief
```

Run standalone:
```bash
streamlit run M6_feature/ui_cards.py
```

---

## Acceptance Criteria

| Check | How to verify |
|-------|--------------|
| Branch exists | `git branch --list feature/event-causality-auto-brief-ghostnode` |
| Features run offline | `bash scripts/run_features.sh` (no network) |
| Outputs written | `ls demo_cache/feature_outputs/` |
| Brief PDF valid | `file demo_cache/feature_outputs/brief_case_A.pdf` |
| Manifest present | `cat demo_cache/feature_outputs/manifest_case_A.json` |
| Tests pass | `pytest -q tests/test_event_chains.py tests/test_auto_brief.py tests/test_ghost_node.py` |
| Safe language | No "guilty"/"perpetrator" anywhere in outputs |

---

## Safe Language Reference

All UI copy and output text must follow these conventions:

| Instead of… | Use… |
|-------------|------|
| "suspect is guilty" | "elevated-priority candidate — requires investigator verification" |
| "criminal network" | "communication/transaction network pattern" |
| "proves involvement" | "indicates a pattern that warrants further investigation" |
| "the accused" | "the identified candidate" |

---

## Dependencies

No new pip packages are required. All features use:
- `csv`, `json`, `hashlib`, `hmac`, `math`, `textwrap` — Python stdlib
- `numpy` — already in `requirements.txt`
- `pytest` — already in `requirements.txt`

> **Note:** The auto-brief PDF writer is a minimal pure-Python implementation.
> For production use, replace with reportlab or fpdf2 for richer formatting.

---

*Generated for PS 26152 — AI-Powered Criminal Network Analysis System.*
