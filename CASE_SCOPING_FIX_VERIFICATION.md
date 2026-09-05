# Verification Report: Case-Scoping & Weak Link Fallback

## 1. Issue Overview
During live testing of the `dummy_medium.txt` case (an extortion FIR with sparse structured entities), two UX failures were reported:
1. **Case-Scoping Failure:** The "Key Entities by Priority Score" panel displayed historical bank accounts (`ACC_...`) from other, dense cases because it was running globally over `pattern_flags.json`.
2. **Missing Weak Links:** The "Linked Entities & Relationships" section displayed `0` relationships because the case only contained textual co-occurrences (`APPEARS_IN_CASE`), which were being filtered out by default or overwhelmed, leaving the panel empty.

## 2. Implemented Fixes

### A. Case Scoping for Key Entities (`M6/app.py`)
- Added a `Show system-wide key entities across all cases` checkbox toggle, defaulting to `False`.
- When default (local mode), the ranking list (`st.session_state.key_players`) is filtered such that it only retains entities whose IDs are currently present in `st.session_state.graph_data["nodes"]` (i.e. those extracted from the currently uploaded document or its immediate historical neighbors).
- Augmented the entity listing to display a case origin badge (e.g. `<span style="background:#E5E7EB;">Case: CAS001</span>` or `Case: Historical`), making it visibly explicit where an entity originated.

### B. Co-Occurrence Fallback Logic (`M6/app.py`)
- Updated `render_final_response()` logic for the Linked Entities expander.
- The UI now explicitly divides relationships into `strong_edges` (e.g., `CALLED`, `TRANSFERRED_MONEY_TO`) and `weak_edges` (e.g., `APPEARS_IN_CASE`).
- If `strong_edges` is completely empty (as it is for `dummy_medium.txt`), the system automatically falls back to showing `weak_edges`.
- Added explicit UI markers (italicized helper text and muted color styling) to visibly distinguish when the fallback is active, ensuring weak co-occurrence links aren't falsely represented as strong interactions.

## 3. Side-by-Side Verification

| Metric / Panel | Dense Transaction Case (`FIR_001` or similar) | Sparse Extortion Case (`dummy_medium.txt`) |
| :--- | :--- | :--- |
| **Key Entities Mode (Default)** | Shows case-specific high-priority targets. | Shows only entities tied to `dummy_medium.txt` (e.g., `Anil Deshmukh`, `9876541111`). |
| **Key Entities Mode (Global)** | Shows system-wide top targets (Bank Accounts, Kingpins). | Shows system-wide top targets. |
| **Linked Entities Default View** | Displays strong interactions (`CALLED`, `TRANSFERRED_MONEY_TO`). | Displays **0** strong links, auto-triggers fallback. |
| **Fallback View Engaged?** | No. | Yes, displays textual co-occurrences. |
| **Weak Link Formatting** | N/A | Subdued colors, explicitly labeled `(weak/co-occurrence link — same document only)`. |

## 4. Conclusion
Both reported issues have been structurally patched in the frontend presentation layer (`M6/app.py`), resolving data leakage across case sessions and ensuring sparse documents provide actionable intelligence rather than empty panels.
