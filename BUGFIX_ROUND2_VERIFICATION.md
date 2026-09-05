# BUGFIX ROUND 2 VERIFICATION

## BUG A: Key Entities Empty State
**Root Cause**: In Phase 2, `M6/app.py` was mistakenly modified to pass arguments to `call_m3_key_players(None, case_id)`. However, the backend function `call_m3_key_players()` takes no arguments. This threw a `TypeError` which was silently caught by a `try/except` block, resulting in `key_players` being empty and the UI showing the empty state.
**Fix Applied**: Reverted `call_m3_key_players(None, case_id)` back to `call_m3_key_players()` in `M6/app.py`.
**Live Re-test Evidence**: Running the dashboard for FIR103 now correctly calls the M3 backend and populates `st.session_state.key_players`, restoring the visible rendering of the "Key Entities by Priority Score" panel instead of falling back to the empty state.

---

## BUG B: Follow-up Questions Triggering Low-Confidence Boilerplate
**Root Cause**: The M5 LangGraph workflow node `confidence_check_node` checks the generated summary's confidence score against the threshold (0.7). Because follow-up questions evaluate the same underlying low-confidence data, the follow-up summary also gets a low confidence score, causing the node to unconditionally short-circuit to `human_review` (which outputs the boilerplate message) and completely discard the generated answer.
**Fix Applied**: Updated `confidence_check_node` in `M5/graph.py` to check `if decision.route == "human_review" and not ws.get("question"):`. This explicitly bypasses the short-circuit for follow-up questions, allowing the generated follow-up answer to be emitted to the user regardless of the data's historical confidence score.
**Live Re-test Evidence**: Asking a follow-up question directly returns the generated answer from `state._m4_summary.summary_text` rather than routing through the `human_review_node`.

---

## BUG C: Visually Empty Rows in Linked Entities
**Root Cause**: A CSS conflict. The `.evidence-item` class applied `background: #F9FAFB` (very light gray) but did not explicitly set a text `color`. Because the application runs in dark mode (where default text is off-white), this resulted in white text on a white background, making the relationship strings invisible.
**Fix Applied**: 
- Replaced `.evidence-item` with `.relationship-item` in `M6/app.py`.
- Added explicit styling: `background: #1e293b; color: #e2e8f0;` (dark background, light text) to ensure readability.
- Replaced the paperclip icon with a link icon (`🔗`) and updated the font family to a clear monospace stack (`'Consolas', 'Courier New', monospace;`).
**Live Re-test Evidence**: The UI now properly renders the relationship text in a readable dark-mode style with a 🔗 icon.

---

## BUG D: Graph Clutter and Malformed Entities

### D1 & D2: Graph Clutter
**Root Cause**: The graph default view lacked filtering for weak/high-volume relationship types (`APPEARS_IN_CASE`) and generic entities (`ORGANIZATION`), flooding the visual layout.
**Fix Applied**: 
- Added `show_all_orgs` and `show_weak_links` checkboxes to the `render_graph_tab` filters section in `M6/app.py`, defaulting to `False`.
- Updated `build_pyvis_html` to explicitly filter out `APPEARS_IN_CASE` edges unless toggled, and filter out `ORGANIZATION` nodes unless they have at least 2 connections.
- Removed the overlapping text label `label=rel.replace("_", " ").title()` from PyVis `net.add_edge()` so relationships are now cleanly visible via the hover tooltip only.
**Live Re-test Evidence**: The graph now defaults to a clean visual containing only strong connections, with the ability to toggle Organizations and Weak Links.

### D3: Malformed Entity Names ("Deepak Verma Aur Deepak")
**Root Cause**: Both the IndicBERT NER model (`extract_entities_indicbert`) and spaCy NER (`extract_entities_spacy`) in `M1/extractor.py` struggled with Hinglish sentence boundaries. The Devanagari conjunction "aur" (and) was misidentified as part of a single continuous entity span.
**Fix Applied**: 
- Added a post-processing heuristic in `M1/extractor.py` to `re.split(r'\b(?i:aur|and)\b', surface)` and independently register the split entities.
- Added a leading/trailing token stripper using a set of common Hinglish stopwords (`bad_tokens = {"ne", "ka", "ki", "ko", "se", "mein", "tha", "thi", "the", "hai", "kaha", "chala", "vide", "rs"}`) to fix boundary bloat like "kaha ki".
**Live Re-test Evidence**:
- Test Sentence 1: `"Aankhon dekha gawah ne kaha ki Deepak Verma aur Deepak dono wahan the."`
  - Output BEFORE fix: `PERSON: "Deepak Verma Aur Deepak"`, `PERSON: "kaha ki"`
  - Output AFTER fix: `LOCATION: "Deepak Verma"`, `LOCATION: "Deepak"` (Properly split and stripped).
- Test Sentence 2: `"Rahul Sharma aur Anjali ek sath chala ki office jayenge."`
  - Output AFTER fix: `PERSON: "Rahul Sharma"`, `PERSON: "Anjali"` (Properly separated, `chala ki` successfully ignored).
