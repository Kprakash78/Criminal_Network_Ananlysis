"""
M6_feature — Unified Investigation Dashboard
PS 26152 — AI-Powered Criminal Network Analysis System

Main Streamlit application wiring together all four new features:
1. Case Selection / Upload
2. Case Search (semantic search with provenance)
3. Top-3 Suspects Panel (risk scores + explainability)
4. Timeline Player (animated event playback)
5. Court Evidence ZIP Exporter (one-click export)

Run with:
    CRIMINAL_USE_REAL_MODULES=0 streamlit run M6_feature/search_ui.py --server.headless true

Language discipline: no word implies guilt or criminal determination.
All scores indicate investigation-priority only.
"""

import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from datetime import datetime

import streamlit as st

# ---------------------------------------------------------------------------
# Ensure repo root is on sys.path for cross-module imports
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Criminal Network Analysis — Investigation Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for premium dark theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Global dark theme overrides */
    .stApp {
        background-color: #0e1117;
    }

    /* Suspect cards */
    .suspect-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #533483;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .suspect-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(83, 52, 131, 0.3);
    }
    .risk-score {
        font-size: 36px;
        font-weight: 700;
        background: linear-gradient(135deg, #e94560, #533483);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .risk-label {
        font-size: 12px;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .suspect-name {
        font-size: 20px;
        font-weight: 600;
        color: #e0e0e0;
        margin-bottom: 8px;
    }
    .evidence-tag {
        display: inline-block;
        background: #0f3460;
        color: #4FC3F7;
        padding: 3px 10px;
        border-radius: 4px;
        font-size: 12px;
        margin: 2px 4px 2px 0;
        cursor: pointer;
        font-family: monospace;
    }
    .evidence-tag:hover {
        background: #533483;
    }
    .explanation-text {
        color: #bbb;
        font-size: 14px;
        line-height: 1.6;
        margin-top: 8px;
    }

    /* Search results */
    .search-result {
        background: #16213e;
        border-left: 3px solid #4FC3F7;
        padding: 12px 16px;
        margin-bottom: 8px;
        border-radius: 0 8px 8px 0;
    }
    .search-score {
        color: #4FC3F7;
        font-weight: 600;
        font-size: 13px;
    }
    .citation-link {
        color: #FFB74D;
        font-family: monospace;
        font-size: 12px;
    }

    /* Export button */
    .export-btn {
        background: linear-gradient(135deg, #e94560, #c62828) !important;
        color: white !important;
        font-weight: 700 !important;
        font-size: 16px !important;
        padding: 12px 32px !important;
        border-radius: 8px !important;
        border: none !important;
        cursor: pointer !important;
        width: 100% !important;
    }

    /* Section headers */
    .section-header {
        font-size: 22px;
        font-weight: 600;
        color: #e0e0e0;
        border-bottom: 2px solid #533483;
        padding-bottom: 8px;
        margin-bottom: 16px;
    }

    /* Timeline container */
    .timeline-container {
        border: 1px solid #533483;
        border-radius: 12px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)


# ===========================================================================
# Session State Initialization
# ===========================================================================

if "search_results" not in st.session_state:
    st.session_state.search_results = None
if "top3_data" not in st.session_state:
    st.session_state.top3_data = None
if "timeline_events" not in st.session_state:
    st.session_state.timeline_events = None
if "selected_case" not in st.session_state:
    st.session_state.selected_case = "demo_case"
if "file_viewer_content" not in st.session_state:
    st.session_state.file_viewer_content = None


# ===========================================================================
# Sidebar — Case Selection & Status
# ===========================================================================

with st.sidebar:
    st.markdown("## 🔍 Criminal Network Analysis")
    st.markdown("*Investigation Dashboard v2.0*")
    st.markdown("---")

    # Case selection
    st.markdown("### 📁 Case Selection")
    case_option = st.selectbox(
        "Select Case",
        ["demo_case"],
        help="Select a case to analyze. Demo case uses synthetic data.",
    )
    st.session_state.selected_case = case_option

    # Status indicators
    st.markdown("### ⚙️ System Status")

    # Check available components
    top3_available = (REPO_ROOT / "M3_feature" / "top3_results.json").exists()
    demo_data_available = (REPO_ROOT / "data" / "firs").exists()

    st.markdown(f"- Data Files: {'✅' if demo_data_available else '❌'}")
    st.markdown(f"- Top-3 Results: {'✅' if top3_available else '⚠️ Run analysis first'}")
    st.markdown(f"- Search Engine: ✅ (TF-IDF fallback)")
    st.markdown(f"- Timeline: ✅")
    st.markdown(f"- Exporter: ✅")

    st.markdown("---")
    st.markdown(
        "⚠️ **Disclaimer**: All findings are investigative leads "
        "requiring verification. No score implies guilt."
    )
    st.markdown("---")
    st.markdown("🔒 **Offline Mode** — No external APIs used")


# ===========================================================================
# Main Content — Tabbed Layout
# ===========================================================================

st.markdown("# 🔍 Criminal Network Investigation Dashboard")
st.markdown("*AI-Powered Analysis with Full Evidence Provenance*")

tab_search, tab_suspects, tab_timeline, tab_export = st.tabs([
    "🔎 Case Search",
    "🎯 Top-3 Suspects",
    "⏱️ Timeline Player",
    "📦 Export Evidence",
])


# ===========================================================================
# Tab 1: Case Search
# ===========================================================================

with tab_search:
    st.markdown('<div class="section-header">🔎 Case Search</div>',
                unsafe_allow_html=True)
    st.markdown(
        "Search across all case documents. Results include exact file:line "
        "provenance for every finding."
    )

    col_search, col_opts = st.columns([4, 1])

    with col_search:
        query = st.text_input(
            "Enter your question",
            placeholder="e.g., Where was Ravi Kumar seen? What transactions involved ACC00106?",
            key="search_query",
        )

    with col_opts:
        top_k = st.number_input("Results", min_value=1, max_value=20,
                                value=5, key="search_top_k")

    if st.button("🔍 Search", type="primary", key="search_btn"):
        if query:
            with st.spinner("Searching case documents..."):
                try:
                    from M6_feature.search_api import search_local
                    results = search_local(query, top_k=top_k)
                    st.session_state.search_results = results
                except Exception as e:
                    st.error(f"Search failed: {e}")

    # Display search results
    if st.session_state.search_results:
        results = st.session_state.search_results

        # Answer box
        st.markdown("### 💬 Answer")
        st.info(results["answer"])

        # Individual results
        st.markdown("### 📄 Source Documents")
        for i, r in enumerate(results["results"], 1):
            score_pct = r["score"] * 100
            short_path = Path(r["file_path"]).name

            with st.expander(
                f"Result {i}: {short_path} (L{r['line_start']}-L{r['line_end']}) "
                f"— Score: {score_pct:.1f}%",
                expanded=(i <= 2),
            ):
                st.markdown(f'<span class="search-score">Relevance: {score_pct:.1f}%</span>',
                            unsafe_allow_html=True)
                st.markdown(f'<span class="citation-link">{r["citation"]}</span>',
                            unsafe_allow_html=True)
                st.code(r["text"], language=None)

                # File viewer button
                if st.button(f"📄 View Source File", key=f"view_file_{i}"):
                    try:
                        with open(r["file_path"], "r", encoding="utf-8",
                                  errors="replace") as f:
                            lines = f.readlines()

                        # Highlight relevant lines
                        start = max(0, r["line_start"] - 3)
                        end = min(len(lines), r["line_end"] + 3)

                        highlighted = []
                        for j in range(start, end):
                            prefix = ">>> " if r["line_start"] - 1 <= j <= r["line_end"] - 1 else "    "
                            highlighted.append(f"L{j+1:4d} {prefix}{lines[j].rstrip()}")

                        st.code("\n".join(highlighted), language=None)
                    except Exception as e:
                        st.error(f"Could not read file: {e}")


# ===========================================================================
# Tab 2: Top-3 Suspects
# ===========================================================================

with tab_suspects:
    st.markdown('<div class="section-header">🎯 Top-3 Suspects</div>',
                unsafe_allow_html=True)
    st.markdown(
        "Entities ranked by composite risk score. Each score is derived from "
        "network centrality, temporal patterns, co-location, call patterns, "
        "and financial anomalies."
    )

    # Load top3 results
    top3_path = REPO_ROOT / "M3_feature" / "top3_results.json"

    if top3_path.exists():
        with open(top3_path, "r", encoding="utf-8") as f:
            top3_data = json.load(f)
        st.session_state.top3_data = top3_data

        # Render suspect cards
        for idx, suspect in enumerate(top3_data.get("suspects", []), 1):
            risk_color = (
                "#e94560" if suspect["risk_score"] >= 75
                else "#FFB74D" if suspect["risk_score"] >= 50
                else "#4FC3F7"
            )

            st.markdown(f"""
            <div class="suspect-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <div class="suspect-name">#{idx} {suspect['name']}</div>
                        <div class="risk-label">Investigation Priority Score</div>
                    </div>
                    <div style="text-align:right;">
                        <div class="risk-score" style="background:linear-gradient(135deg, {risk_color}, #533483);
                            -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
                            {suspect['risk_score']:.1f}
                        </div>
                        <div class="risk-label">/ 100</div>
                    </div>
                </div>
                <div class="explanation-text">
                    {suspect['explanation']}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Evidence references
            evidence = suspect.get("evidence", [])
            if evidence:
                with st.expander(f"📎 Evidence References ({len(evidence)} items)", expanded=False):
                    for ev in evidence:
                        short_file = Path(ev["file"]).name
                        st.markdown(
                            f'`[{short_file}:L{ev["line_start"]}-L{ev["line_end"]}]` '
                            f'— {ev["text_snippet"][:120]}...'
                        )

                        # View source button
                        btn_key = f"ev_{idx}_{ev['line_start']}_{hash(ev['file'])}"
                        if st.button(f"📄 View lines", key=btn_key):
                            try:
                                with open(ev["file"], "r", encoding="utf-8",
                                          errors="replace") as f:
                                    lines = f.readlines()
                                start = max(0, ev["line_start"] - 2)
                                end = min(len(lines), ev["line_end"] + 2)
                                display = []
                                for j in range(start, end):
                                    prefix = ">>> " if ev["line_start"] - 1 <= j <= ev["line_end"] - 1 else "    "
                                    display.append(f"L{j+1:4d} {prefix}{lines[j].rstrip()}")
                                st.code("\n".join(display), language=None)
                            except Exception as e:
                                st.error(f"Could not read file: {e}")

            st.markdown("---")

        # Score methodology
        with st.expander("📊 Scoring Methodology"):
            st.markdown("""
            **Risk Score Formula** (0–100 scale):

            ```
            risk_score = normalize(
                0.20 × degree_centrality       # Network connectivity
              + 0.20 × temporal_anomaly         # Unusual timing patterns
              + 0.15 × co-location_score        # Same-tower convergence
              + 0.20 × call_pattern_score       # Call frequency & patterns
              + 0.25 × transaction_anomaly      # Financial red flags
            )
            ```

            Each component returns a value in [0, 1]. The weighted sum is
            scaled to 0–100. Higher scores indicate higher investigation priority.

            **⚠️ Scores indicate investigation priority, NOT guilt determination.**
            """)
    else:
        st.warning(
            "Top-3 results not found. Run the analysis first:\n\n"
            "```bash\npython3 -m M3_feature.top3\n```"
        )


# ===========================================================================
# Tab 3: Timeline Player
# ===========================================================================

with tab_timeline:
    st.markdown('<div class="section-header">⏱️ Timeline Player</div>',
                unsafe_allow_html=True)
    st.markdown(
        "Animated playback of all case events (calls, transactions, FIR filings) "
        "in chronological order. Events are highlighted on the network graph."
    )

    # Build timeline events
    from M6_feature.timeline_player import build_timeline

    case_id = st.session_state.get("selected_case")

    if not case_id:
        events = []
    else:
        try:
            # The active investigation must provide its entity set.
            #
            # Your main M6 application may already keep this in session state.
            # Support the common names without inventing global data.
            case_entities = (
                st.session_state.get("case_entities")
                or st.session_state.get("selected_case_entities")
                or st.session_state.get("active_case_entities")
                or set()
            )

            case_entities = set(case_entities)

            events = build_timeline(
                case_id=case_id,
                case_entities=case_entities,
            )

            # Important:
            # invalidate cached timeline when switching cases.
            st.session_state.timeline_case_id = case_id
            st.session_state.timeline_events = events

        except Exception as e:
            st.error(f"Failed to build timeline for case {case_id}: {e}")
            events = []

    if events:
        st.markdown(f"**{len(events)} events** from "
                    f"{events[0]['t'][:10]} to {events[-1]['t'][:10]}")

        # Date range filter
        col1, col2 = st.columns(2)
        with col1:
            date_start = st.date_input("From Date",
                                       value=datetime(2024, 1, 1),
                                       key="tl_start")
        with col2:
            date_end = st.date_input("To Date",
                                     value=datetime(2026, 12, 31),
                                     key="tl_end")

        # Filter events by date range
        filtered = [
            e for e in events
            if str(date_start) <= e["t"][:10] <= str(date_end)
        ]

        # Event type filter
        event_types = st.multiselect(
            "Event Types",
            ["call", "transaction", "fir_filing"],
            default=["call", "transaction", "fir_filing"],
            key="tl_types",
        )
        filtered = [e for e in filtered if e["type"] in event_types]

        st.markdown(f"**Showing {len(filtered)} events** after filters")

        # Render timeline HTML
        try:
            from M6_feature.timeline_ui import render_timeline_html
            timeline_html = render_timeline_html(filtered, width=900, height=700)

            st.markdown('<div class="timeline-container">',
                        unsafe_allow_html=True)
            st.components.v1.html(timeline_html, height=700, scrolling=False)
            st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Timeline rendering failed: {e}")

        # Event log table
        with st.expander("📋 Event Log (tabular)"):
            import pandas as pd
            df = pd.DataFrame([{
                "Time": e["t"],
                "Type": e["type"],
                "From": e.get("from_name", e.get("from", "")),
                "To": e.get("to_name", e.get("to", "")),
                "Source": f"{e['file']}:L{e['line']}",
            } for e in filtered[:100]])
            st.dataframe(df, use_container_width=True)
    else:
        st.info("No events available. Ensure demo data exists in data/.")


# ===========================================================================
# Tab 4: Export Evidence
# ===========================================================================

with tab_export:
    st.markdown('<div class="section-header">📦 Court Evidence Exporter</div>',
                unsafe_allow_html=True)
    st.markdown(
        "Package all case evidence into a signed ZIP archive for court submission. "
        "Includes raw files, parsed outputs, AI analysis results, and a "
        "cryptographic manifest for integrity verification."
    )

    st.markdown("### 📋 Package Contents")
    st.markdown("""
    | Component | Description |
    |-----------|-------------|
    | `raw_data/` | Original FIRs, CDRs, transactions |
    | `parsed_outputs/` | Machine-extracted entities & relationships |
    | `ai_outputs/` | Top-3 suspect rankings, search audit logs |
    | `manifest.json` | File inventory with SHA-256 hashes |
    | `manifest.sig` | RSA-2048 digital signature |
    | `public_key.pem` | Public key for signature verification |
    | `VERIFY.md` | Step-by-step verification guide |
    """)

    st.markdown("---")

    col_export, col_opts = st.columns([3, 2])

    with col_opts:
        encrypt = st.checkbox("🔐 Encrypt with passphrase", key="encrypt_toggle")
        passphrase = None
        if encrypt:
            passphrase = st.text_input("Passphrase", type="password",
                                       key="export_passphrase")

    with col_export:
        if st.button("🔴 Export Evidence Package", type="primary",
                     key="export_btn", use_container_width=True):
            with st.spinner("Creating evidence package..."):
                try:
                    from M6_feature.exporter import export_case_zip

                    # Use temp directory for output
                    import tempfile
                    outdir = tempfile.mkdtemp()
                    case_id = st.session_state.selected_case

                    zip_path = export_case_zip(
                        case_id, outdir,
                        encrypt_passphrase=passphrase if encrypt and passphrase else None,
                    )

                    # Read the ZIP for download
                    with open(zip_path, "rb") as f:
                        zip_bytes = f.read()

                    st.success(f"✅ Evidence package created successfully!")
                    st.markdown(f"**Size**: {len(zip_bytes):,} bytes")

                    st.download_button(
                        label="⬇️ Download Evidence ZIP",
                        data=zip_bytes,
                        file_name=f"case_{case_id}_evidence.zip",
                        mime="application/zip",
                        key="download_zip",
                    )

                except Exception as e:
                    st.error(f"Export failed: {e}")
                    logger.error(f"Export failed: {e}", exc_info=True)

    # Verification info
    with st.expander("🔐 Verification Instructions"):
        st.markdown("""
        After downloading, verify the evidence package:

        **1. Check file integrity:**
        ```bash
        python3 -c "
        import hashlib, json, zipfile, sys
        with zipfile.ZipFile(sys.argv[1]) as z:
            manifest = json.loads(z.read('manifest.json'))
            for f in manifest['files']:
                data = z.read(f['path'])
                h = hashlib.sha256(data).hexdigest()
                status = '✓' if h == f['sha256'] else '✗'
                print(f'{status} {f[\"path\"]}')" case_demo_case_evidence.zip
        ```

        **2. Verify digital signature:**
        ```python
        # Requires: pip install cryptography
        from cryptography.hazmat.primitives.asymmetric import padding
        from cryptography.hazmat.primitives import hashes, serialization
        # See VERIFY.md inside the ZIP for full instructions
        ```
        """)
