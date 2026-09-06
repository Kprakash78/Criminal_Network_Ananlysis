"""
M6_feature/ui_cards.py
PS 26152 — AI-Powered Criminal Network Analysis System
======================================================
Streamlit UI Cards for the four new analytical features.

Three buttons:
  1. "Show Chains"           — displays event-causality chains as a table
                               with provenance links.
  2. "Show Ghost Suggestions" — displays ghost-node suggestions with
                               Adamic-Adar / Jaccard scores.
  3. "Export Tactical Brief" — generates and serves the auto-brief PDF.

Designed to be imported into M6_feature/search_ui.py or run standalone:
    streamlit run M6_feature/ui_cards.py

All language is safe (no guilt determinations).
Fully offline — reads only cached JSON files from demo_cache/.
"""

from __future__ import annotations

import json
from pathlib import Path

# Guard import so module can be imported without streamlit (for tests)
try:
    import streamlit as st
    _HAS_ST = True
except ImportError:
    _HAS_ST = False

REPO_ROOT = Path(__file__).resolve().parent.parent
FEAT_OUT  = REPO_ROOT / "demo_cache" / "feature_outputs"
M3_DIR    = REPO_ROOT / "M3_feature"


# ── Data loaders ──────────────────────────────────────────────────────────────

def _load_json(path: Path) -> dict | list | None:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return None
    return None


def _find(filename: str) -> Path | None:
    for d in (FEAT_OUT, M3_DIR):
        p = d / filename
        if p.exists():
            return p
    return None


def _chains_data(case_id: str) -> list[dict]:
    p = _find("event_chains.json")
    if not p:
        return []
    data = _load_json(p)
    if not data or "chains" not in data:
        return []
    return [c for c in data["chains"] if c.get("case_id") == case_id]


def _ghost_data(case_id: str) -> list[dict]:
    p = _find("ghost_suggestions.json")
    if not p:
        return []
    data = _load_json(p)
    if not data or "suggestions" not in data:
        return []
    suggs = data["suggestions"]
    if isinstance(suggs, dict):
        return suggs.get(case_id, [])
    return suggs   # flat list fallback


def _motif_data(case_id: str) -> list[dict]:
    p = _find("motif_flags.json")
    if not p:
        return []
    data = _load_json(p)
    if not data or "flags" not in data:
        return []
    return data["flags"].get(case_id, [])


def _brief_pdf_path(case_id: str) -> Path:
    return FEAT_OUT / f"brief_{case_id}.pdf"


# ── Public card functions (safe to call without streamlit) ────────────────────

def render_chains_card(case_id: str = "case_A") -> None:
    """Render the Event-Causality Chains card in a Streamlit UI."""
    if not _HAS_ST:
        return

    st.subheader("📋 Event-Causality Chains")
    chains = _chains_data(case_id)

    if not chains:
        st.warning(
            "No chain data found. Run:\n"
            "`python -m M3_feature.event_causality build_all`"
        )
        return

    st.caption(
        f"{len(chains)} chain(s) found for **{case_id}**. "
        "All events require investigator verification."
    )

    for chain in chains[:5]:
        chain_id  = chain["chain_id"]
        n_steps   = chain["n_steps"]
        pivot     = chain.get("pivot", {})
        pivot_idx = pivot.get("step_index", -1)
        pivot_sc  = pivot.get("score", 0)

        with st.expander(
            f"⛓ Chain `{chain_id}` — {n_steps} steps  "
            f"| Pivot step {pivot_idx} (score {pivot_sc:.2f})"
        ):
            rows = []
            for si, step in enumerate(chain["steps"]):
                rows.append({
                    "Step":   si,
                    "Time":   step.get("t", "")[:16].replace("T", " "),
                    "Actor":  step.get("actor", ""),
                    "Action": step.get("action", ""),
                    "Target": step.get("target", ""),
                    "Source": f"{step.get('source','')} :L{step.get('line','')}",
                    "Pivot":  "🔴" if si == pivot_idx else "",
                })
            st.table(rows)
            st.caption(chain.get("note", ""))


def render_ghost_card(case_id: str = "case_A") -> None:
    """Render the Ghost-Node Suggestions card in a Streamlit UI."""
    if not _HAS_ST:
        return

    st.subheader("👻 Ghost-Node Suggestions")
    suggestions = _ghost_data(case_id)

    if not suggestions:
        st.warning(
            "No ghost suggestions found. Run:\n"
            "`python -m M3_feature.ghost_node suggest_all`"
        )
        return

    st.caption(
        f"{len(suggestions)} possible unobserved intermediary suggestion(s) "
        f"for **{case_id}**. All require investigator verification."
    )

    rows = []
    for s in suggestions[:10]:
        scores = s.get("scores", {})
        rows.append({
            "Pair (A ↔ B)":    " ↔ ".join(s.get("pair", ["?", "?"])),
            "Cross-Cluster":   "✅" if s.get("cross_cluster") else "—",
            "Common Nbrs":     scores.get("common_neighbours", 0),
            "Adamic-Adar":     f"{scores.get('adamic_adar', 0):.3f}",
            "Jaccard":         f"{scores.get('jaccard', 0):.3f}",
            "Composite":       f"{scores.get('composite', 0):.4f}",
        })
    st.table(rows)

    with st.expander("🔍 Explanations"):
        for s in suggestions[:5]:
            st.markdown(f"- {s.get('explanation', '')}")


def render_motif_card(case_id: str = "case_A") -> None:
    """Render the Temporal Motif Flags card in a Streamlit UI."""
    if not _HAS_ST:
        return

    st.subheader("🕐 Temporal Anomaly Flags")
    flags = _motif_data(case_id)

    if not flags:
        st.info("No temporal anomalies detected above threshold for this case.")
        return

    st.caption(
        f"{len(flags)} temporal anomaly flag(s) for **{case_id}**. "
        "Requires investigator verification."
    )
    rows = []
    for fl in flags:
        rows.append({
            "Name":         fl.get("name", ""),
            "Phone":        fl.get("phone", ""),
            "Off-hr Calls": fl.get("off_hour_calls", 0),
            "Peak Calls":   fl.get("peak_hour_calls", 0),
            "Ratio":        f"{fl.get('multiplier_observed', 0):.1f}×",
            "Source":       fl.get("evidence", {}).get("cdr_file", ""),
        })
    st.table(rows)


def render_brief_card(case_id: str = "case_A") -> None:
    """Render the Export Tactical Brief card in a Streamlit UI."""
    if not _HAS_ST:
        return

    st.subheader("📄 Tactical Brief Export")
    pdf_path = _brief_pdf_path(case_id)

    if pdf_path.exists():
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        st.download_button(
            label=f"⬇️ Download Brief — {case_id}.pdf",
            data=pdf_bytes,
            file_name=f"brief_{case_id}.pdf",
            mime="application/pdf",
        )
        manifest_path = FEAT_OUT / f"manifest_{case_id}.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            st.caption(
                f"SHA-256: `{manifest.get('pdf_sha256','')[:20]}…`  "
                f"| Generated: {manifest.get('generated_at', '')}"
            )
    else:
        st.info("Brief not yet generated.")
        if st.button("Generate Brief Now"):
            with st.spinner("Generating PDF brief..."):
                try:
                    import subprocess, sys
                    subprocess.run(
                        [sys.executable, "-m", "M6_feature.auto_brief",
                         "export", case_id],
                        cwd=str(REPO_ROOT), check=True,
                        capture_output=True
                    )
                    st.success("Brief generated! Refresh to download.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Generation failed: {e}")


# ── Standalone Streamlit app ──────────────────────────────────────────────────

def _standalone_app() -> None:
    st.set_page_config(
        page_title="CNA — Investigation Features",
        page_icon="🔍",
        layout="wide",
    )
    st.title("🔍 Criminal Network Analysis — Investigation Features")
    st.caption(
        "All findings are **preliminary** and require **investigator verification**. "
        "No guilt determinations are made by this system."
    )

    case_id = st.sidebar.selectbox(
        "Select Case", ["case_A", "case_B", "case_C"], index=0
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "⛓ Event Chains",
        "👻 Ghost Nodes",
        "🕐 Temporal Flags",
        "📄 Tactical Brief",
    ])

    with tab1:
        render_chains_card(case_id)
    with tab2:
        render_ghost_card(case_id)
    with tab3:
        render_motif_card(case_id)
    with tab4:
        render_brief_card(case_id)


if __name__ == "__main__" and _HAS_ST:
    _standalone_app()
