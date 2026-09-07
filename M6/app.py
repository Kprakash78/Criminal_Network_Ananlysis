"""
M6 — Investigator Dashboard
PS 26152 — AI-Powered Criminal Network Analysis System

Run with:
    streamlit run M6/app.py  (from repo root)

This file implements all 10 milestones (M6.1–M6.10):
  M6.1  Basic Streamlit shell + upload widget
  M6.2  Result rendering (summary, evidence, confidence, human-review flag)
  M6.3  Graph visualization (PyVis interactive graph from M2 query layer)
  M6.4  Search & filter (entity search + date range filter)
  M6.5  Key players view (ranked by M3 priority score)
  M6.6  Follow-up question / conversational UI within a session
  M6.7  Export (CSV, JSON, PDF)
  M6.8  Real vs mock flag (USE_REAL_MODULES env var, surfaced in sidebar)
  M6.9  Full demo walkthrough (same script a judge would see, invokable in-app)
  M6.10 Polish (confidence color-coding, type icons, clean layout)

Language discipline: no word in this file implies guilt/criminality.
Banned words: criminal, guilty, confirmed, suspect (when used as a verdict),
prove, proof, convicted — see BANNED_WORDS in ui_helpers section.
"""

import os
import sys
import logging
import tempfile
import json
import io
from datetime import datetime, timedelta
from pathlib import Path

import streamlit as st
import pandas as pd

# ---------------------------------------------------------------------------
# Ensure repo root is on sys.path for imports from M5/M6/etc.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = str(REPO_ROOT)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Imports — backend call layer (never import M5/M2/M3 directly here)
# ---------------------------------------------------------------------------
from M6.backend_calls import (
    call_m5, call_m2_subgraph, call_m2_search, call_m3_key_players,
    USE_REAL_MODULES, get_backend_status,
)
from M6.export import export_csv, export_json, export_pdf

# M5 request types — needed for type instantiation
try:
    from M5.models import NewCaseUpload, FollowUpQuestion
except ImportError:
    # Fallback lightweight shims so UI loads even without M5 installed
    from dataclasses import dataclass

    @dataclass
    class NewCaseUpload:
        case_id: str
        case_text: str
        source_dir: str = ""

    @dataclass
    class FollowUpQuestion:
        question: str


# ===========================================================================
# M6.10 — Color / style helpers (polished, non-accusatory)
# ===========================================================================

# Confidence color-coding: green ≥ 0.7 | amber 0.5–0.7 | red < 0.5
def confidence_color(score: float) -> str:
    if score >= 0.7:
        return "#2ECC71"   # green
    elif score >= 0.5:
        return "#F39C12"   # amber
    else:
        return "#E74C3C"   # red


def confidence_label(score: float) -> str:
    if score >= 0.7:
        return "High"
    elif score >= 0.5:
        return "Moderate"
    else:
        return "Low — requires human review"


# Entity type to emoji icon (for visual scannability)
TYPE_ICONS = {
    "PERSON": "🧑",
    "PHONE": "📱",
    "ACCOUNT": "🏦",
    "LOCATION": "📍",
    "VEHICLE": "🚗",
    "ORGANIZATION": "🏢",
    "UNKNOWN": "❓",
}

BANNED_WORDS = {"criminal", "guilty", "confirmed", "convicted", "proven", "proof"}

FLAG_LABELS = {
    "COMMUNICATION_SPIKE": "Communication Spike",
    "HIGH_TRANSACTION_FREQUENCY": "High Transaction Frequency",
    "MULTI_SUSPECT_SHARED_ACCOUNT": "Shared Account (Multiple Persons)",
    "RAPID_FUND_MOVEMENT": "Rapid Fund Movement",
    "INCIDENT_TIMING_CLUSTER": "Incident Timing Cluster",
    "DENSE_CLUSTER_MEMBERSHIP": "Dense Network Cluster",
}


# ===========================================================================
# M6.3 — Graph visualization helper (PyVis)
# ===========================================================================

def build_pyvis_html(graph_data: dict, filter_from: datetime = None, filter_to: datetime = None, show_all_orgs: bool = False, show_weak_links: bool = False, selected_types: list = None) -> str:
    """
    Convert {nodes, edges} dict to an interactive PyVis HTML string.
    Applies date-range filter to edges (M6.4).
    Color-codes nodes by entity type.
    Bug 3 fix: cap edges at MAX_VIS_EDGES to prevent browser blank-render.
    Bug 3 fix: use a proper temp file path (Windows NamedTemporaryFile can't
               be read while open on some systems).

    Visual style: detective evidence board (corkboard background, square icon
    tiles per entity type, red string edges distinguishing strong vs. weak links).
    """
    from pyvis.network import Network
    import os
    import base64

    MAX_VIS_EDGES = 300  # beyond this browsers freeze / render blank

    TYPE_COLORS = {
        "PERSON":       "#AF5D4E",   # terracotta
        "PHONE":        "#B88B45",   # ochre
        "ACCOUNT":      "#8C6D5B",   # warm brown
        "LOCATION":     "#72826C",   # muted green
        "VEHICLE":      "#825E6C",   # muted plum
        "ORGANIZATION": "#6B7C8A",   # muted blue-grey
        "UNKNOWN":      "#9CA3AF",   # grey
    }

    # -----------------------------------------------------------------------
    # Load icon images as base64 data URIs (fully local — no network calls)
    # Icons live in M6/assets/icons/{type}.png (80×80 RGBA PNG).
    # -----------------------------------------------------------------------
    _ICONS_DIR = Path(__file__).resolve().parent / "assets" / "icons"

    def _load_icon_b64(name: str) -> str:
        p = _ICONS_DIR / f"{name}.png"
        if p.exists():
            return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()
        # Fallback: 1×1 transparent PNG so vis-network never crashes on missing icon
        return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="

    TYPE_ICONS_B64 = {
        "PERSON":       _load_icon_b64("person"),
        "PHONE":        _load_icon_b64("phone"),
        "ACCOUNT":      _load_icon_b64("account"),
        "LOCATION":     _load_icon_b64("location"),
        "VEHICLE":      _load_icon_b64("vehicle"),
        "ORGANIZATION": _load_icon_b64("organization"),
        "DATE":         _load_icon_b64("date"),
        "UNKNOWN":      _load_icon_b64("unknown"),
    }

    # -----------------------------------------------------------------------
    # Edge styling: RED STRING
    # Strong links (CALLED, TRANSFERRED_MONEY_TO, OWNS, …):
    #   solid crimson, width ∝ weight (2–6 px)
    # Weak links (APPEARS_IN_CASE / APPEARS_IN_SAME_DOCUMENT):
    #   dashed dark maroon, width 1.5 px — clearly subordinate
    # This preserves the pre-existing strong-vs-weak distinction.
    # -----------------------------------------------------------------------
    STRONG_EDGE_COLOR = "#9E5748"   # muted terracotta
    WEAK_EDGE_COLOR   = "#C1A598"   # lighter warm clay
    EDGE_HIGHLIGHT    = "#7B4336"   # dark terracotta on hover
    WEAK_REL_TYPES = {"APPEARS_IN_CASE", "APPEARS_IN_SAME_DOCUMENT"}

    # height must match iframe height in st.components.v1.html()
    # Task 4: increased from 580px → 850px for a proper detective-board canvas
    net = Network(height="850px", width="100%", directed=True,
                  bgcolor="#E8D8C2",       # warm kraft-paper tone
                  font_color="#3E2723",
                  notebook=False)

    net.set_options("""
    {
      "nodes": {
        "shape": "box",
        "margin": 10,
        "font": {
          "size": 14,
          "color": "#3E2723",
          "multi": "html",
          "align": "left"
        },
        "shadow": {"enabled": true, "color": "rgba(62,39,35,0.15)", "size": 6, "x": 2, "y": 3},
        "shapeProperties": {"borderRadius": 4},
        "borderWidth": 1,
        "borderWidthSelected": 2
      },
      "edges": {
        "smooth": {"enabled": true, "type": "curvedCW", "roundness": 0.15},
        "arrows": {"to": {"enabled": true, "scaleFactor": 0.7, "type": "arrow"}},
        "font": {"size": 10, "color": "#5D4037", "strokeWidth": 2, "strokeColor": "#E8D8C2", "align": "middle"},
        "selectionWidth": 2
      },
      "physics": {
        "stabilization": {"iterations": 100},
        "barnesHut": {"gravitationalConstant": -9000, "centralGravity": 0.3, "springLength": 130}
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100,
        "navigationButtons": true,
        "keyboard": true
      }
    }
    """)

    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])

    # Collect IDs of nodes that actually appear in edges (after capping) so we
    # don't render isolated nodes for invisible edges.
    # Check if there are ANY strong edges in the entire graph data
    has_strong_edges = any(e.get("type") != "APPEARS_IN_CASE" for e in edges)

    # First: filter + cap edges
    filtered_edges = []
    for edge in edges:
        ts_str = edge.get("timestamp", "")
        if ts_str and (filter_from or filter_to):
            try:
                ts = datetime.strptime(ts_str[:16], "%Y-%m-%d %H:%M")
                if filter_from and ts < filter_from:
                    continue
                if filter_to and ts > filter_to:
                    continue
            except ValueError:
                pass
                
        # D2 Fix: Filter weak links by default, UNLESS there are no strong links at all (fallback)
        if not show_weak_links and edge.get("type") == "APPEARS_IN_CASE":
            if has_strong_edges:
                continue
            
        filtered_edges.append(edge)

    # Cap by weight descending so we keep the most significant connections
    if len(filtered_edges) > MAX_VIS_EDGES:
        filtered_edges = sorted(
            filtered_edges, key=lambda e: e.get("weight", 1.0), reverse=True
        )[:MAX_VIS_EDGES]

    # Determine which node IDs appear in capped edges
    connected_ids: set[str] = set()
    for edge in filtered_edges:
        connected_ids.add(edge["source"])
        connected_ids.add(edge["target"])

    # Add nodes (only those connected, or all if no edges)
    nodes_to_show = nodes if not connected_ids else [
        n for n in nodes if n["id"] in connected_ids
    ]
    
    # Filter by selected_types
    if selected_types is not None:
        nodes_to_show = [n for n in nodes_to_show if n.get("type", "UNKNOWN") in selected_types]

    # Always show at least the first 80 nodes if no connections
    if not nodes_to_show and not selected_types:
        nodes_to_show = nodes[:80]
        
    # D1 Fix: Filter organizations by default
    if not show_all_orgs:
        # A "seriously involved" organization might be in key_players or have strong links.
        # For simplicity, if it's an ORGANIZATION and we are hiding them, only keep it if it has >1 connection.
        # Count connections for each node:
        degree = {}
        for edge in filtered_edges:
            degree[edge["source"]] = degree.get(edge["source"], 0) + 1
            degree[edge["target"]] = degree.get(edge["target"], 0) + 1
            
        filtered_nodes = []
        for n in nodes_to_show:
            if n.get("type") == "ORGANIZATION":
                if degree.get(n["id"], 0) >= 2:
                    filtered_nodes.append(n)
            else:
                filtered_nodes.append(n)
        nodes_to_show = filtered_nodes

    # Re-filter edges to ensure we don't add edges for nodes we just removed
    final_node_ids = {n["id"] for n in nodes_to_show}
    final_edges = [
        e for e in filtered_edges
        if e["source"] in final_node_ids and e["target"] in final_node_ids
    ]

    # ── Build adjacency map for tooltip connection lists (Task 2) ──────────
    # node_id → list of (neighbour_name, rel_type, is_weak, confidence)
    # sorted by confidence desc, capped at 7 in the tooltip renderer below.
    node_name_map: dict[str, str] = {
        n["id"]: n.get("name", n.get("label", n.get("id", ""))) for n in nodes
    }
    adjacency: dict[str, list[tuple]] = {nid: [] for nid in final_node_ids}
    for edge in final_edges:
        src, tgt = edge["source"], edge["target"]
        rel = edge.get("type", "CONNECTED_TO")
        conf_e = edge.get("confidence", 1.0)
        is_weak_e = rel in WEAK_REL_TYPES
        if src in adjacency:
            adjacency[src].append((node_name_map.get(tgt, tgt), rel, is_weak_e, conf_e))
        if tgt in adjacency:
            adjacency[tgt].append((node_name_map.get(src, src), rel, is_weak_e, conf_e))
    # sort each list by confidence descending
    for nid in adjacency:
        adjacency[nid].sort(key=lambda x: x[3], reverse=True)

    TOOLTIP_CAP = 7   # max connections shown in tooltip before "+ N more"

    # ── Add nodes as square image tiles ───────────────────────────────────
    for node in nodes_to_show:
        ntype = node.get("type", "UNKNOWN")
        icon_emoji = TYPE_ICONS.get(ntype, "❓")
        color = TYPE_COLORS.get(ntype, "#95A5A6")
        conf = node.get("confidence", 1.0)
        label_text = node.get("name", node.get("label", node.get("id", "")))
        icon_uri = TYPE_ICONS_B64.get(ntype, TYPE_ICONS_B64["UNKNOWN"])

        # Build connection list for tooltip (Task 2)
        conns = adjacency.get(node["id"], [])
        conn_html = ""
        if conns:
            conn_html = "<br><br><b>Connected to:</b><br>"
            shown = conns[:TOOLTIP_CAP]
            for nbr_name, rel_type, is_weak_e, _ in shown:
                rel_label = rel_type.replace("_", " ").title()
                if is_weak_e:
                    # Weak link — show in muted colour with tag
                    conn_html += (
                        f"<span style='color:#FFAB91;'>&#8226; {nbr_name} "
                        f"<i>({rel_label})</i></span><br>"
                    )
                else:
                    conn_html += f"&#8226; {nbr_name} ({rel_label})<br>"
            remainder = len(conns) - TOOLTIP_CAP
            if remainder > 0:
                conn_html += f"<i style='color:#FFD54F;'>+{remainder} more connections</i>"

        # NOTE: tooltip is stored as a plain string here.
        # Task 1 fix: a post-generation JS patch below converts every node's
        # title string into a real DOM <div> via innerHTML so vis-network
        # renders HTML instead of displaying escaped tag characters.
        tooltip = (
            f"<b>{icon_emoji} {label_text}</b><br>"
            f"Type: {ntype}<br>"
            f"Confidence: {conf:.0%}"
            f"{conn_html}"
        )
        card_label = f"<b>{icon_emoji}  {ntype}</b>\n\n{label_text}"
        net.add_node(
            node["id"],
            label=card_label,
            title=tooltip,
            color={
                "background": "#FDFBF7",
                "border": color,
                "highlight": {"border": color, "background": "#F5EFE6"},
            },
        )

    # ── Add edges as red string lines ──────────────────────────────────────
    for edge in final_edges:
        rel = edge.get("type", "")
        conf = edge.get("confidence", 1.0)
        weight = edge.get("weight", 1.0)
        tooltip = f"<b>{rel}</b><br>Confidence: {conf:.0%}<br>Weight: {weight}"

        is_weak = rel in WEAK_REL_TYPES
        if is_weak:
            # Dashed dark maroon — weak/co-occurrence link
            edge_color = {"color": WEAK_EDGE_COLOR, "highlight": EDGE_HIGHLIGHT, "opacity": 0.75}
            width = 1.5
            dashes = [6, 4]
        else:
            # Solid crimson — strong direct evidence; thickness ∝ weight
            edge_color = {"color": STRONG_EDGE_COLOR, "highlight": EDGE_HIGHLIGHT, "opacity": 1.0}
            width = min(max(weight * 2.5, 2.0), 6.0)
            dashes = False

        net.add_edge(
            edge["source"], edge["target"],
            title=tooltip,
            width=width,
            color=edge_color,
            dashes=dashes,
        )

    # ── Generate HTML and inject corkboard CSS ─────────────────────────────
    # Bug 3 fix: on Windows, NamedTemporaryFile cannot be read while open.
    # Write to a named path, close the handle first, then read.
    tmp_path = os.path.join(tempfile.gettempdir(), f"pyvis_{os.getpid()}.html")
    try:
        net.save_graph(tmp_path)
        html = Path(tmp_path).read_text(encoding="utf-8")
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    # Patch the corkboard background into the rendered HTML.
    # PyVis sets bgcolor on the Network(), but we need the CSS to cover the
    # full canvas area and add the SVG noise texture overlay.
    CORKBOARD_CSS = """
<style>
/* ── Detective Evidence Board — Corkboard Background ─────────────────── */
body, html {
    margin: 0; padding: 0;
    background:
        url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='200' height='200'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.75' numOctaves='4' stitchTiles='stitch'/%3E%3CfeColorMatrix type='saturate' values='0'/%3E%3C/filter%3E%3Crect width='200' height='200' filter='url(%23n)' opacity='0.04'/%3E%3C/svg%3E"),
        linear-gradient(160deg, #F5EFE6 0%, #E8D8C2 100%);
}
#mynetwork, .card {
    background:
        url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='200' height='200'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.75' numOctaves='4' stitchTiles='stitch'/%3E%3CfeColorMatrix type='saturate' values='0'/%3E%3C/filter%3E%3Crect width='200' height='200' filter='url(%23n)' opacity='0.04'/%3E%3C/svg%3E"),
        linear-gradient(160deg, #F5EFE6 0%, #E8D8C2 100%) !important;
    border: 1px solid #DED5C8 !important;
    border-radius: 4px;
}
/* vis canvas must be transparent so the body texture shows through */
canvas { background: transparent !important; }
</style>
"""
    html = html.replace("</head>", CORKBOARD_CSS + "</head>", 1)

    # ── Task 1: Fix HTML tooltip rendering ────────────────────────────────
    # PyVis JSON-encodes the title string, so '<b>' becomes '&lt;b&gt;' and
    # vis-network displays literal tag characters instead of rendering HTML.
    # Fix: inject a JS snippet that runs AFTER network initialisation and
    # replaces every node's title (a string) with a real DOM <div> element
    # (set via innerHTML). vis-network renders DOM elements as HTML.
    # Edge titles get the same treatment.
    TOOLTIP_FIX_JS = """
<script>
// ── Tooltip HTML fix: convert title strings to DOM elements ────────────
// vis-network only renders HTML in tooltips when title is a DOM node.
// This runs after the network is created and patches all titles.
(function patchTooltips() {
    function waitForNetwork(attempts) {
        if (typeof network === 'undefined' || !network.body) {
            if (attempts > 0) setTimeout(function(){ waitForNetwork(attempts - 1); }, 150);
            return;
        }
        // Patch node titles
        var nodeUpdates = [];
        network.body.data.nodes.forEach(function(node) {
            if (typeof node.title === 'string' && node.title.length > 0) {
                var div = document.createElement('div');
                div.style.cssText = [
                    'background:#FDFBF7',
                    'color:#3E2723',
                    'padding:10px 14px',
                    'border-radius:8px',
                    'border:1px solid #DED5C8',
                    'font-family:Inter,Segoe UI,sans-serif',
                    'font-size:13px',
                    'line-height:1.6',
                    'max-width:260px',
                    'box-shadow:0 4px 16px rgba(62,39,35,0.15)'
                ].join(';');
                div.innerHTML = node.title;
                nodeUpdates.push({id: node.id, title: div});
            }
        });
        if (nodeUpdates.length) network.body.data.nodes.update(nodeUpdates);

        // Patch edge titles
        var edgeUpdates = [];
        network.body.data.edges.forEach(function(edge) {
            if (typeof edge.title === 'string' && edge.title.length > 0) {
                var div = document.createElement('div');
                div.style.cssText = [
                    'background:#FDFBF7',
                    'color:#9E5748',
                    'padding:8px 12px',
                    'border-radius:6px',
                    'border:1px solid #C1A598',
                    'font-family:Inter,Segoe UI,sans-serif',
                    'font-size:12px',
                    'line-height:1.5',
                    'box-shadow:0 2px 8px rgba(62,39,35,0.1)'
                ].join(';');
                div.innerHTML = edge.title;
                edgeUpdates.push({id: edge.id, title: div});
            }
        });
        if (edgeUpdates.length) network.body.data.edges.update(edgeUpdates);
    }
    // Wait up to 3 seconds for vis-network to initialise
    waitForNetwork(20);
})();
</script>
"""
    html = html.replace("</body>", TOOLTIP_FIX_JS + "</body>", 1)
    return html


# ===========================================================================
# Streamlit page setup  (M6.1)
# ===========================================================================

st.set_page_config(
    page_title="Investigator Dashboard — PS 26152",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Inline CSS — M6.10 polish
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* Global */
header[data-testid="stHeader"] { display: none !important; }
body, [data-testid="stApp"] {
    font-family: 'Inter', 'Geist', 'Manrope', -apple-system, BlinkMacSystemFont, sans-serif;
    background-color: #F6F1E8 !important;
    color: #211C18 !important;
}

/* Base header and text colors */
h1, h2, h3, h4, p, span, div {
    /* color: #211C18; */
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #F6F1E8 !important;
    border-right: 1px solid #DED5C8;
    box-shadow: none;
}
[data-testid="stSidebar"] * { 
    color: #66584C !important; 
    font-size: 15px;
}
[data-testid="stSidebar"] .stButton > button {
    background: transparent;
    border: none;
    color: #211C18 !important;
    width: 100%;
    font-weight: 500;
    font-size: 15px !important;
    text-align: left;
    padding-left: 0;
    box-shadow: none;
    transition: all 0.2s ease;
}
[data-testid="stSidebar"] .stButton > button * {
    color: #211C18 !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background-color: transparent !important;
}
[data-testid="stSidebar"] .stButton > button:hover * {
    color: #3A2A22 !important;
}

/* Primary Streamlit Buttons */
button[kind="primary"], button[kind="primaryFormSubmit"] {
    background-color: #2B211C !important;
    color: #F7F2E8 !important;
    border: none !important;
    border-radius: 4px !important;
    font-size: 15px !important;
    font-weight: 500 !important;
    padding: 8px 24px !important;
    box-shadow: none !important;
    transition: all 0.2s ease !important;
}
button[kind="primary"] *, button[kind="primaryFormSubmit"] * {
    color: #F7F2E8 !important;
}
button[kind="primary"]:hover, button[kind="primaryFormSubmit"]:hover {
    transform: translateX(3px) !important;
    background-color: #3A2A22 !important;
}
button[kind="primary"]:hover *, button[kind="primaryFormSubmit"]:hover * {
    color: #F7F2E8 !important;
}
button[kind="primary"]:disabled, button[kind="primaryFormSubmit"]:disabled {
    background-color: #CFC5B7 !important;
    transform: none !important;
    cursor: not-allowed !important;
}
button[kind="primary"]:disabled *, button[kind="primaryFormSubmit"]:disabled * {
    color: #6B5D50 !important;
}

/* Secondary Buttons */
button[kind="secondary"] {
    background-color: transparent !important;
    color: #211C18 !important;
    border: 1px solid #DED5C8 !important;
    border-radius: 4px !important;
    font-size: 15px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease;
}
button[kind="secondary"] * {
    color: #211C18 !important;
}
button[kind="secondary"]:hover {
    background-color: rgba(58, 42, 34, 0.04) !important;
}
button[kind="secondary"]:disabled {
    border-color: #E9E1D5 !important;
    background-color: transparent !important;
    cursor: not-allowed !important;
}
button[kind="secondary"]:disabled * {
    color: #908271 !important;
}

/* Inputs / Text Areas / Forms */
.stTextInput > div > div > input, .stTextArea > div > div > textarea {
    background-color: transparent !important;
    border: none !important;
    border-bottom: 1px solid #DED5C8 !important;
    border-radius: 0 !important;
    color: #211C18 !important;
    padding-left: 0 !important;
    padding-bottom: 8px !important;
    font-size: 15px !important;
}
.stTextInput > div > div > input:focus, .stTextArea > div > div > textarea:focus {
    border-bottom: 1px solid #3A2A22 !important;
    box-shadow: none !important;
}
div[data-testid="stForm"] {
    border: none !important;
    background: transparent !important;
    padding: 0 !important;
}

/* Upload dropzone styling */
[data-testid="stFileUploadDropzone"] {
    border: 1px solid #DED5C8 !important;
    background-color: #F6F1E8 !important;
    border-radius: 4px !important;
    transition: all 0.2s ease;
    padding: 32px !important;
}
[data-testid="stFileUploadDropzone"]:hover {
    border-color: #C0B5A6 !important;
    background-color: #F0EAE1 !important;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 64px;
    background: transparent;
    padding: 0 0 16px 0;
}
.stTabs [data-baseweb="tab"] {
    background-color: transparent;
    border: none;
    border-radius: 0;
    font-weight: 500;
    font-size: 16px;
    letter-spacing: 0.02em;
    padding: 8px 0;
    color: #6B5D50;
    transition: all 0.2s;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #3A2A22;
}
.stTabs [aria-selected="true"] {
    background-color: transparent !important;
    color: #2B211C !important;
    border-bottom: 3px solid #2B211C !important;
}

/* Multiselect / Entity Chips - Covers Streamlit versions 1.20 to 1.36+ */
div[data-testid="stMultiSelect"] [data-baseweb="tag"],
div[data-testid="stMultiSelect"] [data-testid="stMultiSelectTag"],
.st-key-graph_entity_types [data-testid="stMultiSelectTag"] {
    background-color: #E9E1D5 !important;
    color: #2B211C !important;
    border-radius: 4px !important;
    padding: 4px 10px !important;
    font-weight: 500 !important;
    border: 1px solid #DDD3C5 !important;
}

div[data-testid="stMultiSelect"] [data-baseweb="tag"] span,
div[data-testid="stMultiSelect"] [data-testid="stMultiSelectTag"] span {
    color: #2B211C !important;
}

div[data-testid="stMultiSelect"] [data-baseweb="tag"] svg,
div[data-testid="stMultiSelect"] [data-testid="stMultiSelectTag"] svg {
    color: #6B5D50 !important;
}

div[data-testid="stMultiSelect"] [data-baseweb="tag"] svg:hover,
div[data-testid="stMultiSelect"] [data-testid="stMultiSelectTag"] svg:hover {
    color: #2B211C !important;
}

/* Inline error */
.inline-error {
    color: #EF4444;
    font-size: 13px;
    margin-top: -12px;
    margin-bottom: 16px;
}

/* Metric cards */
div[data-testid="metric-container"] {
    background: transparent;
    border: none;
    border-radius: 0;
    padding: 0;
    box-shadow: none;
}
div[data-testid="metric-container"] label {
    color: #6B5D50 !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color: #2B211C !important;
    font-weight: 500 !important;
    font-size: 26px !important;
}

/* Chat bubbles */
.chat-investigator {
    background: rgba(58, 42, 34, 0.04);
    color: #211C18;
    border-radius: 4px;
    padding: 12px 16px;
    margin: 6px 0;
    max-width: 85%;
    margin-left: auto;
    font-size: 15px;
    border: none;
}
.chat-system {
    background: #FFFFFF;
    color: #211C18;
    border-radius: 4px;
    padding: 12px 16px;
    margin: 6px 0;
    max-width: 85%;
    font-size: 15px;
    border: 1px solid #DED5C8;
    box-shadow: 0 4px 20px rgba(0,0,0,0.02);
}
.chat-label {
    font-size: 11px;
    font-weight: 600;
    color: #66584C;
    margin-bottom: 4px;
    letter-spacing: 0.1em;
}

/* Flag badges */
.flag-badge {
    display: inline-block;
    background: #F0EAE1;
    border: 1px solid #DED5C8;
    border-radius: 4px;
    padding: 4px 10px;
    font-size: 12px;
    font-weight: 500;
    margin: 2px;
    color: #3A2A22;
}

/* Priority score bar */
.priority-bar-outer {
    background: #E8E0D5;
    border-radius: 100px;
    height: 4px;
    width: 100%;
}
.priority-bar-inner {
    border-radius: 100px;
    height: 4px;
}

/* Human review alert */
.review-alert {
    background: transparent;
    color: #991B1B;
    border: 1px solid #DED5C8;
    border-radius: 4px;
    padding: 12px 16px;
    margin: 8px 0;
}

/* Relationship item */
.relationship-item {
    background: transparent;
    color: #211C18;
    border: none;
    border-left: 2px solid #DED5C8;
    padding: 10px 14px;
    margin: 6px 0;
    border-radius: 0;
    font-size: 14px;
    box-shadow: none;
}
</style>
""", unsafe_allow_html=True)


# ===========================================================================
# Session state initialization  (M6.1, M6.6)
# ===========================================================================

def _init_state():
    defaults = {
        "session_id": None,
        "case_id": None,
        "final_response": None,
        "graph_data": None,
        "key_players": None,
        "conversation_history": [],
        "active_tab": "Upload & Analyze",
        "search_results": [],
        "last_search_query": "",
        "date_filter_from": None,
        "date_filter_to": None,
        "demo_mode": False,
        "timeline_events": None,
        "uploaded_case_text": "",
        "uploaded_source_label": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ===========================================================================
# M6.2 helper — render the FINAL RESPONSE
# ===========================================================================

def render_final_response(response_dict: dict):
    """Render analysis result (summary, confidence, evidence, human-review flag)."""
    confidence = response_dict.get("confidence", 0.0)
    requires_review = response_dict.get("requires_human_review", False)

    # Human review alert banner
    if requires_review:
        st.markdown(
            '<div class="review-alert">'
            '<b>⚠️ Requires Human Review</b> — The automated analysis confidence is low. '
            'Please escalate this case to a senior investigator for manual verification.'
            '</div>',
            unsafe_allow_html=True,
        )

    # Confidence metric
    col1, col2, col3 = st.columns(3)
    with col1:
        color = confidence_color(confidence)
        label = confidence_label(confidence)
        st.markdown(f"""
        <div style="background:#f8f9fa;border-radius:8px;padding:16px;border:1px solid #dee2e6;text-align:center;">
            <div style="font-size:0.85em;color:#6B7280;font-weight:600;">Confidence Score</div>
            <div style="font-size:2.2em;font-weight:700;color:{color};">{confidence:.0%}</div>
            <div style="font-size:0.8em;color:{color};font-weight:500;">{label}</div>
        </div>
        """, unsafe_allow_html=True)
        with st.expander("ℹ️ What does this mean?"):
            st.caption("This score represents data connectivity — how strongly this document links to already known accounts, entities, or historical cases. It is not a prediction of guilt.")
    with col2:
        evidence_count = len(response_dict.get("evidence", []))
        st.markdown(f"""
        <div style="background:#f8f9fa;border-radius:8px;padding:16px;border:1px solid #dee2e6;text-align:center;">
            <div style="font-size:0.85em;color:#6B7280;font-weight:600;">Evidence References</div>
            <div style="font-size:2.2em;font-weight:700;color:#3B82F6;">{evidence_count}</div>
            <div style="font-size:0.8em;color:#6B7280;">items cited</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        review_icon = "⚠️" if requires_review else "✅"
        review_text = "Requires Review" if requires_review else "Threshold Met"
        review_color = "#E74C3C" if requires_review else "#2ECC71"
        st.markdown(f"""
        <div style="background:#f8f9fa;border-radius:8px;padding:16px;border:1px solid #dee2e6;text-align:center;">
            <div style="font-size:0.85em;color:#6B7280;font-weight:600;">Review Status</div>
            <div style="font-size:1.8em;">{review_icon}</div>
            <div style="font-size:0.8em;color:{review_color};font-weight:600;">{review_text}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Summary text
    st.markdown("#### 📋 Analysis Summary")
    summary_text = response_dict.get("response_text", "")
    # Check for banned words in summary (M6 language enforcement — flag internally)
    found_banned = [w for w in BANNED_WORDS if w.lower() in summary_text.lower()]
    if found_banned:
        st.warning(f"⚠️ Internal language check: summary contains flagged words: {found_banned}. Please review.")
    st.info(summary_text)

    # Linked Entities & Relationships
    edges = st.session_state.graph_data.get("edges", []) if getattr(st.session_state, "graph_data", None) else []
    if edges:
        with st.expander(f"🔗 Linked Entities & Relationships ({len(edges)} links)", expanded=True):
            # Build a lookup for node names
            nodes = st.session_state.graph_data.get("nodes", [])
            node_names = {n["id"]: n.get("name", n.get("label", n["id"])) for n in nodes}
            
            # Dropdown to filter by entity
            unique_names = sorted(list({name for name in node_names.values() if name}))
            selected_entity = st.selectbox("Filter relationships by Entity:", options=["All"] + unique_names)
            
            strong_edges = [e for e in edges if e.get("type") != "APPEARS_IN_CASE"]
            weak_edges = [e for e in edges if e.get("type") == "APPEARS_IN_CASE"]
            
            # Fallback logic: if no strong edges exist but weak ones do, show the weak ones
            display_edges = strong_edges
            showing_fallback = False
            if not strong_edges and weak_edges:
                display_edges = weak_edges
                showing_fallback = True
                st.markdown(
                    "<div style='margin-bottom:12px;'>"
                    "<span style='color:#9CA3AF;font-size:0.95em;font-style:italic;'>"
                    "No strong relationships found. Showing co-occurrence (weak) links instead."
                    "</span></div>", 
                    unsafe_allow_html=True
                )
            
            displayed_edges = 0
            for edge in display_edges:
                src_id = edge["source"]
                tgt_id = edge["target"]
                
                # Resolve names
                src_name = node_names.get(src_id) or src_id
                tgt_name = node_names.get(tgt_id) or tgt_id
                
                if selected_entity != "All" and selected_entity not in (src_name, tgt_name):
                    continue
                
                displayed_edges += 1
                rel = edge["type"].replace("_", " ").title()
                src_rec = edge.get("source_record", "Unknown")
                
                if showing_fallback:
                    st.markdown(
                        f'<div class="relationship-item" style="color:#6B7280;">🔗 **{src_name}** ⟷ **{rel}** ⟷ **{tgt_name}** <br><span style="font-size:0.85em;color:#9CA3AF;">(weak/co-occurrence link — same document only: {src_rec})</span></div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="relationship-item">🔗 **{src_name}** ⟷ **{rel}** ⟷ **{tgt_name}** (via {src_rec})</div>',
                        unsafe_allow_html=True
                    )
            
            if displayed_edges == 0:
                st.caption(f"No relationships found involving '{selected_entity}'.")
    else:
        st.caption("No relationships recorded for this analysis.")

    st.caption(
        f"*Session ID: {response_dict.get('session_id', 'N/A')}  |  "
        f"All findings require investigator verification before any action is taken.*"
    )


# ===========================================================================
# M6.3 — Graph visualization tab
# ===========================================================================

def render_graph_tab():
    st.markdown("### 🕸️ Relationship Network")

    if not st.session_state.graph_data:
        st.info("Upload and analyze a case first to see the network graph.")
        return

    graph_data = st.session_state.graph_data
    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])

    if not nodes:
        st.warning("No network data found for this case.")
        return

    # --- M6.4 Date & View filter controls ---
    st.markdown("#### 🗓️ Filters & Views")
    
    all_types = list(TYPE_ICONS.keys())
    selected_types = st.multiselect(
        "Filter Entity Types to Display", 
        options=all_types, 
        default=[t for t in all_types if t != "UNKNOWN"], 
        key="graph_entity_types"
    )

    col_from, col_to, col_orgs, col_edges = st.columns([2, 2, 2, 2])
    with col_from:
        date_from = st.date_input("From date", value=None, key="graph_date_from")
    with col_to:
        date_to = st.date_input("To date", value=None, key="graph_date_to")
    with col_orgs:
        st.markdown("<br>", unsafe_allow_html=True)
        show_all_orgs = st.checkbox("Show all Organizations", value=False, help="Include all organizations, not just highly connected ones.", key="graph_show_orgs")
    with col_edges:
        st.markdown("<br>", unsafe_allow_html=True)
        show_weak_links = st.checkbox("Show 'Appears In Case' links", value=False, help="Include weak document co-occurrence links.", key="graph_show_edges")
        
    def reset_filters():
        for k in ["graph_date_from", "graph_date_to", "graph_show_orgs", "graph_show_edges", "graph_entity_types"]:
            if k in st.session_state:
                del st.session_state[k]

    if st.button("Reset Filters", key="graph_reset_date", on_click=reset_filters):
        pass

    filter_from = datetime.combine(date_from, datetime.min.time()) if date_from else None
    filter_to = datetime.combine(date_to, datetime.max.time()) if date_to else None

    # Stats strip
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Entities", len(nodes))
    with col_b:
        st.metric("Relationships", len(edges))

    # Legend
    st.markdown("""
    **Legend:**
    🧑 Person &nbsp;&nbsp; 📱 Phone &nbsp;&nbsp; 🏦 Account &nbsp;&nbsp;
    📍 Location &nbsp;&nbsp; 🚗 Vehicle &nbsp;&nbsp; 🏢 Organisation &nbsp;&nbsp; ❓ Unknown
    """)

    # Build and embed graph
    with st.spinner("Rendering network graph…"):
        try:
            html = build_pyvis_html(graph_data, filter_from, filter_to, show_all_orgs, show_weak_links, selected_types)
            # Task 4: height increased from 600 → 880 to match larger canvas (850px + chrome)
            st.components.v1.html(html, height=880, scrolling=False)
        except Exception as e:
            st.error(f"Graph rendering failed: {e}")
            return

    # Node table below graph
    with st.expander("📋 Show Entity Table"):
        df_nodes = pd.DataFrame([
            {
                "Entity ID": n["id"],
                "Name": n.get("label", n["id"]),
                "Type": TYPE_ICONS.get(n.get("type", "UNKNOWN"), "") + " " + n.get("type", ""),
                "Confidence": f"{n.get('confidence', 1.0):.0%}",
            }
            for n in nodes
        ])
        st.dataframe(df_nodes, use_container_width=True, hide_index=True)


# ===========================================================================
# M6.4 — Search & filter tab
# ===========================================================================

def render_search_tab():
    st.markdown('<div class="section-header">🔎 Case Search</div>',
                unsafe_allow_html=True)
    active_case = st.session_state.get("case_id")
    case_text = st.session_state.get("uploaded_case_text", "")
    if not active_case or not case_text:
        st.info("Upload and analyze a case first. Search is limited to the active uploaded case.")
        return
    st.markdown(
        f"Search within uploaded case `{active_case}`. Results include exact line provenance "
        "from the active case document only."
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
                    results = search_local(
                        query,
                        top_k=top_k,
                        case_text=case_text,
                        source_label=st.session_state.get("uploaded_source_label", "uploaded_case.txt"),
                    )
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
                        source_path = Path(r["file_path"])
                        if source_path.exists():
                            with open(source_path, "r", encoding="utf-8", errors="replace") as f:
                                lines = f.readlines()
                        else:
                            lines = case_text.splitlines(keepends=True)

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
# M6.4.5 — Video Evidence Tab (Semantic Analysis integration)
# ===========================================================================

def render_video_analysis_tab():
    st.markdown('<div class="section-header">🎥 Video Evidence Analysis</div>',
                unsafe_allow_html=True)
    st.markdown(
        "The Video Evidence Analysis tool runs as a completely separate application to ensure stability. "
        "It allows you to upload surveillance footage and run semantic queries across visual content, "
        "spoken audio, objects, and text in the video."
    )
    
    st.info("💡 **How to launch the Video Analysis Tool:**\n"
            "Open a new terminal and run:\n"
            "`python -m uvicorn semantic_analysis.integration.app:app --port 8001`")
            
    st.markdown(
        '<a href="http://localhost:8001" target="_blank" style="'
        'display: inline-block; padding: 10px 20px; background-color: #3A2A22; '
        'color: white; text-decoration: none; border-radius: 5px; font-weight: bold; '
        'margin-top: 20px;">'
        '↗️ Open Video Analysis Dashboard</a>',
        unsafe_allow_html=True
    )

    video_dir = REPO_ROOT / "data" / "videos"
    videos = sorted(p for p in video_dir.glob("*") if p.is_file()) if video_dir.exists() else []
    if videos:
        video = st.selectbox("Video file", videos, format_func=lambda p: p.name, key="video_download_file")
        st.download_button(
            "⬇️ Download video",
            data=video.read_bytes(),
            file_name=video.name,
            mime="video/mp4" if video.suffix.lower() == ".mp4" else "application/octet-stream",
            key="download_video",
        )
    else:
        st.caption("No processed video is available in the local case workspace yet.")

# ===========================================================================
# M6.5 — Key players view
# ===========================================================================

def render_key_players_tab():
    st.markdown('<div class="section-header">👥 Case Entities</div>',
                unsafe_allow_html=True)
    case_id = st.session_state.get("case_id")
    graph_data = st.session_state.get("graph_data") or {}
    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])
    if not case_id:
        st.info("Upload and analyze a case first. Entities are limited to the active uploaded case.")
        return
    if not nodes:
        st.info(f"No entities were extracted from case {case_id}.")
        return

    degrees = {n["id"]: 0 for n in nodes}
    for edge in edges:
        for endpoint in (edge.get("source"), edge.get("target")):
            if endpoint in degrees:
                degrees[endpoint] += 1
    rows = [{
        "Entity ID": n.get("id", ""),
        "Name": n.get("label", n.get("name", n.get("id", ""))),
        "Type": n.get("type", "UNKNOWN"),
        "Connections in case": degrees.get(n.get("id"), 0),
        "Confidence": f"{n.get('confidence', 1.0):.0%}",
    } for n in nodes]
    # Keep people visually prominent; the remaining case entities follow in
    # name order, independent of confidence.
    rows.sort(key=lambda row: (row["Type"] != "PERSON", row["Name"].casefold()))
    st.caption(f"{len(rows)} entities from uploaded case `{case_id}`")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
# ===========================================================================
# M6.6 — Follow-up conversation tab
# ===========================================================================

def render_conversation_tab():
    st.markdown("### 💬 Follow-Up Questions")

    if not st.session_state.session_id:
        st.info("Upload and analyze a case first to start a conversation.")
        return

    st.caption(
        f"Active session: `{st.session_state.session_id}` | "
        f"Case: `{st.session_state.case_id}` | "
        "All responses are analytical findings — verify before acting."
    )

    # Display conversation history
    history = st.session_state.conversation_history
    if history:
        st.markdown("#### Conversation History")
        for turn in history:
            role = turn.get("role", "")
            content = turn.get("content", "")
            if role == "investigator":
                st.markdown(
                    f'<div class="chat-label">🧑 You</div>'
                    f'<div class="chat-investigator">{content}</div>',
                    unsafe_allow_html=True,
                )
            elif role == "system":
                st.markdown(
                    f'<div class="chat-label">🤖 System Analysis</div>'
                    f'<div class="chat-system">{content}</div>',
                    unsafe_allow_html=True,
                )
        st.markdown("")

    # Input area
    st.markdown("#### Ask a Follow-Up Question")
    with st.form("followup_form", clear_on_submit=True):
        question = st.text_area(
            "Your question",
            placeholder=(
                "e.g. What are the known connections of P001? "
                "or: Are there any unusual patterns in the financial transactions?"
            ),
            height=80,
        )
        submitted = st.form_submit_button("Ask Question →")

    if submitted and question.strip():
        with st.spinner("Analyzing your question…"):
            try:
                req = FollowUpQuestion(question=question.strip())
                resp = call_m5(st.session_state.session_id, req)
                resp_dict = resp.to_dict() if hasattr(resp, "to_dict") else resp

                # Update state
                st.session_state.final_response = resp_dict
                st.session_state.conversation_history.append(
                    {"role": "investigator", "content": question.strip()}
                )
                st.session_state.conversation_history.append(
                    {"role": "system", "content": resp_dict.get("response_text", "")}
                )
                st.rerun()

            except Exception as e:
                st.error(
                    "The follow-up analysis could not be completed. "
                    "Please try again or escalate for manual review. "
                    f"(Detail: {type(e).__name__})"
                )
                logger.error(f"[UI] Follow-up question failed: {e}", exc_info=True)


# ===========================================================================
# Timeline Tab  (pixel-perfect re-implementation)
# ===========================================================================


def render_timeline_tab():
    # Resolve the active case before any cached-event or filter branch. Streamlit
    # reruns can preserve timeline_events while skipping the event builder.
    case_id = getattr(st.session_state, 'case_id', None) or "no_case"
    st.markdown('<div class="section-header">⏱️ Timeline Player</div>',
                unsafe_allow_html=True)
    st.markdown(
        "Animated playback of all case events (incidents, calls, transactions, FIR filings) "
        "in chronological order. Events are highlighted on the network graph."
    )

    # Build timeline events
    if getattr(st.session_state, 'timeline_events', None) is None:
        try:
            active_case_id = getattr(st.session_state, 'case_id', None)

            if active_case_id:
                from M6_feature.timeline_player import build_timeline, build_uploaded_case_timeline
                uploaded_text = st.session_state.get("uploaded_case_text", "")
                if uploaded_text:
                    # An uploaded case is authoritative. Do not search the
                    # historical FIR/CDR corpus by case ID: IDs can collide.
                    events = build_uploaded_case_timeline(
                        case_id=active_case_id,
                        case_text=uploaded_text,
                        source_label=st.session_state.get("uploaded_source_label", "uploaded_case.txt"),
                    )
                else:
                    graph_data = getattr(st.session_state, 'graph_data', {})
                    nodes = graph_data.get("nodes", []) if graph_data else []
                    valid_entities = set()
                    for node in nodes:
                        for key in ("id", "label", "name"):
                            value = node.get(key)
                            if value:
                                valid_entities.add(str(value))
                    events = build_timeline(case_id=active_case_id, case_entities=valid_entities or None)
            else:
                events = []
                
            st.session_state.timeline_events = events
        except Exception as e:
            st.error(f"Failed to build timeline: {e}")
            events = []
    else:
        events = st.session_state.timeline_events

    if events:
        st.markdown(f"**{len(events)} events** from "
                    f"{events[0]['t'][:10]} to {events[-1]['t'][:10]}")

        # Date range filter
        event_start = datetime.fromisoformat(events[0]["t"][:10]).date()
        event_end = datetime.fromisoformat(events[-1]["t"][:10]).date()
        col1, col2 = st.columns(2)
        with col1:
            date_start = st.date_input("From Date",
                                       value=event_start,
                                       min_value=event_start,
                                       max_value=event_end,
                                       key=f"tl_start_{case_id}")
        with col2:
            date_end = st.date_input("To Date",
                                     value=event_end,
                                     min_value=event_start,
                                     max_value=event_end,
                                     key=f"tl_end_{case_id}")

        # Filter events by date range
        filtered = [
            e for e in events
            if str(date_start) <= e["t"][:10] <= str(date_end)
        ]

        # Event type filter
        event_types = st.multiselect(
            "Event Types",
            ["incident", "call", "transaction", "fir_filing"],
            default=["incident", "call", "transaction", "fir_filing"],
            key="tl_types",
        )
        filtered = [e for e in filtered if e["type"] in event_types]

        st.markdown(f"**Showing {len(filtered)} events** after filters")

        # Render timeline HTML
        # Height grows with event count: ~150px per event + 120px for controls.
        # scrolling=True lets the iframe scroll for very large event sets.
        # Min 400px (even for 1 event), max 6000px to cap iframe size.
        try:
            from M6_feature.timeline_ui import render_timeline_html
            timeline_html = render_timeline_html(filtered)
            iframe_height = max(400, min(6000, len(filtered) * 160 + 120))
            st.components.v1.html(timeline_html, height=iframe_height, scrolling=True)
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
        case_id = getattr(st.session_state, 'case_id', None)
        if not case_id:
            st.info("No active case selected.")
        else:
            st.info("No timeline events found for this case.")
# ===========================================================================
# M6.7 — Export tab
# ===========================================================================

def render_export_tab():
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
                    case_id = getattr(st.session_state, 'case_id', None)
                    if not case_id:
                        raise ValueError("No active case selected. Please upload or select a case first.")

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
# ===========================================================================
# M6.1 + M6.2 — Upload & Analyze tab
# ===========================================================================

def render_upload_tab():
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    col_left, col_right = st.columns([1.5, 1])
    
    with col_left:
        st.markdown("<div style='font-size: 11px; font-weight: 500; letter-spacing: 0.12em; color: #6B5D50; margin-bottom: 8px;'>CASE ANALYSIS &nbsp;|&nbsp; 01 / DOCUMENT INGESTION</div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size: 56px; font-weight: 500; line-height: 1.05; margin-bottom: 12px; letter-spacing: -0.03em; max-width: 500px; color: #2B211C;'>UPLOAD & ANALYZE<br>CASE DOCUMENT</div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size: 16px; color: #6B5D50; margin-bottom: 48px; max-width: 400px;'>Turn case documents into an investigative network.</div>", unsafe_allow_html=True)

        with st.form("upload_form"):
            case_id_input = st.text_input(
                "CASE REFERENCE",
                placeholder="FIR-2026-00931",
                max_chars=30,
            )
            
            st.markdown("<div style='margin-top: 32px; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.12em; color: #211C18; margin-bottom: 8px;'>DOCUMENT</div>", unsafe_allow_html=True)
            st.markdown("<div style='font-size: 13px; color: #66584C; margin-bottom: 16px;'>↑ Drop case files here &nbsp;·&nbsp; TXT / PDF &nbsp;·&nbsp; up to 200MB</div>", unsafe_allow_html=True)
            
            uploaded_files = st.file_uploader(
                "Upload Dropzone",
                label_visibility="collapsed",
                type=["txt", "pdf"],
                help="Plain text (.txt) is recommended. PDF text will be extracted.",
                accept_multiple_files=True,
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
            submit_btn = st.form_submit_button("Analyze →", type="primary")

    with col_right:
        st.markdown("<div style='padding-left: 64px;'>", unsafe_allow_html=True)
        status = get_backend_status()
        
        # Subtle Japanese-inspired detail
        st.markdown("<div style='font-size: 11px; letter-spacing: 0.1em; color: #6B5D50; text-align: right; margin-bottom: 40px;'>分析 / ANALYSIS</div>", unsafe_allow_html=True)
        
        st.markdown("<div style='font-size:12px; font-weight:600; color:#6B5D50; text-transform:uppercase; letter-spacing:0.12em; margin-bottom:12px;'>SYSTEM</div>", unsafe_allow_html=True)
        
        if status["use_real_modules"]:
            st.markdown("<div style='font-size:14px; font-weight:500; color:#2B211C; margin-bottom:32px;'>● OPERATIONAL</div>", unsafe_allow_html=True)
            
            st.markdown(f"<div style='font-size:28px; font-weight:500; color:#2B211C; letter-spacing: -0.02em;'>{status['graph_nodes']:,}</div>", unsafe_allow_html=True)
            st.markdown("<div style='font-size:11px; color:#6B5D50; text-transform:uppercase; letter-spacing:0.12em; margin-bottom:24px;'>NODES</div>", unsafe_allow_html=True)
            
            st.markdown(f"<div style='font-size:28px; font-weight:500; color:#2B211C; letter-spacing: -0.02em;'>{status['graph_edges']:,}</div>", unsafe_allow_html=True)
            st.markdown("<div style='font-size:11px; color:#6B5D50; text-transform:uppercase; letter-spacing:0.12em; margin-bottom:32px;'>CONNECTIONS</div>", unsafe_allow_html=True)
            
            st.markdown(f"<div style='font-size:14px; font-weight:500; color:#2B211C; margin-bottom:6px;'>M4 RAG &nbsp;&nbsp;<span style='color:#7A8064;font-weight:400; font-size:11px;'>{'READY' if status['pipeline_loaded'] else 'OFFLINE'}</span></div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:14px; font-weight:500; color:#2B211C;'>M3 PATTERNS &nbsp;&nbsp;<span style='color:#7A8064;font-weight:400; font-size:11px;'>{'READY' if status['m3_flags_available'] else 'OFFLINE'}</span></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='font-size:13px; font-weight:500; color:#A66A4C; margin-bottom:32px;'>● MOCK MODE</div>", unsafe_allow_html=True)
            st.markdown("<div style='font-size:13px; color:#66584C;'>Set CRIMINAL_USE_REAL_MODULES=1 for full backend.</div>", unsafe_allow_html=True)
            
        st.markdown("</div>", unsafe_allow_html=True)

    if submit_btn:
        # --- Validation (M6 FR11 — pre-processing validation) ---
        if not case_id_input.strip():
            with col_left:
                st.markdown("<div class='inline-error'>Please enter a case reference number.</div>", unsafe_allow_html=True)
            return
        if not uploaded_files:
            with col_left:
                st.markdown("<div class='inline-error'>Please upload at least one case document (.txt or .pdf).</div>", unsafe_allow_html=True)
            return

        # --- Extract text ---
        case_texts = []
        for file in uploaded_files:
            if file.type == "text/plain" or file.name.endswith(".txt"):
                case_texts.append(file.read().decode("utf-8", errors="replace"))
            elif file.name.endswith(".pdf"):
                try:
                    import io as _io
                    raw = file.read()
                    try:
                        import pypdf
                        reader = pypdf.PdfReader(_io.BytesIO(raw))
                        case_texts.append("\n".join(page.extract_text() or "" for page in reader.pages))
                    except ImportError:
                        decoded = raw.decode("latin-1", errors="replace")
                        case_texts.append("".join(c for c in decoded if c.isprintable() or c in "\n\r\t"))
                        st.warning(f"PDF text extraction for {file.name} is limited without pypdf.")
                except Exception as e:
                    st.error(f"Could not read the PDF file {file.name}: {e}")
                    return
            else:
                st.error(f"Unsupported file type: {file.name}")
                return
        
        case_text = "\n\n--- NEXT DOCUMENT ---\n\n".join(case_texts)

        if not case_text.strip():
            st.error(
                "The uploaded file appears to be empty or contains no readable text. "
                "Please check the file and try again."
            )
            return

        case_id = case_id_input.strip().upper()

        # --- Call M5 ---
        with st.spinner(f"Analyzing case {case_id}… (this may take up to 60 seconds for the first case)"):
            try:
                req = NewCaseUpload(case_id=case_id, case_text=case_text)
                resp = call_m5(None, req)
                resp_dict = resp.to_dict() if hasattr(resp, "to_dict") else resp
            except Exception as e:
                st.error(
                    "Analysis could not complete. Please try again or escalate for manual review. "
                    f"(Error: {type(e).__name__}: {e})"
                )
                logger.error(f"[UI] M5 call failed for case {case_id}: {e}", exc_info=True)
                return

        # --- Bug 2 fix: build graph_data from entities extracted from the uploaded doc ---
        # Previously this called call_m2_subgraph(case_id) which falls back to the full
        # 247-node static graph when case_id isn't in the pre-built index. That caused
        # entities from unrelated FIRs (Fatima Begum, Kolkata etc.) to appear.
        extracted_entities = resp_dict.get("extracted_entities", [])
        if extracted_entities:
            # Build graph_data from extracted entities + neighbors from the real graph
            logger.info(f"[UI] Using {len(extracted_entities)} extracted entities for graph display")
            # Nodes from extraction
            graph_nodes = [
                {
                    "id": e["entity_id"],
                    "label": e.get("name", e["entity_id"]),
                    "type": e.get("type", "UNKNOWN"),
                    "confidence": e.get("confidence", 0.75),
                }
                for e in extracted_entities
            ]
            # Keep only relationships explicitly recorded for this case.
            # Shared entity IDs must not pull historical relationships into
            # the uploaded case view.
            graph_edges = []
            try:
                from M6.backend_calls import _get_graph
                real_g = _get_graph()
                if real_g is not None:
                    extracted_ids = {e["entity_id"] for e in extracted_entities}
                    for u, v, data in real_g.edges(data=True):
                        if (u in extracted_ids and v in extracted_ids
                                and data.get("source_record") == case_id):
                            graph_edges.append({
                                "source": u, "target": v,
                                "type": data.get("relationship", ""),
                                "confidence": data.get("confidence", 1.0),
                                "weight": data.get("weight", 1.0),
                                "timestamp": data.get("timestamp", ""),
                                "source_record": data.get("source_record", "Unknown"),
                            })
            except Exception as eg:
                logger.warning(f"[UI] Could not pull graph edges for extracted entities: {eg}")
                
            # Synthesize co-occurrence edges for the newly extracted entities
            # so they are connected by APPEARS_IN_CASE in the graph
            extracted_ids_list = list({e["entity_id"] for e in extracted_entities})
            for i in range(len(extracted_ids_list)):
                for j in range(i + 1, len(extracted_ids_list)):
                    # Avoid adding duplicates if the real graph somehow already had them
                    if not any((e["source"] == extracted_ids_list[i] and e["target"] == extracted_ids_list[j]) or 
                               (e["source"] == extracted_ids_list[j] and e["target"] == extracted_ids_list[i]) 
                               for e in graph_edges):
                        graph_edges.append({
                            "source": extracted_ids_list[i],
                            "target": extracted_ids_list[j],
                            "type": "APPEARS_IN_CASE",
                            "confidence": 1.0,
                            "weight": 1.0,
                            "timestamp": "",
                            "source_record": case_id,
                        })

            graph_data = {"nodes": graph_nodes, "edges": graph_edges}
        else:
            # Fallback: use the subgraph query (for follow-ups or when M1 extraction was skipped)
            with st.spinner("Loading network data…"):
                try:
                    graph_data = call_m2_subgraph(None, case_id)
                except Exception as e:
                    logger.warning(f"[UI] Graph load failed: {e}")
                    graph_data = {"nodes": [], "edges": []}

        with st.spinner("Ranking key entities…"):
            try:
                key_players = call_m3_key_players()
            except Exception as e:
                logger.warning(f"[UI] Key players load failed: {e}")
                key_players = []

        # --- Store in session state ---
        new_session_id = resp_dict.get("session_id", None)
        st.session_state.session_id = new_session_id
        st.session_state.case_id = case_id
        st.session_state.final_response = resp_dict
        st.session_state.graph_data = graph_data
        st.session_state.key_players = key_players
        st.session_state.timeline_events = None
        st.session_state.search_results = []
        st.session_state.last_search_query = ""
        st.session_state.uploaded_case_text = case_text
        st.session_state.uploaded_source_label = (
            uploaded_files[0].name if len(uploaded_files) == 1 else f"{len(uploaded_files)} uploaded files"
        )
        st.session_state.conversation_history = [
            {"role": "investigator", "content": f"[Uploaded: {case_id}]"},
            {"role": "system", "content": resp_dict.get("response_text", "")},
        ]
        st.session_state._just_analyzed = True
        st.success(f"✅ Case {case_id} analyzed. See tabs below for results.")
        st.rerun()

    # --- Show results if available ---
    if st.session_state.final_response:
        if st.session_state.get("auto_switch_graph") and st.session_state.get("_just_analyzed"):
            st.session_state._just_analyzed = False
            import streamlit.components.v1 as components
            components.html(
                """
                <script>
                window.parent.document.querySelectorAll('button[data-baseweb="tab"]')[1].click();
                </script>
                """,
                height=0
            )

        st.markdown("---")
        st.markdown(f"#### Results — Case `{st.session_state.case_id}`")
        render_final_response(st.session_state.final_response)

        # Bug 2 fix: show entity table drawn from extracted_entities (doc-specific)
        extracted = st.session_state.final_response.get("extracted_entities", [])
        if extracted:
            st.markdown("#### 🏷️ Entities Extracted from This Document")
            filtered = [e for e in extracted if len(e.get("name", "")) > 2]
            if filtered:
                df_extracted = pd.DataFrame([
                    {
                        "Entity ID": e["entity_id"],
                        "Name": e.get("name", e["entity_id"]),
                        "Type": TYPE_ICONS.get(e.get("type", "UNKNOWN"), "❓") + " " + e.get("type", ""),
                        "Confidence": f"{e.get('confidence', 0.75):.0%}",
                    }
                    for e in filtered
                ])
                st.dataframe(df_extracted, use_container_width=True, hide_index=True)
            else:
                st.caption("No named entities were extracted from this document.")


# ===========================================================================
# M6.8 + M6.9 — Sidebar controls
# ===========================================================================

def render_sidebar():
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    st.sidebar.markdown("<div style='font-size: 13px; color:#211C18; font-weight:600; letter-spacing:0.12em;'>PS 26152</div>", unsafe_allow_html=True)
    st.sidebar.markdown("<div style='font-size: 18px; font-weight:500; color:#211C18; margin-top:8px; line-height:1.2; letter-spacing: -0.02em;'>Investigator<br>Dashboard</div>", unsafe_allow_html=True)
    st.sidebar.markdown("<div style='font-size: 15px; color:#66584C; margin-top:12px;'>AI-Powered Network<br>Analysis System</div>", unsafe_allow_html=True)
    st.sidebar.markdown("<hr style='border:none; border-top:1px solid #DED5C8; margin:32px 0;'>", unsafe_allow_html=True)
    
    st.sidebar.markdown("<div style='font-size:12px; font-weight:600; color:#66584C; text-transform:uppercase; letter-spacing:0.12em; margin-bottom:16px;'>DEMO</div>", unsafe_allow_html=True)
    if st.sidebar.button("Run Demo →", key="run_demo"):
        st.session_state.demo_mode = True
        st.rerun()

    st.sidebar.markdown("<hr style='border:none; border-top:1px solid #DED5C8; margin:32px 0;'>", unsafe_allow_html=True)
    
    # Active session info
    if st.session_state.session_id:
        st.sidebar.markdown("<div style='font-size:12px; font-weight:600; color:#66584C; text-transform:uppercase; letter-spacing:0.12em; margin-bottom:8px;'>ACTIVE CASE</div>", unsafe_allow_html=True)
        st.sidebar.markdown(f"<div style='font-size: 15px; font-weight: 500; color:#211C18; margin-bottom:16px;'>{st.session_state.case_id}</div>", unsafe_allow_html=True)
        if st.sidebar.button("Clear Session"):
            for key in ["session_id", "case_id", "final_response", "graph_data",
                        "key_players", "conversation_history", "search_results",
                        "uploaded_case_text", "uploaded_source_label"]:
                st.session_state[key] = None if key not in ["conversation_history", "search_results"] else []
            st.rerun()
        st.sidebar.markdown("<hr style='border:none; border-top:1px solid #DED5C8; margin:32px 0;'>", unsafe_allow_html=True)

    st.sidebar.markdown("<div style='font-size:12px; font-weight:600; color:#66584C; text-transform:uppercase; letter-spacing:0.12em; margin-bottom:16px;'>SETTINGS</div>", unsafe_allow_html=True)
    st.session_state["auto_switch_graph"] = st.sidebar.checkbox(
        "Auto-switch to Graph", 
        value=False
    )
    
    st.sidebar.markdown("<br><br><br>", unsafe_allow_html=True)
    st.sidebar.markdown(
        "<div style='font-size: 13px; color: #66584C; line-height: 1.5;'>⚖️ All findings require verification.<br>Not for legal liability.</div>", 
        unsafe_allow_html=True
    )


# ===========================================================================
# M6.9 — Demo mode auto-script
# ===========================================================================

def run_demo_script():
    """
    Executes the full intended demo flow automatically:
    Upload → Analyze → View graph → Ask follow-up → Ready for export.
    Displayed inline so a judge can follow along step by step.
    """
    st.markdown("## 🎬 Demo Script: Full Investigator Workflow")
    st.info(
        "This demo runs the complete workflow: case upload → analysis → "
        "network graph → follow-up question. Export is available on the Export tab."
    )

    DEMO_CASE_ID = "FIR103"
    DEMO_TEXT = """
Case No.: FIR103
Date: 15-Jan-2024
Reporting Officer: Inspector A. Sharma

Details:
Ravi Kumar (DOB: 12-Mar-1985, ID: DL-20199874) was identified at three ATMs
on the day of the incident. Phone 9876543210 registered to Ravi Kumar
was used repeatedly within a 2-hour window between 10:00 and 12:00.
Account ACC00102 received INR 200,000 from an unknown sender at 11:45.

Associate Meena Rao (DOB: 4-Jun-1988, ID: DL-20234561) was seen near the
ATM at Lajpat Nagar at 11:30. Phone 9123456789 shows 14 calls to the
first number in a 2-hour window.

Suresh Patel was mentioned in two prior cases in the same locality.
""".strip()

    progress = st.progress(0, text="Starting demo…")

    # Step 1: Upload
    st.markdown("### Step 1 — Upload case document")
    progress.progress(10, text="Uploading case FIR103…")
    req = NewCaseUpload(case_id=DEMO_CASE_ID, case_text=DEMO_TEXT)
    try:
        resp = call_m5(None, req)
        resp_dict = resp.to_dict() if hasattr(resp, "to_dict") else resp
        progress.progress(35, text="Case uploaded and analyzed.")
        st.success(f"✅ Case {DEMO_CASE_ID} analyzed")
    except Exception as e:
        st.error(f"Demo failed at Step 1 (M5 call): {e}")
        progress.progress(0)
        return

    # Step 2: Load graph + key players
    progress.progress(50, text="Loading network data…")
    graph_data = call_m2_subgraph(None, DEMO_CASE_ID)
    key_players = call_m3_key_players()

    # Save to session state
    st.session_state.session_id = resp_dict.get("session_id")
    st.session_state.case_id = DEMO_CASE_ID
    st.session_state.final_response = resp_dict
    st.session_state.graph_data = graph_data
    st.session_state.key_players = key_players
    st.session_state.conversation_history = [
        {"role": "investigator", "content": f"[Uploaded: {DEMO_CASE_ID}]"},
        {"role": "system", "content": resp_dict.get("response_text", "")},
    ]

    # Step 3: Show result
    st.markdown("### Step 2 — Analysis Result")
    render_final_response(resp_dict)
    progress.progress(65, text="Analysis displayed.")

    # Step 4: Show graph
    st.markdown("### Step 3 — Relationship Network")
    try:
        html = build_pyvis_html(graph_data)
        st.components.v1.html(html, height=500)
        progress.progress(80, text="Network graph rendered.")
    except Exception as e:
        st.warning(f"Graph rendering failed: {e}")

    # Step 5: Follow-up question
    st.markdown("### Step 4 — Follow-Up Question")
    followup = "What connections does Ravi Kumar have to financial accounts?"
    st.info(f"📨 Investigator: *\"{followup}\"*")
    try:
        req2 = FollowUpQuestion(question=followup)
        resp2 = call_m5(st.session_state.session_id, req2)
        resp2_dict = resp2.to_dict() if hasattr(resp2, "to_dict") else resp2
        st.success("🤖 System Response:")
        st.info(resp2_dict.get("response_text", ""))
        st.session_state.conversation_history.append(
            {"role": "investigator", "content": followup}
        )
        st.session_state.conversation_history.append(
            {"role": "system", "content": resp2_dict.get("response_text", "")}
        )
        progress.progress(95, text="Follow-up answered.")
    except Exception as e:
        st.warning(f"Follow-up question demo step failed: {e}")

    progress.progress(100, text="Demo complete. Use the Export tab to download results.")
    st.success(
        "🎉 Demo complete! Use the **Export** tab to download results as CSV, JSON, or PDF. "
        "Use the **Follow-Up** tab to ask more questions."
    )
    st.session_state.demo_mode = False


# ===========================================================================
# Main layout
# ===========================================================================

def main():
    render_sidebar()

    # Demo mode takes over the whole page
    if st.session_state.demo_mode:
        run_demo_script()
        return

    # Tab navigation (M6.1–M6.9)
    tabs = st.tabs([
        "Upload",
        "Graph",
        "Timeline",
        "Search",
        "Video Evidence",
        "Entities",
        "Export",
    ])

    with tabs[0]:
        render_upload_tab()
    with tabs[1]:
        render_graph_tab()
    with tabs[2]:
        render_timeline_tab()
    with tabs[3]:
        render_search_tab()
    with tabs[4]:
        render_video_analysis_tab()
    with tabs[5]:
        render_key_players_tab()
    with tabs[6]:
        render_export_tab()


if __name__ == "__main__":
    main()
