"""
M6 — Export Module  (M6.7)
PS 26152 — AI-Powered Criminal Network Analysis System

Generates CSV, JSON, and PDF exports from the currently displayed case data.
All functions return (bytes, filename) for Streamlit's st.download_button().
Raises ValueError with a human-readable message on failure (callers should catch).
"""

import csv
import io
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_csv(case_id: str, final_response: dict,
               graph_data: dict, key_players: list[dict]) -> tuple[bytes, str]:
    """
    Export the currently displayed case as a CSV file.
    Three sections: summary metadata, entities/nodes, key-player rankings.
    Returns (csv_bytes, filename).
    """
    buf = io.StringIO()
    writer = csv.writer(buf)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Section 1: case summary metadata
    writer.writerow(["SECTION", "CASE SUMMARY"])
    writer.writerow(["Case ID", case_id])
    writer.writerow(["Session ID", final_response.get("session_id", "")])
    writer.writerow(["Confidence Score", final_response.get("confidence", "")])
    writer.writerow(["Requires Human Review", final_response.get("requires_human_review", "")])
    writer.writerow(["Summary Text", final_response.get("response_text", "")])
    writer.writerow([])

    # Section 2: evidence
    writer.writerow(["SECTION", "EVIDENCE"])
    writer.writerow(["Evidence ID"])
    for ev in final_response.get("evidence", []):
        writer.writerow([ev])
    writer.writerow([])

    # Section 3: entities / graph nodes
    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])

    writer.writerow(["SECTION", "ENTITIES"])
    writer.writerow(["Entity ID", "Label", "Type", "Confidence"])
    for node in nodes:
        writer.writerow([
            node.get("id", ""),
            node.get("label", ""),
            node.get("type", ""),
            node.get("confidence", ""),
        ])
    writer.writerow([])

    # Section 4: relationships / edges
    writer.writerow(["SECTION", "RELATIONSHIPS"])
    writer.writerow(["Source", "Target", "Relationship Type", "Confidence", "Timestamp"])
    for edge in edges:
        writer.writerow([
            edge.get("source", ""),
            edge.get("target", ""),
            edge.get("type", ""),
            edge.get("confidence", ""),
            edge.get("timestamp", ""),
        ])
    writer.writerow([])

    # Section 5: key players
    writer.writerow(["SECTION", "KEY ENTITIES BY PRIORITY SCORE"])
    writer.writerow(["Entity ID", "Name", "Priority Score", "Flags", "Evidence"])
    for player in key_players:
        writer.writerow([
            player.get("entity_id", ""),
            player.get("name", ""),
            player.get("priority_score", ""),
            "; ".join(player.get("flags", [])),
            " | ".join(player.get("evidence", [])),
        ])

    content = buf.getvalue().encode("utf-8-sig")  # utf-8-sig for Excel compatibility
    filename = f"case_{case_id}_{timestamp}.csv"
    logger.info(f"[Export] CSV generated: {filename}")
    return content, filename


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------

def export_json(case_id: str, final_response: dict,
                graph_data: dict, key_players: list[dict]) -> tuple[bytes, str]:
    """
    Export the currently displayed case as a structured JSON file.
    Returns (json_bytes, filename).
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    payload = {
        "export_metadata": {
            "case_id": case_id,
            "exported_at": datetime.now().isoformat(),
            "system": "PS 26152 — AI-Powered Criminal Network Analysis System",
            "disclaimer": (
                "All findings are investigative leads requiring verification. "
                "This data does not constitute evidence of any offence."
            ),
        },
        "analysis_result": final_response,
        "network_data": {
            "nodes": graph_data.get("nodes", []),
            "edges": graph_data.get("edges", []),
        },
        "key_entities_by_priority": key_players,
    }

    content = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
    filename = f"case_{case_id}_{timestamp}.json"
    logger.info(f"[Export] JSON generated: {filename}")
    return content, filename


# ---------------------------------------------------------------------------
# PDF export
# ---------------------------------------------------------------------------

def export_pdf(case_id: str, final_response: dict,
               graph_data: dict, key_players: list[dict]) -> tuple[bytes, str]:
    """
    Export the currently displayed case as a PDF report using fpdf2.
    Returns (pdf_bytes, filename).
    """
    from fpdf import FPDF

    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    filename = f"case_{case_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_margins(15, 15, 15)

    # --- Header ---
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Investigation Analysis Report", new_x="LMARGIN", new_y="NEXT", align="C")

    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Case ID: {case_id}    |    Generated: {timestamp_str}", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.cell(0, 6, "PS 26152 -- AI-Powered Criminal Network Analysis System", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(4)

    # --- Disclaimer ---
    pdf.set_fill_color(255, 243, 205)
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(
        0, 6,
        "DISCLAIMER: All findings are investigative leads requiring verification before any action is taken. "
        "This report does not constitute evidence of any offence and must not be used as a basis for "
        "any accusation, arrest, or prosecution without independent verification.",
        fill=True, border=1
    )
    pdf.ln(4)

    # --- Analysis Summary ---
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "1. Analysis Summary", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)

    confidence = final_response.get("confidence", 0.0)
    requires_review = final_response.get("requires_human_review", False)

    pdf.cell(60, 6, "Confidence Score:", border=0)
    pdf.cell(0, 6, f"{confidence:.1%}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(60, 6, "Requires Human Review:", border=0)
    pdf.cell(0, 6, "YES -- escalate to senior investigator" if requires_review else "No (automated confidence threshold met)", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Summary:", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    summary = final_response.get("response_text", "")
    # Sanitize characters that FPDF can't handle
    summary = summary.encode("latin-1", "replace").decode("latin-1")
    pdf.multi_cell(0, 6, summary)
    pdf.ln(3)

    # --- Evidence ---
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "2. Evidence References", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for i, ev in enumerate(final_response.get("evidence", []), 1):
        pdf.cell(0, 6, f"  {i}. {ev}", new_x="LMARGIN", new_y="NEXT")
    if not final_response.get("evidence"):
        pdf.cell(0, 6, "  No evidence references recorded.", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # --- Key Entities ---
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "3. Key Entities by Priority Score", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(40, 7, "Entity ID", border=1)
    pdf.cell(50, 7, "Name", border=1)
    pdf.cell(30, 7, "Priority Score", border=1)
    pdf.cell(0, 7, "Flags", border=1, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    for player in key_players:
        pdf.cell(40, 6, player.get("entity_id", "")[:15], border=1)
        pdf.cell(50, 6, player.get("name", "")[:20], border=1)
        pdf.cell(30, 6, f"{player.get('priority_score', 0):.2f}", border=1)
        flags_str = ", ".join(player.get("flags", []))
        flags_str = flags_str.encode("latin-1", "replace").decode("latin-1")
        pdf.cell(0, 6, flags_str[:40], border=1, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # --- Network Summary ---
    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "4. Network Summary", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Total entities in network: {len(nodes)}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Total relationships in network: {len(edges)}", new_x="LMARGIN", new_y="NEXT")

    # Entity type breakdown
    types: dict = {}
    for node in nodes:
        t = node.get("type", "UNKNOWN")
        types[t] = types.get(t, 0) + 1
    for t, count in sorted(types.items()):
        pdf.cell(0, 6, f"  {t}: {count}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # --- Footer ---
    pdf.set_font("Helvetica", "I", 8)
    pdf.cell(
        0, 6,
        "This report is generated by an automated system. All findings require investigator verification.",
        new_x="LMARGIN", new_y="NEXT", align="C"
    )

    content = pdf.output()
    if isinstance(content, str):
        content = content.encode("latin-1")

    logger.info(f"[Export] PDF generated: {filename}")
    return bytes(content), filename
