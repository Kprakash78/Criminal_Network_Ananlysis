"""
M6_feature — Timeline Playback UI Component
PS 26152 — AI-Powered Criminal Network Analysis System

Streamlit-compatible timeline visualization with Play/Pause/Speed controls.
Renders events as an animated timeline with PyVis network graph highlighting.

This module provides reusable Streamlit components that can be embedded
in the main search_ui.py dashboard.
"""

import json
import logging
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent

# Add repo root to path
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def render_timeline_html(events: list[dict], width: int = 800, height: int = 600) -> str:
    """
    Generate a standalone HTML page with an animated timeline player.

    Uses vanilla JavaScript for animation and a simple SVG-based network
    visualization. This avoids heavy JS library dependencies and works
    in Streamlit's iframe embedding.

    Args:
        events: List of event dicts from timeline_player.build_timeline()
        width:  Canvas width in pixels
        height: Canvas height in pixels

    Returns:
        Complete HTML string ready for embedding
    """
    # Collect unique entities for the network graph
    entities = set()
    for e in events:
        if e.get("from_name"):
            entities.add(e["from_name"])
        elif e.get("from"):
            entities.add(e["from"])
        if e.get("to_name"):
            entities.add(e["to_name"])
        elif e.get("to") and e["to"]:
            entities.add(e["to"])

    entities = sorted(entities)

    # Assign positions in a circle layout
    import math
    positions = {}
    cx, cy = width // 2, (height - 200) // 2
    radius = min(cx, cy) - 60
    for i, entity in enumerate(entities):
        angle = 2 * math.pi * i / max(len(entities), 1)
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        positions[entity] = (int(x), int(y))

    # Serialize for JavaScript
    events_json = json.dumps(events, ensure_ascii=False)
    positions_json = json.dumps(positions, ensure_ascii=False)

    # Color mapping for event types
    type_colors = {
        "call": "#4FC3F7",
        "transaction": "#FFB74D",
        "fir_filing": "#EF5350",
    }
    colors_json = json.dumps(type_colors)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
        font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
        background: #1a1a2e;
        color: #e0e0e0;
    }}
    #controls {{
        padding: 12px 16px;
        background: #16213e;
        display: flex;
        align-items: center;
        gap: 12px;
        border-bottom: 1px solid #0f3460;
    }}
    button {{
        background: #0f3460;
        color: #e0e0e0;
        border: 1px solid #533483;
        padding: 8px 18px;
        border-radius: 6px;
        cursor: pointer;
        font-size: 14px;
        transition: background 0.2s;
    }}
    button:hover {{ background: #533483; }}
    button.active {{ background: #e94560; border-color: #e94560; }}
    #speed-label {{ font-size: 13px; color: #aaa; }}
    select {{
        background: #0f3460;
        color: #e0e0e0;
        border: 1px solid #533483;
        padding: 6px 10px;
        border-radius: 4px;
        font-size: 13px;
    }}
    #progress-bar {{
        width: 100%;
        height: 4px;
        background: #16213e;
        position: relative;
    }}
    #progress-fill {{
        height: 100%;
        background: linear-gradient(90deg, #e94560, #533483);
        width: 0%;
        transition: width 0.3s;
    }}
    #canvas-container {{
        position: relative;
        width: 100%;
        height: {height - 200}px;
    }}
    canvas {{ display: block; }}
    #caption-box {{
        padding: 12px 16px;
        background: #16213e;
        min-height: 80px;
        border-top: 1px solid #0f3460;
        font-size: 14px;
        line-height: 1.6;
    }}
    .event-type {{
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 600;
        margin-right: 8px;
    }}
    .event-call {{ background: #4FC3F720; color: #4FC3F7; }}
    .event-transaction {{ background: #FFB74D20; color: #FFB74D; }}
    .event-fir_filing {{ background: #EF535020; color: #EF5350; }}
    #event-counter {{
        font-size: 12px;
        color: #888;
    }}
    .source-link {{
        color: #4FC3F7;
        text-decoration: underline;
        cursor: pointer;
    }}
</style>
</head>
<body>
<div id="controls">
    <button id="btn-play" onclick="togglePlay()">▶ Play</button>
    <button onclick="stepBack()">⏮ Prev</button>
    <button onclick="stepForward()">⏭ Next</button>
    <button onclick="resetTimeline()">⏹ Reset</button>
    <span id="speed-label">Speed:</span>
    <select id="speed-select" onchange="setSpeed(this.value)">
        <option value="2000">1x</option>
        <option value="1000" selected>2x</option>
        <option value="400">5x</option>
        <option value="200">10x</option>
    </select>
    <span id="event-counter">Event 0 / 0</span>
</div>
<div id="progress-bar"><div id="progress-fill"></div></div>
<div id="canvas-container">
    <canvas id="graph-canvas" width="{width}" height="{height - 200}"></canvas>
</div>
<div id="caption-box">
    <em style="color:#888">Press Play to start the timeline animation</em>
</div>

<script>
const events = {events_json};
const positions = {positions_json};
const typeColors = {colors_json};

let currentIdx = -1;
let isPlaying = false;
let playInterval = null;
let speed = 1000;

const canvas = document.getElementById('graph-canvas');
const ctx = canvas.getContext('2d');

function drawGraph(activeFrom, activeTo, eventType) {{
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw edges for active connection
    if (activeFrom && activeTo && positions[activeFrom] && positions[activeTo]) {{
        const [x1, y1] = positions[activeFrom];
        const [x2, y2] = positions[activeTo];
        const color = typeColors[eventType] || '#4FC3F7';

        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.setLineDash([8, 4]);
        ctx.stroke();
        ctx.setLineDash([]);

        // Animated pulse on the edge
        const mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
        ctx.beginPath();
        ctx.arc(mx, my, 6, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
    }}

    // Draw nodes
    for (const [name, [x, y]] of Object.entries(positions)) {{
        const isActive = (name === activeFrom || name === activeTo);
        const radius = isActive ? 22 : 14;
        const color = isActive ? (typeColors[eventType] || '#4FC3F7') : '#533483';

        // Glow effect for active nodes
        if (isActive) {{
            ctx.beginPath();
            ctx.arc(x, y, radius + 8, 0, Math.PI * 2);
            ctx.fillStyle = color + '30';
            ctx.fill();
        }}

        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
        ctx.strokeStyle = isActive ? '#fff' : '#0f3460';
        ctx.lineWidth = 2;
        ctx.stroke();

        // Label
        ctx.fillStyle = isActive ? '#fff' : '#aaa';
        ctx.font = isActive ? 'bold 11px sans-serif' : '10px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';

        // Truncate long names
        let displayName = name.length > 15 ? name.substring(0, 13) + '…' : name;
        ctx.fillText(displayName, x, y + radius + 4);
    }}
}}

function showEvent(idx) {{
    if (idx < 0 || idx >= events.length) return;
    currentIdx = idx;
    const e = events[idx];

    const fromName = e.from_name || e.from || '';
    const toName = e.to_name || e.to || '';

    drawGraph(fromName, toName, e.type);

    // Update caption
    const captionBox = document.getElementById('caption-box');
    const typeClass = 'event-' + e.type;
    const typeLabel = e.type.replace('_', ' ').toUpperCase();

    captionBox.innerHTML = `
        <span class="event-type ${{typeClass}}">${{typeLabel}}</span>
        <strong>${{e.t}}</strong><br>
        ${{e.caption || ''}}<br>
        <span style="color:#888;font-size:12px">
            📁 ${{e.file}}:L${{e.line}}
        </span>
    `;

    // Update progress
    const pct = ((idx + 1) / events.length) * 100;
    document.getElementById('progress-fill').style.width = pct + '%';
    document.getElementById('event-counter').textContent =
        `Event ${{idx + 1}} / ${{events.length}}`;
}}

function togglePlay() {{
    if (isPlaying) {{
        clearInterval(playInterval);
        isPlaying = false;
        document.getElementById('btn-play').textContent = '▶ Play';
        document.getElementById('btn-play').classList.remove('active');
    }} else {{
        isPlaying = true;
        document.getElementById('btn-play').textContent = '⏸ Pause';
        document.getElementById('btn-play').classList.add('active');
        playInterval = setInterval(() => {{
            if (currentIdx >= events.length - 1) {{
                clearInterval(playInterval);
                isPlaying = false;
                document.getElementById('btn-play').textContent = '▶ Play';
                document.getElementById('btn-play').classList.remove('active');
                return;
            }}
            showEvent(currentIdx + 1);
        }}, speed);
    }}
}}

function stepForward() {{
    if (currentIdx < events.length - 1) showEvent(currentIdx + 1);
}}

function stepBack() {{
    if (currentIdx > 0) showEvent(currentIdx - 1);
}}

function resetTimeline() {{
    if (isPlaying) togglePlay();
    currentIdx = -1;
    drawGraph(null, null, null);
    document.getElementById('caption-box').innerHTML =
        '<em style="color:#888">Press Play to start the timeline animation</em>';
    document.getElementById('progress-fill').style.width = '0%';
    document.getElementById('event-counter').textContent = `Event 0 / ${{events.length}}`;
}}

function setSpeed(ms) {{
    speed = parseInt(ms);
    if (isPlaying) {{
        clearInterval(playInterval);
        playInterval = setInterval(() => {{
            if (currentIdx >= events.length - 1) {{
                clearInterval(playInterval);
                isPlaying = false;
                document.getElementById('btn-play').textContent = '▶ Play';
                document.getElementById('btn-play').classList.remove('active');
                return;
            }}
            showEvent(currentIdx + 1);
        }}, speed);
    }}
}}

// Initial draw
drawGraph(null, null, null);
document.getElementById('event-counter').textContent = `Event 0 / ${{events.length}}`;
</script>
</body>
</html>"""

    return html


def render_pyvis_graph(events: list[dict]) -> str:
    """
    Alternative: Generate a PyVis network graph HTML.
    Falls back to SVG-based rendering if PyVis unavailable.
    """
    try:
        from pyvis.network import Network

        net = Network(height="500px", width="100%", bgcolor="#1a1a2e",
                      font_color="#e0e0e0", directed=True)
        net.toggle_physics(True)

        # Collect unique entities
        entities = set()
        for e in events:
            from_name = e.get("from_name", e.get("from", ""))
            to_name = e.get("to_name", e.get("to", ""))
            if from_name:
                entities.add(from_name)
            if to_name:
                entities.add(to_name)

        # Add nodes
        for entity in entities:
            net.add_node(entity, label=entity, color="#533483",
                         font={"color": "#e0e0e0"})

        # Add edges from events (deduplicated)
        edges_seen = set()
        for e in events:
            from_name = e.get("from_name", e.get("from", ""))
            to_name = e.get("to_name", e.get("to", ""))
            if from_name and to_name and (from_name, to_name) not in edges_seen:
                color = {"call": "#4FC3F7", "transaction": "#FFB74D",
                         "fir_filing": "#EF5350"}.get(e["type"], "#888")
                net.add_edge(from_name, to_name, color=color,
                             title=e.get("caption", ""))
                edges_seen.add((from_name, to_name))

        # Write to temp file
        tmp = tempfile.NamedTemporaryFile(suffix=".html", delete=False)
        net.save_graph(tmp.name)
        with open(tmp.name, "r") as f:
            return f.read()

    except ImportError:
        logger.warning("[Timeline] PyVis not available, using built-in renderer")
        return render_timeline_html(events)
