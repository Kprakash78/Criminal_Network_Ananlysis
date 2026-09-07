"""
M6_feature — Timeline Playback UI Component
PS 26152 — AI-Powered Criminal Network Analysis System

Renders events as a true vertical investigation timeline:
  - ONE continuous vertical spine via CSS ::before pseudo-element
  - ONE marker per event, sitting on the spine
  - Event cards alternate LEFT / RIGHT using CSS Grid (3 columns)
  - Normal document flow — no fixed canvas, no graph layout, no coordinates
  - Grows naturally: 2 events → short; 220 events → long vertical page

Controls:
  Play / Pause | Prev | Next | Reset | Speed selector
"""

import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def render_timeline_html(events: list[dict], width: int = 800, height: int = 600) -> str:
    """
    Generate a standalone HTML page with an animated vertical investigation timeline.

    The height parameter is ignored; the page grows with content.
    The width parameter is informational only — CSS uses 100% width.

    Args:
        events: List of event dicts from timeline_player.build_timeline()
        width:  Unused (CSS handles width with 100%)
        height: Unused (page grows with content naturally)

    Returns:
        Complete HTML string ready for embedding via st.components.v1.html()
    """
    events_json = json.dumps(events, ensure_ascii=False)

    type_colors = {
        "call":        "#4A6FA5",   # muted blue
        "transaction": "#B8860B",   # dark goldenrod
        "fir_filing":  "#8B0000",   # dark red
    }
    colors_json = json.dumps(type_colors)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}

body {{
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: #F9F6F0;
    color: #211C18;
    /* No fixed height — the page grows with content */
}}

/* ── Sticky control bar ─────────────────────────────────────────────── */
#controls {{
    padding: 10px 16px;
    background: #F9F6F0;
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
    border-bottom: 1px solid #DED5C8;
    position: sticky;
    top: 0;
    z-index: 100;
}}
button {{
    background: transparent;
    color: #211C18;
    border: 1px solid #DED5C8;
    padding: 6px 14px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 13px;
    transition: background 0.15s;
}}
button:hover {{ background: rgba(58,42,34,0.06); }}
button.active {{ background: #2B211C; color: #F7F2E8; border-color: #2B211C; }}
#speed-label {{ font-size: 12px; color: #6B5D50; margin-left: 4px; }}
select {{
    background: transparent;
    color: #211C18;
    border: 1px solid #DED5C8;
    padding: 5px 8px;
    border-radius: 4px;
    font-size: 12px;
}}
#event-counter {{
    font-size: 12px;
    color: #6B5D50;
    margin-left: auto;
}}

/* ── Thin progress bar ──────────────────────────────────────────────── */
#progress-bar {{
    width: 100%;
    height: 3px;
    background: #DED5C8;
    position: sticky;
    top: 43px;
    z-index: 99;
}}
#progress-fill {{
    height: 100%;
    background: linear-gradient(90deg, #9E5748, #3A2A22);
    width: 0%;
    transition: width 0.3s;
}}

/* ── Timeline body — grows with content ─────────────────────────────── */
#timeline-body {{
    width: 100%;
    background: #F9F6F0;
    padding: 40px 20px 60px;
}}

/* ── Centre column container ─────────────────────────────────────────── */
#timeline-content {{
    position: relative;
    display: flex;
    flex-direction: column;
    gap: 20px;
    max-width: 820px;
    margin: 0 auto;
}}

/* ── THE ONE CONTINUOUS VERTICAL SPINE ──────────────────────────────── */
/* Runs from the very top of the first event to the very bottom of the   */
/* last event, centred on the marker column.                             */
#timeline-content::before {{
    content: '';
    position: absolute;
    top: 0;
    bottom: 0;
    left: 50%;
    transform: translateX(-50%);
    width: 2px;
    background: #DED5C8;
    z-index: 1;
}}

/* ── 3-column grid row: [card] [marker] [card] ───────────────────────── */
.timeline-row {{
    display: grid;
    grid-template-columns: minmax(0, 1fr) 40px minmax(0, 1fr);
    align-items: center;
    position: relative;
    z-index: 2;
    min-height: 100px;
}}

/* ── Centre marker ───────────────────────────────────────────────────── */
.marker-col {{
    grid-column: 2;
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 3;
}}
.node-marker {{
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: #F9F6F0;
    border: 3px solid #6B5D50;
    transition: transform 0.25s, box-shadow 0.25s, background 0.25s;
    flex-shrink: 0;
}}
.timeline-row.active .node-marker {{
    transform: scale(1.5);
    box-shadow: 0 0 0 4px rgba(43,33,28,0.12);
}}

/* ── LEFT card (odd rows) ────────────────────────────────────────────── */
.card-left {{
    grid-column: 1;
    padding-right: 24px;
    position: relative;
    text-align: right;
}}
.card-left::after {{
    content: '';
    position: absolute;
    right: 0;
    top: 50%;
    transform: translateY(-50%);
    width: 24px;
    height: 2px;
    background: #DED5C8;
    z-index: 1;
}}
/* Spacer on the right for left-side rows */
.card-right-empty {{
    grid-column: 3;
}}

/* ── RIGHT card (even rows) ──────────────────────────────────────────── */
.card-right {{
    grid-column: 3;
    padding-left: 24px;
    position: relative;
}}
.card-right::before {{
    content: '';
    position: absolute;
    left: 0;
    top: 50%;
    transform: translateY(-50%);
    width: 24px;
    height: 2px;
    background: #DED5C8;
    z-index: 1;
}}
/* Spacer on the left for right-side rows */
.card-left-empty {{
    grid-column: 1;
}}

/* ── The event card ──────────────────────────────────────────────────── */
.event-card {{
    background: #FFFFFF;
    border: 1px solid #DED5C8;
    border-radius: 6px;
    padding: 14px 16px;
    cursor: pointer;
    transition: transform 0.15s, box-shadow 0.15s, border-color 0.15s;
    position: relative;
    z-index: 2;
}}
.event-card:hover {{
    transform: translateY(-2px);
    box-shadow: 0 4px 14px rgba(0,0,0,0.06);
    border-color: #C0B5A6;
}}
.timeline-row.active .event-card {{
    border-color: #3A2A22;
    box-shadow: 0 0 0 2px rgba(58,42,34,0.2);
    background: #FDFBF8;
}}

/* Card header: badge + timestamp */
.card-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
    flex-wrap: wrap;
}}
/* Left-side cards flip the header direction */
.card-left .card-header {{
    flex-direction: row-reverse;
}}
.event-type-badge {{
    display: inline-block;
    padding: 2px 7px;
    border-radius: 3px;
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}}
.card-time {{
    font-size: 11px;
    color: #6B5D50;
    font-family: monospace;
    white-space: nowrap;
}}
.card-body {{
    font-size: 13px;
    color: #211C18;
    line-height: 1.5;
    margin-bottom: 8px;
}}
.card-footer {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 11px;
    color: #8C8276;
    gap: 8px;
    flex-wrap: wrap;
}}
.card-left .card-footer {{
    flex-direction: row-reverse;
}}
.confidence-badge {{
    font-size: 10px;
    font-weight: 600;
    padding: 1px 5px;
    border-radius: 3px;
}}
.conf-high   {{ background: #D4EDDA; color: #155724; }}
.conf-medium {{ background: #FFF3CD; color: #856404; }}
.conf-low    {{ background: #F8D7DA; color: #721C24; }}

/* ── Mobile layout: spine on left, all cards on right ───────────────── */
@media (max-width: 680px) {{
    #timeline-content::before {{
        left: 20px;
        transform: none;
    }}
    .timeline-row {{
        grid-template-columns: 40px minmax(0, 1fr);
    }}
    .marker-col    {{ grid-column: 1; }}
    .card-left, .card-right {{
        grid-column: 2;
        padding-left: 20px;
        padding-right: 0;
        text-align: left;
    }}
    .card-left::after, .card-right::before {{
        left: 0; right: auto;
        width: 20px;
        transform: translateY(-50%);
    }}
    .card-right-empty, .card-left-empty {{ display: none; }}
    .card-left .card-header,
    .card-left .card-footer {{ flex-direction: row; }}
}}

/* ── Empty state ─────────────────────────────────────────────────────── */
#empty-state {{
    text-align: center;
    padding: 60px 20px;
    color: #6B5D50;
    font-size: 14px;
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
        <option value="2000">0.5×</option>
        <option value="1000" selected>1×</option>
        <option value="500">2×</option>
        <option value="200">5×</option>
        <option value="100">10×</option>
    </select>
    <span id="event-counter">0 events</span>
</div>
<div id="progress-bar"><div id="progress-fill"></div></div>

<div id="timeline-body">
    <div id="timeline-content">
        <!-- rows injected by JS -->
    </div>
    <div id="empty-state" style="display:none;">
        No events to display. Upload and analyze a case first.
    </div>
</div>

<script>
const events = {events_json};
const typeColors = {colors_json};

let currentIdx = -1;
let isPlaying   = false;
let playTimer   = null;
let speed       = 1000;

// ── Icon per event type ─────────────────────────────────────────────────
function eventIcon(type) {{
    if (type === 'call')        return '☎';
    if (type === 'transaction') return '₹';
    if (type === 'fir_filing')  return '📄';
    return '●';
}}

// ── Confidence badge ────────────────────────────────────────────────────
function confBadge(conf) {{
    const c = conf !== undefined ? conf : 0.85;
    if (c >= 0.8) return '<span class="confidence-badge conf-high">High</span>';
    if (c >= 0.5) return '<span class="confidence-badge conf-medium">Medium</span>';
    return '<span class="confidence-badge conf-low">Low</span>';
}}

// ── Build the entire timeline DOM ───────────────────────────────────────
function initTimeline() {{
    const container = document.getElementById('timeline-content');
    const emptyState = document.getElementById('empty-state');

    if (events.length === 0) {{
        container.style.display = 'none';
        emptyState.style.display = 'block';
        document.getElementById('event-counter').textContent = '0 events';
        return;
    }}

    events.forEach((e, idx) => {{
        const isLeft = (idx % 2 === 0);   // even index → left card
        const typeColor = typeColors[e.type] || '#6B5D50';
        const typeLabel = (e.type || '').replace(/_/g, ' ').toUpperCase();
        const icon      = eventIcon(e.type);

        let dateStr = e.t || '';
        if (dateStr.length > 10) dateStr = dateStr.replace('T', ' ').substring(0, 16);

        const fromTo = e.from_name
            ? (e.to_name ? e.from_name + ' → ' + e.to_name : e.from_name)
            : '';

        const sourceRef = (e.file && e.line)
            ? '<span title="' + e.file + ':L' + e.line + '">📎 ' + e.file.split('/').pop() + ':L' + e.line + '</span>'
            : '';

        const cardHTML = `
            <div class="card-header">
                <span class="event-type-badge"
                      style="border:1px solid ${{typeColor}};color:${{typeColor}};">
                    ${{icon}} ${{typeLabel}}
                </span>
                <span class="card-time">${{dateStr}}</span>
            </div>
            <div class="card-body">${{e.caption || (typeLabel + ' event')}}</div>
            <div class="card-footer">
                <span>${{fromTo}}</span>
                <span style="display:flex;gap:6px;align-items:center;">
                    ${{confBadge(e.confidence)}}
                    ${{sourceRef}}
                </span>
            </div>
        `;

        const row = document.createElement('div');
        row.className = 'timeline-row';
        row.id = 'row-' + idx;

        // Marker — always grid-column 2
        const markerCol = document.createElement('div');
        markerCol.className = 'marker-col';
        const marker = document.createElement('div');
        marker.className = 'node-marker';
        marker.style.borderColor = typeColor;
        markerCol.appendChild(marker);

        if (isLeft) {{
            // LEFT: [card] [marker] [empty]
            const cardWrap = document.createElement('div');
            cardWrap.className = 'card-left';
            const card = document.createElement('div');
            card.className = 'event-card';
            card.onclick = () => showEvent(idx);
            card.innerHTML = cardHTML;
            cardWrap.appendChild(card);

            const spacer = document.createElement('div');
            spacer.className = 'card-right-empty';

            row.appendChild(cardWrap);
            row.appendChild(markerCol);
            row.appendChild(spacer);
        }} else {{
            // RIGHT: [empty] [marker] [card]
            const spacer = document.createElement('div');
            spacer.className = 'card-left-empty';

            const cardWrap = document.createElement('div');
            cardWrap.className = 'card-right';
            const card = document.createElement('div');
            card.className = 'event-card';
            card.onclick = () => showEvent(idx);
            card.innerHTML = cardHTML;
            cardWrap.appendChild(card);

            row.appendChild(spacer);
            row.appendChild(markerCol);
            row.appendChild(cardWrap);
        }}

        container.appendChild(row);
    }});

    document.getElementById('event-counter').textContent =
        events.length + ' event' + (events.length !== 1 ? 's' : '') + ' in this case';
}}

// ── Activate an event ───────────────────────────────────────────────────
function showEvent(idx) {{
    if (idx < 0 || idx >= events.length) return;

    // Deactivate old
    if (currentIdx >= 0) {{
        const oldRow = document.getElementById('row-' + currentIdx);
        if (oldRow) {{
            oldRow.classList.remove('active');
            const oldMarker = oldRow.querySelector('.node-marker');
            if (oldMarker) oldMarker.style.background = '#F9F6F0';
        }}
    }}

    currentIdx = idx;
    const e = events[idx];
    const typeColor = typeColors[e.type] || '#211C18';

    const newRow = document.getElementById('row-' + idx);
    if (newRow) {{
        newRow.classList.add('active');
        const marker = newRow.querySelector('.node-marker');
        if (marker) marker.style.background = typeColor;
        // Scroll into view without moving to absolute top
        newRow.scrollIntoView({{ behavior: 'smooth', block: 'nearest' }});
    }}

    // Progress
    const pct = ((idx + 1) / events.length) * 100;
    document.getElementById('progress-fill').style.width = pct + '%';
    document.getElementById('event-counter').textContent =
        'Event ' + (idx + 1) + ' of ' + events.length + ' in this case';
}}

// ── Playback controls ───────────────────────────────────────────────────
function togglePlay() {{
    if (isPlaying) {{
        clearInterval(playTimer);
        isPlaying = false;
        document.getElementById('btn-play').textContent = '▶ Play';
        document.getElementById('btn-play').classList.remove('active');
    }} else {{
        // If at end, restart from beginning
        if (currentIdx >= events.length - 1) resetTimeline();
        isPlaying = true;
        document.getElementById('btn-play').textContent = '⏸ Pause';
        document.getElementById('btn-play').classList.add('active');
        playTimer = setInterval(() => {{
            if (currentIdx >= events.length - 1) {{
                clearInterval(playTimer);
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
    if (currentIdx >= 0) {{
        const oldRow = document.getElementById('row-' + currentIdx);
        if (oldRow) {{
            oldRow.classList.remove('active');
            const m = oldRow.querySelector('.node-marker');
            if (m) m.style.background = '#F9F6F0';
        }}
    }}
    currentIdx = -1;
    document.getElementById('progress-fill').style.width = '0%';
    document.getElementById('event-counter').textContent =
        events.length + ' event' + (events.length !== 1 ? 's' : '') + ' in this case';
    window.scrollTo({{ top: 0, behavior: 'smooth' }});
}}

function setSpeed(ms) {{
    speed = parseInt(ms);
    if (isPlaying) {{
        clearInterval(playTimer);
        playTimer = setInterval(() => {{
            if (currentIdx >= events.length - 1) {{
                clearInterval(playTimer);
                isPlaying = false;
                document.getElementById('btn-play').textContent = '▶ Play';
                document.getElementById('btn-play').classList.remove('active');
                return;
            }}
            showEvent(currentIdx + 1);
        }}, speed);
    }}
}}

// ── Boot ────────────────────────────────────────────────────────────────
initTimeline();
</script>
</body>
</html>"""

    return html
