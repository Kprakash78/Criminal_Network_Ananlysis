"""
M6_feature — Investigation Timeline UI

A real chronological investigation timeline.

Layout:
    LEFT CARD  |  MARKER  |  RIGHT CARD
                 |
    LEFT CARD  |  MARKER  |
                 |
                 |  MARKER  |  RIGHT CARD

No graph layout.
No coordinates.
No SVG graph.
No radial positioning.
"""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def render_timeline_html(
    events: list[dict],
    width: int = 1100,
    height: int = 900,
) -> str:
    """
    Render the investigation timeline as ordinary HTML/CSS.

    The height argument is intentionally only used as a fallback
    by the embedding application. The timeline itself grows naturally.
    """

    # Always chronological.
    events = sorted(
        events or [],
        key=lambda e: str(e.get("t", "")),
    )

    events_json = json.dumps(
        events,
        ensure_ascii=False,
        default=str,
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">

<style>

* {{
    box-sizing: border-box;
}}

html,
body {{
    margin: 0;
    padding: 0;
    width: 100%;
}}

body {{
    background: #F7F2E8;
    color: #2B211C;
    font-family:
        Inter,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;
}}

/* ============================================================
   CONTROL BAR
   ============================================================ */

.timeline-controls {{
    position: sticky;
    top: 0;
    z-index: 50;

    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;

    padding: 14px 20px;

    background: rgba(247, 242, 232, 0.97);
    border-bottom: 1px solid #D8CEC0;
}}

.timeline-controls button {{
    appearance: none;

    border: 1px solid #CFC3B4;
    background: #FFFCF7;
    color: #2B211C;

    border-radius: 6px;

    padding: 9px 15px;

    font-size: 13px;
    font-weight: 600;

    cursor: pointer;

    transition:
        background 0.15s ease,
        border-color 0.15s ease,
        transform 0.15s ease;
}}

.timeline-controls button:hover {{
    background: #EFE8DC;
    border-color: #AFA092;
}}

.timeline-controls button.active {{
    background: #2B211C;
    color: #F7F2E8;
    border-color: #2B211C;
}}

.speed-label {{
    margin-left: 4px;

    font-size: 12px;
    color: #6B5D50;
}}

.timeline-controls select {{
    border: 1px solid #CFC3B4;
    background: #FFFCF7;
    color: #2B211C;

    border-radius: 6px;

    padding: 8px 10px;

    font-size: 13px;
}}

.event-counter {{
    margin-left: auto;

    font-size: 12px;
    color: #6B5D50;
}}

/* ============================================================
   PROGRESS
   ============================================================ */

.timeline-progress {{
    height: 3px;
    width: 100%;

    background: #DED5C8;
}}

.timeline-progress-fill {{
    width: 0%;
    height: 100%;

    background: #8C4F3E;

    transition: width 0.25s ease;
}}

/* ============================================================
   TIMELINE CONTAINER
   ============================================================ */

.timeline-shell {{
    width: 100%;
    padding: 42px 28px 70px;
}}

.timeline {{
    position: relative;

    width: min(1120px, 100%);

    margin: 0 auto;

    /*
     * THREE COLUMNS:
     *
     * LEFT CARD | SPINE | RIGHT CARD
     */
    display: flex;
    flex-direction: column;
}}

/* ============================================================
   THE ONE CONTINUOUS SPINE
   ============================================================ */

.timeline::before {{
    content: "";

    position: absolute;

    top: 0;
    bottom: 0;

    left: 50%;

    width: 2px;

    transform: translateX(-50%);

    background: #C9BFB1;

    z-index: 0;
}}

/* ============================================================
   EVENT ROW
   ============================================================ */

.timeline-row {{
    position: relative;

    display: grid;

    grid-template-columns:
        minmax(0, 1fr)
        56px
        minmax(0, 1fr);

    align-items: center;

    min-height: 150px;

    z-index: 1;
}}

/* ============================================================
   CENTER MARKER
   ============================================================ */

.timeline-marker {{
    grid-column: 2;

    display: flex;
    align-items: center;
    justify-content: center;

    width: 56px;
    height: 100%;

    position: relative;

    z-index: 5;
}}

.marker-dot {{
    width: 15px;
    height: 15px;

    border-radius: 50%;

    background: #F7F2E8;

    border: 3px solid #6B5D50;

    box-shadow:
        0 0 0 5px #F7F2E8;

    transition:
        transform 0.2s ease,
        box-shadow 0.2s ease,
        background 0.2s ease;
}}

.timeline-row.active .marker-dot {{
    transform: scale(1.35);

    box-shadow:
        0 0 0 5px #F7F2E8,
        0 0 0 8px rgba(43, 33, 28, 0.12);
}}

/* ============================================================
   CARD WRAPPERS
   ============================================================ */

.timeline-left {{
    grid-column: 1;

    display: flex;
    justify-content: flex-end;

    padding-right: 24px;

    position: relative;
}}

.timeline-right {{
    grid-column: 3;

    display: flex;
    justify-content: flex-start;

    padding-left: 24px;

    position: relative;
}}

/* ============================================================
   CONNECTORS
   ============================================================ */

.timeline-left::after {{
    content: "";

    position: absolute;

    right: 0;
    top: 50%;

    width: 24px;
    height: 2px;

    transform: translateY(-50%);

    background: #C9BFB1;
}}

.timeline-right::before {{
    content: "";

    position: absolute;

    left: 0;
    top: 50%;

    width: 24px;
    height: 2px;

    transform: translateY(-50%);

    background: #C9BFB1;
}}

/* ============================================================
   EVENT CARD
   ============================================================ */

.event-card {{
    width: min(500px, 100%);

    background: #FFFCF7;

    border:
        1px solid
        #D8CEC0;

    border-radius: 8px;

    padding: 17px 19px;

    cursor: pointer;

    transition:
        transform 0.18s ease,
        border-color 0.18s ease,
        box-shadow 0.18s ease,
        background 0.18s ease;
}}

.event-card:hover {{
    transform: translateY(-2px);

    border-color: #B9AA99;

    box-shadow:
        0 8px 22px rgba(43, 33, 28, 0.07);
}}

.timeline-row.active .event-card {{
    border-color: #2B211C;

    background: #FFFDF9;

    box-shadow:
        0 0 0 2px rgba(43, 33, 28, 0.10),
        0 8px 24px rgba(43, 33, 28, 0.08);
}}

/* ============================================================
   CARD HEADER
   ============================================================ */

.event-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;

    gap: 12px;

    margin-bottom: 10px;
}}

.event-type {{
    display: inline-flex;
    align-items: center;

    padding: 4px 8px;

    border-radius: 4px;

    border: 1px solid currentColor;

    font-size: 10px;
    font-weight: 700;

    letter-spacing: 0.6px;
    text-transform: uppercase;
}}

.event-time {{
    font-family:
        "SFMono-Regular",
        Consolas,
        monospace;

    font-size: 11px;

    color: #6B5D50;

    white-space: nowrap;
}}

.event-caption {{
    font-size: 14px;

    line-height: 1.55;

    color: #2B211C;

    margin-bottom: 12px;
}}

.event-footer {{
    display: flex;

    align-items: center;
    justify-content: space-between;

    gap: 10px;

    flex-wrap: wrap;

    font-size: 11px;

    color: #8A7D70;
}}

.confidence {{
    padding: 3px 7px;

    border-radius: 4px;

    font-size: 10px;
    font-weight: 700;
}}

.conf-high {{
    background: #E5EFE5;
    color: #35613A;
}}

.conf-medium {{
    background: #F4EBD2;
    color: #775C1D;
}}

.conf-low {{
    background: #F3DFDA;
    color: #7A3D32;
}}

.source {{
    font-family:
        "SFMono-Regular",
        Consolas,
        monospace;

    overflow: hidden;

    text-overflow: ellipsis;

    white-space: nowrap;

    max-width: 260px;
}}

/* ============================================================
   EMPTY
   ============================================================ */

.timeline-empty {{
    padding: 80px 20px;

    text-align: center;

    color: #6B5D50;

    font-size: 14px;
}}

/* ============================================================
   MOBILE
   ============================================================ */

@media (max-width: 760px) {{

    .timeline-shell {{
        padding:
            30px 14px 50px;
    }}

    .timeline {{
        padding-left: 34px;
    }}

    .timeline::before {{
        left: 14px;
        transform: none;
    }}

    .timeline-row {{
        grid-template-columns: 34px minmax(0, 1fr);

        min-height: 125px;
    }}

    .timeline-marker {{
        grid-column: 1;

        width: 34px;
    }}

    .timeline-left,
    .timeline-right {{
        grid-column: 2;

        padding-left: 16px;
        padding-right: 0;

        justify-content: flex-start;
    }}

    .timeline-left::after,
    .timeline-right::before {{
        left: 0;
        right: auto;

        width: 16px;
    }}

    .event-card {{
        width: 100%;
    }}

    .event-header {{
        align-items: flex-start;

        flex-direction: column;

        gap: 6px;
    }}

    .event-time {{
        white-space: normal;
    }}

    .event-counter {{
        margin-left: 0;
    }}
}}

</style>
</head>

<body>

<div class="timeline-controls">

    <button id="playButton" onclick="togglePlay()">
        ▶ Play
    </button>

    <button onclick="previousEvent()">
        ⏮ Prev
    </button>

    <button onclick="nextEvent()">
        ⏭ Next
    </button>

    <button onclick="resetTimeline()">
        ■ Reset
    </button>

    <span class="speed-label">
        Speed:
    </span>

    <select id="speedSelect" onchange="changeSpeed(this.value)">
        <option value="2000">0.5×</option>
        <option value="1000" selected>1×</option>
        <option value="500">2×</option>
        <option value="200">5×</option>
        <option value="100">10×</option>
    </select>

    <span
        id="eventCounter"
        class="event-counter">
    </span>

</div>

<div class="timeline-progress">
    <div
        id="progressFill"
        class="timeline-progress-fill">
    </div>
</div>

<div class="timeline-shell">

    <div id="timeline" class="timeline"></div>

    <div
        id="empty"
        class="timeline-empty"
        style="display:none;">
        No timeline events found for this case.
    </div>

</div>

<script>

const EVENTS = {events_json};

const COLORS = {{
    call: "#4A6FA5",
    transaction: "#B8860B",
    fir_filing: "#8B0000"
}};

let currentIndex = -1;
let playing = false;
let timer = null;
let interval = 1000;


function escapeHtml(value) {{
    const div = document.createElement("div");
    div.textContent = value ?? "";
    return div.innerHTML;
}}


function eventColor(type) {{
    return COLORS[type] || "#6B5D50";
}}


function eventLabel(type) {{
    return String(type || "event")
        .replaceAll("_", " ")
        .toUpperCase();
}}


function confidenceClass(confidence) {{

    const value =
        typeof confidence === "number"
            ? confidence
            : 0.85;

    if (value >= 0.8)
        return "conf-high";

    if (value >= 0.5)
        return "conf-medium";

    return "conf-low";
}}


function confidenceLabel(confidence) {{

    const value =
        typeof confidence === "number"
            ? confidence
            : 0.85;

    if (value >= 0.8)
        return "High";

    if (value >= 0.5)
        return "Medium";

    return "Low";
}}


function formatTimestamp(timestamp) {{

    if (!timestamp)
        return "";

    const value = String(timestamp);

    if (value.length >= 16)
        return value
            .replace("T", " ")
            .substring(0, 16);

    return value;
}}


function createCard(event, index) {{

    const color = eventColor(event.type);

    const card = document.createElement("div");

    card.className = "event-card";

    card.dataset.index = index;

    const header =
        document.createElement("div");

    header.className = "event-header";

    const badge =
        document.createElement("span");

    badge.className = "event-type";

    badge.style.color = color;

    badge.textContent =
        eventLabel(event.type);

    const time =
        document.createElement("span");

    time.className = "event-time";

    time.textContent =
        formatTimestamp(event.t);

    header.appendChild(badge);
    header.appendChild(time);


    const caption =
        document.createElement("div");

    caption.className =
        "event-caption";

    caption.textContent =
        event.caption ||
        `${{eventLabel(event.type)}} event`;


    const footer =
        document.createElement("div");

    footer.className =
        "event-footer";


    const confidence =
        document.createElement("span");

    confidence.className =
        "confidence " +
        confidenceClass(event.confidence);

    confidence.textContent =
        confidenceLabel(event.confidence);


    const source =
        document.createElement("span");

    source.className = "source";

    if (event.file) {{

        source.textContent =
            event.file +
            (event.line
                ? `:L${{event.line}}`
                : "");
    }}


    footer.appendChild(confidence);

    if (event.file)
        footer.appendChild(source);


    card.appendChild(header);
    card.appendChild(caption);
    card.appendChild(footer);


    card.addEventListener(
        "click",
        () => activateEvent(index, true)
    );

    return card;
}}


function renderTimeline() {{

    const timeline =
        document.getElementById("timeline");

    const empty =
        document.getElementById("empty");

    timeline.innerHTML = "";

    if (!EVENTS.length) {{

        timeline.style.display = "none";
        empty.style.display = "block";

        updateCounter();

        return;
    }}

    timeline.style.display = "flex";
    empty.style.display = "none";


    EVENTS.forEach((event, index) => {{

        const row =
            document.createElement("div");

        row.className =
            "timeline-row";

        row.id =
            `timeline-row-${{index}}`;


        const marker =
            document.createElement("div");

        marker.className =
            "timeline-marker";


        const dot =
            document.createElement("div");

        dot.className =
            "marker-dot";

        dot.style.borderColor =
            eventColor(event.type);


        marker.appendChild(dot);


        const card =
            createCard(event, index);


        if (index % 2 === 0) {{

            const left =
                document.createElement("div");

            left.className =
                "timeline-left";

            left.appendChild(card);

            row.appendChild(left);
            row.appendChild(marker);

            const emptyRight =
                document.createElement("div");

            emptyRight.style.gridColumn = "3";

            row.appendChild(emptyRight);

        }} else {{

            const emptyLeft =
                document.createElement("div");

            emptyLeft.style.gridColumn = "1";

            row.appendChild(emptyLeft);
            row.appendChild(marker);

            const right =
                document.createElement("div");

            right.className =
                "timeline-right";

            right.appendChild(card);

            row.appendChild(right);
        }}


        timeline.appendChild(row);
    }});

    updateCounter();
}}


function activateEvent(index, scroll) {{

    if (
        index < 0 ||
        index >= EVENTS.length
    )
        return;


    document
        .querySelectorAll(".timeline-row.active")
        .forEach(row => {{
            row.classList.remove("active");

            const dot =
                row.querySelector(".marker-dot");

            if (dot)
                dot.style.background = "#F7F2E8";
        }});


    const row =
        document.getElementById(
            `timeline-row-${{index}}`
        );

    if (!row)
        return;


    row.classList.add("active");


    const dot =
        row.querySelector(".marker-dot");

    if (dot) {{

        dot.style.background =
            eventColor(EVENTS[index].type);
    }}


    currentIndex = index;

    const percentage =
        ((index + 1) / EVENTS.length) * 100;

    document
        .getElementById("progressFill")
        .style.width =
        percentage + "%";


    updateCounter();


    if (scroll) {{

        row.scrollIntoView({{
            behavior: "smooth",
            block: "center"
        }});
    }}
}}


function updateCounter() {{

    const counter =
        document.getElementById(
            "eventCounter"
        );

    if (!EVENTS.length) {{

        counter.textContent =
            "0 events";

        return;
    }}


    if (currentIndex < 0) {{

        counter.textContent =
            `${{EVENTS.length}} event${{
                EVENTS.length === 1 ? "" : "s"
            }} in this case`;

        return;
    }}


    counter.textContent =
        `Event ${{currentIndex + 1}} of ${{EVENTS.length}}`;
}}


function nextEvent() {{

    if (!EVENTS.length)
        return;

    const next =
        currentIndex + 1;

    if (next < EVENTS.length)
        activateEvent(next, true);
}}


function previousEvent() {{

    if (currentIndex > 0)
        activateEvent(
            currentIndex - 1,
            true
        );
}}


function resetTimeline() {{

    stopPlayback();

    document
        .querySelectorAll(".timeline-row.active")
        .forEach(row => {{
            row.classList.remove("active");

            const dot =
                row.querySelector(".marker-dot");

            if (dot)
                dot.style.background =
                    "#F7F2E8";
        }});


    currentIndex = -1;

    document
        .getElementById("progressFill")
        .style.width = "0%";

    updateCounter();

    window.scrollTo({{
        top: 0,
        behavior: "smooth"
    }});
}}


function stopPlayback() {{

    if (timer) {{

        clearInterval(timer);
        timer = null;
    }}

    playing = false;

    const button =
        document.getElementById(
            "playButton"
        );

    button.textContent =
        "▶ Play";

    button.classList.remove("active");
}}


function togglePlay() {{

    if (playing) {{

        stopPlayback();

        return;
    }}


    if (!EVENTS.length)
        return;


    if (
        currentIndex >=
        EVENTS.length - 1
    ) {{
        resetTimeline();
    }}


    playing = true;

    const button =
        document.getElementById(
            "playButton"
        );

    button.textContent =
        "⏸ Pause";

    button.classList.add("active");


    timer = setInterval(() => {{

        if (
            currentIndex >=
            EVENTS.length - 1
        ) {{

            stopPlayback();

            return;
        }}

        activateEvent(
            currentIndex + 1,
            true
        );

    }}, interval);
}}


function changeSpeed(value) {{

    interval =
        parseInt(value, 10);

    if (playing) {{

        stopPlayback();

        togglePlay();
    }}
}}


renderTimeline();

</script>

</body>
</html>
"""