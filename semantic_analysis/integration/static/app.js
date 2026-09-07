/**
 * PS2614 Multimodal Video Search — Frontend Client Script
 * Handles search queries, API calls, segment rendering, and precise HTML5 video timestamp seeking.
 */

document.addEventListener("DOMContentLoaded", () => {
  initCorpusVideos();

  // Add event listener to update video duration when metadata loads
  const player = document.getElementById("videoPlayer");
  player.addEventListener("loadedmetadata", () => {
    const durSpan = document.getElementById("videoDuration");
    if (durSpan && !isNaN(player.duration)) {
      durSpan.textContent = formatTime(player.duration);
    }
  });

  // Theme toggle logic
  const themeToggleBtn = document.querySelector(".theme-toggle");
  if (themeToggleBtn) {
    themeToggleBtn.addEventListener("click", () => {
      document.body.classList.toggle("dark-mode");
    });
  }
});

let currentResults = [];
let activeSegmentId = null;
let currentVideos = {};

function formatTime(seconds) {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

/** Fetch available videos from API and populate the dropdown selector */
async function initCorpusVideos() {
  const selectEl = document.getElementById("videoSelect");
  try {
    const res = await fetch("/videos");
    const data = await res.json();

    selectEl.innerHTML = "";
    if (data.videos && data.videos.length > 0) {
      data.videos.forEach((vid) => {
        const opt = document.createElement("option");
        opt.value = vid.video_id;
        opt.textContent = `${vid.filename} (${vid.size_mb} MB)`;
        selectEl.appendChild(opt);
        currentVideos[vid.video_id] = vid;
      });

      // Load first video into player
      loadSelectedVideo();
    } else {
      selectEl.innerHTML = '<option value="">No videos in corpus</option>';
    }
  } catch (err) {
    console.error("Failed to fetch video corpus:", err);
    selectEl.innerHTML = '<option value="">Error loading corpus</option>';
  }
}

/** Change active video in player */
function loadSelectedVideo() {
  const selectEl = document.getElementById("videoSelect");
  const videoId = selectEl.value;
  if (!videoId) return;

  const player = document.getElementById("videoPlayer");
  const vidMeta = document.getElementById("currentVidId");
  const sizeMeta = document.getElementById("videoSize");
  
  if (vidMeta) vidMeta.textContent = currentVideos[videoId]?.filename || videoId;
  if (sizeMeta) sizeMeta.textContent = `${currentVideos[videoId]?.size_mb || 0} MB`;
  
  player.src = `/video/${videoId}`;
}

/** Set query from quick suggestion chip */
function setQuery(text) {
  const inputEl = document.getElementById("queryInput");
  inputEl.value = text;
  executeSearch();
}

/** Execute multi-modal search query */
async function executeSearch() {
  const inputEl = document.getElementById("queryInput");
  const query = inputEl.value.trim();
  if (!query) return;

  const loadingEl = document.getElementById("loadingState");
  const listEl = document.getElementById("resultsList");
  const countEl = document.getElementById("resultsCount");

  loadingEl.classList.remove("hidden");
  listEl.innerHTML = "";
  countEl.textContent = "Searching...";

  try {
    const res = await fetch("/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: query, top_k: 5 }),
    });

    if (!res.ok) {
      throw new Error(`Server returned status ${res.status}`);
    }

    const data = await res.json();
    currentResults = data.results || [];

    // Render Answer & Evidence Bullets
    renderAnswerCard(data);

    // Render Candidate Segment Cards
    renderResultsList(currentResults);

    countEl.textContent = `${currentResults.length}`;

    // Automatically seek to top candidate segment if available
    if (currentResults.length > 0) {
      const topSeg = currentResults[0];
      playSegment(topSeg.video_id, topSeg.start_ts, topSeg.end_ts, topSeg.segment_id);
    }
  } catch (err) {
    console.error("Search failed:", err);
    listEl.innerHTML = `
      <div class="empty-state">
        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="text-purple opacity-50"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>
        <p style="color: #ef4444">Search failed: ${err.message}</p>
      </div>
    `;
    countEl.textContent = "Error";
  } finally {
    loadingEl.classList.add("hidden");
  }
}

/** Render AI Answer Card */
function renderAnswerCard(data) {
  const answerTextEl = document.getElementById("answerText");
  const badgeEl = document.getElementById("confidenceBadge");
  const evidenceSection = document.getElementById("evidenceSection");
  const bulletsEl = document.getElementById("evidenceBullets");
  const statusDot = document.getElementById("statusDot");

  answerTextEl.innerHTML = `<strong>Synthesis:</strong> ${data.answer || "No answer generated."}`;

  // Confidence badge formatting
  const conf = (data.confidence || "MEDIUM").toUpperCase();
  
  // Dynamic card heading
  const answerHeaderEl = document.querySelector("#answerCard .card-title h3");
  if (conf === "LOW") {
    answerHeaderEl.textContent = "[ LOW_CONFIDENCE_MATCH ]";
  } else {
    answerHeaderEl.textContent = "[ GROUNDED_SYNTHESIS ]";
  }

  badgeEl.textContent = `${conf} CONFIDENCE`;

  // Reset dot classes
  statusDot.className = "dot";
  if (data.no_match || conf === "NO_MATCH") {
    statusDot.classList.add("dot-idle");
  } else if (conf === "HIGH") {
    statusDot.classList.add("dot-high");
  } else if (conf === "LOW") {
    statusDot.classList.add("dot-low");
  } else {
    statusDot.classList.add("dot-medium");
  }

  // Evidence Bullets
  if (data.explanation_bullets && data.explanation_bullets.length > 0) {
    evidenceSection.classList.remove("hidden");
    bulletsEl.innerHTML = "";
    data.explanation_bullets.forEach((b) => {
      const li = document.createElement("li");
      li.textContent = b;
      bulletsEl.appendChild(li);
    });
  } else {
    evidenceSection.classList.add("hidden");
  }
}

/** Render Candidate Segments List */
function renderResultsList(results) {
  const listEl = document.getElementById("resultsList");
  listEl.innerHTML = "";

  if (!results || results.length === 0) {
    listEl.innerHTML = `
      <div class="empty-state">
        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="text-purple opacity-50"><rect x="2" y="2" width="20" height="20" rx="2.18" ry="2.18"></rect><line x1="7" y1="2" x2="7" y2="22"></line><line x1="17" y1="2" x2="17" y2="22"></line><line x1="2" y1="12" x2="22" y2="12"></line><line x1="2" y1="7" x2="7" y2="7"></line><line x1="2" y1="17" x2="7" y2="17"></line><line x1="17" y1="17" x2="22" y2="17"></line><line x1="17" y1="7" x2="22" y2="7"></line></svg>
        <p style="font-family: monospace;">[ NO DATA PRESENT ]</p>
      </div>
    `;
    return;
  }

  results.forEach((seg, idx) => {
    const card = document.createElement("div");
    card.className = "segment-card";
    card.id = `card_${seg.segment_id}`;
    card.onclick = () => playSegment(seg.video_id, seg.start_ts, seg.end_ts, seg.segment_id);

    const scores = seg.scores || {};
    const finalScore = (scores.final_score || 0).toFixed(3);
    const fillPercent = Math.min(100, Math.max(0, scores.final_score * 100)).toFixed(0);

    // Objects tags
    let objHtml = "";
    if (seg.objects && seg.objects.length > 0) {
      const objTags = seg.objects.map((o) => {
        const attrs = o.attributes && o.attributes.length ? ` (${o.attributes.join(", ")})` : "";
        return `<span class="tag">${o.label}${attrs}</span>`;
      }).join(" ");
      objHtml = `<div class="seg-detail-item"><span class="detail-label">Objects:</span><span class="tag-list">${objHtml || objTags}</span></div>`;
    }

    // OCR tags
    let ocrHtml = "";
    if (seg.ocr && seg.ocr.length > 0) {
      const ocrTags = seg.ocr.map((t) => `<span class="tag tag-ocr">${t}</span>`).join(" ");
      ocrHtml = `<div class="seg-detail-item"><span class="detail-label">OCR Text:</span><span class="tag-list">${ocrTags}</span></div>`;
    }

    card.innerHTML = `
      <div class="segment-top-row">
        <span class="segment-id-tag">#${idx + 1} — ${currentVideos[seg.video_id]?.filename || seg.video_id}</span>
        <span class="timestamp-badge">T=${seg.start_ts.toFixed(1)}s - ${seg.end_ts.toFixed(1)}s</span>
      </div>

      <div class="score-bar-wrapper">
        <div class="score-bar-bg">
          <div class="score-bar-fill" style="width: ${fillPercent}%"></div>
        </div>
        <span class="score-val">Score: ${finalScore}</span>
      </div>

      ${seg.visual_description ? `<div class="seg-detail-item"><span class="detail-label">Visual:</span> ${seg.visual_description}</div>` : ""}
      ${seg.audio_transcript ? `<div class="seg-detail-item"><span class="detail-label">Speech:</span> "${seg.audio_transcript}"</div>` : ""}
      ${objHtml}
      ${ocrHtml}

      <div class="scores-breakdown-chips">
        <span>Vis: ${(scores.visual_similarity || 0).toFixed(2)}</span>
        <span>• Aud: ${(scores.transcript_similarity || 0).toFixed(2)}</span>
        <span>• Obj: ${(scores.object_match || 0).toFixed(2)}</span>
        <span>• OCR: ${(scores.ocr_match || 0).toFixed(2)}</span>
      </div>
    `;

    listEl.appendChild(card);
  });
}

/**
 * Play video and jump to exact timestamp.
 * DEMO-CRITICAL: Uses onloadedmetadata / canplay listener to ensure player seeks AFTER metadata is loaded.
 */
function playSegment(videoId, startTs, endTs, segmentId) {
  const player = document.getElementById("videoPlayer");
  const banner = document.getElementById("seekBanner");
  const tsVal = document.getElementById("seekTsVal");
  const vidMeta = document.getElementById("currentVidId");
  const sizeMeta = document.getElementById("videoSize");

  // Highlight active segment card
  if (activeSegmentId) {
    const prevCard = document.getElementById(`card_${activeSegmentId}`);
    if (prevCard) prevCard.classList.remove("active-segment");
  }
  activeSegmentId = segmentId;
  const newCard = document.getElementById(`card_${segmentId}`);
  if (newCard) newCard.classList.add("active-segment");

  if (vidMeta) vidMeta.textContent = currentVideos[videoId]?.filename || videoId;
  if (sizeMeta) sizeMeta.textContent = `${currentVideos[videoId]?.size_mb || 0} MB`;
  
  if (tsVal) tsVal.textContent = `${startTs.toFixed(1)}s`;

  // Show seeking banner
  if (banner) {
    banner.classList.remove("hidden");
    setTimeout(() => banner.classList.add("hidden"), 3000);
  }

  const targetSrc = `/video/${videoId}`;
  const isSameVideo = player.src.endsWith(targetSrc) || player.src.includes(videoId);

  if (isSameVideo && player.readyState >= 1) {
    // Video is already loaded — seek directly
    player.currentTime = startTs;
    player.play().catch(() => {});
  } else {
    // Video changed or metadata not ready — set currentTime on metadata load
    player.src = targetSrc;

    const onMetadataLoaded = () => {
      player.currentTime = startTs;
      player.play().catch(() => {});
      player.removeEventListener("loadedmetadata", onMetadataLoaded);
    };

    player.addEventListener("loadedmetadata", onMetadataLoaded);
  }
}

/** Handle user video file upload via form input */
async function handleVideoUpload(event) {
  const file = event.target.files[0];
  if (!file) return;

  const btnText = event.target.parentElement.querySelector('.btn-text');
  const originalText = btnText ? btnText.textContent : "Upload Video";
  if (btnText) btnText.textContent = "Uploading...";

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch("/upload", {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      throw new Error(`Upload failed (Status ${res.status})`);
    }

    const data = await res.json();
    alert(`Success! Ingested video '${data.filename}' (${data.segments_count} segments created).`);

    // Reload video selector list
    await initCorpusVideos();
    const selectEl = document.getElementById("videoSelect");
    selectEl.value = data.video_id;
    loadSelectedVideo();
  } catch (err) {
    console.error("Video upload error:", err);
    alert(`Upload failed: ${err.message}`);
  } finally {
    if (btnText) btnText.textContent = originalText;
  }
}
