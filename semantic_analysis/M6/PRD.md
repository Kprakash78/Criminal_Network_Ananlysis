# M6 — Full Stack / Integration — PRD

## 1. What M6 owns
The user-facing surface and the wiring that turns M1–M5's separately-built pieces into
one working demo: a FastAPI backend exposing upload/search/ask endpoints, and a
frontend (video upload, a search box, results with playable video jumping to the cited
timestamp, and the natural-language answer + explainability from M5). M6 does NOT
implement any of the ML/retrieval logic itself — it calls into M1–M5's existing
functions/pipelines. M6's job is orchestration and presentation, not algorithms.

## 2. Why this exists in the architecture
Everything M1–M5 built is only a demo if a judge can type a query, see a result, click
play, and watch the cited moment happen. M6 is what makes the system legible and
demoable in 60 seconds instead of requiring someone to read logs or run scripts on
stage. It's also the module most likely to reveal integration bugs across every other
module at once, since it's the first place all five pipelines get called in sequence
against real, uploaded video.

## 3. MVP scope (must ship by 24 Aug) vs optional
**MVP:**
- FastAPI backend with three endpoints:
  - `POST /upload` — accept a video file, run it through the ingestion pipeline
    (M1 → M2 → M3 → M4 ingestion, in whatever order those modules actually need),
    return a `video_id` once processing completes
  - `POST /search` — accept a raw text query, call M5's query parsing → M4's
    `search()` → M5's answer generation, return the final answer + cited
    segment(s) + explainability bullets + scores
  - `GET /video/{video_id}` — stream/serve the uploaded video file for playback
- A minimal frontend (plain HTML/JS is an acceptable MVP choice — React/Next.js only
  if time allows and someone on the team is already comfortable with it):
  - Upload form (or a pre-loaded demo video picker, since live-uploading during a demo
    is riskier than having the corpus pre-ingested)
  - Search box + submit
  - Result display: answer text, explainability bullets, a `<video>` element that
    seeks to the cited timestamp on load/click
  - Basic loading/error states (searching..., no results found, upload failed)
- Pipeline orchestration script/endpoint that runs a batch of demo videos through
  M1→M2→M3→M4 ingestion ahead of the actual live demo, so the live demo only exercises
  the fast `/search` path, not the slow `/upload` path, on stage.

**Optional / only if time remains, in priority order:**
1. Live upload during the demo (only if ingestion time is short enough — test this
   explicitly, don't assume) — riskier for a live demo but more impressive if it works
   reliably.
2. Polished frontend styling (React + a component library, animations, etc.) — only
   after the plain-functional version works end-to-end. Never trade a working demo for
   a prettier broken one.
3. Multi-result display (show top 3 candidates, not just the top-1 answer) so judges
   can see the ranking, not just the single best match.
4. A simple "why this ranked here" expandable view showing the raw `scores` object per
   result, for technically-curious judges.

## 4. Non-goals
- No user auth/accounts — single-demo-session app, no login flow needed.
- No production deployment concerns (load balancing, HTTPS certs, CDN) — local or
  simple single-instance hosting is enough for a hackathon demo.
- No mobile-responsive polish unless time allows — optimize for "works well on the
  demo laptop's screen," not general device support.
- No async job queue for ingestion — a synchronous (or simple background-task) call is
  fine at this scale; don't build Celery/RQ infrastructure for a few dozen videos.

## 5. Success criteria for the demo
- A judge can type one of the team's rehearsed queries into the search box and, within
  a few seconds, see an answer with a video player that jumps to the correct moment
  when played.
- The upload → ingestion path has been run successfully for the entire demo video
  corpus at least once, ahead of time, with no manual intervention needed to fix
  broken records.
- The UI clearly shows the explainability bullets from M5, not just a bare answer
  string — this is a key differentiator the team should be able to point to.
- No unhandled crash/blank-white-screen state for a query with no strong match — the
  "no strong match" path from M5 renders as a clear message, not an error.

## 6. Integration contract (what M6 promises/expects from M1–M5)
- M6 calls M1/M2/M3's ingestion functions (or a single orchestrated ingestion function
  if the team consolidates them) in whatever order produces valid input for M4's
  ingestion/join step — confirm the actual call order and function signatures with
  each module owner, don't assume the docs' example code matches exactly what got built.
- M6 calls M4's `search(structured_query, raw_query, top_k)` — but note the *query
  parsing* step (raw text → structured_query) is M5's job, so M6's `/search` endpoint
  actually calls: M5.parse_query() → M4.search() → M5.generate_answer(), in that order.
- M6 calls M5's `generate_answer()` with M4's returned segments to get the final
  natural-language answer + explainability bullets.
- M6 owns turning `video_id` + `start_ts`/`end_ts` from a returned segment into an
  actual playable moment in the `<video>` element (via the HTML5 video `currentTime`
  API) — this is the payoff moment of the whole system and deserves explicit testing,
  not an assumption that "obviously" a timestamp maps to `currentTime` correctly.
- M6 does not duplicate any ML/retrieval logic locally — if a piece of logic doesn't
  exist yet in M1–M5, that's a signal to go build/fix it in the owning module, not to
  reimplement a shortcut version inside the API layer.