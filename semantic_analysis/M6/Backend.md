# M6 — Full Stack / Integration — Backend / Implementation Guide

## 1. Concepts you need (in order, don't over-study)
1. **FastAPI basics**: routes, request/response models (Pydantic), file upload
   handling, serving static/media files. If you know Flask, this transfers almost
   directly.
2. **Calling into other modules' code**: importing M1–M5's functions directly (as a
   monorepo) is simpler than spinning up separate microservices for a hackathon —
   don't build inter-service HTTP calls between your own modules unless there's a
   concrete reason (e.g. different runtime environments per module).
3. **HTML5 `<video>` element + `currentTime`**: how to make a video element jump to a
   specific timestamp programmatically — this is the one genuinely new frontend
   concept for this module.
4. **Basic async/background tasks in FastAPI** (`BackgroundTasks` or just running
   ingestion synchronously) — only needed if `/upload` needs to not block the response
   while ingestion runs; for a hackathon, blocking is often simpler and fine.

You do NOT need to learn: a frontend framework's full ecosystem if plain HTML/JS gets
the job done, WebSocket-based streaming, or a production-grade file storage system.

## 2. Libraries and installs
```bash
pip install fastapi uvicorn python-multipart   # python-multipart is required for file uploads
```
Frontend: no build step required for the MVP — plain HTML + vanilla JS served as a
static file by FastAPI is a legitimate, fast choice. If the team wants React/Next.js
and someone's already comfortable with it, that's fine too, but don't spend day 1
setting up a frontend build pipeline if a static HTML page gets you to a working demo
faster.

## 3. FastAPI backend — minimal structure
```python
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import shutil, uuid, os

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

VIDEO_DIR = "/data/videos"
os.makedirs(VIDEO_DIR, exist_ok=True)

class SearchRequest(BaseModel):
    query: str

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    video_id = str(uuid.uuid4())[:8]
    video_path = os.path.join(VIDEO_DIR, f"{video_id}.mp4")
    with open(video_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Call into the ingestion pipeline -- exact call order/signatures depend on what
    # M1/M2/M3/M4 actually built; confirm with each module owner rather than assuming.
    run_full_ingestion(video_id, video_path)  # your integration glue function

    return {"video_id": video_id, "status": "ingested"}

@app.post("/search")
async def search(req: SearchRequest):
    structured_query = m5_parse_query(req.query)             # M5
    results = m4_search(structured_query, req.query, top_k=5) # M4
    answer = m5_generate_answer(req.query, results)            # M5

    if not results or results[0]["scores"]["final_score"] < 0.5:
        return {"answer": "No strong match found for this query.", "results": []}

    return {
        "answer": answer,
        "results": [
            {
                "video_id": r["video_id"],
                "start_ts": r["start_ts"],
                "end_ts": r["end_ts"],
                "scores": r["scores"],
            }
            for r in results[:3]
        ],
    }

@app.get("/video/{video_id}")
async def get_video(video_id: str):
    path = os.path.join(VIDEO_DIR, f"{video_id}.mp4")
    return FileResponse(path, media_type="video/mp4")
```
Keep `run_full_ingestion`, `m5_parse_query`, `m4_search`, `m5_generate_answer` as thin
wrappers that import and call each module's actual functions — this file should have
almost no logic of its own, just orchestration.

## 4. Ingestion orchestration
`run_full_ingestion(video_id, video_path)` needs to call, in order:
1. M1: extract frames (or share extraction with M3 if that's how the team wired it),
   embed, describe, aggregate into visual segments.
2. M2: extract audio, transcribe, chunk, embed into audio segments.
3. M3: extract frames (or reuse M1's), run YOLO+OCR, aggregate into object/OCR records.
4. M4: join M1/M2/M3's output into unified segments, write to FAISS + metadata store.

Confirm the *actual* function names/signatures each module owner built — the example
code in each module's own BACKEND.md is illustrative, not guaranteed to match 1:1 what
got implemented under time pressure. This orchestration function is exactly where
mismatches between modules' assumed interfaces get caught — expect to spend real
debugging time here, and budget for it (see PART N-equivalent day-by-day plan).

For the demo itself, prefer running this ingestion once, ahead of time, against the
whole demo video corpus via a small CLI script (`python ingest_all.py`) rather than
relying on the live `/upload` endpoint during the actual judged demo — see PRD §3's
MVP note on why live upload during the demo is the riskier path.

## 5. Frontend — minimal HTML/JS example
```html
<!DOCTYPE html>
<html>
<head><title>PS2614 Video Search</title></head>
<body>
  <input id="query" type="text" placeholder="Find videos where..." />
  <button onclick="search()">Search</button>
  <div id="answer"></div>
  <video id="player" controls width="480"></video>

  <script>
    async function search() {
      const query = document.getElementById("query").value;
      document.getElementById("answer").innerText = "Searching...";
      const res = await fetch("/search", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({query})
      });
      const data = await res.json();
      document.getElementById("answer").innerText = data.answer;

      if (data.results.length > 0) {
        const top = data.results[0];
        const player = document.getElementById("player");
        player.src = `/video/${top.video_id}`;
        player.onloadedmetadata = () => { player.currentTime = top.start_ts; };
      }
    }
  </script>
</body>
</html>
```
Key detail: `currentTime` must be set in the `onloadedmetadata` handler (or after
confirming the video's metadata has loaded), not immediately after setting `src` —
setting it too early is a common bug that silently does nothing because the video
hasn't loaded enough to seek yet.

## 6. Testing methodology
1. **Endpoint smoke test**: hit `/upload` with a real video file via `curl` or the
   FastAPI auto-generated `/docs` page, confirm it returns a `video_id` and doesn't
   error, before touching the frontend at all.
2. **Search smoke test**: hit `/search` directly (via `/docs` or `curl`) with one of
   the team's rehearsed demo queries, confirm the JSON response has a sensible answer
   and results array — this isolates backend bugs from frontend bugs.
3. **Timestamp-seek test**: manually load the frontend, run a search, confirm the
   video element actually jumps to the right moment when the page loads the result —
   this is the single most demo-critical interaction, test it explicitly and
   repeatedly, don't assume it "just works."
4. **Full corpus ingestion dry run**: run the batch ingestion script against the whole
   demo video corpus at least once, well before the deadline, and fix any videos that
   fail rather than discovering a broken video live during the demo.
5. **Demo rehearsal**: run through the exact sequence of queries planned for the judge
   demo, timed, at least once end-to-end through the actual UI (not just the API)
   the day before the deadline.

## 7. Common failure modes + debugging checklist
| Symptom | Likely cause | Fix |
|---|---|---|
| `/upload` hangs or times out | Ingestion pipeline is genuinely slow (Whisper/YOLO/CLIP all running synchronously in the request) | Pre-ingest the demo corpus via a CLI script ahead of time (§4) instead of relying on live upload during the demo |
| `/search` returns an error | A downstream module's function signature doesn't match what M6's wrapper assumes | Check the actual function signature in the real module code, not just its BACKEND.md example |
| Video doesn't seek to the right timestamp | `currentTime` set before video metadata loaded | Set it inside `onloadedmetadata`, per §5 |
| Video won't play at all | Wrong `media_type` in `FileResponse`, or file path mismatch between `/upload`'s save location and `/video/{id}`'s read location | Confirm both endpoints agree on the exact file path convention |
| CORS errors if frontend is served separately from the API | Frontend and backend on different origins/ports | Serve the frontend as static files from the same FastAPI app (§3's `StaticFiles` mount) to sidestep CORS entirely for the MVP |
| Search results look right in `/docs` but wrong in the UI | Frontend not reading the actual response shape correctly (e.g. `results[0]` vs `results` typo) | Console.log the raw response in the browser devtools before debugging further upstream |

## 8. Integration contract recap
See PRD §6. M6 orchestrates calls into M1–M5's actual functions in the correct order,
owns the upload/search/video-serving endpoints and the frontend, and does not
reimplement any ML/retrieval logic itself.