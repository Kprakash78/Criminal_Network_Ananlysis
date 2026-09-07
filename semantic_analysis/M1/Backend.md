# M1 — Video / VLM — Backend / Implementation Guide

## 1. Concepts you need (in order, don't over-study)
1. **Frame extraction/sampling**: same idea as M3 — pull a manageable subset of frames
   from the video, not every frame. Coordinate the exact interval with M3 (see PRD §7).
2. **CLIP/SigLIP basics**: a model that maps both images and text into the same
   embedding space, so cosine similarity between a text query's embedding and an
   image's embedding measures semantic relatedness. You don't need to understand the
   contrastive training objective — just that image and text embeddings are directly
   comparable.
3. **Zero-shot classification**: using CLIP's text-image similarity against a fixed
   list of candidate labels to produce a rough "what's in this frame" description
   without any captioning model — this is your cheap MVP fallback for descriptions.
4. **(Optional) Image captioning**: a separate model class (e.g. BLIP-2) that generates
   free-text descriptions directly, rather than classifying against fixed labels.
   Better quality, more compute, only pursue if MVP is done early.

You do NOT need to learn: CLIP's training procedure, transformer attention internals,
or how to fine-tune a VLM.

## 2. Libraries and installs
```bash
pip install open_clip_torch      # CLIP/SigLIP checkpoints, easy to swap between them
pip install torch torchvision
pip install opencv-python
pip install ffmpeg-python         # or shell out to the ffmpeg binary directly
# Optional, only if pursuing real captioning:
pip install transformers          # for BLIP-2 via HuggingFace
```
Recommended pretrained checkpoint: `ViT-B-32` CLIP (fast, good enough for MVP) or a
SigLIP checkpoint via `open_clip` if slightly better zero-shot accuracy is worth the
minor extra compute. Don't spend time benchmarking multiple checkpoints — pick one on
day 1 (ViT-B-32 is the safe default) and move on; swapping later is a one-line change
if it turns out to matter.

## 3. Frame sampling strategy
- Use the **same interval convention as M3** (fixed time interval, e.g. every 1s) —
  coordinate directly with whoever's building M3 so you're either sharing one
  extraction pass or at least aligned on timestamps. Don't let each module invent its
  own sampling rate independently; it makes downstream aggregation across modalities
  needlessly painful.
- FFmpeg CLI extraction, same as M3's approach:
  `ffmpeg -i in.mp4 -vf fps=1 out_%04d.jpg`
- Store frames under `/data/frames/<video_id>/` (shared location — again, ideally one
  extraction pass feeds both M1 and M3).

## 4. Minimal code example (CLIP embedding + zero-shot description per frame)
```python
import open_clip
import torch
from PIL import Image
import glob, os, uuid

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai"
)
tokenizer = open_clip.get_tokenizer("ViT-B-32")
model.eval()

# Small candidate label set for the cheap zero-shot description fallback.
# Tune this list to your actual demo video content, not a generic ontology.
CANDIDATE_LABELS = [
    "a person", "a motorcycle", "a kitchen", "an air fryer", "a chef cooking",
    "someone repairing a vehicle", "an outdoor scene", "an indoor scene",
    "a person wearing a helmet", "a workshop",
]
label_tokens = tokenizer(CANDIDATE_LABELS)
with torch.no_grad():
    label_embeds = model.encode_text(label_tokens)
    label_embeds /= label_embeds.norm(dim=-1, keepdim=True)

def embed_and_describe(frame_path, top_k=3):
    img = preprocess(Image.open(frame_path)).unsqueeze(0)
    with torch.no_grad():
        img_embed = model.encode_image(img)
        img_embed /= img_embed.norm(dim=-1, keepdim=True)
        sims = (img_embed @ label_embeds.T).squeeze(0)
        top_idx = sims.topk(top_k).indices.tolist()
    description = ", ".join(CANDIDATE_LABELS[i] for i in top_idx)
    return img_embed.squeeze(0).numpy(), description

def process_video_frames(video_id, frame_dir):
    records = []
    for frame_path in sorted(glob.glob(os.path.join(frame_dir, "frame_*.jpg"))):
        ts = float(os.path.basename(frame_path).replace("frame_", "").replace(".jpg", ""))
        embedding, description = embed_and_describe(frame_path)
        embedding_id = f"vis_{video_id}_{uuid.uuid4().hex[:8]}"
        records.append({
            "video_id": video_id,
            "timestamp": ts,
            "visual": {"description": description, "embedding_id": embedding_id},
            "_embedding_vector": embedding,  # handed to FAISS writer, not stored in JSON
        })
    return records
```
Swapping to a real captioning model later (optional tier) only changes how
`description` is produced — the embedding path and downstream schema stay identical,
so this is a safe MVP-first choice.

## 5. Timestamp handling
Identical convention to M3 (see M3's BACKEND.md §4): filename encodes seconds directly
(`frame_23.5.jpg`), timestamps always stored as seconds (float), never frame indices —
this is what lets M1's, M2's, and M3's records line up without a conversion step.

## 6. Batching / GPU-CPU considerations
- `open_clip` supports batched image encoding (`model.encode_image` on a stacked
  tensor of multiple frames) — meaningfully faster on GPU, still worth batching in
  groups of 8–16 even on CPU to amortize model call overhead.
- ViT-B-32 CLIP runs acceptably on CPU for a hackathon-scale corpus; test actual
  per-video throughput on day 1 (same advice as M2/M3) so you know your real budget.
- If pursuing the optional captioning-model upgrade (BLIP-2), budget meaningfully more
  compute — it's a heavier model than CLIP embedding alone. Don't attempt it without
  GPU access unless processing time is a non-issue for your corpus size.

## 7. Aggregating frame-level embeddings into segment-level records
Same spirit as M3's dedup problem, different signal: instead of a fixed 30-frame dump
per continuous shot, average (or take the centroid of) embeddings across consecutive
frames whose cosine similarity to each other stays above a threshold (e.g. >0.9),
treating that run as one segment. This keeps segment boundaries semantically
meaningful rather than arbitrary time slices.
```python
def aggregate_to_segments(frame_records, video_id, sim_threshold=0.9):
    import numpy as np
    segments = []
    current = None
    for rec in frame_records:
        vec = rec["_embedding_vector"]
        if current is None:
            current = {"video_id": video_id, "start_ts": rec["timestamp"],
                        "end_ts": rec["timestamp"], "_vectors": [vec],
                        "description": rec["visual"]["description"]}
        else:
            centroid = np.mean(current["_vectors"], axis=0)
            sim = float(np.dot(centroid, vec) / (np.linalg.norm(centroid) * np.linalg.norm(vec)))
            if sim >= sim_threshold:
                current["end_ts"] = rec["timestamp"]
                current["_vectors"].append(vec)
            else:
                segments.append(current)
                current = {"video_id": video_id, "start_ts": rec["timestamp"],
                            "end_ts": rec["timestamp"], "_vectors": [vec],
                            "description": rec["visual"]["description"]}
    if current:
        segments.append(current)
    return segments
```
This is intentionally simple — a genuine shot-boundary-detection algorithm is the
optional upgrade (PRD §3.2) if time allows, not a prerequisite for MVP.

## 8. Testing methodology
1. **Standalone retrieval sanity check** (do this before M4 exists): embed a handful of
   hand-written text queries with CLIP's text encoder, compute cosine similarity
   directly against your stored visual embeddings for 2–3 demo videos, and confirm the
   top match is intuitively correct. This validates the whole M1 pipeline in isolation.
2. **Description spot-check**: for 3 demo videos, read the generated descriptions
   against the actual frames — are they roughly right, not wildly off (e.g. describing
   an indoor kitchen scene as "an outdoor scene" would be a real bug to chase).
3. **Aggregation check**: confirm segment boundaries roughly match real scene changes
   by eye, and that one continuous shot doesn't produce dozens of near-duplicate
   segments.
4. **Schema validation**: confirm output matches `COMMON_DATA_CONTRACT.md`'s `visual`
   field shape exactly.
5. **Cross-module timestamp check**: pick a segment, confirm its `start_ts`/`end_ts`
   plausibly overlaps with M3's object detections and M2's transcript for the same
   real-world moment in the video — this is the first real integration smoke test
   across M1/M2/M3, worth doing as soon as all three have some output to compare.

## 9. Common failure modes + debugging checklist
| Symptom | Likely cause | Fix |
|---|---|---|
| Descriptions consistently wrong/generic | Candidate label set too narrow or mismatched to actual video content | Expand/tune `CANDIDATE_LABELS` to reflect your actual demo videos, not a generic list |
| Retrieval sanity check returns wrong segment | Embedding normalization bug, or text/image embeddings not from the same checkpoint | Confirm both `encode_image` and `encode_text` use the identical loaded model, and both are L2-normalized before the dot product |
| Segment boundaries too fragmented or too coarse | `sim_threshold` in §7 not tuned to your content | Test at 0.85 vs 0.9 vs 0.95, pick what best matches real scene changes by eye |
| Slow processing | No batching, or unnecessarily heavy checkpoint | Batch frames per §6; confirm you're using ViT-B-32, not a much larger checkpoint, unless GPU headroom is confirmed |
| Timestamps misaligned with M2/M3 | Different sampling interval or units mismatch | Confirm the shared interval convention from PRD §7 is actually being used, and everything's in seconds |

## 10. Integration contract recap
See PRD §7. M1 extracts its own frames if needed (ideally shared with M3), outputs
segment-level `visual` records matching the common contract, and coordinates with M4
on who owns the actual FAISS write for the visual index.