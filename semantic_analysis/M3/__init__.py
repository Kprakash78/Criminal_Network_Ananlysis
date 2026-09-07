# M3 — Vision Intelligence Module
# Turns sampled video frames into timestamped structured metadata:
#   object labels + confidence (YOLOv8), on-screen text (EasyOCR),
#   frame-quality gate (Laplacian variance), aggregated segment records.
# Output: append-only metadata.jsonl per COMMON_DATA_CONTRACT.md
