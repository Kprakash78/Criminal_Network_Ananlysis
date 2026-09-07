"""
M4 Retrieval — Ingestion Pipeline
Main entry point for ingesting multi-modal outputs (M1 visual, M2 audio, M3 objects/OCR),
joining them into unified segment records, updating FAISS indices, and saving metadata.jsonl.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import RetrievalConfig as RetrievalConfig, DEFAULT_CONFIG
from .index_manager import FAISSIndexManager
from .joiner import join_segment_records

log = logging.getLogger(__name__)


def run_ingestion(
    visual_records: Optional[List[Dict[str, Any]]] = None,
    audio_records: Optional[List[Dict[str, Any]]] = None,
    object_records: Optional[List[Dict[str, Any]]] = None,
    ocr_records: Optional[List[Dict[str, Any]]] = None,
    cfg: RetrievalConfig = DEFAULT_CONFIG,
    index_mgr: Optional[FAISSIndexManager] = None,
) -> List[Dict[str, Any]]:
    """
    Run ingestion workflow:
      1. Load input records (from parameters or data/metadata.jsonl)
      2. Join multi-modal signals into unified segment records
      3. Index embeddings into FAISS (visual & audio indices)
      4. Persist updated metadata.jsonl & id_map.json

    Returns list of unified segment records.
    """
    log.info("=== Starting M4 Retrieval Ingestion Pipeline ===")

    # 1. Load existing records if not provided
    if visual_records is None and cfg.metadata_jsonl.exists():
        visual_records = _read_jsonl_records(cfg.metadata_jsonl)
        log.info("Loaded %d existing records from %s for ingestion", len(visual_records), cfg.metadata_jsonl)

    if visual_records is None:
        visual_records = []
    if audio_records is None:
        audio_records = []
    if object_records is None:
        object_records = []
    if ocr_records is None:
        ocr_records = []

    # 2. Join records by video_id and overlap ratio
    unified_segments = join_segment_records(
        visual_records=visual_records,
        audio_records=audio_records,
        object_records=object_records,
        ocr_records=ocr_records,
        cfg=cfg,
    )

    if not unified_segments:
        log.warning("No unified segment records created during ingestion")
        return []

    # 3. M4 owns FAISS writes: update FAISS indices & id_map.json
    if index_mgr is None:
        index_mgr = FAISSIndexManager(cfg=cfg)

    indexed_vis = 0
    indexed_aud = 0

    for seg in unified_segments:
        # Visual embedding indexing
        vis_info = seg.get("visual") or {}
        if "_embedding_vector" in vis_info and "embedding_id" in vis_info:
            index_mgr.add_visual(vis_info["embedding_id"], vis_info["_embedding_vector"])
            indexed_vis += 1
        elif "_embedding_vector" in seg and "visual" in seg and "embedding_id" in seg["visual"]:
            index_mgr.add_visual(seg["visual"]["embedding_id"], seg["_embedding_vector"])
            indexed_vis += 1

        # Audio embedding indexing
        aud_info = seg.get("audio") or {}
        if "_embedding_vector" in aud_info and "embedding_id" in aud_info:
            index_mgr.add_audio(aud_info["embedding_id"], aud_info["_embedding_vector"])
            indexed_aud += 1

    index_mgr.save()
    log.info("Indexed %d visual vectors, %d audio vectors into FAISS", indexed_vis, indexed_aud)

    # 4. Save metadata.jsonl (stripping transient vector fields)
    _save_jsonl_records(unified_segments, cfg.metadata_jsonl)

    log.info("=== M4 Ingestion Pipeline Completed: %d unified segments saved ===", len(unified_segments))
    return unified_segments


def _read_jsonl_records(path: Path) -> List[Dict[str, Any]]:
    records = []
    if path.exists():
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return records


def _save_jsonl_records(records: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_map: Dict[str, Dict[str, Any]] = {}
    
    if path.exists():
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        rec = json.loads(line)
                        sid = rec.get("segment_id")
                        if sid:
                            existing_map[sid] = rec
                    except json.JSONDecodeError:
                        pass

    for rec in records:
        clean_rec = {k: v for k, v in rec.items() if not k.startswith("_")}
        if "visual" in clean_rec and isinstance(clean_rec["visual"], dict):
            clean_rec["visual"] = {k: v for k, v in clean_rec["visual"].items() if not k.startswith("_")}
        if "audio" in clean_rec and isinstance(clean_rec["audio"], dict):
            clean_rec["audio"] = {k: v for k, v in clean_rec["audio"].items() if not k.startswith("_")}
        
        sid = clean_rec.get("segment_id")
        if sid:
            existing_map[sid] = clean_rec

    with open(path, "w", encoding="utf-8") as fh:
        for rec in existing_map.values():
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
