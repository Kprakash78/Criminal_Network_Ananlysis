"""
M1 Video / VLM — Writer & Indexer
Writes segment records matching COMMON_DATA_CONTRACT.md to data/metadata.jsonl.
Maintains FAISS visual index (data/faiss_visual.index) and mapping (data/id_map.json).

If a segment_id already exists in metadata.jsonl (e.g. created by M3/M2),
it updates/merges the 'visual' key in place instead of creating duplicate records.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from .config import M1Config, DEFAULT_CONFIG

log = logging.getLogger(__name__)

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    log.warning("faiss package not installed; FAISS index operations will use numpy fallback")


def write_segments_and_index(
    segments: List[Dict[str, Any]],
    cfg: M1Config = DEFAULT_CONFIG,
) -> Tuple[int, int]:
    """
    Write or merge segment records into cfg.metadata_jsonl, and add vector embeddings
    to cfg.faiss_visual_index & update cfg.id_map_json.

    Returns tuple (records_written_or_updated, vectors_indexed).
    """
    if not segments:
        log.info("No segments to write")
        return 0, 0

    jsonl_path = cfg.metadata_jsonl
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Update / Append metadata.jsonl
    records_count = _upsert_metadata_jsonl(segments, jsonl_path)

    # 2. Update FAISS Index & ID map
    vectors_count = _update_faiss_index(segments, cfg)

    return records_count, vectors_count


def _upsert_metadata_jsonl(segments: List[Dict[str, Any]], jsonl_path: Path) -> int:
    """Read existing lines, update or append segment records, write back cleanly."""
    existing_records: List[Dict[str, Any]] = []
    id_to_index: Dict[str, int] = {}

    if jsonl_path.exists():
        with open(jsonl_path, "r", encoding="utf-8") as fh:
            for idx, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    existing_records.append(obj)
                    sid = obj.get("segment_id")
                    if sid:
                        id_to_index[sid] = len(existing_records) - 1
                except json.JSONDecodeError as exc:
                    log.warning("Skipping bad JSON line: %s", exc)

    updated_or_added = 0
    for seg in segments:
        # Prepare clean record without private vector field
        record_to_save = {k: v for k, v in seg.items() if not k.startswith("_")}
        sid = record_to_save.get("segment_id")

        if sid and sid in id_to_index:
            # Merge visual field into existing record
            idx = id_to_index[sid]
            existing_records[idx]["visual"] = record_to_save.get("visual", {})
            updated_or_added += 1
        else:
            existing_records.append(record_to_save)
            if sid:
                id_to_index[sid] = len(existing_records) - 1
            updated_or_added += 1

    # Atomic rewrite
    tmp_path = jsonl_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as fh:
        for rec in existing_records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp_path.replace(jsonl_path)

    log.info("Wrote/updated %d records in %s", updated_or_added, jsonl_path)
    return updated_or_added


def _update_faiss_index(segments: List[Dict[str, Any]], cfg: M1Config) -> int:
    """Save/update vectors into FAISS visual index and update id_map.json."""
    vectors = [seg["_embedding_vector"] for seg in segments if "_embedding_vector" in seg]
    embedding_ids = [seg["visual"]["embedding_id"] for seg in segments if "visual" in seg and "embedding_id" in seg["visual"]]

    if not vectors or not embedding_ids:
        return 0

    vec_array = np.vstack(vectors).astype(np.float32)
    dim = vec_array.shape[1]

    # Load existing id_map.json
    id_map_path = cfg.id_map_json
    id_map_path.parent.mkdir(parents=True, exist_ok=True)
    id_map: Dict[str, Any] = {"visual": {}, "audio": {}}
    if id_map_path.exists():
        try:
            with open(id_map_path, "r", encoding="utf-8") as fh:
                id_map = json.load(fh)
        except Exception as exc:
            log.warning("Could not load existing id_map.json (%s), starting fresh", exc)

    if "visual" not in id_map:
        id_map["visual"] = {}

    index_path = cfg.faiss_visual_index

    if HAS_FAISS:
        if index_path.exists():
            try:
                index = faiss.read_index(str(index_path))
            except Exception:
                index = faiss.IndexFlatIP(dim)
        else:
            index = faiss.IndexFlatIP(dim)

        start_idx = index.ntotal
        index.add(vec_array)
        faiss.write_index(index, str(index_path))

        for i, emb_id in enumerate(embedding_ids):
            id_map["visual"][emb_id] = start_idx + i

    else:
        # Fallback numpy vector store if faiss is not installed
        np_index_path = index_path.with_suffix(".npy")
        if np_index_path.exists():
            existing_vecs = np.load(np_index_path)
            start_idx = existing_vecs.shape[0]
            new_vecs = np.vstack([existing_vecs, vec_array])
        else:
            start_idx = 0
            new_vecs = vec_array

        np.save(np_index_path, new_vecs)

        for i, emb_id in enumerate(embedding_ids):
            id_map["visual"][emb_id] = start_idx + i

    with open(id_map_path, "w", encoding="utf-8") as fh:
        json.dump(id_map, fh, indent=2)

    log.info("Indexed %d visual embedding vectors in %s", len(vectors), index_path)
    return len(vectors)
