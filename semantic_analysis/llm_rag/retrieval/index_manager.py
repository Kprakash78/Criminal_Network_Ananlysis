"""
M4 Retrieval — FAISS Index & ID Map Manager
Manages persistence, vector insertion, and inner-product (cosine) search for:
  - Visual FAISS index (512-dim CLIP ViT-B-32 vectors)
  - Audio FAISS index (384-dim Sentence-Transformers vectors)
  - Bi-directional ID mapping (embedding_id <-> row index) in id_map.json
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np

from .config import RetrievalConfig as RetrievalConfig, DEFAULT_CONFIG

log = logging.getLogger(__name__)


class FAISSIndexManager:
    """
    Manages FAISS IndexFlatIP vector indices and id_map.json persistence.
    Vectors are assumed L2-normalized so inner product equals cosine similarity.
    """

    def __init__(self, cfg: RetrievalConfig = DEFAULT_CONFIG):
        self.cfg = cfg

        # Initialize FAISS IndexFlatIP
        self.visual_index = faiss.IndexFlatIP(cfg.visual_dim)
        self.audio_index = faiss.IndexFlatIP(cfg.audio_dim)

        # Bi-directional ID maps
        # id_map["visual"][embedding_id] = row_idx
        # id_map["audio"][embedding_id] = row_idx
        self.id_map: Dict[str, Dict[str, int]] = {"visual": {}, "audio": {}}
        self.rev_id_map: Dict[str, Dict[int, str]] = {"visual": {}, "audio": {}}

        self.load()

    def load(self) -> None:
        """Load FAISS indices and id_map.json from disk if present."""
        data_dir = self.cfg.data_root
        data_dir.mkdir(parents=True, exist_ok=True)

        # 1. Load visual index
        vis_path = self.cfg.faiss_visual_index
        if vis_path.exists():
            try:
                self.visual_index = faiss.read_index(str(vis_path))
                log.info("Loaded visual FAISS index (%d vectors) from %s", self.visual_index.ntotal, vis_path)
            except Exception as exc:
                log.warning("Could not read visual FAISS index: %s", exc)

        # 2. Load audio index
        aud_path = self.cfg.faiss_audio_index
        if aud_path.exists():
            try:
                self.audio_index = faiss.read_index(str(aud_path))
                log.info("Loaded audio FAISS index (%d vectors) from %s", self.audio_index.ntotal, aud_path)
            except Exception as exc:
                log.warning("Could not read audio FAISS index: %s", exc)

        # 3. Load id_map.json
        id_map_path = self.cfg.id_map_json
        if id_map_path.exists():
            try:
                with open(id_map_path, "r", encoding="utf-8") as fh:
                    raw_map = json.load(fh)
                    self.id_map["visual"] = raw_map.get("visual", {})
                    self.id_map["audio"] = raw_map.get("audio", {})
                self._rebuild_reverse_maps()
                log.info("Loaded id_map.json (%d visual, %d audio entries)",
                         len(self.id_map["visual"]), len(self.id_map["audio"]))
            except Exception as exc:
                log.warning("Could not load id_map.json: %s", exc)

    def save(self) -> None:
        """Persist visual index, audio index, and id_map.json to disk."""
        data_dir = self.cfg.data_root
        data_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.visual_index, str(self.cfg.faiss_visual_index))
        faiss.write_index(self.audio_index, str(self.cfg.faiss_audio_index))

        with open(self.cfg.id_map_json, "w", encoding="utf-8") as fh:
            json.dump(self.id_map, fh, indent=2)

        log.info("Persisted FAISS indices and id_map.json to %s", data_dir)

    def _rebuild_reverse_maps(self) -> None:
        """Rebuild integer row_idx -> embedding_id lookup dicts."""
        self.rev_id_map["visual"] = {int(v): k for k, v in self.id_map["visual"].items()}
        self.rev_id_map["audio"] = {int(v): k for k, v in self.id_map["audio"].items()}

    def add_visual(self, embedding_id: str, vector: Any) -> int:
        """
        Add a 512-dim visual vector to the visual FAISS index.
        Idempotent: if embedding_id already exists in the index, returns the
        existing row index without re-inserting the vector.
        Returns the assigned (or pre-existing) row index.
        """
        if embedding_id in self.id_map["visual"]:
            existing_row = self.id_map["visual"][embedding_id]
            log.debug("Visual embedding_id '%s' already in FAISS (row %d); skipping add.", embedding_id, existing_row)
            return existing_row

        vector = np.array(vector, dtype=np.float32).reshape(1, -1)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        row_idx = self.visual_index.ntotal
        self.visual_index.add(vector)
        self.id_map["visual"][embedding_id] = row_idx
        self.rev_id_map["visual"][row_idx] = embedding_id
        return row_idx

    def add_audio(self, embedding_id: str, vector: Any) -> int:
        """
        Add a 384-dim audio vector to the audio FAISS index.
        Idempotent: if embedding_id already exists in the index, returns the
        existing row index without re-inserting the vector.
        Returns the assigned (or pre-existing) row index.
        """
        if embedding_id in self.id_map["audio"]:
            existing_row = self.id_map["audio"][embedding_id]
            log.debug("Audio embedding_id '%s' already in FAISS (row %d); skipping add.", embedding_id, existing_row)
            return existing_row

        vector = np.array(vector, dtype=np.float32).reshape(1, -1)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        row_idx = self.audio_index.ntotal
        self.audio_index.add(vector)
        self.id_map["audio"][embedding_id] = row_idx
        self.rev_id_map["audio"][row_idx] = embedding_id
        return row_idx

    def search_visual(
        self, query_vector: np.ndarray, top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Search visual FAISS index with L2-normalized query vector.
        Returns list of (embedding_id, similarity_score) tuples sorted by score descending.
        """
        if self.visual_index.ntotal == 0:
            return []

        k = min(top_k, self.visual_index.ntotal)
        vec = query_vector.astype(np.float32).reshape(1, -1)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        scores, indices = self.visual_index.search(vec, k)
        results: List[Tuple[str, float]] = []
        for sim, idx in zip(scores[0], indices[0]):
            if idx in self.rev_id_map["visual"]:
                emb_id = self.rev_id_map["visual"][idx]
                results.append((emb_id, float(sim)))
        return results

    def search_audio(
        self, query_vector: np.ndarray, top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Search audio FAISS index with L2-normalized query vector.
        Returns list of (embedding_id, similarity_score) tuples sorted by score descending.
        """
        if self.audio_index.ntotal == 0:
            return []

        k = min(top_k, self.audio_index.ntotal)
        vec = query_vector.astype(np.float32).reshape(1, -1)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        scores, indices = self.audio_index.search(vec, k)
        results: List[Tuple[str, float]] = []
        for sim, idx in zip(scores[0], indices[0]):
            if idx in self.rev_id_map["audio"]:
                emb_id = self.rev_id_map["audio"][idx]
                results.append((emb_id, float(sim)))
        return results
