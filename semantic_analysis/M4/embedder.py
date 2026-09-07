"""
M4 Retrieval — Query Embedder
Encodes natural language query string into:
  1. Visual vector (512-dim) via CLIP ViT-B-32 (exact checkpoint matching M1)
  2. Audio/transcript vector (384-dim) via SentenceTransformers all-MiniLM-L6-v2 (matching M2)
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import open_clip
import torch

from .config import M4Config, DEFAULT_CONFIG

log = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    log.warning("sentence_transformers package not available; audio query embedding will use fallback")


class QueryEmbedder:
    """
    Encapsulates text encoding models for visual (CLIP) and audio (SentenceTransformers) queries.
    """

    def __init__(self, cfg: M4Config = DEFAULT_CONFIG):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

        # 1. CLIP Text Embedder (512-dim)
        log.info("Loading CLIP model '%s' (pretrained='%s') for visual query embedding...",
                 cfg.clip_model_name, cfg.clip_pretrained)
        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            cfg.clip_model_name, pretrained=cfg.clip_pretrained
        )
        self.clip_tokenizer = open_clip.get_tokenizer(cfg.clip_model_name)
        self.clip_model.to(self.device)
        self.clip_model.eval()

        # 2. SentenceTransformers Text Embedder (384-dim)
        self.sentence_model: Optional[Any] = None
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                log.info("Loading SentenceTransformers model '%s' for transcript query embedding...",
                         cfg.sentence_model_name)
                self.sentence_model = SentenceTransformer(cfg.sentence_model_name, device=str(self.device))
            except Exception as exc:
                log.warning("Failed to load SentenceTransformers model (%s); audio query embedding fallback active", exc)

    def embed_visual_query(self, raw_query: str) -> np.ndarray:
        """
        Embed text query using CLIP text encoder.
        Returns L2-normalized 512-dim float32 numpy vector.
        """
        tokens = self.clip_tokenizer([raw_query]).to(self.device)
        with torch.no_grad():
            embed = self.clip_model.encode_text(tokens)
            embed = embed / embed.norm(dim=-1, keepdim=True)
        vec = embed.cpu().numpy().squeeze(0).astype(np.float32)
        return vec

    def embed_audio_query(self, raw_query: str) -> np.ndarray:
        """
        Embed text query using SentenceTransformers all-MiniLM-L6-v2.
        Returns L2-normalized 384-dim float32 numpy vector.
        """
        if self.sentence_model is not None:
            embed = self.sentence_model.encode(raw_query, convert_to_numpy=True, normalize_embeddings=True)
            return embed.squeeze().astype(np.float32)

        # Fallback zero vector if model is unavailable
        log.warning("Audio query embedding model not loaded; returning zero vector")
        return np.zeros(self.cfg.audio_dim, dtype=np.float32)
