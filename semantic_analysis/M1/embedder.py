"""
M1 Video / VLM — CLIP Embedder & Zero-Shot Scene Classifier
Uses open_clip (ViT-B-32 by default) to generate normalized 512-dim visual embeddings
and zero-shot scene descriptions per sampled frame.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import open_clip
import torch
from PIL import Image

from .config import M1Config, DEFAULT_CONFIG

log = logging.getLogger(__name__)


class CLIPEmbedder:
    """
    Encapsulates CLIP model loading, image embedding extraction,
    text encoding, and zero-shot scene description generation.
    """

    def __init__(self, cfg: M1Config = DEFAULT_CONFIG):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        log.info(
            "Loading CLIP model '%s' (pretrained='%s') on device %s...",
            cfg.clip_model_name,
            cfg.clip_pretrained,
            self.device,
        )
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            cfg.clip_model_name, pretrained=cfg.clip_pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(cfg.clip_model_name)
        self.model.to(self.device)
        self.model.eval()

        self._candidate_labels = cfg.candidate_labels
        self._candidate_embeds = self._precompute_label_embeddings()

    def _precompute_label_embeddings(self) -> torch.Tensor:
        """Encode and L2-normalize candidate text labels for zero-shot classification."""
        tokens = self.tokenizer(self._candidate_labels).to(self.device)
        with torch.no_grad():
            embeds = self.model.encode_text(tokens)
            embeds = embeds / embeds.norm(dim=-1, keepdim=True)
        return embeds

    def encode_text(self, text: str | List[str]) -> np.ndarray:
        """
        Encode single text query or list of queries into L2-normalized numpy embedding(s).
        Returns 1D array if single str, 2D array if list[str].
        """
        is_single = isinstance(text, str)
        text_list = [text] if is_single else text
        tokens = self.tokenizer(text_list).to(self.device)
        with torch.no_grad():
            embeds = self.model.encode_text(tokens)
            embeds = embeds / embeds.norm(dim=-1, keepdim=True)
        np_embeds = embeds.cpu().numpy().astype(np.float32)
        return np_embeds[0] if is_single else np_embeds

    def process_frames(
        self, frames: List[Tuple[float, Path]]
    ) -> List[Dict[str, Any]]:
        """
        Process (timestamp, frame_path) pairs in batches:
        Extracts L2-normalized image embedding and zero-shot scene description.
        Returns list of dicts with 'timestamp', 'frame_path', 'embedding', 'description'.
        """
        if not frames:
            return []

        results: List[Dict[str, Any]] = []
        batch_size = self.cfg.batch_size

        for i in range(0, len(frames), batch_size):
            batch_items = frames[i : i + batch_size]
            batch_imgs = []
            valid_items = []

            for ts, path in batch_items:
                try:
                    img = Image.open(path).convert("RGB")
                    tensor = self.preprocess(img)
                    batch_imgs.append(tensor)
                    valid_items.append((ts, path))
                except Exception as exc:
                    log.warning("Failed to read image %s: %s; skipping", path, exc)

            if not batch_imgs:
                continue

            imgs_tensor = torch.stack(batch_imgs).to(self.device)
            with torch.no_grad():
                img_embeds = self.model.encode_image(imgs_tensor)
                img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)

                # Zero-shot classification similarities
                sims = img_embeds @ self._candidate_embeds.T

            img_embeds_np = img_embeds.cpu().numpy().astype(np.float32)
            sims_np = sims.cpu().numpy()

            top_k = min(self.cfg.description_top_k, len(self._candidate_labels))
            for idx, (ts, path) in enumerate(valid_items):
                top_indices = np.argsort(sims_np[idx])[::-1][:top_k]
                top_labels = [self._candidate_labels[j] for j in top_indices]
                description = ", ".join(top_labels)

                results.append(
                    {
                        "timestamp": ts,
                        "frame_path": str(path),
                        "embedding": img_embeds_np[idx],
                        "description": description,
                        "top_labels": top_labels,
                    }
                )

        log.info("Processed %d frame embeddings & descriptions", len(results))
        return results
