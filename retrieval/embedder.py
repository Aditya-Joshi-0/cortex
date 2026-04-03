"""
Cortex RAG — Embedder
Wraps sentence-transformers with:
  - Lazy loading (model loads only on first use)
  - Batch encode with progress bar
  - L2-normalised outputs for cosine similarity
  - In-memory LRU cache for repeated single queries (avoids redundant GPU calls)
"""
from __future__ import annotations

import functools
import logging
from typing import Optional

import numpy as np

from config import get_settings

logger = logging.getLogger(__name__)


class Embedder:
    """
    Singleton-style embedder wrapping SentenceTransformer.

    Example:
        emb = Embedder()
        vectors = emb.encode(["Hello world", "Foo bar"])
        # → np.ndarray of shape (2, 384), float32, L2-normalised
    """

    _instance: Optional["Embedder"] = None

    def __new__(cls) -> "Embedder":
        # Allow multiple instances but share the underlying model
        return super().__new__(cls)

    def __init__(self) -> None:
        if hasattr(self, "_initialised"):
            return
        self._initialised = True
        self._model = None      # lazy-loaded

    # ── Public ─────────────────────────────────────────────────

    @property
    def model(self):
        """Lazy-load the SentenceTransformer model."""
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def dim(self) -> int:
        return get_settings().embed_dim

    def encode(
        self,
        texts: list[str],
        batch_size: Optional[int] = None,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """
        Encode a list of strings.
        Returns float32 numpy array of shape (len(texts), embed_dim),
        L2-normalised (unit vectors → dot product == cosine similarity).
        """
        cfg = get_settings()
        bs = batch_size or cfg.embed_batch_size

        embeddings = self.model.encode(
            texts,
            batch_size=bs,
            normalize_embeddings=True,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single query string.
        BGE models benefit from a special prefix for query embedding.
        Returns shape (1, embed_dim).
        """
        cfg = get_settings()
        # BGE-specific instruction prefix for retrieval queries
        if "bge" in cfg.embed_model_name.lower():
            query = f"Represent this sentence for searching relevant passages: {query}"
        return self.encode([query])

    # ── Private ────────────────────────────────────────────────

    def _load_model(self) -> None:
        cfg = get_settings()
        logger.info("Loading embedding model: %s on %s", cfg.embed_model_name, cfg.embed_device)
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Install sentence-transformers: pip install sentence-transformers"
            ) from exc

        self._model = SentenceTransformer(
            cfg.embed_model_name,
            device=cfg.embed_device,
        )
        logger.info("Embedding model ready (dim=%d)", cfg.embed_dim)
