"""
Cortex RAG — Semantic Chunker

Two-pass chunking strategy:
  1. Sentence-level semantic boundary detection via cosine similarity.
     When consecutive sentences drop below `similarity_threshold`, we
     treat that as a topic boundary and start a new chunk.
  2. Parent-child hierarchy: the child chunk (≈256 tokens) is indexed in
     Milvus for precise retrieval; the parent chunk (≈1 024 tokens) is
     stored and returned to the LLM for wider context.
     This is "small-to-big" retrieval and significantly boosts both
     precision and answer quality.
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from config import get_settings
from ingestion.document_loader import Document

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A single text chunk ready for embedding and storage."""
    chunk_id: str                # sha256(doc_id + chunk_index)
    doc_id: str                  # parent document ID
    source: str                  # original file path
    title: str                   # document title
    text: str                    # child chunk text (indexed in Milvus)
    parent_text: str             # parent chunk text (returned to LLM)
    chunk_index: int             # position within document
    start_char: int
    end_char: int
    metadata: dict = field(default_factory=dict)

    @staticmethod
    def make_id(doc_id: str, index: int) -> str:
        key = f"{doc_id}:{index}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]


class SemanticChunker:
    """
    Splits documents into semantically coherent chunks.

    Steps:
      1. Split text into sentences (regex + edge-case handling).
      2. Embed each sentence (cached in-memory per document).
      3. Compute rolling cosine similarity between adjacent sentences.
      4. Insert a boundary where similarity < threshold.
      5. Merge sentences within each boundary into child chunks
         (~`chunk_size_tokens` tokens).
      6. Group adjacent child chunks into parent chunks
         (~`parent_chunk_size_tokens` tokens).
    """

    def __init__(self, embedder=None) -> None:
        cfg = get_settings()
        self.threshold = cfg.semantic_similarity_threshold
        self.child_max_tokens = cfg.chunk_size_tokens
        self.parent_max_tokens = cfg.parent_chunk_size_tokens
        self.overlap_tokens = cfg.chunk_overlap_tokens
        self._embedder = embedder           # injected; lazy-loaded if None

    # ── Public ─────────────────────────────────────────────────

    def chunk_document(self, doc: Document) -> list[Chunk]:
        """Return all child chunks (each containing its parent text)."""
        sentences = self._split_sentences(doc.text)
        if not sentences:
            logger.warning("Empty document: %s", doc.source)
            return []

        embeddings = self._get_sentence_embeddings(sentences)
        boundaries = self._find_boundaries(sentences, embeddings)

        child_chunks = self._build_child_chunks(sentences, boundaries, doc)
        self._attach_parent_chunks(child_chunks, doc.text)

        logger.info(
            "Chunked '%s' → %d child chunks", doc.title, len(child_chunks)
        )
        return child_chunks

    def chunk_documents(self, docs: list[Document]) -> list[Chunk]:
        all_chunks: list[Chunk] = []
        for doc in docs:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks

    # ── Sentence splitting ─────────────────────────────────────

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """
        Sentence splitter that handles:
          - Abbreviations (Dr., Fig., U.S.)
          - Numbered lists
          - Paragraph breaks
        Falls back to NLTK punkt if available.
        """
        try:
            import nltk  # type: ignore
            try:
                return nltk.sent_tokenize(text)
            except LookupError:
                nltk.download("punkt", quiet=True)
                return nltk.sent_tokenize(text)
        except ImportError:
            pass

        # Regex-based fallback
        abbreviations = r"(?<!\b(?:Dr|Mr|Ms|Mrs|Prof|Fig|vs|etc|approx|U\.S|e\.g|i\.e))"
        pattern = abbreviations + r"(?<=[.!?])\s+(?=[A-Z])"
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]

    # ── Embedding ─────────────────────────────────────────────

    def _get_sentence_embeddings(self, sentences: list[str]) -> np.ndarray:
        """Embed sentences in batches. Returns (N, dim) float32 array."""
        embedder = self._get_embedder()
        cfg = get_settings()
        batch_size = cfg.embed_batch_size
        all_embs: list[np.ndarray] = []

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            embs = embedder.encode(
                batch,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            all_embs.append(embs)

        return np.vstack(all_embs).astype(np.float32)

    def _get_embedder(self):
        if self._embedder is None:
            from retrieval.embedder import Embedder
            self._embedder = Embedder()
        return self._embedder.model

    # ── Boundary detection ────────────────────────────────────

    def _find_boundaries(
        self,
        sentences: list[str],
        embeddings: np.ndarray,
    ) -> list[int]:
        """
        Returns indices (into `sentences`) where a new chunk should start.
        Index 0 is always a boundary.
        """
        if len(sentences) == 1:
            return [0]

        boundaries = [0]
        current_chunk_tokens = self._count_tokens(sentences[0])

        for i in range(1, len(sentences)):
            # Cosine similarity between adjacent sentence embeddings
            sim = float(
                np.dot(embeddings[i - 1], embeddings[i])
                / (np.linalg.norm(embeddings[i - 1]) * np.linalg.norm(embeddings[i]) + 1e-8)
            )
            token_count = self._count_tokens(sentences[i])

            # Force a boundary if:
            # (a) semantic similarity drops below threshold, OR
            # (b) current chunk would exceed child max tokens
            if (
                sim < self.threshold
                or current_chunk_tokens + token_count > self.child_max_tokens
            ):
                boundaries.append(i)
                current_chunk_tokens = token_count
            else:
                current_chunk_tokens += token_count

        return boundaries

    # ── Chunk construction ────────────────────────────────────

    def _build_child_chunks(
        self,
        sentences: list[str],
        boundaries: list[int],
        doc: Document,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        boundary_set = set(boundaries)
        current_sentences: list[str] = []
        char_cursor = 0

        for i, sentence in enumerate(sentences):
            if i in boundary_set and current_sentences:
                chunk_text = " ".join(current_sentences)
                start = doc.text.find(current_sentences[0], char_cursor)
                end = start + len(chunk_text)
                chunks.append(
                    Chunk(
                        chunk_id=Chunk.make_id(doc.doc_id, len(chunks)),
                        doc_id=doc.doc_id,
                        source=doc.source,
                        title=doc.title,
                        text=chunk_text,
                        parent_text="",   # filled in _attach_parent_chunks
                        chunk_index=len(chunks),
                        start_char=max(0, start),
                        end_char=end,
                        metadata={**doc.metadata, "doc_type": doc.doc_type},
                    )
                )
                char_cursor = max(0, end - self.overlap_tokens * 4)
                current_sentences = []

            current_sentences.append(sentence)

        # Flush remaining sentences
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            start = doc.text.find(current_sentences[0], char_cursor)
            end = start + len(chunk_text)
            chunks.append(
                Chunk(
                    chunk_id=Chunk.make_id(doc.doc_id, len(chunks)),
                    doc_id=doc.doc_id,
                    source=doc.source,
                    title=doc.title,
                    text=chunk_text,
                    parent_text="",
                    chunk_index=len(chunks),
                    start_char=max(0, start),
                    end_char=min(end, len(doc.text)),
                    metadata={**doc.metadata, "doc_type": doc.doc_type},
                )
            )

        return chunks

    def _attach_parent_chunks(
        self, chunks: list[Chunk], full_text: str
    ) -> None:
        """
        Assign parent_text to each child chunk.
        Parent = the window of text centred on the child chunk,
        up to `parent_max_tokens` tokens wide.
        """
        parent_chars = self.parent_max_tokens * 4  # rough char estimate

        for chunk in chunks:
            half = parent_chars // 2
            p_start = max(0, chunk.start_char - half)
            p_end = min(len(full_text), chunk.end_char + half)

            # Snap to nearest sentence boundary
            p_start = self._snap_left(full_text, p_start)
            p_end = self._snap_right(full_text, p_end)

            chunk.parent_text = full_text[p_start:p_end].strip()

    # ── Utilities ─────────────────────────────────────────────

    @staticmethod
    def _count_tokens(text: str) -> int:
        """Fast token approximation: 1 token ≈ 4 chars."""
        return max(1, len(text) // 4)

    @staticmethod
    def _snap_left(text: str, pos: int) -> int:
        """Move pos left to the start of the nearest sentence."""
        while pos > 0 and text[pos] not in ".!?\n":
            pos -= 1
        return pos + 1 if pos > 0 else 0

    @staticmethod
    def _snap_right(text: str, pos: int) -> int:
        """Move pos right to the end of the nearest sentence."""
        while pos < len(text) and text[pos] not in ".!?\n":
            pos += 1
        return min(pos + 1, len(text))
