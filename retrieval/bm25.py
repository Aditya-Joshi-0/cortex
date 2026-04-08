"""
Cortex RAG — BM25 Sparse Retriever

Why BM25 alongside dense vectors?
  Dense embeddings excel at semantic similarity but blur exact terms.
  A query for "PagedAttention" or "RLHF" may retrieve semantically
  adjacent but lexically different chunks. BM25 catches exact keyword
  matches that dense search misses. The two signals are complementary
  and combining them (via RRF) consistently outperforms either alone.

Implementation
  - rank_bm25 (Okapi BM25) on preprocessed token lists
  - Index is built at ingestion time and persisted as a pickle
  - Corpus (chunk_id → text + metadata) stored alongside the index
  - Preprocessing: lowercase, stop-word removal, optional stemming
"""
from __future__ import annotations

import logging
import pickle
import re
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from config import get_settings

from retrieval.dense import RetrievedChunk

logger = logging.getLogger(__name__)

cfg = get_settings()
# Path where the BM25 index is persisted between runs
_DEFAULT_INDEX_PATH = Path(cfg.bm25_path)

# Common English stop words (avoids NLTK download requirement)
_STOP_WORDS = frozenset("""
a an the is are was were be been being have has had do does did will would
could should may might shall can of in on at to for with by from as into
through during before after above below between out off over under again
further then once here there when where why how all both each few more most
other some such no nor not only own same so than too very just because if
or and but i me my we our you your he she it they them their what which who
this that these those am""".split())


@dataclass
class BM25Corpus:
    """Serialisable corpus stored alongside the BM25 index."""
    chunk_ids:    list[str]     = field(default_factory=list)
    doc_ids:      list[str]     = field(default_factory=list)
    sources:      list[str]     = field(default_factory=list)
    titles:       list[str]     = field(default_factory=list)
    texts:        list[str]     = field(default_factory=list)   # child text
    parent_texts: list[str]     = field(default_factory=list)
    chunk_indices:list[int]     = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.chunk_ids)

    def add(self, chunk: "Chunk") -> None:  # type: ignore[name-defined]
        self.chunk_ids.append(chunk.chunk_id)
        self.doc_ids.append(chunk.doc_id)
        self.sources.append(chunk.source)
        self.titles.append(chunk.title)
        self.texts.append(chunk.text)
        self.parent_texts.append(chunk.parent_text)
        self.chunk_indices.append(chunk.chunk_index)

    def get_chunk_ids(self) -> set[str]:
        return set(self.chunk_ids)


class BM25Retriever:
    """
    Sparse keyword retriever using Okapi BM25.

    Usage:
        retriever = BM25Retriever()
        retriever.add_chunks(chunks)          # at ingestion time
        results = retriever.search("RLHF fine-tuning", top_k=15)
    """

    def __init__(self, index_path: str | Path = _DEFAULT_INDEX_PATH) -> None:
        self._index_path = Path(index_path)
        self._index = None       # rank_bm25.BM25Okapi — lazy built
        self._corpus = BM25Corpus()
        self._load_if_exists()

    # ── Public API ─────────────────────────────────────────────

    def add_chunks(self, chunks: list) -> int:
        """
        Add new chunks to the BM25 index.
        Skips chunks already in the corpus (deduplication by chunk_id).
        Rebuilds and persists the index after insertion.
        Returns number of chunks actually added.
        """
        existing = self._corpus.get_chunk_ids()
        new_chunks = [c for c in chunks if c.chunk_id not in existing]

        if not new_chunks:
            logger.info("BM25: all chunks already indexed.")
            return 0

        for chunk in new_chunks:
            self._corpus.add(chunk)

        self._rebuild_index()
        self._save()
        logger.info("BM25: added %d chunks (total: %d)", len(new_chunks), len(self._corpus))
        return len(new_chunks)

    def search(self, query: str, top_k: int = 15) -> list[RetrievedChunk]:
        """
        Keyword search. Returns RetrievedChunk list sorted by BM25 score desc.
        Scores are normalised to [0, 1] relative to the top result.
        """
        if self._index is None or len(self._corpus) == 0:
            logger.warning("BM25 index is empty — returning no results.")
            return []

        tokens = self._tokenise(query)
        if not tokens:
            return []

        raw_scores = self._index.get_scores(tokens)

        # Get top-k indices by score
        top_indices = sorted(
            range(len(raw_scores)),
            key=lambda i: raw_scores[i],
            reverse=True,
        )[:top_k]

        max_score = raw_scores[top_indices[0]] if top_indices else 1.0
        if max_score == 0:
            return []

        results: list[RetrievedChunk] = []
        for idx in top_indices:
            score = raw_scores[idx]
            if score <= 0:
                break
            results.append(
                RetrievedChunk(
                    chunk_id=self._corpus.chunk_ids[idx],
                    doc_id=self._corpus.doc_ids[idx],
                    source=self._corpus.sources[idx],
                    title=self._corpus.titles[idx],
                    text=self._corpus.texts[idx],
                    parent_text=self._corpus.parent_texts[idx],
                    chunk_index=self._corpus.chunk_indices[idx],
                    score=float(score / max_score),  # normalise
                    retriever="bm25",
                )
            )
        return results

    def corpus_size(self) -> int:
        return len(self._corpus)

    # ── Index management ───────────────────────────────────────

    def _rebuild_index(self) -> None:
        """Rebuild BM25 index from current corpus."""
        try:
            from rank_bm25 import BM25Okapi  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Install rank-bm25: pip install rank-bm25") from exc

        tokenised_corpus = [self._tokenise(text) for text in self._corpus.texts]
        self._index = BM25Okapi(tokenised_corpus)
        logger.debug("BM25 index rebuilt over %d documents.", len(self._corpus))

    def _save(self) -> None:
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._index_path, "wb") as fh:
            pickle.dump({"corpus": self._corpus, "index": self._index}, fh)
        logger.debug("BM25 index persisted to %s", self._index_path)

    def _load_if_exists(self) -> None:
        if not self._index_path.exists():
            return
        try:
            with open(self._index_path, "rb") as fh:
                data = pickle.load(fh)
            self._corpus = data["corpus"]
            self._index  = data["index"]
            logger.info(
                "BM25 index loaded from %s (%d chunks)",
                self._index_path, len(self._corpus)
            )
        except Exception as exc:
            logger.warning("Failed to load BM25 index (%s) — starting fresh.", exc)

    # ── Text preprocessing ─────────────────────────────────────

    @staticmethod
    def _tokenise(text: str) -> list[str]:
        """
        Lowercase → remove punctuation → split → strip stop words.
        Keeps numbers and hyphenated terms (e.g. "multi-head").
        """
        text = text.lower()
        # Remove punctuation except hyphens
        text = re.sub(r"[^\w\s\-]", " ", text)
        tokens = text.split()
        return [
            tok for tok in tokens
            if tok not in _STOP_WORDS and len(tok) > 1
        ]
