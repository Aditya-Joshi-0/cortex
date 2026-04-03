"""
Cortex RAG — Milvus Dense Store

Handles:
  - Collection creation with schema (child text + parent text + metadata)
  - IVF_FLAT index (swap to HNSW for recall-critical workloads)
  - Batched upsert with automatic embedding
  - Similarity search returning RetrievedChunk objects
  - Deduplication check by doc_id
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from config import get_settings
from ingestion.chunker import Chunk
from retrieval.embedder import Embedder

logger = logging.getLogger(__name__)

# Milvus field name constants
_FLD_ID         = "chunk_id"
_FLD_DOC_ID     = "doc_id"
_FLD_SOURCE     = "source"
_FLD_TITLE      = "title"
_FLD_TEXT       = "text"
_FLD_PARENT     = "parent_text"
_FLD_CHUNK_IDX  = "chunk_index"
_FLD_VECTOR     = "vector"


@dataclass
class RetrievedChunk:
    """A chunk returned by a similarity search, with its relevance score."""
    chunk_id: str
    doc_id: str
    source: str
    title: str
    text: str               # child chunk (used for display / citation)
    parent_text: str        # parent chunk (passed to LLM as context)
    chunk_index: int
    score: float            # cosine similarity (0–1)
    retriever: str = "dense"
    metadata: dict = field(default_factory=dict)


class MilvusStore:
    """
    Vector store backed by Milvus.

    Collection schema
    ─────────────────
    chunk_id   VARCHAR(32)   PK
    doc_id     VARCHAR(32)
    source     VARCHAR(512)
    title      VARCHAR(512)
    text       VARCHAR(4096)   ← child chunk
    parent_text VARCHAR(8192)  ← parent chunk
    chunk_index INT64
    vector     FLOAT_VECTOR(384)
    """

    # Milvus field length limits
    _MAX_CHUNK_TEXT  = 4_096
    _MAX_PARENT_TEXT = 8_192
    _MAX_VARCHAR     = 512

    def __init__(self, embedder: Optional[Embedder] = None) -> None:
        self._embedder = embedder or Embedder()
        self._collection = None
        self._connected = False

    # ── Connection ─────────────────────────────────────────────

    def connect(self) -> None:
        if self._connected:
            return
        cfg = get_settings()
        try:
            from pymilvus import connections  # type: ignore
            connections.connect(
                alias="default",
                host=cfg.milvus_host,
                port=str(cfg.milvus_port),
            )
            self._connected = True
            logger.info("Connected to Milvus at %s:%d", cfg.milvus_host, cfg.milvus_port)
        except Exception as exc:
            raise RuntimeError(
                f"Cannot connect to Milvus at {cfg.milvus_host}:{cfg.milvus_port}. "
                "Is the Docker container running? (docker-compose up -d)"
            ) from exc

    def _ensure_collection(self):
        if self._collection is not None:
            return self._collection
        self.connect()
        self._collection = self._get_or_create_collection()
        return self._collection

    # ── Schema & collection ───────────────────────────────────

    def _get_or_create_collection(self):
        from pymilvus import (  # type: ignore
            Collection, CollectionSchema, DataType, FieldSchema, utility
        )
        cfg = get_settings()
        name = cfg.milvus_collection

        if utility.has_collection(name):
            coll = Collection(name)
            coll.load()
            logger.info("Loaded existing collection '%s'", name)
            return coll

        logger.info("Creating collection '%s'", name)
        fields = [
            FieldSchema(_FLD_ID,         DataType.VARCHAR, max_length=32,  is_primary=True),
            FieldSchema(_FLD_DOC_ID,     DataType.VARCHAR, max_length=32),
            FieldSchema(_FLD_SOURCE,     DataType.VARCHAR, max_length=self._MAX_VARCHAR),
            FieldSchema(_FLD_TITLE,      DataType.VARCHAR, max_length=self._MAX_VARCHAR),
            FieldSchema(_FLD_TEXT,       DataType.VARCHAR, max_length=self._MAX_CHUNK_TEXT),
            FieldSchema(_FLD_PARENT,     DataType.VARCHAR, max_length=self._MAX_PARENT_TEXT),
            FieldSchema(_FLD_CHUNK_IDX,  DataType.INT64),
            FieldSchema(_FLD_VECTOR,     DataType.FLOAT_VECTOR, dim=cfg.embed_dim),
        ]
        schema = CollectionSchema(fields, description="Cortex RAG chunks")
        coll = Collection(name, schema)

        # IVF_FLAT index — good balance of speed vs. recall for <1M vectors
        coll.create_index(
            field_name=_FLD_VECTOR,
            index_params={
                "index_type": cfg.milvus_index_type,
                "metric_type": cfg.milvus_metric_type,
                "params": {"nlist": cfg.milvus_nlist},
            },
        )
        coll.load()
        logger.info("Collection '%s' created and loaded.", name)
        return coll

    # ── Upsert ─────────────────────────────────────────────────

    def upsert_chunks(
        self,
        chunks: list[Chunk],
        batch_size: int = 256,
    ) -> int:
        """Embed and insert chunks. Returns count stored."""
        coll = self._ensure_collection()
        stored = 0

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c.text for c in batch]
            embeddings = self._embedder.encode(texts, show_progress_bar=False)

            data = [
                [c.chunk_id                                             for c in batch],  # chunk_id
                [c.doc_id                                               for c in batch],  # doc_id
                [self._trunc(c.source,      self._MAX_VARCHAR)         for c in batch],  # source
                [self._trunc(c.title,       self._MAX_VARCHAR)         for c in batch],  # title
                [self._trunc(c.text,        self._MAX_CHUNK_TEXT)      for c in batch],  # text
                [self._trunc(c.parent_text, self._MAX_PARENT_TEXT)     for c in batch],  # parent_text
                [c.chunk_index                                          for c in batch],  # chunk_index
                embeddings.tolist(),                                                       # vector
            ]
            coll.insert(data)
            stored += len(batch)
            logger.debug("Inserted batch %d–%d", i, i + len(batch))

        coll.flush()
        logger.info("Upserted %d chunks.", stored)
        return stored

    # ── Search ─────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> list[RetrievedChunk]:
        """
        Semantic search. Returns RetrievedChunk list sorted by cosine score desc.
        """
        cfg = get_settings()
        k = top_k or cfg.retrieval_top_k
        coll = self._ensure_collection()

        query_vector = self._embedder.encode_query(query)

        results = coll.search(
            data=query_vector.tolist(),
            anns_field=_FLD_VECTOR,
            param={"metric_type": cfg.milvus_metric_type, "params": {"nprobe": cfg.milvus_nprobe}},
            limit=k,
            output_fields=[
                _FLD_ID, _FLD_DOC_ID, _FLD_SOURCE, _FLD_TITLE,
                _FLD_TEXT, _FLD_PARENT, _FLD_CHUNK_IDX,
            ],
        )

        chunks: list[RetrievedChunk] = []
        for hit in results[0]:
            e = hit.entity
            chunks.append(
                RetrievedChunk(
                    chunk_id=e.get(_FLD_ID, ""),
                    doc_id=e.get(_FLD_DOC_ID, ""),
                    source=e.get(_FLD_SOURCE, ""),
                    title=e.get(_FLD_TITLE, ""),
                    text=e.get(_FLD_TEXT, ""),
                    parent_text=e.get(_FLD_PARENT, ""),
                    chunk_index=e.get(_FLD_CHUNK_IDX, 0),
                    score=float(hit.score),
                )
            )
        return chunks

    # ── Deduplication ─────────────────────────────────────────

    def get_existing_doc_ids(self) -> set[str]:
        """Return all doc_ids currently stored in the collection."""
        try:
            coll = self._ensure_collection()
            results = coll.query(
                expr="chunk_index == 0",    # one row per doc
                output_fields=[_FLD_DOC_ID],
                limit=16_384,
            )
            return {r[_FLD_DOC_ID] for r in results}
        except Exception:
            return set()

    def collection_stats(self) -> dict:
        """Return basic stats for the collection."""
        try:
            coll = self._ensure_collection()
            return {
                "collection": get_settings().milvus_collection,
                "entity_count": coll.num_entities,
            }
        except Exception as exc:
            return {"error": str(exc)}

    # ── Utilities ─────────────────────────────────────────────

    @staticmethod
    def _trunc(text: str, max_len: int) -> str:
        """Truncate string to Milvus VARCHAR limit."""
        return text[:max_len] if len(text) > max_len else text
