"""
Cortex RAG — Ingestion Pipeline

Orchestrates: DocumentLoader → SemanticChunker → Embedder → MilvusStore + BM25.
Phase 2: chunks are now indexed in BOTH Milvus (dense) and BM25 (sparse)
during ingestion. Both indexes are deduplicated independently.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import logging
import time
from pathlib import Path
from typing import Optional

from config import get_settings
from ingestion.chunker import Chunk, SemanticChunker
from ingestion.document_loader import Document, DocumentLoader
from retrieval.embedder import Embedder
from retrieval.dense import MilvusStore
from retrieval.bm25 import BM25Retriever
from retrieval.graph_builder import KnowledgeGraphBuilder

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    End-to-end ingestion pipeline.

    Usage:
        pipeline = IngestionPipeline()
        stats = pipeline.ingest_directory("data/documents")
        print(stats)
    """

    def __init__(
        self,
        loader: Optional[DocumentLoader] = None,
        chunker: Optional[SemanticChunker] = None,
        embedder: Optional[Embedder] = None,
        store: Optional[MilvusStore] = None,
        bm25: Optional[BM25Retriever] = None,
        graph: Optional[KnowledgeGraphBuilder] = None,

    ) -> None:
        self._loader = loader or DocumentLoader()
        self._embedder = embedder or Embedder()
        self._chunker = chunker or SemanticChunker(embedder=self._embedder)
        self._store = store or MilvusStore(embedder=self._embedder)
        self._bm25  = bm25  or BM25Retriever()
        self._graph = graph or KnowledgeGraphBuilder()

    # ── Public ─────────────────────────────────────────────────

    def ingest_file(self, path: str | Path) -> dict:
        """Ingest a single file. Returns per-file stats dict."""
        return self._run([self._loader.load_file(path)])

    def ingest_directory(
        self,
        directory: str | Path,
        recursive: bool = True,
    ) -> dict:
        """Ingest all supported files in a directory."""
        docs = self._loader.load_directory(directory, recursive=recursive)
        return self._run(docs)

    def ingest_documents(self, docs: list[Document]) -> dict:
        """Ingest an already-loaded list of Document objects."""
        return self._run(docs)

    # ── Core pipeline ──────────────────────────────────────────

    def _run(self, docs: list[Document]) -> dict:
        t0 = time.perf_counter()
        stats = {
            "documents_processed": 0,
            "documents_skipped": 0,
            "chunks_created": 0,
            "chunks_stored": 0,
            "errors": [],
        }

        if not docs:
            logger.warning("No documents to ingest.")
            return stats

        # ── Deduplication ──────────────────────────────────────
        existing_ids = self._store.get_existing_doc_ids()
        new_docs = [d for d in docs if d.doc_id not in existing_ids]
        stats["documents_skipped"] = len(docs) - len(new_docs)

        if not new_docs:
            logger.info("All %d documents already ingested.", len(docs))
            return stats

        logger.info(
            "Ingesting %d new documents (%d already exist).",
            len(new_docs),
            stats["documents_skipped"],
        )

        # ── Chunk ──────────────────────────────────────────────
        all_chunks: list[Chunk] = []
        for doc in new_docs:
            try:
                chunks = self._chunker.chunk_document(doc)
                all_chunks.extend(chunks)
                stats["documents_processed"] += 1
                logger.debug("  %s → %d chunks", doc.title, len(chunks))
            except Exception as exc:
                logger.error("Chunking failed for %s: %s", doc.source, exc)
                stats["errors"].append({"source": doc.source, "error": str(exc)})

        stats["chunks_created"] = len(all_chunks)

        if not all_chunks:
            logger.warning("No chunks produced.")
            return stats

        # ── Embed + store in Milvus (dense) ───────────────────
        try:
            stored = self._store.upsert_chunks(all_chunks)
            stats["chunks_stored"] = stored
        except Exception as exc:
            logger.error("Milvus storage failed: %s", exc)
            stats["errors"].append({"source": "milvus_upsert", "error": str(exc)})

        # ── Index in BM25 (sparse) — Phase 2 ──────────────────
        try:
            bm25_added = self._bm25.add_chunks(all_chunks)
            stats["bm25_indexed"] = bm25_added
        except Exception as exc:
            logger.error("BM25 indexing failed: %s", exc)
            stats["errors"].append({"source": "bm25_index", "error": str(exc)})
        
        # ── Build knowledge graph (NER + relations) — Phase 3 ──
        try:
            graph_stats = self._graph.process_chunks(all_chunks)
            stats["graph_entities"] = graph_stats.get("entities", 0)
            stats["graph_triples"]  = graph_stats.get("triples", 0)
        except Exception as exc:
            logger.error("Graph extraction failed: %s", exc)
            stats["errors"].append({"source": "graph_build", "error": str(exc)})
        
        elapsed = time.perf_counter() - t0
        logger.info(
            "Ingestion complete in %.1fs — %d docs, %d chunks stored.",
            elapsed,
            stats["documents_processed"],
            stats["chunks_stored"],
        )
        return stats


# ── CLI entry point ────────────────────────────────────────────

def main() -> None:
    import argparse, sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    )
    parser = argparse.ArgumentParser(description="Cortex ingestion pipeline")
    parser.add_argument("path", help="File or directory to ingest")
    parser.add_argument(
        "--no-recursive", action="store_true",
        help="Only ingest top-level files (no subdirectories)"
    )
    args = parser.parse_args()

    pipeline = IngestionPipeline()
    p = Path(args.path)

    if p.is_file():
        stats = pipeline.ingest_file(p)
    elif p.is_dir():
        stats = pipeline.ingest_directory(p, recursive=not args.no_recursive)
    else:
        print(f"Error: {p} is not a valid file or directory.", file=sys.stderr)
        sys.exit(1)

    print("\n── Ingestion summary ──────────────────────────")
    for k, v in stats.items():
        print(f"  {k:30s}: {v}")


if __name__ == "__main__":
    main()
