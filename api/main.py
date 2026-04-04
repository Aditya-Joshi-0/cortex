"""
Cortex RAG — FastAPI Application

Endpoints
─────────
GET  /health          → system health check
POST /ingest          → trigger ingestion pipeline
POST /query           → blocking query (JSON response)
POST /query/stream    → streaming query (Server-Sent Events)

Phase 1 uses dense-only retrieval.
Later phases will add routing, graph, BM25, and CRAG via the same endpoint.
"""
from __future__ import annotations

import json
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from api.schemas import (
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    ChunkResponse,
    CitationResponse,
)
from config import get_settings
from generation.generator import Generator, GenerationRequest
from ingestion.pipeline import IngestionPipeline
from retrieval.dense import MilvusStore
from retrieval.embedder import Embedder
from retrieval.bm25 import BM25Retriever
from retrieval.orchestrator import MultiStrategyRetriever

logger = logging.getLogger(__name__)

# ── Shared singletons ──────────────────────────────────────────
# Created once on startup, shared across requests

_embedder: Embedder = None
_store: MilvusStore = None
_bm25: BM25Retriever = None
_retriever: MultiStrategyRetriever = None
_generator: Generator = None
_pipeline: IngestionPipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise shared resources on startup, clean up on shutdown."""
    global _embedder, _store, _bm25, _retriever, _generator, _pipeline
    logger.info("Cortex starting up...")

    _embedder  = Embedder()
    _store     = MilvusStore(embedder=_embedder)
    _bm25      = BM25Retriever()
    _retriever = MultiStrategyRetriever(embedder=_embedder, store=_store, bm25=_bm25)
    _generator = Generator()
    _pipeline  = IngestionPipeline(embedder=_embedder, store=_store, bm25=_bm25)

    # Warm up: trigger model load immediately so first request is fast
    _ = _embedder.model

    logger.info("Cortex ready.")
    yield
    logger.info("Cortex shutting down.")


# ── App factory ────────────────────────────────────────────────

def create_app() -> FastAPI:
    cfg = get_settings()

    app = FastAPI(
        title="Cortex RAG API",
        description=(
            "Production-grade Retrieval-Augmented Generation system "
            "with multi-strategy retrieval, CRAG, and RAGAS evaluation."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # tighten in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app


app = create_app()


# ── Routes ─────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health() -> HealthResponse:
    """
    Returns the health of all system components.
    Use this to verify Milvus is reachable and the model is loaded.
    """
    milvus_status = "ok"
    collection_stats = {}

    try:
        collection_stats = _store.collection_stats()
    except Exception as exc:
        milvus_status = f"error: {exc}"

    embedder_status = "loaded" if _embedder and _embedder._model else "not_loaded"

    return HealthResponse(
        status="ok" if milvus_status == "ok" else "degraded",
        milvus=milvus_status,
        embedder=embedder_status,
        collection_stats=collection_stats,
    )


@app.post("/ingest", response_model=IngestResponse, tags=["ingestion"])
async def ingest(req: IngestRequest) -> IngestResponse:
    """
    Trigger the ingestion pipeline for a file or directory.

    - Deduplicates by doc_id (SHA-256 of file path)
    - Returns counts for documents processed, chunks created, and errors
    """
    import os
    path = req.path

    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Path not found: {path}")

    try:
        if os.path.isfile(path):
            stats = _pipeline.ingest_file(path)
        else:
            stats = _pipeline.ingest_directory(path, recursive=req.recursive)
    except Exception as exc:
        logger.exception("Ingestion error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return IngestResponse(**stats)


@app.post("/query", response_model=QueryResponse, tags=["retrieval"])
async def query(req: QueryRequest) -> QueryResponse:
    """
    Blocking query endpoint.
    Retrieves top-k chunks and returns a complete cited answer.
    """
    cfg = get_settings()
    k = req.top_k or cfg.retrieval_top_k

    try:
        retrieval = _retriever.retrieve(req.query, top_k_candidates=k, final_top_k=cfg.final_top_k)
    except Exception as exc:
        logger.exception("Retrieval error")
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {exc}")

    if retrieval.empty:
        return QueryResponse(
            query=req.query,
            answer="No relevant documents found in the knowledge base.",
            citations=[],
            retrieved_chunks=[],
            model="",
            usage={},
        )

    final_chunks = retrieval.chunks

    try:
        result = _generator.generate(
            GenerationRequest(query=req.query, chunks=final_chunks)
        )
    except Exception as exc:
        logger.exception("Generation error")
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}")

    return QueryResponse(
        query=req.query,
        answer=result.answer,
        citations=[
            CitationResponse(
                number=c.number,
                title=c.title,
                source=c.source,
                chunk_id=c.chunk_id,
                score=c.score,
            )
            for c in result.citations
        ],
        retrieved_chunks=[
            ChunkResponse(
                chunk_id=ch.chunk_id,
                doc_id=ch.doc_id,
                source=ch.source,
                title=ch.title,
                text=ch.text,
                score=ch.score,
            )
            for ch in final_chunks
        ],
        model=result.model,
        usage=result.usage,
    )


@app.post("/query/stream", tags=["retrieval"])
async def query_stream(req: QueryRequest):
    """
    Streaming query endpoint using Server-Sent Events (SSE).

    Event types emitted:
      - data: {"type": "chunk_meta", "chunks": [...]}   — retrieved chunks
      - data: {"type": "token", "text": "..."}           — answer tokens
      - data: {"type": "sources", "text": "..."}         — sources block
      - data: {"type": "done"}                           — stream complete
      - data: {"type": "error", "message": "..."}        — error event
    """
    cfg = get_settings()
    k = req.top_k or cfg.retrieval_top_k

    async def event_stream() -> AsyncGenerator[str, None]:
        try:
            # 1. Retrieve
            # 1. Multi-strategy retrieval: router → dense+BM25 → RRF → cross-encoder
            result = _retriever.retrieve(req.query, top_k_candidates=k, final_top_k=cfg.final_top_k)
            final_chunks = result.chunks

            # 2. Emit chunk metadata + routing decision so UI shows sources + strategy info immediately
            chunk_meta = [
                {
                    "chunk_id": c.chunk_id,
                    "title": c.title,
                    "source": c.source,
                    "score": round(c.score, 4),
                    "retriever": c.retriever,
                    "text_snippet": c.text[:200],
                }
                for c in final_chunks
            ]
            yield _sse_event({
                "type": "chunk_meta",
                "chunks": chunk_meta,
                "routing": {
                    "intent": result.decision.intent.value,
                    "strategies": result.decision.strategies,
                    "retriever_hits": result.retriever_hits,
                    "reasoning": result.decision.reasoning,
                },
            })

            if not final_chunks:
                yield _sse_event({
                    "type": "token",
                    "text": "No relevant documents found in the knowledge base.",
                })
                yield _sse_event({"type": "done"})
                return

            # 3. Stream answer tokens
            gen_request = GenerationRequest(
                query=req.query,
                chunks=final_chunks,
                stream=True,
            )
            for token in _generator.stream(gen_request):
                yield _sse_event({"type": "token", "text": token})

            # 4. Emit sources block
            sources = _generator.build_sources_block(final_chunks)
            yield _sse_event({"type": "sources", "text": sources})

            # 5. Signal completion
            yield _sse_event({"type": "done"})

        except Exception as exc:
            logger.exception("Streaming error")
            yield _sse_event({"type": "error", "message": str(exc)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering
        },
    )


# ── SSE helper ─────────────────────────────────────────────────

def _sse_event(data: dict) -> str:
    """Format a dict as a Server-Sent Event string."""
    return f"data: {json.dumps(data)}\n\n"


# ── Dev server entry point ─────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    cfg = get_settings()
    logging.basicConfig(
        level=getattr(logging, cfg.log_level),
        format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    )
    uvicorn.run(
        "api.main:app",
        host=cfg.api_host,
        port=cfg.api_port,
        reload=cfg.api_reload,
    )
