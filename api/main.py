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
from generation.crag import CRAGGate
from evaluation.store import EvalStore, QueryLogEntry
from evaluation.ragas_eval import RAGASEvaluator, EvalInput
from retrieval.cache import CachedRetriever
from ingestion.pipeline import IngestionPipeline
from retrieval.dense import MilvusStore
from retrieval.embedder import Embedder
from retrieval.bm25 import BM25Retriever
from retrieval.orchestrator import MultiStrategyRetriever

logger = logging.getLogger(__name__)
cfg = get_settings()

# ── Shared singletons ──────────────────────────────────────────
# Created once on startup, shared across requests

_embedder: Embedder = None
_store: MilvusStore = None
_bm25: BM25Retriever = None
_retriever: MultiStrategyRetriever = None
_crag: CRAGGate = None
_eval_store: EvalStore = None
_evaluator: RAGASEvaluator = None
_generator: Generator = None
_pipeline: IngestionPipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise shared resources on startup, clean up on shutdown."""
    global _embedder, _store, _bm25, _retriever, _crag, _generator, _pipeline, _eval_store, _evaluator
    logger.info("Cortex starting up...")

    _embedder  = Embedder()
    _store     = MilvusStore(embedder=_embedder)
    _bm25      = BM25Retriever()
    _retriever = MultiStrategyRetriever(embedder=_embedder, store=_store, bm25=_bm25)
    _crag       = CRAGGate()
    _eval_store = EvalStore(db_path=cfg.eval_db_path)
    _evaluator  = RAGASEvaluator(store=_eval_store)
    _generator  = Generator()
    # Wrap retriever with Redis cache (degrades gracefully if Redis is absent)
    _retriever  = CachedRetriever(_retriever)
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

    graph_stats = {}
    try:
        graph_stats = _retriever.graph_builder.stats()
    except Exception:
        pass

    return HealthResponse(
        status="ok" if milvus_status == "ok" else "degraded",
        milvus=milvus_status,
        embedder=embedder_status,
        collection_stats=collection_stats,
        graph_stats=graph_stats,
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

@app.get("/metrics", tags=["evaluation"])
async def get_metrics(limit: int = 100, days: int = 7):
    """
    Query performance metrics and RAGAS scores for the dashboard.
    Returns summary stats, recent query logs, and hourly timeseries.
    """
    return {
        "summary":    _eval_store.get_summary_stats(),
        "recent":     _eval_store.get_recent_queries(limit=limit),
        "timeseries": _eval_store.get_metric_timeseries(days=days),
        "cache":      _retriever.cache_stats(),
    }


@app.post("/cache/flush", tags=["system"])
async def flush_cache():
    """Flush all Redis retrieval cache entries."""
    deleted = _retriever.flush_all()
    return {"deleted": deleted}

@app.post("/query", response_model=QueryResponse, tags=["retrieval"])
async def query(req: QueryRequest) -> QueryResponse:
    """
    Blocking query endpoint.
    Retrieves top-k chunks and returns a complete cited answer.
    """
    cfg = get_settings()
    k = req.top_k or cfg.retrieval_top_k

    import time as _time
    _t0 = _time.perf_counter()

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

    # CRAG gate: grade, rewrite if POOR, web-search fallback if ABSENT
    crag_result = _crag.evaluate(
        query=req.query,
        chunks=final_chunks,
        retriever_fn=lambda q: _retriever.retrieve(q).chunks,
    )
    final_chunks = crag_result.final_chunks

    try:
        result = _generator.generate(
            GenerationRequest(query=req.query, chunks=final_chunks)
        )
    except Exception as exc:
        logger.exception("Generation error")
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}")

    latency_ms = (_time.perf_counter() - _t0) * 1000

    log_id = _eval_store.log_query(QueryLogEntry(
        query=req.query,
        intent=retrieval.decision.intent.value,
        strategies=retrieval.decision.strategies,
        retriever_hits=retrieval.retriever_hits,
        crag_grade=crag_result.grade.value,
        crag_rewritten=bool(crag_result.rewritten_query),
        web_search_used=crag_result.web_search_used,
        num_chunks=len(final_chunks),
        top_chunk_score=final_chunks[0].score if final_chunks else 0.0,
        latency_ms=latency_ms,
        model=result.model,
    ))

    if cfg.eval_enabled:
        _evaluator.evaluate_async(EvalInput(
            query_log_id=log_id,
            query=req.query,
            answer=result.answer,
            chunks=final_chunks,
        ))

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

            # 3. CRAG gate — grade, optionally rewrite + re-retrieve
            crag_result = _crag.evaluate(
                query=req.query,
                chunks=final_chunks,
                retriever_fn=lambda q: _retriever.retrieve(q).chunks,
            )
            final_chunks = crag_result.final_chunks

            # Emit CRAG event if something interesting happened
            if crag_result.grade.value != "GOOD" or crag_result.web_search_used:
                yield _sse_event({
                    "type": "crag_update",
                    "grade": crag_result.grade.value,
                    "rewritten_query": crag_result.rewritten_query,
                    "web_search_used": crag_result.web_search_used,
                    "reasoning": crag_result.reasoning,
                })

            # 4. Stream answer tokens
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
