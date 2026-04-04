"""
Cortex RAG — Multi-Strategy Retrieval Orchestrator

This is the central retrieval entry point for Phase 2+.
It replaces the direct `MilvusStore.search()` call in api/main.py.

Query flow
──────────
  1. QueryRouter   — classify intent → strategy set (e.g. ["dense", "bm25"])
  2. Run enabled retrievers in parallel (ThreadPoolExecutor)
  3. RRFFusion     — merge ranked lists into a single fused ranking
  4. CrossEncoder  — rerank top-N RRF candidates → top-K final chunks
  5. Return to generator

Adding a new retriever in Phase 3 (GraphRAG)
  - Implement a class with .search(query, top_k) → list[RetrievedChunk]
  - Register it in _RETRIEVERS dict below
  - Map intent type "multi_hop" → ["dense", "graph", "bm25"] in router.py

No changes to generator.py or api/main.py endpoints required.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from config import get_settings
from retrieval.bm25 import BM25Retriever
from retrieval.dense import MilvusStore, RetrievedChunk
from retrieval.embedder import Embedder
from retrieval.fusion import CrossEncoderReranker, RRFFusion
from retrieval.router import QueryRouter, RoutingDecision

logger = logging.getLogger(__name__)


class MultiStrategyRetriever:
    """
    Orchestrates all retrieval strategies for a single query.

    Typical usage (in api/main.py):
        retriever = MultiStrategyRetriever(...)
        result = retriever.retrieve(query)
        # result.chunks → list[RetrievedChunk] ready for the generator
        # result.decision → RoutingDecision (for logging / debug UI)
    """

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        store: Optional[MilvusStore] = None,
        bm25: Optional[BM25Retriever] = None,
        router: Optional[QueryRouter] = None,
        fuser: Optional[RRFFusion] = None,
        reranker: Optional[CrossEncoderReranker] = None,
    ) -> None:
        self._embedder = embedder or Embedder()
        self._dense    = store    or MilvusStore(embedder=self._embedder)
        self._bm25     = bm25     or BM25Retriever()
        self._router   = router   or QueryRouter()
        self._fuser    = fuser    or RRFFusion()
        self._reranker = reranker or CrossEncoderReranker()

    # ── Public API ─────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k_candidates: Optional[int] = None,
        final_top_k: Optional[int] = None,
    ) -> "RetrievalResult":
        """
        Full pipeline: route → retrieve → fuse → rerank.

        Returns a RetrievalResult with the final chunks and routing metadata.
        """
        cfg = get_settings()
        candidates_k = top_k_candidates or cfg.retrieval_top_k
        final_k      = final_top_k      or cfg.final_top_k

        # 1. Classify intent → strategy set
        decision = self._router.route(query)

        # 2. Run retrievers in parallel
        retriever_results = self._run_retrievers(
            query, decision.strategies, top_k=candidates_k
        )

        if not retriever_results:
            return RetrievalResult(chunks=[], decision=decision, retriever_hits={})

        # 3. RRF fusion
        fused = self._fuser.fuse(retriever_results, top_k=candidates_k)

        # 4. Cross-encoder rerank
        final = self._reranker.rerank(query, fused, top_k=final_k)

        hits = {name: len(results) for name, results in retriever_results.items()}

        logger.info(
            "Retrieval: intent=%s strategies=%s hits=%s → %d final chunks",
            decision.intent.value, decision.strategies, hits, len(final)
        )
        return RetrievalResult(chunks=final, decision=decision, retriever_hits=hits)

    # ── BM25 corpus management ────────────────────────────────

    def index_chunks(self, chunks: list) -> int:
        """
        Add chunks to the BM25 index (call from ingestion pipeline).
        Dense indexing is handled separately by MilvusStore.
        """
        return self._bm25.add_chunks(chunks)

    # ── Private: parallel retrieval ───────────────────────────

    def _run_retrievers(
        self,
        query: str,
        strategies: list[str],
        top_k: int,
    ) -> dict[str, list[RetrievedChunk]]:
        """
        Dispatch retriever calls concurrently.
        Unknown strategy names are logged and skipped.
        """
        retriever_map = {
            "dense": lambda q, k: self._dense.search(q, top_k=k),
            "bm25":  lambda q, k: self._bm25.search(q, top_k=k),
            # "graph" will be registered here in Phase 3
        }

        results: dict[str, list[RetrievedChunk]] = {}
        active = [(name, retriever_map[name]) for name in strategies if name in retriever_map]

        if not active:
            logger.warning("No enabled retrievers for strategies: %s", strategies)
            return results

        # Use threads — both Milvus and BM25 are IO/CPU bound but release the GIL
        with ThreadPoolExecutor(max_workers=len(active)) as pool:
            futures = {
                pool.submit(fn, query, top_k): name
                for name, fn in active
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    hits = future.result()
                    results[name] = hits
                    logger.debug("%s retriever: %d hits", name, len(hits))
                except Exception as exc:
                    logger.error("%s retriever failed: %s", name, exc)

        return results


# ── Result dataclass ───────────────────────────────────────────

from dataclasses import dataclass, field


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]
    decision: RoutingDecision
    retriever_hits: dict[str, int] = field(default_factory=dict)

    @property
    def empty(self) -> bool:
        return len(self.chunks) == 0
