"""
Cortex RAG — Retrieval Cache (Redis, Phase 4)

What gets cached
─────────────────
The output of the full retrieval pipeline — after RRF fusion and
cross-encoder reranking — is serialised and stored in Redis with a
configurable TTL (default 1 hour).

Cache key: SHA-256 of (query.lower().strip() + str(top_k))
This means the same query with different capitalisation or trailing
spaces hits the same cache entry, which is almost always correct for RAG.

What does NOT get cached
─────────────────────────
CRAG evaluation and generation are NOT cached. The CRAG grade depends
on the current state of the knowledge base (which changes after ingestion),
and generation is fast enough (streaming) that caching it adds complexity
without meaningful latency savings.

Graceful degradation
─────────────────────
If Redis is unreachable on startup, the cache silently disables itself
and logs a warning. Every query falls through to the live retrieval
pipeline unchanged. No exceptions surface to the user.

This means you can develop without Redis running locally and only enable
it in production (Railway, Render) where Redis add-ons are available.
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Optional

from retrieval.dense import RetrievedChunk
from retrieval.orchestrator import MultiStrategyRetriever, RetrievalResult
from retrieval.router import QueryIntent, RoutingDecision

logger = logging.getLogger(__name__)


def _make_cache_key(query: str, top_k: int) -> str:
    raw = f"{query.lower().strip()}:{top_k}"
    return "cortex:retrieval:" + hashlib.sha256(raw.encode()).hexdigest()[:24]


def _serialise_result(result: RetrievalResult) -> str:
    """JSON-serialise a RetrievalResult for Redis storage."""
    return json.dumps({
        "chunks": [
            {
                "chunk_id":    c.chunk_id,
                "doc_id":      c.doc_id,
                "source":      c.source,
                "title":       c.title,
                "text":        c.text,
                "parent_text": c.parent_text,
                "chunk_index": c.chunk_index,
                "score":       c.score,
                "retriever":   c.retriever,
            }
            for c in result.chunks
        ],
        "decision": {
            "intent":        result.decision.intent.value,
            "strategies":    result.decision.strategies,
            "confidence":    result.decision.confidence,
            "reasoning":     result.decision.reasoning,
        },
        "retriever_hits": result.retriever_hits,
    })


def _deserialise_result(raw: str) -> RetrievalResult:
    """Reconstruct a RetrievalResult from its JSON representation."""
    data = json.loads(raw)

    chunks = [
        RetrievedChunk(
            chunk_id=c["chunk_id"],
            doc_id=c["doc_id"],
            source=c["source"],
            title=c["title"],
            text=c["text"],
            parent_text=c["parent_text"],
            chunk_index=c["chunk_index"],
            score=c["score"],
            retriever=c["retriever"],
        )
        for c in data["chunks"]
    ]

    d = data["decision"]
    decision = RoutingDecision(
        intent=QueryIntent(d["intent"]),
        strategies=d["strategies"],
        confidence=d["confidence"],
        reasoning=d["reasoning"],
    )

    return RetrievalResult(
        chunks=chunks,
        decision=decision,
        retriever_hits=data.get("retriever_hits", {}),
    )


class CachedRetriever:
    """
    Drop-in wrapper around MultiStrategyRetriever that adds Redis caching.

    Usage (replaces MultiStrategyRetriever in api/main.py):
        retriever = CachedRetriever(MultiStrategyRetriever(...))
        result = retriever.retrieve(query)
        print(retriever.cache_stats())  # {"hits": 3, "misses": 7, "enabled": True}
    """

    def __init__(
        self,
        inner: MultiStrategyRetriever,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        self._inner = inner
        self._redis = self._connect_redis()
        self._ttl   = ttl_seconds or self._default_ttl()
        self._hits   = 0
        self._misses = 0

    # ── Public API (matches MultiStrategyRetriever interface) ──

    def retrieve(
        self,
        query: str,
        top_k_candidates: Optional[int] = None,
        final_top_k: Optional[int] = None,
    ) -> RetrievalResult:
        """
        Retrieve with cache. Falls through to live retrieval on miss or error.
        """
        from config import get_settings
        cfg = get_settings()
        k = final_top_k or cfg.final_top_k
        key = _make_cache_key(query, k)

        # ── Cache lookup ───────────────────────────────────────
        if self._redis:
            try:
                cached = self._redis.get(key)
                if cached:
                    self._hits += 1
                    logger.debug("Cache HIT for query: %s…", query[:40])
                    result = _deserialise_result(cached)
                    result.from_cache = True
                    return result
            except Exception as exc:
                logger.warning("Redis GET failed: %s — falling through.", exc)

        # ── Cache miss: live retrieval ─────────────────────────
        self._misses += 1
        logger.debug("Cache MISS for query: %s…", query[:40])
        result = self._inner.retrieve(query, top_k_candidates, final_top_k)
        result.from_cache = False

        # ── Write to cache ─────────────────────────────────────
        if self._redis and not result.empty:
            try:
                self._redis.setex(key, self._ttl, _serialise_result(result))
            except Exception as exc:
                logger.warning("Redis SET failed: %s", exc)

        return result

    def invalidate(self, query: str, top_k: int) -> bool:
        """Manually invalidate a cache entry (e.g. after re-ingestion)."""
        if not self._redis:
            return False
        try:
            return bool(self._redis.delete(_make_cache_key(query, top_k)))
        except Exception:
            return False

    def flush_all(self) -> int:
        """Delete all Cortex cache keys. Returns count deleted."""
        if not self._redis:
            return 0
        try:
            keys = self._redis.keys("cortex:retrieval:*")
            if keys:
                return self._redis.delete(*keys)
            return 0
        except Exception:
            return 0

    def cache_stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "enabled":   self._redis is not None,
            "hits":      self._hits,
            "misses":    self._misses,
            "hit_rate":  round(self._hits / total, 3) if total else 0.0,
            "ttl_s":     self._ttl,
        }

    # ── Pass-through for orchestrator methods ──────────────────

    def index_chunks(self, chunks: list) -> int:
        return self._inner.index_chunks(chunks)

    def build_graph(self, chunks: list) -> dict:
        return self._inner.build_graph(chunks)

    @property
    def graph_builder(self):
        return self._inner.graph_builder

    # ── Redis connection ───────────────────────────────────────

    @staticmethod
    def _connect_redis():
        from config import get_settings
        cfg = get_settings()
        url = getattr(cfg, "redis_url", "redis://localhost:6379")
        try:
            import redis  # type: ignore
            client = redis.from_url(url, socket_connect_timeout=2, decode_responses=True)
            client.ping()
            logger.info("Redis cache connected at %s", url)
            return client
        except ImportError:
            logger.info("redis-py not installed — cache disabled. pip install redis")
            return None
        except Exception as exc:
            logger.warning("Redis unavailable (%s) — cache disabled.", exc)
            return None

    @staticmethod
    def _default_ttl() -> int:
        from config import get_settings
        return getattr(get_settings(), "cache_ttl_seconds", 3600)
