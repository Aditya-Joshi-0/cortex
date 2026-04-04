"""
Cortex RAG — RRF Fusion + Cross-Encoder Reranker

Reciprocal Rank Fusion (RRF)
────────────────────────────
Each retriever returns a ranked list of chunks. RRF merges these lists
by assigning every chunk a fused score:

    score(chunk) = Σ  1 / (k + rank_i(chunk))
                  retrievers

where k=60 is the standard smoothing constant (Cormack et al., 2009).
The same chunk appearing at rank 3 in dense AND rank 5 in BM25 gets a
much higher fused score than a chunk that only appeared in one list.

Why not just concatenate and deduplicate?
  Concatenation ignores rank information — the 1st-place dense result
  and the 15th-place dense result would be treated identically once
  deduplicated. RRF preserves relative rank from each signal.

Cross-Encoder Reranker
───────────────────────
RRF merges retriever rankings but still uses similarity scores as a
proxy for relevance. A cross-encoder (CE) directly models the query–
chunk relevance by jointly encoding the (query, chunk) pair, producing
a true relevance score rather than a similarity proxy.

Pipeline: top-15 RRF candidates → CE reranker → top-5 returned to LLM.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - 22M params, CPU-fast (~20ms per pair)
  - Trained on MS MARCO passage ranking (165k Q&A pairs)
  - Significantly better relevance than cosine similarity alone
"""
from __future__ import annotations

import logging
from typing import Optional

from retrieval.dense import RetrievedChunk

logger = logging.getLogger(__name__)

# RRF smoothing constant — standard value from the original paper
_RRF_K = 60


class RRFFusion:
    """
    Merges ranked result lists from multiple retrievers using RRF.

    Usage:
        fuser = RRFFusion()
        merged = fuser.fuse({
            "dense": dense_results,
            "bm25":  bm25_results,
        })
    """

    def fuse(
        self,
        retriever_results: dict[str, list[RetrievedChunk]],
        top_k: Optional[int] = None,
    ) -> list[RetrievedChunk]:
        """
        Fuse multiple ranked lists via RRF.

        Args:
            retriever_results: mapping of retriever name → ranked chunk list
            top_k: number of results to return (None = return all)

        Returns:
            Deduplicated list of RetrievedChunk sorted by RRF score desc.
            The returned chunks carry the retriever tag of whichever
            retriever ranked them highest.
        """
        # chunk_id → cumulative RRF score
        rrf_scores: dict[str, float] = {}
        # chunk_id → best chunk object (from highest-scoring retriever)
        chunk_map:  dict[str, RetrievedChunk] = {}
        # chunk_id → which retrievers contributed
        sources:    dict[str, list[str]] = {}

        for retriever_name, results in retriever_results.items():
            for rank, chunk in enumerate(results, start=1):
                cid = chunk.chunk_id
                contribution = 1.0 / (_RRF_K + rank)
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + contribution

                if cid not in chunk_map:
                    chunk_map[cid] = chunk
                    sources[cid]   = [retriever_name]
                else:
                    sources[cid].append(retriever_name)
                    # Keep the chunk object from the retriever with the
                    # higher per-retriever score (better text quality signal)
                    if chunk.score > chunk_map[cid].score:
                        chunk_map[cid] = chunk

        # Sort by RRF score
        sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)
        if top_k:
            sorted_ids = sorted_ids[:top_k]

        # Annotate each chunk with its fused score and source retrievers
        fused: list[RetrievedChunk] = []
        for cid in sorted_ids:
            chunk = chunk_map[cid]
            chunk.score     = rrf_scores[cid]
            chunk.retriever = "+".join(sorted(sources[cid]))
            fused.append(chunk)

        logger.debug(
            "RRF: merged %d retriever(s), %d unique chunks → top %d",
            len(retriever_results),
            len(rrf_scores),
            len(fused),
        )
        return fused


class CrossEncoderReranker:
    """
    Reranks a candidate set using a cross-encoder relevance model.

    The cross-encoder jointly encodes (query, passage) pairs and outputs
    a calibrated relevance logit — far more accurate than cosine similarity,
    at the cost of O(N) inference calls per query.

    We run it only on the top-N RRF candidates (default: 15) to keep
    latency low, then return the top-K (default: 5) to the generator.
    """

    def __init__(self) -> None:
        self._model = None   # lazy-loaded

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """
        Rerank `chunks` against `query`. Returns top_k chunks sorted by
        cross-encoder score (highest first).

        Falls back to the original RRF order if the model fails to load.
        """
        if not chunks:
            return chunks

        try:
            model = self._get_model()
        except Exception as exc:
            logger.warning("Cross-encoder unavailable (%s) — using RRF order.", exc)
            return chunks[:top_k]

        pairs = [(query, chunk.parent_text or chunk.text) for chunk in chunks]

        try:
            scores = model.predict(pairs, show_progress_bar=False)
        except Exception as exc:
            logger.warning("Cross-encoder inference failed (%s) — using RRF order.", exc)
            return chunks[:top_k]

        # Attach CE scores and sort
        scored = sorted(
            zip(chunks, scores),
            key=lambda x: float(x[1]),
            reverse=True,
        )

        reranked = []
        for chunk, ce_score in scored[:top_k]:
            chunk.score = float(ce_score)   # replace RRF score with CE score
            reranked.append(chunk)

        logger.debug(
            "Cross-encoder reranked %d → top %d (top score: %.3f)",
            len(chunks), top_k, reranked[0].score if reranked else 0,
        )
        return reranked

    # ── Model loading ──────────────────────────────────────────

    def _get_model(self):
        if self._model is None:
            self._model = self._load_model()
        return self._model

    @staticmethod
    def _load_model():
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Install sentence-transformers: pip install sentence-transformers"
            ) from exc

        logger.info("Loading cross-encoder model…")
        model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            max_length=512,
        )
        logger.info("Cross-encoder ready.")
        return model
