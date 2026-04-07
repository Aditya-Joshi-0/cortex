"""
Cortex RAG — RAGAS Evaluation Harness (Phase 4)

Why reference-free metrics?
────────────────────────────
Classic RAG evaluation requires ground-truth answers (golden QA pairs).
We don't have those at runtime. RAGAS provides three metrics that need
only (question, answer, retrieved_contexts):

  faithfulness        — Does the answer make claims supported by the context?
                        Computed by asking an LLM to identify each claim in
                        the answer, then checking each claim against the context.
                        Score = supported_claims / total_claims.

  answer_relevancy    — Does the answer actually address the question?
                        Computed by generating N hypothetical questions from the
                        answer and measuring cosine similarity to the original
                        question. Low score = answer talks about something else.

  context_precision   — Are the retrieved chunks actually relevant to the query?
                        Computed by asking an LLM whether each chunk is useful
                        for answering the query. Score = relevant_chunks / total.

We also compute two lightweight custom metrics without any LLM calls:

  context_utilisation — What fraction of the retrieved chunks are cited in the
                        answer? (Count [1], [2]... citation markers.) A low score
                        means the generator ignored most of what was retrieved.

  mean_chunk_score    — Average retrieval score (post-reranking) of the final
                        chunks. Tracks retrieval quality independently of answer
                        quality. Useful for spotting when CRAG rewrites help.

Running mode
────────────
Evaluation is async — it runs in a background thread after the response
has been streamed to the user, so it never adds latency to the query path.
Results are written to the EvalStore (SQLite) and appear in the dashboard.

If RAGAS is not installed or the LLM call fails, only the two custom
metrics (context_utilisation, mean_chunk_score) are computed and stored.
This ensures the evaluation pipeline never blocks ingestion or queries.
"""
from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from typing import Optional

from evaluation.store import EvalMetricEntry, EvalStore
from retrieval.dense import RetrievedChunk

logger = logging.getLogger(__name__)


@dataclass
class EvalInput:
    """Everything needed to evaluate one query-response pair."""
    query_log_id: int
    query: str
    answer: str
    chunks: list[RetrievedChunk] = field(default_factory=list)


@dataclass
class EvalResult:
    faithfulness:        Optional[float] = None
    answer_relevancy:    Optional[float] = None
    context_precision:   Optional[float] = None
    context_utilisation: Optional[float] = None
    mean_chunk_score:    Optional[float] = None

    def as_store_entry(self, query_log_id: int) -> EvalMetricEntry:
        return EvalMetricEntry(
            query_log_id=query_log_id,
            faithfulness=self.faithfulness,
            answer_relevancy=self.answer_relevancy,
            context_precision=self.context_precision,
            context_utilisation=self.context_utilisation,
            mean_chunk_score=self.mean_chunk_score,
        )


class RAGASEvaluator:
    """
    Computes RAGAS + custom metrics for a query-response pair.

    Usage — fire-and-forget (non-blocking):
        evaluator = RAGASEvaluator(store)
        evaluator.evaluate_async(EvalInput(
            query_log_id=log_id,
            query="What is attention?",
            answer="Attention is...",
            chunks=final_chunks,
        ))

    Usage — blocking (for testing):
        result = evaluator.evaluate(eval_input)
    """

    def __init__(self, store: Optional[EvalStore] = None) -> None:
        self._store = store or EvalStore()
        self._ragas_available = self._check_ragas()

    # ── Public API ─────────────────────────────────────────────

    def evaluate_async(self, inp: EvalInput) -> None:
        """
        Run evaluation in a daemon thread. Returns immediately.
        Results are written to EvalStore when complete.
        """
        thread = threading.Thread(
            target=self._run_and_store,
            args=(inp,),
            daemon=True,
            name=f"ragas-eval-{inp.query_log_id}",
        )
        thread.start()

    def evaluate(self, inp: EvalInput) -> EvalResult:
        """Blocking evaluation. Returns EvalResult."""
        result = EvalResult()

        # ── Custom metrics (no LLM, always computed) ──────────
        result.context_utilisation = self._context_utilisation(inp.answer, inp.chunks)
        result.mean_chunk_score    = self._mean_chunk_score(inp.chunks)

        # ── RAGAS metrics (LLM-based, may be skipped) ─────────
        if self._ragas_available and inp.chunks:
            ragas_scores = self._run_ragas(inp)
            result.faithfulness      = ragas_scores.get("faithfulness")
            result.answer_relevancy  = ragas_scores.get("answer_relevancy")
            result.context_precision = ragas_scores.get("context_precision")
        else:
            if not self._ragas_available:
                logger.debug("RAGAS not installed — only custom metrics computed.")

        return result

    # ── Private ────────────────────────────────────────────────

    def _run_and_store(self, inp: EvalInput) -> None:
        try:
            result = self.evaluate(inp)
            self._store.log_metrics(result.as_store_entry(inp.query_log_id))
            logger.debug(
                "Eval stored for query %d: faith=%.2f rel=%.2f prec=%.2f util=%.2f",
                inp.query_log_id,
                result.faithfulness      or 0,
                result.answer_relevancy  or 0,
                result.context_precision or 0,
                result.context_utilisation or 0,
            )
        except Exception as exc:
            logger.warning("Eval failed for query %d: %s", inp.query_log_id, exc)

    def _run_ragas(self, inp: EvalInput) -> dict:
        """
        Call RAGAS library. Returns dict of metric_name → score.
        Returns empty dict on any failure.
        """
        try:
            from datasets import Dataset  # type: ignore
            from ragas import evaluate as ragas_evaluate  # type: ignore
            from ragas.metrics import (  # type: ignore
                answer_relevancy,
                context_precision,
                faithfulness,
            )
            from config import get_settings
            cfg = get_settings()

            # RAGAS expects a HuggingFace Dataset
            data = {
                "question":  [inp.query],
                "answer":    [inp.answer],
                "contexts":  [[c.parent_text or c.text for c in inp.chunks]],
                # reference not available at runtime — omit context_recall
            }
            dataset = Dataset.from_dict(data)

            scores = ragas_evaluate(
                dataset,
                metrics=[faithfulness, answer_relevancy, context_precision],
                raise_exceptions=False,
            )
            df = scores.to_pandas()
            return {
                "faithfulness":      float(df["faithfulness"].iloc[0])      if "faithfulness"      in df else None,
                "answer_relevancy":  float(df["answer_relevancy"].iloc[0])  if "answer_relevancy"  in df else None,
                "context_precision": float(df["context_precision"].iloc[0]) if "context_precision" in df else None,
            }

        except Exception as exc:
            logger.warning("RAGAS evaluation failed: %s", exc)
            return {}

    # ── Custom metrics (no LLM required) ──────────────────────

    @staticmethod
    def _context_utilisation(answer: str, chunks: list[RetrievedChunk]) -> float:
        """
        Fraction of retrieved chunks cited in the answer.
        Looks for inline [N] citation markers.
        """
        if not chunks:
            return 0.0
        cited_indices = set(int(n) for n in re.findall(r"\[(\d+)\]", answer))
        cited = sum(1 for i in range(1, len(chunks) + 1) if i in cited_indices)
        return round(cited / len(chunks), 3)

    @staticmethod
    def _mean_chunk_score(chunks: list[RetrievedChunk]) -> float:
        """Average retrieval score of the final chunks."""
        if not chunks:
            return 0.0
        return round(sum(c.score for c in chunks) / len(chunks), 3)

    @staticmethod
    def _check_ragas() -> bool:
        try:
            import ragas  # type: ignore  # noqa: F401
            import datasets  # type: ignore  # noqa: F401
            return True
        except ImportError:
            return False
