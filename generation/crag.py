"""
Cortex RAG — Corrective RAG (CRAG) Gate (Phase 3)

The problem CRAG solves
────────────────────────
Standard RAG always passes retrieved chunks to the LLM, even when:
  - The query is ambiguous and the retrieved chunks are off-topic
  - The knowledge base simply doesn't contain the answer
  - The retrieved chunks contradict each other

In all three cases, the LLM will either hallucinate or produce a
confused answer. CRAG adds a grading step BEFORE generation:

                 ┌─── GOOD ────► Generator (proceed normally)
Query → Retrieve ┤
                 ├─── POOR ────► Rewrite query → Re-retrieve → Generator
                 └─── ABSENT ──► Web search fallback → Generator

Grading
────────
An LLM-as-judge evaluates (query, retrieved_chunks) and returns:
  {
    "grade": "GOOD" | "POOR" | "ABSENT",
    "relevance_score": 0.0–1.0,
    "has_sufficient_context": true | false,
    "reasoning": "..."
  }

Grade definitions:
  GOOD    — chunks are relevant and sufficient for the query
  POOR    — chunks are partially relevant; try rewriting the query
  ABSENT  — knowledge base clearly doesn't contain the answer;
             fall back to web search

Query rewriting
────────────────
When grade == POOR, we expand the query using chain-of-thought:
the grader's `reasoning` field (why did retrieval fail?) is fed
back as context for a rewrite prompt. This makes the rewrite
semantically targeted, not just rephrased.

Web search fallback
────────────────────
When grade == ABSENT, we call Tavily (preferred) or DuckDuckGo
(no API key needed) and package the top-3 web results as synthetic
RetrievedChunk objects with source="web_search". These flow into
the same generator unchanged.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from config import get_settings
from retrieval.dense import RetrievedChunk

logger = logging.getLogger(__name__)


# ── Grade enum ─────────────────────────────────────────────────

class RetrievalGrade(str, Enum):
    GOOD   = "GOOD"     # proceed to generation
    POOR   = "POOR"     # rewrite query and re-retrieve
    ABSENT = "ABSENT"   # fall back to web search


# ── Result dataclass ───────────────────────────────────────────

@dataclass
class CRAGResult:
    grade: RetrievalGrade
    relevance_score: float
    has_sufficient_context: bool
    reasoning: str
    final_chunks: list[RetrievedChunk]    # chunks to pass to generator
    rewritten_query: Optional[str] = None # set if grade was POOR
    web_search_used: bool = False


# ── Prompt templates ───────────────────────────────────────────

_GRADER_PROMPT = """\
You are a retrieval quality judge. Given a user query and retrieved passages,
assess whether the passages contain sufficient information to answer the query.

Return ONLY a JSON object in this exact format (no markdown, no preamble):
{{
  "grade": "<GOOD|POOR|ABSENT>",
  "relevance_score": <float 0.0-1.0>,
  "has_sufficient_context": <true|false>,
  "reasoning": "<one sentence explaining your assessment>"
}}

Grades:
  GOOD   — passages are clearly relevant and contain enough information to answer
  POOR   — passages are partially relevant but incomplete or off-topic; retrieval should be retried
  ABSENT — the knowledge base clearly does not contain information about this query

User query: {query}

Retrieved passages:
{passages}
"""

_REWRITE_PROMPT = """\
A retrieval system failed to find good results for the following query.
The grader's feedback explains why the results were poor.

Original query: {query}
Grader feedback: {reasoning}

Rewrite the query to be more specific and likely to retrieve better results.
Apply these strategies: expand acronyms, add domain context, use alternative terms.

Return ONLY the rewritten query string, no explanation.
"""


# ── CRAG Gate ──────────────────────────────────────────────────

class CRAGGate:
    """
    Corrective RAG gate that sits between retrieval and generation.

    Usage (in orchestrator):
        crag = CRAGGate()
        result = crag.evaluate(
            query=user_query,
            chunks=retrieved_chunks,
            retriever_fn=retriever.retrieve,   # callable for re-retrieval
        )
        # result.final_chunks → pass to generator
        # result.grade → log for evaluation dashboard
    """

    def __init__(self) -> None:
        self._llm = None

    # ── Public API ─────────────────────────────────────────────

    def evaluate(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        retriever_fn: Optional[callable] = None,
        max_retries: int = 1,
    ) -> CRAGResult:
        """
        Grade retrieved chunks and apply corrective action if needed.

        Args:
            query:        the user's original query
            chunks:       chunks returned by the retrieval pipeline
            retriever_fn: callable(query: str) → list[RetrievedChunk]
                          used for re-retrieval on POOR grade
            max_retries:  max number of rewrite+re-retrieve cycles
        """
        # Grade the initial retrieval
        grade_result = self._grade(query, chunks)
        logger.info(
            "CRAG grade: %s (score=%.2f, sufficient=%s) — %s",
            grade_result["grade"],
            grade_result["relevance_score"],
            grade_result["has_sufficient_context"],
            grade_result["reasoning"][:80],
        )

        grade = RetrievalGrade(grade_result["grade"])

        # ── GOOD: pass through unchanged ──────────────────────
        if grade == RetrievalGrade.GOOD:
            return CRAGResult(
                grade=grade,
                relevance_score=grade_result["relevance_score"],
                has_sufficient_context=True,
                reasoning=grade_result["reasoning"],
                final_chunks=chunks,
            )

        # ── POOR: rewrite query and re-retrieve ───────────────
        if grade == RetrievalGrade.POOR and retriever_fn and max_retries > 0:
            rewritten = self._rewrite_query(query, grade_result["reasoning"])
            logger.info("CRAG rewrite: '%s' → '%s'", query[:50], rewritten[:50])

            try:
                new_chunks = retriever_fn(rewritten)
                # Re-grade the new results (once — no infinite loop)
                new_grade = self._grade(rewritten, new_chunks)
                return CRAGResult(
                    grade=RetrievalGrade(new_grade["grade"]),
                    relevance_score=new_grade["relevance_score"],
                    has_sufficient_context=new_grade["has_sufficient_context"],
                    reasoning=new_grade["reasoning"],
                    final_chunks=new_chunks or chunks,  # fall back if retry also empty
                    rewritten_query=rewritten,
                )
            except Exception as exc:
                logger.warning("Re-retrieval after rewrite failed: %s", exc)
                # Fall through to returning original chunks with POOR grade
                return CRAGResult(
                    grade=grade,
                    relevance_score=grade_result["relevance_score"],
                    has_sufficient_context=False,
                    reasoning=grade_result["reasoning"],
                    final_chunks=chunks,
                    rewritten_query=rewritten,
                )

        # ── ABSENT: web search fallback ────────────────────────
        if grade == RetrievalGrade.ABSENT:
            web_chunks = self._web_search_fallback(query)
            if web_chunks:
                return CRAGResult(
                    grade=grade,
                    relevance_score=0.0,
                    has_sufficient_context=True,
                    reasoning=grade_result["reasoning"],
                    final_chunks=web_chunks,
                    web_search_used=True,
                )
            # Web search also failed — return original chunks with warning
            return CRAGResult(
                grade=grade,
                relevance_score=0.0,
                has_sufficient_context=False,
                reasoning=f"Knowledge base: {grade_result['reasoning']}. Web search also returned no results.",
                final_chunks=chunks,
            )

        # Default: return original chunks unchanged
        return CRAGResult(
            grade=grade,
            relevance_score=grade_result["relevance_score"],
            has_sufficient_context=grade_result["has_sufficient_context"],
            reasoning=grade_result["reasoning"],
            final_chunks=chunks,
        )

    # ── LLM grader ────────────────────────────────────────────

    def _grade(self, query: str, chunks: list[RetrievedChunk]) -> dict:
        """Call LLM to grade retrieval quality. Returns parsed dict."""
        if not chunks:
            return {
                "grade": "ABSENT",
                "relevance_score": 0.0,
                "has_sufficient_context": False,
                "reasoning": "No chunks were retrieved.",
            }

        passages = "\n\n".join(
            f"[{i}] {c.title}: {c.text[:400]}"
            for i, c in enumerate(chunks[:5], 1)
        )

        try:
            client = self._get_llm()
            cfg = get_settings()
            response = client.chat.completions.create(
                model=cfg.groq_model,
                messages=[{
                    "role": "user",
                    "content": _GRADER_PROMPT.format(query=query, passages=passages),
                }],
                temperature=0.0,
                max_tokens=200,
            )
            raw = response.choices[0].message.content or "{}"
            return self._parse_grade(raw)

        except Exception as exc:
            logger.warning("CRAG grader LLM call failed: %s", exc)
            # Safe default: assume GOOD to avoid blocking the pipeline
            return {
                "grade": "GOOD",
                "relevance_score": 0.5,
                "has_sufficient_context": True,
                "reasoning": f"Grader unavailable ({exc}); passing through.",
            }

    def _parse_grade(self, raw: str) -> dict:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return {
                "grade": "GOOD", "relevance_score": 0.5,
                "has_sufficient_context": True, "reasoning": "Parse error.",
            }

        grade_str = data.get("grade", "GOOD").upper()
        if grade_str not in {"GOOD", "POOR", "ABSENT"}:
            grade_str = "GOOD"

        return {
            "grade": grade_str,
            "relevance_score": float(data.get("relevance_score", 0.5)),
            "has_sufficient_context": bool(data.get("has_sufficient_context", True)),
            "reasoning": str(data.get("reasoning", "")),
        }

    # ── Query rewriter ────────────────────────────────────────

    def _rewrite_query(self, original_query: str, reasoning: str) -> str:
        try:
            client = self._get_llm()
            cfg = get_settings()
            response = client.chat.completions.create(
                model=cfg.groq_model,
                messages=[{
                    "role": "user",
                    "content": _REWRITE_PROMPT.format(
                        query=original_query, reasoning=reasoning
                    ),
                }],
                temperature=0.3,
                max_tokens=128,
            )
            rewritten = (response.choices[0].message.content or "").strip()
            return rewritten if rewritten else original_query
        except Exception as exc:
            logger.warning("Query rewrite failed: %s", exc)
            return original_query

    # ── Web search fallback ───────────────────────────────────

    def _web_search_fallback(self, query: str) -> list[RetrievedChunk]:
        """
        Try Tavily first (better quality), then DuckDuckGo (no API key).
        Returns synthetic RetrievedChunk objects from web results.
        """
        chunks = self._tavily_search(query) or self._duckduckgo_search(query)
        if chunks:
            logger.info("CRAG web fallback: %d results for '%s'", len(chunks), query[:50])
        return chunks

    def _tavily_search(self, query: str) -> list[RetrievedChunk]:
        try:
            from tavily import TavilyClient  # type: ignore
            cfg = get_settings()
            api_key = cfg.tavily_api_key
            if not api_key:
                return []
            client = TavilyClient(api_key=api_key)
            results = client.search(query, max_results=3)
            return [
                self._web_result_to_chunk(r.get("content", ""), r.get("url", ""), r.get("title", "Web"))
                for r in results.get("results", [])
                if r.get("content")
            ]
        except Exception:
            return []

    def _duckduckgo_search(self, query: str) -> list[RetrievedChunk]:
        try:
            from duckduckgo_search import DDGS  # type: ignore
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=3):
                    results.append(
                        self._web_result_to_chunk(
                            r.get("body", ""), r.get("href", ""), r.get("title", "Web")
                        )
                    )
            return results
        except Exception:
            return []

    @staticmethod
    def _web_result_to_chunk(text: str, url: str, title: str) -> RetrievedChunk:
        import hashlib
        cid = hashlib.sha256(url.encode()).hexdigest()[:16]
        return RetrievedChunk(
            chunk_id=cid,
            doc_id="web",
            source=url,
            title=title,
            text=text[:1500],
            parent_text=text[:1500],
            chunk_index=0,
            score=0.6,       # neutral score for web results
            retriever="web_search",
        )

    # ── Groq client ───────────────────────────────────────────

    def _get_llm(self):
        if self._llm is None:
            cfg = get_settings()
            if not cfg.groq_api_key:
                raise RuntimeError("GROQ_API_KEY not set")
            from groq import Groq  # type: ignore
            self._llm = Groq(api_key=cfg.groq_api_key)
        return self._llm
