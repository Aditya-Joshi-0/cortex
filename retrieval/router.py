"""
Cortex RAG — Query Router

Classifies the user's query into one of five intent types and maps
each type to the optimal combination of retrieval strategies.

Intent taxonomy
───────────────
  factual_lookup   "What is the definition of attention?"
                   → Dense only (semantic match is sufficient)

  keyword_exact    "Show me papers mentioning PagedAttention or vLLM"
                   → BM25 primary + Dense secondary
                     (exact term matching beats semantic blur)

  comparison       "What are the differences between GPT-4 and Llama 2?"
                   → Dense + BM25 (need breadth across multiple sources)

  multi_hop        "Who wrote the attention paper, and what did they work on later?"
                   → Dense + Graph (Phase 3 adds graph traversal)

  code_or_formula  "Show the pseudocode for scaled dot-product attention"
                   → BM25 + Dense (code keywords are exact; semantic helps)

Classification uses a zero-shot LLM call with structured JSON output.
Falls back to ["dense", "bm25"] (safe default) if the call fails.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from config import get_settings

logger = logging.getLogger(__name__)


class QueryIntent(str, Enum):
    FACTUAL_LOOKUP  = "factual_lookup"
    KEYWORD_EXACT   = "keyword_exact"
    COMPARISON      = "comparison"
    MULTI_HOP       = "multi_hop"
    CODE_OR_FORMULA = "code_or_formula"


@dataclass
class RoutingDecision:
    intent: QueryIntent
    strategies: list[str]       # e.g. ["dense", "bm25"]
    confidence: float           # 0–1
    reasoning: str              # one-sentence explanation (useful for debug UI)


# Maps each intent to its retrieval strategy set
_STRATEGY_MAP: dict[QueryIntent, list[str]] = {
    QueryIntent.FACTUAL_LOOKUP:  ["dense"],
    QueryIntent.KEYWORD_EXACT:   ["bm25", "dense"],
    QueryIntent.COMPARISON:      ["dense", "bm25"],
    QueryIntent.MULTI_HOP:       ["dense", "graph"],   # graph added in Phase 3
    QueryIntent.CODE_OR_FORMULA: ["bm25", "dense"],
}

_CLASSIFICATION_PROMPT = """\
You are a query classifier for a RAG system. Your job is to classify the user's
question into exactly ONE of these intent types:

  factual_lookup  — asks for a definition, explanation, or single-fact answer
  keyword_exact   — relies on specific technical terms, product names, or acronyms
  comparison      — asks to compare, contrast, or differentiate two or more things
  multi_hop       — requires connecting information across multiple sources or
                    following a chain of relationships (who wrote X and what did they later do)
  code_or_formula — asks for code, pseudocode, equations, or algorithmic steps

Respond ONLY with a JSON object in this exact format (no markdown, no preamble):
{{
  "intent": "<one of the five types above>",
  "confidence": <float 0.0–1.0>,
  "reasoning": "<one brief sentence explaining your choice>"
}}

User query: {query}
"""

class QueryRouter:
    """
    Classifies query intent using a zero-shot Groq/LLM call.

    Falls back gracefully: if the API call fails or returns unparseable
    JSON, returns a safe default (dense + bm25) with confidence 0.0.
    """

    def __init__(self) -> None:
        self._client = None

    # ── Public ─────────────────────────────────────────────────

    def route(self, query: str) -> RoutingDecision:
        """
        Classify query intent and return retrieval strategy set.

        Example:
            decision = router.route("What is PagedAttention?")
            # RoutingDecision(intent=keyword_exact, strategies=["bm25","dense"], ...)
        """
        try:
            decision = self._classify_via_llm(query)
        except Exception as exc:
            logger.warning("Router LLM call failed (%s) — using default.", exc)
            decision = self._default_decision()

        logger.info(
            "Router: '%s' → %s (strategies=%s, conf=%.2f)",
            query[:60], decision.intent.value, decision.strategies, decision.confidence
        )
        return decision

    # ── LLM classification ────────────────────────────────────

    def _classify_via_llm(self, query: str) -> RoutingDecision:
        client = self._get_client()
        cfg = get_settings()

        response = client.chat.completions.create(
            model=cfg.groq_model,
            messages=[
                {
                    "role": "user",
                    "content": _CLASSIFICATION_PROMPT.format(query=query),
                }
            ],
            temperature=0.0,        # deterministic classification
            max_tokens=2048,
        )

        raw = response.choices[0].message.content or ""
        return self._parse_response(raw)

    def _parse_response(self, raw: str) -> RoutingDecision:
        """Parse the JSON response. Raises ValueError on bad output."""
        # Strip any accidental markdown fences
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        data = json.loads(raw)

        intent_str = data.get("intent", "").strip().lower()
        try:
            intent = QueryIntent(intent_str)
        except ValueError:
            # Graceful fallback if the LLM returns a non-standard string
            intent = QueryIntent.FACTUAL_LOOKUP

        strategies = _STRATEGY_MAP[intent]
        confidence = float(data.get("confidence", 0.8))
        reasoning  = str(data.get("reasoning", ""))

        return RoutingDecision(
            intent=intent,
            strategies=strategies,
            confidence=confidence,
            reasoning=reasoning,
        )

    @staticmethod
    def _default_decision() -> RoutingDecision:
        return RoutingDecision(
            intent=QueryIntent.FACTUAL_LOOKUP,
            strategies=["dense", "bm25"],
            confidence=0.0,
            reasoning="Default fallback — LLM classification unavailable.",
        )

    # ── Groq client ───────────────────────────────────────────

    def _get_client(self):
        if self._client is None:
            cfg = get_settings()
            if not cfg.groq_api_key:
                raise RuntimeError("GROQ_API_KEY is not set.")
            from groq import Groq  # type: ignore
            self._client = Groq(api_key=cfg.groq_api_key)
        return self._client
