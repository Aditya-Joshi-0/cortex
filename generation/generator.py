"""
Cortex RAG — LLM Generator

Handles:
  - Context assembly (parent chunks + citations)
  - Prompt construction with strict grounding instructions
  - Streaming generation via Groq API
  - Structured citation extraction from the response
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Generator, Optional

from config import get_settings
from retrieval.dense import RetrievedChunk

logger = logging.getLogger(__name__)

# ── Prompt templates ───────────────────────────────────────────

SYSTEM_PROMPT = """\
You are Cortex, an expert research assistant with access to a curated knowledge base.

Rules you MUST follow:
1. Answer ONLY using the provided context passages. Do not use prior knowledge.
2. After each factual claim, add an inline citation using the format [N] where N is the
   passage number from the context.
3. If the context does not contain enough information to answer, say:
   "I don't have sufficient information in the knowledge base to answer this."
4. Keep your answer focused and precise. Use markdown formatting where helpful.
5. At the end of your response, list the cited sources under a "## Sources" heading.
"""

USER_PROMPT_TEMPLATE = """\
## Context passages

{context}

---

## Question

{query}

Answer based strictly on the context passages above. Include inline [N] citations.
"""


# ── Data classes ──────────────────────────────────────────────

@dataclass
class GenerationRequest:
    query: str
    chunks: list[RetrievedChunk]
    stream: bool = True


@dataclass
class Citation:
    number: int
    title: str
    source: str
    chunk_id: str
    score: float


@dataclass
class GenerationResponse:
    answer: str
    citations: list[Citation] = field(default_factory=list)
    model: str = ""
    usage: dict = field(default_factory=dict)


# ── Generator ─────────────────────────────────────────────────

class Generator:
    """
    Generates grounded, cited answers from retrieved chunks.

    Streaming example:
        gen = Generator()
        for token in gen.stream(GenerationRequest(query, chunks)):
            print(token, end="", flush=True)
    """

    def __init__(self) -> None:
        self._client = None

    # ── Public API ─────────────────────────────────────────────

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Blocking generation. Returns full response with citations."""
        messages = self._build_messages(request)
        cfg = get_settings()
        client = self._get_client()

        response = client.chat.completions.create(
            model=cfg.groq_model,
            messages=messages,
            temperature=cfg.groq_temperature,
            max_tokens=cfg.groq_max_tokens,
            stream=False,
        )
        answer = response.choices[0].message.content or ""
        citations = self._build_citations(request.chunks)

        return GenerationResponse(
            answer=answer,
            citations=citations,
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            },
        )

    def stream(
        self,
        request: GenerationRequest,
    ) -> Generator[str, None, None]:
        """
        Token-by-token streaming generator.
        Yields string tokens as they arrive from Groq.

        Usage:
            for token in generator.stream(request):
                yield token   # pass to SSE / websocket / Streamlit
        """
        messages = self._build_messages(request)
        cfg = get_settings()
        client = self._get_client()

        stream = client.chat.completions.create(
            model=cfg.groq_model,
            messages=messages,
            temperature=cfg.groq_temperature,
            max_tokens=cfg.groq_max_tokens,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

    def build_sources_block(self, chunks: list[RetrievedChunk]) -> str:
        """
        Returns a markdown sources block for appending after the streamed answer.
        Example:
            ## Sources
            [1] **Attention Is All You Need** — attention_paper.pdf (score: 0.94)
        """
        lines = ["", "## Sources"]
        for i, chunk in enumerate(chunks, start=1):
            lines.append(
                f"[{i}] **{chunk.title}** — `{chunk.source}` "
                f"*(relevance: {chunk.score:.2f})*"
            )
        return "\n".join(lines)

    # ── Prompt construction ───────────────────────────────────

    @staticmethod
    def _build_messages(request: GenerationRequest) -> list[dict]:
        context_parts: list[str] = []
        for i, chunk in enumerate(request.chunks, start=1):
            # Use parent_text for LLM context (wider context window),
            # child text is used only for citation display
            context_text = chunk.parent_text or chunk.text
            context_parts.append(
                f"[{i}] (Source: {chunk.title})\n{context_text}"
            )
        context_str = "\n\n---\n\n".join(context_parts)

        user_content = USER_PROMPT_TEMPLATE.format(
            context=context_str,
            query=request.query,
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ]

    @staticmethod
    def _build_citations(chunks: list[RetrievedChunk]) -> list[Citation]:
        return [
            Citation(
                number=i,
                title=chunk.title,
                source=chunk.source,
                chunk_id=chunk.chunk_id,
                score=chunk.score,
            )
            for i, chunk in enumerate(chunks, start=1)
        ]

    # ── Groq client ───────────────────────────────────────────

    def _get_client(self):
        if self._client is None:
            self._client = self._init_client()
        return self._client

    @staticmethod
    def _init_client():
        cfg = get_settings()
        if not cfg.groq_api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to your .env file."
            )
        try:
            from groq import Groq  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Install groq: pip install groq"
            ) from exc

        return Groq(api_key=cfg.groq_api_key)
