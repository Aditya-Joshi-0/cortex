"""
Cortex RAG — LLM Generator

Supported providers
────────────────────
  groq        https://api.groq.com/openai/v1         (default)
  nvidia_nim  https://integrate.api.nvidia.com/v1
  openai      https://api.openai.com/v1
  custom      any OpenAI-compatible endpoint

All four expose the same OpenAI chat completions API, so one client
handles everything. The `openai` package is used for all providers;
Groq's own SDK is no longer required (though it still works if present).

Runtime override
─────────────────
GenerationRequest now accepts optional provider/model/api_key fields.
When set, they override the .env defaults for that single request.
This is how the UI model-selector works — it sends the chosen
provider+model with every query without touching server config.

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
from typing import Generator, Iterator, Optional

from config import get_settings
from retrieval.dense import RetrievedChunk

logger = logging.getLogger(__name__)

# ── Provider registry ──────────────────────────────────────────
PROVIDERS: dict[str, dict] = {
    "groq": {
        "label":    "Groq",
        "base_url": "https://api.groq.com/openai/v1",
        "env_key":  "groq_api_key",
        "models": [
            {"id": "openai/gpt-oss-120b",      "label": "OpenAI GPT-OSS-120B"},
            {"id": "llama-3.3-70b-versatile",  "label": "Llama 3.3 70B"},
            {"id": "llama-3.1-8b-instant",     "label": "Llama 3.1 8B"},
            {"id": "mixtral-8x7b-32768",       "label": "Mixtral 8×7B"},
            {"id": "gemma2-9b-it",             "label": "Gemma 2 9B"},
        ],
    },
    "nvidia_nim": {
        "label":    "NVIDIA NIM",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "env_key":  "nvidia_api_key",
        "models": [
            {"id": "google/gemma-4-31b-it",  "label": "Gemma 4 31B"},
            {"id": "openai/gpt-oss-120b",      "label": "OpenAI GPT-OSS-120B"},
            {"id": "meta/llama-3.3-70b-instruct",  "label": "Llama 3.3 70B"},
            {"id": "meta/llama-3.1-8b-instruct",   "label": "Llama 3.1 8B"},
            {"id": "mistralai/mixtral-8x22b-instruct", "label": "Mixtral 8×22B"},
            {"id": "microsoft/phi-3-medium-128k-instruct", "label": "Phi-3 Medium"},
            {"id": "google/gemma-2-27b-it",        "label": "Gemma 2 27B"},
        ],
    },
    "openai": {
        "label":    "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "env_key":  "openai_api_key",
        "models": [
            {"id": "gpt-4o",        "label": "GPT-4o"},
            {"id": "gpt-4o-mini",   "label": "GPT-4o mini"},
            {"id": "gpt-4-turbo",   "label": "GPT-4 Turbo"},
            {"id": "gpt-3.5-turbo", "label": "GPT-3.5 Turbo"},
        ],
    },
    "custom": {
        "label":    "Custom",
        "base_url": "",   # user-supplied at runtime
        "env_key":  "custom_api_key",
        "models": [],     # user-supplied at runtime
    },
}

# ── Prompts ────────────────────────────────────────────────────
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
6. You have access to the conversation history above. Use it to resolve follow-up
   references but always ground factual claims in the provided context passages.
"""

USER_PROMPT_TEMPLATE = """\
## Context passages

{context}

---

## Question

{query}

Answer based strictly on the context passages above. Include inline [N] citations.
"""

REWRITE_PROMPT = """\
You are a query rewriter for a retrieval system.
Given a conversation history and a follow-up question, rewrite the follow-up as a \
fully self-contained question that makes sense without the conversation history.

Rules:
- Resolve all pronouns (it, this, they, that, those, them) to their actual referents
- Expand vague references like "the first one", "that paper", "the approach above"
- If the question is already standalone and unambiguous, return it EXACTLY as-is
- Return ONLY the rewritten question — no explanation, no preamble

Conversation history:
{history}

Follow-up question: {query}"""

# ── Data classes ──────────────────────────────────────────────

@dataclass
class GenerationRequest:
    query:    str
    chunks:   list[RetrievedChunk]
    stream:   bool = True
    conversation: list[dict] = field(default_factory=list)  # [{role, content}, ...]    provider: Optional[str] = None   # e.g. "groq", "nvidia_nim", "openai", "custom"
    model:    Optional[str] = None   # model id string
    api_key:  Optional[str] = None   # override .env key for this request
    base_url: Optional[str] = None   # only used when provider == "custom"



@dataclass
class Citation:
    number:   int
    title:    str
    source:   str
    chunk_id: str
    score:    float


@dataclass
class GenerationResponse:
    answer:    str
    citations: list[Citation] = field(default_factory=list)
    model:     str = ""
    provider:  str = ""
    usage:     dict = field(default_factory=dict)


# ── Generator ─────────────────────────────────────────────────

class Generator:
    """
    Generates grounded, cited answers from retrieved chunks.
    Multi-provider LLM generator.
    The client is built fresh per unique (provider, model, api_key) tuple
    and cached in a small dict to avoid redundant instantiation across
    requests that share the same settings.

    Memory is injected as prior conversation turns in the message list:
    [system] → [user turn 1] → [assistant turn 1] → ... → [user + context]

    The retrieval context (RAG passages) is attached only to the FINAL
    user message. Prior turns are plain Q&A without context — the LLM
    uses them purely to resolve pronouns and follow-up references.

    Streaming example:
        gen = Generator()
        for token in gen.stream(GenerationRequest(query, chunks)):
            print(token, end="", flush=True)
    """

    def __init__(self) -> None:
        self._clients: dict[tuple, object] = {}

    # ── Public API ─────────────────────────────────────────────

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Blocking generation. Returns full response with citations."""
        client, resolved = self._resolve_client(request)
        messages = self._build_messages(request)

        response = client.chat.completions.create(
            model=resolved["model"],
            messages=messages,
            temperature=resolved["temperature"],
            max_tokens=resolved["max_tokens"],
            stream=False,
        )
        answer = response.choices[0].message.content or ""

        return GenerationResponse(
            answer=answer,
            citations=self._build_citations(request.chunks),
            model=response.model,
            provider=resolved["provider"],
            usage={
                "prompt_tokens":     getattr(response.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(response.usage, "completion_tokens", 0),
            },
        )
    def stream(self, request: GenerationRequest) -> Iterator[str]:
        """Token-by-token streaming. Yields raw string tokens."""
        client, resolved = self._resolve_client(request)

        messages = self._build_messages(request)

        stream_obj = client.chat.completions.create(
            model=resolved["model"],
            messages=messages,
            temperature=resolved["temperature"],
            max_tokens=resolved["max_tokens"],
            stream=True,
        )
        for chunk in stream_obj:
            # Guard against empty choices — the final [DONE] sentinel chunk
            # from some providers (e.g. NVIDIA NIM) arrives as choices:[].
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

    def rewrite_query(
        self,
        query: str,
        conversation: list[dict],
        provider: Optional[str] = None,
        model:    Optional[str] = None,
        api_key:  Optional[str] = None,
    ) -> str:
        """
        Rewrite a follow-up query into a standalone question using conversation
        history. Returns the original query unchanged if:
          - There is no prior conversation (nothing to resolve)
          - The rewrite call fails (safe fallback)
          - The rewritten text is empty

        Uses temperature=0 and max_tokens=200 — the cheapest possible call.

        Example:
            conversation = [
                {"role": "user",      "content": "What is the attention mechanism?"},
                {"role": "assistant", "content": "Attention allows the model to ..."},
            ]
            query = "Who invented it?"
            → "Who invented the attention mechanism?"
        """
        if not conversation or len(conversation) < 2:
            return query   # no history — nothing to resolve

        # Build a compact history string from the last 4 turns (2 exchanges)
        # to keep the rewrite prompt short and fast
        recent = conversation[-4:]
        history_str = "\n".join(
            f"{t['role'].upper()}: {t['content'][:300]}"
            for t in recent
        )

        prompt = REWRITE_PROMPT.format(history=history_str, query=query)

        try:
            # Build a minimal request just for the rewrite call
            class _MinimalReq:
                provider = provider
                model    = model
                api_key  = api_key
                base_url = None

            client, resolved = self._resolve_client(_MinimalReq())
            response = client.chat.completions.create(
                model=resolved["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=200,
                stream=False,
            )
            rewritten = (response.choices[0].message.content or "").strip()

            if rewritten and rewritten != query:
                logger.info(
                    "Memory rewrite: '%s' → '%s'", query[:60], rewritten[:60]
                )
                return rewritten

        except Exception as exc:
            logger.debug("Query rewrite failed (%s) — using original query", exc)

        return query

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

    # ── Client resolution ──────────────────────────────────────

    def _resolve_client(self, request: GenerationRequest) -> tuple:
        """
        Build (or retrieve cached) OpenAI-compatible client for the request.
        Returns (client, resolved_params_dict).
        """
        cfg = get_settings()

        provider_id = request.provider or getattr(cfg, "default_provider", "groq")
        provider    = PROVIDERS.get(provider_id, PROVIDERS["groq"])

        model = request.model or getattr(cfg, "groq_model", "llama-3.3-70b-versatile")
        # base_url: for known providers always use the registry URL — the client
        # may send a stale URL from a previous session (e.g. Groq's URL while
        # NVIDIA NIM is selected). Only trust request.base_url for "custom".
        if provider_id == "custom":
            base_url = request.base_url or getattr(cfg, "custom_base_url", "")
            if not base_url:
                raise RuntimeError(
                    "Custom provider requires a base URL. "
                    "Enter it in the model selector or set CUSTOM_BASE_URL in .env."
                )
        else:
            base_url = provider["base_url"]   # always authoritative for known providers

        # API key priority: request override → provider-specific env var
        # Never fall back to a different provider's key — that causes 401s.

        env_key_name = provider["env_key"]
        api_key = request.api_key or getattr(cfg, env_key_name, "")

        if not api_key:
            env_var = env_key_name.upper()
            raise RuntimeError(
                f"No API key for provider '{provider_id}'. "
                f"Set {env_var} in your .env file, or enter it in the model selector."
            )

        cache_key = (provider_id, model, api_key, base_url)
        if cache_key not in self._clients:
            self._clients[cache_key] = self._build_client(api_key, base_url)
            logger.info(
                "Built client for provider=%s model=%s base_url=%s",
                provider_id, model, base_url
            )

        resolved = {
            "provider":    provider_id,
            "model":       model,
            "temperature": getattr(cfg, "groq_temperature", 0.1),
            "max_tokens":  getattr(cfg, "groq_max_tokens", 4096),
        }
        return self._clients[cache_key], resolved

    @staticmethod
    def _build_client(api_key: str, base_url: str):
        """Build an OpenAI-compatible client pointing at base_url."""
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Install openai: pip install openai>=1.0"
            ) from exc
        return OpenAI(api_key=api_key, base_url=base_url)

    # ── Prompt helpers ─────────────────────────────────────────

    @staticmethod
    def _build_messages(request: GenerationRequest) -> list[dict]:
        """
        Build the full message list for the LLM call.

        Structure with conversation history:
          [system]
          [user: prior question 1]         ← conversation turns (no context)
          [assistant: prior answer 1]
          [user: prior question 2]
          [assistant: prior answer 2]
          ...
          [user: current question + RAG context passages]

        Without conversation history (or first turn):
          [system]
          [user: current question + RAG context passages]

        The RAG context is ONLY attached to the final user message.
        Prior turns are plain Q&A — they exist solely so the LLM can
        resolve pronouns and follow-up references from prior exchanges.
        """
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Insert prior conversation turns (without context — plain Q&A)
        for turn in request.conversation:
            messages.append({"role": turn["role"], "content": turn["content"]})

        # Final user message: current question + retrieved context
        context_parts = []

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
        messages.append({"role": "user", "content": user_content})

        return messages


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

    # def _get_client(self):
    #     if self._client is None:
    #         self._client = self._init_client()
    #     return self._client

    # @staticmethod
    # def _init_client():
    #     cfg = get_settings()
    #     if not cfg.groq_api_key:
    #         raise RuntimeError(
    #             "GROQ_API_KEY is not set. Add it to your .env file."
    #         )
    #     try:
    #         from groq import Groq  # type: ignore
    #     except ImportError as exc:
    #         raise RuntimeError(
    #             "Install groq: pip install groq"
    #         ) from exc

    #     return Groq(api_key=cfg.groq_api_key)
