"""
Cortex RAG — Relation Extractors

Strategy pattern: both extractors share the same interface.
Switch between them with GRAPH_EXTRACTOR=rebel|llm in .env.

  RelationExtractor (abstract)
    ├── REBELExtractor   — local model, no API calls, Wikidata predicates
    └── LLMExtractor     — Mistral/LLM, free-form predicates, rate-limited

KnowledgeGraphBuilder accepts either via dependency injection, or
auto-selects based on config.get_settings().graph_extractor.

Adding a new extractor in the future:
  1. Subclass RelationExtractor
  2. Implement extract(chunk) → list[Triple]
  3. Register the name in _EXTRACTOR_REGISTRY at the bottom of this file
"""
from __future__ import annotations

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from config import get_settings
from ingestion.chunker import Chunk

logger = logging.getLogger(__name__)


# ── Shared dataclass ───────────────────────────────────────────

@dataclass
class Triple:
    subject: str
    predicate: str
    object: str
    chunk_id: str
    source: str
    extractor: str = "unknown"   # tracks which extractor produced this triple


# ── Abstract base ──────────────────────────────────────────────

class RelationExtractor(ABC):
    """
    Common interface for all relation extraction strategies.
    Subclasses must implement extract() only.
    """

    @abstractmethod
    def extract(self, chunk: Chunk) -> list[Triple]:
        """
        Extract (subject, predicate, object) triples from a single chunk.
        Must never raise — return [] on any failure.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in logging and triple.extractor field."""
        ...

    def extract_batch(self, chunks: list[Chunk]) -> dict[str, list[Triple]]:
        """
        Extract triples from a list of chunks.
        Default: calls extract() sequentially.
        Subclasses can override for true batching (e.g. REBEL).

        Returns: {chunk_id: [Triple, ...]}
        """
        return {chunk.chunk_id: self.extract(chunk) for chunk in chunks}


# ── REBEL extractor ────────────────────────────────────────────

# REBEL relation types that map cleanly to RAG-useful edges.
# The full Wikidata set has 220 types; we keep the ~40 most useful.
_REBEL_KEEP_RELATIONS = {
    "author", "developer", "creator", "founded by", "owned by",
    "instance of", "subclass of", "part of", "has part",
    "country", "country of origin", "located in", "headquarters location",
    "employer", "member of", "affiliation", "educated at",
    "award received", "occupation", "field of work", "notable work",
    "based on", "followed by", "follows", "influenced by", "has edition",
    "product or material produced", "used by", "manufacturer",
    "publication date", "academic degree", "applies to jurisdiction",
    "published in", "platform", "programming language", "license",
}


class REBELExtractor(RelationExtractor):
    """
    Local relation extraction using REBEL (Babelscape/rebel-large).

    Model facts:
      - 406M params (BART-large fine-tuned on Wikipedia + Wikidata)
      - Input: raw text sentence(s)
      - Output: decoded triplet string → parsed into (head, type, tail)
      - CPU inference: ~80–150ms per chunk on modern hardware
      - No API calls, no rate limits, fully offline after first download

    Batching:
      REBEL's tokeniser handles variable-length batches natively.
      extract_batch() sends all chunks in one forward pass, which is
      significantly faster than calling extract() in a loop.
      Max batch size is controlled by REBEL_BATCH_SIZE in config
      (default 8 — safe for 8GB RAM; raise to 16–32 with more RAM).

    Predicate normalisation:
      REBEL outputs Wikidata relation labels (e.g. "country of origin").
      We keep only relations in _REBEL_KEEP_RELATIONS (40 types) and
      discard the rest — this prevents graph noise from obscure predicates
      like "Wikimedia disambiguation page" or "image" polluting the graph.

    Download:
      First run downloads ~1.6GB to ~/.cache/huggingface/hub/.
      Subsequent runs load from cache in ~3s.
    """

    _REBEL_MODEL = "Babelscape/rebel-large"
    _MAX_INPUT_TOKENS = 256     # REBEL was trained on short passages
    _MAX_OUTPUT_TOKENS = 512

    def __init__(self) -> None:
        self._tokenizer = None
        self._model = None

    @property
    def name(self) -> str:
        return "rebel"

    # ── Public ──────────────────────────────────────────────────

    def extract(self, chunk: Chunk) -> list[Triple]:
        """Single-chunk extraction (sequential). Prefer extract_batch for speed."""
        results = self.extract_batch([chunk])
        return results.get(chunk.chunk_id, [])

    def extract_batch(self, chunks: list[Chunk]) -> dict[str, list[Triple]]:
        """
        True batched extraction. All chunks processed in a single model call.
        Falls back to sequential on memory errors.
        """
        if not chunks:
            return {}

        tok, model = self._load()
        cfg = get_settings()
        batch_size = getattr(cfg, "rebel_batch_size", 8)

        # Chunk text is truncated to avoid exceeding REBEL's context window
        texts = [c.text[:1200] for c in chunks]
        all_triples: dict[str, list[Triple]] = {c.chunk_id: [] for c in chunks}

        for batch_start in range(0, len(chunks), batch_size):
            batch_chunks = chunks[batch_start : batch_start + batch_size]
            batch_texts  = texts[batch_start : batch_start + batch_size]

            try:
                inputs = tok(
                    batch_texts,
                    max_length=self._MAX_INPUT_TOKENS,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                generated = model.generate(
                    **inputs,
                    max_length=self._MAX_OUTPUT_TOKENS,
                    num_beams=3,
                    early_stopping=True,
                )
                decoded = tok.batch_decode(generated, skip_special_tokens=False)

                for chunk, raw_output in zip(batch_chunks, decoded):
                    triples = self._parse_rebel_output(raw_output, chunk)
                    all_triples[chunk.chunk_id] = triples

            except Exception as exc:
                logger.warning("REBEL batch %d failed: %s", batch_start, exc)
                # Mark as empty rather than crashing the whole ingestion
                for chunk in batch_chunks:
                    all_triples[chunk.chunk_id] = []

        return all_triples

    # ── REBEL output parser ────────────────────────────────────

    def _parse_rebel_output(self, decoded: str, chunk: Chunk) -> list[Triple]:
        """
        Parse REBEL's special-token output format.

        REBEL outputs a string like:
          <triplet> Vaswani <subj> Attention Is All You Need <obj> author
          <triplet> Transformer <subj> NLP <obj> field of work

        We extract each triplet, filter to keep relations, normalise,
        and return Triple dataclasses.
        """
        triples: list[Triple] = []

        # Split on <triplet> delimiter
        raw_triplets = decoded.split("<triplet>")

        for raw in raw_triplets:
            raw = raw.strip()
            if not raw or "<subj>" not in raw or "<obj>" not in raw:
                continue

            try:
                # Format: "SUBJECT <subj> OBJECT <obj> RELATION"
                subj_split = raw.split("<subj>")
                subject = subj_split[0].strip()

                obj_rel = subj_split[1].split("<obj>")
                obj    = obj_rel[0].strip()
                relation = obj_rel[1].strip()

                # Clean up any residual special tokens
                for tok_str in ["</s>", "<s>", "<pad>"]:
                    relation = relation.replace(tok_str, "").strip()
                    subject  = subject.replace(tok_str, "").strip()
                    obj      = obj.replace(tok_str, "").strip()

                if not subject or not obj or not relation:
                    continue

                # Filter to useful relation types only
                if relation.lower() not in _REBEL_KEEP_RELATIONS:
                    continue

                triples.append(Triple(
                    subject=subject.title(),
                    predicate=relation.lower(),
                    object=obj.title(),
                    chunk_id=chunk.chunk_id,
                    source=chunk.source,
                    extractor=self.name,
                ))

            except (IndexError, AttributeError):
                continue

        return triples[:8]   # cap per chunk

    # ── Model loading ──────────────────────────────────────────

    def _load(self):
        if self._tokenizer is None or self._model is None:
            try:
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "Install transformers: pip install transformers"
                ) from exc

            logger.info("Loading REBEL model '%s' (first run downloads ~1.6GB)…", self._REBEL_MODEL)
            t0 = time.perf_counter()
            self._tokenizer = AutoTokenizer.from_pretrained(self._REBEL_MODEL)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self._REBEL_MODEL)
            self._model.eval()   # inference mode — disables dropout
            logger.info("REBEL loaded in %.1fs", time.perf_counter() - t0)

        return self._tokenizer, self._model


# ── LLM extractor (original method, preserved) ─────────────────

_LLM_PROMPT = """\
Extract factual relationships from the passage below.
Return ONLY a JSON array of triples. Each triple is:
  {{"subject": "...", "predicate": "...", "object": "..."}}

Rules:
- subject and object must be named entities (people, orgs, systems, concepts)
- predicate is a short verb phrase ("developed", "is based on", "introduced", "authored")
- Extract 0–5 triples maximum. If there are none, return []
- Return ONLY the JSON array, no explanation, no markdown

Passage:
{text}
"""


class LLMExtractor(RelationExtractor):
    """
    Relation extraction via Mistral LLM (the original Phase 3 method).

    Produces free-form, human-readable predicates ("introduced the concept of",
    "co-authored with") rather than the fixed Wikidata vocabulary that REBEL uses.

    Use this when:
      - You want rich, domain-specific predicate labels
      - Your corpus is small enough that rate limits aren't a problem
      - You want to fine-tune the extraction prompt for your specific domain

    Rate limiting:
      Set MISTRAL_RELATION_RPM in .env to cap requests-per-minute.
      Default is 0 (no cap). Mistral free tier allows ~30 RPM.
    """

    def __init__(self) -> None:
        self._llm = None

    @property
    def name(self) -> str:
        return "llm"

    def extract(self, chunk: Chunk) -> list[Triple]:
        try:
            client = self._get_llm()
            cfg = get_settings()

            if cfg.llm_server == "ollama":
                response = client.chat.complete(
                    model=cfg.ollama_model,
                    messages=[{
                    "role": "user",
                    "content": _LLM_PROMPT.format(text=chunk.text[:2000]),
                }],
                )
            else:    
                response = client.chat.complete(
                    model=cfg.mistral_model,
                    messages=[{
                        "role": "user",
                        "content": _LLM_PROMPT.format(text=chunk.text[:2000]),
                    }],
                    temperature=0.0,
                    max_tokens=512,
                )
            raw = response.choices[0].message.content or "[]"
            return self._parse(raw, chunk)

        except Exception as exc:
            logger.debug("LLM extractor failed for chunk %s: %s", chunk.chunk_id, exc)
            return []

    def _parse(self, raw: str, chunk: Chunk) -> list[Triple]:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
        try:
            items = json.loads(raw)
        except json.JSONDecodeError:
            return []

        triples: list[Triple] = []
        for item in items[:5]:
            if not isinstance(item, dict):
                continue
            s = str(item.get("subject", "")).strip()
            p = str(item.get("predicate", "")).strip()
            o = str(item.get("object", "")).strip()
            if s and p and o:
                triples.append(Triple(
                    subject=s.title(),
                    predicate=p.lower(),
                    object=o.title(),
                    chunk_id=chunk.chunk_id,
                    source=chunk.source,
                    extractor=self.name,
                ))
        return triples

    def _get_llm(self):
        if self._llm is None:
            cfg = get_settings()
            llm_server = cfg.llm_server

            if llm_server == "ollama":
                try:
                    from ollama import Client as ollama_client  # type: ignore
                except ImportError as exc:
                    raise RuntimeError(
                        "Install ollama client: pip install ollama"
                    ) from exc
                self._llm = ollama_client(host=cfg.ollama_host)
            else:
                if not cfg.mistral_api_key:
                    raise RuntimeError("MISTRAL_API_KEY not set")
                from mistralai.client import Mistral  # type: ignore
                self._llm = Mistral(api_key=cfg.mistral_api_key)
        return self._llm

# ── Entity density filter (Option 4) ──────────────────────────

class EntityDensityFilter(RelationExtractor):
    """
    Decorator that wraps any extractor and skips low-entity-density chunks.

    Rationale
    ─────────
    Chunks with 0–1 named entities rarely yield useful triples — a
    paragraph of methodology boilerplate has no entities to link.
    Scoring by entity density (entities per 100 tokens) and processing
    only the top N% of chunks cuts extraction time by ~70% with
    negligible graph quality loss.

    How density is computed
    ───────────────────────
    density = (spaCy NER entity count) / (token count / 100)

    This normalises for chunk length — a 50-token chunk with 3 entities
    scores higher than a 500-token chunk with the same 3 entities.

    Usage
    ─────
    # Wrap REBEL, keep top 30% of chunks (default):
    extractor = EntityDensityFilter(REBELExtractor())

    # Wrap LLM, keep top 20%, only chunks with ≥2 entities:
    extractor = EntityDensityFilter(
        LLMExtractor(),
        top_fraction=0.20,
        min_entity_count=2,
    )

    # Via config (wraps whatever GRAPH_EXTRACTOR is set to):
    GRAPH_EXTRACTOR=rebel-filtered   # rebel + density filter
    GRAPH_EXTRACTOR=llm-filtered     # llm   + density filter
    """

    def __init__(
        self,
        inner: RelationExtractor,
        top_fraction: Optional[float] = None,
        min_entity_count: Optional[int] = None,
    ) -> None:
        cfg = get_settings()
        self._inner = inner
        # top_fraction: process only the top X% most entity-dense chunks
        self._top_fraction   = top_fraction   or getattr(cfg, "density_top_fraction", 0.30)
        # min_entity_count: hard floor — never extract from chunks below this
        self._min_entity_count = min_entity_count or getattr(cfg, "density_min_entities", 2)
        self._nlp = None

    @property
    def name(self) -> str:
        return f"{self._inner.name}-filtered"

    # ── Public ──────────────────────────────────────────────────

    def extract(self, chunk: Chunk) -> list[Triple]:
        """Single-chunk extraction with density pre-check."""
        if not self._passes_density_check([chunk]):
            logger.debug("Chunk %s skipped (low entity density)", chunk.chunk_id)
            return []
        return self._inner.extract(chunk)

    def extract_batch(self, chunks: list[Chunk]) -> dict[str, list[Triple]]:
        """
        Filter chunks by density score, then delegate only the qualifying
        subset to the inner extractor's batch method.

        Steps:
          1. Score every chunk by entity density (fast — pure spaCy)
          2. Apply min_entity_count hard floor
          3. Keep top_fraction of remaining chunks by density score
          4. Pass filtered set to inner.extract_batch()
          5. Return merged result (skipped chunks → empty list)
        """
        if not chunks:
            return {}

        # Score all chunks
        scored = self._score_chunks(chunks)   # list of (chunk, density, entity_count)

        # Hard floor: drop chunks below minimum entity count
        above_floor = [(c, d, n) for c, d, n in scored if n >= self._min_entity_count]

        # Top-fraction cut: sort by density desc, keep top N%
        above_floor.sort(key=lambda x: x[1], reverse=True)
        cutoff = max(1, int(len(above_floor) * self._top_fraction))
        selected = [c for c, _, _ in above_floor[:cutoff]]
        skipped  = len(chunks) - len(selected)

        if skipped:
            logger.info(
                "Density filter: %d/%d chunks selected (top %.0f%%, min_entities=%d)",
                len(selected), len(chunks),
                self._top_fraction * 100, self._min_entity_count,
            )

        # Delegate to inner extractor
        if not selected:
            return {c.chunk_id: [] for c in chunks}

        inner_results = self._inner.extract_batch(selected)

        # Merge: unselected chunks get empty lists
        selected_ids = {c.chunk_id for c in selected}
        return {
            c.chunk_id: inner_results.get(c.chunk_id, []) if c.chunk_id in selected_ids else []
            for c in chunks
        }

    # ── Density scoring ────────────────────────────────────────

    def _score_chunks(
        self, chunks: list[Chunk]
    ) -> list[tuple[Chunk, float, int]]:
        """
        Returns list of (chunk, density_score, entity_count).
        density_score = entities per 100 tokens (approx).
        """
        nlp = self._get_nlp()
        results = []
        for chunk in chunks:
            doc = nlp(chunk.text[:5000])
            entity_count = len([e for e in doc.ents if len(e.text.strip()) > 1])
            token_count  = max(len(doc), 1)
            density      = (entity_count / token_count) * 100
            results.append((chunk, density, entity_count))
        return results

    def _passes_density_check(self, chunks: list[Chunk]) -> bool:
        """Quick single-chunk density check for extract()."""
        if not chunks:
            return False
        _, _, entity_count = self._score_chunks(chunks)[0]
        return entity_count >= self._min_entity_count

    # ── spaCy ──────────────────────────────────────────────────

    def _get_nlp(self):
        if self._nlp is None:
            import spacy  # type: ignore
            try:
                self._nlp = spacy.load("en_core_web_sm")
            except OSError:
                raise RuntimeError("Run: python -m spacy download en_core_web_sm")
        return self._nlp

# ── Registry + factory ─────────────────────────────────────────

_EXTRACTOR_REGISTRY: dict[str, type[RelationExtractor]] = {
    "rebel":         REBELExtractor,
    "llm":           LLMExtractor,
    # Density-filtered variants are constructed specially — see build_extractor()
}

# Names that trigger density-filter wrapping
_FILTERED_VARIANTS = {
    "rebel-filtered": "rebel",
    "llm-filtered":   "llm",
}


def build_extractor(name: Optional[str] = None) -> RelationExtractor:
    """
    Available values for GRAPH_EXTRACTOR:
    "rebel"          — REBEL local model, no API calls (default)
    "llm"            — Groq LLM, free-form predicates
    "rebel-filtered" — REBEL + entity density pre-filter (option 4)
    "llm-filtered"   — LLM   + entity density pre-filter (option 4)

    Explicit usage in code:
    extractor = build_extractor("rebel-filtered")

    # Or compose manually for full control:
    extractor = EntityDensityFilter(
        REBELExtractor(),
        top_fraction=0.25,
        min_entity_count=3,
    )
    """
    cfg = get_settings()
    extractor_name = (name or getattr(cfg, "graph_extractor", "rebel")).lower()

    # Density-filtered variant: build inner extractor then wrap it
    if extractor_name in _FILTERED_VARIANTS:
        inner_name = _FILTERED_VARIANTS[extractor_name]
        inner_cls  = _EXTRACTOR_REGISTRY[inner_name]
        inner      = inner_cls()
        logger.info(
            "Using relation extractor: %s (inner=%s, top_fraction=%.0f%%, min_entities=%d)",
            extractor_name, inner_name,
            getattr(cfg, "density_top_fraction", 0.30) * 100,
            getattr(cfg, "density_min_entities", 2),
        )
        return EntityDensityFilter(inner)

    # Plain extractor
    cls = _EXTRACTOR_REGISTRY.get(extractor_name)
    if cls is None:
        available = list(_EXTRACTOR_REGISTRY.keys()) + list(_FILTERED_VARIANTS.keys())
        raise ValueError(
            f"Unknown extractor '{extractor_name}'. "
            f"Available: {available}. "
            f"Set GRAPH_EXTRACTOR in .env to one of these."
        )

    logger.info("Using relation extractor: %s", extractor_name)
    return cls()
