"""
Cortex RAG — Knowledge Graph Builder (Phase 3)

What this does
──────────────
During ingestion, every chunk is processed to extract:
  1. Named entities  (spaCy NER: PERSON, ORG, WORK_OF_ART, PRODUCT, …)
  2. Relations       (few-shot LLM: subject → predicate → object triples)

These are assembled into a NetworkX undirected graph where:
  - Nodes  = entities (label + type + first-seen source)
  - Edges  = relations (predicate label + list of source chunk_ids)

Each node also carries a list of chunk_ids it appeared in, so the
graph retriever can map entity → chunks without an extra lookup.

The graph is persisted as a JSON file (graphs are small — a 100-doc
corpus typically has <10k nodes). On reload the full graph is
reconstructed in seconds from the JSON.

──────────────
(Phase 3, refactored)

The builder is now responsible ONLY for:
  - spaCy NER (entities are always extracted the same way)
  - Assembling triples into a NetworkX graph
  - Persisting / loading the graph

Relation extraction is delegated to a RelationExtractor strategy:
  - REBELExtractor  (default) — local model, no API calls
  - LLMExtractor              — Groq, free-form predicates

Switch via .env:
  GRAPH_EXTRACTOR=rebel    # default, recommended
  GRAPH_EXTRACTOR=llm      # original method

Or pass explicitly:
  builder = KnowledgeGraphBuilder(extractor=LLMExtractor())

"""
from __future__ import annotations

import json
import logging

from pathlib import Path
from typing import Optional

import networkx as nx

from ingestion.chunker import Chunk
from retrieval.relation_extractors import (
    RelationExtractor,
    Triple,
    build_extractor,
)

logger = logging.getLogger(__name__)

_DEFAULT_GRAPH_PATH = Path("data/knowledge_graph.json")

# spaCy entity types we care about for RAG
_ENTITY_TYPES = {
    "PERSON", "ORG", "GPE", "PRODUCT", "WORK_OF_ART",
    "EVENT", "LAW", "NORP", "FAC", "LOC",
}

class KnowledgeGraphBuilder:
    """
    Builds and maintains the knowledge graph.

    Usage (at ingestion time):
        # REBEL (default — no API calls)
        builder = KnowledgeGraphBuilder()
        builder.process_chunks(chunks)

        # LLM method (original)
        from retrieval.relation_extractors import LLMExtractor
        builder = KnowledgeGraphBuilder(extractor=LLMExtractor())
        builder.process_chunks(chunks)

    Usage (at query time):
        builder = KnowledgeGraphBuilder()
        G = builder.graph    # loaded from disk automatically
    """

    def __init__(
        self,
        graph_path: str | Path = _DEFAULT_GRAPH_PATH,
        extractor: Optional[RelationExtractor] = None,
    ) -> None:
        self._path = Path(graph_path)
        self._graph: nx.Graph = nx.Graph()
        # If no extractor is injected, build_extractor() reads GRAPH_EXTRACTOR from .env
        self._extractor: RelationExtractor = extractor or build_extractor()
        self._nlp = None
        self._load_if_exists()
        logger.info(
            "KnowledgeGraphBuilder ready (extractor=%s)", self._extractor.name
        )

    # ── Public API ─────────────────────────────────────────────

    @property
    def graph(self) -> nx.Graph:
        return self._graph

    @property
    def extractor_name(self) -> str:
        return self._extractor.name

    def process_chunks(self, chunks: list[Chunk]) -> dict:
        """
        Extract entities and relations from chunks; update and save graph.
        Uses the configured extractor's extract_batch() for efficiency.
        Returns stats dict.
        """
        if not chunks:
            return {"chunks": 0, "entities": 0, "triples": 0, "errors": 0}

        stats = {"chunks": len(chunks), "entities": 0, "triples": 0, "errors": 0}

        # ── Batch relation extraction ──────────────────────────
        # REBEL processes all chunks in one forward pass.
        # LLM falls back to sequential (one API call per chunk).
        try:
            triple_map = self._extractor.extract_batch(chunks)
        except Exception as exc:
            logger.error("Batch extraction failed, falling back to sequential: %s", exc)
            triple_map = {}
            for chunk in chunks:
                try:
                    triple_map[chunk.chunk_id] = self._extractor.extract(chunk)
                except Exception as e:
                    logger.warning("Extraction failed for %s: %s", chunk.chunk_id, e)
                    triple_map[chunk.chunk_id] = []
                    stats["errors"] += 1

        # ── Entity extraction + graph update ───────────────────
        for chunk in chunks:
            try:
                entities = self._extract_entities(chunk.text)
                triples  = triple_map.get(chunk.chunk_id, [])

                self._add_entities_to_graph(entities, chunk)
                self._add_triples_to_graph(triples)

                stats["entities"] += len(entities)
                stats["triples"]  += len(triples)

            except Exception as exc:
                logger.warning("Graph update failed for chunk %s: %s", chunk.chunk_id, exc)
                stats["errors"] += 1

        self.save()
        logger.info(
            "Graph updated via %s: +%d entities, +%d triples (nodes=%d, edges=%d)",
            self._extractor.name,
            stats["entities"], stats["triples"],
            self._graph.number_of_nodes(), self._graph.number_of_edges(),
        )
        return stats

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = nx.node_link_data(self._graph)
        with open(self._path, "w") as fh:
            json.dump(data, fh, indent=2)
        logger.debug("Graph saved to %s", self._path)

    def stats(self) -> dict:
        return {
            "nodes":      self._graph.number_of_nodes(),
            "edges":      self._graph.number_of_edges(),
            "extractor":  self._extractor.name,
            "graph_path": str(self._path),
        }

    # ── Entity extraction (always spaCy — same for both methods) ─

    def _extract_entities(self, text: str) -> list[tuple[str, str]]:
        nlp = self._get_nlp()
        doc = nlp(text[:10_000])

        seen: set[str] = set()
        entities: list[tuple[str, str]] = []
        for ent in doc.ents:
            if ent.label_ not in _ENTITY_TYPES:
                continue
            normalised = ent.text.strip().title()
            if normalised in seen or len(normalised) < 2:
                continue
            seen.add(normalised)
            entities.append((normalised, ent.label_))
        return entities

    # ── Graph construction (shared by both methods) ────────────

    def _add_entities_to_graph(
        self, entities: list[tuple[str, str]], chunk: Chunk
    ) -> None:
        for label, etype in entities:
            if self._graph.has_node(label):
                existing = self._graph.nodes[label].get("chunk_ids", [])
                if chunk.chunk_id not in existing:
                    existing.append(chunk.chunk_id)
                self._graph.nodes[label]["chunk_ids"] = existing
            else:
                self._graph.add_node(
                    label,
                    entity_type=etype,
                    chunk_ids=[chunk.chunk_id],
                    source=chunk.source,
                )

    def _add_triples_to_graph(self, triples: list[Triple]) -> None:
        for triple in triples:
            for node in (triple.subject, triple.object):
                if not self._graph.has_node(node):
                    self._graph.add_node(
                        node,
                        entity_type="UNKNOWN",
                        chunk_ids=[],
                        source=triple.source,
                        extractor=triple.extractor,
                    )

            if self._graph.has_edge(triple.subject, triple.object):
                edge = self._graph[triple.subject][triple.object]
                predicates = edge.get("predicates", [])
                chunk_ids  = edge.get("chunk_ids", [])
                if triple.predicate not in predicates:
                    predicates.append(triple.predicate)
                if triple.chunk_id not in chunk_ids:
                    chunk_ids.append(triple.chunk_id)
                edge["predicates"] = predicates
                edge["chunk_ids"]  = chunk_ids
            else:
                self._graph.add_edge(
                    triple.subject, triple.object,
                    predicates=[triple.predicate],
                    chunk_ids=[triple.chunk_id],
                    source=triple.source,
                    extractor=triple.extractor,
                )

    # ── Persistence ───────────────────────────────────────────

    def _load_if_exists(self) -> None:
        if not self._path.exists():
            return
        try:
            with open(self._path) as fh:
                data = json.load(fh)
            self._graph = nx.node_link_graph(data)
            logger.info(
                "Knowledge graph loaded: %d nodes, %d edges",
                self._graph.number_of_nodes(),
                self._graph.number_of_edges(),
            )
        except Exception as exc:
            logger.warning("Failed to load graph (%s) — starting fresh.", exc)

    # ── spaCy ─────────────────────────────────────────────────

    def _get_nlp(self):
        if self._nlp is None:
            try:
                import spacy  # type: ignore
            except ImportError as exc:
                raise RuntimeError("Install spacy: pip install spacy") from exc
            try:
                self._nlp = spacy.load("en_core_web_sm")
            except OSError:
                raise RuntimeError(
                    "Run: python -m spacy download en_core_web_sm"
                )
        return self._nlp
