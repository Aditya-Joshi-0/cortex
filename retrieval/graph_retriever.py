"""
Cortex RAG — Graph Retriever (Phase 3)

How multi-hop retrieval works
──────────────────────────────
Standard dense retrieval can answer: "What is attention?"
It cannot answer: "Who wrote the attention paper, and what did they later
build that addresses memory bottlenecks in inference?"

That question requires:
  Step 1: Find entity "Attention Is All You Need" in the graph
  Step 2: Follow "authored_by" edges → Vaswani, Shazeer, Parmar, …
  Step 3: Follow those author nodes' other edges →
          Shazeer: "introduced" → "Multi-Query Attention"
          Leviathan: "developed" → "Speculative Decoding"
  Step 4: Collect all chunk_ids linked to visited nodes
  Step 5: Fetch those chunks from Milvus → return to RRF pool

The BFS depth (default: 2 hops) is the key parameter. 1 hop = only
direct neighbours; 2 hops = neighbours of neighbours. 3+ hops tends to
explode in scope and include irrelevant context.

Entity matching
───────────────
The query "Who developed PagedAttention?" must match graph nodes like
"Paged Attention" or "PagedAttention". We do:
  1. Exact match (case-insensitive)
  2. Partial match (query entity substring of node label)
  3. spaCy NER on the query to extract candidate entity strings first
"""
from __future__ import annotations

import logging
from typing import Optional

from retrieval.dense import MilvusStore, RetrievedChunk
from retrieval.graph_builder import KnowledgeGraphBuilder

logger = logging.getLogger(__name__)


class GraphRetriever:
    """
    Retrieves chunks via knowledge graph traversal.

    Returns RetrievedChunk objects fetched from Milvus, so they carry
    the same structure as dense/BM25 results and can flow into RRF.
    """

    def __init__(
        self,
        graph_builder: Optional[KnowledgeGraphBuilder] = None,
        store: Optional[MilvusStore] = None,
        max_hops: int = 2,
    ) -> None:
        self._builder = graph_builder or KnowledgeGraphBuilder()
        self._store   = store or MilvusStore()
        self._max_hops = max_hops
        self._nlp = None

    # ── Public API ─────────────────────────────────────────────

    def search(self, query: str, top_k: int = 15) -> list[RetrievedChunk]:
        """
        Graph traversal retrieval for a given query.

        Pipeline:
          1. Extract named entities from query (spaCy)
          2. Anchor each entity to matching graph nodes (fuzzy match)
          3. BFS up to max_hops from anchors
          4. Collect chunk_ids from all visited nodes + traversed edges
          5. Fetch chunks from Milvus by chunk_id
          6. Score by graph centrality (number of graph links to query entities)
        """
        G = self._builder.graph
        if G.number_of_nodes() == 0:
            logger.debug("Graph is empty — skipping graph retrieval.")
            return []

        # 1. Extract query entities
        query_entities = self._extract_query_entities(query)
        if not query_entities:
            logger.debug("No named entities in query — skipping graph retrieval.")
            return []

        logger.debug("Graph query entities: %s", query_entities)

        # 2. Find anchor nodes
        anchor_nodes = self._find_anchor_nodes(query_entities, G)
        if not anchor_nodes:
            logger.debug("No anchor nodes found in graph.")
            return []

        logger.debug("Anchor nodes: %s", anchor_nodes)

        # 3 + 4. BFS traversal → collect chunk_ids
        chunk_id_scores: dict[str, float] = {}
        visited_nodes: set[str] = set()

        for anchor in anchor_nodes:
            self._bfs_collect(
                G, anchor, self._max_hops,
                chunk_id_scores, visited_nodes
            )

        if not chunk_id_scores:
            return []

        # 5. Sort chunk_ids by score and fetch from Milvus
        sorted_ids = sorted(
            chunk_id_scores, key=lambda cid: chunk_id_scores[cid], reverse=True
        )[:top_k]

        chunks = self._fetch_chunks_from_milvus(sorted_ids, chunk_id_scores)
        logger.info(
            "Graph retriever: %d anchors, %d nodes visited, %d chunks returned",
            len(anchor_nodes), len(visited_nodes), len(chunks)
        )
        return chunks

    # ── BFS traversal ─────────────────────────────────────────

    def _bfs_collect(
        self,
        G,
        start_node: str,
        max_hops: int,
        chunk_scores: dict[str, float],
        visited: set[str],
    ) -> None:
        """
        BFS from start_node up to max_hops.
        Scores chunks by hop distance: 1.0 at hop 0, 0.5 at hop 1, 0.25 at hop 2.
        """
        queue: list[tuple[str, int]] = [(start_node, 0)]
        local_visited: set[str] = set()

        while queue:
            node, depth = queue.pop(0)
            if node in local_visited or depth > max_hops:
                continue
            local_visited.add(node)
            visited.add(node)

            # Score = 1 / 2^depth (1.0 at anchor, 0.5 one hop away, etc.)
            hop_score = 1.0 / (2 ** depth)

            # Collect chunk_ids from this node
            node_data = G.nodes[node]
            for cid in node_data.get("chunk_ids", []):
                chunk_scores[cid] = max(chunk_scores.get(cid, 0.0), hop_score)

            # Collect chunk_ids from edges (relations)
            for neighbour in G.neighbors(node):
                edge_data = G[node][neighbour]
                for cid in edge_data.get("chunk_ids", []):
                    chunk_scores[cid] = max(chunk_scores.get(cid, 0.0), hop_score * 0.8)

                if depth < max_hops:
                    queue.append((neighbour, depth + 1))

    # ── Entity extraction ──────────────────────────────────────

    def _extract_query_entities(self, query: str) -> list[str]:
        """
        Extract named entities from the query using spaCy NER.
        Falls back to noun chunks if NER finds nothing.
        """
        try:
            nlp = self._get_nlp()
            doc = nlp(query)
            entities = [ent.text.strip().title() for ent in doc.ents if len(ent.text.strip()) > 1]
            if not entities:
                # Fallback: try noun chunks (catches "attention mechanism", etc.)
                entities = [
                    chunk.text.strip().title()
                    for chunk in doc.noun_chunks
                    if len(chunk.text.strip()) > 3
                ]
            return entities
        except Exception as exc:
            logger.debug("Entity extraction failed: %s", exc)
            return []

    # ── Node matching ─────────────────────────────────────────

    @staticmethod
    def _find_anchor_nodes(query_entities: list[str], G) -> list[str]:
        """
        Find graph nodes that match query entities.
        Priority: exact match → partial match.
        """
        all_nodes = list(G.nodes())
        lower_nodes = {n.lower(): n for n in all_nodes}

        anchors: list[str] = []
        for qe in query_entities:
            qe_lower = qe.lower()

            # Exact match (case-insensitive)
            if qe_lower in lower_nodes:
                anchors.append(lower_nodes[qe_lower])
                continue

            # Partial match: query entity is substring of a node label
            for node_lower, node in lower_nodes.items():
                if qe_lower in node_lower or node_lower in qe_lower:
                    if node not in anchors:
                        anchors.append(node)

        return anchors[:10]   # cap to avoid explosion on generic queries

    # ── Milvus fetch ──────────────────────────────────────────

    def _fetch_chunks_from_milvus(
        self,
        chunk_ids: list[str],
        scores: dict[str, float],
    ) -> list[RetrievedChunk]:
        """
        Fetch specific chunks from Milvus by chunk_id.
        Tags each chunk with retriever="graph".
        """
        if not chunk_ids:
            return []

        try:
            # Milvus IN query
            id_list = '", "'.join(chunk_ids)
            expr = f'chunk_id in ["{id_list}"]'

            coll = self._store._ensure_collection()
            results = coll.query(
                expr=expr,
                output_fields=["chunk_id", "doc_id", "source", "title",
                                "text", "parent_text", "chunk_index"],
                limit=len(chunk_ids),
            )

            chunks: list[RetrievedChunk] = []
            for row in results:
                cid = row["chunk_id"]
                chunks.append(RetrievedChunk(
                    chunk_id=cid,
                    doc_id=row["doc_id"],
                    source=row["source"],
                    title=row["title"],
                    text=row["text"],
                    parent_text=row["parent_text"],
                    chunk_index=row["chunk_index"],
                    score=scores.get(cid, 0.1),
                    retriever="graph",
                ))
            return sorted(chunks, key=lambda c: c.score, reverse=True)

        except Exception as exc:
            logger.warning("Milvus fetch for graph chunks failed: %s", exc)
            return []

    # ── spaCy ─────────────────────────────────────────────────

    def _get_nlp(self):
        if self._nlp is None:
            import spacy  # type: ignore
            try:
                self._nlp = spacy.load("en_core_web_sm")
            except OSError:
                raise RuntimeError(
                    "Download spaCy model: python -m spacy download en_core_web_sm"
                )
        return self._nlp
