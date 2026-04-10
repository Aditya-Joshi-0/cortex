---
title: Cortex
sdk: docker
emoji: 📚
colorFrom: blue
colorTo: purple
---
# Cortex RAG — Phase 1

> Production-grade Retrieval-Augmented Generation with dense vector search,
> semantic chunking, parent-child hierarchy, and streaming generation.

## Architecture (Phase 1)

```
Documents (PDF/HTML/TXT)
    │
    ▼
DocumentLoader          # pdfplumber / bs4 / plain text
    │
    ▼
SemanticChunker         # sentence-level cosine similarity boundaries
    │  ├─ child chunk  (~256 tokens)  → embedded & stored in Milvus
    │  └─ parent chunk (~1024 tokens) → stored alongside; returned to LLM
    ▼
Embedder                # BAAI/bge-small-en-v1.5, L2-normalised
    │
    ▼
MilvusStore             # IVF_FLAT index, cosine metric
    │
    │   Query
    │     │
    │     ▼
    │   Dense search → top-15 chunks
    │     │
    │     ▼
    │   LLM Generator (Groq / Llama 3.3-70B)
    │     │ streaming SSE
    │     ▼
    │   Streamlit UI (tabbed: Ask / Ingest / System)
```

## Quick start

### 1. Clone and install

```bash
git clone <repo>
cd cortex
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m nltk.downloader punkt
python -m spacy download en_core_web_sm
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env — set GROQ_API_KEY at minimum
```

Get a free Groq API key at https://console.groq.com

### 3. Start Milvus

```bash
docker-compose up -d
# Wait ~30s for Milvus to be healthy
docker-compose ps   # all three services should show "healthy"
```

### 4. Ingest documents

```bash
mkdir -p data/documents
# Copy PDFs / HTML / TXT files into data/documents/

python -m ingestion.pipeline data/documents
```

Or use the CLI:
```bash
python ingestion/pipeline.py data/documents
python ingestion/pipeline.py data/documents/paper.pdf
```

### 5. Start the API

```bash
uvicorn api.main:app --reload --port 8000
```

### 6. Start the UI

```bash
streamlit run ui/app.py
```

Open http://localhost:8501 in your browser.

---

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET  | `/health` | Component health check |
| POST | `/ingest` | Trigger ingestion pipeline |
| POST | `/query` | Blocking query (full JSON) |
| POST | `/query/stream` | Streaming query (SSE) |

### Example — blocking query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is attention in transformers?", "top_k": 5}'
```

### Example — streaming query

```bash
curl -X POST http://localhost:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain PagedAttention", "stream": true}'
```

---

## Key design decisions

### Semantic chunking
Fixed-size chunking (e.g. 1000 chars with 200 overlap) splits mid-sentence
and mid-concept. Semantic chunking detects topic boundaries using cosine
similarity between consecutive sentence embeddings, producing chunks that
align with natural concept transitions. Combined with a fallback on token
count (child_max = 256 tokens), chunks are both semantically coherent and
bounded in size.

### Parent-child hierarchy
The child chunk (≈256 tokens) is what gets embedded and indexed — small,
precise, high-relevance. When a child chunk is retrieved, its parent chunk
(≈1024 tokens, centred on the child) is what goes into the LLM context.
This separates the **retrieval granularity** from the **generation context
width**, giving you the precision of small chunks with the coherence of
large ones.

### BGE query prefix
`BAAI/bge-small-en-v1.5` is trained to expect a task-specific prefix on
query strings for retrieval tasks:
`"Represent this sentence for searching relevant passages: <query>"`
Documents are embedded as-is. Skipping this prefix typically costs 3-5
points on retrieval benchmarks.

---

## Phase roadmap

| Phase | Status | What's added |
|-------|--------|--------------|
| 1 | ✅ Done | Dense RAG, semantic chunking, parent-child, streaming UI |
| 2 | 🔜 Next | BM25 sparse, query router, RRF fusion, cross-encoder reranking |
| 3 | Planned | GraphRAG (spaCy NER + NetworkX), CRAG gate, web fallback |
| 4 | Planned | RAGAS eval harness, Redis cache, evaluation dashboard |