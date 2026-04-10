---
title: Cortex RAG
sdk: docker
emoji: 🧠
colorFrom: blue
colorTo: purple
---

# Cortex RAG — Next-Gen Retrieval-Augmented Generation

<div align="center">

**Production-grade RAG system with dense retrieval, semantic chunking, knowledge graph integration, CRAG gating, and multi-provider LLM support.**

![Python](https://img.shields.io/badge/Python-3.10+-3776ab?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

</div>

---

## 🎯 Overview

**Cortex** is a production-ready Retrieval-Augmented Generation (RAG) framework that combines:

- **Dense Vector Search** — Fast, accurate document retrieval using BAAI embeddings (384-dim)
- **Semantic Chunking** — Intelligent split boundaries based on sentence-level cosine similarity
- **Parent-Child Chunks** — 256-token child chunks for precision, 1024-token parents for context
- **Multi-Strategy Retrieval** — Dense search, BM25 hybrid, knowledge graph traversal
- **CRAG Gating** — Automatic relevance assessment with fallback to web search
- **Multi-Provider LLM** — Support for Groq, OpenAI, NVIDIA NIM, and custom endpoints
- **Streaming Responses** — Real-time SSE-based answer generation with inline citations
- **Knowledge Graphs** — Automatic relation extraction and entity-based retrieval
- **Caching Layer** — Redis integration for query result caching
- **Evaluation Framework** — RAGAS-based RAG evaluation metrics

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Document Ingestion                            │
├─────────────────────────────────────────────────────────────────┤
│  PDF/HTML/TXT → DocumentLoader → SemanticChunker                 │
│                                       ↓                          │
│                   Child (~256 tokens) + Parent (~1024 tokens)    │
├─────────────────────────────────────────────────────────────────┤
│                        Embedding Layer                            │
├─────────────────────────────────────────────────────────────────┤
│  BAAI/bge-small-en-v1.5 (384-dim, L2-normalized)                │
│  → Milvus Store (IVF_FLAT, COSINE metric)                        │
│  → BM25 Index (keyword search)                                   │
│  → Knowledge Graph (entities, relations, triples)                │
├─────────────────────────────────────────────────────────────────┤
│                       Query Processing                            │
├─────────────────────────────────────────────────────────────────┤
│  Dense Search (top-15) → Reranking → CRAG Gate                   │
│         ↓                                    ↓                    │
│  High Confidence?                    Low Confidence?             │
│         ↓                                    ↓                    │
│    Use KnowledgeBase               ⚠️ Web Search (Tavily)        │
├─────────────────────────────────────────────────────────────────┤
│                    LLM Generation (Streaming)                     │
├─────────────────────────────────────────────────────────────────┤
│  Groq Llama 3.3-70B / OpenAI GPT-4o / NVIDIA NIM / Custom        │
│  Process context → Generate answer → Extract citations           │
│  Stream via SSE → Client receives real-time response             │
├─────────────────────────────────────────────────────────────────┤
│                  Frontend Interfaces                              │
├─────────────────────────────────────────────────────────────────┤
│  Streamlit UI (Ask/Ingest/System) | REST API (FastAPI)           │
└─────────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

| Feature | Details |
|---------|---------|
| 🔍 **Dense Retrieval** | Sub-50ms semantic search via Milvus with 384-dim embeddings |
| 📚 **Smart Chunking** | Semantic splits + parent-child hierarchy for precision + context |
| 🧬 **Knowledge Graphs** | Automatic relation extraction (REBEL or LLM-based) |
| 🚨 **CRAG Gating** | Relevance assessment with web search fallback |
| 🔗 **Multi-Strategy** | Dense + BM25 keyword + graph traversal combined |
| 💾 **Redis Cache** | Query result caching with configurable TTL |
| 🌐 **Multi-Provider LLM** | Groq, OpenAI, NVIDIA NIM, Ollama, custom OpenAI-compatible |
| 📊 **Evaluation** | RAGAS metrics for answer relevance, faithfulness, context precision |
| 🎨 **Streaming UI** | Real-time responses with inline citations and source cards |
| 🐳 **Docker Ready** | Full Docker Compose setup with Milvus, Redis, API, UI |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Docker & Docker Compose (optional, for containerized setup)
- GROQ API key (default LLM provider)

### 1. Clone & Setup

```bash
# Clone repository
git clone <repo-url>
cd cortex

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration

Create `.env` file in project root:

```bash
# LLM Providers
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_TEMPERATURE=0.1

# Optional: Other LLM providers
OPENAI_API_KEY=your_openai_key
MISTRAL_API_KEY=your_mistral_key
NVIDIA_API_KEY=your_nvidia_key

# Embedding & Storage
EMBED_MODEL_NAME=BAAI/bge-small-en-v1.5
EMBED_DEVICE=cpu  # "cuda" if GPU available

# Milvus Vector Store
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_COLLECTION=cortex_chunks
MILVUS_INDEX_TYPE=IVF_FLAT

# Redis Cache (optional)
REDIS_URL=redis://localhost:6379

# Retrieval
RETRIEVAL_TOP_K=15
FINAL_TOP_K=5

# CRAG (Consistency-based Retrieval Augmented Generation)
CRAG_ENABLED=true
CRAG_RELEVANCE_THRESHOLD=0.5

# Knowledge Graph
GRAPH_ENABLED=true
GRAPH_EXTRACTOR=llm-filtered  # "rebel", "llm", "rebel-filtered", "llm-filtered"
GRAPH_MAX_HOPS=2

# API
API_HOST=0.0.0.0
API_PORT=8000
```

### 3. Start Services

**Option A: Docker Compose (Recommended)**

```bash
docker-compose up -d
# API: http://localhost:8000
# Streamlit UI: http://localhost:8501
# Milvus: http://localhost:19530
```

**Option B: Local Setup**

Make sure Milvus is running:

```bash
# Using Milvus Docker (if not using compose)
docker run -d -p 19530:19530 -p 9091:9091 milvusdb/milvus:latest

# Start API
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# In another terminal, start UI
streamlit run ui/app.py
```

### 4. Ingest Documents

**Via Streamlit UI:**
- Open http://localhost:8501
- Go to "📥 Ingest" tab
- Upload PDF/HTML/TXT or provide directory path

**Via REST API:**

```bash
curl -X POST "http://localhost:8000/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "directory",
    "path": "/path/to/documents"
  }'
```

### 5. Ask Questions

**Via Streamlit UI:**
- Go to "🔍 Ask" tab
- Type your question
- Watch streaming response with citations

**Via REST API:**

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "provider": "groq",
    "top_k": 5
  }' | jq .
```

**Streaming Response:**

```bash
curl -X POST "http://localhost:8000/query/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Your question here",
    "provider": "groq"
  }'
```

---

## 📡 REST API Endpoints

### Health & Status

```http
GET /health
```

Returns system health, Milvus status, collection stats.

```json
{
  "status": "healthy",
  "milvus": {
    "connected": true,
    "collection_count": 2500,
    "index_type": "IVF_FLAT"
  }
}
```

### Document Ingestion

```http
POST /ingest
Content-Type: application/json

{
  "mode": "directory|file|upload",
  "path": "/path/to/documents",
  "chunk_size": 256,
  "overlap": 32
}
```

### Query (Blocking)

```http
POST /query
Content-Type: application/json

{
  "query": "Your question",
  "provider": "groq",
  "model": "llama-3.3-70b-versatile",
  "top_k": 5,
  "crag": true,
  "graph": true
}
```

**Response:**

```json
{
  "answer": "Answer text with citations [1][2]...",
  "chunks": [
    {
      "id": "chunk_001",
      "text": "...",
      "score": 0.87,
      "source": "document_name.pdf"
    }
  ],
  "citations": [1, 2],
  "latency_ms": 1245
}
```

### Query (Streaming)

```http
POST /query/stream
Content-Type: application/json

{
  "query": "Your question",
  "provider": "groq"
}
```

**Response:** Server-Sent Events (SSE) stream

```
data: {"type": "start"}
data: {"type": "chunk", "content": "Answer "}
data: {"type": "chunk", "content": "is "}
data: {"type": "chunk", "content": "streaming..."}
data: {"type": "citations", "citations": [1, 2]}
data: {"type": "end"}
```

### Model Information

```http
GET /providers
```

Lists all available LLM providers and models.

---

## 🛠️ Configuration Guide

### Retrieval Configuration

```env
# Chunk sizes (tokens)
CHUNK_SIZE_TOKENS=256                    # Child chunk size
PARENT_CHUNK_SIZE_TOKENS=1024            # Parent chunk size
SEMANTIC_SIMILARITY_THRESHOLD=0.82       # Split boundary threshold
CHUNK_OVERLAP_TOKENS=32                  # Overlap padding

# Retrieval settings
RETRIEVAL_TOP_K=15                       # Candidates before reranking
FINAL_TOP_K=5                            # Chunks sent to LLM
```

### Embedding Configuration

```env
EMBED_MODEL_NAME=BAAI/bge-small-en-v1.5  # Model identifier
EMBED_DIM=384                             # Output dimension
EMBED_BATCH_SIZE=64                       # Batch size for processing
EMBED_DEVICE=cpu                          # cpu or cuda
```

### Milvus Configuration

```env
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_COLLECTION=cortex_chunks
MILVUS_INDEX_TYPE=IVF_FLAT                # or HNSW for larger corpora
MILVUS_METRIC_TYPE=COSINE                 # Vector similarity metric
MILVUS_NLIST=128                          # clustering parameter for IVF
MILVUS_NPROBE=16                          # search parameter
```

### LLM Provider Configuration

**Groq (Default)**
```env
GROQ_API_KEY=your_key
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_TEMPERATURE=0.1
GROQ_MAX_TOKENS=1024
GROQ_TIMEOUT=30
```

**OpenAI**
```env
OPENAI_API_KEY=your_key
```

**NVIDIA NIM**
```env
NVIDIA_API_KEY=your_key
```

**Custom/Ollama**
```env
CUSTOM_BASE_URL=http://localhost:11434/v1
CUSTOM_API_KEY=your_key
```

### CRAG (Consistency-based Retrieval Augmented Generation)

```env
CRAG_ENABLED=true
CRAG_RELEVANCE_THRESHOLD=0.5             # Grade boundary
TAVILY_API_KEY=your_tavily_key           # For web search fallback
```

The CRAG gate automatically assesses retrieval quality:
- **High confidence** (score ≥ threshold) → Use knowledge base
- **Low confidence** (score < threshold) → Augment with web search

### Knowledge Graph

```env
GRAPH_ENABLED=true
GRAPH_EXTRACTOR=llm-filtered             # rebel|llm|rebel-filtered|llm-filtered
GRAPH_MAX_HOPS=2                          # Traversal depth
GRAPH_PATH=/data/storage/knowledge_graph.json

# Density filtering (for "filtered" extractors)
DENSITY_TOP_FRACTION=0.30                 # Process top 30% entity-dense chunks
DENSITY_MIN_ENTITIES=2                    # Minimum entities per chunk
```

### Caching

```env
REDIS_URL=redis://localhost:6379
CACHE_TTL_SECONDS=3600                    # 1 hour
```

### Evaluation

```env
EVAL_DB_PATH=/data/storage/eval.db
```

---

## 📁 Project Structure

```
cortex/
├── api/                          # FastAPI REST endpoints
│   ├── main.py                   # App initialization, endpoints
│   └── schemas.py                # Request/response Pydantic models
│
├── ingestion/                    # Document processing pipeline
│   ├── pipeline.py               # Orchestration
│   ├── document_loader.py        # PDF/HTML/TXT parsing
│   ├── chunker.py                # Semantic chunking
│   └── __init__.py
│
├── retrieval/                    # Multi-strategy retrieval
│   ├── orchestrator.py           # Coordinate retrieval strategies
│   ├── dense.py                  # Milvus vector search
│   ├── bm25.py                   # Keyword search index
│   ├── embedder.py               # HuggingFace embedding model
│   ├── router.py                 # Query routing logic
│   ├── fusion.py                 # Result fusion & reranking
│   ├── graph_builder.py          # Build knowledge graphs
│   ├── graph_retriever.py        # Entity-based retrieval
│   ├── relation_extractors.py    # REBEL + LLM extractors
│   ├── cache.py                  # Redis caching wrapper
│   └── __init__.py
│
├── generation/                   # LLM generation & CRAG
│   ├── generator.py              # Multi-provider LLM wrapper
│   ├── crag.py                   # CRAG gate logic
│   └── __init__.py
│
├── evaluation/                   # RAG evaluation metrics
│   ├── ragas_eval.py             # RAGAS evaluator
│   ├── store.py                  # Evaluation database
│   └── __init__.py
│
├── ui/                           # Streamlit frontend
│   ├── app.py                    # Main UI
│   └── static/                   # (Optional) HTML/CSS/JS
│
├── data/                         # Data storage
│   ├── documents/                # Input documents
│   ├── storage/                  # Persistent storage
│   │   ├── knowledge_graph.json
│   │   ├── bm25_index.pkl
│   │   └── uploads/
│   └── synthetic_knowledge_items.txt
│
├── config.py                     # Configuration & settings
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Docker image build
├── docker-compose.yml            # Multi-container orchestration
├── test.py                       # Test suite
└── README.md                     # This file
```

---

## 🐳 Docker & Deployment

### Docker Compose Quick Deploy

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

**Services:**
- `milvus` — Vector database (port 19530)
- `redis` — Caching layer (port 6379)
- `api` — FastAPI backend (port 8000)
- `ui` — Streamlit frontend (port 8501)

### Environment Variables in Compose

Edit `docker-compose.yml` to customize:

```yaml
services:
  api:
    environment:
      - GROQ_API_KEY=${GROQ_API_KEY}
      - GROQ_MODEL=llama-3.3-70b-versatile
      - MILVUS_HOST=milvus
      - REDIS_URL=redis://redis:6379
      - GRAPH_EXTRACTOR=llm-filtered
```

### Production Deployment

For production, consider:

1. **Use HNSW index** instead of IVF_FLAT for better recall:
   ```env
   MILVUS_INDEX_TYPE=HNSW
   ```

2. **Enable caching** for frequently asked questions:
   ```env
   REDIS_URL=redis://redis-prod:6379
   ```

3. **Use stronger embedding model** for higher quality:
   ```env
   EMBED_MODEL_NAME=BAAI/bge-base-en-v1.5  # 768-dim, better quality
   ```

4. **Configure CRAG** for reliability:
   ```env
   CRAG_ENABLED=true
   CRAG_RELEVANCE_THRESHOLD=0.6
   TAVILY_API_KEY=your_key
   ```

---

## 🔄 Workflow Examples

### Example 1: Legal Document Q&A

```bash
# 1. Ingest legal documents
curl -X POST "http://localhost:8000/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "directory",
    "path": "/data/legal_documents"
  }'

# 2. Query with graph enabled for relation extraction
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the penalties for breach of contract?",
    "provider": "groq",
    "graph": true,
    "crag": true
  }'
```

### Example 2: Research Paper Analysis

```bash
# Ingest PDF papers
python -c "
from ingestion.pipeline import IngestionPipeline
from retrieval.embedder import Embedder
from retrieval.dense import MilvusStore

embedder = Embedder()
store = MilvusStore(embedder=embedder)
pipeline = IngestionPipeline(embedder=embedder, store=store, bm25=None)

pipeline.ingest('/data/papers', mode='pdf')
"

# Query for specific findings
curl -X POST "http://localhost:8000/query/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the key findings about transformer performance?",
    "model": "gpt-4o"
  }'
```

### Example 3: Customer Support Bot

```bash
# 1. Ingest FAQ and documentation
# 2. Set up CRAG with relevant threshold
# 3. Route low-confidence queries to web search

CRAG_RELEVANCE_THRESHOLD=0.6
TAVILY_API_KEY=your_key
```

---

## 📊 Advanced Features

### Knowledge Graph Extraction

Three modes available:

| Mode | Backend | Speed | Quality | Cost |
|------|---------|-------|---------|------|
| `rebel` | Local REBEL model | Fast | Good | Free |
| `llm` | LLM (Groq/OpenAI) | Slower | Excellent | $$ |
| `rebel-filtered` | REBEL + entity filtering | Fast | Good | Free |
| `llm-filtered` | LLM + entity filtering | Slower | Excellent | $$ |

Switch via config:
```env
GRAPH_EXTRACTOR=llm-filtered
```

### CRAG (Consistency-based RAG)

Automatically:
1. Evaluates retrieval confidence
2. Assigns relevance grade (Correct/Partially-Correct/Missing)
3. Supplements low-confidence with web search via Tavily

```python
from generation.crag import CRAGGate

crag = CRAGGate()
response = crag.evaluate(query, context, answer)
# Returns: grade, supplemental_docs
```

### Evaluation & Metrics

RAGAS-based evaluation:

```python
from evaluation.ragas_eval import RAGASEvaluator
from evaluation.store import EvalStore

evaluator = RAGASEvaluator(store=EvalStore())
metrics = evaluator.evaluate(query, context, answer)
# Returns: answer_relevance, faithfulness, context_precision
```

### Caching Strategy

```python
from retrieval.cache import CachedRetriever

retriever = CachedRetriever(base_retriever)
# First call: 1000ms (database query)
# Second call: 5ms (Redis cache hit, TTL: 1 hour)
results = retriever.retrieve("machine learning basics")
```

---

## ⚙️ Performance Tuning

### For Speed

```env
# Smaller embedding model
EMBED_MODEL_NAME=BAAI/bge-small-en-v1.5

# Smaller chunks
CHUNK_SIZE_TOKENS=128
PARENT_CHUNK_SIZE_TOKENS=512

# Faster index
MILVUS_INDEX_TYPE=IVF_FLAT
MILVUS_NPROBE=8  # Lower = faster

# Enable cache
REDIS_URL=redis://localhost:6379

# Fewer LLM tokens
GROQ_MAX_TOKENS=512
```

### For Quality

```env
# Larger embedding model
EMBED_MODEL_NAME=BAAI/bge-base-en-v1.5

# Optimal chunks
CHUNK_SIZE_TOKENS=512
PARENT_CHUNK_SIZE_TOKENS=2048

# More precise index
MILVUS_INDEX_TYPE=HNSW
MILVUS_NPROBE=32

# Better LLM
GROQ_MODEL=llama-3.3-70b-versatile

# Enable CRAG
CRAG_ENABLED=true
```

---

## 🐛 Troubleshooting

### Milvus Connection Failed

```bash
# Check if Milvus is running
curl http://localhost:19530/healthz

# Restart Milvus
docker-compose restart milvus

# Verify in settings
python -c "from config import get_settings; print(get_settings().milvus_host)"
```

### Low Retrieval Quality

1. **Check chunk quality:**
   ```python
   from ingestion.chunker import SemanticChunker
   chunker = SemanticChunker()
   chunks = chunker.chunk("your document text")
   print([c.text for c in chunks[:3]])
   ```

2. **Verify embeddings:**
   ```python
   from retrieval.embedder import Embedder
   embedder = Embedder()
   emb = embedder.embed("test query")
   print(f"Embedding dim: {len(emb)}, sample: {emb[:5]}")
   ```

3. **Enable CRAG** for automatic augmentation:
   ```env
   CRAG_ENABLED=true
   ```

### Slow Response Times

1. Check cache hit rate
2. Reduce `MILVUS_NPROBE`
3. Use streaming endpoint (`/query/stream`)
4. Enable Redis caching

### Out of Memory

```env
# Reduce batch sizes
EMBED_BATCH_SIZE=16

# Reduce chunk sizes
CHUNK_SIZE_TOKENS=128

# Switch to CPU if using GPU
EMBED_DEVICE=cpu
```

---

## 📈 Monitoring & Evaluation

### Health Check

```bash
curl http://localhost:8000/health | jq .
```

### Collection Statistics

```python
from retrieval.dense import MilvusStore
from retrieval.embedder import Embedder

store = MilvusStore(embedder=Embedder())
stats = store.get_stats()
print(f"Documents: {stats['collection_count']}")
```

### Query Evaluation

```python
from evaluation.ragas_eval import RAGASEvaluator
from evaluation.store import EvalStore

evaluator = RAGASEvaluator(store=EvalStore(db_path="/data/storage/eval.db"))
metrics = evaluator.evaluate(query, context, answer)
print(f"Answer Relevance: {metrics['answer_relevance']:.2f}")
print(f"Faithfulness: {metrics['faithfulness']:.2f}")
print(f"Context Precision: {metrics['context_precision']:.2f}")
```

---

## 🤝 Contributing

Contributions welcome! Areas for enhancement:

- [ ] Multi-language support
- [ ] Fine-tuned domain-specific embeddings
- [ ] Advanced reranking strategies
- [ ] GraphQL API
- [ ] Persistent trace logging
- [ ] A/B testing framework

---

## 📝 License

MIT License — see LICENSE file for details

---

## 🔗 Resources

- [Milvus Documentation](https://milvus.io/docs)
- [FastAPI Guide](https://fastapi.tiangolo.com/)
- [RAGAS Evaluation Framework](https://github.com/explorerx3/ragas)
- [Groq API Reference](https://console.groq.com/docs/api-reference)
- [CRAG Paper](https://arxiv.org/abs/2401.15884)

---

**Questions?** Open an issue on GitHub or check the documentation.
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
| 2 | ✅ Done | BM25 sparse, query router, RRF fusion, cross-encoder reranking |
| 3 | ✅ Done | GraphRAG (spaCy NER + NetworkX), CRAG gate, web fallback |
| 4 | ✅ Done | RAGAS eval harness, Redis cache, evaluation dashboard |