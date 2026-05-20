# Cortex RAG — Architecture & Implementation Guide

This document provides a deep dive into the architecture of **Cortex**, a production-grade Retrieval-Augmented Generation (RAG) system. This guide is structured to help you explain the "what", "how", and "why" of each layer during your GenAI Engineer interview.

---

## 🏗️ 1. High-Level Architecture Overview

Cortex follows a modular, multi-layer RAG architecture designed for high precision, scalability, and reliability. It moves beyond "naive RAG" by implementing:
- **Semantic Data Ingestion** (instead of fixed-size chunking)
- **Hybrid Multi-Strategy Retrieval** (Dense + Sparse + Knowledge Graph)
- **Corrective Gating (CRAG)** (to handle retrieval failures)
- **Reference-Free Evaluation** (using RAGAS)

---

## 📥 2. Ingestion Layer: "Context-Aware Processing"

### **Document Loading**
- Supports multiple formats: PDF, HTML, and TXT.
- **Implementation:** `DocumentLoader` handles parsing and basic cleaning.

### **Semantic Chunker (`ingestion/chunker.py`)**
- **The Problem:** Fixed-size chunking (e.g., 512 tokens) often splits mid-sentence or mid-concept, losing semantic coherence.
- **The Solution:** We use **Sentence-Level Semantic Boundary Detection**.
- **How it works:**
    1. Split text into individual sentences.
    2. Embed each sentence using `BAAI/bge-small-en-v1.5`.
    3. Compute the **cosine similarity** between consecutive sentence embeddings.
    4. Insert a chunk boundary whenever the similarity drops below a certain threshold (e.g., 0.82) or the token limit is reached.
- **Why?** This ensures each chunk contains a single, coherent topic.

### **Parent-Child Hierarchy**
- **The Problem:** Small chunks are better for retrieval precision, but large chunks provide better context for generation.
- **The Implementation:** 
    - **Child Chunks (~256 tokens):** These are the units indexed in the vector database. They represent a specific "nugget" of information.
    - **Parent Chunks (~1024 tokens):** A wider window of text centered on the child. When a child is retrieved, its **parent text** is what gets sent to the LLM.
- **Why?** It decouples **retrieval granularity** (find exactly what you need) from **context width** (give the LLM enough room to understand).

---

## 🔍 3. Retrieval Layer: "Multi-Strategy Orchestration"

Cortex doesn't just rely on vector search; it uses a `MultiStrategyRetriever` to combine different search paradigms.

### **A. Dense Retrieval (Milvus)**
- **Embeddings:** `bge-small-en-v1.5` (384-dim).
- **Vector DB:** Milvus (Dockerized).
- **Indexing:** `IVF_FLAT` with `COSINE` similarity metric.
- **Why?** Captures semantic meaning (e.g., "puppy" matches "dog").

### **B. Sparse Retrieval (BM25)**
- **Implementation:** `rank_bm25` library.
- **Why?** Essential for exact keyword matching, acronyms, and specific names (e.g., "Project Cortex-X1") where vector search might be too "fuzzy".

### **C. Knowledge Graph (GraphRAG)**
- **Extraction:** During ingestion, we use **spaCy** for Named Entity Recognition (NER) and **REBEL** (or LLM) for relation extraction.
- **Storage:** A NetworkX graph storing triples: `(Subject) --[Predicate]--> (Object)`.
- **Retrieval:**
    1. Extract entities from the user query.
    2. Traverse the graph to find related nodes (multi-hop traversal).
    3. Retrieve the chunks associated with those nodes.
- **Why?** Solves "multi-hop" queries where the answer requires connecting disparate pieces of information across the document.

### **D. Fusion & Reranking (`retrieval/fusion.py`)**
- **RRF (Reciprocal Rank Fusion):** Combines the ranked lists from Milvus, BM25, and the Graph into one unified list.
- **Cross-Encoder Reranker:** We take the top-15 fused candidates and run them through a Cross-Encoder (e.g., `BAAI/bge-reranker-base`).
- **Why?** Cross-encoders are much more accurate (but slower) than vector search because they look at the query and chunk simultaneously. Using them as a final "filter" boosts precision significantly.

---

## 🧠 4. Generation Layer: "Corrective RAG (CRAG)"

The `CRAGGate` (`generation/crag.py`) acts as a "quality control" layer between retrieval and the LLM.

### **The CRAG Workflow**
1. **Grading:** An LLM-as-judge assesses if the retrieved chunks are relevant to the query.
2. **Action Categories:**
    - **GOOD:** Chunks are relevant. Proceed to generation.
    - **POOR:** Chunks are partially relevant. **Rewrite the query** (using CoT) and re-retrieve to find better results.
    - **ABSENT:** Knowledge base doesn't have the answer. **Fallback to Web Search** (Tavily/DuckDuckGo).
3. **LLM Generation:** Uses Groq (Llama 3), OpenAI, or NVIDIA NIM to generate the final answer with **inline citations** (e.g., "The sky is blue [1].").

---

## 📊 5. Evaluation Layer: "Reference-Free Metrics"

Since production RAG systems often lack "ground truth" answers, we use the **RAGAS** framework (`evaluation/ragas_eval.py`).

### **Key Metrics**
- **Faithfulness:** Does the answer stay true to the retrieved context? (Prevents hallucinations).
- **Answer Relevancy:** Does the answer actually address the user's question?
- **Context Precision:** Were the retrieved chunks actually useful?
- **Context Utilisation:** What % of retrieved chunks were actually cited?

### **Implementation**
- Evaluations run **asynchronously** in background threads so they don't slow down the user's response time.
- Results are stored in a local SQLite DB for monitoring.

---

## 🛠️ 6. System & Infrastructure

- **API:** FastAPI for high-performance, asynchronous endpoints.
- **UI:** Streamlit for a clean, interactive dashboard (Ask, Ingest, Monitor).
- **Cache:** Redis for caching query results (TTL-based) to save LLM costs and latency.
- **Deployment:** Full **Docker Compose** setup for Milvus, Redis, API, and UI.

---

## 💡 Interview Tip: "Why this architecture?"

If asked why you built it this way, emphasize these three points:
1. **Precision:** By using **Semantic Chunking** and **Cross-Encoder Reranking**, we ensure only the most relevant context reaches the LLM.
2. **Reliability:** **CRAG** ensures the system doesn't hallucinate when the knowledge base is missing information.
3. **Observability:** By integrating **RAGAS**, we have an automated way to track performance and catch regressions.

Good luck with your interview! 🚀
