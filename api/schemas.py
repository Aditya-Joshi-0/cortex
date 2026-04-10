"""
Cortex RAG — API Schemas (Pydantic v2)
"""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    provider: Optional[str] = Field(default=None, description="Provider id: groq|nvidia_nim|openai|custom")
    model:    Optional[str] = Field(default=None, description="Model id string")
    api_key:  Optional[str] = Field(default=None, description="API key override for this request")
    base_url: Optional[str] = Field(default=None, description="Base URL (custom provider only)")


class QueryRequest(BaseModel):
    query:  str = Field(..., min_length=3, max_length=2048, description="User question")
    top_k:  Optional[int] = Field(default=None, ge=1, le=20, description="Override default top-k")
    stream: bool = Field(default=True, description="Stream tokens via SSE")
    llm:    Optional[LLMConfig] = Field(default=None, description="LLM provider/model override")


class RoutingResponse(BaseModel):
    intent: str
    strategies: list[str]
    confidence: float
    reasoning: str
    retriever_hits: dict = {}


class ChunkResponse(BaseModel):
    chunk_id: str
    doc_id: str
    source: str
    title: str
    text: str           # child chunk (shown as citation snippet)
    score: float
    retriever: str = "dense"


class CitationResponse(BaseModel):
    number: int
    title: str
    source: str
    chunk_id: str
    score: float


class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: list[CitationResponse]
    retrieved_chunks: list[ChunkResponse]
    routing: Optional[RoutingResponse] = None
    crag_grade: Optional[str] = None
    crag_rewritten_query: Optional[str] = None
    web_search_used: bool = False
    model: str
    usage: dict


class IngestRequest(BaseModel):
    path: str = Field(..., description="File or directory path on server")
    recursive: bool = True


class IngestResponse(BaseModel):
    documents_processed: int
    documents_skipped: int
    chunks_created: int
    chunks_stored: int
    bm25_indexed: int = 0
    graph_entities: int = 0
    graph_triples: int = 0
    errors: list[dict] = []


class HealthResponse(BaseModel):
    status: str
    milvus: str
    embedder: str
    collection_stats: dict
    graph_stats: dict = {}



class ModelInfo(BaseModel):
    id:    str
    label: str


class ProviderInfo(BaseModel):
    id:        str
    label:     str
    base_url:  str
    models:    list[ModelInfo]
    configured: bool   # True if an API key is set in .env


class ProvidersResponse(BaseModel):
    providers:        list[ProviderInfo]
    default_provider: str
    default_model:    str
