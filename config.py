"""
Cortex RAG System — Configuration
All runtime settings sourced from environment variables with safe defaults.
"""
import os
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Milvus ──────────────────────────────────────────────
    milvus_host: str = os.getenv("MILVUS_HOST", "localhost")
    milvus_port: int = int(os.getenv("MILVUS_PORT", 19530))
    milvus_collection: str = os.getenv("MILVUS_COLLECTION", "cortex_chunks")
    milvus_index_type: str = os.getenv("MILVUS_INDEX_TYPE", "IVF_FLAT")   # swap to HNSW for larger corpora
    milvus_metric_type: str = os.getenv("MILVUS_METRIC_TYPE", "COSINE")
    milvus_nlist: int = int(os.getenv("MILVUS_NLIST", 128))               # IVF nlist; ~sqrt(num_vectors)
    milvus_nprobe: int = 16               # search nprobe

    # ── Embedding model ─────────────────────────────────────
    embed_model_name: str = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-small-en-v1.5")
    embed_dim: int = 384                  # bge-small output dim
    embed_batch_size: int = 64
    embed_device: str = os.getenv("EMBED_DEVICE", "cpu")             # "cuda" if GPU available

    # ── Chunking ─────────────────────────────────────────────
    chunk_size_tokens: int = 256          # child chunk (small, precise)
    parent_chunk_size_tokens: int = 1024  # parent chunk (wide context)
    semantic_similarity_threshold: float = 0.82  # cosine cutoff for splits
    chunk_overlap_tokens: int = 32

    # ── Retrieval ────────────────────────────────────────────
    retrieval_top_k: int = 15            # candidates before reranking
    final_top_k: int = 5                 # chunks sent to LLM

    # ── LLM / Groq ───────────────────────────────────────────
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")  # must be set in .env for LLM classification to work
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    groq_temperature: float = float(os.getenv("GROQ_TEMPERATURE", 0.1))
    groq_max_tokens: int = int(os.getenv("GROQ_MAX_TOKENS", 1024))
    groq_timeout: int = int(os.getenv("GROQ_TIMEOUT", 30))  # seconds before Groq client timeout

    # ── FastAPI ──────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True

    # ── Paths ─────────────────────────────────────────────────
    data_dir: str = "data/documents"
    log_level: str = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
