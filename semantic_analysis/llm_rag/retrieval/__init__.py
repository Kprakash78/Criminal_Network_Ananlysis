# llm_rag/retrieval/ — Retrieval module (was M4)
from .config import RetrievalConfig, DEFAULT_CONFIG
from .pipeline import run_ingestion
from .search import search

__all__ = ["RetrievalConfig", "DEFAULT_CONFIG", "run_ingestion", "search"]
