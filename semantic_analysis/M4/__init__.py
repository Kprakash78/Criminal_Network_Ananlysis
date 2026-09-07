# M4 — Retrieval Module
# Convergence point: multi-modal fusion scoring, FAISS index management, search() API
from .config import M4Config, DEFAULT_CONFIG
from .embedder import QueryEmbedder
from .index_manager import FAISSIndexManager
from .pipeline import run_ingestion
from .search import search
from .validator import validate_search_results

__all__ = [
    "M4Config",
    "DEFAULT_CONFIG",
    "QueryEmbedder",
    "FAISSIndexManager",
    "run_ingestion",
    "search",
    "validate_search_results",
]
