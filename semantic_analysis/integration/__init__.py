"""
integration/ — Full Stack / Integration module (was M6)
"""
from .config import IntegrationConfig, DEFAULT_CONFIG
from .orchestrator import run_full_ingestion

__all__ = [
    "IntegrationConfig",
    "DEFAULT_CONFIG",
    "run_full_ingestion",
]
