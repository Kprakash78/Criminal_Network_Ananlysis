# llm_rag/ — Local query parsing + template answer generation (was M5)
# NOTE: No cloud API required. Works fully offline.
from .schemas import StructuredQuery, AttributedObject, M5Answer, CitedSegment
from .query_parser import parse_query
from .generator import generate_answer
from .pipeline import run_pipeline, format_answer_for_display

__all__ = [
    "StructuredQuery", "AttributedObject", "M5Answer", "CitedSegment",
    "parse_query",
    "generate_answer",
    "run_pipeline",
    "format_answer_for_display",
]
