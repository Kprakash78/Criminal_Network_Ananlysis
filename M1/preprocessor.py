"""
M1 Preprocessor — strip_structural_noise()
PS 26152 — AI-Powered Criminal Network Analysis System

Public API preserved for backward compatibility with existing tests and pipeline
callers. Internally delegates to the document structure classifier (doc_structure.py)
rather than re-implementing its own ad-hoc line filtering.

The old implementation's hand-written rules have been moved into doc_structure.py
where they live as part of the unified classifier, applied consistently.
"""

import logging
from M1.doc_structure import classify_document_structure, reconstruct_clean_text

logger = logging.getLogger(__name__)


def strip_structural_noise(text: str) -> str:
    """
    Preprocess document text to strip lines that are clearly structural or
    formatting artifacts before they reach the NER model.

    Returns a cleaned string suitable for NER. Header field values are retained
    (labels stripped); table name columns are retained as plain text; structural
    noise is removed entirely.

    This function preserves backward compatibility: existing callers that pass
    raw text and expect clean text back will continue to work correctly.
    """
    blocks = classify_document_structure(text)
    return reconstruct_clean_text(blocks)
