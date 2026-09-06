"""
M1 — Document Structure Classifier
PS 26152 — AI-Powered Criminal Network Analysis System

Before ANY text reaches spaCy/IndicBERT NER, every line (or logical block)
of the input document is classified into exactly one of:

  HEADER_FIELD    — "Label: Value" style metadata lines
  TABLE_ROW       — a line that is part of a delimited or aligned table
  STRUCTURAL_NOISE — section headers, ALL-CAPS titles, separators, doc metadata
  NARRATIVE       — prose sentences → the ONLY type sent to NER unchanged

Usage:
    from M1.doc_structure import classify_document_structure, ClassifiedBlock
    blocks = classify_document_structure(raw_text)
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional, Literal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ClassifiedBlock — output unit of the classifier
# ---------------------------------------------------------------------------

@dataclass
class ClassifiedBlock:
    """
    A classified span of the document.

    type         : HEADER_FIELD | TABLE_ROW | STRUCTURAL_NOISE | NARRATIVE
    raw_text     : the original source text for this block (may be multi-line
                   for TABLE sections)
    parsed_fields: structured data extracted from the block, or None.
                   HEADER_FIELD → {"label": str, "value": str}
                   TABLE_ROW   → {"name": str, "age": str|None, "role": str|None,
                                  "_header": list[str], "_row": list[str]}
                   others      → None
    line_numbers : (start, end) 1-indexed, inclusive
    """
    type: Literal["HEADER_FIELD", "TABLE_ROW", "STRUCTURAL_NOISE", "NARRATIVE"]
    raw_text: str
    parsed_fields: Optional[dict]
    line_numbers: tuple  # (start_line, end_line)


# ---------------------------------------------------------------------------
# Compiled patterns used by the classifier
# ---------------------------------------------------------------------------

# Markdown separators: ---, ===, ⸻, ─, ___ (3+ chars, line by itself)
_MARKDOWN_SEP_RE = re.compile(r"^\s*([=\-─⸻_]{3,})\s*$")

# Numbered section headers: "1. INITIAL FIR", "2. PERSON DATABASE"
_NUMBERED_HEADER_RE = re.compile(r"^\s*\d+\.\s+[A-Z][A-Za-z0-9\s&:-]+$")

# Header field: label (≤6 short words, no terminal punct) followed by ": value"
# The label part must be short and not look like a prose sentence.
_HEADER_FIELD_RE = re.compile(
    r"^(?P<label>[A-Za-z][A-Za-z0-9\s/()\-]{0,50}):\s*(?P<value>.+)$"
)

# Sentence-like label disqualifiers: label that ends with common phrase endings,
# or contains words that indicate it's part of a prose sentence, not a field label.
_SENTENCE_INDICATORS = re.compile(
    r"\b(was|is|are|were|has|have|had|does|did|will|would|can|could|should|"
    r"said|told|saw|went|came|called|reported|stated|noted|found|observed)\b",
    re.IGNORECASE,
)

# Version / project-code lines: "PS 26152", "CAS 123"
_VERSION_CODE_RE = re.compile(r"^[A-Z]{2,4}\s?\d{3,6}$")

# Document-end sentinel patterns
_DOC_END_SENTINEL_RE = re.compile(
    r"(---\s*end\s*of\s*document|note\s*for\s*testing\s*purposes)",
    re.IGNORECASE,
)

# ASCII art / box drawing / arrows
_ASCII_ART_RE = re.compile(r"[→↓▼─⸻┌┐└┘├┤┬┴┼═║╔╗╚╝╠╣╦╩╬]")
_ARROW_NOTATION_RE = re.compile(r"->|=>")

# Tech / UI vocabulary that indicates structural headings, not narrative
_TECH_VOCAB = {
    "dashboard", "search", "graph", "export", "review", "pipeline",
    "upload", "extract", "system", "engine", "module", "analysis",
    "workflow", "documentation", "centrality", "detection", "nlp", "llm", "rag",
    "ui", "streamlit", "scorer", "normalizer", "langgraph", "interactive",
    "csv", "pdf", "louvain", "community", "extraction", "scoring",
}

# Words that strongly indicate a legitimate org/location name even if title-case short
_LEGIT_ORG_WORDS = {
    "board", "center", "centre", "group", "department", "association", "company",
    "inc", "ltd", "corp", "agency", "council", "commission", "force", "police",
    "bank", "hospital", "school", "university", "court", "station", "ministry",
}

# Name-column fuzzy keywords for table header matching
_NAME_COLUMN_KEYWORDS = {"name", "person", "witness", "suspect", "individual", "accused"}
_AGE_COLUMN_KEYWORDS = {"age"}
_ROLE_COLUMN_KEYWORDS = {"role", "relationship", "occupation", "designation", "position"}


# ---------------------------------------------------------------------------
# Internal helper: table block detection
# ---------------------------------------------------------------------------

def _detect_delimiter(line: str) -> Optional[str]:
    """
    Detect whether a line uses a consistent delimiter: tab, pipe, or 2+ spaces.
    Returns 'tab', 'pipe', or 'spaces', or None if no delimiter found.
    """
    if "\t" in line:
        return "tab"
    if "|" in line and line.count("|") >= 1:
        return "pipe"
    if re.search(r"  +", line):
        return "spaces"
    return None


def _split_by_delimiter(line: str, delimiter: str) -> list[str]:
    """Split a line using the detected delimiter type."""
    if delimiter == "tab":
        return [p.strip() for p in line.split("\t")]
    if delimiter == "pipe":
        parts = [p.strip() for p in line.split("|")]
        # strip empty leading/trailing parts from lines like "| A | B | C |"
        if parts and not parts[0]:
            parts = parts[1:]
        if parts and not parts[-1]:
            parts = parts[:-1]
        return parts
    if delimiter == "spaces":
        return [p.strip() for p in re.split(r"  +", line) if p.strip()]
    return [line]


def _match_header_semantics(headers: list[str]) -> dict:
    """
    Given a list of column header strings, return a dict mapping semantic
    roles ('name', 'age', 'role') to column indices.
    Returns {} if no name column found (not a person table).

    Handles slash-compound headers like 'Relationship / Role' by splitting
    on '/' and matching keywords against all sub-tokens.
    """
    mapping = {}
    for idx, h in enumerate(headers):
        # Flatten slash-separated sub-headers: "Relationship / Role" → ["relationship", "role"]
        sub_tokens = [t.strip().lower() for t in re.split(r'[/&,]+', h)]
        combined = " ".join(sub_tokens)
        if any(kw in combined for kw in _NAME_COLUMN_KEYWORDS):
            if "name" not in mapping:
                mapping["name"] = idx
        if any(kw in combined for kw in _AGE_COLUMN_KEYWORDS):
            if "age" not in mapping:
                mapping["age"] = idx
        if any(kw in combined for kw in _ROLE_COLUMN_KEYWORDS):
            if "role" not in mapping:
                mapping["role"] = idx
    return mapping


def _extract_table_row_fields(
    row_parts: list[str],
    col_mapping: dict,
    header_parts: list[str],
) -> dict:
    """
    Extract semantic fields from a table row given the column mapping.
    Returns a dict with 'name', 'age', 'role' (any may be None).
    Also stores the raw _header and _row for debugging.
    """
    def _get(key):
        idx = col_mapping.get(key)
        if idx is not None and idx < len(row_parts):
            val = row_parts[idx].strip()
            return val if val else None
        return None

    return {
        "name": _get("name"),
        "age": _get("age"),
        "role": _get("role"),
        "_header": header_parts,
        "_row": row_parts,
    }


# ---------------------------------------------------------------------------
# Internal helper: header field detection
# ---------------------------------------------------------------------------

def _is_header_field(line: str) -> Optional[dict]:
    """
    Return {"label": ..., "value": ...} if `line` is a HEADER_FIELD, else None.

    Rules:
      1. Must match label: value pattern
      2. Label must be ≤6 words
      3. Label must NOT contain sentence-verb indicators
      4. Value must be non-empty
      5. Line must not end with common punctuation that implies it's a sentence
         followed by a colon (like "at 10:30 pm, he said:")
    """
    m = _HEADER_FIELD_RE.match(line.strip())
    if not m:
        return None

    label = m.group("label").strip()
    value = m.group("value").strip()

    if not value:
        return None

    # Label word count check
    label_words = label.split()
    if len(label_words) > 6:
        return None

    # Label must not contain sentence-verb indicators
    if _SENTENCE_INDICATORS.search(label):
        return None

    # If the label looks like a prose phrase ending with a noun/adjective
    # and the colon is inside a quote or time reference, it's narrative.
    # Heuristic: if the label contains a comma, it's probably part of a sentence.
    if "," in label:
        return None

    # If there's a digit:digit pattern in the label (time reference like "10:30"),
    # the FIRST colon in the line was a time separator, not a field delimiter.
    # Disqualify: label contains digits (e.g. "At 10" from "At 10:30 pm, ...").
    if re.search(r'\d', label):
        return None

    return {"label": label, "value": value}


# ---------------------------------------------------------------------------
# Internal helper: structural noise detection (single line)
# ---------------------------------------------------------------------------

def _is_structural_noise(line: str) -> bool:
    """
    Return True if `line` is structural noise that should be discarded.
    Does NOT handle table headers or header fields (those are classified above).
    """
    stripped = line.strip()
    if not stripped:
        return False  # Empty lines are kept (as spacing)

    # Markdown-style separators
    if _MARKDOWN_SEP_RE.match(stripped):
        return True

    # ASCII art / box drawing
    if _ASCII_ART_RE.search(stripped):
        return True

    # Arrow notation fragments (standalone lines like "Upload -> Extract")
    if _ARROW_NOTATION_RE.search(stripped):
        return True

    # Version/project code tokens
    if _VERSION_CODE_RE.match(stripped):
        return True

    # ALL CAPS short lines (likely section headers)
    words = stripped.split()
    if len(words) <= 5 and stripped.isupper() and len(stripped) > 3:
        if re.match(r"^[A-Z\s&:()\-]+$", stripped):
            return True

    # Numbered section headers (short)
    if _NUMBERED_HEADER_RE.match(stripped) and len(words) <= 8:
        return True

    # Title-Case short phrases with tech vocabulary
    if 0 < len(words) <= 6:
        capitalized_count = sum(1 for w in words if w and w[0].isupper())
        if capitalized_count >= len(words) * 0.7:
            words_lower = set(re.sub(r"[^\w\s]", "", stripped.lower()).split())
            if any(t in words_lower for t in _TECH_VOCAB):
                if not any(org_w in words_lower for org_w in _LEGIT_ORG_WORDS):
                    return True
            # Module acronyms (M1, M2, RAG, NLP, LLM)
            if any(w.isupper() and len(w) > 1 for w in words):
                if any(t in stripped.lower() for t in {"rag", "nlp", "llm", "m2", "m3", "m4", "m1", "m5", "m6"}):
                    return True

    return False


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------

def classify_document_structure(raw_text: str) -> list[ClassifiedBlock]:
    """
    Classify every line (or logical multi-line block) of `raw_text` into one
    of: HEADER_FIELD, TABLE_ROW, STRUCTURAL_NOISE, NARRATIVE.

    Returns a list of ClassifiedBlock objects in document order.

    Table blocks spanning multiple lines produce one ClassifiedBlock per
    data row, all with type TABLE_ROW. The header row itself produces a
    STRUCTURAL_NOISE block (it is metadata, not extractable content).
    """
    lines = raw_text.splitlines()
    blocks: list[ClassifiedBlock] = []

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        line_num = i + 1  # 1-indexed

        # ----------------------------------------------------------------
        # 0. Document-end sentinel — discard rest of document
        # ----------------------------------------------------------------
        if _DOC_END_SENTINEL_RE.search(stripped):
            # Everything from here is noise
            blocks.append(ClassifiedBlock(
                type="STRUCTURAL_NOISE",
                raw_text=stripped,
                parsed_fields=None,
                line_numbers=(line_num, len(lines)),
            ))
            break  # Stop processing

        # ----------------------------------------------------------------
        # 1. Empty lines — pass through as NARRATIVE spacing (NER ignores them)
        # ----------------------------------------------------------------
        if not stripped:
            blocks.append(ClassifiedBlock(
                type="NARRATIVE",
                raw_text=line,
                parsed_fields=None,
                line_numbers=(line_num, line_num),
            ))
            i += 1
            continue

        # ----------------------------------------------------------------
        # 2. Table detection — look-ahead for multi-line consistency
        #    We classify a line as a potential table header if:
        #      a) It has a delimiter (tab, pipe, or 2+ spaces)
        #      b) The NEXT line also has the same delimiter
        #      c) The header row contains a name-column keyword
        # ----------------------------------------------------------------
        delimiter = _detect_delimiter(stripped)
        if delimiter is not None and i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            next_delimiter = _detect_delimiter(next_line)

            if next_delimiter == delimiter:
                # Check if this line is a table header with a name column
                header_parts = _split_by_delimiter(stripped, delimiter)
                col_mapping = _match_header_semantics(header_parts)

                if col_mapping.get("name") is not None:
                    # This IS a table. Emit header as STRUCTURAL_NOISE.
                    blocks.append(ClassifiedBlock(
                        type="STRUCTURAL_NOISE",
                        raw_text=line,
                        parsed_fields={"_role": "table_header", "_columns": header_parts},
                        line_numbers=(line_num, line_num),
                    ))
                    i += 1

                    # Process data rows
                    while i < len(lines):
                        row_line = lines[i]
                        row_stripped = row_line.strip()

                        # Table ends on empty line, sentinel, or line that
                        # can't be split with the same delimiter into ≥2 parts
                        if not row_stripped:
                            break
                        if _DOC_END_SENTINEL_RE.search(row_stripped):
                            break

                        row_parts = _split_by_delimiter(row_stripped, delimiter)
                        if len(row_parts) < 2:
                            break

                        fields = _extract_table_row_fields(row_parts, col_mapping, header_parts)
                        name_val = fields.get("name")

                        if name_val and len(name_val) >= 2 and not name_val.startswith("---"):
                            blocks.append(ClassifiedBlock(
                                type="TABLE_ROW",
                                raw_text=row_line,
                                parsed_fields=fields,
                                line_numbers=(i + 1, i + 1),
                            ))
                        else:
                            # Separator row or empty name — noise
                            blocks.append(ClassifiedBlock(
                                type="STRUCTURAL_NOISE",
                                raw_text=row_line,
                                parsed_fields=None,
                                line_numbers=(i + 1, i + 1),
                            ))
                        i += 1

                    continue  # outer while loop — don't fall through

        # ----------------------------------------------------------------
        # 3. Header field detection
        # ----------------------------------------------------------------
        header_match = _is_header_field(line)
        if header_match is not None:
            blocks.append(ClassifiedBlock(
                type="HEADER_FIELD",
                raw_text=line,
                parsed_fields=header_match,
                line_numbers=(line_num, line_num),
            ))
            i += 1
            continue

        # ----------------------------------------------------------------
        # 4. Structural noise detection
        # ----------------------------------------------------------------
        if _is_structural_noise(line):
            blocks.append(ClassifiedBlock(
                type="STRUCTURAL_NOISE",
                raw_text=line,
                parsed_fields=None,
                line_numbers=(line_num, line_num),
            ))
            i += 1
            continue

        # ----------------------------------------------------------------
        # 5. Default: NARRATIVE
        # ----------------------------------------------------------------
        blocks.append(ClassifiedBlock(
            type="NARRATIVE",
            raw_text=line,
            parsed_fields=None,
            line_numbers=(line_num, line_num),
        ))
        i += 1

    return blocks


# ---------------------------------------------------------------------------
# Convenience: reconstruct clean text for legacy preprocessor compatibility
# ---------------------------------------------------------------------------

def reconstruct_clean_text(blocks: list[ClassifiedBlock]) -> str:
    """
    Reconstruct a cleaned text string from classified blocks, suitable for
    passing to the old-style NER pipeline or for legacy preprocessor tests.

    - HEADER_FIELD: emit the VALUE only (label dropped)
    - TABLE_ROW: emit "name_value" only (one per row, as plain text)
    - STRUCTURAL_NOISE: omit entirely
    - NARRATIVE: emit as-is

    Note: This is used by the refactored strip_structural_noise() and is NOT
    what extract_all() uses. extract_all() routes blocks per-type.
    """
    output_lines = []
    for block in blocks:
        if block.type == "STRUCTURAL_NOISE":
            continue
        if block.type == "HEADER_FIELD":
            value = block.parsed_fields.get("value", "").strip()
            if value:
                output_lines.append(value)
        elif block.type == "TABLE_ROW":
            name = block.parsed_fields.get("name", "").strip() if block.parsed_fields else ""
            if name:
                output_lines.append(name)
        else:
            # NARRATIVE (including empty lines)
            output_lines.append(block.raw_text)

    return "\n".join(output_lines)
