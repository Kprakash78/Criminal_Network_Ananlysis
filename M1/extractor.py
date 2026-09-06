"""
M1.3 — NER Extraction (English, spaCy)
M1.4 — NER Extraction (Hinglish/Hindi, IndicBERT via HuggingFace)
M1.5 — Regex Extractors (PHONE, VEHICLE, ACCOUNT numbers)
PS 26152 — AI-Powered Criminal Network Analysis System

This module houses all entity extraction logic:
  - English NER: spaCy en_core_web_sm
  - Hinglish/Hindi NER: IndicBERT token classification
  - Structured field regex: Indian phone numbers, vehicle plates, account numbers

Output: list[RawEntity] per document — pre-normalization, pre-resolution.
The RawEntity type is an intermediate; the final shared schema type is in schema.py.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Intermediate extraction result (pre-normalisation)
# ---------------------------------------------------------------------------

@dataclass
class RawEntity:
    """
    A single entity as extracted from one document, before normalisation
    or resolution.  Multiple RawEntity objects may refer to the same
    real-world entity — that's what the resolver handles.
    """
    text: str                  # raw surface form as it appeared in text
    entity_type: str           # PERSON | LOCATION | ORGANIZATION | PHONE |
                               # VEHICLE | ACCOUNT | DATE | CASE
    source_doc_id: str         # doc_id of the Document it came from
    extraction_method: str     # 'spacy' | 'indicbert' | 'regex'
    raw_confidence: float      # raw confidence from the extractor (0-1)
    span_start: int = -1       # character offset in clean_text (-1 if N/A)
    span_end: int = -1


# ---------------------------------------------------------------------------
# spaCy label → M1 entity type mapping
# ---------------------------------------------------------------------------

SPACY_LABEL_MAP = {
    "PERSON":   "PERSON",
    "PER":      "PERSON",
    "ORG":      "ORGANIZATION",
    "GPE":      "LOCATION",     # geo-political entity (city, country, state)
    "LOC":      "LOCATION",     # non-GPE locations
    "FAC":      "LOCATION",     # facility
    "DATE":     "DATE",
    "TIME":     "DATE",
    "CARDINAL": None,           # skip plain numbers
    "MONEY":    None,
    "PERCENT":  None,
}

# Labels we actively want (after mapping)
WANTED_TYPES = {"PERSON", "LOCATION", "ORGANIZATION", "DATE"}


# ---------------------------------------------------------------------------
# M1.3 — spaCy English NER
# ---------------------------------------------------------------------------

_spacy_nlp = None   # lazy-loaded singleton


def _get_spacy_nlp():
    global _spacy_nlp
    if _spacy_nlp is None:
        import spacy
        try:
            _spacy_nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning(
                "en_core_web_sm not found — run: python -m spacy download en_core_web_sm"
            )
            raise
    return _spacy_nlp


def extract_entities_spacy(doc_id: str, text: str) -> list[RawEntity]:
    """
    Run spaCy NER on `text` and return RawEntity objects for wanted types.
    Confidence is fixed at 0.75 for spaCy sm (model doesn't expose per-span scores).
    """
    nlp = _get_spacy_nlp()
    results: list[RawEntity] = []

    try:
        spacy_doc = nlp(text)
    except Exception as exc:
        logger.warning(f"[spaCy] Failed on doc {doc_id}: {exc}")
        return []

    for ent in spacy_doc.ents:
        mapped = SPACY_LABEL_MAP.get(ent.label_)
        if mapped is None:
            continue
        if mapped not in WANTED_TYPES:
            continue

        surface = ent.text.strip()
        surface = surface.replace("##", "")
        
        sub_surfaces = re.split(r'\b(?i:aur|and)\b', surface)
        bad_tokens = {"ne", "ka", "ki", "ko", "se", "mein", "tha", "thi", "the", "hai", "kaha", "chala", "vide", "rs", "kya", "hua", "call", "kiya"}
        
        for sub in sub_surfaces:
            words = sub.strip().split()
            while words and words[0].lower() in bad_tokens:
                words.pop(0)
            while words and words[-1].lower() in bad_tokens:
                words.pop()
            clean_surface = " ".join(words)
            
            if not clean_surface or len(clean_surface) < 2:
                continue

            results.append(RawEntity(
                text=clean_surface,
                entity_type=mapped,
                source_doc_id=doc_id,
                extraction_method="spacy",
                raw_confidence=0.75,
                span_start=ent.start_char,
                span_end=ent.end_char,
            ))

    return results


# ---------------------------------------------------------------------------
# M1.4 — IndicBERT Hinglish/Hindi NER
# ---------------------------------------------------------------------------

_indicbert_pipeline = None   # lazy-loaded singleton
_INDICBERT_MODEL = "ai4bharat/IndicBERT-MLM-only"

# IndicBERT NER label → M1 type mapping (BIO tagging scheme)
_INDICBERT_LABEL_MAP = {
    "B-PER": "PERSON", "I-PER": "PERSON",
    "B-PERSON": "PERSON", "I-PERSON": "PERSON",
    "B-LOC": "LOCATION", "I-LOC": "LOCATION",
    "B-LOCATION": "LOCATION", "I-LOCATION": "LOCATION",
    "B-ORG": "ORGANIZATION", "I-ORG": "ORGANIZATION",
    "B-ORGANIZATION": "ORGANIZATION", "I-ORGANIZATION": "ORGANIZATION",
    "B-DATE": "DATE", "I-DATE": "DATE",
}


def _get_indicbert_pipeline():
    """
    Attempt to load an Indic NER model from HuggingFace.
    Falls back gracefully if the model cannot be loaded (no GPU, offline, etc.).

    We try ai4bharat/IndicNER first; it's a dedicated NER model for Indian languages.
    """
    global _indicbert_pipeline
    if _indicbert_pipeline is not None:
        return _indicbert_pipeline

    from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
    import torch

    # List of models to try (in order of preference)
    candidate_models = [
        "ai4bharat/IndicNER",          # Dedicated Indic NER model
        "Davlan/distilbert-base-multilingual-cased-ner-hrl",   # multilingual fallback
    ]

    for model_name in candidate_models:
        try:
            logger.info(f"[IndicBERT] Loading NER model: {model_name}")
            # local_files_only=True: model must already be cached locally.
            # This enforces the privacy requirement — no case text or metadata
            # is sent to HuggingFace during inference. The model is loaded from
            # ~/.cache/huggingface/hub/ on disk only.
            tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            model = AutoModelForTokenClassification.from_pretrained(model_name, local_files_only=True)
            _indicbert_pipeline = pipeline(
                "ner",
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy="simple",
                device=-1,          # CPU — no GPU dependency for M1
            )
            logger.info(f"[IndicBERT] Loaded: {model_name}")
            return _indicbert_pipeline
        except Exception as exc:
            logger.warning(f"[IndicBERT] Could not load {model_name}: {exc}")
            continue

    logger.warning("[IndicBERT] No Indic NER model could be loaded. Hindi/Hinglish NER will be skipped.")
    _indicbert_pipeline = False   # sentinel: tried and failed
    return None


def _is_hinglish(text: str) -> bool:
    """
    Heuristic: return True if text contains Devanagari script or common
    Hindi romanisation markers. Used to decide whether to run IndicBERT.
    """
    # Devanagari unicode block: U+0900–U+097F
    if re.search(r"[\u0900-\u097F]", text):
        return True
    # Common Hinglish words that signal code-mixing
    hinglish_words = {
        "ne", "ka", "ki", "ko", "mein", "se", "hai", "tha", "thi", "the",
        "aur", "par", "jo", "woh", "yeh", "bataya", "dekha", "pakda",
        "giraftaar", "uska", "paas", "baith", "bhaag", "chura",
    }
    words = set(re.findall(r"\b\w+\b", text.lower()))
    return len(hinglish_words & words) >= 2


# IndicBERT label set varies by model; we do a flexible lookup.
def _map_indicbert_label(label: str) -> Optional[str]:
    mapped = _INDICBERT_LABEL_MAP.get(label)
    if mapped:
        return mapped
    # Handle labels like "PER", "LOC", "ORG" without BIO prefix
    if label in ("PER", "PERSON"):
        return "PERSON"
    if label in ("LOC", "LOCATION"):
        return "LOCATION"
    if label in ("ORG", "ORGANIZATION"):
        return "ORGANIZATION"
    return None


def extract_entities_indicbert(doc_id: str, text: str) -> list[RawEntity]:
    """
    Run IndicBERT NER on `text`. Only called on sentences detected as Hinglish/Hindi.
    Returns empty list if model unavailable (pipeline continues without crash).
    """
    if not _is_hinglish(text):
        return []

    pipe = _get_indicbert_pipeline()
    if not pipe:
        return []

    results: list[RawEntity] = []
    try:
        ner_output = pipe(text)
    except Exception as exc:
        logger.warning(f"[IndicBERT] Inference failed on doc {doc_id}: {exc}")
        return []

    for ent in ner_output:
        entity_group = ent.get("entity_group") or ent.get("entity", "")
        mapped = _map_indicbert_label(entity_group)
        if mapped is None:
            continue

        surface = ent.get("word", "").strip()
        
        # --- BUG D3: Boundary detection cleanup ---
        # 1. Clean up tokenizer artifacts
        surface = surface.replace("##", "")
        
        # 2. Split multiple entities joined by ' aur ' (and)
        sub_surfaces = re.split(r'\b(?i:aur|and)\b', surface)
        
        bad_tokens = {"ne", "ka", "ki", "ko", "se", "mein", "tha", "thi", "the", "hai", "kaha", "chala", "vide", "rs", "kya", "hua", "call", "kiya"}
        
        for sub in sub_surfaces:
            words = sub.strip().split()
            # Strip trailing/leading bad tokens
            while words and words[0].lower() in bad_tokens:
                words.pop(0)
            while words and words[-1].lower() in bad_tokens:
                words.pop()
                
            clean_surface = " ".join(words)
            if not clean_surface or len(clean_surface) < 2:
                continue

            score = float(ent.get("score", 0.6))
            results.append(RawEntity(
                text=clean_surface,
                entity_type=mapped,
                source_doc_id=doc_id,
                extraction_method="indicbert",
                raw_confidence=score,
            ))

    return results


# ---------------------------------------------------------------------------
# M1.5 — Regex Extractors
# ---------------------------------------------------------------------------

# Indian mobile number: starts with 6-9, total 10 digits
# May be prefixed by +91 or 0
_PHONE_RE = re.compile(
    r"(?<!\d)"                       # not preceded by a digit
    r"(?:\+91[-.\s]?|0)?([6-9]\d{9})"
    r"(?!\d)"                        # not followed by a digit
)

# Indian vehicle registration plate: AB12CD3456
# 2 state-code letters, 2 district digits, 2 series letters, 4 sequence digits
_VEHICLE_RE = re.compile(
    r"\b([A-Z]{2}\d{2}[A-Z]{1,2}\d{4})\b"
)

# Account numbers in our synthetic dataset: ACC followed by 5 digits
# Also catch generic bank account-like patterns (10-18 digit sequences preceded by "account")
# Added support for hex-style account IDs like ACC_dac31588
_ACCOUNT_RE = re.compile(
    r"\b(ACC\d{5}|ACC_[a-fA-F0-9]{8})\b"
    r"|(?:account\s+(?:number\s+)?|a/c\s*(?:no\.?\s*)?)([0-9]{9,18})\b",
    re.IGNORECASE,
)

# Bulleted names: catches capitalized names following "1. ", "2. ", etc.
# Helps catch entities that spaCy misses in list contexts (e.g., "PRIMARY ACCUSED:\n1. Rakesh Kumar").
_BULLETED_NAME_RE = re.compile(
    r"(?m)^\s*\d+\.\s+([A-Z][A-Za-z\-]+(?:\s+[A-Z][A-Za-z\-]+)*)"
)


def extract_phones(doc_id: str, text: str) -> list[RawEntity]:
    """Extract Indian phone numbers via regex. High confidence (regex = deterministic)."""
    results = []
    for m in _PHONE_RE.finditer(text):
        number = m.group(1)
        results.append(RawEntity(
            text=number,
            entity_type="PHONE",
            source_doc_id=doc_id,
            extraction_method="regex",
            raw_confidence=0.98,
            span_start=m.start(1),
            span_end=m.end(1),
        ))
    return results


def extract_vehicles(doc_id: str, text: str) -> list[RawEntity]:
    """Extract Indian vehicle plate numbers via regex."""
    results = []
    for m in _VEHICLE_RE.finditer(text):
        plate = m.group(1)
        results.append(RawEntity(
            text=plate,
            entity_type="VEHICLE",
            source_doc_id=doc_id,
            extraction_method="regex",
            raw_confidence=0.98,
            span_start=m.start(1),
            span_end=m.end(1),
        ))
    return results


def extract_accounts(doc_id: str, text: str) -> list[RawEntity]:
    """Extract account numbers via regex."""
    results = []
    for m in _ACCOUNT_RE.finditer(text):
        # Group 1 = ACC##### style; group 2 = numeric account after keyword
        acc = m.group(1) or m.group(2)
        if acc:
            results.append(RawEntity(
                text=acc.strip(),
                entity_type="ACCOUNT",
                source_doc_id=doc_id,
                extraction_method="regex",
                raw_confidence=0.95,
                span_start=m.start(),
                span_end=m.end(),
            ))
    return results


def extract_bulleted_names(doc_id: str, text: str) -> list[RawEntity]:
    """Extract capitalized names from numbered bullet lists using regex."""
    results = []
    for m in _BULLETED_NAME_RE.finditer(text):
        name = m.group(1)
        results.append(RawEntity(
            text=name.strip(),
            entity_type="PERSON",
            source_doc_id=doc_id,
            extraction_method="regex",
            raw_confidence=0.85,  # heuristic confidence
            span_start=m.start(1),
            span_end=m.end(1),
        ))
    return results


# ---------------------------------------------------------------------------
# Combined extractor — called per Document by the pipeline
# ---------------------------------------------------------------------------

def extract_all(doc_id: str, text: str) -> list[RawEntity]:
    """
    Run all extractors on `text` and return the combined list of RawEntity objects.

    Uses the document structure classifier (M1.doc_structure) to route each
    block through a dedicated extraction path:

      HEADER_FIELD   → NER on VALUE only (isolated, never concatenated with label)
      TABLE_ROW      → direct PERSON entity from name column (NER bypassed)
      STRUCTURAL_NOISE → discarded, never reaches any extractor
      NARRATIVE      → existing spaCy + IndicBERT NER pipeline, unchanged

    Regex extractors (phone / vehicle / account) run on the full original text
    because they are precise enough and numbers can appear anywhere in a document.
    """
    from M1.doc_structure import classify_document_structure

    entities: list[RawEntity] = []

    # ---- Step 1: Classify the document structure ----
    blocks = classify_document_structure(text)

    # ---- Step 2: Route each block to its extraction path ----
    for block in blocks:

        if block.type == "STRUCTURAL_NOISE":
            # Discard entirely — never reaches NER
            continue

        elif block.type == "HEADER_FIELD":
            # Extract entities from the VALUE ONLY — isolated, not the full line.
            # This prevents "Case Type: Suspected Financial Fraud Network" from
            # being fed as "Case Type Suspected Financial Fraud Network" to NER.
            value = block.parsed_fields.get("value", "").strip() if block.parsed_fields else ""
            if value and len(value) >= 2:
                entities.extend(extract_entities_spacy(doc_id, value))
                # IndicBERT on header values: only if they look Hinglish
                entities.extend(extract_entities_indicbert(doc_id, value))

        elif block.type == "TABLE_ROW":
            # The column header already told us this is a PERSON name field.
            # Bypass NER entirely — running NER on "Solapur" would hallucinate
            # or fail; we already know the semantic type from the table structure.
            name = block.parsed_fields.get("name", "").strip() if block.parsed_fields else ""
            if name and len(name) >= 2:
                entities.append(RawEntity(
                    text=name,
                    entity_type="PERSON",
                    source_doc_id=doc_id,
                    extraction_method="table",
                    raw_confidence=0.88,
                ))

        elif block.type == "NARRATIVE":
            # Existing NER pipeline — spaCy + IndicBERT, unchanged.
            # Also extract bulleted names from narrative (e.g. "PRIMARY ACCUSED:\n1. Raka")
            narrative_text = block.raw_text
            entities.extend(extract_entities_spacy(doc_id, narrative_text))
            entities.extend(extract_entities_indicbert(doc_id, narrative_text))
            entities.extend(extract_bulleted_names(doc_id, narrative_text))

    # ---- Step 3: Regex extractors on full original text ----
    # Regex is precise enough that it doesn't need structural filtering.
    # Phone/vehicle/account numbers appear anywhere in documents.
    entities.extend(extract_phones(doc_id, text))
    entities.extend(extract_vehicles(doc_id, text))
    entities.extend(extract_accounts(doc_id, text))

    return entities
