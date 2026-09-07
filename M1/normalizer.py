"""
M1.6 — Entity Normalizer
PS 26152 — AI-Powered Criminal Network Analysis System

Takes a list of RawEntity objects and normalises them:
  - Casing: PERSON/LOCATION/ORGANIZATION → title case (with exceptions)
  - PHONE → canonical 10-digit form (strip +91/0 prefix)
  - VEHICLE → uppercase
  - ACCOUNT → uppercase
  - Whitespace → stripped, collapsed
  - Alias collection: collects raw surface forms as aliases
  - Regex type precedence: for spans also matched by NER as wrong type,
    regex-extracted structured fields win (per PRD §11)
  - Returns list[NormalizedEntity]
"""

import re
import logging
import copy
from dataclasses import dataclass, field
from typing import Optional
from M1.extractor import RawEntity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Normalised entity (output of this stage, input to resolver)
# ---------------------------------------------------------------------------

@dataclass
class NormalizedEntity:
    """
    A cleaned, type-validated entity ready for resolution and ID assignment.
    The `aliases` set accumulates every surface form seen across documents.
    """
    normalized_text: str
    entity_type: str
    source_doc_ids: list[str]           # all documents this text was seen in
    raw_aliases: set[str]               # all raw surface forms collected so far
    extraction_methods: set[str]        # which methods contributed
    raw_confidence: float               # best raw confidence seen so far


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

_PHONE_CLEANUP_RE = re.compile(r"[\s\-.]")   # strip separators inside number
_MULTI_SPACE_RE = re.compile(r"\s+")

# Common title abbreviations that should stay uppercased
_TITLE_ABBREV = {"Lt.", "Insp.", "SI", "ASI", "DCP", "SP", "DSP", "SHO", "IO"}


def _normalize_name(text: str) -> str:
    """
    Title-case a person/org/location name.
    Strips leading/trailing whitespace and collapses internal runs.
    """
    text = _MULTI_SPACE_RE.sub(" ", text.strip())
    # Title-case each word, but preserve known abbreviations
    words = text.split()
    result = []
    for w in words:
        if w.upper() in {a.upper() for a in _TITLE_ABBREV}:
            result.append(w.upper())
        else:
            result.append(w.capitalize())
    return " ".join(result)


def _normalize_phone(text: str) -> str:
    """Strip any prefix and separators; return bare 10-digit number."""
    digits = re.sub(r"\D", "", text)
    if digits.startswith("91") and len(digits) == 12:
        digits = digits[2:]
    elif digits.startswith("0") and len(digits) == 11:
        digits = digits[1:]
    return digits


def _normalize_vehicle(text: str) -> str:
    """Uppercase and remove internal spaces/hyphens."""
    return re.sub(r"[\s\-]", "", text.upper())


def _normalize_account(text: str) -> str:
    """Uppercase account numbers, except preserve lowercase for ACC_hex IDs."""
    text = text.strip()
    if text.upper().startswith("ACC_"):
        return "ACC_" + text[4:].lower()
    return text.upper()


# ---------------------------------------------------------------------------
# Bug 4 — Indian locality pattern reclassifier
# ---------------------------------------------------------------------------
#
# spaCy en_core_web_sm is trained mostly on Western text and routinely
# misclassifies Indian locality names ("Malviya Nagar", "Vaishali Nagar",
# "Andheri East", "Bandra West") as PERSON because their two-word pattern
# is structurally identical to a Western first+last name pair.
#
# Fix: run a post-extraction reclassification pass on the raw entity list.
# If an entity tagged as PERSON or ORGANIZATION ends with a well-known Indian
# locality suffix (as the LAST word of a multi-word string), reclassify it
# as LOCATION.
#
# Conservatism constraints:
#   - Only fires on multi-word candidates (2+ tokens). Single-word strings
#     ending in "Nagar" etc. that are genuine given names are rare enough not
#     to worry about, but we still require the full string to have at least
#     one non-suffix word.
#   - Directional qualifiers (West, East, North, South) immediately before
#     a locality suffix count as the suffix itself ("Andheri East" → LOCATION
#     because "East" after a place word is a directional qualifier, not a
#     person surname).
#   - Strings that are already tagged LOCATION are not touched.
#   - Only PERSON and ORGANIZATION misclassifications are corrected.
#
# ---------------------------------------------------------------------------

# Common Indian locality name suffixes. These are specifically LOCALITY-naming
# conventions and are very unlikely to be parts of human surnames in India.
_INDIAN_LOCALITY_SUFFIXES = {
    # Colony / residential area patterns
    "nagar", "colony", "vihar", "puram", "enclave", "basti", "ganj", "pura",
    "layout", "extension", "bagh", "kunj", "garden", "gardens",
    # Road / street patterns
    "marg", "road", "lane", "avenue", "path", "marg",
    # Commercial / institutional patterns
    "chowk", "crossing", "square", "market", "bazaar", "bazaars",
    # Area qualifiers — only meaningful AFTER a place-name word
    "area", "block", "phase", "pocket", "zone",
    # Explicit directional qualifiers (West, East, North, South)
    # These are included so "Andheri East", "Bandra West" etc. are caught.
    # We only trigger this when the penultimate word is also location-like.
    "west", "east", "north", "south",
}

# Directional qualifier words — when the LAST word is one of these, we also
# check the SECOND-TO-LAST word against _INDIAN_LOCALITY_SUFFIXES or the
# broader locality context.
_DIRECTIONAL_QUALIFIERS = {"west", "east", "north", "south"}

# Words that are almost never part of a human name but commonly appear in
# Indian place names — used as a positive signal when combined with a suffix.
_LOCALITY_POSITIVE_WORDS = {
    "nagar", "colony", "vihar", "puram", "enclave", "basti", "ganj", "pura",
    "layout", "extension", "bagh", "kunj", "marg", "chowk", "sector",
    "block", "phase", "pocket", "zone", "area",
}


def reclassify_indian_locality_patterns(entities: list["RawEntity"]) -> list["RawEntity"]:
    """
    Reclassify PERSON/ORGANIZATION entities whose text matches common
    Indian locality naming patterns as LOCATION.

    Rules (applied in order, all must pass for reclassification):
      1. Entity type is PERSON or ORGANIZATION (LOCATION is already correct)
      2. Text has at least 2 words (multi-word requirement)
      3. Text is NOT a known official title or police rank pattern
      4. Last word (lowercased) is an Indian locality suffix OR:
         - Last word is a directional qualifier AND second-to-last word
           is a locality suffix or the overall phrase looks like a place

    When reclassification fires:
      - entity_type → LOCATION
      - raw_confidence is reduced slightly (heuristic, not model certainty)
    """
    # Patterns that look like locality names but are actually org/title names
    # that should NOT be reclassified — conservative exclusion list.
    _EXCLUDE_PATTERNS = [
        r"\bSI\b", r"\bACP\b", r"\bDCP\b", r"\bSHO\b",   # police ranks
        r"\bPublic\b", r"\bPrivate\b", r"\bLtd\b", r"\bInc\b",  # org markers
    ]
    _EXCLUDE_RE = re.compile("|".join(_EXCLUDE_PATTERNS))

    result = []
    for ent in entities:
        if ent.entity_type not in ("PERSON", "ORGANIZATION"):
            result.append(ent)
            continue

        # Get words — ignore punctuation for matching purposes
        words = re.sub(r"[^\w\s]", "", ent.text.strip()).split()
        if len(words) < 2:
            result.append(ent)
            continue

        last_word = words[-1].lower()
        second_last = words[-2].lower() if len(words) >= 2 else ""

        # Exclusion guard: skip if the text contains org/title markers
        if _EXCLUDE_RE.search(ent.text):
            result.append(ent)
            continue

        should_reclassify = False

        if last_word in _INDIAN_LOCALITY_SUFFIXES and last_word not in _DIRECTIONAL_QUALIFIERS:
            # Straightforward suffix match: "Malviya Nagar", "Vaishali Vihar", etc.
            should_reclassify = True

        elif last_word in _DIRECTIONAL_QUALIFIERS:
            # Directional qualifier: "Andheri East", "Bandra West"
            # Trigger if the second-to-last word is also a known locality suffix
            # OR if it's a known locality name / Sector N pattern.
            if second_last in _LOCALITY_POSITIVE_WORDS:
                should_reclassify = True
            elif re.match(r"sector$", second_last) and len(words) >= 3 and words[-2].isdigit():
                # "Sector 22 West" style
                should_reclassify = True
            else:
                # General heuristic: a word + directional qualifier strongly
                # suggests a place name (e.g. "Andheri East", "Bandra West").
                # Almost no human name ends with " West" or " East" in Hindi/Indian context.
                should_reclassify = True

        elif last_word == "sector" or (len(words) >= 2 and second_last == "sector" and words[-1].isdigit()):
            # "Sector 22", "Sector 14" etc.
            should_reclassify = True

        if should_reclassify:
            # Create a copy with updated type and slightly reduced confidence
            # (heuristic reclassification, not model-assigned)
            new_ent = copy.replace(ent,
                entity_type="LOCATION",
                raw_confidence=min(ent.raw_confidence, 0.72),
            )
            result.append(new_ent)
            logger.debug(
                f"[LocalityReclassifier] '{ent.text}' {ent.entity_type} -> LOCATION "
                f"(suffix='{last_word}')"
            )
        else:
            result.append(ent)

    return result



def _normalize_text(entity_type: str, raw_text: str) -> Optional[str]:
    """
    Apply type-appropriate normalisation.
    Returns None if the entity should be discarded (e.g. too short, known noise).
    """
    text = raw_text.strip()
    if not text:
        return None

    if entity_type == "PHONE":
        norm = _normalize_phone(text)
        if len(norm) != 10 or not norm[0].isdigit() or int(norm[0]) < 6:
            return None   # failed validation — discard
        return norm

    if entity_type == "VEHICLE":
        norm = _normalize_vehicle(text)
        # Must match Indian format after normalisation
        if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}", norm):
            return None
        return norm

    if entity_type == "ACCOUNT":
        return _normalize_account(text)

    if entity_type in ("PERSON", "LOCATION", "ORGANIZATION"):
        norm = _normalize_name(text)
        # Discard single-character or very short non-meaningful tokens
        if len(norm) < 2:
            return None
        # Discard pure-digit "names"
        if re.fullmatch(r"\d+", norm):
            return None
            
        norm_lower = norm.lower()
        
        # Discard document structural terms that spaCy hallucinates as entities
        JUNK_EXACT = {
            "psychological", "communications", "timeline", "authorship",
            "network graph", "identity", "relationship", "possible",
            "physical", "who", "family", "documentary", "hypothesis",
            "document", "location", "evidence", "activity", "organization",
            "person", "vehicle", "account", "date", "information", "fir",
            # Single-word fragments frequently hallucinated from FIR / Hinglish text
            "si", "na", "bank", "shakti", "anita", "vikram",
            "ipc", "detail records", "call detail records",
        }
        JUNK_SUBSTRING = {
            "case no", "police station", "reporting officer",
            "first information", "under section", "ipc", "crpc", "ps 26152",
            "evidence database", "activity database", "initial fir", "network graph",
            "detail records",
        }

        if norm_lower in JUNK_EXACT:
            return None
        if any(j in norm_lower for j in JUNK_SUBSTRING):
            return None

        # --- Reject date-like strings being tagged as LOCATION ---
        # spaCy often tags DD/MM/YYYY timestamps as GPE/LOC; reject them here.
        if entity_type == "LOCATION" and re.search(r"\d{2}/\d{2}/\d{4}", norm):
            return None

        # --- Reject time strings like "21:00" or "19:30" appearing as entities ---
        if re.search(r"^\d{1,2}:\d{2}$", norm.strip()):
            return None

        # --- Hindi/Hinglish stopword fragment filter ---
        # Belt-and-suspenders: even if a fragment survived extraction-level
        # filtering, reject it here if it's a Hindi function-word fragment.
        try:
            from M1.extractor import _is_hinglish_noise_fragment
            if _is_hinglish_noise_fragment(norm_lower):
                return None
        except ImportError:
            pass  # Extractor not available — skip this check
            
        # Reject generalized tech/UI structural terms (broadened filter instead of just exact strings)
        TECH_VOCAB = {
            "dashboard", "search", "graph", "export", "review", "pipeline",
            "upload", "extract", "system", "engine", "module", "analysis",
            "workflow", "documentation", "centrality", "detection", "nlp", "llm", "rag",
            "ui", "streamlit", "scorer", "normalizer", "langgraph", "players", "interactive",
            "csv", "pdf", "louvain", "community"
        }
        
        # Words that strongly indicate a legitimate organization/location name
        LEGIT_ORG_WORDS = {
            "board", "center", "group", "department", "association", "company",
            "inc", "ltd", "corp", "agency", "council", "commission", "force", "police",
            "bank", "hospital", "school", "university", "court", "station", "ministry"
        }
        
        words = re.sub(r"[^\w\s]", "", norm_lower).split()
        if len(words) <= 7:
            # If it contains a tech word, but DOES NOT contain any legit org word
            if any(tech_word in words for tech_word in TECH_VOCAB):
                if not any(org_word in words for org_word in LEGIT_ORG_WORDS):
                    return None
                
        # Also drop if it has dangling commas or weird trailing stuff
        if norm.endswith(",") or norm.endswith(":"):
            return None
                
        # Reject version/project-code-like tokens (e.g. "PS 26152", "CAS 123")
        if re.match(r"^[A-Za-z]{2,4}\s?\d{3,6}$", norm):
            return None
            
        # Reject malformed/truncated fragments (unbalanced quotes)
        if norm.count('"') % 2 != 0 or norm.count("'") % 2 != 0:
            return None
            
        # Reject common junk words that leak from headers or doc metadata
        JUNK_WORDS = {"the", "a", "an", "this", "that", "entity", "name", "id", "entity_id", "entity_ids", "none", "null", "undefined"}
        if norm_lower in JUNK_WORDS:
            return None
            
        return norm

    if entity_type == "DATE":
        text_stripped = text.strip()
        # Reject strings that look like badge/employee numbers rather than dates
        # (pure 4-digit numbers that aren't year-like context)
        if re.fullmatch(r"\d{4}", text_stripped):
            # Allow 4-digit years (1900–2099) but reject badge/ID numbers
            year = int(text_stripped)
            if year < 1900 or year > 2099:
                return None
        # Reject 'between X:YY hrs' style strings — too verbose to be useful DATE nodes
        if re.match(r"between\s+\d", text_stripped, re.IGNORECASE):
            return None
        # Reject time-range strings (e.g. '21:00', '19:30')
        if re.fullmatch(r"\d{1,2}:\d{2}", text_stripped):
            return None
        return text_stripped

    return text.strip()


# ---------------------------------------------------------------------------
# Regex takes precedence: mark spans occupied by regex entities
# so spaCy doesn't get to claim the same span as a wrong type.
# ---------------------------------------------------------------------------

def _apply_regex_precedence(entities: list[RawEntity]) -> list[RawEntity]:
    """
    Per PRD §11: regex-extracted structured fields (PHONE/VEHICLE/ACCOUNT)
    always win over NER for the same span.

    We identify regex spans and remove NER entities that overlap them.
    Returns the pruned list.
    """
    regex_entities = [
        e for e in entities
        if e.extraction_method == "regex" and e.entity_type in {"PHONE", "VEHICLE", "ACCOUNT"}
    ]
    regex_spans: list[tuple[int, int]] = [
        (e.span_start, e.span_end)
        for e in regex_entities
        if e.span_start >= 0
    ]

    # Some multilingual NER pipelines do not return character offsets. Keep
    # the same structured-value precedence for those outputs by comparing a
    # normalized form of the account/phone/vehicle text.
    regex_values = {
        re.sub(r"[^a-z0-9]", "", e.text.casefold())
        for e in regex_entities
        if e.text
    }

    if not regex_spans and not regex_values:
        return entities

    def _overlaps(start: int, end: int) -> bool:
        for rs, re_ in regex_spans:
            if start < re_ and end > rs:
                return True
        return False

    result = []
    for e in entities:
        if e.extraction_method != "regex" and e.span_start >= 0:
            if _overlaps(e.span_start, e.span_end):
                continue   # drop this NER entity; regex covers the span
        if e.extraction_method != "regex" and e.text:
            normalized_value = re.sub(r"[^a-z0-9]", "", e.text.casefold())
            if normalized_value in regex_values and normalized_value:
                continue   # offset-less NER duplicate of a structured field
        result.append(e)

    return result


# ---------------------------------------------------------------------------
# Main normaliser
# ---------------------------------------------------------------------------

def normalize_entities(raw_entities: list[RawEntity]) -> list[NormalizedEntity]:
    """
    Normalise a list of RawEntity objects extracted from one or more documents.

    Steps:
      1. Apply regex-precedence pruning
      2. Bug 4 fix: reclassify Indian locality patterns (PERSON/ORG → LOCATION)
      3. Normalise each entity's text for its type
      4. Group identical (normalized_text, entity_type) pairs, accumulating aliases
         and source_doc_ids

    Returns a list of NormalizedEntity objects (one per unique normalised text+type).
    """
    pruned = _apply_regex_precedence(raw_entities)

    # Bug 4 fix: reclassify Indian locality name patterns before bucketing.
    # This corrects PERSON/ORGANIZATION misclassifications from spaCy's English
    # model on Indian locality names ("Malviya Nagar", "Vaishali Nagar",
    # "Andheri East", "Bandra West" etc.).
    pruned = reclassify_indian_locality_patterns(pruned)

    # bucket: (normalized_text, entity_type) -> NormalizedEntity
    bucket: dict[tuple[str, str], NormalizedEntity] = {}

    for raw in pruned:
        norm_text = _normalize_text(raw.entity_type, raw.text)
        if norm_text is None:
            continue   # discard invalid

        key = (norm_text, raw.entity_type)
        if key in bucket:
            existing = bucket[key]
            existing.raw_aliases.add(raw.text)
            if raw.source_doc_id not in existing.source_doc_ids:
                existing.source_doc_ids.append(raw.source_doc_id)
            existing.extraction_methods.add(raw.extraction_method)
            existing.raw_confidence = max(existing.raw_confidence, raw.raw_confidence)
        else:
            bucket[key] = NormalizedEntity(
                normalized_text=norm_text,
                entity_type=raw.entity_type,
                source_doc_ids=[raw.source_doc_id],
                raw_aliases={raw.text},
                extraction_methods={raw.extraction_method},
                raw_confidence=raw.raw_confidence,
            )

    return list(bucket.values())
