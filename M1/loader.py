"""
M1.2 — Loader + Preprocessor
PS 26152 — AI-Powered Criminal Network Analysis System

Reads raw FIR text files, CDR CSVs, and transaction CSVs into a unified
internal Document representation with basic cleaning applied.

The Document type is the primary data carrier for all downstream M1 stages.
"""

import csv
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Unified Document representation
# ---------------------------------------------------------------------------

@dataclass
class Document:
    """
    Unified representation of one source record after loading and preprocessing.

    doc_id       : stable identifier derived from source filename / row index
    doc_type     : 'FIR' | 'CDR' | 'TRANSACTION'
    source_path  : absolute path to the originating file
    raw_text     : original content (joined row fields for CSV sources)
    clean_text   : after encoding normalisation, noise removal, whitespace clean
    sentences    : clean_text split into sentences
    metadata     : type-specific structured fields (e.g. caller/callee for CDR)
    """
    doc_id: str
    doc_type: str          # 'FIR' | 'CDR' | 'TRANSACTION'
    source_path: str
    raw_text: str
    clean_text: str
    sentences: list[str]
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Text cleaning helpers
# ---------------------------------------------------------------------------

# Regex for stripping control characters while keeping newlines
_CONTROL_RE = re.compile(r"[^\S\n]+")            # collapse horizontal whitespace
_MULTI_NL_RE = re.compile(r"\n{3,}")             # collapse 3+ blank lines to 2

def _normalize_unicode(text: str) -> str:
    """NFC-normalise and strip invisible control chars (except newlines/tabs)."""
    text = unicodedata.normalize("NFC", text)
    # Remove control characters except \n and \t
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in "\n\t")
    return text


def _clean_text(raw: str) -> str:
    """
    Apply encoding normalisation, noise removal, and whitespace cleanup.
    Preserves Devanagari and Latin characters (Hinglish-safe).
    """
    text = _normalize_unicode(raw)
    text = _CONTROL_RE.sub(" ", text)       # collapse runs of whitespace (not \n)
    text = _MULTI_NL_RE.sub("\n\n", text)   # max two consecutive blank lines
    text = text.strip()
    return text


# Simple sentence splitter that handles:
#  - English sentence endings (.!?)
#  - Devanagari danda (।)
#  - Line-break boundaries in FIR blocks
_SENT_SPLIT_RE = re.compile(
    r"(?<=[.!?।])\s+(?=[A-Z\u0900-\u097F])"   # after . ! ? । before capital/Devanagari
    r"|(?<=\n)\s*(?=[A-Z\u0900-\u097F])"       # at line boundaries
)


def _split_sentences(text: str) -> list[str]:
    """Split cleaned text into sentence-sized chunks. Returns at least [text]."""
    parts = _SENT_SPLIT_RE.split(text)
    sentences = [s.strip() for s in parts if s.strip()]
    return sentences if sentences else [text]


# ---------------------------------------------------------------------------
# FIR loader
# ---------------------------------------------------------------------------

def load_fir(file_path: Path) -> Document | None:
    """
    Load a single FIR text file.
    Returns None (with a warning) if the file cannot be read.
    """
    try:
        raw = file_path.read_text(encoding="utf-8")
    except Exception as exc:
        # Failure handling: log and return None — caller must skip
        print(f"[WARN] Could not read FIR {file_path}: {exc}")
        return None

    clean = _clean_text(raw)
    return Document(
        doc_id=file_path.stem,          # e.g. "FIR_001"
        doc_type="FIR",
        source_path=str(file_path),
        raw_text=raw,
        clean_text=clean,
        sentences=_split_sentences(clean),
        metadata={"filename": file_path.name},
    )


def load_fir_directory(fir_dir: Path) -> list[Document]:
    """Load all *.txt files in a directory. Skips unreadable files."""
    docs = []
    for fir_file in sorted(fir_dir.glob("*.txt")):
        doc = load_fir(fir_file)
        if doc is not None:
            docs.append(doc)
    return docs


# ---------------------------------------------------------------------------
# CDR loader
# ---------------------------------------------------------------------------

def load_cdrs(cdr_path: Path) -> list[Document]:
    """
    Load CDR CSV. Each row becomes one Document whose clean_text is a
    natural-language description of the call (so NER can run on it).
    """
    docs = []
    try:
        with cdr_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                try:
                    caller = row.get("caller", "").strip()
                    callee = row.get("callee", "").strip()
                    ts = row.get("timestamp", "").strip()
                    duration = row.get("duration_seconds", "").strip()
                    location = row.get("tower_location", "").strip()

                    raw_text = (
                        f"Call from {caller} to {callee} "
                        f"at {ts} lasting {duration} seconds "
                        f"via tower at {location}."
                    )
                    clean = _clean_text(raw_text)
                    docs.append(Document(
                        doc_id=f"CDR_{str(i + 1).zfill(4)}",
                        doc_type="CDR",
                        source_path=str(cdr_path),
                        raw_text=raw_text,
                        clean_text=clean,
                        sentences=[clean],
                        metadata={
                            "caller": caller,
                            "callee": callee,
                            "timestamp": ts,
                            "duration_seconds": duration,
                            "tower_location": location,
                        },
                    ))
                except Exception as exc:
                    print(f"[WARN] Could not parse CDR row {i}: {exc}")
    except Exception as exc:
        print(f"[WARN] Could not read CDR file {cdr_path}: {exc}")
    return docs


# ---------------------------------------------------------------------------
# Transaction loader
# ---------------------------------------------------------------------------

def load_transactions(trans_path: Path) -> list[Document]:
    """
    Load transaction CSV. Each row becomes one Document.
    clean_text is a natural-language description of the transaction.
    """
    docs = []
    try:
        with trans_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                try:
                    src = row.get("source_account", "").strip()
                    tgt = row.get("target_account", "").strip()
                    amount = row.get("amount_inr", "").strip()
                    ts = row.get("timestamp", "").strip()
                    remarks = row.get("remarks", "").strip()

                    raw_text = (
                        f"Transaction from account {src} to account {tgt} "
                        f"amount INR {amount} on {ts}. Remarks: {remarks}."
                    )
                    clean = _clean_text(raw_text)
                    docs.append(Document(
                        doc_id=f"TXN_{str(i + 1).zfill(4)}",
                        doc_type="TRANSACTION",
                        source_path=str(trans_path),
                        raw_text=raw_text,
                        clean_text=clean,
                        sentences=[clean],
                        metadata={
                            "source_account": src,
                            "target_account": tgt,
                            "amount_inr": amount,
                            "timestamp": ts,
                            "remarks": remarks,
                        },
                    ))
                except Exception as exc:
                    print(f"[WARN] Could not parse transaction row {i}: {exc}")
    except Exception as exc:
        print(f"[WARN] Could not read transaction file {trans_path}: {exc}")
    return docs


# ---------------------------------------------------------------------------
# Unified loader — called by the pipeline
# ---------------------------------------------------------------------------

def load_all(source_dir: Path) -> list[Document]:
    """
    Load all source types from `source_dir/data/` hierarchy.

    Expected layout (produced by generate_dataset.py):
      source_dir/data/firs/*.txt
      source_dir/data/cdrs/cdr.csv
      source_dir/data/transactions/transactions.csv

    Skips missing paths gracefully (logs warning, continues).
    """
    documents: list[Document] = []

    fir_dir = source_dir / "data" / "firs"
    cdr_path = source_dir / "data" / "cdrs" / "cdr.csv"
    trans_path = source_dir / "data" / "transactions" / "transactions.csv"

    if fir_dir.exists():
        fir_docs = load_fir_directory(fir_dir)
        documents.extend(fir_docs)
    else:
        print(f"[WARN] FIR directory not found: {fir_dir}")

    if cdr_path.exists():
        cdr_docs = load_cdrs(cdr_path)
        documents.extend(cdr_docs)
    else:
        print(f"[WARN] CDR file not found: {cdr_path}")

    if trans_path.exists():
        trans_docs = load_transactions(trans_path)
        documents.extend(trans_docs)
    else:
        print(f"[WARN] Transaction file not found: {trans_path}")

    return documents
