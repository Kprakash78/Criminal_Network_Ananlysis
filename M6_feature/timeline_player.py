"""
M6_feature — Timeline Event Player Backend
PS 26152 — AI-Powered Criminal Network Analysis System

Transforms CDR, transaction, and FIR data into a chronologically sorted
event stream for timeline visualization. Each event includes:
- Timestamp
- Event type (call, transaction, fir_filing)
- Participants
- File:line provenance
- Natural language caption

Usage:
    from M6_feature.timeline_player import build_timeline
    events = build_timeline(cdr_path, txn_path, fir_dir)
"""

import csv
import re
import logging
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
ACCOUNT_RE = re.compile(r"\bACC(?:\d{5}|_[A-Za-z0-9]{8})\b")
PHONE_RE = re.compile(r"(?:\+?91[\s-]?)?\b[6-9]\d{9}\b")
INCIDENT_RE = re.compile(
    r"\b(?:died|death|deceased|killed|killing|murder(?:ed)?|dead|body|bodies)\b",
    re.IGNORECASE,
)
DATE_PATTERNS = [
    (re.compile(r"\bDate\s*:\s*(\d{2}-\d{2}-\d{4})", re.IGNORECASE), "%d-%m-%Y"),
    (re.compile(r"\bDate\s*:\s*(\d{2}/\d{2}/\d{4})", re.IGNORECASE), "%d/%m/%Y"),
    (re.compile(r"\bDate\s*:\s*(\d{4}-\d{2}-\d{2})", re.IGNORECASE), "%Y-%m-%d"),
]

# Phone -> person name (for captions)
PHONE_TO_NAME = {
    "9876543210": "Ravi Kumar",
    "9123456789": "Sunita Sharma",
    "9988776655": "Mohammed Iqbal",
    "8877665544": "Priya Nair",
    "7766554433": "Deepak Verma",
    "9654321098": "Anjali Singh",
    "9543210987": "Rakesh Yadav",
    "9432109876": "Fatima Begum",
    "9321098765": "Suresh Patil",
    "9210987654": "Kavya Reddy",
}

ACCOUNT_TO_NAME = {
    "ACC00101": "Ravi Kumar",
    "ACC00102": "Sunita Sharma",
    "ACC00103": "Mohammed Iqbal",
    "ACC00104": "Priya Nair",
    "ACC00105": "Deepak Verma",
    "ACC00106": "Anjali Singh",
    "ACC00107": "Rakesh Yadav",
    "ACC00108": "Fatima Begum",
    "ACC00109": "Suresh Patil",
    "ACC00110": "Kavya Reddy",
}


def _entity_match_values(value: object) -> set[str]:
    """Return normalized forms that may appear in UI entities or raw records."""
    if value is None:
        return set()

    raw = str(value).strip()
    if not raw:
        return set()

    lowered = raw.lower()
    values = {lowered}

    # Entity IDs commonly look like phone_9876543210 or account_acc00101.
    if "_" in lowered:
        values.add(lowered.rsplit("_", 1)[-1])

    compact = re.sub(r"[^a-z0-9]", "", lowered)
    if compact:
        values.add(compact)

    digits = re.sub(r"\D", "", raw)
    if digits:
        values.add(digits[-10:] if len(digits) >= 10 else digits)

    return values


def _short_path(filepath: str) -> str:
    """Convert absolute path to relative path from repo root."""
    try:
        return str(Path(filepath).relative_to(REPO_ROOT))
    except ValueError:
        return Path(filepath).name


def _find_case_date(case_text: str) -> tuple[datetime, str, int]:
    """Extract a filing/reference date from uploaded text."""
    lines = case_text.splitlines()
    for line_no, line in enumerate(lines, start=1):
        for pattern, fmt in DATE_PATTERNS:
            match = pattern.search(line)
            if match:
                raw_date = match.group(1)
                try:
                    return datetime.strptime(raw_date, fmt), raw_date, line_no
                except ValueError:
                    continue
    return datetime.now().replace(hour=9, minute=0, second=0, microsecond=0), "uploaded date", 1


def build_uploaded_case_timeline(
    case_id: str,
    case_text: str,
    source_label: str = "uploaded_case.txt",
) -> list[dict]:
    """
    Build timeline events directly from an uploaded case document.

    Ad-hoc uploads are not present in the static CDR/transaction/FIR folders, so
    this fallback gives the Timeline tab visible, source-backed events.
    """
    if not case_text or not case_text.strip():
        return []

    case_date, raw_date, date_line = _find_case_date(case_text)
    source_label = source_label or "uploaded_case.txt"
    events = [
        {
            "t": case_date.strftime("%Y-%m-%dT09:00:00"),
            "type": "fir_filing",
            "from": case_id,
            "to": "",
            "file": source_label,
            "line": date_line,
            "confidence": 0.9,
            "caption": (
                f"09:00 - Uploaded case {case_id} reference date {raw_date}. "
                f"[source: {source_label}:L{date_line}]"
            ),
        }
    ]

    seen_accounts: set[tuple[str, int]] = set()
    seen_phones: set[tuple[str, int]] = set()
    seen_incidents: set[int] = set()
    for line_no, line in enumerate(case_text.splitlines(), start=1):
        clean_line = " ".join(line.split())
        if not clean_line:
            continue

        if INCIDENT_RE.search(line) and line_no not in seen_incidents:
            seen_incidents.add(line_no)
            timestamp = case_date.replace(hour=11, minute=0) + timedelta(minutes=line_no)
            events.append(
                {
                    "t": timestamp.strftime("%Y-%m-%dT%H:%M:%S"),
                    "type": "incident",
                    "from": case_id,
                    "to": "",
                    "file": source_label,
                    "line": line_no,
                    "confidence": 0.75,
                    "caption": (
                        f"{timestamp.strftime('%H:%M')} - Incident reference: "
                        f"{clean_line[:220]} [source: {source_label}:L{line_no}]"
                    ),
                }
            )

        for account in ACCOUNT_RE.findall(line):
            key = (account.upper(), line_no)
            if key in seen_accounts:
                continue
            seen_accounts.add(key)
            timestamp = case_date.replace(hour=9, minute=0) + timedelta(minutes=line_no)
            events.append(
                {
                    "t": timestamp.strftime("%Y-%m-%dT%H:%M:%S"),
                    "type": "transaction",
                    "from": account.upper(),
                    "to": "",
                    "from_name": account.upper(),
                    "to_name": "",
                    "amount": 0,
                    "file": source_label,
                    "line": line_no,
                    "confidence": 0.85,
                    "caption": (
                        f"{timestamp.strftime('%H:%M')} - Account reference {account.upper()}: "
                        f"{clean_line[:180]} [source: {source_label}:L{line_no}]"
                    ),
                }
            )

        for phone in PHONE_RE.findall(line):
            normalized_phone = re.sub(r"\D", "", phone)[-10:]
            key = (normalized_phone, line_no)
            if key in seen_phones:
                continue
            seen_phones.add(key)
            timestamp = case_date.replace(hour=10, minute=0) + timedelta(minutes=line_no)
            events.append(
                {
                    "t": timestamp.strftime("%Y-%m-%dT%H:%M:%S"),
                    "type": "call",
                    "from": normalized_phone,
                    "to": "",
                    "from_name": normalized_phone,
                    "to_name": "",
                    "file": source_label,
                    "line": line_no,
                    "confidence": 0.8,
                    "caption": (
                        f"{timestamp.strftime('%H:%M')} - Phone reference {normalized_phone}: "
                        f"{clean_line[:180]} [source: {source_label}:L{line_no}]"
                    ),
                }
            )

    events.sort(key=lambda e: e.get("t", ""))
    logger.info(f"[Timeline] Built {len(events)} uploaded-document events for case {case_id}")
    return events


def build_timeline(
    cdr_path: str | Path = None,
    txn_path: str | Path = None,
    fir_dir: str | Path = None,
    case_id: str = None,
    case_entities: set[str] = None
) -> list[dict]:
    """
    Build a chronologically sorted event timeline from case data.
    Filters to only include events relevant to the given case_id if provided.

    Args:
        cdr_path: Path to CDR CSV file
        txn_path: Path to transactions CSV file
        fir_dir:  Path to directory containing FIR text files
        case_id:  The active case identifier to filter by
        case_entities: Set of entity IDs associated with the case

    Returns:
        Sorted list of event dicts:
        {
            "t": "2024-01-15T17:26:05",   # ISO timestamp
            "type": "call" | "transaction" | "fir_filing",
            "from": "+91...",              # source entity
            "to": "+91...",                # target entity (if applicable)
            "location": "...",             # location (if available)
            "amount": 50000,              # amount for transactions
            "file": "data/cdrs/cdr.csv",  # source file
            "line": 2,                    # line number in source file
            "caption": "..."              # natural language description
        }
    """
    if cdr_path is None:
        # Keep the timeline on the same source tree as the real M1/M2 graph.
        # The repository also contains demo copies under data/, which can be
        # stale or incomplete when the dashboard is running in real mode.
        real_data_root = REPO_ROOT / "M1" / "data"
        data_root = real_data_root if real_data_root.exists() else REPO_ROOT / "data"
        cdr_path = data_root / "cdrs" / "cdr.csv"
    if txn_path is None:
        data_root = REPO_ROOT / "M1" / "data" if (REPO_ROOT / "M1" / "data").exists() else REPO_ROOT / "data"
        txn_path = data_root / "transactions" / "transactions.csv"
    if fir_dir is None:
        data_root = REPO_ROOT / "M1" / "data" if (REPO_ROOT / "M1" / "data").exists() else REPO_ROOT / "data"
        fir_dir = data_root / "firs"

    cdr_path = Path(cdr_path)
    txn_path = Path(txn_path)
    fir_dir = Path(fir_dir)

    events = []

    # --- CDR events ---
    if cdr_path.exists():
        events.extend(_parse_cdr_events(cdr_path))

    # --- Transaction events ---
    if txn_path.exists():
        events.extend(_parse_txn_events(txn_path))

    # --- FIR events ---
    if fir_dir.exists():
        events.extend(_parse_fir_events(fir_dir))

    # --- Case Filtering ---
    if case_id:
        case_id_normalized = str(case_id).strip().lower()

        entity_set: set[str] = set()
        for value in case_entities or set():
            entity_set.update(_entity_match_values(value))

        filtered_events = []

        for event in events:
            event_type = event.get("type")

            # FIRs are directly associated with a case number.
            if event_type == "fir_filing":
                event_case = str(event.get("from", "")).strip().lower()
                if event_case == case_id_normalized:
                    filtered_events.append(event)
                continue

            # CDR / transaction records do not contain a case ID in the current
            # data contract. Include them only when one of their participants is
            # explicitly known to belong to the active case.
            if not entity_set:
                continue

            involved_values: set[str] = set()
            for field in ("from", "to", "from_name", "to_name"):
                involved_values.update(_entity_match_values(event.get(field, "")))

            if involved_values & entity_set:
                filtered_events.append(event)

        events = filtered_events

    # Sort chronologically
    events.sort(key=lambda e: e.get("t", ""))

    logger.info(f"[Timeline] Built {len(events)} events for case {case_id}")
    return events


def _parse_cdr_events(cdr_path: Path) -> list[dict]:
    """Parse CDR CSV into call events."""
    events = []
    with open(cdr_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=2):
            caller = row.get("caller", "").strip()
            callee = row.get("callee", "").strip()
            timestamp = row.get("timestamp", "").strip()
            duration = row.get("duration_seconds", "0").strip()
            location = row.get("tower_location", "").strip()

            if not timestamp:
                continue

            caller_name = PHONE_TO_NAME.get(caller, caller)
            callee_name = PHONE_TO_NAME.get(callee, callee)

            # Format time for caption
            try:
                ts = datetime.fromisoformat(timestamp)
                time_str = ts.strftime("%H:%M")
                date_str = ts.strftime("%d %b %Y")
            except (ValueError, TypeError):
                time_str = timestamp
                date_str = ""

            caption = (
                f"{time_str} — Call from {caller_name} to {callee_name} "
                f"({duration}s) at {location}. "
                f"[source: {_short_path(str(cdr_path))}:L{i}]"
            )

            events.append({
                "t": timestamp,
                "type": "call",
                "from": caller,
                "to": callee,
                "from_name": caller_name,
                "to_name": callee_name,
                "location": location,
                "duration": int(duration) if duration.isdigit() else 0,
                "file": _short_path(str(cdr_path)),
                "line": i,
                "caption": caption,
            })

    return events


def _parse_txn_events(txn_path: Path) -> list[dict]:
    """Parse transactions CSV into transaction events."""
    events = []
    with open(txn_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=2):
            src = row.get("source_account", "").strip()
            tgt = row.get("target_account", "").strip()
            timestamp = row.get("timestamp", "").strip()
            amount_str = row.get("amount_inr", "0").strip()
            remarks = row.get("remarks", "").strip()

            if not timestamp:
                continue

            try:
                amount = float(amount_str)
            except (ValueError, TypeError):
                amount = 0.0

            src_name = ACCOUNT_TO_NAME.get(src, src)
            tgt_name = ACCOUNT_TO_NAME.get(tgt, tgt)

            try:
                ts = datetime.fromisoformat(timestamp)
                time_str = ts.strftime("%H:%M")
            except (ValueError, TypeError):
                time_str = timestamp

            caption = (
                f"{time_str} — Transaction ₹{amount:,.0f} from {src_name} to {tgt_name}"
                f"{' (' + remarks + ')' if remarks and remarks != 'Unknown' else ''}. "
                f"[source: {_short_path(str(txn_path))}:L{i}]"
            )

            events.append({
                "t": timestamp,
                "type": "transaction",
                "from": src,
                "to": tgt,
                "from_name": src_name,
                "to_name": tgt_name,
                "amount": amount,
                "remarks": remarks,
                "file": _short_path(str(txn_path)),
                "line": i,
                "caption": caption,
            })

    return events


def _parse_fir_events(fir_dir: Path) -> list[dict]:
    """Parse FIR files to extract filing date events."""
    events = []
    date_pattern = re.compile(r"Date:\s*(\d{2}-\d{2}-\d{4})")

    for fir_file in sorted(fir_dir.glob("*.txt")):
        try:
            text = fir_file.read_text(encoding="utf-8", errors="replace")
        except (OSError, IOError):
            continue

        match = date_pattern.search(text)
        if not match:
            continue

        date_str = match.group(1)
        try:
            dt = datetime.strptime(date_str, "%d-%m-%Y")
            iso_ts = dt.strftime("%Y-%m-%dT09:00:00")  # Default to 9 AM
        except ValueError:
            continue

        # Find case number
        case_match = re.search(r"Case No\.?:\s*(FIR\w+)", text, re.IGNORECASE)
        case_no = case_match.group(1) if case_match else fir_file.stem

        caption = (
            f"09:00 — FIR {case_no} filed on {date_str}. "
            f"[source: {_short_path(str(fir_file))}:L{match.start() // 80 + 1}]"
        )

        events.append({
            "t": iso_ts,
            "type": "fir_filing",
            "from": case_no,
            "to": "",
            "file": _short_path(str(fir_file)),
            "line": 3,  # Date is typically on line 3 in our FIR format
            "caption": caption,
        })

    return events


if __name__ == "__main__":
    events = build_timeline()
    print(f"[Timeline] Generated {len(events)} events")
    for e in events[:5]:
        print(f"  {e['t']} [{e['type']}] {e['caption'][:80]}...")
