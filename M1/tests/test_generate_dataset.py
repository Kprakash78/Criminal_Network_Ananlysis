"""
Tests for M1.1 — Synthetic Dataset Generator
"""
import csv
import sys
import os
from pathlib import Path

# Allow running from either project root or M1/ directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from M1.generate_dataset import generate_all, PERSONS, LOCATIONS, ORGANIZATIONS


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    base = tmp_path_factory.mktemp("dataset")
    result = generate_all(base_dir=base)
    return result, base


def test_fir_count(dataset):
    result, base = dataset
    firs = list((base / "data" / "firs").glob("*.txt"))
    assert len(firs) >= 30, f"Expected ≥30 FIRs, got {len(firs)}"


def test_fir_nonempty(dataset):
    result, base = dataset
    for fir_path in (base / "data" / "firs").glob("*.txt"):
        content = fir_path.read_text(encoding="utf-8")
        assert len(content.strip()) > 50, f"{fir_path.name} is nearly empty"


def test_cdr_count(dataset):
    result, base = dataset
    with open(base / "data" / "cdrs" / "cdr.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 100, f"Expected ≥100 CDR rows, got {len(rows)}"


def test_cdr_schema(dataset):
    result, base = dataset
    with open(base / "data" / "cdrs" / "cdr.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_cols = {"caller", "callee", "timestamp", "duration_seconds", "tower_location"}
        assert required_cols.issubset(set(reader.fieldnames)), \
            f"Missing CDR columns. Got: {reader.fieldnames}"
        rows = list(reader)
    assert all(row["caller"] != row["callee"] for row in rows), \
        "Self-calls found in CDR — caller and callee should differ"


def test_transaction_count(dataset):
    result, base = dataset
    with open(base / "data" / "transactions" / "transactions.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 50, f"Expected ≥50 transaction rows, got {len(rows)}"


def test_transaction_schema(dataset):
    result, base = dataset
    with open(base / "data" / "transactions" / "transactions.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_cols = {"source_account", "target_account", "amount_inr", "timestamp", "remarks"}
        assert required_cols.issubset(set(reader.fieldnames)), \
            f"Missing transaction columns. Got: {reader.fieldnames}"


def test_entity_overlap_across_firs(dataset):
    """
    Verify that the same canonical person name or one of their aliases
    appears in multiple FIR files — this is the overlap the resolver relies on.
    """
    result, base = dataset
    fir_dir = base / "data" / "firs"
    firs = sorted(fir_dir.glob("*.txt"))

    # Build index: name_fragment -> set of FIRs containing it
    person_firs: dict[str, set] = {}
    all_name_forms = []
    for p in PERSONS:
        all_name_forms.append((p[0], p[0]))          # canonical
        for alias in p[1]:
            all_name_forms.append((p[0], alias))     # alias

    for fir_path in firs:
        content = fir_path.read_text(encoding="utf-8")
        for canonical, name_form in all_name_forms:
            if name_form in content:
                person_firs.setdefault(canonical, set()).add(fir_path.name)

    multi_doc_persons = {k: v for k, v in person_firs.items() if len(v) >= 2}
    assert len(multi_doc_persons) >= 3, (
        f"Expected ≥3 persons appearing in ≥2 FIRs (overlap for resolver). "
        f"Got: {multi_doc_persons}"
    )


def test_hinglish_present(dataset):
    """
    Verify that Hinglish sentences (code-mixed Hindi) are present in FIR files.
    We detect them by looking for common Hindi words that appear in the templates.
    """
    result, base = dataset
    fir_dir = base / "data" / "firs"
    hindi_markers = ["bataya", "mobile", "gaadi", "pakda", "giraftaar", "account", "baar", "paas", "uska", "baith"]
    hinglish_firs = []
    for fir_path in fir_dir.glob("*.txt"):
        content = fir_path.read_text(encoding="utf-8")
        if any(marker in content for marker in hindi_markers):
            hinglish_firs.append(fir_path.name)

    assert len(hinglish_firs) >= 5, (
        f"Expected ≥5 FIRs with Hinglish content, got {len(hinglish_firs)}"
    )


def test_phone_numbers_in_firs(dataset):
    """Verify that Indian phone numbers (10 digit) appear in FIR text."""
    import re
    result, base = dataset
    phone_pattern = re.compile(r"\b[6-9]\d{9}\b")
    firs_with_phones = []
    for fir_path in (base / "data" / "firs").glob("*.txt"):
        content = fir_path.read_text(encoding="utf-8")
        if phone_pattern.search(content):
            firs_with_phones.append(fir_path.name)
    assert len(firs_with_phones) >= 10, (
        f"Expected ≥10 FIRs containing phone numbers, got {len(firs_with_phones)}"
    )


def test_vehicle_numbers_in_firs(dataset):
    """Verify Indian vehicle plate patterns appear in FIR text."""
    import re
    result, base = dataset
    # Indian format: XX00XX0000 (2 letters, 2 digits, 2 letters, 4 digits)
    vehicle_pattern = re.compile(r"\b[A-Z]{2}\d{2}[A-Z]{2}\d{4}\b")
    firs_with_vehicles = []
    for fir_path in (base / "data" / "firs").glob("*.txt"):
        content = fir_path.read_text(encoding="utf-8")
        if vehicle_pattern.search(content):
            firs_with_vehicles.append(fir_path.name)
    assert len(firs_with_vehicles) >= 5, (
        f"Expected ≥5 FIRs containing vehicle numbers, got {len(firs_with_vehicles)}"
    )


def test_communication_spike_planted(dataset):
    """
    The CDR generator plants at least one detectable communication spike —
    some phone number has ≥5 call events within a single 24-hour window.
    (This is the signal M3's COMMUNICATION_SPIKE rule relies on.)
    """
    from collections import defaultdict
    from datetime import datetime, timedelta
    result, base = dataset
    calls: dict[str, list] = defaultdict(list)
    with open(base / "data" / "cdrs" / "cdr.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ts = datetime.strptime(row["timestamp"].strip(), "%Y-%m-%dT%H:%M:%S")
            calls[row["caller"].strip()].append(ts)
            calls[row["callee"].strip()].append(ts)

    window = timedelta(hours=24)
    spikes = []
    for phone, timestamps in calls.items():
        sorted_ts = sorted(timestamps)
        for i, t0 in enumerate(sorted_ts):
            in_window = sum(1 for t in sorted_ts[i:] if t <= t0 + window)
            if in_window >= 5:
                spikes.append((phone, in_window))
                break

    assert spikes, (
        "No planted communication spike found — expected at least one phone "
        "with ≥5 calls within a 24-hour window"
    )


def test_reproducibility(tmp_path):
    """Same seed must produce identical output across two runs."""
    r1 = generate_all(base_dir=tmp_path / "run1")
    r2 = generate_all(base_dir=tmp_path / "run2")

    # Read first FIR text from both runs
    fir1_content = sorted((Path(r1["fir_dir"])).glob("*.txt"))[0].read_text()
    fir2_content = sorted((Path(r2["fir_dir"])).glob("*.txt"))[0].read_text()
    assert fir1_content == fir2_content, "Dataset generation is not reproducible (seed not applied correctly)"
