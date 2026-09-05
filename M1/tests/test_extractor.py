"""
Tests for M1.3 (spaCy NER), M1.4 (IndicBERT NER), M1.5 (Regex extractors)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from M1.extractor import (
    RawEntity,
    extract_entities_spacy,
    extract_entities_indicbert,
    extract_phones,
    extract_vehicles,
    extract_accounts,
    extract_all,
    _is_hinglish,
    _PHONE_RE,
    _VEHICLE_RE,
)


# ---------------------------------------------------------------------------
# M1.3 — spaCy English NER tests
# ---------------------------------------------------------------------------

def test_spacy_person_extraction():
    text = "Ravi Kumar was arrested by the police at Connaught Place."
    entities = extract_entities_spacy("TEST_001", text)
    persons = [e for e in entities if e.entity_type == "PERSON"]
    assert len(persons) >= 1, "Expected at least one PERSON entity"
    names = [e.text for e in persons]
    assert any("Ravi" in n for n in names), f"Expected 'Ravi' in extracted persons, got {names}"


def test_spacy_location_extraction():
    text = "The accused was seen near Lajpat Nagar, Delhi."
    entities = extract_entities_spacy("TEST_002", text)
    locs = [e for e in entities if e.entity_type == "LOCATION"]
    assert len(locs) >= 1, f"Expected at least one LOCATION, got {[e.text for e in entities]}"


def test_spacy_org_extraction():
    text = "He transferred money via Sunrise Finance Ltd. to another account."
    entities = extract_entities_spacy("TEST_003", text)
    orgs = [e for e in entities if e.entity_type == "ORGANIZATION"]
    assert len(orgs) >= 1, f"Expected at least one ORGANIZATION, got {[e.text for e in entities]}"


def test_spacy_returns_rawentity_type():
    text = "Mohammed Iqbal lives in Mumbai."
    entities = extract_entities_spacy("TEST_004", text)
    for e in entities:
        assert isinstance(e, RawEntity)
        assert e.extraction_method == "spacy"
        assert 0.0 <= e.raw_confidence <= 1.0


def test_spacy_empty_text():
    """Empty text should return empty list, not raise."""
    entities = extract_entities_spacy("TEST_005", "")
    assert entities == []


def test_spacy_no_crash_on_long_text():
    text = "Priya Nair and Deepak Verma met at Bandra West, Mumbai. " * 50
    entities = extract_entities_spacy("TEST_006", text)
    assert isinstance(entities, list)


def test_spacy_precision_on_synthetic_fir(tmp_path):
    """
    Run spaCy on a synthetic FIR and check that PERSON/LOCATION recall
    is reasonable (≥1 entity of each type expected, per FR2 ≥80% precision goal).
    """
    from M1.generate_dataset import generate_all
    generate_all(base_dir=tmp_path)
    fir_dir = tmp_path / "data" / "firs"
    fir_file = sorted(fir_dir.glob("*.txt"))[0]
    text = fir_file.read_text(encoding="utf-8")
    entities = extract_entities_spacy(fir_file.stem, text)
    types_found = {e.entity_type for e in entities}
    # At minimum we expect persons or locations from a FIR
    assert types_found & {"PERSON", "LOCATION"}, \
        f"Expected at least one PERSON or LOCATION. Got: {types_found}"


# ---------------------------------------------------------------------------
# M1.4 — IndicBERT / Hinglish NER tests
# ---------------------------------------------------------------------------

def test_is_hinglish_devanagari():
    assert _is_hinglish("रवि कुमार को गिरफ्तार किया गया।") is True


def test_is_hinglish_romanised():
    text = "Ravi ne bataya ki uska mobile pakda gaya."
    assert _is_hinglish(text) is True


def test_is_hinglish_pure_english():
    text = "The accused was arrested at the police station."
    assert _is_hinglish(text) is False


def test_indicbert_hinglish_sentence():
    """
    Run IndicBERT on a Hinglish sentence. Accept either successful extraction
    or graceful empty list if model is unavailable (offline environment).
    This test must NOT crash either way.
    """
    text = "Ravi Kumar ne bataya ki uska mobile 9876543210 par call aaya tha."
    result = extract_entities_indicbert("TEST_HI_001", text)
    # Either list of RawEntity OR empty list — no crash
    assert isinstance(result, list)
    for e in result:
        assert isinstance(e, RawEntity)
        assert e.extraction_method == "indicbert"


def test_indicbert_skips_pure_english():
    """IndicBERT should return empty list for pure English (not worth running on it)."""
    text = "The accused was arrested at the police station on 15 August 2025."
    result = extract_entities_indicbert("TEST_HI_002", text)
    assert result == []


# ---------------------------------------------------------------------------
# M1.5 — Regex extractor tests
# ---------------------------------------------------------------------------

# --- Phone number tests ---
@pytest.mark.parametrize("text,expected_count", [
    ("Call 9876543210 for details.", 1),
    ("Caller 9876543210 called 9123456789.", 2),
    ("+91-9876543210 is the suspect's number.", 1),
    ("The number 12345 is not a phone.", 0),
    ("Account 9876543210 — wait, that's 10 digits starting with 9.", 1),
])
def test_phone_extraction(text, expected_count):
    result = extract_phones("TEST_PH", text)
    assert len(result) == expected_count, \
        f"Text: '{text}' → expected {expected_count} phones, got {[e.text for e in result]}"


def test_phone_type_and_confidence():
    result = extract_phones("TEST_PH2", "Phone: 9543210987")
    assert len(result) == 1
    assert result[0].entity_type == "PHONE"
    assert result[0].raw_confidence >= 0.95
    assert result[0].extraction_method == "regex"


def test_phone_strips_prefix():
    """The extracted text should be the 10-digit number, not the +91 prefix."""
    result = extract_phones("TEST_PH3", "+91 9876543210")
    assert len(result) == 1
    assert result[0].text == "9876543210"


# --- Vehicle number tests ---
@pytest.mark.parametrize("text,expected_count", [
    ("Vehicle DL01AB1234 was seen at the scene.", 1),
    ("Plates MH12CD5678 and UP32EF9012 were found.", 2),
    ("Random text without vehicle plate.", 0),
    # DL01A1234 is a valid Indian single-letter series plate format (9 chars)
    ("Partial plate DL01A1234 is a single-letter series.", 1),
    # Truly invalid: wrong number of digits in district code
    ("Bogus plate DL1AB1234 is not a plate.", 0),
])
def test_vehicle_extraction(text, expected_count):
    result = extract_vehicles("TEST_VH", text)
    assert len(result) == expected_count, \
        f"Text: '{text}' → expected {expected_count} vehicles, got {[e.text for e in result]}"


def test_vehicle_type_and_confidence():
    result = extract_vehicles("TEST_VH2", "The vehicle RJ14IJ7890 belongs to the accused.")
    assert len(result) == 1
    assert result[0].entity_type == "VEHICLE"
    assert result[0].raw_confidence >= 0.95
    assert result[0].extraction_method == "regex"


# --- Account number tests ---
@pytest.mark.parametrize("text,expected_min", [
    ("Account number ACC00101 is suspicious.", 1),
    ("Transaction from account ACC00103 to ACC00105.", 2),
    ("No account here.", 0),
])
def test_account_extraction_acc_format(text, expected_min):
    result = extract_accounts("TEST_AC", text)
    assert len(result) >= expected_min, \
        f"Text: '{text}' → expected ≥{expected_min} accounts, got {[e.text for e in result]}"


def test_account_type():
    result = extract_accounts("TEST_AC2", "Funds transferred from ACC00108.")
    assert len(result) >= 1
    assert result[0].entity_type == "ACCOUNT"


# ---------------------------------------------------------------------------
# M1.3+1.5 combined: regex takes precedence over NER for phone/vehicle
# ---------------------------------------------------------------------------

def test_extract_all_returns_list():
    text = "Ravi Kumar called 9876543210 from vehicle DL01AB1234 near Delhi."
    result = extract_all("TEST_ALL", text)
    assert isinstance(result, list)
    assert all(isinstance(e, RawEntity) for e in result)


def test_extract_all_has_phone_and_vehicle():
    text = "Ravi Kumar called 9876543210 from vehicle DL01AB1234 near Delhi."
    result = extract_all("TEST_ALL2", text)
    types = {e.entity_type for e in result}
    assert "PHONE" in types
    assert "VEHICLE" in types


def test_extract_all_no_crash_empty():
    result = extract_all("TEST_EMPTY", "")
    assert result == []
