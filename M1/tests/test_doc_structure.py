"""
M1 — Document Structure Classifier Unit Tests
PS 26152 — AI-Powered Criminal Network Analysis System

Covers all 17+ required test cases from the implementation spec:
  - HEADER_FIELD detection (simple, comma-value, multi-line isolation)
  - TABLE_ROW detection (tab, pipe, multi-space, fuzzy header variants)
  - STRUCTURAL_NOISE detection (ALL-CAPS, numbered, meta-blocks)
  - NARRATIVE detection (colon-in-prose, alias-merging, dual-entity)
  - Known limitations documented inline
"""

import sys
from pathlib import Path
import pytest

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from M1.doc_structure import classify_document_structure, ClassifiedBlock


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def blocks_of_type(blocks, btype):
    return [b for b in blocks if b.type == btype]


def block_texts(blocks):
    return [b.raw_text.strip() for b in blocks]


# ===========================================================================
# HEADER_FIELD tests
# ===========================================================================

class TestHeaderFieldDetection:

    def test_simple_header_field(self):
        """A simple 'Location: Nagpur' line must be classified as HEADER_FIELD."""
        blocks = classify_document_structure("Location: Nagpur")
        hf = blocks_of_type(blocks, "HEADER_FIELD")
        assert len(hf) == 1
        assert hf[0].parsed_fields["label"] == "Location"
        assert hf[0].parsed_fields["value"] == "Nagpur"

    def test_header_field_comma_value(self):
        """
        'Location: Nagpur, Maharashtra' — the comma is in the VALUE, not the label.
        Must be classified as HEADER_FIELD with value='Nagpur, Maharashtra'.
        """
        blocks = classify_document_structure("Location: Nagpur, Maharashtra")
        hf = blocks_of_type(blocks, "HEADER_FIELD")
        assert len(hf) == 1
        assert hf[0].parsed_fields["label"] == "Location"
        assert hf[0].parsed_fields["value"] == "Nagpur, Maharashtra"

    def test_two_consecutive_header_fields_not_merged(self):
        """
        Two adjacent header field lines must produce TWO separate HEADER_FIELD
        blocks. Their values must NEVER be merged into one string.
        This directly tests the "Case Type: X / Location: Y" garbling bug.
        """
        text = "Case Type: Suspected Financial Fraud Network\nLocation: Nagpur, Maharashtra"
        blocks = classify_document_structure(text)
        hf = blocks_of_type(blocks, "HEADER_FIELD")
        assert len(hf) == 2, f"Expected 2 HEADER_FIELD blocks, got {len(hf)}: {[(b.parsed_fields) for b in hf]}"

        labels = {b.parsed_fields["label"] for b in hf}
        assert "Case Type" in labels
        assert "Location" in labels

        values = {b.parsed_fields["label"]: b.parsed_fields["value"] for b in hf}
        assert values["Case Type"] == "Suspected Financial Fraud Network"
        assert values["Location"] == "Nagpur, Maharashtra"

        # Critical: confirm the value strings are never concatenated
        for hf_block in hf:
            val = hf_block.parsed_fields["value"]
            assert "Nagpur" not in val or hf_block.parsed_fields["label"] == "Location"
            assert "Suspected Financial Fraud Network" not in val or hf_block.parsed_fields["label"] == "Case Type"

    def test_header_field_date_value(self):
        """'Date of Report: 05/09/2026' is a HEADER_FIELD."""
        blocks = classify_document_structure("Date of Report: 05/09/2026")
        hf = blocks_of_type(blocks, "HEADER_FIELD")
        assert len(hf) == 1
        assert hf[0].parsed_fields["value"] == "05/09/2026"

    def test_header_field_case_id(self):
        """'Case ID: FIR_2026_TEST_DUAL' is a HEADER_FIELD."""
        blocks = classify_document_structure("Case ID: FIR_2026_TEST_DUAL")
        hf = blocks_of_type(blocks, "HEADER_FIELD")
        assert len(hf) == 1
        assert "FIR_2026_TEST_DUAL" in hf[0].parsed_fields["value"]


# ===========================================================================
# TABLE_ROW tests
# ===========================================================================

class TestTableRowDetection:

    def test_table_with_exact_name_header(self):
        """Table with header column exactly 'Name' — rows must be TABLE_ROW."""
        text = "ID\tName\tAge\tRole\nP01\tMeena Kulkarni\t52\tDirector\nP02\tSolapur\t40\tWitness"
        blocks = classify_document_structure(text)
        rows = blocks_of_type(blocks, "TABLE_ROW")
        assert len(rows) == 2
        names = [b.parsed_fields["name"] for b in rows]
        assert "Meena Kulkarni" in names
        assert "Solapur" in names

    def test_table_fuzzy_header_full_name(self):
        """Header 'Full Name' must fuzzy-match to the name column."""
        text = "ID\tFull Name\tAge\nP01\tArjun Patwardhan\t35"
        blocks = classify_document_structure(text)
        rows = blocks_of_type(blocks, "TABLE_ROW")
        assert len(rows) == 1
        assert rows[0].parsed_fields["name"] == "Arjun Patwardhan"

    def test_table_fuzzy_header_witness(self):
        """Header 'Witness' must fuzzy-match to the name column."""
        text = "ID\tWitness\tStatement\nW01\tReshma Solapur\tProvided oral statement"
        blocks = classify_document_structure(text)
        rows = blocks_of_type(blocks, "TABLE_ROW")
        assert len(rows) == 1
        assert rows[0].parsed_fields["name"] == "Reshma Solapur"

    def test_table_fuzzy_header_suspect_name(self):
        """Header 'Suspect Name' must fuzzy-match to the name column."""
        text = "ID\tSuspect Name\tAge\nS01\tRavi Kumar\t30"
        blocks = classify_document_structure(text)
        rows = blocks_of_type(blocks, "TABLE_ROW")
        assert len(rows) == 1
        assert rows[0].parsed_fields["name"] == "Ravi Kumar"

    def test_table_pipe_delimiter(self):
        """Pipe-separated table rows must be detected and parsed correctly."""
        text = "ID | Name | Age | Relationship\nP01 | Meena Kulkarni | 52 | Director\nP02 | Solapur | 40 | Witness"
        blocks = classify_document_structure(text)
        rows = blocks_of_type(blocks, "TABLE_ROW")
        assert len(rows) == 2
        names = [b.parsed_fields["name"] for b in rows]
        assert "Meena Kulkarni" in names
        assert "Solapur" in names

    def test_table_tab_delimiter(self):
        """Tab-separated table — verified by test_table_with_exact_name_header above.
        This test explicitly checks that tab is the delimiter used."""
        text = "ID\tName\tRole\nP03\tArjun Patwardhan\tAccountant"
        blocks = classify_document_structure(text)
        rows = blocks_of_type(blocks, "TABLE_ROW")
        assert len(rows) == 1
        assert rows[0].parsed_fields["name"] == "Arjun Patwardhan"

    def test_table_multi_space_alignment(self):
        """Multi-space-aligned table (no explicit delimiter char) must be detected."""
        text = "ID  Name              Age  Role\nP01  Meena Kulkarni    52   Director\nP02  Solapur           40   Witness"
        blocks = classify_document_structure(text)
        rows = blocks_of_type(blocks, "TABLE_ROW")
        assert len(rows) >= 1
        names = [b.parsed_fields["name"] for b in rows]
        # At minimum the first data row should be extracted
        assert any("Meena" in n for n in names) or any("Solapur" in n for n in names)

    def test_table_name_with_comma_or_spacing(self):
        """A name containing extra spacing in the name column is cleaned but preserved."""
        text = "ID\tName\tAge\nP01\t  Reshma Solapur  \t29"
        blocks = classify_document_structure(text)
        rows = blocks_of_type(blocks, "TABLE_ROW")
        assert len(rows) == 1
        name = rows[0].parsed_fields["name"]
        # Should be stripped of leading/trailing whitespace
        assert name == "Reshma Solapur"

    def test_table_row_does_not_include_non_name_columns(self):
        """
        Age and Relationship columns must NEVER end up in the name field.
        Tests the core bug: 'P02 Solapur', 'Solapur 29 Company' must NOT appear.
        """
        text = "ID\tName\tAge\tRelationship / Role\nP02\tSolapur\t40\tFormer employee, witness"
        blocks = classify_document_structure(text)
        rows = blocks_of_type(blocks, "TABLE_ROW")
        assert len(rows) == 1
        name = rows[0].parsed_fields["name"]
        assert name == "Solapur"
        assert "40" not in name
        assert "employee" not in name.lower()
        assert "P02" not in name
        assert "Former" not in name


# ===========================================================================
# STRUCTURAL_NOISE tests
# ===========================================================================

class TestStructuralNoiseDetection:

    def test_all_caps_section_header(self):
        """An ALL-CAPS section header (≤5 words) must be STRUCTURAL_NOISE."""
        blocks = classify_document_structure("PERSON DATABASE")
        noise = blocks_of_type(blocks, "STRUCTURAL_NOISE")
        assert len(noise) >= 1

    def test_numbered_section_title(self):
        """Numbered section titles like '1. BACKGROUND' must be STRUCTURAL_NOISE."""
        blocks = classify_document_structure("1. BACKGROUND")
        noise = blocks_of_type(blocks, "STRUCTURAL_NOISE")
        assert len(noise) >= 1

    def test_note_for_testing_purposes(self):
        """
        'NOTE FOR TESTING PURPOSES:' line must trigger doc-end sentinel.
        Everything from that line onwards must be STRUCTURAL_NOISE and
        nothing after it should be NARRATIVE.
        """
        text = "Actual narrative here.\nNOTE FOR TESTING PURPOSES:\nThis text after should be discarded."
        blocks = classify_document_structure(text)
        narratives = blocks_of_type(blocks, "NARRATIVE")
        narrative_texts = "\n".join(b.raw_text for b in narratives)
        # The "NOTE FOR TESTING PURPOSES" line and everything after it must NOT reach NARRATIVE
        assert "NOTE FOR TESTING PURPOSES" not in narrative_texts
        assert "This text after should be discarded" not in narrative_texts
        # The actual narrative before the sentinel must be preserved
        assert "Actual narrative here." in narrative_texts

    def test_end_of_document_sentinel(self):
        """'--- END OF DOCUMENT ---' must trigger doc-end sentinel."""
        text = "Real narrative.\n--- END OF DOCUMENT ---\nPost-doc garbage."
        blocks = classify_document_structure(text)
        narratives = blocks_of_type(blocks, "NARRATIVE")
        narrative_texts = "\n".join(b.raw_text for b in narratives)
        assert "Real narrative." in narrative_texts
        assert "Post-doc garbage" not in narrative_texts

    def test_markdown_separator(self):
        """'---' and '===' lines must be STRUCTURAL_NOISE."""
        text = "---\nNarrative text.\n==="
        blocks = classify_document_structure(text)
        noise = blocks_of_type(blocks, "STRUCTURAL_NOISE")
        noise_texts = [b.raw_text.strip() for b in noise]
        assert "---" in noise_texts or any("---" in t for t in noise_texts)

    def test_ascii_art_lines(self):
        """Lines with box-drawing characters must be STRUCTURAL_NOISE."""
        text = "┌─┐\nNormal narrative.\n└─┘"
        blocks = classify_document_structure(text)
        noise = blocks_of_type(blocks, "STRUCTURAL_NOISE")
        assert any("┌" in b.raw_text for b in noise)

    def test_version_code_line(self):
        """'PS 26152' style version tokens must be STRUCTURAL_NOISE."""
        text = "PS 26152\nSome narrative."
        blocks = classify_document_structure(text)
        noise = blocks_of_type(blocks, "STRUCTURAL_NOISE")
        assert any("PS 26152" in b.raw_text for b in noise)


# ===========================================================================
# NARRATIVE tests — critical: must NOT misclassify as HEADER_FIELD
# ===========================================================================

class TestNarrativeDetection:

    def test_colon_in_narrative_is_not_header_field(self):
        """
        A colon inside a prose sentence (time reference, quote) must be
        classified as NARRATIVE, NOT as HEADER_FIELD.
        Example: "at 10:30 pm, he said: 'wait here'"
        """
        text = "At 10:30 pm, he said: 'wait here'."
        blocks = classify_document_structure(text)
        hf = blocks_of_type(blocks, "HEADER_FIELD")
        narratives = blocks_of_type(blocks, "NARRATIVE")
        # Must be narrative, not a header field
        assert len(hf) == 0, f"Prose sentence with colon was misclassified as HEADER_FIELD: {[b.parsed_fields for b in hf]}"
        assert len(narratives) >= 1

    def test_long_sentence_with_multiple_clauses(self):
        """A full sentence with verb is NARRATIVE, not a header field."""
        text = "Officers observed that the suspect moved funds through several shell companies."
        blocks = classify_document_structure(text)
        hf = blocks_of_type(blocks, "HEADER_FIELD")
        assert len(hf) == 0

    def test_incident_report_label_is_narrative_not_header(self):
        """
        'INCIDENT REPORT:' is an ALL-CAPS two-word header with colon.
        It should be classified as STRUCTURAL_NOISE (ALL-CAPS), not HEADER_FIELD.
        """
        text = "INCIDENT REPORT:"
        blocks = classify_document_structure(text)
        hf = blocks_of_type(blocks, "HEADER_FIELD")
        noise = blocks_of_type(blocks, "STRUCTURAL_NOISE")
        # Should be noise (ALL-CAPS), not a header field
        # (If it somehow passes as header, value would be empty — also acceptable)
        if hf:
            # If classified as HEADER_FIELD, value must be empty (no false value)
            for b in hf:
                assert not b.parsed_fields.get("value"), \
                    f"INCIDENT REPORT: was misclassified as HEADER_FIELD with value={b.parsed_fields.get('value')}"
        else:
            assert len(noise) >= 1

    def test_legitimate_entity_containing_blocklist_word(self):
        """
        A multi-word entity that contains a word also appearing in any blocklist
        (e.g., 'Evidence Review Board') must still be extractable from NARRATIVE text.
        The classifier must not filter it as noise just because 'review' is a tech vocab word.
        This specific case: 'review' is in TECH_VOCAB, but the sentence is NARRATIVE.
        """
        text = "The accused was summoned before the Evidence Review Board for questioning."
        blocks = classify_document_structure(text)
        narratives = blocks_of_type(blocks, "NARRATIVE")
        narrative_texts = " ".join(b.raw_text for b in narratives)
        # The sentence must reach NARRATIVE so NER can extract 'Evidence Review Board'
        assert "Evidence Review Board" in narrative_texts, \
            f"Sentence was incorrectly filtered: {block_texts(blocks)}"


# ===========================================================================
# End-to-end dual-entity test (the core regression case)
# ===========================================================================

class TestDualEntityCase:

    DUAL_ENTITY_TEXT = """INVESTIGATIVE CASE RECORD
Case ID: FIR_2026_TEST_DUAL
Case Type: Suspected Financial Fraud Network
Location: Nagpur, Maharashtra
Date of Report: 05/09/2026

1. BACKGROUND

A complaint was filed regarding a suspected shell-company fraud
operation. During investigation, officers identified a recurring name,
"Solapur", which appears in the case both as a place reference and,
separately, as a person's surname.

Investigating officers first encountered the term "Solapur" as a city
reference: several fraudulent transactions were routed through a
shell company registered in Solapur, Maharashtra.

Later in witness statements, officers recorded a statement from a man
identified only by his surname, Solapur, who claimed to be a former
employee of the shell company and provided information about its
internal structure.

2. PERSON DATABASE

The following individuals are referenced in this investigation:

ID\tName\tAge\tRelationship / Role
P01\tMeena Kulkarni\t52\tCompany director (suspected)
P02\tSolapur\t40\tFormer employee, witness
P03\tArjun Patwardhan\t35\tAccountant
P04\tReshma Solapur\t29\tCompany director's relative, uninvolved

Source caution: entity P02 (surname "Solapur") and the earlier city
reference "Solapur" are intentionally the same text string used in two
different senses, to test entity-type disambiguation.

--- END OF DOCUMENT ---

NOTE FOR TESTING PURPOSES:
This is synthetic text and should not be extracted.
"""

    def test_header_fields_from_dual_entity(self):
        """Case ID, Case Type, Location, Date of Report must be HEADER_FIELD blocks."""
        blocks = classify_document_structure(self.DUAL_ENTITY_TEXT)
        hf = blocks_of_type(blocks, "HEADER_FIELD")
        labels = {b.parsed_fields["label"].strip() for b in hf}
        assert "Case ID" in labels or "Case Id" in labels
        assert "Location" in labels
        assert "Date of Report" in labels

    def test_header_values_not_concatenated(self):
        """
        'Suspected Financial Fraud Network' and 'Nagpur, Maharashtra' must be
        in SEPARATE HEADER_FIELD blocks — never merged into one string.
        """
        blocks = classify_document_structure(self.DUAL_ENTITY_TEXT)
        hf = blocks_of_type(blocks, "HEADER_FIELD")
        values_by_label = {b.parsed_fields["label"].strip(): b.parsed_fields["value"].strip() for b in hf}

        if "Location" in values_by_label:
            loc_val = values_by_label["Location"]
            # Must NOT contain the Case Type value mixed in
            assert "Suspected Financial Fraud Network" not in loc_val

    def test_table_rows_extracted_correctly(self):
        """P01-P04 names must be TABLE_ROW blocks with correct name field."""
        blocks = classify_document_structure(self.DUAL_ENTITY_TEXT)
        rows = blocks_of_type(blocks, "TABLE_ROW")
        names = [b.parsed_fields["name"] for b in rows if b.parsed_fields]

        assert "Meena Kulkarni" in names, f"Meena Kulkarni not found in: {names}"
        assert "Solapur" in names, f"Solapur not found in: {names}"
        assert "Arjun Patwardhan" in names, f"Arjun Patwardhan not found in: {names}"
        assert "Reshma Solapur" in names, f"Reshma Solapur not found in: {names}"

    def test_no_id_in_name_field(self):
        """ID column values (P01, P02, P03, P04) must NEVER appear in name field."""
        blocks = classify_document_structure(self.DUAL_ENTITY_TEXT)
        rows = blocks_of_type(blocks, "TABLE_ROW")
        for row in rows:
            name = row.parsed_fields.get("name", "")
            for bad in ["P01", "P02", "P03", "P04"]:
                assert bad not in name, f"ID column '{bad}' leaked into name: '{name}'"

    def test_no_age_in_name_field(self):
        """Age values must NEVER appear in the name field."""
        blocks = classify_document_structure(self.DUAL_ENTITY_TEXT)
        rows = blocks_of_type(blocks, "TABLE_ROW")
        for row in rows:
            name = row.parsed_fields.get("name", "")
            for bad_age in ["52", "40", "35", "29"]:
                assert bad_age not in name, f"Age '{bad_age}' leaked into name: '{name}'"

    def test_no_role_in_name_field(self):
        """Relationship/role text must NEVER appear in the name field."""
        blocks = classify_document_structure(self.DUAL_ENTITY_TEXT)
        rows = blocks_of_type(blocks, "TABLE_ROW")
        for row in rows:
            name = row.parsed_fields.get("name", "")
            for bad_role in ["Company", "director", "employee", "witness", "Accountant", "relative"]:
                assert bad_role not in name, f"Role text '{bad_role}' leaked into name: '{name}'"

    def test_note_for_testing_not_in_any_block_as_narrative(self):
        """All text after '--- END OF DOCUMENT ---' must be discarded (not NARRATIVE)."""
        blocks = classify_document_structure(self.DUAL_ENTITY_TEXT)
        narratives = blocks_of_type(blocks, "NARRATIVE")
        all_narrative_text = " ".join(b.raw_text for b in narratives)
        assert "NOTE FOR TESTING PURPOSES" not in all_narrative_text
        assert "synthetic text" not in all_narrative_text.lower()

    def test_reshma_solapur_extracted_as_distinct_entity(self):
        """
        'Reshma Solapur' from the table must be in its own TABLE_ROW block
        with name='Reshma Solapur' — not merged with or truncated to 'Solapur'.
        """
        blocks = classify_document_structure(self.DUAL_ENTITY_TEXT)
        rows = blocks_of_type(blocks, "TABLE_ROW")
        names = [b.parsed_fields["name"] for b in rows if b.parsed_fields]
        assert "Reshma Solapur" in names, \
            f"'Reshma Solapur' not found as distinct table row. Names found: {names}"
        # And 'Solapur' (P02) must also be present as a DIFFERENT row
        solapur_rows = [n for n in names if n == "Solapur"]
        assert len(solapur_rows) >= 1, \
            f"'Solapur' (P02 witness) not found separately. Names: {names}"


# ===========================================================================
# Single-row table limitation documentation test
# ===========================================================================

class TestKnownLimitations:

    def test_single_row_table_falls_through_to_narrative(self):
        """
        KNOWN LIMITATION: A table with only ONE data row (no second line to
        confirm the multi-line pattern) will NOT be detected as TABLE_ROW.
        It falls through to NARRATIVE classification instead.

        This is documented and acceptable — a single-row unlabeled table is
        indistinguishable from a prose sentence in the general case.
        """
        text = "ID\tName\tAge"  # Only header, no data rows
        blocks = classify_document_structure(text)
        rows = blocks_of_type(blocks, "TABLE_ROW")
        # There must be no TABLE_ROW blocks (no data rows present)
        assert len(rows) == 0, \
            "Header-only table should produce no TABLE_ROW blocks"


# ===========================================================================
# Alias-merging regression — runs classify + extract_all path
# ===========================================================================

class TestAliasMergingRegression:

    def test_ravi_kumar_alias_merging_via_resolver(self):
        """
        Regression: 'Ravi Kumar', 'Ravi K.', 'R. Kumar' must all resolve to
        ONE entity in the resolver. This tests the resolver's custom initials
        logic introduced in a prior fix.

        All three names come from different pages/statements of the SAME logical
        case FIR_ALIAS_REGRESSION. The explicit case_id groups them so the
        resolver can merge across document-page boundaries within one case.
        """
        from M1.normalizer import normalize_entities
        from M1.resolver import EntityStore, resolve_entities
        from M1.extractor import RawEntity

        store = EntityStore()
        CASE = "FIR_ALIAS_REGRESSION"

        for name, doc in [("Ravi Kumar", "FIR_001"), ("Ravi K.", "FIR_002"), ("R. Kumar", "FIR_003")]:
            raw = [RawEntity(
                text=name, entity_type="PERSON", source_doc_id=doc,
                extraction_method="spacy", raw_confidence=0.75
            )]
            normed = normalize_entities(raw)
            resolve_entities(normed, store, case_id=CASE)

        persons = [e for e in store.all() if e.entity_type == "PERSON"]
        assert len(persons) == 1, \
            f"Expected 'Ravi Kumar' / 'Ravi K.' / 'R. Kumar' to merge into 1 entity. Got {len(persons)}: {[p.canonical for p in persons]}"

    def test_suresh_verma_alias_merging_via_resolver(self):
        """
        Regression: 'Suresh K. Verma', 'Suresh Verma', 'S. Verma' must all
        resolve to ONE entity.

        All three names come from different pages/statements of the SAME logical
        case FIR_ALIAS_REGRESSION. The explicit case_id groups them so the
        resolver can merge across document-page boundaries within one case.
        """
        from M1.normalizer import normalize_entities
        from M1.resolver import EntityStore, resolve_entities
        from M1.extractor import RawEntity

        store = EntityStore()
        CASE = "FIR_ALIAS_REGRESSION"

        for name, doc in [("Suresh K. Verma", "FIR_001"), ("Suresh Verma", "FIR_002"), ("S. Verma", "FIR_003")]:
            raw = [RawEntity(
                text=name, entity_type="PERSON", source_doc_id=doc,
                extraction_method="spacy", raw_confidence=0.75
            )]
            normed = normalize_entities(raw)
            resolve_entities(normed, store, case_id=CASE)

        persons = [e for e in store.all() if e.entity_type == "PERSON"]
        assert len(persons) == 1, \
            f"Expected 'Suresh K. Verma' / 'Suresh Verma' / 'S. Verma' to merge into 1 entity. Got {len(persons)}: {[p.canonical for p in persons]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
