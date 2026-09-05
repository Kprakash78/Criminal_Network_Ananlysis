import unittest
from M1.preprocessor import strip_structural_noise

class TestStructuralNoisePreprocessor(unittest.TestCase):
    def test_all_caps_headers(self):
        text = "EVIDENCE DATABASE\nSome real narrative here.\nPERSON DATABASE\nMore narrative."
        cleaned = strip_structural_noise(text)
        self.assertNotIn("EVIDENCE DATABASE", cleaned)
        self.assertNotIn("PERSON DATABASE", cleaned)
        self.assertIn("Some real narrative here.", cleaned)
        
    def test_numbered_section_headers(self):
        text = "1. INITIAL FIR\nNarrative begins.\n2. PERSON DATABASE\nMore text."
        cleaned = strip_structural_noise(text)
        self.assertNotIn("1. INITIAL FIR", cleaned)
        self.assertNotIn("2. PERSON DATABASE", cleaned)
        self.assertIn("Narrative begins.", cleaned)
        
    def test_table_field_labels(self):
        text = "Entity type: PERSON\nInvestigative relevance: HIGH\nCase ID: FIR123\nThis is a completely normal sentence: it continues."
        cleaned = strip_structural_noise(text)
        self.assertNotIn("Entity type:", cleaned)
        self.assertNotIn("Investigative relevance:", cleaned)
        self.assertNotIn("Case ID:", cleaned)
        self.assertIn("PERSON", cleaned)
        self.assertIn("HIGH", cleaned)
        self.assertIn("FIR123", cleaned)
        self.assertIn("This is a completely normal sentence: it continues.", cleaned)
        
    def test_markdown_separators(self):
        text = "---\nNarrative\n===\nMore\n⸻"
        cleaned = strip_structural_noise(text)
        self.assertNotIn("---", cleaned)
        self.assertNotIn("===", cleaned)
        self.assertNotIn("⸻", cleaned)
        self.assertIn("Narrative", cleaned)
        
    def test_ascii_art(self):
        text = "A → B\nNormal sentence here.\n┌─┐\n│ │\n└─┘"
        cleaned = strip_structural_noise(text)
        self.assertNotIn("A → B", cleaned)
        self.assertNotIn("┌─┐", cleaned)
        self.assertIn("Normal sentence here.", cleaned)
        
    def test_arrow_notation(self):
        text = "Upload -> Extract\nNormal sentence => continues."
        cleaned = strip_structural_noise(text)
        self.assertNotIn("Upload -> Extract", cleaned)
        self.assertNotIn("Normal sentence => continues.", cleaned)
        
    def test_legitimate_entity_with_structural_word(self):
        # This tests that the preprocessor doesn't indiscriminately drop lines
        # containing words like 'Case' or 'Group' if they don't match the label rules.
        text = "The Ramesh Chander Case Group held a meeting.\nWe reviewed the Annual Report Center."
        cleaned = strip_structural_noise(text)
        self.assertIn("The Ramesh Chander Case Group held a meeting.", cleaned)
        self.assertIn("We reviewed the Annual Report Center.", cleaned)
        
    def test_table_parsing(self):
        text = "ID\tName\tAge\tRelationship\nP01\tMeena Kulkarni\t52\tDirector\nP02\tSolapur\t40\tWitness"
        cleaned = strip_structural_noise(text)
        self.assertIn("99. Meena Kulkarni", cleaned)
        self.assertIn("99. Solapur", cleaned)
        self.assertNotIn("Director", cleaned)
        self.assertNotIn("Witness", cleaned)
        
    def test_title_case_headings(self):
        text = "Entity Search\nM2 Knowledge Graph\nRAG Pipeline\nJohn Smith was seen today.\nThe system uses M3 for Analysis."
        cleaned = strip_structural_noise(text)
        self.assertNotIn("Entity Search", cleaned)
        self.assertNotIn("M2 Knowledge Graph", cleaned)
        self.assertNotIn("RAG Pipeline", cleaned)
        self.assertIn("John Smith was seen today.", cleaned)
        self.assertIn("The system uses M3 for Analysis.", cleaned)
        
    def test_version_codes(self):
        text = "PS 26152\nVersion v2.0 is out."
        cleaned = strip_structural_noise(text)
        self.assertNotIn("PS 26152", cleaned)
        self.assertIn("Version v2.0 is out.", cleaned)
        
    def test_keep_legitimate_narrative(self):
        text = "The evidence database was accessed by the person database administrator.\nAn initial FIR was filed."
        cleaned = strip_structural_noise(text)
        self.assertIn("The evidence database was accessed by the person database administrator.", cleaned)
        self.assertIn("An initial FIR was filed.", cleaned)

if __name__ == "__main__":
    unittest.main()
