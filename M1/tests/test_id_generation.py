import unittest
from M1.resolver import _make_entity_id

class TestEntityIdGeneration(unittest.TestCase):
    def test_different_types_different_ids(self):
        # Even with identical names, different types MUST produce different IDs
        loc_id = _make_entity_id("LOCATION", "Solapur", case_prefix="")
        per_id = _make_entity_id("PERSON", "Solapur", case_prefix="")
        
        self.assertNotEqual(loc_id, per_id, "IDs must differ for different entity types")
        self.assertTrue(loc_id.startswith("LOC_"))
        self.assertTrue(per_id.startswith("PER_"))
        
        # Test hash suffix differs
        loc_hash = loc_id.split("_")[-1]
        per_hash = per_id.split("_")[-1]
        self.assertNotEqual(loc_hash, per_hash, "Hash suffixes must differ to prevent collision")

if __name__ == '__main__':
    unittest.main()
