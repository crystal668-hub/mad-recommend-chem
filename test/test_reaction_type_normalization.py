import unittest

from utils.reaction_types import canonical_reaction_type, reaction_type_matches, is_supported_reaction_type


class ReactionTypeNormalizationTests(unittest.TestCase):
    def test_old_reaction_acronyms_stay_uppercase(self):
        self.assertEqual(canonical_reaction_type("orr"), "ORR")
        self.assertEqual(canonical_reaction_type("CO2RR"), "CO2RR")

    def test_new_categories_match_chroma_metadata_labels(self):
        self.assertEqual(canonical_reaction_type("conductivity"), "conductivity")
        self.assertEqual(canonical_reaction_type("thermal_conductivity"), "thermal conductivity")
        self.assertEqual(
            canonical_reaction_type("photothermal conversion efficiency"),
            "photothermal conversion efficiency",
        )

    def test_matches_ignore_case_spacing_and_underscores(self):
        self.assertTrue(reaction_type_matches("THERMAL_CONDUCTIVITY", "thermal conductivity"))
        self.assertTrue(reaction_type_matches("photothermal_conversion_efficiency", "photothermal conversion efficiency"))
        self.assertFalse(reaction_type_matches("conductivity", "thermal conductivity"))

    def test_supported_type_check(self):
        self.assertTrue(is_supported_reaction_type("ferrimagnetism"))
        self.assertFalse(is_supported_reaction_type("unknown category"))


if __name__ == "__main__":
    unittest.main()
