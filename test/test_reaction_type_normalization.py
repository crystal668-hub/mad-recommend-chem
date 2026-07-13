import unittest

from agents.react_agent import _text_mentions_reaction
from utils.reaction_types import canonical_reaction_type, reaction_type_matches, is_supported_reaction_type


class ReactionTypeNormalizationTests(unittest.TestCase):
    def test_old_reaction_acronyms_stay_uppercase(self):
        self.assertEqual(canonical_reaction_type("orr"), "ORR")
        self.assertEqual(canonical_reaction_type("CO2RR"), "CO2RR")

    def test_new_categories_match_chroma_metadata_labels(self):
        self.assertEqual(canonical_reaction_type("Antiferromagnetism"), "antiferromagnetism")
        self.assertEqual(canonical_reaction_type("conductivity"), "conductivity")
        self.assertEqual(canonical_reaction_type("thermal_conductivity"), "thermal conductivity")
        self.assertEqual(
            canonical_reaction_type("photothermal conversion efficiency"),
            "photothermal conversion efficiency",
        )
        self.assertEqual(canonical_reaction_type("antibacterial"), "Antibacterial")
        self.assertEqual(canonical_reaction_type("THERMOELECTRIC"), "Thermoelectric")
        self.assertEqual(
            canonical_reaction_type("photocatalytic_h2o2_production"),
            "photocatalytic H2O2 production",
        )

    def test_matches_ignore_case_spacing_and_underscores(self):
        self.assertTrue(reaction_type_matches("THERMAL_CONDUCTIVITY", "thermal conductivity"))
        self.assertTrue(reaction_type_matches("photothermal_conversion_efficiency", "photothermal conversion efficiency"))
        self.assertTrue(reaction_type_matches("Photocatalytic H2O2 Production", "photocatalytic_h2o2_production"))
        self.assertFalse(reaction_type_matches("conductivity", "thermal conductivity"))

    def test_supported_type_check(self):
        self.assertTrue(is_supported_reaction_type("antiferromagnetism"))
        self.assertTrue(is_supported_reaction_type("ferrimagnetism"))
        self.assertTrue(is_supported_reaction_type("Antibacterial"))
        self.assertTrue(is_supported_reaction_type("thermoelectric"))
        self.assertTrue(is_supported_reaction_type("photocatalytic H2O2 production"))
        self.assertFalse(is_supported_reaction_type("unknown category"))

    def test_new_literature_type_query_terms(self):
        self.assertTrue(_text_mentions_reaction("Strong antimicrobial activity", "Antibacterial"))
        self.assertTrue(_text_mentions_reaction("High thermoelectric figure of merit", "Thermoelectric"))
        self.assertTrue(
            _text_mentions_reaction(
                "Photocatalytic hydrogen peroxide production under visible light",
                "photocatalytic H2O2 production",
            )
        )


if __name__ == "__main__":
    unittest.main()
