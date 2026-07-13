import unittest

from database.literature_types import LITERATURE_TYPE_CONFIGS


class LiteratureTypeConfigsTests(unittest.TestCase):
    def test_contains_all_existing_types_with_csv_metadata(self):
        expected = {
            "CO2RR",
            "EOR",
            "HER",
            "HOR",
            "HZOR",
            "O5H",
            "OER",
            "ORR",
            "UOR",
            "Antibacterial",
            "Thermoelectric",
            "antiferromagnetism",
            "conductivity",
            "ferrimagnetism",
            "ferromagnetism",
            "photocatalytic H2O2 production",
            "photothermal conversion efficiency",
            "thermal conductivity",
        }

        self.assertEqual(set(LITERATURE_TYPE_CONFIGS), expected)
        for config in LITERATURE_TYPE_CONFIGS.values():
            self.assertEqual(set(config), {"path", "metadata_csv"})
            self.assertTrue(config["metadata_csv"].endswith(".csv"))

    def test_new_literature_types_have_expected_paths(self):
        self.assertEqual(
            LITERATURE_TYPE_CONFIGS["Antibacterial"],
            {"path": "Antibacterial", "metadata_csv": "./metadata/Antibacterial.csv"},
        )
        self.assertEqual(
            LITERATURE_TYPE_CONFIGS["Thermoelectric"],
            {"path": "Thermoelectric", "metadata_csv": "./metadata/Thermoelectric.csv"},
        )
        self.assertEqual(
            LITERATURE_TYPE_CONFIGS["photocatalytic H2O2 production"],
            {
                "path": "photocatalytic H2O2 production",
                "metadata_csv": "./metadata/Photocatalytic H2O2 production.csv",
            },
        )


if __name__ == "__main__":
    unittest.main()
