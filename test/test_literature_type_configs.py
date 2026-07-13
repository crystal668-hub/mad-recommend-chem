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
            "antiferromagnetism",
            "conductivity",
            "ferrimagnetism",
            "ferromagnetism",
            "photothermal conversion efficiency",
            "thermal conductivity",
        }

        self.assertEqual(set(LITERATURE_TYPE_CONFIGS), expected)
        for config in LITERATURE_TYPE_CONFIGS.values():
            self.assertEqual(set(config), {"path", "metadata_csv"})
            self.assertTrue(config["metadata_csv"].endswith(".csv"))


if __name__ == "__main__":
    unittest.main()
