import json
import unittest


class StrictJsonRepairTests(unittest.TestCase):
    def test_repairs_literal_newlines_inside_json_strings(self):
        from agents.react_agent import _repair_strict_json_text

        bad = """{
  "reaction_type": "OER",
  "electrode_composition": "Co(57.08%), Ni(23.16%)",
  "catalyst_metal_elements": ["Co", "Ni"],
  "products": "N/A",
  "performance_metrics": "320 mV overpotential at 10 mA/cm^2",
  "confidence": "medium",
  "evidence": [{"source_id": "llm"}],
  "rationale": "Line1
Mismatch: composition differs.
Mechanism: strain effect shifts adsorption energetics.
Adjustment: +10 mV due to lower active-site density."
}"""

        with self.assertRaises(json.JSONDecodeError):
            json.loads(bad)

        repaired, reason = _repair_strict_json_text(bad, system_prompt="STRICT JSON proposal schema")
        self.assertIsNotNone(repaired, msg=f"repair failed: {reason}")

        parsed = json.loads(repaired)
        self.assertEqual(parsed.get("reaction_type"), "OER")
        self.assertEqual(parsed.get("performance_metrics"), "320 mV overpotential at 10 mA/cm^2")
        self.assertIn("Mismatch:", parsed.get("rationale", ""))
        self.assertIn("Mechanism:", parsed.get("rationale", ""))
        self.assertIn("Adjustment:", parsed.get("rationale", ""))
        # The repaired JSON text should escape newlines inside the rationale string.
        self.assertIn("\\nMismatch:", repaired)

    def test_valid_json_is_left_parseable(self):
        from agents.react_agent import _repair_strict_json_text

        good = """{
  "reaction_type": "OER",
  "electrode_composition": "Co(57.08%), Ni(23.16%)",
  "catalyst_metal_elements": ["Co", "Ni"],
  "products": "N/A",
  "performance_metrics": "320 mV overpotential at 10 mA/cm^2",
  "confidence": "medium",
  "evidence": [{"source_id": "llm"}],
  "rationale": "Line1\\nMismatch: ok\\nMechanism: ok\\nAdjustment: ok"
}"""

        json.loads(good)  # sanity

        repaired, reason = _repair_strict_json_text(good, system_prompt="STRICT JSON proposal schema")
        self.assertIsNotNone(repaired, msg=f"unexpected repair failure: {reason}")
        parsed = json.loads(repaired)
        self.assertEqual(parsed.get("reaction_type"), "OER")


if __name__ == "__main__":
    unittest.main()

