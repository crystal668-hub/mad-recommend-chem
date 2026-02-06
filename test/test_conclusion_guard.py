import unittest


class ConclusionGuardTests(unittest.TestCase):
    def test_all_required_elements_with_extra_element_is_allowed(self):
        from agents.react_agent import _validate_conclusion_against_task_with_evidence

        required = ["Ni", "Fe", "Co"]
        conclusion = (
            "Reaction Type: ORR\n"
            "Catalyst metal elements (exactly as provided): Ni, Fe, Co\n"
            "Benchmark: Pt/C (commercial) is often used for comparison.\n"
        )

        ok, reason = _validate_conclusion_against_task_with_evidence(conclusion, required, trajectory=None)
        self.assertTrue(ok)
        # Extra elements should be warning-only (or empty), not a blocker.
        self.assertTrue((reason or "").strip() == "" or "warning:" in (reason or "").lower())

    def test_missing_required_element_blocks(self):
        from agents.react_agent import _validate_conclusion_against_task

        required = ["Ni", "Fe", "Co"]
        conclusion = "Reaction Type: ORR\nCatalyst metal elements: Ni, Fe\n"

        ok, reason = _validate_conclusion_against_task(conclusion, required)
        self.assertFalse(ok)
        self.assertIn("missing required", (reason or "").lower())


if __name__ == "__main__":
    unittest.main()

