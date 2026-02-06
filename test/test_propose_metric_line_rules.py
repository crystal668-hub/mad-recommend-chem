import unittest


class ProposeMetricLineRulesTests(unittest.TestCase):
    def test_metrics_line_with_plus_minus_is_warning_only(self):
        from agents.react_agent import _validate_conclusion_against_task_with_evidence

        required = ["Ni", "Fe", "Co"]
        conclusion = (
            "Reaction Type: ORR\n"
            "Catalyst metal elements (exactly as provided): Ni, Fe, Co\n"
            "Performance Metrics: E1/2 = 0.86 \u00b1 0.02 V (Confidence: Medium)\n"
        )

        ok, reason = _validate_conclusion_against_task_with_evidence(conclusion, required, trajectory=None)
        self.assertTrue(ok)
        self.assertIn("warning:", (reason or "").lower())
        self.assertIn("performance_metrics_should_be_point_estimate_plus_confidence", (reason or ""))

    def test_range_in_rationale_does_not_trigger_metrics_warning(self):
        from agents.react_agent import _validate_conclusion_against_task_with_evidence

        required = ["Ni", "Fe", "Co"]
        conclusion = (
            "Reaction Type: ORR\n"
            "Catalyst metal elements (exactly as provided): Ni, Fe, Co\n"
            "Performance Metrics: E1/2 = 0.86 V (Confidence: Medium)\n"
            "Rationale: prior reports span 0.83\u20130.88 V under similar conditions.\n"
        )

        ok, reason = _validate_conclusion_against_task_with_evidence(conclusion, required, trajectory=None)
        self.assertTrue(ok)
        self.assertNotIn("performance_metrics_should_be_point_estimate_plus_confidence", (reason or ""))

    def test_metrics_line_missing_confidence_is_warning(self):
        from agents.react_agent import _validate_conclusion_against_task_with_evidence

        required = ["Ni", "Fe", "Co"]
        conclusion = (
            "Reaction Type: ORR\n"
            "Catalyst metal elements (exactly as provided): Ni, Fe, Co\n"
            "Performance Metrics: E1/2 = 0.86 V\n"
        )

        ok, reason = _validate_conclusion_against_task_with_evidence(conclusion, required, trajectory=None)
        self.assertTrue(ok)
        self.assertIn("warning:", (reason or "").lower())
        self.assertIn("missing_confidence_on_performance_metrics_line", (reason or ""))


if __name__ == "__main__":
    unittest.main()

