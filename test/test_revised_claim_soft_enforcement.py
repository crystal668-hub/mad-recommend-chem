import unittest


class RevisedClaimSoftEnforcementTests(unittest.TestCase):
    def test_restore_metric_when_withheld_na(self):
        from debate.langgraph_coordinator import (
            _AUTO_NOTE_REVISED_CLAIM_METRIC_WITHHELD,
            _normalize_claim_newlines,
            _soft_enforce_revised_claim_metrics,
        )

        prev_claim = (
            "Reaction Type: OER\n"
            "Products: N/A\n"
            "Performance Metrics: 330 mV overpotential at 10 mA/cm^2 (Confidence: low)\n"
            "Evidence: llm\n"
        )
        revised_claim = (
            "Reaction Type: OER\\n"
            "Products: N/A\\n"
            "Performance Metrics: Overpotential at 10 mA/cm^2: N/A (not asserted; composition-specific data not provided).\\n"
            "Rationale: placeholder text."
        )
        revised_claim = _normalize_claim_newlines(revised_claim)

        patched, flags = _soft_enforce_revised_claim_metrics(revised_claim, prev_claim)
        self.assertTrue(flags.get("withheld_detected"))
        self.assertTrue(flags.get("restored_from_prev"))
        self.assertIn(
            "Performance Metrics: 330 mV overpotential at 10 mA/cm^2 (Confidence: low)",
            patched,
        )
        self.assertNotIn("Overpotential at 10 mA/cm^2: N/A", patched)
        self.assertIn(_AUTO_NOTE_REVISED_CLAIM_METRIC_WITHHELD, patched)

        # Idempotent: no repeated note.
        patched2, _flags2 = _soft_enforce_revised_claim_metrics(patched, prev_claim)
        self.assertEqual(patched2.count(_AUTO_NOTE_REVISED_CLAIM_METRIC_WITHHELD), 1)

    def test_insert_metric_when_missing_line(self):
        from debate.langgraph_coordinator import (
            _AUTO_NOTE_REVISED_CLAIM_METRIC_WITHHELD,
            _normalize_claim_newlines,
            _soft_enforce_revised_claim_metrics,
        )

        prev_claim = (
            "Reaction Type: OER\n"
            "Products: N/A\n"
            "Performance Metrics: 310 mV overpotential at 10 mA/cm^2 (Confidence: medium-low)\n"
        )
        revised_claim = "Reaction Type: OER\\nProducts: N/A\\nRationale: no direct match."
        revised_claim = _normalize_claim_newlines(revised_claim)

        patched, flags = _soft_enforce_revised_claim_metrics(revised_claim, prev_claim)
        self.assertTrue(flags.get("withheld_detected"))
        self.assertTrue(flags.get("restored_from_prev"))
        self.assertIn(
            "Performance Metrics: 310 mV overpotential at 10 mA/cm^2 (Confidence: low)",
            patched,
        )
        self.assertIn(_AUTO_NOTE_REVISED_CLAIM_METRIC_WITHHELD, patched)

    def test_no_change_when_numeric_present(self):
        from debate.langgraph_coordinator import _soft_enforce_revised_claim_metrics

        prev_claim = "Performance Metrics: 330 mV overpotential at 10 mA/cm^2 (Confidence: low)"
        revised_claim = (
            "Reaction Type: OER\n"
            "Products: N/A\n"
            "Performance Metrics: 340 mV overpotential at 10 mA/cm^2 (Confidence: low)\n"
            "Rationale: conservative revision."
        )
        patched, flags = _soft_enforce_revised_claim_metrics(revised_claim, prev_claim)
        self.assertFalse(flags.get("withheld_detected"))
        self.assertEqual(patched, revised_claim.strip())

    def test_placeholder_when_prev_missing_metric(self):
        from debate.langgraph_coordinator import (
            _AUTO_NOTE_REVISED_CLAIM_METRIC_WITHHELD,
            _soft_enforce_revised_claim_metrics,
        )

        prev_claim = "Reaction Type: OER\nProducts: N/A\n"
        revised_claim = (
            "Reaction Type: OER\n"
            "Products: N/A\n"
            "Performance Metrics: unknown\n"
            "Rationale: uncertain."
        )
        patched, flags = _soft_enforce_revised_claim_metrics(revised_claim, prev_claim)
        self.assertTrue(flags.get("withheld_detected"))
        self.assertFalse(flags.get("restored_from_prev"))
        self.assertTrue(flags.get("inserted_placeholder"))
        self.assertIn("Performance Metrics: (missing) (Confidence: low)", patched)
        self.assertIn(_AUTO_NOTE_REVISED_CLAIM_METRIC_WITHHELD, patched)


if __name__ == "__main__":
    unittest.main()

