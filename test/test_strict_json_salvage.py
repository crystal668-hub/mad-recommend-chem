import json
import unittest


class StrictJsonSalvageTests(unittest.TestCase):
    def test_salvage_proposal_from_plain_text_metric(self):
        from agents.react_agent import _salvage_invalid_strict_json_payload

        system_prompt = 'STRICT JSON schema: {"reaction_type": "OER", "electrode_composition": "X"}'
        full_query = (
            "Please propose...\n"
            "Target reaction: OER\n"
            "Electrode composition (relative %): Co(57.08%), Ni(23.16%), Zn(14.97%), Fe(2.92%), Cu(1.87%)\n"
        )
        text = "327 mV overpotential at 10 mA/cm^2\nMore text that should be ignored."
        out, reason = _salvage_invalid_strict_json_payload(
            text,
            system_prompt=system_prompt,
            full_query=full_query,
            task_reaction="OER",
            task_components=["Co", "Ni", "Zn", "Fe", "Cu"],
        )
        self.assertIsNotNone(out, msg=f"salvage failed: {reason}")
        parsed = json.loads(out)
        self.assertEqual(parsed.get("reaction_type"), "OER")
        self.assertIn("Co(57.08%)", parsed.get("electrode_composition", ""))
        self.assertEqual(parsed.get("performance_metrics"), "327 mV overpotential at 10 mA/cm^2")
        self.assertEqual(parsed.get("confidence"), "low")
        self.assertEqual(parsed.get("evidence"), [{"source_id": "llm"}])

    def test_salvage_reviews_from_plain_text_critique(self):
        from agents.react_agent import _salvage_invalid_strict_json_payload

        system_prompt = 'STRICT JSON schema: {"reviews": []}'
        full_query = (
            "REVIEW phase...\n"
            "--- TARGET PROPOSAL ---\n"
            "target_proposal_id: agent4\n"
            "trajectory_steps:\n"
            "- step_number=1 action=search_literature\n"
            "- step_number=3 action=conclude\n"
        )
        text = "The proposal copies a metric without justifying composition mismatch."
        out, reason = _salvage_invalid_strict_json_payload(
            text,
            system_prompt=system_prompt,
            full_query=full_query,
            task_reaction="OER",
            task_components=["Co", "Ni"],
        )
        self.assertIsNotNone(out, msg=f"salvage failed: {reason}")
        parsed = json.loads(out)
        self.assertIsInstance(parsed.get("reviews"), list)
        self.assertEqual(len(parsed["reviews"]), 1)
        item = parsed["reviews"][0]
        self.assertEqual(item.get("target_proposal_id"), "agent4")
        self.assertEqual(item.get("target_step_number"), 3)
        self.assertEqual(item.get("flaw_type"), "other")
        self.assertTrue(str(item.get("critique") or "").startswith("AUTO-SALVAGE:"))
        self.assertEqual(item.get("evidence"), [{"source_id": "llm"}])

    def test_salvage_rebuttals_from_plain_text_response(self):
        from agents.react_agent import _salvage_invalid_strict_json_payload

        system_prompt = 'STRICT JSON schema: {"rebuttals": []}'
        full_query = (
            "REBUTTAL phase...\n"
            "--- REVIEWS AGAINST YOU (valid only) ---\n"
            "- review_id=rev_r1_agent2_0 from=agent2 target_step=4 flaw_type=wrong_inference\n"
            "- review_id=rev_r1_agent4_0 from=agent4 target_step=3 flaw_type=missing_evidence\n"
        )
        text = "I accept the critique and will revise to a verifiable benchmark."
        out, reason = _salvage_invalid_strict_json_payload(
            text,
            system_prompt=system_prompt,
            full_query=full_query,
            task_reaction="OER",
            task_components=["Co"],
        )
        self.assertIsNotNone(out, msg=f"salvage failed: {reason}")
        parsed = json.loads(out)
        self.assertIsInstance(parsed.get("rebuttals"), list)
        self.assertEqual(len(parsed["rebuttals"]), 2)
        ids = [r.get("target_review_id") for r in parsed["rebuttals"]]
        self.assertEqual(ids, ["rev_r1_agent2_0", "rev_r1_agent4_0"])
        for r in parsed["rebuttals"]:
            self.assertIn(r.get("response_mode"), {"defend", "no_response"})
            self.assertTrue(str(r.get("response") or "").startswith("AUTO-SALVAGE:"))
            self.assertEqual(r.get("evidence"), [{"source_id": "llm"}])
        self.assertIsNone(parsed.get("revised_claim"))

    def test_salvage_returns_none_on_empty_text(self):
        from agents.react_agent import _salvage_invalid_strict_json_payload

        out, reason = _salvage_invalid_strict_json_payload(
            "",
            system_prompt='{"reviews": []}',
            full_query="",
            task_reaction=None,
            task_components=[],
        )
        self.assertIsNone(out)
        self.assertEqual(reason, "empty")


if __name__ == "__main__":
    unittest.main()

