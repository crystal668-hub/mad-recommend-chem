from __future__ import annotations

import json
import unittest

from prompts import qa_prompts


class LedgerPromptTests(unittest.TestCase):
    def test_router_semantic_system_prompt_includes_contract_and_examples(self):
        prompt = qa_prompts.ROUTER_SEMANTIC_SYSTEM_PROMPT

        self.assertIn("Output contract:", prompt)
        self.assertIn("Correct JSON example:", prompt)
        self.assertIn("Common invalid outputs:", prompt)
        self.assertIn("primary_question_type", prompt)

    def test_query_planner_user_prompt_includes_output_contract(self):
        prompt = qa_prompts.build_query_planner_user_prompt(
            question="How does Pt/C compare with NiMo catalysts for HER activity in alkaline media?",
            task_spec={"question_type": "comparison"},
            entity_pack={"entities": []},
            baseline_plans=[],
        )
        payload = json.loads(prompt)

        self.assertIn("output_contract", payload)
        self.assertIn("example", payload["output_contract"])
        self.assertIn("invalid_examples", payload["output_contract"])
        self.assertEqual(["review", "frontier", "data", "contrarian"], payload["output_contract"]["allowed_lanes"])

    def test_reviewer_user_prompt_includes_output_contract(self):
        prompt = qa_prompts.build_reviewer_user_prompt(
            review_kind="citation",
            task_spec=None,
            claim={"claim_id": "claim-1"},
            evidence_snippets=[],
            focus_flag_types=["Unsupported"],
            allowed_flag_types=["Unsupported", "Weak_Evidence"],
        )
        payload = json.loads(prompt)

        self.assertIn("output_contract", payload)
        self.assertEqual(["Unsupported", "Weak_Evidence"], payload["output_contract"]["allowed_flag_types"])
        self.assertIn("flags", payload["output_contract"]["example"])

    def test_synthesizer_user_prompt_includes_example_contract(self):
        prompt = qa_prompts.build_synthesizer_user_prompt({"claims": []})
        payload = json.loads(prompt)

        self.assertIn("output_contract", payload)
        self.assertIn("final_answer", payload["output_contract"]["example"])
        self.assertIn("sections", payload["output_contract"]["required_top_level_keys"])


if __name__ == "__main__":
    unittest.main()
