from __future__ import annotations

import json
import unittest

from prompts import qa_prompts
from prompts.ledger.render import load_template


class LedgerPromptTests(unittest.TestCase):
    def test_all_ledger_system_prompt_templates_exist(self):
        template_names = [
            "router_semantic_system.txt",
            "router_localization_system.txt",
            "entity_mention_extraction_system.txt",
            "entity_resolver_system.txt",
            "query_planner_system.txt",
            "evidence_extractor_system.txt",
            "claim_miner_system.txt",
            "methodology_reviewer_system.txt",
            "citation_reviewer_system.txt",
            "contradiction_reviewer_system.txt",
            "claim_revision_system.txt",
            "review_merge_system.txt",
            "synthesis_system.txt",
        ]

        for template_name in template_names:
            self.assertTrue(load_template(template_name).strip(), template_name)

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

    def test_entity_mention_extraction_system_prompt_renders_allowed_entity_types_from_template(self):
        prompt = qa_prompts.ENTITY_MENTION_EXTRACTION_SYSTEM_PROMPT

        self.assertIn('"catalyst"', prompt)
        self.assertIn('"condition"', prompt)
        self.assertIn("exact contiguous spans", prompt)


if __name__ == "__main__":
    unittest.main()
