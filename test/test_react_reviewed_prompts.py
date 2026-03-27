from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from prompts.react_reviewed import (
    build_proposer_action_prompt,
    build_proposer_system_prompt,
    build_reviewer_action_prompt,
    build_screening_system_prompt,
)
from prompts.react_reviewed.render import load_template


class ReactReviewedPromptTests(unittest.TestCase):
    def test_all_react_reviewed_templates_exist_and_are_non_empty(self):
        template_names = [
            "proposer_system.yaml",
            "proposer_action.yaml",
            "reviewer_system.yaml",
            "reviewer_action.yaml",
            "screening_system.yaml",
            "proposer_repair_system.yaml",
            "reviewer_repair_system.yaml",
        ]

        for template_name in template_names:
            self.assertTrue(load_template(template_name).strip(), template_name)

    def test_proposer_action_prompt_renders_template_placeholders(self):
        prompt = build_proposer_action_prompt(
            tool_names=("plan_queries", "search_papers", "conclude"),
            retrieval_tools=("plan_queries", "search_papers"),
            proposer_candidate_target=10,
            conclude_contract={
                "tool_call_rule": "Call conclude with exactly {\"submission\": {...}}.",
                "tool_call_example": {"submission": {"submission_id": "submission_cycle_1"}},
                "invalid_examples": [{"payload": {"submission_id": "submission_cycle_1"}}],
            },
        )

        self.assertIn("Allowed tools: plan_queries, search_papers, conclude.", prompt)
        self.assertIn("Treat these as retrieval/inspection tools: plan_queries, search_papers.", prompt)
        self.assertIn("Proposer candidate target: 10 cumulative strict-PDF candidates within the current cycle.", prompt)
        self.assertIn("cycle-level cumulative threshold", prompt)
        self.assertIn('Call conclude with exactly {"submission": {...}}.', prompt)
        self.assertIn('"submission_id": "submission_cycle_1"', prompt)
        self.assertNotIn("Once one search_papers call has produced usable PDF-downloadable candidates", prompt)

    def test_proposer_system_prompt_renders_candidate_target_threshold(self):
        prompt = build_proposer_system_prompt(
            proposer_candidate_target=8,
            conclude_contract={
                "tool_call_rule": "Call conclude with exactly {\"submission\": {...}}.",
                "tool_call_example": {"submission": {"submission_id": "submission_cycle_1"}},
                "invalid_examples": [{"payload": {"submission_id": "submission_cycle_1"}}],
            },
        )

        self.assertIn("Proposer candidate target: 8 cumulative strict-PDF candidates within the current cycle.", prompt)
        self.assertIn("cycle-level cumulative threshold", prompt)
        self.assertNotIn("As soon as search_papers returns a usable PDF-downloadable candidate set", prompt)

    def test_reviewer_action_prompt_renders_template_placeholders(self):
        prompt = build_reviewer_action_prompt(
            tool_names=("inspect_submission_anchor", "conclude"),
            retrieval_budget=2,
            conclude_contract={
                "tool_call_rule": "Call conclude with exactly {\"review\": {\"review_items\": [...]}}.",
                "tool_call_example": {"review": {"review_items": [{"review_id": "review_1"}]}},
                "invalid_examples": [{"review_items": [{"review_id": "review_1"}]}],
            },
        )

        self.assertIn("Allowed tools: inspect_submission_anchor, conclude.", prompt)
        self.assertIn("Charged retrieval budget: 2 cache-miss actions.", prompt)
        self.assertIn('Call conclude with exactly {"review": {"review_items": [...]}}.', prompt)
        self.assertIn('"review_id": "review_1"', prompt)

    def test_screening_system_prompt_renders_yaml_template(self):
        prompt = build_screening_system_prompt(max_candidates=3)

        self.assertIn("Lock at most 3 papers.", prompt)
        self.assertIn("Correct JSON example:", prompt)
        self.assertIn("Common invalid outputs:", prompt)

    def test_load_template_requires_prompt_field(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            template_dir = Path(tmpdir)
            template_path = template_dir / "missing_prompt.yaml"
            template_path.write_text("title: Missing prompt\n", encoding="utf-8")
            with patch("prompts.react_reviewed.render._TEMPLATE_DIR", template_dir):
                load_template.cache_clear()
                with self.assertRaisesRegex(ValueError, "must define 'prompt' as a string"):
                    load_template("missing_prompt.yaml")
                load_template.cache_clear()

    def test_load_template_rejects_empty_prompt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            template_dir = Path(tmpdir)
            template_path = template_dir / "empty_prompt.yaml"
            template_path.write_text("prompt: \"   \"\n", encoding="utf-8")
            with patch("prompts.react_reviewed.render._TEMPLATE_DIR", template_dir):
                load_template.cache_clear()
                with self.assertRaisesRegex(ValueError, "must define a non-empty 'prompt' string"):
                    load_template("empty_prompt.yaml")
                load_template.cache_clear()

    def test_load_template_rejects_invalid_yaml(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            template_dir = Path(tmpdir)
            template_path = template_dir / "invalid.yaml"
            template_path.write_text("prompt: [unclosed\n", encoding="utf-8")
            with patch("prompts.react_reviewed.render._TEMPLATE_DIR", template_dir):
                load_template.cache_clear()
                with self.assertRaisesRegex(ValueError, "is not valid YAML"):
                    load_template("invalid.yaml")
                load_template.cache_clear()


if __name__ == "__main__":
    unittest.main()
