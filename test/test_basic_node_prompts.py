from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from prompts.basic_node import (
    build_entity_mention_extraction_system_prompt,
    build_query_planner_system_prompt,
)
from prompts.basic_node.render import load_template


class BasicNodePromptTests(unittest.TestCase):
    def test_all_basic_node_templates_exist_and_are_non_empty(self):
        template_names = [
            "claim_miner_system.yaml",
            "entity_mention_extraction_system.yaml",
            "entity_resolver_system.yaml",
            "evidence_extractor_system.yaml",
            "query_planner_system.yaml",
            "router_localization_system.yaml",
            "router_semantic_system.yaml",
        ]

        for template_name in template_names:
            self.assertTrue(load_template(template_name).strip(), template_name)

    def test_entity_mention_extraction_prompt_renders_yaml_template(self):
        prompt = build_entity_mention_extraction_system_prompt(
            allowed_entity_types=["catalyst", "reaction"],
        )

        self.assertIn('"catalyst"', prompt)
        self.assertIn('"reaction"', prompt)
        self.assertIn('"selected_entity_type": "catalyst"', prompt)
        self.assertNotIn("$allowed_entity_types_json", prompt)

    def test_query_planner_prompt_renders_allowed_lanes(self):
        prompt = build_query_planner_system_prompt()

        for lane in ("review", "frontier", "data", "contrarian"):
            self.assertIn(f'"{lane}"', prompt)
        self.assertIn('"lane": "review"', prompt)
        self.assertNotIn("$allowed_lanes_json", prompt)

    def test_load_template_requires_prompt_field(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            template_dir = Path(tmpdir)
            template_path = template_dir / "missing_prompt.yaml"
            template_path.write_text("title: Missing prompt\n", encoding="utf-8")
            with patch("prompts.basic_node.render._TEMPLATE_DIR", template_dir):
                load_template.cache_clear()
                with self.assertRaisesRegex(ValueError, "must define 'prompt' as a string"):
                    load_template("missing_prompt.yaml")
                load_template.cache_clear()

    def test_load_template_rejects_empty_prompt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            template_dir = Path(tmpdir)
            template_path = template_dir / "empty_prompt.yaml"
            template_path.write_text("prompt: \"   \"\n", encoding="utf-8")
            with patch("prompts.basic_node.render._TEMPLATE_DIR", template_dir):
                load_template.cache_clear()
                with self.assertRaisesRegex(ValueError, "must define a non-empty 'prompt' string"):
                    load_template("empty_prompt.yaml")
                load_template.cache_clear()

    def test_load_template_rejects_invalid_yaml(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            template_dir = Path(tmpdir)
            template_path = template_dir / "invalid.yaml"
            template_path.write_text("prompt: [unclosed\n", encoding="utf-8")
            with patch("prompts.basic_node.render._TEMPLATE_DIR", template_dir):
                load_template.cache_clear()
                with self.assertRaisesRegex(ValueError, "is not valid YAML"):
                    load_template("invalid.yaml")
                load_template.cache_clear()


if __name__ == "__main__":
    unittest.main()
