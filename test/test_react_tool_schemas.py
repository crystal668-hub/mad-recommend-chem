from __future__ import annotations

import unittest

from langchain_core.tools import StructuredTool
from pydantic import ValidationError

from agents import react_tool_schemas as tool_schemas


class ReactToolSchemaTests(unittest.TestCase):
    def _canonical_submission(self) -> dict:
        return {
            "submission_id": "submission_cycle_1",
            "question": "How does Pt/C compare with NiMo catalysts for HER activity in alkaline media?",
            "version": 1,
            "sections": [
                {
                    "section_id": "comparison_summary",
                    "title": "Comparison Summary",
                    "content": "Pt/C remains the benchmark, while NiMo often approaches it under alkaline HER conditions.",
                    "citation_ids": ["CIT-1"],
                    "step_refs": [{"trajectory_id": "traj_1", "step_number": 1}],
                    "issue_refs": [],
                    "section_confidence": {
                        "level": "medium",
                        "score": 0.55,
                        "rationale": "One anchored citation supports this section.",
                    },
                }
            ],
            "citations": [
                {
                    "citation_id": "CIT-1",
                    "paper_id": "paper-1",
                    "doi": "10.1000/example",
                    "title": "Pt/C and NiMo for alkaline HER",
                    "year": 2024,
                    "venue": "J. Catalysis",
                    "section_ids": ["sec_results"],
                    "evidence_ids": ["ev-1"],
                }
            ],
            "limitations": ["Evidence coverage is still limited to one anchored citation."],
            "overall_confidence": {
                "level": "medium",
                "score": 0.55,
                "rationale": "Grounded in one anchored citation and one extracted evidence item.",
            },
            "trajectory_id": "traj_1",
            "step_refs": [{"trajectory_id": "traj_1", "step_number": 1}],
            "issue_refs": [],
        }

    def test_proposer_conclude_tool_schema_accepts_canonical_submission(self):
        def conclude(submission):
            """test conclude"""
            return submission

        tool = StructuredTool.from_function(
            conclude,
            name="conclude",
            args_schema=tool_schemas.ProposerConcludeToolInput,
        )

        payload = tool.invoke({"submission": self._canonical_submission()})

        self.assertEqual("submission_cycle_1", payload.submission_id)

    def test_proposer_conclude_tool_schema_rejects_schema_drift(self):
        def conclude(submission):
            """test conclude"""
            return submission

        tool = StructuredTool.from_function(
            conclude,
            name="conclude",
            args_schema=tool_schemas.ProposerConcludeToolInput,
        )

        drift_payload = {
            "submission": {
                "cycle_number": 1,
                "normalized_question": "How does Pt/C compare with NiMo catalysts for HER activity in alkaline media?",
                "answer_sections": [
                    {
                        "section_id": "comparison_summary",
                        "title": "Comparison Summary",
                        "content": "No anchored evidence available.",
                        "citations": [],
                    }
                ],
                "overall_confidence": 0.05,
                "limitations": "No tool-anchored evidence.",
                "evidence_items": [],
            }
        }

        with self.assertRaises(ValidationError):
            tool.invoke(drift_payload)

    def test_reviewer_conclude_tool_schema_accepts_canonical_payload(self):
        def conclude(review):
            """test conclude"""
            return review

        tool = StructuredTool.from_function(
            conclude,
            name="conclude",
            args_schema=tool_schemas.ReviewerConcludeToolInput,
        )

        payload = tool.invoke(
            {
                "review": {
                    "review_items": [
                        {
                            "review_id": "rev-1",
                            "reviewer_role": "search_coverage",
                            "anchor_kind": "global",
                            "severity": "warning",
                            "flaw_type": "needs_manual_review",
                            "critique": "Coverage may be incomplete.",
                            "required_action": "Check one more supporting paper.",
                        }
                    ]
                }
            }
        )

        self.assertEqual("rev-1", payload.review_items[0].review_id)

    def test_generic_conclude_schema_is_text_only(self):
        self.assertIs(
            tool_schemas.GenericTextConcludeToolInput,
            tool_schemas.get_generic_conclude_args_schema("any"),
        )

    def test_generic_conclude_schema_rejects_non_text_shape(self):
        def conclude(conclusion):
            """test conclude"""
            return conclusion

        tool = StructuredTool.from_function(
            conclude,
            name="conclude",
            args_schema=tool_schemas.GenericTextConcludeToolInput,
        )

        with self.assertRaises(ValidationError):
            tool.invoke({"conclusion": {"summary": "not text"}})


if __name__ == "__main__":
    unittest.main()
