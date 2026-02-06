import unittest


class ReactConcludeGuardWarningOnlyTests(unittest.TestCase):
    def test_forbidden_elements_in_cited_evidence_is_warning_not_block(self):
        # Import the private helper directly; this is a unit test for guard behavior.
        from agents.react_agent import _validate_conclusion_against_task_with_evidence
        from agents.react_reasoning import ReActTrajectory, ReActStep, ToolCallRecord

        sid = "rag:chroma/test_collection/doi:10.1234#chunk:1"
        traj = ReActTrajectory(query="dummy")
        traj.add_step(
            ReActStep(
                step_number=1,
                thought="dummy",
                action="search_literature",
                action_input={},
                observation="dummy",
                tool_calls=[
                    ToolCallRecord(
                        tool_name="search_literature",
                        tool_call_id="call_1",
                        tool_args={"query": "dummy", "top_k": 1},
                        observation="dummy",
                        observation_data=[
                            {"source_id": sid, "forbidden_elements": ["Ru"]},
                        ],
                    )
                ],
            )
        )

        required = ["Pt", "Pd", "Ni", "Fe", "Co"]
        conclusion = (
            "Reaction Type: OER\n"
            "Catalyst metal elements (exactly as provided): Pt, Pd, Ni, Fe, Co\n"
            f"Evidence: {sid}\n"
        )

        ok, reason = _validate_conclusion_against_task_with_evidence(conclusion, required, traj)
        self.assertTrue(ok)
        self.assertIn("warning:", reason)


if __name__ == "__main__":
    unittest.main()
