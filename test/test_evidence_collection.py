import unittest

from agents.react_reasoning import ReActStep, ReActTrajectory, ToolCallRecord
from agents.react_agent import _collect_retrieved_source_ids_from_trajectory
from debate.langgraph_coordinator import _collect_retrieved_source_ids


class EvidenceCollectionTests(unittest.TestCase):
    def test_collects_source_ids_from_multi_tool_steps(self):
        traj = ReActTrajectory(query="q")
        traj.add_step(
            ReActStep(
                step_number=1,
                thought="t",
                action="multi_tool",
                action_input={},
                observation="o",
                tool_calls=[
                    ToolCallRecord(
                        tool_name="search_literature",
                        tool_call_id="call_1",
                        tool_args={"query": "x"},
                        observation="obs",
                        observation_data=[
                            {"source_id": "rag:chroma/c/doi:10.1#chunk:1"},
                            {"source_id": "rag:chroma/c/doi:10.2#chunk:2"},
                        ],
                    )
                ],
            )
        )

        sids = _collect_retrieved_source_ids(traj)
        self.assertIn("rag:chroma/c/doi:10.1#chunk:1", sids)
        self.assertIn("rag:chroma/c/doi:10.2#chunk:2", sids)

    def test_collects_source_ids_robust_to_non_dict_items(self):
        traj = ReActTrajectory(query="q")
        traj.add_step(
            ReActStep(
                step_number=1,
                thought="t",
                action="multi_tool",
                action_input={},
                observation="o",
                tool_calls=[
                    ToolCallRecord(
                        tool_name="search_literature",
                        tool_call_id="call_1",
                        tool_args={"query": "x"},
                        observation="obs",
                        observation_data=[
                            "not_a_dict",
                            {"source_id": "rag:chroma/c/doi:10.1#chunk:1"},
                        ],
                    )
                ],
            )
        )

        sids = _collect_retrieved_source_ids(traj)
        self.assertEqual({"rag:chroma/c/doi:10.1#chunk:1"}, sids)

    def test_react_agent_collect_source_ids_skips_error_payloads(self):
        traj = ReActTrajectory(query="q")
        traj.add_step(
            ReActStep(
                step_number=1,
                thought="t",
                action="multi_tool",
                action_input={},
                observation="o",
                tool_calls=[
                    # Simulate blocked retrieval budget payloads (dict) that should be ignored.
                    ToolCallRecord(
                        tool_name="search_literature",
                        tool_call_id="call_blocked",
                        tool_args={"query": "x"},
                        observation="Policy: retrieval budget exceeded for this phase.",
                        observation_data={"error": "retrieval_budget_exceeded"},
                    ),
                    # And a normal RAG return (list of dicts) that should be collected.
                    ToolCallRecord(
                        tool_name="search_literature",
                        tool_call_id="call_ok",
                        tool_args={"query": "y"},
                        observation="Found 1 relevant document.",
                        observation_data=[{"source_id": "rag:chroma/c/doi:10.2#chunk:2"}],
                    ),
                ],
            )
        )

        sids = _collect_retrieved_source_ids_from_trajectory(traj)
        self.assertEqual({"rag:chroma/c/doi:10.2#chunk:2"}, sids)


if __name__ == "__main__":
    unittest.main()
