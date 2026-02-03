import unittest

from agents.react_reasoning import ReActStep, ReActTrajectory, ToolCallRecord
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
                        tool_name="search_rag",
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


if __name__ == "__main__":
    unittest.main()

