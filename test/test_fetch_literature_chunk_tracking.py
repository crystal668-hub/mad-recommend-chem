import unittest


class FetchLiteratureChunkTrackingTests(unittest.TestCase):
    def test_react_and_coordinator_collect_fetch_source_ids(self):
        from agents.react_agent import _collect_retrieved_source_ids_from_trajectory
        from agents.react_reasoning import ReActStep, ReActTrajectory, ToolCallRecord
        from debate.langgraph_coordinator import _collect_retrieved_source_ids

        sid = "rag:chroma/electrochemistry_literature_agent3/doi:10.1002/adma.202109108#chunk:17"

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
                        tool_name="fetch_literature_chunk",
                        tool_call_id="c1",
                        tool_args={"source_id": sid},
                        observation="fetched",
                        observation_data=[{"source_id": sid, "text": "x", "metadata": {}}],
                    )
                ],
            )
        )

        self.assertIn(sid, _collect_retrieved_source_ids_from_trajectory(traj))
        self.assertIn(sid, _collect_retrieved_source_ids(traj))

    def test_collects_union_of_search_and_fetch(self):
        from agents.react_agent import _collect_retrieved_source_ids_from_trajectory
        from agents.react_reasoning import ReActStep, ReActTrajectory, ToolCallRecord
        from debate.langgraph_coordinator import _collect_retrieved_source_ids

        sid1 = "rag:chroma/electrochemistry_literature_agent1/doi:10.1/abc#chunk:1"
        sid2 = "rag:chroma/electrochemistry_literature_agent2/doi:10.2/def#chunk:2"

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
                        tool_call_id="c1",
                        tool_args={"query": "x"},
                        observation="obs",
                        observation_data=[{"source_id": sid1}],
                    ),
                    ToolCallRecord(
                        tool_name="fetch_literature_chunk",
                        tool_call_id="c2",
                        tool_args={"source_id": sid2},
                        observation="obs2",
                        observation_data=[{"source_id": sid2}],
                    ),
                ],
            )
        )

        s_react = _collect_retrieved_source_ids_from_trajectory(traj)
        s_coord = _collect_retrieved_source_ids(traj)
        self.assertEqual({sid1, sid2}, set(s_react))
        self.assertEqual({sid1, sid2}, set(s_coord))


if __name__ == "__main__":
    unittest.main()

