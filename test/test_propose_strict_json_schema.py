import unittest


class ProposeStrictJsonSchemaTests(unittest.TestCase):
    def test_coerce_proposal_output_patches_from_prompt_and_trajectory(self):
        from agents.react_reasoning import ReActStep, ReActTrajectory, ToolCallRecord
        from debate.langgraph_coordinator import _coerce_proposal_output, _render_proposal_claim

        prompt = (
            "Please propose an evidence-backed prediction for the target electrochemical reaction.\n"
            "Target reaction: OER\n"
            "Electrode composition (relative %): Ni(69.00%), Co(19.07%), Fe(11.48%)\n"
            "Metal catalyst elements: Ni, Co, Fe\n"
        )

        traj = ReActTrajectory(query=prompt)
        rag_item = {"source_id": "rag:chroma/dummy/doi:10.1#chunk:1"}
        call = ToolCallRecord(
            tool_name="search_literature",
            tool_call_id="call_1",
            tool_args={"query": "x", "top_k": 1},
            observation="Found 1 relevant documents:",
            observation_data=[rag_item],
        )
        traj.add_step(
            ReActStep(
                step_number=1,
                thought="t",
                action="search_literature",
                action_input={"query": "x"},
                observation="obs",
                tool_calls=[call],
            )
        )

        # Missing electrode composition / evidence in the model output should be patched.
        parsed = {
            "reaction_type": "OER",
            "electrode_composition": "",
            "catalyst_metal_elements": [],
            "products": "",
            "performance_metrics": "310 mV overpotential at 10 mA/cm^2",
            "confidence": "medium",
            "evidence": [],
            "rationale": "r",
        }

        out, schema_ok = _coerce_proposal_output(
            parsed=parsed,
            prompt=prompt,
            components=["Ni", "Co", "Fe"],
            reaction_type="OER",
            trajectory=traj,
        )
        self.assertTrue(schema_ok)

        claim = _render_proposal_claim(out)
        self.assertIn(
            "Electrode composition (exactly as provided): Ni(69.00%), Co(19.07%), Fe(11.48%)",
            claim,
        )
        self.assertIn("Metal catalyst elements (explicit): Ni, Co, Fe", claim)
        self.assertIn("Evidence: rag:chroma/dummy/doi:10.1#chunk:1", claim)
        self.assertIn("Performance Metrics: 310 mV overpotential at 10 mA/cm^2 (Confidence: medium)", claim)


if __name__ == "__main__":
    unittest.main()

