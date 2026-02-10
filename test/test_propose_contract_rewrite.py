from __future__ import annotations

import unittest


class ProposeContractRewriteTests(unittest.TestCase):
    def test_coerce_propose_conclusion_to_contract_adds_required_lines(self):
        from agents.react_agent import _coerce_propose_conclusion_to_contract
        from agents.react_reasoning import ReActTrajectory, ReActStep, ToolCallRecord

        full_query = (
            "Please propose an evidence-backed prediction for the target electrochemical reaction.\n"
            "Target reaction: OER\n"
            "Electrode composition (relative %): Ni(69.00%), Co(19.07%), Fe(11.48%), Cu(0.40%), Zn(0.05%)\n"
            "Metal catalyst elements: Ni, Co, Fe, Cu, Zn\n"
        )
        traj = ReActTrajectory(query=full_query)

        # Provide one on-reaction, non-forbidden chunk with a clear overpotential at 10 mA.
        rag_item = {
            "text": "This catalyst shows an overpotential of 170 mV at a current density of 10 mA cm-2 in 1 M KOH.",
            "source_id": "rag:chroma/dummy/doi:10.0000/dummy-oer#chunk:1",
            "reaction_match": True,
            "forbidden_elements": [],
        }
        call = ToolCallRecord(
            tool_name="search_literature",
            tool_call_id="call_1",
            tool_args={"query": "dummy", "top_k": 1},
            observation="Found 1 relevant documents:",
            observation_data=[rag_item],
        )
        step = ReActStep(
            step_number=1,
            thought="t",
            action="search_literature",
            action_input={"query": "dummy"},
            observation="obs",
            tool_calls=[call],
        )
        traj.add_step(step)

        bad_draft = (
            "Conclusion out of scope: The retrieved composition is close but not exact.\n"
            "I will search again in the next step."
        )
        out = _coerce_propose_conclusion_to_contract(
            draft=bad_draft,
            full_query=full_query,
            task_reaction="OER",
            task_components=["Ni", "Co", "Fe", "Cu", "Zn"],
            trajectory=traj,
        )

        self.assertIn("Reaction Type: OER", out)
        self.assertIn(
            "Electrode composition (exactly as provided): Ni(69.00%), Co(19.07%), Fe(11.48%), Cu(0.40%), Zn(0.05%)",
            out,
        )
        self.assertIn("Products: N/A", out)
        self.assertIn("Performance Metrics:", out)
        self.assertIn("Confidence:", out)
        self.assertIn("Evidence:", out)
        # Should cite the rag source_id from the trajectory rather than defaulting to llm.
        self.assertIn("rag:chroma/dummy/doi:10.0000/dummy-oer#chunk:1", out)
        # Preserve the original text for debugging.
        self.assertIn("Rationale:", out)
        self.assertIn("Conclusion out of scope", out)


if __name__ == "__main__":
    unittest.main()

