import unittest


from debate.langgraph_coordinator import LangGraphDebateCoordinator, ProposalState


class TestStalemateResolver(unittest.TestCase):
    def _coordinator(self):
        # No agents needed for deterministic stalemate resolution helpers.
        return LangGraphDebateCoordinator(agents=[], config={})

    def test_stalemate_prefers_strict_composition_over_evidence(self):
        coord = self._coordinator()

        expected_rt = "OER"
        expected_electrode = "Co(57.08%), Ni(23.16%), Zn(14.97%), Fe(2.92%), Cu(1.87%)"
        expected_elements = ["Co", "Ni", "Zn", "Fe", "Cu"]

        p1 = ProposalState(
            proposal_id="agent1",
            agent_name="agent1",
            status="active",
            claim=(
                "Reaction Type: OER\n"
                "Electrode composition (exactly as provided): Co(57.08%), Ni(23.16%), Zn(14.97%), Fe(2.92%), Cu(1.87%)\n"
                "Metal catalyst elements (explicit): Co, Ni, Zn, Fe, Cu\n"
                "Products: N/A\n"
                "Performance Metrics: 305-345 mV overpotential at 10 mA/cm^2 (Confidence: low)\n"
                "Evidence: llm\n"
            ),
        )
        p3 = ProposalState(
            proposal_id="agent3",
            agent_name="agent3",
            status="active",
            claim=(
                "Reaction Type: OER\n"
                "Electrode composition: CoNiFeCu (atomic ratio 1:1:1:0.5)\n"
                "Metal catalyst elements (explicit): Co, Ni, Fe, Cu\n"
                "Products: N/A\n"
                "Performance Metrics: 291.5 mV @ 10 mA cm^-2 (Confidence: high)\n"
                "Evidence: rag:chroma/electrochemistry_literature_agent3/doi:10.1002/adma.202109108#chunk:17\n"
            ),
        )

        winner_id, final_products, final_claim, details = coord._resolve_stalemate_score(
            surviving=[p1, p3],
            expected_reaction_type=expected_rt,
            expected_electrode_composition=expected_electrode,
            expected_elements=expected_elements,
            percent_tolerance=0.05,
            range_strategy="conservative",
        )

        self.assertEqual(winner_id, "agent1")
        self.assertEqual(final_products, "N/A")
        self.assertIn(f"Electrode composition (exactly as provided): {expected_electrode}", final_claim)
        # OER is lower-is-better => conservative picks the worse (higher) bound.
        self.assertIn("Performance Metrics: 345 mV", final_claim)
        self.assertIn("Evidence: llm", final_claim)
        self.assertIn("AUTO-NOTE: Detected metric range; conservative bound selected for final output.", final_claim)
        self.assertIn("AUTO-NOTE: No verifiable rag:chroma source_id found in winning claim", final_claim)
        self.assertIsInstance(details, dict)

    def test_evidence_beats_confidence(self):
        coord = self._coordinator()

        expected_rt = "OER"
        expected_electrode = "Co(10.00%), Ni(20.00%), Zn(30.00%), Fe(25.00%), Cu(15.00%)"
        expected_elements = ["Co", "Ni", "Zn", "Fe", "Cu"]

        with_evidence = ProposalState(
            proposal_id="agent1",
            agent_name="agent1",
            status="active",
            claim=(
                "Reaction Type: OER\n"
                "Electrode composition (exactly as provided): Co(10.00%), Ni(20.00%), Zn(30.00%), Fe(25.00%), Cu(15.00%)\n"
                "Metal catalyst elements (explicit): Co, Ni, Zn, Fe, Cu\n"
                "Products: N/A\n"
                "Performance Metrics: 320 mV overpotential at 10 mA/cm^2 (Confidence: low)\n"
                "Evidence: rag:chroma/c/doi:10.0000/example#chunk:1\n"
            ),
        )
        high_conf_no_evidence = ProposalState(
            proposal_id="agent3",
            agent_name="agent3",
            status="active",
            claim=(
                "Reaction Type: OER\n"
                "Electrode composition (exactly as provided): Co(10.00%), Ni(20.00%), Zn(30.00%), Fe(25.00%), Cu(15.00%)\n"
                "Metal catalyst elements (explicit): Co, Ni, Zn, Fe, Cu\n"
                "Products: N/A\n"
                "Performance Metrics: 320 mV overpotential at 10 mA/cm^2 (Confidence: high)\n"
                "Evidence: llm\n"
            ),
        )

        winner_id, _final_products, final_claim, _details = coord._resolve_stalemate_score(
            surviving=[high_conf_no_evidence, with_evidence],
            expected_reaction_type=expected_rt,
            expected_electrode_composition=expected_electrode,
            expected_elements=expected_elements,
            percent_tolerance=0.05,
            range_strategy="conservative",
        )

        self.assertEqual(winner_id, "agent1")
        self.assertIn("Evidence: rag:chroma/c/doi:10.0000/example#chunk:1", final_claim)

    def test_confidence_tie_break(self):
        coord = self._coordinator()

        expected_rt = "OER"
        expected_electrode = "Co(10.00%), Ni(20.00%), Zn(30.00%), Fe(25.00%), Cu(15.00%)"
        expected_elements = ["Co", "Ni", "Zn", "Fe", "Cu"]

        low = ProposalState(
            proposal_id="agent1",
            agent_name="agent1",
            status="active",
            claim=(
                "Reaction Type: OER\n"
                "Electrode composition (exactly as provided): Co(10.00%), Ni(20.00%), Zn(30.00%), Fe(25.00%), Cu(15.00%)\n"
                "Metal catalyst elements (explicit): Co, Ni, Zn, Fe, Cu\n"
                "Products: N/A\n"
                "Performance Metrics: 320 mV overpotential at 10 mA/cm^2 (Confidence: low)\n"
                "Evidence: llm\n"
            ),
        )
        high = ProposalState(
            proposal_id="agent3",
            agent_name="agent3",
            status="active",
            claim=(
                "Reaction Type: OER\n"
                "Electrode composition (exactly as provided): Co(10.00%), Ni(20.00%), Zn(30.00%), Fe(25.00%), Cu(15.00%)\n"
                "Metal catalyst elements (explicit): Co, Ni, Zn, Fe, Cu\n"
                "Products: N/A\n"
                "Performance Metrics: 320 mV overpotential at 10 mA/cm^2 (Confidence: medium-high)\n"
                "Evidence: llm\n"
            ),
        )

        winner_id, _final_products, _final_claim, _details = coord._resolve_stalemate_score(
            surviving=[low, high],
            expected_reaction_type=expected_rt,
            expected_electrode_composition=expected_electrode,
            expected_elements=expected_elements,
            percent_tolerance=0.05,
            range_strategy="conservative",
        )

        self.assertEqual(winner_id, "agent3")


if __name__ == "__main__":
    unittest.main()

