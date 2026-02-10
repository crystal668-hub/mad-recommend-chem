import unittest


class ProposeMechanismEnforcementTests(unittest.TestCase):
    def test_downgrades_confidence_and_appends_note_when_sections_missing(self):
        from debate.langgraph_coordinator import EvidenceItem, ProposalOutput, _enforce_proposal_mechanism_sections

        p = ProposalOutput(
            reaction_type="OER",
            electrode_composition="Ni(50%), Fe(50%)",
            catalyst_metal_elements=["Ni", "Fe"],
            products="N/A",
            performance_metrics="300 mV overpotential at 10 mA/cm^2",
            confidence="high",
            evidence=[EvidenceItem(source_id="rag:chroma/dummy/doi:10.1#chunk:1")],
            rationale="Uses literature benchmark but does not explain mismatch.",
        )

        out, sections_ok, auto_downgraded = _enforce_proposal_mechanism_sections(p)
        self.assertFalse(sections_ok)
        self.assertTrue(auto_downgraded)
        self.assertEqual("low", out.confidence)
        self.assertIn("AUTO-NOTE:", out.rationale)
        self.assertIn("Mismatch/Mechanism/Adjustment", out.rationale)

    def test_no_downgrade_when_sections_present(self):
        from debate.langgraph_coordinator import EvidenceItem, ProposalOutput, _enforce_proposal_mechanism_sections

        p = ProposalOutput(
            reaction_type="OER",
            electrode_composition="Ni(50%), Fe(50%)",
            catalyst_metal_elements=["Ni", "Fe"],
            products="N/A",
            performance_metrics="300 mV overpotential at 10 mA/cm^2",
            confidence="medium",
            evidence=[EvidenceItem(source_id="rag:chroma/dummy/doi:10.1#chunk:1")],
            rationale="Mismatch: composition differs.\nMechanism: d-band center shift changes OH* binding.\nAdjustment: expect lower overpotential vs benchmark due to stronger electronic coupling.",
        )

        out, sections_ok, auto_downgraded = _enforce_proposal_mechanism_sections(p)
        self.assertTrue(sections_ok)
        self.assertFalse(auto_downgraded)
        self.assertEqual("medium", out.confidence)
        self.assertNotIn("AUTO-NOTE:", out.rationale)

    def test_no_check_when_only_llm_evidence(self):
        from debate.langgraph_coordinator import EvidenceItem, ProposalOutput, _enforce_proposal_mechanism_sections

        p = ProposalOutput(
            reaction_type="OER",
            electrode_composition="Ni(50%), Fe(50%)",
            catalyst_metal_elements=["Ni", "Fe"],
            products="N/A",
            performance_metrics="300 mV overpotential at 10 mA/cm^2",
            confidence="high",
            evidence=[EvidenceItem(source_id="llm")],
            rationale="Parametric estimate without cited evidence.",
        )

        out, sections_ok, auto_downgraded = _enforce_proposal_mechanism_sections(p)
        self.assertTrue(sections_ok)
        self.assertFalse(auto_downgraded)
        self.assertEqual("high", out.confidence)
        self.assertNotIn("AUTO-NOTE:", out.rationale)

    def test_sections_detected_with_literal_backslash_n_escapes(self):
        from debate.langgraph_coordinator import EvidenceItem, ProposalOutput, _enforce_proposal_mechanism_sections

        # Simulate a model that double-escapes newlines in JSON (decoded text contains literal "\n").
        p = ProposalOutput(
            reaction_type="OER",
            electrode_composition="Ni(50%), Fe(50%)",
            catalyst_metal_elements=["Ni", "Fe"],
            products="N/A",
            performance_metrics="300 mV overpotential at 10 mA/cm^2",
            confidence="medium",
            evidence=[EvidenceItem(source_id="rag:chroma/dummy/doi:10.1#chunk:1")],
            rationale="Intro\\\\n\\\\nMismatch: composition differs.\\\\nMechanism: d-band shift.\\\\nAdjustment: +10 mV.",
        )

        out, sections_ok, auto_downgraded = _enforce_proposal_mechanism_sections(p)
        self.assertTrue(sections_ok)
        self.assertFalse(auto_downgraded)
        self.assertEqual("medium", out.confidence)


if __name__ == "__main__":
    unittest.main()
