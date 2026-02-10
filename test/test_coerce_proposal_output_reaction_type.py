import unittest


class CoerceProposalOutputReactionTypeTests(unittest.TestCase):
    def test_task_reaction_type_overrides_model_unknown(self):
        from debate.langgraph_coordinator import _coerce_proposal_output

        parsed = {"reaction_type": "UNKNOWN"}
        prompt = (
            "Please propose...\n"
            "Target reaction: OER\n"
            "Electrode composition (relative %): Co(50%), Ni(50%)\n"
        )
        out, _ok = _coerce_proposal_output(
            parsed=parsed,
            prompt=prompt,
            components=["Co(50%)", "Ni(50%)"],
            reaction_type="OER",
            trajectory=None,
        )
        self.assertEqual(out.reaction_type, "OER")

    def test_model_reaction_used_when_task_unknown(self):
        from debate.langgraph_coordinator import _coerce_proposal_output

        parsed = {"reaction_type": "OER"}
        prompt = (
            "Please propose...\n"
            "Target reaction: UNKNOWN\n"
            "Electrode composition (relative %): Co(50%), Ni(50%)\n"
        )
        out, _ok = _coerce_proposal_output(
            parsed=parsed,
            prompt=prompt,
            components=["Co(50%)", "Ni(50%)"],
            reaction_type="UNKNOWN",
            trajectory=None,
        )
        self.assertEqual(out.reaction_type, "OER")


if __name__ == "__main__":
    unittest.main()

