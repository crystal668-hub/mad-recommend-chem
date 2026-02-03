import unittest

from debate.langgraph_coordinator import (
    LangGraphDebateCoordinator,
    ProposalState,
    DebateReview,
    DebateRebuttal,
)


class LangGraphRuleAdjudicationTests(unittest.TestCase):
    def _coordinator(self, no_response_threshold: int = 2) -> LangGraphDebateCoordinator:
        return LangGraphDebateCoordinator(
            agents=[],
            config={
                "max_rounds": 3,
                "no_response_threshold": no_response_threshold,
                "max_reviews_per_target": 1,
            },
        )

    def test_defeated_after_two_consecutive_no_response_rounds(self):
        coord = self._coordinator(no_response_threshold=2)

        proposals = {
            "agent1": ProposalState(proposal_id="agent1", agent_name="A"),
            "agent2": ProposalState(proposal_id="agent2", agent_name="B"),
        }

        # Round 1: valid review against agent1, but no rebuttal.
        reviews_r1 = [
            DebateReview(
                review_id="rev_r1_agent2_0",
                round_number=1,
                from_proposal_id="agent2",
                target_proposal_id="agent1",
                target_step_number=1,
                flaw_type="missing_evidence",
                critique="No support for step 1",
                evidence=[{"source_id": "rag:chroma/x/doi:10.1#chunk:1"}],
                valid=True,
            )
        ]
        rebuttals_r1 = []
        changed, consensus = coord._rule_adjudicate(proposals, 1, reviews_r1, rebuttals_r1)
        self.assertTrue(changed)
        self.assertFalse(consensus)
        self.assertEqual(proposals["agent1"].no_response_streak, 1)
        self.assertEqual(proposals["agent1"].status, "active")

        # Round 2: again no rebuttal -> defeated.
        reviews_r2 = [
            DebateReview(
                review_id="rev_r2_agent2_0",
                round_number=2,
                from_proposal_id="agent2",
                target_proposal_id="agent1",
                target_step_number=1,
                flaw_type="missing_evidence",
                critique="Still no support",
                evidence=[{"source_id": "rag:chroma/x/doi:10.1#chunk:1"}],
                valid=True,
            )
        ]
        rebuttals_r2 = []
        changed, _consensus = coord._rule_adjudicate(proposals, 2, reviews_r2, rebuttals_r2)
        self.assertTrue(changed)
        self.assertEqual(proposals["agent1"].no_response_streak, 2)
        self.assertEqual(proposals["agent1"].status, "defeated")

    def test_streak_resets_when_all_valid_reviews_are_responded(self):
        coord = self._coordinator(no_response_threshold=2)

        proposals = {
            "agent1": ProposalState(proposal_id="agent1", agent_name="A"),
            "agent2": ProposalState(proposal_id="agent2", agent_name="B"),
        }
        proposals["agent1"].no_response_streak = 1

        reviews = [
            DebateReview(
                review_id="rev_r1_agent2_0",
                round_number=1,
                from_proposal_id="agent2",
                target_proposal_id="agent1",
                target_step_number=1,
                flaw_type="wrong_inference",
                critique="Bad inference",
                evidence=[{"source_id": "rag:chroma/x/doi:10.1#chunk:1"}],
                valid=True,
            )
        ]
        rebuttals = [
            DebateRebuttal(
                rebuttal_id="reb_r1_agent1_0",
                round_number=1,
                from_proposal_id="agent1",
                target_review_id="rev_r1_agent2_0",
                response_mode="defend",
                response="Here is my evidence.",
                evidence=[{"source_id": "rag:chroma/x/doi:10.2#chunk:3"}],
                valid=True,
            )
        ]

        changed, _consensus = coord._rule_adjudicate(proposals, 1, reviews, rebuttals)
        self.assertTrue(changed)
        self.assertEqual(proposals["agent1"].no_response_streak, 0)
        self.assertEqual(proposals["agent1"].status, "active")


if __name__ == "__main__":
    unittest.main()

