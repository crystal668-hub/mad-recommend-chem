from __future__ import annotations

import unittest


class LangGraphParametricReviewTests(unittest.TestCase):
    def _coordinator(self):
        from debate.langgraph_coordinator import LangGraphDebateCoordinator

        return LangGraphDebateCoordinator(
            agents=[],
            config={
                "max_rounds": 2,
                "no_response_threshold": 2,
                "max_reviews_per_target": 1,
            },
        )

    def _proposals_with_step1(self):
        from debate.langgraph_coordinator import ProposalState
        from agents.react_reasoning import ReActTrajectory, ReActStep

        traj = ReActTrajectory(query="dummy")
        traj.add_step(
            ReActStep(
                step_number=1,
                thought="t",
                action="search_literature",
                action_input={},
                observation="obs",
            )
        )

        proposals = {
            "agent1": ProposalState(proposal_id="agent1", agent_name="A", propose_trajectory=traj),
            "agent2": ProposalState(proposal_id="agent2", agent_name="B", propose_trajectory=traj),
        }
        return proposals

    def test_parametric_review_is_valid_without_evidence(self):
        from debate.langgraph_coordinator import ReviewItem

        coord = self._coordinator()
        proposals = self._proposals_with_step1()

        item = ReviewItem(
            target_proposal_id="agent1",
            target_step_number=1,
            flaw_type="other",
            critique="Parametric critique with no evidence.",
            evidence=[],
        )
        review = coord._validate_review_item(
            review_id="rev_r1_agent2_0",
            round_number=1,
            from_id="agent2",
            item=item,
            proposals=proposals,
            retrieved_source_ids=set(),
        )
        self.assertTrue(review.valid)
        self.assertEqual(review.evidence, [])

    def test_parametric_reviews_block_consensus_and_require_response(self):
        from debate.langgraph_coordinator import DebateReview

        coord = self._coordinator()
        proposals = self._proposals_with_step1()

        # A valid review with evidence=[] should still affect adjudication (evidence is optional).
        reviews = [
            DebateReview(
                review_id="rev_r1_agent2_0",
                round_number=1,
                from_proposal_id="agent2",
                target_proposal_id="agent1",
                target_step_number=1,
                flaw_type="other",
                critique="Parametric critique.",
                evidence=[],
                valid=True,
            )
        ]

        changed, consensus = coord._rule_adjudicate(
            proposals=proposals,
            round_number=1,
            round_reviews=reviews,
            round_rebuttals=[],
            round_review_calls=[{"error": None}],
        )

        self.assertTrue(changed)
        self.assertFalse(consensus)
        self.assertEqual(proposals["agent1"].no_response_streak, 1)

    def test_parametric_rebuttal_is_valid_without_evidence(self):
        from debate.langgraph_coordinator import RebuttalItem

        coord = self._coordinator()

        item = RebuttalItem(
            target_review_id="rev_r1_agent2_0",
            response_mode="defend",
            response="I defend this claim based on parametric knowledge.",
            evidence=[],
        )
        rebuttal = coord._validate_rebuttal_item(
            rebuttal_id="reb_r1_agent1_0",
            round_number=1,
            from_id="agent1",
            item=item,
            valid_review_ids={"rev_r1_agent2_0"},
            retrieved_source_ids=set(),
        )
        self.assertTrue(rebuttal.valid)

    def test_defend_or_revise_requires_non_empty_response(self):
        from debate.langgraph_coordinator import RebuttalItem

        coord = self._coordinator()

        item = RebuttalItem(
            target_review_id="rev_r1_agent2_0",
            response_mode="defend",
            response="",
            evidence=[],
        )
        rebuttal = coord._validate_rebuttal_item(
            rebuttal_id="reb_r1_agent1_0",
            round_number=1,
            from_id="agent1",
            item=item,
            valid_review_ids={"rev_r1_agent2_0"},
            retrieved_source_ids=set(),
        )
        self.assertFalse(rebuttal.valid)
        self.assertEqual(rebuttal.invalid_reason, "empty_response")


if __name__ == "__main__":
    unittest.main()
