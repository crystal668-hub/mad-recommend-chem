from __future__ import annotations

import unittest


class SourceIdNormalizationTests(unittest.TestCase):
    def test_normalize_doc_id_strips_markdown_wrappers(self):
        from utils.source_id import normalize_doc_id

        self.assertEqual(normalize_doc_id("10.1002/adma.202109108**"), "10.1002/adma.202109108")
        self.assertEqual(normalize_doc_id("**10.1002/adma.202109108**"), "10.1002/adma.202109108")
        self.assertEqual(normalize_doc_id("`10.1002/adma.202109108`"), "10.1002/adma.202109108")
        self.assertEqual(normalize_doc_id("_10.1002/adma.202109108_"), "10.1002/adma.202109108")

    def test_normalize_chroma_source_id_strips_doc_id_wrappers(self):
        from utils.source_id import normalize_chroma_source_id

        raw = "rag:chroma/c/doi:10.1002/adma.202109108**#chunk:23"
        want = "rag:chroma/c/doi:10.1002/adma.202109108#chunk:23"
        self.assertEqual(normalize_chroma_source_id(raw), want)

    def test_review_evidence_verification_allows_normalized_matching(self):
        from debate.langgraph_coordinator import LangGraphDebateCoordinator, ProposalState, ReviewItem
        from agents.react_reasoning import ReActTrajectory, ReActStep

        coord = LangGraphDebateCoordinator(agents=[], config={"max_rounds": 1})

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

        retrieved = {"rag:chroma/c/doi:10.1002/adma.202109108**#chunk:23"}
        cited = "rag:chroma/c/doi:10.1002/adma.202109108#chunk:23"

        item = ReviewItem(
            target_proposal_id="agent1",
            target_step_number=1,
            flaw_type="missing_evidence",
            critique="Evidence should be verifiable even if DOI wrappers differ.",
            evidence=[{"source_id": cited}],
        )
        review = coord._validate_review_item(
            review_id="rev_r1_agent2_0",
            round_number=1,
            from_id="agent2",
            item=item,
            proposals=proposals,
            retrieved_source_ids=retrieved,
        )
        self.assertTrue(review.valid)

    def test_rebuttal_evidence_verification_allows_normalized_matching(self):
        from debate.langgraph_coordinator import LangGraphDebateCoordinator, RebuttalItem

        coord = LangGraphDebateCoordinator(agents=[], config={"max_rounds": 1})

        retrieved = {"rag:chroma/c/doi:10.1002/adma.202109108**#chunk:23"}
        cited = "rag:chroma/c/doi:10.1002/adma.202109108#chunk:23"

        item = RebuttalItem(
            target_review_id="rev_r1_agent2_0",
            response_mode="defend",
            response="OK",
            evidence=[{"source_id": cited}],
        )
        reb = coord._validate_rebuttal_item(
            rebuttal_id="reb_r1_agent1_0",
            round_number=1,
            from_id="agent1",
            item=item,
            valid_review_ids={"rev_r1_agent2_0"},
            retrieved_source_ids=retrieved,
        )
        self.assertTrue(reb.valid)


if __name__ == "__main__":
    unittest.main()

