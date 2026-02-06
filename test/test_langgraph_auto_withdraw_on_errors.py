import unittest


class LangGraphAutoWithdrawOnErrorsTests(unittest.TestCase):
    def test_auto_withdraw_after_repeated_timeouts(self):
        from debate.langgraph_coordinator import LangGraphDebateCoordinator, ProposalState

        coordinator = LangGraphDebateCoordinator(
            agents=[],
            config={
                "auto_withdraw_on_call_errors": True,
                "auto_withdraw_call_error_threshold": 2,
                "auto_withdraw_call_error_types": ["timeout", "invalid_json"],
                "auto_withdraw_status": "withdrawn",
            },
        )

        proposals = {"agent2": ProposalState(proposal_id="agent2", agent_name="agent2")}
        call_history = []

        coordinator._apply_auto_withdraw_policy(
            proposals=proposals,
            from_id="agent2",
            phase="review",
            err="timeout",
            round_number=1,
            call_history=call_history,
        )
        self.assertEqual(proposals["agent2"].status, "active")

        coordinator._apply_auto_withdraw_policy(
            proposals=proposals,
            from_id="agent2",
            phase="review",
            err="timeout",
            round_number=1,
            call_history=call_history,
        )
        self.assertEqual(proposals["agent2"].status, "withdrawn")
        self.assertTrue(any((e or {}).get("type") == "auto_withdraw" for e in call_history))

    def test_error_streak_resets_on_success(self):
        from debate.langgraph_coordinator import LangGraphDebateCoordinator, ProposalState

        coordinator = LangGraphDebateCoordinator(
            agents=[],
            config={
                "auto_withdraw_on_call_errors": True,
                "auto_withdraw_call_error_threshold": 2,
                "auto_withdraw_call_error_types": ["timeout", "invalid_json"],
                "auto_withdraw_status": "withdrawn",
            },
        )

        proposals = {"agent2": ProposalState(proposal_id="agent2", agent_name="agent2")}
        call_history = []

        coordinator._apply_auto_withdraw_policy(
            proposals=proposals,
            from_id="agent2",
            phase="review",
            err="timeout",
            round_number=1,
            call_history=call_history,
        )
        self.assertEqual(proposals["agent2"].call_error_streak, 1)

        coordinator._apply_auto_withdraw_policy(
            proposals=proposals,
            from_id="agent2",
            phase="review",
            err=None,
            round_number=1,
            call_history=call_history,
        )
        self.assertEqual(proposals["agent2"].status, "active")
        self.assertEqual(proposals["agent2"].call_error_streak, 0)
        self.assertFalse(any((e or {}).get("type") == "auto_withdraw" for e in call_history))


if __name__ == "__main__":
    unittest.main()

