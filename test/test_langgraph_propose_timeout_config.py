import unittest


class LangGraphProposeTimeoutConfigTests(unittest.TestCase):
    def test_propose_timeout_overrides_round_timeout(self):
        from debate.langgraph_coordinator import LangGraphDebateCoordinator

        coord = LangGraphDebateCoordinator(
            agents=[],
            config={"timeout": 300, "round_timeout": 900, "propose_timeout": 1200},
        )
        self.assertAlmostEqual(coord.propose_timeout_seconds, 1200.0)
        # Deprecated alias should match.
        self.assertAlmostEqual(coord.round_timeout_seconds, 1200.0)

    def test_round_timeout_used_when_propose_timeout_missing(self):
        from debate.langgraph_coordinator import LangGraphDebateCoordinator

        coord = LangGraphDebateCoordinator(
            agents=[],
            config={"timeout": 300, "round_timeout": 900},
        )
        self.assertAlmostEqual(coord.propose_timeout_seconds, 900.0)
        self.assertAlmostEqual(coord.round_timeout_seconds, 900.0)

    def test_timeout_fallback_when_both_missing(self):
        from debate.langgraph_coordinator import LangGraphDebateCoordinator

        coord = LangGraphDebateCoordinator(
            agents=[],
            config={"timeout": 300},
        )
        self.assertAlmostEqual(coord.propose_timeout_seconds, 300.0)
        self.assertAlmostEqual(coord.round_timeout_seconds, 300.0)


if __name__ == "__main__":
    unittest.main()

