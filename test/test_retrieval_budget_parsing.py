import unittest


class RetrievalBudgetParsingTests(unittest.TestCase):
    def test_parse_retrieval_budgets_from_debate_prompts(self):
        from agents.react_agent import _parse_retrieval_budget_from_system_prompt
        from prompts.debate_phase_prompts import (
            DEBATE_PROPOSE_SYSTEM_PROMPT,
            DEBATE_REVIEW_SYSTEM_PROMPT,
            DEBATE_REBUTTAL_SYSTEM_PROMPT,
        )

        self.assertEqual(_parse_retrieval_budget_from_system_prompt(DEBATE_PROPOSE_SYSTEM_PROMPT), 2)
        self.assertEqual(_parse_retrieval_budget_from_system_prompt(DEBATE_REVIEW_SYSTEM_PROMPT), 1)
        self.assertEqual(_parse_retrieval_budget_from_system_prompt(DEBATE_REBUTTAL_SYSTEM_PROMPT), 1)


if __name__ == "__main__":
    unittest.main()

