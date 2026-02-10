import unittest


class DebatePhasePromptContentTests(unittest.TestCase):
    def test_propose_prompt_has_reduced_step_budget_and_parallel_retrieval(self):
        from prompts import debate_phase_prompts as dp

        p = dp.DEBATE_PROPOSE_SYSTEM_PROMPT
        self.assertIn("at most 5 ReAct steps", p)
        self.assertIn("FIRST ACTION: emit >=3 retrieval tool_calls", p)
        self.assertIn("Retrieval budget: at most TWO ACTION steps", p)
        self.assertIn("meaningfully DISTINCT", p)
        # PROPOSE now uses STRICT JSON to reduce format-related rework.
        self.assertIn("STRICT JSON ONLY", p)
        self.assertIn("Output schema (STRICT JSON)", p)
        self.assertIn("\"reaction_type\"", p)
        self.assertIn("\"electrode_composition\"", p)
        self.assertIn("\"catalyst_metal_elements\"", p)
        self.assertIn("\"performance_metrics\"", p)
        self.assertIn("\"confidence\"", p)
        self.assertIn("\"evidence\"", p)
        self.assertIn("\"rationale\"", p)
        self.assertIn('\"source_id\": \"llm\"', p)
        self.assertIn("point estimate", p.lower())
        self.assertIn("confidence", p.lower())
        # Mechanism-based correction scaffold when citing mismatched evidence.
        self.assertIn("Mismatch:", p)
        self.assertIn("Mechanism:", p)
        self.assertIn("Adjustment:", p)
        self.assertIn("mixed_search_and_analysis", p)

    def test_review_prompt_prefers_parallel_retrieval_then_conclude(self):
        from prompts import debate_phase_prompts as dp

        p = dp.DEBATE_REVIEW_SYSTEM_PROMPT
        self.assertIn("You have at most 3 ReAct steps.", p)
        self.assertIn("Retrieval budget: at most ONE ACTION step", p)
        self.assertIn("fetch_literature_chunk", p)
        self.assertIn("Preferred workflows:", p)
        self.assertIn("ACTION 1 =", p)
        self.assertIn("ACTION 2 = `conclude`", p)
        self.assertIn("You MAY return an empty reviews list", p)
        self.assertIn("STRICT JSON ONLY", p)
        self.assertIn('\"source_id\": \"llm\"', p)
        self.assertIn("Mismatch/Mechanism/Adjustment", p)

    def test_rebuttal_prompt_has_four_step_budget(self):
        from prompts import debate_phase_prompts as dp

        p = dp.DEBATE_REBUTTAL_SYSTEM_PROMPT
        self.assertIn("You have at most 4 ReAct steps.", p)
        self.assertIn("Retrieval budget: at most ONE ACTION step", p)
        self.assertIn("fetch_literature_chunk", p)
        self.assertIn("ACTION 1:", p)
        self.assertIn("ACTION 3: `conclude` with STRICT JSON", p)
        self.assertIn("withdraw` or `no_response`, do NOT retrieve", p)
        self.assertIn('\"source_id\": \"llm\"', p)


if __name__ == "__main__":
    unittest.main()
