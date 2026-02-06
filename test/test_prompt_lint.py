import re
import unittest


def _assert_max_len(testcase: unittest.TestCase, name: str, text: str, max_chars: int) -> None:
    testcase.assertLessEqual(
        len(text),
        max_chars,
        msg=f"{name} too long: {len(text)} chars (budget {max_chars})",
    )


def _assert_no_excessive_duplicate_lines(testcase: unittest.TestCase, name: str, text: str) -> None:
    """
    A lightweight safeguard against accidental prompt duplication (often caused by bad composition).

    We only flag lines that are:
    - non-empty
    - "meaningful" length (>= 30 chars)
    - repeated 3+ times verbatim after whitespace normalization
    """

    norm_lines = []
    for raw in (text or "").splitlines():
        s = re.sub(r"\s+", " ", raw).strip()
        if not s:
            continue
        if len(s) < 30:
            continue
        norm_lines.append(s)

    counts = {}
    for s in norm_lines:
        counts[s] = counts.get(s, 0) + 1

    offenders = [(s, n) for s, n in counts.items() if n >= 3]
    testcase.assertFalse(
        offenders,
        msg=f"{name} contains repeated lines (>=3x): {offenders[:3]}",
    )


class PromptLintTests(unittest.TestCase):
    def test_prompt_length_budgets(self):
        from prompts.system_prompts import UNIFIED_SYSTEM_PROMPT
        from prompts.debate_phase_prompts import (
            DEBATE_PROPOSE_SYSTEM_PROMPT,
            DEBATE_REVIEW_SYSTEM_PROMPT,
            DEBATE_REBUTTAL_SYSTEM_PROMPT,
        )

        # Rough token budget: ~chars/4. Keep these tight to prevent prompt bloat.
        _assert_max_len(self, "UNIFIED_SYSTEM_PROMPT", UNIFIED_SYSTEM_PROMPT, max_chars=3000)
        _assert_max_len(self, "DEBATE_PROPOSE_SYSTEM_PROMPT", DEBATE_PROPOSE_SYSTEM_PROMPT, max_chars=4800)
        _assert_max_len(self, "DEBATE_REVIEW_SYSTEM_PROMPT", DEBATE_REVIEW_SYSTEM_PROMPT, max_chars=2000)
        _assert_max_len(self, "DEBATE_REBUTTAL_SYSTEM_PROMPT", DEBATE_REBUTTAL_SYSTEM_PROMPT, max_chars=2000)

    def test_prompt_has_no_accidental_duplicate_lines(self):
        from prompts.system_prompts import UNIFIED_SYSTEM_PROMPT
        from prompts.debate_phase_prompts import (
            DEBATE_PROPOSE_SYSTEM_PROMPT,
            DEBATE_REVIEW_SYSTEM_PROMPT,
            DEBATE_REBUTTAL_SYSTEM_PROMPT,
        )

        _assert_no_excessive_duplicate_lines(self, "UNIFIED_SYSTEM_PROMPT", UNIFIED_SYSTEM_PROMPT)
        _assert_no_excessive_duplicate_lines(self, "DEBATE_PROPOSE_SYSTEM_PROMPT", DEBATE_PROPOSE_SYSTEM_PROMPT)
        _assert_no_excessive_duplicate_lines(self, "DEBATE_REVIEW_SYSTEM_PROMPT", DEBATE_REVIEW_SYSTEM_PROMPT)
        _assert_no_excessive_duplicate_lines(self, "DEBATE_REBUTTAL_SYSTEM_PROMPT", DEBATE_REBUTTAL_SYSTEM_PROMPT)

    def test_propose_prompt_has_must_should_sections(self):
        from prompts.debate_phase_prompts import DEBATE_PROPOSE_SYSTEM_PROMPT

        self.assertIn("MUST (follow all):", DEBATE_PROPOSE_SYSTEM_PROMPT)
        self.assertIn("SHOULD:", DEBATE_PROPOSE_SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()

