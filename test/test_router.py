from __future__ import annotations

import unittest

from qa.nodes.router import RouterExecutionError, RouterNode


def _semantic_response(
    *,
    primary_question_type: str,
    secondary_candidates: list[str] | None = None,
    semantic_confidence: float = 0.86,
    needs_disambiguation: bool = False,
    comparison_intent: bool = False,
    comparison_targets_present: bool = False,
    explicit_metric_requested: bool = False,
    explicit_time_intent: str = "none",
    mechanistic_intent: bool = False,
    causal_intent: bool = False,
    frontier_intent: bool = False,
    notes_on_ambiguity: list[str] | None = None,
) -> dict:
    return {
        "primary_question_type": primary_question_type,
        "secondary_candidates": list(secondary_candidates or []),
        "semantic_confidence": semantic_confidence,
        "needs_disambiguation": needs_disambiguation,
        "comparison_intent": comparison_intent,
        "comparison_targets_present": comparison_targets_present,
        "explicit_metric_requested": explicit_metric_requested,
        "explicit_time_intent": explicit_time_intent,
        "mechanistic_intent": mechanistic_intent,
        "causal_intent": causal_intent,
        "frontier_intent": frontier_intent,
        "notes_on_ambiguity": list(notes_on_ambiguity or []),
    }


def _task_spec_response(
    *,
    question: str,
    normalized_question: str,
    question_type: str,
    recency_policy: str = "none",
    year_from: int | None = None,
    year_to: int | None = None,
    answer_sections: list[dict] | None = None,
    required_condition_axes: list[str] | None = None,
    query_constraints: dict | None = None,
    ambiguity_flags: list[dict] | None = None,
    router_confidence: float = 0.9,
) -> dict:
    return {
        "version": "1.0",
        "question": question,
        "normalized_question": normalized_question,
        "question_type": question_type,
        "recency_policy": recency_policy,
        "year_from": year_from,
        "year_to": year_to,
        "answer_sections": list(answer_sections or []),
        "required_condition_axes": list(required_condition_axes or []),
        "query_constraints": query_constraints
        or {
            "must_include_terms": [],
            "should_include_terms": [],
            "exclude_terms": [],
            "preferred_entity_types": [],
            "allow_broad_expansion": question_type in {"frontier", "comparison"},
        },
        "ambiguity_flags": list(ambiguity_flags or []),
        "router_confidence": router_confidence,
    }


class _FakeLLM:
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.calls: list[list[dict]] = []

    def invoke(self, messages):
        self.calls.append(messages)
        if not self.responses:
            raise AssertionError("LLM invoked more times than expected.")
        return self.responses.pop(0)


class RouterNodeTests(unittest.TestCase):
    def test_mainline_llm_classifies_canonical_question(self):
        question = "Does Pt/C affect HER activity in 1 M KOH?"
        llm = _FakeLLM(
            [
                _semantic_response(
                    primary_question_type="causal",
                    causal_intent=True,
                    explicit_metric_requested=True,
                ),
                _task_spec_response(
                    question=question,
                    normalized_question=question,
                    question_type="causal",
                    required_condition_axes=["electrolyte"],
                    query_constraints={
                        "must_include_terms": ["Pt/C", "HER"],
                        "should_include_terms": ["effect", "activity"],
                        "exclude_terms": [],
                        "preferred_entity_types": ["catalyst", "reaction"],
                        "allow_broad_expansion": False,
                    },
                ),
            ]
        )
        router = RouterNode(llm=llm, current_year=2026)

        result = router.run(question)

        self.assertEqual("causal", result.question_type)
        self.assertEqual(2, len(llm.calls))
        self.assertNotIn("fallback_reason", router.last_run_debug)

    def test_mainline_llm_handles_comparison_without_explicit_vs(self):
        question = "For alkaline HER, is Pt/C or NiMo usually more active?"
        llm = _FakeLLM(
            [
                _semantic_response(
                    primary_question_type="comparison",
                    comparison_intent=True,
                    comparison_targets_present=True,
                    explicit_metric_requested=True,
                ),
                _task_spec_response(
                    question=question,
                    normalized_question=question.lower(),
                    question_type="comparison",
                    query_constraints={
                        "must_include_terms": ["Pt/C", "NiMo", "HER"],
                        "should_include_terms": ["activity"],
                        "exclude_terms": [],
                        "preferred_entity_types": ["catalyst", "reaction"],
                        "allow_broad_expansion": True,
                    },
                    router_confidence=0.88,
                ),
            ]
        )
        router = RouterNode(llm=llm, current_year=2026)

        result = router.run(question)

        self.assertEqual("comparison", result.question_type)
        self.assertTrue(result.query_constraints.allow_broad_expansion)
        self.assertNotIn("fallback_reason", router.last_run_debug)

    def test_mainline_llm_handles_mechanism_without_mechanism_keyword(self):
        question = "Why is Pt/C often more active for alkaline HER in 1 M KOH?"
        llm = _FakeLLM(
            [
                _semantic_response(
                    primary_question_type="mechanism",
                    mechanistic_intent=True,
                    explicit_metric_requested=True,
                ),
                _task_spec_response(
                    question=question,
                    normalized_question=question.lower(),
                    question_type="mechanism",
                    required_condition_axes=["electrolyte"],
                    router_confidence=0.87,
                ),
            ]
        )
        router = RouterNode(llm=llm, current_year=2026)

        result = router.run(question)

        self.assertEqual("mechanism", result.question_type)
        self.assertNotIn("fallback_reason", router.last_run_debug)

    def test_mainline_llm_coerces_why_does_compared_with_to_causal(self):
        question = "Why does Pt/C generally improve HER activity in alkaline electrolyte compared with bare carbon?"
        llm = _FakeLLM(
            [
                _semantic_response(
                    primary_question_type="comparison",
                    comparison_intent=True,
                    comparison_targets_present=True,
                    causal_intent=True,
                ),
                _task_spec_response(
                    question=question,
                    normalized_question=question.lower(),
                    question_type="comparison",
                    required_condition_axes=["electrolyte"],
                    router_confidence=0.86,
                ),
            ]
        )
        router = RouterNode(llm=llm, current_year=2026)

        result = router.run(question)

        self.assertEqual("causal", result.question_type)
        self.assertNotIn("fallback_reason", router.last_run_debug)

    def test_mainline_llm_handles_frontier_recent_progress_question(self):
        question = "What has been the recent progress in alkaline HER catalysts beyond Pt/C?"
        llm = _FakeLLM(
            [
                _semantic_response(
                    primary_question_type="frontier",
                    frontier_intent=True,
                    explicit_time_intent="recent",
                ),
                _task_spec_response(
                    question=question,
                    normalized_question=question.lower(),
                    question_type="frontier",
                    recency_policy="last_3y",
                    year_from=2024,
                    year_to=2026,
                    query_constraints={
                        "must_include_terms": ["Pt/C", "HER"],
                        "should_include_terms": ["recent review", "state of the art"],
                        "exclude_terms": [],
                        "preferred_entity_types": ["catalyst", "reaction"],
                        "allow_broad_expansion": True,
                    },
                    router_confidence=0.9,
                ),
            ]
        )
        router = RouterNode(llm=llm, current_year=2026)

        result = router.run(question)

        self.assertEqual("frontier", result.question_type)
        self.assertEqual("last_3y", result.recency_policy)
        self.assertEqual(2024, result.year_from)
        self.assertEqual(2026, result.year_to)

    def test_semantic_stage_failure_raises_typed_error(self):
        question = "Why does Pt/C improve HER activity in 1 M KOH?"
        llm = _FakeLLM(["not json"])
        router = RouterNode(llm=llm, current_year=2026)

        with self.assertRaises(RouterExecutionError) as ctx:
            router.run(question)

        self.assertEqual(1, len(llm.calls))
        self.assertEqual("semantic", ctx.exception.stage)
        self.assertIn("semantic stage returned unusable output", ctx.exception.reason)
        self.assertIn("semantic stage returned unusable output", router.last_run_debug["failure"]["reason"])

    def test_localization_stage_failure_raises_typed_error(self):
        question = "What has been the recent progress in alkaline HER catalysts beyond Pt/C?"
        llm = _FakeLLM(
            [
                _semantic_response(
                    primary_question_type="frontier",
                    frontier_intent=True,
                    explicit_time_intent="recent",
                ),
                "not json",
            ]
        )
        router = RouterNode(llm=llm, current_year=2026)

        with self.assertRaises(RouterExecutionError) as ctx:
            router.run(question)

        self.assertEqual(2, len(llm.calls))
        self.assertEqual("localization", ctx.exception.stage)
        self.assertIn("localization stage returned unusable output", ctx.exception.reason)
        self.assertIn("localization stage returned unusable output", router.last_run_debug["failure"]["reason"])

    def test_field_level_repair_keeps_llm_mainline_result(self):
        question = "Why is Pt/C more active for alkaline HER in 1 M KOH?"
        llm = _FakeLLM(
            [
                _semantic_response(
                    primary_question_type="mechanism",
                    mechanistic_intent=True,
                    explicit_metric_requested=True,
                ),
                _task_spec_response(
                    question=question,
                    normalized_question=question.lower(),
                    question_type="mechanism",
                    answer_sections=[],
                    required_condition_axes=["bogus_axis"],
                    query_constraints={
                        "must_include_terms": ["Pt/C"],
                        "should_include_terms": ["mechanism"],
                        "exclude_terms": [],
                        "preferred_entity_types": ["bad_type"],
                        "allow_broad_expansion": False,
                    },
                    ambiguity_flags=[{"flag_type": "invalid_flag", "target": "x", "note": "bad", "severity": "high"}],
                    router_confidence=0.82,
                ),
            ]
        )
        router = RouterNode(llm=llm, current_year=2026)

        result = router.run(question)

        self.assertEqual("mechanism", result.question_type)
        self.assertGreaterEqual(len(result.answer_sections), 3)
        self.assertEqual(["electrolyte"], result.required_condition_axes)
        self.assertEqual([], result.query_constraints.preferred_entity_types)
        self.assertNotIn("fallback_reason", router.last_run_debug)


if __name__ == "__main__":
    unittest.main()
