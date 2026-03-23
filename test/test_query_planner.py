from __future__ import annotations

import unittest

from qa.nodes.query_planner import QueryPlannerExecutionError, QueryPlannerNode
from test.qa_test_utils import make_entity_pack, make_task_spec


def _plan(
    lane: str,
    query_text: str,
    *,
    must_terms: list[str] | None = None,
    exclude_terms: list[str] | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    preferred_sources: list[str] | None = None,
) -> dict:
    return {
        "lane": lane,
        "query_text": query_text,
        "must_terms": list(must_terms or ["Pt/C", "HER"]),
        "exclude_terms": list(exclude_terms or []),
        "year_from": year_from,
        "year_to": year_to,
        "preferred_sources": list(preferred_sources or ["openalex", "semantic_scholar"]),
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


class QueryPlannerNodeTests(unittest.TestCase):
    def test_mainline_llm_returns_strict_four_lane_plan(self):
        task_spec = make_task_spec(question_type="mechanism")
        entity_pack = make_entity_pack()
        llm = _FakeLLM(
            [
                {
                    "plans": [
                        _plan("review", "Pt/C HER alkaline review"),
                        _plan("frontier", "Pt/C HER recent advances", year_from=2023, year_to=2026),
                        _plan("data", "Pt/C HER benchmark overpotential"),
                        _plan("contrarian", "Pt/C HER limitations null results", exclude_terms=["hydrazine"]),
                    ]
                }
            ]
        )
        planner = QueryPlannerNode(llm=llm, current_year=2026)

        result = planner.run(task_spec=task_spec, entity_pack=entity_pack)

        self.assertEqual(["review", "frontier", "data", "contrarian"], [item.lane for item in result])
        self.assertEqual(1, len(llm.calls))
        self.assertNotIn("failure", planner.last_run_debug)

    def test_missing_llm_raises_typed_error(self):
        task_spec = make_task_spec()
        entity_pack = make_entity_pack()
        planner = QueryPlannerNode(llm=None, current_year=2026)

        with self.assertRaises(QueryPlannerExecutionError) as ctx:
            planner.run(task_spec=task_spec, entity_pack=entity_pack)

        self.assertEqual("startup", ctx.exception.stage)
        self.assertIn("LLM is unavailable", ctx.exception.reason)
        self.assertEqual("query_planner_execution_failed", planner.last_run_debug["failure"]["error"])

    def test_partial_lane_output_raises_typed_error(self):
        task_spec = make_task_spec(question_type="frontier")
        entity_pack = make_entity_pack()
        llm = _FakeLLM(
            [
                {
                    "plans": [
                        _plan("review", "Pt/C HER alkaline review"),
                        _plan("frontier", "Pt/C HER recent advances", year_from=2024, year_to=2026),
                        _plan("data", "Pt/C HER benchmark overpotential"),
                    ]
                }
            ]
        )
        planner = QueryPlannerNode(llm=llm, current_year=2026)

        with self.assertRaises(QueryPlannerExecutionError) as ctx:
            planner.run(task_spec=task_spec, entity_pack=entity_pack)

        self.assertEqual("planning", ctx.exception.stage)
        self.assertIn("returned unusable output", ctx.exception.reason)
        self.assertIn("failure", planner.last_run_debug)


if __name__ == "__main__":
    unittest.main()
