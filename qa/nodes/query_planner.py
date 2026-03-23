from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

from prompts.qa_prompts import QUERY_PLANNER_SYSTEM_PROMPT, build_query_planner_user_prompt
from qa.llm_utils import invoke_llm, parse_json_object
from qa.retrieval_state import QueryPlan
from qa.retrieval_utils import normalize_text
from qa.state import EntityPack, TaskSpec


LANE_PREFERRED_SOURCES = ["openalex", "semantic_scholar", "crossref"]
LANE_KEYWORDS: Dict[str, Sequence[str]] = {
    "review": ("review", "perspective", "survey"),
    "frontier": ("recent", "latest", "state of the art"),
    "data": ("benchmark", "performance", "yield", "selectivity", "current density"),
    "contrarian": ("limitation", "negative result", "null effect", "controversy", "challenge"),
}
QUESTION_TYPE_TERMS: Dict[str, Sequence[str]] = {
    "fact": ("evidence",),
    "causal": ("effect", "impact", "cause"),
    "mechanism": ("mechanism", "pathway"),
    "comparison": ("comparison", "versus"),
    "frontier": ("recent", "advances"),
}


class QueryPlannerNode:
    def __init__(self, llm: Any = None, current_year: int = 2026) -> None:
        self.llm = llm
        self.current_year = current_year

    def run(self, task_spec: TaskSpec, entity_pack: EntityPack) -> List[QueryPlan]:
        baseline = self._build_rule_plans(task_spec=task_spec, entity_pack=entity_pack)
        if self.llm is None:
            return baseline
        try:
            raw_output = invoke_llm(
                self.llm,
                [
                    {"role": "system", "content": QUERY_PLANNER_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": build_query_planner_user_prompt(
                            question=task_spec.question,
                            task_spec=task_spec.model_dump(exclude_none=True),
                            entity_pack=entity_pack.model_dump(exclude_none=True),
                            baseline_plans=[plan.model_dump(exclude_none=True) for plan in baseline],
                        ),
                    },
                ],
            )
            parsed = parse_json_object(raw_output)
            repaired = self._repair_llm_plans(parsed, baseline)
            if repaired:
                return repaired
        except Exception:
            pass
        return baseline

    __call__ = run

    def _build_rule_plans(self, task_spec: TaskSpec, entity_pack: EntityPack) -> List[QueryPlan]:
        base_terms = self._collect_base_terms(task_spec=task_spec, entity_pack=entity_pack)
        exclude_terms = list(dict.fromkeys(task_spec.query_constraints.exclude_terms))
        year_from, year_to = self._resolve_base_year_window(task_spec)

        return [
            QueryPlan(
                lane="review",
                query_text=self._compose_query(base_terms, QUESTION_TYPE_TERMS[task_spec.question_type], LANE_KEYWORDS["review"]),
                must_terms=self._compose_must_terms(base_terms, ["review"]),
                exclude_terms=exclude_terms,
                year_from=self._lane_year_from("review", year_from),
                year_to=year_to,
                preferred_sources=LANE_PREFERRED_SOURCES,
            ),
            QueryPlan(
                lane="frontier",
                query_text=self._compose_query(base_terms, QUESTION_TYPE_TERMS[task_spec.question_type], LANE_KEYWORDS["frontier"]),
                must_terms=self._compose_must_terms(base_terms, ["recent"]),
                exclude_terms=exclude_terms,
                year_from=year_from,
                year_to=year_to,
                preferred_sources=LANE_PREFERRED_SOURCES,
            ),
            QueryPlan(
                lane="data",
                query_text=self._compose_query(base_terms, QUESTION_TYPE_TERMS[task_spec.question_type], LANE_KEYWORDS["data"]),
                must_terms=self._compose_must_terms(base_terms, ["benchmark"]),
                exclude_terms=exclude_terms,
                year_from=year_from,
                year_to=year_to,
                preferred_sources=LANE_PREFERRED_SOURCES,
            ),
            QueryPlan(
                lane="contrarian",
                query_text=self._compose_query(base_terms, QUESTION_TYPE_TERMS[task_spec.question_type], LANE_KEYWORDS["contrarian"]),
                must_terms=self._compose_must_terms(base_terms, ["limitation", "negative result"]),
                exclude_terms=exclude_terms,
                year_from=year_from,
                year_to=year_to,
                preferred_sources=LANE_PREFERRED_SOURCES,
            ),
        ]

    def _collect_base_terms(self, task_spec: TaskSpec, entity_pack: EntityPack) -> List[str]:
        anchors: List[str] = []
        for entity in entity_pack.entities:
            anchors.extend(entity.query_anchors or [])
            anchors.append(entity.canonical_name)
            if entity.mention != entity.canonical_name:
                anchors.append(entity.mention)
        for condition in entity_pack.condition_mentions:
            anchors.append(condition.raw_value)
        for term in task_spec.query_constraints.must_include_terms:
            anchors.append(term)
        for term in task_spec.query_constraints.should_include_terms:
            anchors.append(term)
        if not anchors:
            anchors.append(task_spec.normalized_question)

        unique_terms: List[str] = []
        seen = set()
        for term in anchors:
            cleaned = normalize_text(term)
            key = cleaned.lower()
            if not cleaned or key in seen:
                continue
            seen.add(key)
            unique_terms.append(cleaned)
        return unique_terms[:8]

    def _resolve_base_year_window(self, task_spec: TaskSpec) -> tuple[Optional[int], Optional[int]]:
        if task_spec.year_from is not None or task_spec.year_to is not None:
            return task_spec.year_from, task_spec.year_to
        if task_spec.question_type == "frontier":
            return self.current_year - 2, self.current_year
        if task_spec.recency_policy == "last_5y":
            return self.current_year - 4, self.current_year
        return None, None

    def _lane_year_from(self, lane: str, year_from: Optional[int]) -> Optional[int]:
        if lane == "review" and year_from is not None:
            return max(1900, year_from - 5)
        return year_from

    def _compose_query(self, base_terms: Iterable[str], question_terms: Iterable[str], lane_terms: Iterable[str]) -> str:
        parts: List[str] = []
        seen = set()
        for value in [*base_terms, *question_terms, *lane_terms]:
            cleaned = normalize_text(value)
            key = cleaned.lower()
            if not cleaned or key in seen:
                continue
            seen.add(key)
            parts.append(cleaned)
        return " ".join(parts)

    def _compose_must_terms(self, base_terms: Iterable[str], lane_terms: Iterable[str]) -> List[str]:
        must_terms: List[str] = []
        seen = set()
        for value in [*list(base_terms)[:4], *lane_terms]:
            cleaned = normalize_text(value)
            key = cleaned.lower()
            if not cleaned or key in seen:
                continue
            seen.add(key)
            must_terms.append(cleaned)
        return must_terms

    def _repair_llm_plans(self, parsed: Optional[Dict[str, Any]], baseline: Sequence[QueryPlan]) -> List[QueryPlan]:
        if not isinstance(parsed, dict):
            return list(baseline)
        raw_plans = parsed.get("plans")
        if not isinstance(raw_plans, list):
            return list(baseline)

        baseline_by_lane = {plan.lane: plan for plan in baseline}
        repaired_by_lane: Dict[str, QueryPlan] = {}
        for raw_plan in raw_plans:
            if not isinstance(raw_plan, dict):
                continue
            lane = str(raw_plan.get("lane") or "").strip()
            baseline_plan = baseline_by_lane.get(lane)
            if baseline_plan is None:
                continue
            repaired_payload = baseline_plan.model_dump(exclude_none=True)
            repaired_payload.update(
                {
                    "query_text": self._safe_text(raw_plan.get("query_text"), fallback=baseline_plan.query_text),
                    "must_terms": self._repair_text_list(raw_plan.get("must_terms"), baseline_plan.must_terms),
                    "exclude_terms": self._repair_text_list(raw_plan.get("exclude_terms"), baseline_plan.exclude_terms),
                    "year_from": self._safe_year(raw_plan.get("year_from"), fallback=baseline_plan.year_from),
                    "year_to": self._safe_year(raw_plan.get("year_to"), fallback=baseline_plan.year_to),
                    "preferred_sources": self._repair_sources(
                        raw_plan.get("preferred_sources"),
                        baseline_plan.preferred_sources,
                    ),
                }
            )
            try:
                repaired_by_lane[lane] = QueryPlan.model_validate(repaired_payload)
            except Exception:
                repaired_by_lane[lane] = baseline_plan

        return [
            repaired_by_lane.get(plan.lane, plan)
            for plan in baseline
        ]

    def _repair_text_list(self, value: Any, fallback: Sequence[str]) -> List[str]:
        if not isinstance(value, list):
            return list(fallback)
        repaired: List[str] = []
        seen = set()
        for item in value:
            cleaned = normalize_text(item)
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            repaired.append(cleaned)
        return repaired or list(fallback)

    def _repair_sources(self, value: Any, fallback: Sequence[str]) -> List[str]:
        if not isinstance(value, list):
            return list(fallback)
        allowed = {"openalex", "crossref", "semantic_scholar"}
        repaired = []
        for item in value:
            cleaned = normalize_text(item).lower()
            if cleaned in allowed and cleaned not in repaired:
                repaired.append(cleaned)
        return repaired or list(fallback)

    def _safe_year(self, value: Any, fallback: Optional[int]) -> Optional[int]:
        try:
            year = int(value)
        except (TypeError, ValueError):
            return fallback
        return year if 1900 <= year <= 2100 else fallback

    def _safe_text(self, value: Any, *, fallback: str) -> str:
        cleaned = normalize_text(value)
        return cleaned or fallback
