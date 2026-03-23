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
PLAN_LANES: Sequence[str] = ("review", "frontier", "data", "contrarian")


class QueryPlannerExecutionError(RuntimeError):
    def __init__(
        self,
        *,
        stage: str,
        reason: str,
        task_spec: TaskSpec,
        debug_payload: Dict[str, Any],
    ) -> None:
        super().__init__(reason)
        self.stage = str(stage or "").strip() or "unknown"
        self.reason = str(reason or "").strip() or "query planner execution failed"
        self.question = str(task_spec.question or "")
        self.normalized_question = str(task_spec.normalized_question or "")
        self.question_type = str(task_spec.question_type or "")
        self.debug_payload = dict(debug_payload or {})

    def to_payload(self) -> Dict[str, Any]:
        return {
            "error": "query_planner_execution_failed",
            "stage": self.stage,
            "reason": self.reason,
            "question": self.question,
            "normalized_question": self.normalized_question,
            "question_type": self.question_type,
        }


class QueryPlannerNode:
    def __init__(self, llm: Any = None, current_year: int = 2026) -> None:
        self.llm = llm
        self.current_year = current_year
        self.last_run_debug: Dict[str, Any] = {}

    def run(self, task_spec: TaskSpec, entity_pack: EntityPack) -> List[QueryPlan]:
        baseline = self._build_rule_plans(task_spec=task_spec, entity_pack=entity_pack)
        self.last_run_debug = {
            "input": {
                "task_spec": task_spec.model_dump(exclude_none=True),
                "entity_pack": entity_pack.model_dump(exclude_none=True),
            },
            "prompt_baseline_plans": [plan.model_dump(exclude_none=True) for plan in baseline],
        }
        if self.llm is None:
            self._raise_failure(
                stage="startup",
                reason="query planner LLM is unavailable",
                task_spec=task_spec,
            )
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
        except Exception as exc:
            self._raise_failure(
                stage="planning",
                reason=f"query planner LLM call failed: {exc}",
                task_spec=task_spec,
            )

        self.last_run_debug["raw_output"] = raw_output
        parsed = parse_json_object(raw_output)
        if parsed is None:
            self._raise_failure(
                stage="planning",
                reason="query planner returned unusable output",
                task_spec=task_spec,
            )

        repaired = self._validate_llm_plans(parsed)
        if repaired is None:
            self._raise_failure(
                stage="planning",
                reason="query planner returned unusable output",
                task_spec=task_spec,
            )

        self.last_run_debug["output"] = [plan.model_dump(exclude_none=True) for plan in repaired]
        return repaired

    __call__ = run

    def _raise_failure(
        self,
        *,
        stage: str,
        reason: str,
        task_spec: TaskSpec,
    ) -> None:
        failure_payload = {
            "error": "query_planner_execution_failed",
            "stage": str(stage or "").strip() or "unknown",
            "reason": str(reason or "").strip() or "query planner execution failed",
        }
        self.last_run_debug["failure"] = failure_payload
        self.last_run_debug.pop("output", None)
        raise QueryPlannerExecutionError(
            stage=failure_payload["stage"],
            reason=failure_payload["reason"],
            task_spec=task_spec,
            debug_payload=self.last_run_debug,
        )

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

    def _validate_llm_plans(self, parsed: Optional[Dict[str, Any]]) -> Optional[List[QueryPlan]]:
        if not isinstance(parsed, dict):
            return None
        raw_plans = parsed.get("plans")
        if not isinstance(raw_plans, list):
            return None
        if len(raw_plans) != len(PLAN_LANES):
            return None

        repaired_by_lane: Dict[str, QueryPlan] = {}
        for raw_plan in raw_plans:
            if not isinstance(raw_plan, dict):
                return None
            lane = str(raw_plan.get("lane") or "").strip()
            if lane not in PLAN_LANES or lane in repaired_by_lane:
                return None
            query_text = normalize_text(raw_plan.get("query_text"))
            must_terms = self._validate_text_list(raw_plan.get("must_terms"))
            exclude_terms = self._validate_text_list(raw_plan.get("exclude_terms"))
            preferred_sources = self._validate_sources(raw_plan.get("preferred_sources"))
            if not query_text or must_terms is None or exclude_terms is None or preferred_sources is None:
                return None
            try:
                repaired_payload = {
                    "lane": lane,
                    "query_text": query_text,
                    "must_terms": must_terms,
                    "exclude_terms": exclude_terms,
                    "year_from": self._validate_year(raw_plan.get("year_from")),
                    "year_to": self._validate_year(raw_plan.get("year_to")),
                    "preferred_sources": preferred_sources,
                }
            except ValueError:
                return None
            try:
                repaired_by_lane[lane] = QueryPlan.model_validate(repaired_payload)
            except Exception:
                return None

        if set(repaired_by_lane) != set(PLAN_LANES):
            return None
        return [repaired_by_lane[lane] for lane in PLAN_LANES]

    def _validate_text_list(self, value: Any) -> Optional[List[str]]:
        if not isinstance(value, list):
            return None
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
        return repaired

    def _validate_sources(self, value: Any) -> Optional[List[str]]:
        if not isinstance(value, list):
            return None
        allowed = {"openalex", "crossref", "semantic_scholar"}
        repaired = []
        for item in value:
            cleaned = normalize_text(item).lower()
            if cleaned in allowed and cleaned not in repaired:
                repaired.append(cleaned)
        return repaired or None

    def _validate_year(self, value: Any) -> Optional[int]:
        if value is None or value == "":
            return None
        try:
            year = int(value)
        except (TypeError, ValueError):
            raise ValueError("invalid year")
        if not 1900 <= year <= 2100:
            raise ValueError("invalid year")
        return year
