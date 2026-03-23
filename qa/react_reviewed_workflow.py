from __future__ import annotations

import copy
import json
import logging
import re
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from agents.chat_models import build_chat_model_from_config, describe_chat_model_config
from agents import react_tool_schemas as tool_schemas
from agents.react_agent import AgentResponse, ReActAgent, ToolResult
from agents.react_reasoning import ReActStep, ReActTrajectory, ToolCallRecord
from prompts.react_reviewed import (
    build_proposer_action_prompt,
    build_proposer_repair_system_prompt,
    build_proposer_system_prompt,
    build_proposer_thought_prompt,
    build_proposer_user_prompt,
    build_review_prompt_contract,
    build_reviewer_action_prompt,
    build_reviewer_system_prompt,
    build_reviewer_thought_prompt,
    build_reviewer_user_prompt,
    build_screening_system_prompt,
    build_submission_prompt_contract,
    build_submission_prompt_scaffold,
)
from qa.artifacts import QAArtifactStore
from qa.evidence import EvidenceExtractor
from qa.handoff import EvidenceExtractorHandoff
from qa.llm_utils import invoke_llm, parse_json_payload
from qa.nodes.document_acquirer import DocumentAcquirerNode
from qa.nodes.entity_resolver import EntityResolverNode
from qa.nodes.query_planner import QueryPlannerExecutionError, QueryPlannerNode
from qa.nodes.retriever import RetrieverNode
from qa.nodes.router import RouterExecutionError, RouterNode
from qa.react_reviewed_state import (
    AnswerSubmission,
    ReviewCompletionStatus,
    ReviewItem,
    ReviewResponse,
    ReviewerRole,
    ReviewerRunStatus,
    SubmissionCitation,
    SubmissionConfidenceRating,
    SubmissionCycleState,
    SubmissionSection,
    SubmissionStepRef,
    SubmissionTraceItem,
)
from qa.retrieval_state import (
    EvidenceItem,
    PaperCandidate,
    PaperRecord,
    QueryPlan,
    RetrievalDiagnosticRecord,
    SectionIndex,
    SectionTextView,
)
from qa.state import EntityPack, TaskSpec
from qa.synthesis_state import AnswerSectionOutput, CitationRecord, ConfidenceRating, QAResult, SectionConfidenceRecord


logger = logging.getLogger("MAD.qa.react_reviewed")

MIN_REACT_REVIEWED_PROPOSER_STEPS = 6

PROPOSER_TOOL_NAMES = (
    "plan_queries",
    "search_papers",
    "screen_papers",
    "acquire_document",
    "read_sections",
    "extract_evidence",
    "fetch_citation_context",
    "inspect_entity_cache",
    "inspect_submission_anchor",
    "analyze_submission_gap",
    "conclude",
)
REVIEWER_TOOL_NAMES: Dict[str, Tuple[str, ...]] = {
    "search_coverage": (
        "inspect_entity_cache",
        "inspect_submission_anchor",
        "read_sections",
        "search_papers",
        "acquire_document",
        "conclude",
    ),
    "evidence_trace": (
        "inspect_entity_cache",
        "inspect_submission_anchor",
        "fetch_citation_context",
        "read_sections",
        "conclude",
    ),
    "reasoning_consistency": (
        "inspect_entity_cache",
        "inspect_submission_anchor",
        "read_sections",
        "conclude",
    ),
    "counterevidence": (
        "inspect_entity_cache",
        "inspect_submission_anchor",
        "search_papers",
        "acquire_document",
        "read_sections",
        "extract_evidence",
        "conclude",
    ),
}
DEFAULT_REVIEWER_ROLES: Tuple[ReviewerRole, ...] = (
    "search_coverage",
    "evidence_trace",
    "reasoning_consistency",
    "counterevidence",
)
DEFAULT_REVIEWER_BUDGET_BY_ROLE: Dict[ReviewerRole, int] = {
    "search_coverage": 1,
    "evidence_trace": 0,
    "reasoning_consistency": 0,
    "counterevidence": 2,
}

_SCREENING_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "be",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "media",
    "of",
    "on",
    "or",
    "the",
    "to",
    "under",
    "vs",
    "what",
    "when",
    "where",
    "which",
    "why",
}


def _lazy_structured_tool_import():
    try:
        from langchain_core.tools import StructuredTool
    except Exception:
        return None
    return StructuredTool


def _json_preview(payload: Any, *, limit: int = 1200) -> str:
    try:
        text = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    except Exception:
        text = str(payload)
    if len(text) > limit:
        return text[:limit] + "\n...(truncated)"
    return text


def _compact_text(value: Any) -> str:
    text = " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split())
    return text.strip()


def _tool_plain_payload(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    return value


def _merge_unique_text(existing: Optional[Sequence[str]], extra: Optional[Sequence[str]]) -> List[str]:
    merged: List[str] = []
    seen = set()
    for item in [*(existing or []), *(extra or [])]:
        cleaned = str(item or "").strip()
        key = cleaned.lower()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        merged.append(cleaned)
    return merged


def _paper_title_fallback(workspace: "ReactReviewedWorkspace", paper_id: str) -> Optional[str]:
    record = workspace.paper_records.get(str(paper_id or "").strip())
    if record is not None and _compact_text(getattr(record, "title", None)):
        return _compact_text(getattr(record, "title", None))
    candidate = workspace.paper_candidates.get(str(paper_id or "").strip())
    if candidate is not None and _compact_text(getattr(candidate, "title", None)):
        return _compact_text(getattr(candidate, "title", None))
    return None


def _primary_screen_entity_terms(entity_pack: EntityPack) -> List[str]:
    terms: List[str] = []
    candidate_axes = {"catalyst", "material", "substrate", "electrode", "support"}
    for condition in list(getattr(entity_pack, "condition_mentions", []) or []):
        axis = str(getattr(condition, "axis", "") or "").strip().lower()
        if axis not in candidate_axes:
            continue
        for value in (
            getattr(condition, "raw_value", None),
            getattr(condition, "normalized_value", None),
        ):
            cleaned = _compact_text(value).lower()
            if cleaned:
                terms.append(cleaned)
    for entity in list(getattr(entity_pack, "entities", []) or []):
        entity_type = str(getattr(entity, "entity_type", "") or "").strip().lower()
        if entity_type not in {"catalyst", "material", "molecule", "electrode", "support"}:
            continue
        for value in (
            getattr(entity, "canonical_name", None),
            *(list(getattr(entity, "aliases", []) or [])),
            *(list(getattr(entity, "query_anchors", []) or [])),
        ):
            cleaned = _compact_text(value).lower()
            if cleaned:
                terms.append(cleaned)
    deduped: List[str] = []
    seen = set()
    for term in sorted(terms, key=len, reverse=True):
        if term in seen:
            continue
        seen.add(term)
        deduped.append(term)
    return deduped


def _is_review_like_title(title: Optional[str]) -> bool:
    normalized_title = _compact_text(title).lower()
    return any(
        marker in normalized_title
        for marker in ("review", "perspective", "overview", "progress", "advance", "advances", "descriptor")
    )


def _mentions_primary_entity_only_as_comparator(*, abstract: Optional[str], primary_terms: Sequence[str]) -> bool:
    normalized_abstract = _compact_text(abstract).lower()
    if not normalized_abstract:
        return False
    comparator_markers = (
        "commercial ",
        "benchmark ",
        "compared to ",
        "compared with ",
        "than ",
        "vs ",
        "versus ",
        "relative to ",
    )
    for term in list(primary_terms or [])[:4]:
        normalized_term = _compact_text(term).lower()
        if not normalized_term:
            continue
        start = normalized_abstract.find(normalized_term)
        if start < 0:
            continue
        window_start = max(0, start - 32)
        window_end = min(len(normalized_abstract), start + len(normalized_term) + 32)
        window = normalized_abstract[window_start:window_end]
        if any(marker in window for marker in comparator_markers):
            return True
    return False


def _screen_query_terms(task_spec: TaskSpec, entity_pack: EntityPack) -> List[str]:
    terms: List[str] = []
    for entity in entity_pack.entities:
        canonical = _compact_text(entity.canonical_name).lower()
        if canonical:
            terms.append(canonical)
        for anchor in entity.query_anchors:
            cleaned = _compact_text(anchor).lower()
            if cleaned:
                terms.append(cleaned)
    for term in task_spec.query_constraints.must_include_terms:
        cleaned = _compact_text(term).lower()
        if cleaned:
            terms.append(cleaned)
    for token in re.findall(r"[a-z0-9][a-z0-9/+\-.]*", str(task_spec.normalized_question or "").lower()):
        if len(token) < 3 and not any(char.isdigit() for char in token):
            continue
        if token in _SCREENING_STOPWORDS:
            continue
        terms.append(token)
    deduped: List[str] = []
    seen = set()
    for term in terms:
        if term in seen:
            continue
        seen.add(term)
        deduped.append(term)
    return deduped


def _paper_relevance_metrics(
    *,
    task_spec: TaskSpec,
    entity_pack: EntityPack,
    title: Optional[str],
    abstract: Optional[str],
) -> Dict[str, Any]:
    normalized_title = _compact_text(title).lower()
    normalized_abstract = _compact_text(abstract).lower()
    corpus = f"{normalized_title} {normalized_abstract}".strip()
    query_terms = _screen_query_terms(task_spec, entity_pack)
    title_matches = [term for term in query_terms if term and term in normalized_title]
    abstract_matches = [term for term in query_terms if term and term in normalized_abstract]
    corpus_matches = [term for term in query_terms if term and term in corpus]
    question_terms = [
        term
        for term in re.findall(r"[a-z0-9][a-z0-9/+\-.]*", str(task_spec.normalized_question or "").lower())
        if (len(term) >= 3 or any(char.isdigit() for char in term)) and term not in _SCREENING_STOPWORDS
    ]
    question_hits = [term for term in question_terms if term and term in corpus]
    return {
        "title_hits": len(title_matches),
        "abstract_hits": len(abstract_matches),
        "question_hits": len(question_hits),
        "matched_terms": corpus_matches[:8],
        "has_abstract": bool(normalized_abstract),
    }


def _paper_has_useful_abstract_support(
    *,
    task_spec: TaskSpec,
    entity_pack: EntityPack,
    title: Optional[str],
    abstract: Optional[str],
) -> bool:
    metrics = _paper_relevance_metrics(
        task_spec=task_spec,
        entity_pack=entity_pack,
        title=title,
        abstract=abstract,
    )
    if not metrics["has_abstract"]:
        return False
    if metrics["abstract_hits"] >= 2:
        return True
    if metrics["abstract_hits"] >= 1 and metrics["title_hits"] >= 1:
        return True
    return metrics["question_hits"] >= 2


def _review_item_priority_terms(open_review_items: Sequence[ReviewItem]) -> List[str]:
    source_text = " ".join(
        _compact_text(part)
        for item in list(open_review_items or [])
        for part in (item.critique, item.required_action)
        if _compact_text(part)
    ).lower()
    candidate_terms = [
        "overpotential",
        "tafel",
        "exchange current",
        "mass activity",
        "specific activity",
        "pt foil",
        "pt black",
        "unsupported pt",
        "commercial pt/c",
        "baseline",
        "comparator",
        "1.0 m koh",
        "0.1 m koh",
        "koh",
        "naoh",
        "rde",
        "mea",
    ]
    return [term for term in candidate_terms if term in source_text]


def _evidence_item_priority_score(item: EvidenceItem) -> float:
    snippet = _compact_text(item.snippet)
    lowered_snippet = snippet.lower()
    lowered_notes = _compact_text(item.extraction_notes).lower()
    score = 4.0 * float(getattr(item, "extraction_confidence", 0.0) or 0.0)
    if str(getattr(item, "source_layer", "") or "").strip().lower() == "fulltext":
        score += 3.0
    if str(getattr(item, "role", "") or "").strip().lower() == "observation":
        score += 1.4
    elif str(getattr(item, "role", "") or "").strip().lower() == "mechanism":
        score += 1.0
    elif str(getattr(item, "role", "") or "").strip().lower() == "condition":
        score += 0.8
    score += 0.7 * float(min(len(list(getattr(item, "metric_mentions", []) or [])), 3))
    score += 0.45 * float(min(len(list(getattr(item, "conditions", {}) or {})), 3))
    if str(getattr(item, "claim_polarity", "") or "").strip().lower() in {"support", "oppose"}:
        score += 0.5
    if str(getattr(item, "section_type", "") or "").strip().lower() in {"results", "discussion", "abstract"}:
        score += 0.4
    if len(snippet) < 40:
        score -= 2.0
    if len(snippet.split()) < 6:
        score -= 1.5
    if snippet.endswith(("0.", "between 0.", "between 0")):
        score -= 1.0
    if any(
        marker in lowered_notes
        for marker in ("administrative", "address-like", "truncated", "incomplete", "no comparative outcome")
    ):
        score -= 2.5
    if any(marker in lowered_snippet for marker in ("institut universitaire", "boulevard", "paris - france")):
        score -= 3.0
    return round(score, 4)


def _align_step_refs_to_trajectory(raw_payload: Dict[str, Any], trajectory_id: Optional[str]) -> Dict[str, Any]:
    normalized_trajectory_id = _compact_text(trajectory_id)
    if not normalized_trajectory_id:
        return raw_payload

    def _normalize_step_refs(values: Any) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for item in list(_normalize_list_payload(values) or []):
            if isinstance(item, dict):
                step_number = item.get("step_number")
                try:
                    parsed_step_number = max(1, int(step_number))
                except (TypeError, ValueError):
                    parsed_step_number = 1
                normalized.append(
                    {
                        **item,
                        "trajectory_id": normalized_trajectory_id,
                        "step_number": parsed_step_number,
                    }
                )
        return normalized or [{"trajectory_id": normalized_trajectory_id, "step_number": 1}]

    raw_payload["trajectory_id"] = normalized_trajectory_id
    raw_payload["step_refs"] = _normalize_step_refs(raw_payload.get("step_refs"))
    normalized_sections: List[Dict[str, Any]] = []
    for section_payload in list(_normalize_list_payload(raw_payload.get("sections")) or []):
        if not isinstance(section_payload, dict):
            continue
        current = copy.deepcopy(section_payload)
        current["step_refs"] = _normalize_step_refs(current.get("step_refs"))
        normalized_sections.append(current)
    if normalized_sections:
        raw_payload["sections"] = normalized_sections
    return raw_payload


def _confidence(level_score: float, rationale: str) -> SubmissionConfidenceRating:
    score = max(0.0, min(round(float(level_score), 2), 1.0))
    if score >= 0.75:
        level = "high"
    elif score >= 0.45:
        level = "medium"
    else:
        level = "low"
    return SubmissionConfidenceRating(level=level, score=score, rationale=str(rationale).strip() or "No rationale.")


def _normalize_confidence_payload(value: Any, *, rationale: str) -> Any:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _confidence(float(value), rationale).model_dump()
    return value


def _normalize_list_payload(value: Any) -> Any:
    if isinstance(value, dict):
        if not value:
            return []
        dict_values = list(value.values())
        if all(isinstance(item, dict) for item in dict_values):
            return dict_values
    return value


def _result_confidence(level_score: float, rationale: str) -> ConfidenceRating:
    rated = _confidence(level_score, rationale)
    return ConfidenceRating(level=rated.level, score=rated.score, rationale=rated.rationale)


def _build_submission_trace(submission: AnswerSubmission) -> List[SubmissionTraceItem]:
    return [
        SubmissionTraceItem(
            section_id=section.section_id,
            citation_ids=list(section.citation_ids),
            step_refs=list(section.step_refs),
            issue_refs=list(section.issue_refs),
        )
        for section in submission.sections
    ]


def _submission_to_citation_records(submission: AnswerSubmission) -> List[CitationRecord]:
    return [
        CitationRecord(
            citation_id=item.citation_id,
            paper_id=item.paper_id,
            doi=item.doi,
            title=item.title,
            year=item.year,
            venue=item.venue,
            supporting_claim_ids=[],
        )
        for item in submission.citations
    ]


def _assemble_final_answer(sections: Sequence[AnswerSectionOutput]) -> str:
    blocks: List[str] = []
    for section in sections:
        content = str(section.content or "").strip()
        if not content:
            continue
        blocks.append(f"## {section.title}\n{content}")
    return "\n\n".join(blocks).strip()


def _serialize_trajectory(trajectory: ReActTrajectory) -> Dict[str, Any]:
    return trajectory.to_dict()


def _step_ref(trajectory: ReActTrajectory, step_number: int) -> SubmissionStepRef:
    return SubmissionStepRef(trajectory_id=trajectory.trajectory_id, step_number=step_number)


@dataclass
class ReviewerBudgetState:
    role: ReviewerRole
    budget_limit: int
    actions_used: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    blocked_calls: int = 0
    charged_tools: Dict[str, int] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def _record_event(
        self,
        *,
        event_type: str,
        tool_name: str,
        cache_key: str,
        charged: bool,
        requested_via: Optional[str] = None,
    ) -> None:
        self.events.append(
            {
                "timestamp": round(time.time(), 6),
                "event": event_type,
                "tool": tool_name,
                "requested_via": requested_via or tool_name,
                "cache_key": cache_key,
                "charged": bool(charged),
                "actions_used": self.actions_used,
                "budget_limit": self.budget_limit,
            }
        )

    def try_charge(self, *, tool_name: str, cache_key: str, requested_via: Optional[str] = None) -> bool:
        with self._lock:
            if self.actions_used >= self.budget_limit:
                self.blocked_calls += 1
                self._record_event(
                    event_type="blocked",
                    tool_name=tool_name,
                    cache_key=cache_key,
                    charged=False,
                    requested_via=requested_via,
                )
                return False
            self.actions_used += 1
            self.cache_misses += 1
            self.charged_tools[tool_name] = self.charged_tools.get(tool_name, 0) + 1
            self._record_event(
                event_type="miss",
                tool_name=tool_name,
                cache_key=cache_key,
                charged=True,
                requested_via=requested_via,
            )
            return True

    def record_hit(self, *, tool_name: str, cache_key: str, requested_via: Optional[str] = None) -> None:
        with self._lock:
            self.cache_hits += 1
            self._record_event(
                event_type="hit",
                tool_name=tool_name,
                cache_key=cache_key,
                charged=False,
                requested_via=requested_via,
            )

    def record_miss(self, *, tool_name: str, cache_key: str, requested_via: Optional[str] = None) -> None:
        with self._lock:
            self._record_event(
                event_type="miss",
                tool_name=tool_name,
                cache_key=cache_key,
                charged=False,
                requested_via=requested_via,
            )

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "role": self.role,
                "budget_limit": self.budget_limit,
                "actions_used": self.actions_used,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "blocked_calls": self.blocked_calls,
                "charged_tools": dict(self.charged_tools),
                "events": list(self.events),
            }


@dataclass
class ReviewerSession:
    reviewer_role: ReviewerRole
    cycle_number: int
    artifact_store: QAArtifactStore
    budget_state: ReviewerBudgetState

    def cache_key_text(self, cache_key: Any) -> str:
        return json.dumps(cache_key, ensure_ascii=False, sort_keys=True, default=str)

    def record_hit(self, *, tool_name: str, cache_key: Any, requested_via: Optional[str] = None) -> None:
        self.budget_state.record_hit(
            tool_name=tool_name,
            cache_key=self.cache_key_text(cache_key),
            requested_via=requested_via,
        )

    def try_charge(self, *, tool_name: str, cache_key: Any, requested_via: Optional[str] = None) -> bool:
        return self.budget_state.try_charge(
            tool_name=tool_name,
            cache_key=self.cache_key_text(cache_key),
            requested_via=requested_via,
        )

    def record_miss(self, *, tool_name: str, cache_key: Any, requested_via: Optional[str] = None) -> None:
        self.budget_state.record_miss(
            tool_name=tool_name,
            cache_key=self.cache_key_text(cache_key),
            requested_via=requested_via,
        )

    def blocked_payload(self, *, tool_name: str, cache_key: Any, requested_via: Optional[str] = None) -> Dict[str, Any]:
        summary = self.budget_state.snapshot()
        return {
            "__budget_blocked__": True,
            "status": "blocked",
            "reason": "budget_exhausted",
            "reviewer_role": self.reviewer_role,
            "tool": tool_name,
            "requested_via": requested_via or tool_name,
            "cache_key": self.cache_key_text(cache_key),
            "message": (
                f"budget exhausted; tool blocked; used {summary['actions_used']} "
                f"of {summary['budget_limit']} retrieval actions"
            ),
            "retrieval_actions_used": summary["actions_used"],
            "retrieval_budget_limit": summary["budget_limit"],
            "budget_blocked_calls": summary["blocked_calls"],
        }

    def write_budget_usage(self) -> str:
        return self.artifact_store.write_json("budget_usage.json", self.budget_state.snapshot())

    def write_run_artifacts(
        self,
        *,
        review_items: Sequence[ReviewItem],
        reviewer_trajectory: Optional[ReActTrajectory],
        reviewer_status: ReviewerRunStatus,
    ) -> None:
        self.artifact_store.write_json(
            "review_items.json",
            [item.model_dump(exclude_none=True) for item in review_items],
        )
        if reviewer_trajectory is not None:
            self.artifact_store.write_json("reviewer_trajectory.json", reviewer_trajectory.to_dict())
        self.artifact_store.write_json("reviewer_status.json", reviewer_status.model_dump(exclude_none=True))
        self.write_budget_usage()


@dataclass
class ReviewerExecutionResult:
    reviewer_role: ReviewerRole
    review_items: List[ReviewItem]
    reviewer_trajectory: Optional[ReActTrajectory]
    reviewer_status: ReviewerRunStatus


@dataclass
class _InflightOperation:
    event: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    error: Optional[BaseException] = None


class ReviewerBudgetBlocked(RuntimeError):
    def __init__(self, payload: Dict[str, Any]) -> None:
        super().__init__(str(payload.get("message") or "Reviewer retrieval budget exhausted."))
        self.payload = payload


class ReactReviewedStructuredOutputError(RuntimeError):
    def __init__(
        self,
        *,
        stage: str,
        cycle_number: int,
        message: str,
        reviewer_role: Optional[ReviewerRole] = None,
        response_content: Any = None,
        structured_output: Any = None,
        trajectory: Optional[ReActTrajectory] = None,
    ) -> None:
        super().__init__(str(message))
        self.stage = str(stage)
        self.cycle_number = int(cycle_number)
        self.reviewer_role = reviewer_role
        self.response_content = response_content
        self.structured_output = structured_output
        self.trajectory = trajectory


class ReactReviewedProposerExecutionError(RuntimeError):
    def __init__(
        self,
        *,
        stage: str,
        cycle_number: int,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        response_content: Any = None,
        structured_output: Any = None,
        trajectory: Optional[ReActTrajectory] = None,
    ) -> None:
        super().__init__(str(message))
        self.stage = str(stage)
        self.cycle_number = int(cycle_number)
        self.details = dict(details or {})
        self.response_content = response_content
        self.structured_output = structured_output
        self.trajectory = trajectory


class ReactReviewedReviewerExecutionError(RuntimeError):
    def __init__(
        self,
        *,
        stage: str,
        cycle_number: int,
        message: str,
        reviewer_role: ReviewerRole,
        details: Optional[Dict[str, Any]] = None,
        response_content: Any = None,
        structured_output: Any = None,
        trajectory: Optional[ReActTrajectory] = None,
    ) -> None:
        super().__init__(str(message))
        self.stage = str(stage)
        self.cycle_number = int(cycle_number)
        self.reviewer_role = reviewer_role
        self.details = dict(details or {})
        self.response_content = response_content
        self.structured_output = structured_output
        self.trajectory = trajectory


@dataclass
class _ProposerRunState:
    evidence_policy: str
    query_plan_ids: List[str] = field(default_factory=list)
    search_ordered_paper_ids: List[str] = field(default_factory=list)
    searched_paper_ids: set[str] = field(default_factory=set)
    acquired_paper_ids: set[str] = field(default_factory=set)
    evidence_ids: set[str] = field(default_factory=set)
    section_ids_by_paper: Dict[str, set[str]] = field(default_factory=dict)
    evidence_ids_by_paper: Dict[str, set[str]] = field(default_factory=dict)
    evidence_layers_by_id: Dict[str, str] = field(default_factory=dict)
    fulltext_status_by_paper: Dict[str, str] = field(default_factory=dict)
    fulltext_available_by_paper: Dict[str, bool] = field(default_factory=dict)
    search_generation: int = 0
    screening_generation: int = 0
    locked_candidate_paper_ids: List[str] = field(default_factory=list)
    dropped_candidate_paper_ids: List[str] = field(default_factory=list)
    candidate_screening: List[Dict[str, Any]] = field(default_factory=list)

    def record_plan_queries(self, payload: Sequence[Dict[str, Any]]) -> None:
        for item in list(payload or []):
            query_plan_id = str(item.get("query_plan_id") or "").strip()
            if query_plan_id and query_plan_id not in self.query_plan_ids:
                self.query_plan_ids.append(query_plan_id)

    def record_search_results(self, payload: Sequence[Dict[str, Any]]) -> None:
        self.search_generation += 1
        for item in list(payload or []):
            paper_id = str(item.get("paper_id") or "").strip()
            if paper_id:
                self.searched_paper_ids.add(paper_id)
                if paper_id not in self.search_ordered_paper_ids:
                    self.search_ordered_paper_ids.append(paper_id)

    def record_screening(self, payload: Dict[str, Any]) -> None:
        self.screening_generation = self.search_generation
        self.locked_candidate_paper_ids = [
            str(paper_id).strip()
            for paper_id in list((payload or {}).get("locked_paper_ids") or [])
            if str(paper_id).strip()
        ]
        self.dropped_candidate_paper_ids = [
            str(paper_id).strip()
            for paper_id in list((payload or {}).get("dropped_paper_ids") or [])
            if str(paper_id).strip()
        ]
        self.candidate_screening = [
            copy.deepcopy(item)
            for item in list((payload or {}).get("ranked_candidates") or [])
            if isinstance(item, dict)
        ]

    def record_acquisition(self, payload: Dict[str, Any]) -> None:
        paper_id = str((payload or {}).get("paper_id") or "").strip()
        if not paper_id:
            return
        self.acquired_paper_ids.add(paper_id)
        self.fulltext_status_by_paper[paper_id] = str((payload or {}).get("fulltext_status") or "").strip()
        self.fulltext_available_by_paper[paper_id] = bool((payload or {}).get("fulltext_available"))

    def record_sections(self, paper_id: str, payload: Sequence[Dict[str, Any]]) -> None:
        normalized_paper_id = str(paper_id or "").strip()
        if not normalized_paper_id:
            return
        bucket = self.section_ids_by_paper.setdefault(normalized_paper_id, set())
        for item in list(payload or []):
            section_id = str(item.get("section_id") or "").strip()
            if section_id:
                bucket.add(section_id)

    def record_evidence(self, paper_id: str, payload: Sequence[Dict[str, Any]]) -> None:
        normalized_paper_id = str(paper_id or "").strip()
        if not normalized_paper_id:
            return
        evidence_bucket = self.evidence_ids_by_paper.setdefault(normalized_paper_id, set())
        for item in list(payload or []):
            evidence_id = str(item.get("evidence_id") or "").strip()
            if not evidence_id:
                continue
            evidence_bucket.add(evidence_id)
            self.evidence_ids.add(evidence_id)
            layer = str(item.get("source_layer") or "").strip().lower()
            if layer:
                self.evidence_layers_by_id[evidence_id] = layer
            section_id = str(item.get("section_id") or "").strip()
            if section_id:
                self.section_ids_by_paper.setdefault(normalized_paper_id, set()).add(section_id)

    def has_fulltext_evidence(self) -> bool:
        return any(layer == "fulltext" for layer in self.evidence_layers_by_id.values())

    def has_any_evidence(self) -> bool:
        return bool(self.evidence_ids)

    def screening_required(self) -> bool:
        return bool(self.searched_paper_ids) and self.screening_generation < self.search_generation

    def fulltext_evidence_ids_for_paper(self, paper_id: str) -> set[str]:
        return {
            evidence_id
            for evidence_id in self.evidence_ids_by_paper.get(str(paper_id or "").strip(), set())
            if self.evidence_layers_by_id.get(evidence_id) == "fulltext"
        }

    def prompt_payload(self) -> Dict[str, Any]:
        return {
            "evidence_policy": self.evidence_policy,
            "query_plan_ids": list(self.query_plan_ids),
            "search_generation": self.search_generation,
            "screening_generation": self.screening_generation,
            "searched_paper_ids": sorted(self.searched_paper_ids),
            "search_ordered_paper_ids": list(self.search_ordered_paper_ids),
            "locked_candidate_paper_ids": list(self.locked_candidate_paper_ids),
            "dropped_candidate_paper_ids": list(self.dropped_candidate_paper_ids),
            "acquired_paper_ids": sorted(self.acquired_paper_ids),
            "evidence_ids": sorted(self.evidence_ids),
            "candidate_screening": copy.deepcopy(self.candidate_screening),
            "fulltext_status_by_paper": {
                paper_id: self.fulltext_status_by_paper.get(paper_id)
                for paper_id in sorted(self.fulltext_status_by_paper)
            },
            "fulltext_available_by_paper": {
                paper_id: bool(self.fulltext_available_by_paper.get(paper_id))
                for paper_id in sorted(self.fulltext_available_by_paper)
            },
            "section_ids_by_paper": {
                paper_id: sorted(section_ids)
                for paper_id, section_ids in sorted(self.section_ids_by_paper.items())
            },
            "evidence_ids_by_paper": {
                paper_id: sorted(evidence_ids)
                for paper_id, evidence_ids in sorted(self.evidence_ids_by_paper.items())
            },
            "evidence_layers_by_id": {
                evidence_id: self.evidence_layers_by_id.get(evidence_id)
                for evidence_id in sorted(self.evidence_layers_by_id)
            },
        }


@dataclass
class AcceptanceDecision:
    status: str
    blocker_codes: List[str] = field(default_factory=list)
    blocker_messages: List[str] = field(default_factory=list)
    blocking_review_ids: List[str] = field(default_factory=list)

    @property
    def accepted(self) -> bool:
        return self.status == "accepted"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "blocker_codes": list(self.blocker_codes),
            "blocker_messages": list(self.blocker_messages),
            "blocking_review_ids": list(self.blocking_review_ids),
        }


def _store_invalid_llm_output(
    *,
    artifact_store: QAArtifactStore,
    prefix: str,
    error: ReactReviewedStructuredOutputError,
) -> None:
    payload = {
        "stage": error.stage,
        "cycle_number": error.cycle_number,
        "reviewer_role": error.reviewer_role,
        "message": str(error),
        "response_content": error.response_content,
        "structured_output": error.structured_output,
    }
    artifact_store.write_json(f"diagnostics/{prefix}_invalid_response.json", payload)
    if error.trajectory is not None:
        artifact_store.write_json(
            f"diagnostics/{prefix}_invalid_trajectory.json",
            error.trajectory.to_dict(),
        )


def _store_execution_failure(
    *,
    artifact_store: QAArtifactStore,
    prefix: str,
    error: ReactReviewedProposerExecutionError,
) -> None:
    payload = {
        "stage": error.stage,
        "cycle_number": error.cycle_number,
        "message": str(error),
        "details": error.details,
        "response_content": error.response_content,
        "structured_output": error.structured_output,
    }
    artifact_store.write_json(f"diagnostics/{prefix}_failure.json", payload)
    if error.trajectory is not None:
        artifact_store.write_json(
            f"diagnostics/{prefix}_failure_trajectory.json",
            error.trajectory.to_dict(),
        )


def _store_reviewer_execution_failure(
    *,
    artifact_store: QAArtifactStore,
    prefix: str,
    error: ReactReviewedReviewerExecutionError,
) -> None:
    payload = {
        "stage": error.stage,
        "cycle_number": error.cycle_number,
        "reviewer_role": error.reviewer_role,
        "message": str(error),
        "details": error.details,
        "response_content": error.response_content,
        "structured_output": error.structured_output,
    }
    artifact_store.write_json(f"diagnostics/{prefix}_failure.json", payload)
    if error.trajectory is not None:
        artifact_store.write_json(
            f"diagnostics/{prefix}_failure_trajectory.json",
            error.trajectory.to_dict(),
        )


def _extract_stage_payload(
    *,
    response: Any,
    expected_kind: str,
) -> Any:
    structured_output = getattr(response, "structured_output", None)
    if structured_output is not None:
        if not isinstance(structured_output, dict):
            raise ValueError("structured_output must be a dict.")
        kind = str(structured_output.get("kind") or "").strip()
        payload = structured_output.get("payload")
        if kind != expected_kind:
            raise ValueError(f"structured_output kind mismatch: expected {expected_kind}, got {kind or 'missing'}.")
        return payload

    parsed = parse_json_payload(getattr(response, "content", None))
    if expected_kind == "submission":
        if isinstance(parsed, dict) and parsed.get("kind") == "submission" and isinstance(parsed.get("payload"), dict):
            return parsed.get("payload")
        if isinstance(parsed, dict) and isinstance(parsed.get("submission"), dict):
            return parsed.get("submission")
        if isinstance(parsed, dict):
            return parsed
        raise ValueError("response.content did not contain a submission JSON object.")
    if expected_kind == "review_items":
        if isinstance(parsed, dict) and parsed.get("kind") == "review_items" and isinstance(parsed.get("payload"), list):
            return parsed.get("payload")
        if isinstance(parsed, dict) and isinstance(parsed.get("review_items"), list):
            return parsed.get("review_items")
        if isinstance(parsed, dict) and isinstance(parsed.get("review"), dict) and isinstance(parsed["review"].get("review_items"), list):
            return parsed["review"].get("review_items")
        if isinstance(parsed, list):
            return parsed
        raise ValueError("response.content did not contain a review_items array.")
    raise ValueError(f"Unknown expected_kind: {expected_kind}")


def _is_reviewer_llm_completed(status: ReviewerRunStatus) -> bool:
    return status.status in {"completed", "salvaged"}


def _extract_json_candidates(value: Any) -> List[Any]:
    candidates: List[Any] = []
    if value is None:
        return candidates
    if isinstance(value, (dict, list)):
        candidates.append(value)
        return candidates
    parsed = parse_json_payload(value)
    if parsed is not None:
        candidates.append(parsed)
    text = str(value or "").strip()
    if not text:
        return candidates
    for match in re.finditer(r"(\{.*?\}|\[.*?\])", text, re.DOTALL):
        snippet = match.group(1).strip()
        parsed = parse_json_payload(snippet)
        if parsed is not None:
            candidates.append(parsed)
    return candidates


def _coerce_salvaged_submission_payload(payload: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return None
    if payload.get("kind") == "submission" and isinstance(payload.get("payload"), dict):
        return payload.get("payload")
    if isinstance(payload.get("submission"), dict):
        return payload.get("submission")
    if any(key in payload for key in ("submission_id", "sections", "citations", "overall_confidence", "trajectory_id")):
        return payload
    return None


@dataclass
class RouterAgentWrapper:
    router: RouterNode

    def run(
        self,
        *,
        question: str,
        context: Optional[str],
        artifact_store: QAArtifactStore,
    ) -> Tuple[TaskSpec, Dict[str, str]]:
        try:
            task_spec = self.router.run(question=question, context=context)
        except RouterExecutionError as exc:
            debug_payload = dict(exc.debug_payload or {})
            semantic_stage_path = None
            localization_stage_path = None
            if isinstance(debug_payload.get("semantic_stage"), dict):
                semantic_stage_path = artifact_store.write_json(
                    "router/semantic_stage.json",
                    debug_payload["semantic_stage"],
                )
            if isinstance(debug_payload.get("localization_stage"), dict):
                localization_stage_path = artifact_store.write_json(
                    "router/localization_stage.json",
                    debug_payload["localization_stage"],
                )
            failure_path = artifact_store.write_json(
                "router/failure.json",
                exc.to_payload(),
            )
            artifact_store.write_json(
                "router/agent_run.json",
                {
                    "agent": "RouterAgent",
                    "input": {"question": question, "context": context},
                    "error": exc.to_payload(),
                    "debug": debug_payload,
                },
            )
            if semantic_stage_path:
                debug_payload["semantic_stage_artifact"] = semantic_stage_path
            if localization_stage_path:
                debug_payload["localization_stage_artifact"] = localization_stage_path
            debug_payload["failure_artifact"] = failure_path
            raise

        debug_payload = dict(getattr(self.router, "last_run_debug", {}) or {})
        task_spec_path = artifact_store.write_json(
            "router/task_spec.json",
            task_spec.model_dump(exclude_none=True),
        )
        semantic_stage_path = None
        localization_stage_path = None
        if isinstance(debug_payload.get("semantic_stage"), dict):
            semantic_stage_path = artifact_store.write_json(
                "router/semantic_stage.json",
                debug_payload["semantic_stage"],
            )
        if isinstance(debug_payload.get("localization_stage"), dict):
            localization_stage_path = artifact_store.write_json(
                "router/localization_stage.json",
                debug_payload["localization_stage"],
            )
        audit_path = artifact_store.write_json(
            "router/agent_run.json",
            {
                "agent": "RouterAgent",
                "input": {"question": question, "context": context},
                "output": task_spec.model_dump(exclude_none=True),
                "debug": debug_payload,
            },
        )
        artifacts = {
            "router_task_spec": task_spec_path,
            "router_agent_run": audit_path,
        }
        if semantic_stage_path:
            artifacts["router_semantic_stage"] = semantic_stage_path
        if localization_stage_path:
            artifacts["router_localization_stage"] = localization_stage_path
        return task_spec, artifacts


@dataclass
class EntityResolverAgentWrapper:
    resolver: EntityResolverNode

    def run(
        self,
        *,
        question: str,
        task_spec: TaskSpec,
        artifact_store: QAArtifactStore,
    ) -> Tuple[EntityPack, Dict[str, str], Dict[str, Any]]:
        resolve_detailed = getattr(self.resolver, "resolve_detailed", None)
        if callable(resolve_detailed):
            resolution_result = resolve_detailed(question=question, task_spec=task_spec)
            entity_pack = resolution_result.entity_pack
            artifact_payloads = dict(resolution_result.artifact_payloads())
            resolution_snapshot = {
                "resolution_index": artifact_payloads.get("entity_resolver/resolution_index.json", {}),
                "provider_calls": artifact_payloads.get("entity_resolver/provider_calls.json", []),
                "seed_suggestions": artifact_payloads.get("entity_resolver/seed_suggestions.json", []),
            }
        else:
            entity_pack = self.resolver.run(question=question, task_spec=task_spec)
            artifact_payloads = {
                "entity_resolver/entity_pack.json": entity_pack.model_dump(exclude_none=True),
                "entity_resolver/resolution_index.json": {"entries": [], "cache_events": []},
                "entity_resolver/provider_calls.json": [],
                "entity_resolver/seed_suggestions.json": [],
            }
            resolution_snapshot = {
                "resolution_index": artifact_payloads["entity_resolver/resolution_index.json"],
                "provider_calls": [],
                "seed_suggestions": [],
            }
        artifact_paths = {
            Path(relative_path).stem: artifact_store.write_json(relative_path, payload)
            for relative_path, payload in artifact_payloads.items()
        }
        audit_path = artifact_store.write_json(
            "entity_resolver/agent_run.json",
            {
                "agent": "EntityResolverAgent",
                "input": {
                    "question": question,
                    "task_spec": task_spec.model_dump(exclude_none=True),
                },
                "output": entity_pack.model_dump(exclude_none=True),
            },
        )
        artifact_paths["entity_resolver_agent_run"] = audit_path
        return entity_pack, artifact_paths, resolution_snapshot


class ReactReviewedWorkspace:
    def __init__(
        self,
        *,
        question: str,
        context: Optional[str],
        task_spec: TaskSpec,
        entity_pack: EntityPack,
        entity_resolution_snapshot: Optional[Dict[str, Any]],
        artifact_store: QAArtifactStore,
        query_planner: QueryPlannerNode,
        retriever: RetrieverNode,
        document_acquirer: DocumentAcquirerNode,
        handoff: EvidenceExtractorHandoff,
        evidence_extractor: EvidenceExtractor,
        stage_watchdog_seconds: float = 120.0,
    ) -> None:
        self.question = question
        self.context = context
        self.task_spec = task_spec
        self.entity_pack = entity_pack
        self.entity_resolution_snapshot = copy.deepcopy(entity_resolution_snapshot or {})
        self.store = artifact_store
        self.query_planner = query_planner
        self.retriever = retriever
        self.document_acquirer = document_acquirer
        self.handoff = handoff
        self.evidence_extractor = evidence_extractor
        self.stage_watchdog_seconds = max(0.01, float(stage_watchdog_seconds))

        self._state_lock = threading.RLock()
        self.query_plans: Dict[str, QueryPlan] = {}
        self.paper_candidates: Dict[str, PaperCandidate] = {}
        self.paper_records: Dict[str, PaperRecord] = {}
        self.section_indices: Dict[str, SectionIndex] = {}
        self.evidence_items: Dict[str, EvidenceItem] = {}
        self.retrieval_diagnostics: List[RetrievalDiagnosticRecord] = []
        self.provider_health: Dict[str, Dict[str, Any]] = {}
        self.execution_warnings: List[str] = []
        self.current_submission: Optional[AnswerSubmission] = None
        self.current_proposer_trajectory: Optional[ReActTrajectory] = None
        self.current_review_items: List[ReviewItem] = []
        self.current_cycle_number: int = 1
        self._ad_hoc_query_plan_ids: Dict[Tuple[str, str], str] = {}
        self._search_result_cache: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
        self._acquire_result_cache: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        self._section_read_cache: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
        self._extract_result_cache: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
        self._citation_context_cache: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        self._search_inflight: Dict[Tuple[Any, ...], _InflightOperation] = {}
        self._acquire_inflight: Dict[Tuple[Any, ...], _InflightOperation] = {}
        self._extract_inflight: Dict[Tuple[Any, ...], _InflightOperation] = {}
        self._stage_events: List[Dict[str, Any]] = []

    def _register_query_plan(self, query_plan: QueryPlan, *, prefix: str = "qp") -> str:
        with self._state_lock:
            query_plan_id = f"{prefix}_{len(self.query_plans) + 1}"
            self.query_plans[query_plan_id] = query_plan
        return query_plan_id

    def _query_plan_signature(self, query_plan: QueryPlan) -> Tuple[Any, ...]:
        def _normalize_terms(values: Optional[Sequence[Any]]) -> Tuple[str, ...]:
            normalized = {
                self._normalize_cache_text(item)
                for item in list(values or [])
                if self._normalize_cache_text(item)
            }
            return tuple(sorted(normalized))

        return (
            str(query_plan.lane or "").strip().lower(),
            self._normalize_cache_text(query_plan.query_text),
            _normalize_terms(query_plan.must_terms),
            _normalize_terms(query_plan.exclude_terms),
            int(query_plan.year_from) if query_plan.year_from is not None else None,
            int(query_plan.year_to) if query_plan.year_to is not None else None,
        )

    def _find_registered_query_plan_id(self, query_plan: QueryPlan, *, prefix: Optional[str] = None) -> Optional[str]:
        target_signature = self._query_plan_signature(query_plan)
        with self._state_lock:
            for query_plan_id, existing in self.query_plans.items():
                if prefix and not str(query_plan_id).startswith(f"{prefix}_"):
                    continue
                if self._query_plan_signature(existing) == target_signature:
                    return query_plan_id
        return None

    def _merge_diagnostics(self, diagnostics: Sequence[RetrievalDiagnosticRecord]) -> None:
        with self._state_lock:
            self.retrieval_diagnostics.extend(list(diagnostics or []))

    def _merge_provider_health(self, payload: Optional[Dict[str, Dict[str, Any]]]) -> None:
        with self._state_lock:
            for provider_name, provider_payload in dict(payload or {}).items():
                current = dict(self.provider_health.get(provider_name) or {})
                merged = dict(current)
                merged.update(dict(provider_payload or {}))
                self.provider_health[provider_name] = merged

    def _merge_execution_warnings(self, warnings: Optional[Sequence[str]]) -> None:
        with self._state_lock:
            self.execution_warnings = _merge_unique_text(self.execution_warnings, warnings)

    def _record_stage_event(
        self,
        *,
        stage: str,
        status: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload = {
            "timestamp": round(time.time(), 6),
            "stage": str(stage),
            "status": str(status),
            "details": copy.deepcopy(details or {}),
        }
        with self._state_lock:
            self._stage_events.append(payload)
            events_payload = copy.deepcopy(self._stage_events)
        self.store.write_json("diagnostics/runtime_stage_events.json", events_payload)
        self.store.write_json("diagnostics/runtime_stage_status.json", payload)

    def _run_stage(
        self,
        *,
        stage: str,
        operation: Callable[[], Any],
        details: Optional[Dict[str, Any]] = None,
    ) -> Any:
        start_details = copy.deepcopy(details or {})
        start_details["watchdog_seconds"] = self.stage_watchdog_seconds
        self._record_stage_event(stage=stage, status="start", details=start_details)
        started_at = time.perf_counter()
        result_holder: Dict[str, Any] = {}
        error_holder: Dict[str, BaseException] = {}
        done = threading.Event()

        def _target() -> None:
            try:
                result_holder["value"] = operation()
            except BaseException as exc:  # pragma: no cover - thread plumbing
                error_holder["error"] = exc
            finally:
                done.set()

        thread = threading.Thread(target=_target, daemon=True)
        thread.start()
        if not done.wait(self.stage_watchdog_seconds):
            elapsed = round(max(0.0, time.perf_counter() - started_at), 3)
            timeout_exc = TimeoutError(
                f"{stage} exceeded stage_watchdog_seconds={self.stage_watchdog_seconds}s "
                f"(elapsed {elapsed}s)"
            )
            timeout_details = copy.deepcopy(details or {})
            timeout_details.update({"elapsed_seconds": elapsed, "error": str(timeout_exc)})
            self._record_stage_event(stage=stage, status="timeout", details=timeout_details)
            raise timeout_exc
        try:
            if "error" in error_holder:
                raise error_holder["error"]
            result = result_holder.get("value")
        except Exception as exc:
            elapsed = round(max(0.0, time.perf_counter() - started_at), 3)
            failure_status = "timeout" if isinstance(exc, TimeoutError) else "failure"
            failure_details = copy.deepcopy(details or {})
            failure_details.update(
                {
                    "elapsed_seconds": elapsed,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            self._record_stage_event(stage=stage, status=failure_status, details=failure_details)
            raise
        elapsed = round(max(0.0, time.perf_counter() - started_at), 3)
        success_details = copy.deepcopy(details or {})
        success_details["elapsed_seconds"] = elapsed
        self._record_stage_event(stage=stage, status="success", details=success_details)
        return result

    def set_review_context(
        self,
        *,
        submission: Optional[AnswerSubmission],
        proposer_trajectory: Optional[ReActTrajectory],
        open_review_items: Optional[Sequence[ReviewItem]],
        cycle_number: int,
    ) -> None:
        with self._state_lock:
            self.current_submission = submission
            self.current_proposer_trajectory = proposer_trajectory
            self.current_review_items = list(open_review_items or [])
            self.current_cycle_number = int(cycle_number)

    def snapshot_mutable_state(self) -> Dict[str, Any]:
        with self._state_lock:
            return {
                "paper_candidates": copy.deepcopy(self.paper_candidates),
                "paper_records": copy.deepcopy(self.paper_records),
                "section_indices": copy.deepcopy(self.section_indices),
                "evidence_items": copy.deepcopy(self.evidence_items),
                "retrieval_diagnostics": copy.deepcopy(self.retrieval_diagnostics),
                "provider_health": copy.deepcopy(self.provider_health),
                "execution_warnings": list(self.execution_warnings),
                "search_result_cache": copy.deepcopy(self._search_result_cache),
                "acquire_result_cache": copy.deepcopy(self._acquire_result_cache),
                "section_read_cache": copy.deepcopy(self._section_read_cache),
                "extract_result_cache": copy.deepcopy(self._extract_result_cache),
                "citation_context_cache": copy.deepcopy(self._citation_context_cache),
            }

    def restore_mutable_state(self, snapshot: Dict[str, Any], *, write_snapshot: bool = False) -> None:
        with self._state_lock:
            self.paper_candidates = copy.deepcopy(snapshot.get("paper_candidates", {}))
            self.paper_records = copy.deepcopy(snapshot.get("paper_records", {}))
            self.section_indices = copy.deepcopy(snapshot.get("section_indices", {}))
            self.evidence_items = copy.deepcopy(snapshot.get("evidence_items", {}))
            self.retrieval_diagnostics = copy.deepcopy(snapshot.get("retrieval_diagnostics", []))
            self.provider_health = copy.deepcopy(snapshot.get("provider_health", {}))
            self.execution_warnings = list(snapshot.get("execution_warnings", []))
            self._search_result_cache = copy.deepcopy(snapshot.get("search_result_cache", {}))
            self._acquire_result_cache = copy.deepcopy(snapshot.get("acquire_result_cache", {}))
            self._section_read_cache = copy.deepcopy(snapshot.get("section_read_cache", {}))
            self._extract_result_cache = copy.deepcopy(snapshot.get("extract_result_cache", {}))
            self._citation_context_cache = copy.deepcopy(snapshot.get("citation_context_cache", {}))
            self._search_inflight = {}
            self._acquire_inflight = {}
            self._extract_inflight = {}
        if write_snapshot:
            self._write_retrieval_snapshot()

    def write_shared_snapshot(self) -> None:
        self._write_retrieval_snapshot()

    def _normalize_cache_text(self, value: Any) -> str:
        return _compact_text(value).lower()

    def _canonical_section_ids(self, section_ids: Optional[Sequence[str]]) -> Tuple[str, ...]:
        return tuple(str(item).strip() for item in (section_ids or []) if str(item).strip())

    def _prepare_cached_operation(
        self,
        *,
        cache: Dict[Tuple[Any, ...], Any],
        inflight_map: Optional[Dict[Tuple[Any, ...], _InflightOperation]],
        cache_key: Tuple[Any, ...],
        session: Optional[ReviewerSession],
        charge_budget: bool,
        tool_name: str,
        requested_via: Optional[str],
    ) -> Tuple[str, Any]:
        with self._state_lock:
            if cache_key in cache:
                return "hit", copy.deepcopy(cache[cache_key])
            if inflight_map is not None:
                inflight = inflight_map.get(cache_key)
                if inflight is not None:
                    return "wait", inflight
            if charge_budget and session is not None and not session.try_charge(
                tool_name=tool_name,
                cache_key=cache_key,
                requested_via=requested_via,
            ):
                return "blocked", session.blocked_payload(
                    tool_name=tool_name,
                    cache_key=cache_key,
                    requested_via=requested_via,
                )
            elif session is not None and not charge_budget:
                session.record_miss(
                    tool_name=tool_name,
                    cache_key=cache_key,
                    requested_via=requested_via,
                )
            if inflight_map is None:
                return "owner", None
            inflight = _InflightOperation()
            inflight_map[cache_key] = inflight
            return "owner", inflight

    def _wait_for_cached_operation(self, inflight: _InflightOperation) -> Any:
        inflight.event.wait()
        if inflight.error is not None:
            raise inflight.error
        return copy.deepcopy(inflight.result)

    def _finalize_cached_operation(
        self,
        *,
        inflight_map: Dict[Tuple[Any, ...], _InflightOperation],
        cache_key: Tuple[Any, ...],
        result: Any = None,
        error: Optional[BaseException] = None,
    ) -> None:
        with self._state_lock:
            inflight = inflight_map.pop(cache_key, None)
        if inflight is None:
            return
        inflight.result = copy.deepcopy(result)
        inflight.error = error
        inflight.event.set()

    def _build_retriever_clone(self) -> Any:
        if isinstance(self.retriever, RetrieverNode):
            return RetrieverNode(
                openalex_client=self.retriever.openalex_client,
                crossref_client=self.retriever.crossref_client,
                semantic_scholar_client=self.retriever.semantic_scholar_client,
                per_lane_limit=self.retriever.per_lane_limit,
                final_top_k=self.retriever.final_top_k,
                lane_reserve=self.retriever.lane_reserve,
                title_similarity_threshold=self.retriever.title_similarity_threshold,
            )
        return self.retriever

    def _build_document_acquirer_clone(self) -> Any:
        if isinstance(self.document_acquirer, DocumentAcquirerNode):
            pdf_extractor = self.document_acquirer.pdf_extractor
            pdf_extractor_clone = pdf_extractor
            if hasattr(pdf_extractor, "config"):
                try:
                    pdf_extractor_clone = pdf_extractor.__class__(config=pdf_extractor.config)
                except Exception:
                    pdf_extractor_clone = pdf_extractor
            return DocumentAcquirerNode(
                unpaywall_client=self.document_acquirer.unpaywall_client,
                fetcher=self.document_acquirer.fetcher,
                pdf_extractor=pdf_extractor_clone,
                document_fetch_timeout_seconds=self.document_acquirer.document_fetch_timeout_seconds,
                document_fetch_total_timeout_seconds=self.document_acquirer.document_fetch_total_timeout_seconds,
            )
        return self.document_acquirer

    def plan_queries(self, *, focus: str = "initial") -> List[Dict[str, Any]]:
        try:
            plans = self.query_planner.run(task_spec=self.task_spec, entity_pack=self.entity_pack)
        except QueryPlannerExecutionError as exc:
            self._write_query_planner_failure_artifacts(error=exc)
            raise
        payloads: List[Dict[str, Any]] = []
        for plan in plans:
            query_plan_id = self._find_registered_query_plan_id(plan, prefix="qp")
            if not query_plan_id:
                query_plan_id = self._register_query_plan(plan, prefix="qp")
            payloads.append(
                {
                    "query_plan_id": query_plan_id,
                    "focus": focus,
                    **plan.model_dump(exclude_none=True),
                }
            )
        self._write_retrieval_snapshot()
        return payloads

    def _write_query_planner_failure_artifacts(self, *, error: QueryPlannerExecutionError) -> None:
        debug_payload = dict(error.debug_payload or {})
        self.store.write_json(
            "query_planner/failure.json",
            error.to_payload(),
        )
        self.store.write_json(
            "query_planner/agent_run.json",
            {
                "agent": "QueryPlannerNode",
                "input": {
                    "task_spec": self.task_spec.model_dump(exclude_none=True),
                    "entity_pack": self.entity_pack.model_dump(exclude_none=True),
                },
                "error": error.to_payload(),
                "debug": debug_payload,
            },
        )

    def _ensure_query_plan(self, query_plan_id: Optional[str], query_text: Optional[str], lane: str) -> Tuple[str, QueryPlan]:
        normalized_lane = lane if lane in {"review", "frontier", "data", "contrarian"} else "data"
        if query_plan_id:
            with self._state_lock:
                query_plan = self.query_plans.get(str(query_plan_id))
            if query_plan is None:
                raise ValueError(f"Unknown query_plan_id: {query_plan_id}")
            return str(query_plan_id), query_plan
        cleaned_query = _compact_text(query_text)
        if not cleaned_query:
            raise ValueError("query_text is required when query_plan_id is omitted.")
        ad_hoc_key = (normalized_lane, self._normalize_cache_text(cleaned_query))
        with self._state_lock:
            existing_id = self._ad_hoc_query_plan_ids.get(ad_hoc_key)
            if existing_id:
                existing_plan = self.query_plans.get(existing_id)
                if existing_plan is not None:
                    return existing_id, existing_plan
        query_plan = QueryPlan(
            lane=normalized_lane,
            query_text=cleaned_query,
            must_terms=[],
            exclude_terms=[],
            year_from=self.task_spec.year_from,
            year_to=self.task_spec.year_to,
            preferred_sources=["openalex", "semantic_scholar", "crossref"],
        )
        ad_hoc_id = self._register_query_plan(query_plan, prefix="ad_hoc")
        with self._state_lock:
            self._ad_hoc_query_plan_ids[ad_hoc_key] = ad_hoc_id
        return ad_hoc_id, query_plan

    def search_papers(
        self,
        *,
        query_plan_id: Optional[str] = None,
        query_text: Optional[str] = None,
        lane: str = "data",
        reason: str = "",
        artifact_store: Optional[QAArtifactStore] = None,
        session: Optional[ReviewerSession] = None,
        charge_budget: bool = False,
        requested_via: Optional[str] = None,
        write_snapshot: bool = True,
    ) -> List[Dict[str, Any]]:
        store = artifact_store or self.store
        resolved_id, query_plan = self._ensure_query_plan(query_plan_id, query_text, lane)
        cache_key = ("query_plan", resolved_id)
        state, payload = self._prepare_cached_operation(
            cache=self._search_result_cache,
            inflight_map=self._search_inflight,
            cache_key=cache_key,
            session=session,
            charge_budget=charge_budget,
            tool_name="search_papers",
            requested_via=requested_via,
        )
        if state == "blocked":
            raise ReviewerBudgetBlocked(payload)
        if state == "wait":
            candidate_payloads = self._wait_for_cached_operation(payload)
            if session is not None:
                session.record_hit(tool_name="search_papers", cache_key=cache_key, requested_via=requested_via)
        elif state == "hit":
            candidate_payloads = payload
            if session is not None:
                session.record_hit(tool_name="search_papers", cache_key=cache_key, requested_via=requested_via)
        else:
            retriever = self._build_retriever_clone()
            try:
                candidates = self._run_stage(
                    stage="search_papers",
                    details={
                        "query_plan_id": resolved_id,
                        "lane": query_plan.lane,
                        "requested_via": requested_via,
                    },
                    operation=lambda: retriever.run(
                        task_spec=self.task_spec,
                        entity_pack=self.entity_pack,
                        query_plans=[query_plan],
                        artifact_store=store,
                    ),
                )
                self._merge_diagnostics(getattr(retriever, "last_diagnostics", []) or [])
                self._merge_provider_health(getattr(retriever, "last_provider_health", {}) or {})
                with self._state_lock:
                    for candidate in candidates:
                        self.paper_candidates[candidate.paper_id] = candidate
                    candidate_payloads = [candidate.model_dump(exclude_none=True) for candidate in candidates]
                    self._search_result_cache[cache_key] = copy.deepcopy(candidate_payloads)
                self._finalize_cached_operation(
                    inflight_map=self._search_inflight,
                    cache_key=cache_key,
                    result=candidate_payloads,
                )
            except Exception as exc:
                self._finalize_cached_operation(
                    inflight_map=self._search_inflight,
                    cache_key=cache_key,
                    error=exc,
                )
                raise
        if write_snapshot:
            self._write_retrieval_snapshot()
        return [
            {
                "query_plan_id": resolved_id,
                "reason": str(reason or "").strip(),
                **item,
            }
            for item in candidate_payloads
        ]

    def acquire_document(
        self,
        *,
        paper_id: str,
        artifact_store: Optional[QAArtifactStore] = None,
        session: Optional[ReviewerSession] = None,
        charge_budget: bool = False,
        requested_via: Optional[str] = None,
        write_snapshot: bool = True,
    ) -> Dict[str, Any]:
        store = artifact_store or self.store
        normalized_paper_id = str(paper_id)
        cache_key = ("paper_id", normalized_paper_id)
        with self._state_lock:
            if cache_key not in self._acquire_result_cache and normalized_paper_id in self.paper_records and normalized_paper_id in self.section_indices:
                paper_record = self.paper_records[normalized_paper_id]
                section_index = self.section_indices[normalized_paper_id]
                self._acquire_result_cache[cache_key] = {
                    "paper_id": paper_record.paper_id,
                    "fulltext_available": paper_record.fulltext_available,
                    "fulltext_status": paper_record.fulltext_status,
                    "section_count": len(section_index.sections),
                    "artifact_path": paper_record.fulltext_artifact_path,
                }
        state, payload = self._prepare_cached_operation(
            cache=self._acquire_result_cache,
            inflight_map=self._acquire_inflight,
            cache_key=cache_key,
            session=session,
            charge_budget=charge_budget,
            tool_name="acquire_document",
            requested_via=requested_via,
        )
        if state == "blocked":
            raise ReviewerBudgetBlocked(payload)
        if state == "wait":
            result = self._wait_for_cached_operation(payload)
            if session is not None:
                session.record_hit(tool_name="acquire_document", cache_key=cache_key, requested_via=requested_via)
        elif state == "hit":
            result = payload
            if session is not None:
                session.record_hit(tool_name="acquire_document", cache_key=cache_key, requested_via=requested_via)
        else:
            with self._state_lock:
                candidate = self.paper_candidates.get(normalized_paper_id)
            if candidate is None:
                error = ValueError(f"Unknown paper_id: {paper_id}")
                self._finalize_cached_operation(
                    inflight_map=self._acquire_inflight,
                    cache_key=cache_key,
                    error=error,
                )
                raise error
            acquirer = self._build_document_acquirer_clone()
            try:
                paper_records, section_indices = self._run_stage(
                    stage="acquire_document",
                    details={
                        "paper_id": normalized_paper_id,
                        "requested_via": requested_via,
                    },
                    operation=lambda: acquirer.run(
                        candidates=[candidate],
                        artifact_store=store,
                    ),
                )
                self._merge_diagnostics(getattr(acquirer, "last_diagnostics", []) or [])
                self._merge_provider_health(getattr(acquirer, "last_provider_health", {}) or {})
                self._merge_execution_warnings(getattr(acquirer, "last_execution_warnings", []) or [])
                with self._state_lock:
                    for paper_record, section_index in zip(paper_records, section_indices):
                        self.paper_records[paper_record.paper_id] = paper_record
                        self.section_indices[section_index.paper_id] = section_index
                    stored_record = self.paper_records[candidate.paper_id]
                    stored_index = self.section_indices[candidate.paper_id]
                    result = {
                        "paper_id": stored_record.paper_id,
                        "fulltext_available": stored_record.fulltext_available,
                        "fulltext_status": stored_record.fulltext_status,
                        "section_count": len(stored_index.sections),
                        "artifact_path": stored_record.fulltext_artifact_path,
                    }
                    self._acquire_result_cache[cache_key] = copy.deepcopy(result)
                self._finalize_cached_operation(
                    inflight_map=self._acquire_inflight,
                    cache_key=cache_key,
                    result=result,
                )
            except Exception as exc:
                self._finalize_cached_operation(
                    inflight_map=self._acquire_inflight,
                    cache_key=cache_key,
                    error=exc,
                )
                raise
        if write_snapshot:
            self._write_retrieval_snapshot()
        return result

    def _ensure_document(
        self,
        paper_id: str,
        *,
        artifact_store: Optional[QAArtifactStore] = None,
        session: Optional[ReviewerSession] = None,
        charge_budget: bool = False,
        requested_via: Optional[str] = None,
        write_snapshot: bool = True,
    ) -> Tuple[PaperRecord, SectionIndex]:
        with self._state_lock:
            has_document = paper_id in self.paper_records and paper_id in self.section_indices
        if not has_document:
            self.acquire_document(
                paper_id=paper_id,
                artifact_store=artifact_store,
                session=session,
                charge_budget=charge_budget,
                requested_via=requested_via,
                write_snapshot=write_snapshot,
            )
        with self._state_lock:
            paper_record = self.paper_records.get(paper_id)
            section_index = self.section_indices.get(paper_id)
        if paper_record is None or section_index is None:
            raise ValueError(f"Failed to acquire paper_id={paper_id}")
        return paper_record, section_index

    def read_sections(
        self,
        *,
        paper_id: str,
        section_ids: Optional[Sequence[str]] = None,
        preferred_sections: bool = False,
        artifact_store: Optional[QAArtifactStore] = None,
        session: Optional[ReviewerSession] = None,
        charge_budget: bool = False,
        requested_via: Optional[str] = None,
        write_snapshot: bool = True,
    ) -> List[Dict[str, Any]]:
        paper_record, section_index = self._ensure_document(
            str(paper_id),
            artifact_store=artifact_store,
            session=session,
            charge_budget=charge_budget,
            requested_via=requested_via or "read_sections",
            write_snapshot=write_snapshot,
        )
        selected_ids = self._canonical_section_ids(section_ids)
        if not preferred_sections and not selected_ids and section_index.sections:
            selected_ids = (section_index.sections[0].section_id,)
        cache_key = ("sections", str(paper_id), selected_ids, bool(preferred_sections))
        state, payload = self._prepare_cached_operation(
            cache=self._section_read_cache,
            inflight_map=None,
            cache_key=cache_key,
            session=session,
            charge_budget=False,
            tool_name="read_sections",
            requested_via=requested_via,
        )
        if state == "hit":
            if session is not None:
                session.record_hit(tool_name="read_sections", cache_key=cache_key, requested_via=requested_via)
            return payload
        def _read_sections_operation() -> List[Dict[str, Any]]:
            if preferred_sections:
                section_views = self.handoff.read_preferred_sections(
                    paper_record=paper_record,
                    section_index=section_index,
                    task_spec=self.task_spec,
                    evidence_is_weak=False,
                    missing_conditions=False,
                )
            else:
                section_views = [
                    self.handoff.read_section_text(
                        paper_record=paper_record,
                        section_index=section_index,
                        section_id=section_id,
                    )
                    for section_id in list(selected_ids)
                ]
                section_views = [view for view in section_views if view is not None]
            if not section_views and paper_record.abstract:
                return [
                    {
                        "paper_id": paper_record.paper_id,
                        "section_id": "sec_abstract",
                        "section_type": "abstract",
                        "heading": "Abstract",
                        "text": paper_record.abstract,
                    }
                ]
            return [
                {
                    "paper_id": view.paper_id,
                    "section_id": view.section_id,
                    "section_type": view.section_type,
                    "heading": view.heading,
                    "text": view.text,
                    "page_start": view.page_start,
                    "page_end": view.page_end,
                }
                for view in section_views
            ]

        payloads = self._run_stage(
            stage="read_sections",
            details={
                "paper_id": str(paper_id),
                "requested_via": requested_via,
                "preferred_sections": bool(preferred_sections),
                "section_ids": list(selected_ids),
            },
            operation=_read_sections_operation,
        )
        with self._state_lock:
            self._section_read_cache[cache_key] = copy.deepcopy(payloads)
        return payloads

    def extract_evidence(
        self,
        *,
        paper_id: str,
        section_ids: Optional[Sequence[str]] = None,
        preferred_sections: bool = False,
        artifact_store: Optional[QAArtifactStore] = None,
        session: Optional[ReviewerSession] = None,
        charge_budget: bool = False,
        requested_via: Optional[str] = None,
        write_snapshot: bool = True,
    ) -> List[Dict[str, Any]]:
        cache_key = ("evidence", str(paper_id), self._canonical_section_ids(section_ids), bool(preferred_sections))
        state, payload = self._prepare_cached_operation(
            cache=self._extract_result_cache,
            inflight_map=self._extract_inflight,
            cache_key=cache_key,
            session=session,
            charge_budget=charge_budget,
            tool_name="extract_evidence",
            requested_via=requested_via,
        )
        if state == "blocked":
            raise ReviewerBudgetBlocked(payload)
        if state == "wait":
            result = self._wait_for_cached_operation(payload)
            if session is not None:
                session.record_hit(tool_name="extract_evidence", cache_key=cache_key, requested_via=requested_via)
        elif state == "hit":
            result = payload
            if session is not None:
                session.record_hit(tool_name="extract_evidence", cache_key=cache_key, requested_via=requested_via)
        else:
            paper_record, section_index = self._ensure_document(
                str(paper_id),
                artifact_store=artifact_store,
                session=session,
                charge_budget=False,
                requested_via="extract_evidence",
                write_snapshot=write_snapshot,
            )
            try:
                def _extract_evidence_operation() -> List[EvidenceItem]:
                    if not section_ids and not preferred_sections:
                        return self.evidence_extractor.run(
                            task_spec=self.task_spec,
                            entity_pack=self.entity_pack,
                            paper_record=paper_record,
                            section_index=section_index,
                        )
                    evidence_items: List[EvidenceItem] = []
                    for section_payload in self.read_sections(
                        paper_id=paper_id,
                        section_ids=section_ids,
                        preferred_sections=preferred_sections,
                        artifact_store=artifact_store,
                        session=session,
                        charge_budget=False,
                        requested_via="extract_evidence",
                        write_snapshot=False,
                    ):
                        if section_payload.get("section_id") == "sec_abstract":
                            fulltext_end = len(str(section_payload.get("text") or ""))
                            section_view = SectionTextView(
                                paper_id=paper_id,
                                section_id="sec_abstract",
                                section_type=section_payload.get("section_type", "abstract"),
                                heading=section_payload.get("heading", "Abstract"),
                                text=section_payload.get("text", ""),
                                page_start=None,
                                page_end=None,
                                fulltext_char_start=0,
                                fulltext_char_end=fulltext_end,
                            )
                        else:
                            view = self.handoff.read_section_text(
                                paper_record=paper_record,
                                section_index=section_index,
                                section_id=str(section_payload["section_id"]),
                            )
                            if view is None:
                                continue
                            section_view = view
                        evidence_items.extend(
                            self.evidence_extractor._extract_from_section(
                                task_spec=self.task_spec,
                                entity_pack=self.entity_pack,
                                paper_record=paper_record,
                                section_view=section_view,
                            )
                        )
                    return evidence_items

                evidence_items = self._run_stage(
                    stage="extract_evidence",
                    details={
                        "paper_id": str(paper_id),
                        "requested_via": requested_via,
                        "preferred_sections": bool(preferred_sections),
                        "section_ids": list(self._canonical_section_ids(section_ids)),
                    },
                    operation=_extract_evidence_operation,
                )
                with self._state_lock:
                    for evidence_item in evidence_items:
                        self.evidence_items[evidence_item.evidence_id] = evidence_item
                    result = [item.model_dump(exclude_none=True) for item in evidence_items]
                    self._extract_result_cache[cache_key] = copy.deepcopy(result)
                self._finalize_cached_operation(
                    inflight_map=self._extract_inflight,
                    cache_key=cache_key,
                    result=result,
                )
            except Exception as exc:
                self._finalize_cached_operation(
                    inflight_map=self._extract_inflight,
                    cache_key=cache_key,
                    error=exc,
                )
                raise
        if write_snapshot:
            self._write_retrieval_snapshot()
        return result

    def fetch_citation_context(
        self,
        *,
        paper_id: Optional[str] = None,
        section_id: Optional[str] = None,
        evidence_id: Optional[str] = None,
        citation_id: Optional[str] = None,
        artifact_store: Optional[QAArtifactStore] = None,
        session: Optional[ReviewerSession] = None,
        charge_budget: bool = False,
        requested_via: Optional[str] = None,
        write_snapshot: bool = True,
    ) -> Dict[str, Any]:
        del charge_budget
        if evidence_id:
            with self._state_lock:
                evidence_item = self.evidence_items.get(str(evidence_id))
            if evidence_item is None:
                raise ValueError(f"Unknown evidence_id: {evidence_id}")
            return evidence_item.model_dump(exclude_none=True)
        with self._state_lock:
            current_submission = self.current_submission
            evidence_items = dict(self.evidence_items)
        if citation_id and current_submission is not None:
            citation = next(
                (item for item in current_submission.citations if item.citation_id == str(citation_id)),
                None,
            )
            if citation is not None:
                if evidence_id is None and citation.evidence_ids:
                    evidence_payloads = [
                        evidence_items[evidence_ref].model_dump(exclude_none=True)
                        for evidence_ref in list(citation.evidence_ids)
                        if evidence_ref in evidence_items
                    ]
                    if evidence_payloads:
                        return {
                            "citation_id": citation.citation_id,
                            "paper_id": citation.paper_id,
                            "section_ids": list(citation.section_ids or []),
                            "evidence_ids": list(citation.evidence_ids or []),
                            "evidence": evidence_payloads,
                        }
                paper_id = citation.paper_id
                if section_id is None and citation.section_ids:
                    section_id = citation.section_ids[0]
        if not paper_id:
            raise ValueError("paper_id or evidence_id or citation_id is required.")
        cache_key = ("citation_context", str(paper_id), str(section_id or ""), str(evidence_id or ""), str(citation_id or ""))
        state, payload = self._prepare_cached_operation(
            cache=self._citation_context_cache,
            inflight_map=None,
            cache_key=cache_key,
            session=session,
            charge_budget=False,
            tool_name="fetch_citation_context",
            requested_via=requested_via,
        )
        if state == "hit":
            if session is not None:
                session.record_hit(tool_name="fetch_citation_context", cache_key=cache_key, requested_via=requested_via)
            return payload
        sections = self.read_sections(
            paper_id=str(paper_id),
            section_ids=[section_id] if section_id else None,
            preferred_sections=section_id is None,
            artifact_store=artifact_store,
            session=session,
            charge_budget=True,
            requested_via=requested_via or "fetch_citation_context",
            write_snapshot=write_snapshot,
        )
        if not sections:
            raise ValueError(f"No section text available for paper_id={paper_id}")
        first_section = dict(sections[0])
        first_section["paper_id"] = paper_id
        with self._state_lock:
            self._citation_context_cache[cache_key] = copy.deepcopy(first_section)
        return first_section

    def inspect_submission_anchor(
        self,
        *,
        section_id: Optional[str] = None,
        step_number: Optional[int] = None,
        review_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        with self._state_lock:
            current_submission = self.current_submission
            current_proposer_trajectory = self.current_proposer_trajectory
            current_review_items = list(self.current_review_items)
            current_cycle_number = self.current_cycle_number
        payload: Dict[str, Any] = {
            "cycle_number": current_cycle_number,
            "submission": None,
            "trajectory_step": None,
            "review_item": None,
        }
        if current_submission is not None:
            if section_id:
                payload["submission"] = next(
                    (
                        item.model_dump(exclude_none=True)
                        for item in current_submission.sections
                        if item.section_id == str(section_id)
                    ),
                    None,
                )
            else:
                payload["submission"] = current_submission.model_dump(exclude_none=True)
        if current_proposer_trajectory is not None and step_number is not None:
            payload["trajectory_step"] = next(
                (
                    step.to_dict()
                    for step in current_proposer_trajectory.steps
                    if step.step_number == int(step_number)
                ),
                None,
            )
        if review_id:
            payload["review_item"] = next(
                (
                    item.model_dump(exclude_none=True)
                    for item in current_review_items
                    if item.review_id == str(review_id)
                ),
                None,
            )
        return payload

    def analyze_submission_gap(self) -> Dict[str, Any]:
        with self._state_lock:
            current_submission = self.current_submission
            current_review_items = list(self.current_review_items)
        missing_section_ids = [
            section.section_id
            for section in self.task_spec.answer_sections
            if current_submission is None
            or not any(item.section_id == section.section_id for item in current_submission.sections)
        ]
        open_blocking_items = [
            item.model_dump(exclude_none=True)
            for item in current_review_items
            if item.status == "open" and item.severity == "blocking"
        ]
        return {
            "missing_section_ids": missing_section_ids,
            "open_blocking_items": open_blocking_items,
            "open_review_item_count": len([item for item in current_review_items if item.status == "open"]),
        }

    def inspect_entity_cache(
        self,
        *,
        name: Optional[str] = None,
        entity_type: Optional[str] = None,
        limit: int = 10,
    ) -> Dict[str, Any]:
        resolution_index = dict(self.entity_resolution_snapshot.get("resolution_index") or {})
        entries = list(resolution_index.get("entries") or [])
        filtered: List[Dict[str, Any]] = []
        normalized_name = _compact_text(name).lower()
        normalized_type = _compact_text(entity_type).lower()
        for entry in entries:
            if normalized_type and str(entry.get("entity_type") or "").strip().lower() != normalized_type:
                continue
            if normalized_name:
                candidate_texts = [
                    str(entry.get("canonical_name") or ""),
                    str(entry.get("formula") or ""),
                    *(str(value) for value in list(entry.get("aliases") or [])),
                    *(str(value) for value in list(entry.get("query_anchors") or [])),
                ]
                normalized_candidates = {_compact_text(value).lower() for value in candidate_texts if _compact_text(value)}
                if normalized_name not in normalized_candidates:
                    continue
            filtered.append(dict(entry))
        filtered = filtered[: max(1, int(limit or 1))]
        return {
            "count": len(filtered),
            "entries": filtered,
            "provider_calls": list(self.entity_resolution_snapshot.get("provider_calls") or []),
        }

    def diagnostics_summary(self) -> str:
        with self._state_lock:
            diagnostics = list(self.retrieval_diagnostics)
        messages: List[str] = []
        for record in diagnostics:
            if any(getattr(record, field) > 0 for field in ("failure_count", "timeout_count", "skipped_count")):
                parts: List[str] = []
                if record.failure_count:
                    parts.append(f"{record.failure_count} failure")
                if record.timeout_count:
                    parts.append(f"{record.timeout_count} timeout")
                if record.skipped_count:
                    parts.append(f"{record.skipped_count} skipped")
                label = record.lane or record.stage
                messages.append(f"{record.provider} {label} had {', '.join(parts)}")
        if not messages:
            return ""
        return "External literature retrieval encountered issues: " + "; ".join(messages) + "."

    def _write_retrieval_snapshot(self) -> None:
        with self._state_lock:
            query_plans = [
                {"query_plan_id": query_plan_id, **query_plan.model_dump(exclude_none=True)}
                for query_plan_id, query_plan in self.query_plans.items()
            ]
            paper_candidates = [item.model_dump(exclude_none=True) for item in self.paper_candidates.values()]
            paper_records = [item.model_dump(exclude_none=True) for item in self.paper_records.values()]
            section_indices = [item.model_dump(exclude_none=True) for item in self.section_indices.values()]
            retrieval_diagnostics = [item.model_dump(exclude_none=True) for item in self.retrieval_diagnostics]
            provider_health = copy.deepcopy(self.provider_health)
            evidence_items = [item.model_dump(exclude_none=True) for item in self.evidence_items.values()]
            execution_warnings = list(self.execution_warnings)
        self.store.write_json("query_plans.json", query_plans)
        self.store.write_json("paper_candidates.json", paper_candidates)
        self.store.write_json("paper_records.json", paper_records)
        self.store.write_json("section_indices.json", section_indices)
        self.store.write_json("retrieval_diagnostics.json", retrieval_diagnostics)
        self.store.write_json("provider_health.json", provider_health)
        self.store.write_json("evidence_items.json", evidence_items)
        self.store.write_json("execution_warnings.json", execution_warnings)


class ReactReviewedProposerAgent:
    def __init__(
        self,
        *,
        model_config: Optional[Dict[str, Any]],
        max_steps_initial: int,
        max_steps_revision: int,
        llm_timeout_seconds: float,
        fallback_mode: str = "fail_fast_only",
        repair_attempts: int = 1,
        evidence_policy: str = "prefer_fulltext",
    ) -> None:
        self.model_config = dict(model_config or {})
        self.max_steps_initial = max(1, int(max_steps_initial))
        self.max_steps_revision = max(1, int(max_steps_revision))
        if self.max_steps_initial < MIN_REACT_REVIEWED_PROPOSER_STEPS:
            raise ValueError(
                "ReactReviewed proposer max_steps_initial must be at least "
                f"{MIN_REACT_REVIEWED_PROPOSER_STEPS} so the required plan/search/acquire/read-or-extract/conclude "
                "tool chain remains feasible under deadline mode."
            )
        if self.max_steps_revision < MIN_REACT_REVIEWED_PROPOSER_STEPS:
            raise ValueError(
                "ReactReviewed proposer max_steps_revision must be at least "
                f"{MIN_REACT_REVIEWED_PROPOSER_STEPS} so revision cycles can still acquire evidence and conclude "
                "before deadline-mode retrieval blocking starts."
            )
        self.llm_timeout_seconds = float(llm_timeout_seconds)
        self.fallback_mode = str(fallback_mode or "fail_fast_only").strip().lower() or "fail_fast_only"
        self.repair_attempts = max(0, int(repair_attempts))
        self.evidence_policy = str(evidence_policy or "prefer_fulltext").strip().lower() or "prefer_fulltext"

    def _score_screen_candidate(
        self,
        *,
        workspace: ReactReviewedWorkspace,
        candidate: PaperCandidate,
        open_review_items: Sequence[ReviewItem],
    ) -> Dict[str, Any]:
        metrics = _paper_relevance_metrics(
            task_spec=workspace.task_spec,
            entity_pack=workspace.entity_pack,
            title=candidate.title,
            abstract=candidate.abstract,
        )
        review_priority_terms = _review_item_priority_terms(open_review_items)
        normalized_corpus = _compact_text(f"{candidate.title} {candidate.abstract or ''}").lower()
        review_term_hits = [term for term in review_priority_terms if term and term in normalized_corpus]
        primary_entity_terms = _primary_screen_entity_terms(workspace.entity_pack)
        normalized_title = _compact_text(candidate.title).lower()
        normalized_abstract = _compact_text(candidate.abstract).lower()
        primary_title_hits = [term for term in primary_entity_terms if term and term in normalized_title]
        primary_abstract_hits = [term for term in primary_entity_terms if term and term in normalized_abstract]
        provider_count = len({str(item).strip().lower() for item in candidate.provider_hits if str(item).strip()})
        provider_hits = {str(item).strip().lower() for item in candidate.provider_hits if str(item).strip()}
        has_oa_url = bool(_compact_text(candidate.oa_url))
        has_doi = bool(_compact_text(candidate.doi))
        fulltext_signal_score = (3.0 if has_oa_url else 0.0) + (1.5 if has_doi else 0.0) + (
            0.5 if provider_count >= 2 else 0.0
        )
        provider_priority_score = (
            (1.25 if "openalex" in provider_hits else 0.0)
            + (0.75 if "semantic_scholar" in provider_hits else 0.0)
            - (0.75 if provider_hits == {"crossref"} else 0.0)
        )
        acquisition_risk = "low" if has_oa_url or (has_doi and provider_count >= 2) else "medium" if has_doi or has_oa_url else "high"
        generic_abstract_penalty = 3.0 if metrics["has_abstract"] and metrics["abstract_hits"] == 0 and metrics["question_hits"] <= 1 else 0.0
        missing_abstract_penalty = 1.5 if not metrics["has_abstract"] else 0.0
        exact_primary_alignment = bool(primary_title_hits)
        strong_primary_alignment = exact_primary_alignment or len(primary_abstract_hits) >= 2
        weak_primary_alignment = not strong_primary_alignment and bool(primary_abstract_hits)
        review_like = _is_review_like_title(candidate.title)
        comparator_only_primary_reference = _mentions_primary_entity_only_as_comparator(
            abstract=candidate.abstract,
            primary_terms=primary_entity_terms,
        )
        score = (
            4.0 * float(candidate.retrieval_score or 0.0)
            + 2.5 * float(metrics["title_hits"])
            + 2.0 * float(metrics["abstract_hits"])
            + 1.5 * float(metrics["question_hits"])
            + 1.25 * float(min(len(review_term_hits), 4))
            + fulltext_signal_score
            + provider_priority_score
            + (4.5 if exact_primary_alignment else 1.5 if strong_primary_alignment else 0.0)
            - generic_abstract_penalty
            - missing_abstract_penalty
            - (2.5 if weak_primary_alignment else 0.0)
            - (3.5 if review_like and not exact_primary_alignment else 0.0)
            - (4.5 if comparator_only_primary_reference and not exact_primary_alignment else 0.0)
        )
        should_drop = (
            metrics["title_hits"] == 0
            and metrics["abstract_hits"] == 0
            and metrics["question_hits"] == 0
        ) or (acquisition_risk == "high" and metrics["abstract_hits"] == 0 and metrics["question_hits"] <= 1) or (
            bool(primary_entity_terms)
            and not exact_primary_alignment
            and not strong_primary_alignment
            and not _paper_has_useful_abstract_support(
                task_spec=workspace.task_spec,
                entity_pack=workspace.entity_pack,
                title=candidate.title,
                abstract=candidate.abstract,
            )
        ) or (
            review_like and not exact_primary_alignment and metrics["abstract_hits"] <= 2
        ) or (
            comparator_only_primary_reference and not exact_primary_alignment
        )
        return {
            "paper_id": candidate.paper_id,
            "title": candidate.title,
            "retrieval_score": round(float(candidate.retrieval_score or 0.0), 4),
            "screen_score": round(score, 4),
            "decision": "drop" if should_drop else "lock",
            "reason": (
                "Low question relevance or weak acquisition signal."
                if should_drop
                else "Strong question alignment with better full-text acquisition signal."
            ),
            "matched_terms": metrics["matched_terms"],
            "review_term_hits": review_term_hits,
            "title_hits": metrics["title_hits"],
            "abstract_hits": metrics["abstract_hits"],
            "question_hits": metrics["question_hits"],
            "has_abstract": metrics["has_abstract"],
            "has_doi": has_doi,
            "has_oa_url": has_oa_url,
            "provider_count": provider_count,
            "acquisition_risk": acquisition_risk,
            "primary_title_hits": primary_title_hits,
            "primary_abstract_hits": primary_abstract_hits,
            "review_like": review_like,
            "comparator_only_primary_reference": comparator_only_primary_reference,
        }

    def _llm_screen_candidate_papers(
        self,
        *,
        workspace: ReactReviewedWorkspace,
        cycle_number: int,
        open_review_items: Sequence[ReviewItem],
        ranked_candidates: Sequence[Dict[str, Any]],
        max_candidates: int,
    ) -> Tuple[Optional[Dict[str, Any]], Any]:
        provider, model_name, has_api_key = describe_chat_model_config(self.model_config)
        if not provider or not model_name or not has_api_key:
            return None, None
        llm = build_chat_model_from_config(self.model_config)
        raw_response = invoke_llm(
            llm,
            [
                {
                    "role": "system",
                    "content": build_screening_system_prompt(max_candidates=max_candidates),
                },
                {
                    "role": "user",
                    "content": _json_preview(
                        {
                            "cycle_number": cycle_number,
                            "question": workspace.question,
                            "task_spec": workspace.task_spec.model_dump(exclude_none=True),
                            "entity_pack": workspace.entity_pack.model_dump(exclude_none=True),
                            "open_review_items": [item.model_dump(exclude_none=True) for item in open_review_items],
                            "candidates": list(ranked_candidates),
                        },
                        limit=16000,
                    ),
                },
            ],
        )
        parsed = parse_json_payload(raw_response)
        if not isinstance(parsed, dict):
            return None, raw_response
        allowed_ids = {str(item.get("paper_id") or "").strip() for item in ranked_candidates if str(item.get("paper_id") or "").strip()}
        decisions = []
        decision_map: Dict[str, Dict[str, Any]] = {}
        for item in list(parsed.get("decisions") or []):
            if not isinstance(item, dict):
                continue
            paper_id = str(item.get("paper_id") or "").strip()
            decision = str(item.get("decision") or "").strip().lower()
            reason = _compact_text(item.get("reason"))
            if paper_id not in allowed_ids or decision not in {"lock", "drop"} or not reason:
                continue
            current = {"paper_id": paper_id, "decision": decision, "reason": reason}
            decisions.append(current)
            decision_map[paper_id] = current
        if not decisions:
            return None, raw_response
        locked_paper_ids: List[str] = []
        dropped_paper_ids: List[str] = []
        ranked_payload: List[Dict[str, Any]] = []
        for item in ranked_candidates:
            paper_id = str(item.get("paper_id") or "").strip()
            llm_decision = decision_map.get(paper_id)
            base_decision = str(item.get("decision") or "drop")
            decision = llm_decision["decision"] if llm_decision else base_decision
            if base_decision == "drop":
                decision = "drop"
            reason = llm_decision["reason"] if llm_decision else _compact_text(item.get("reason"))
            current = copy.deepcopy(item)
            current["decision"] = decision
            current["reason"] = reason
            ranked_payload.append(current)
            if decision == "lock" and paper_id not in locked_paper_ids and len(locked_paper_ids) < max_candidates:
                locked_paper_ids.append(paper_id)
            elif paper_id not in dropped_paper_ids:
                dropped_paper_ids.append(paper_id)
        if not locked_paper_ids:
            return None, raw_response
        return (
            {
                "locked_paper_ids": locked_paper_ids,
                "dropped_paper_ids": dropped_paper_ids,
                "ranked_candidates": ranked_payload,
                "llm_screening_used": True,
            },
            raw_response,
        )

    def _screen_candidate_papers(
        self,
        *,
        workspace: ReactReviewedWorkspace,
        cycle_number: int,
        open_review_items: Sequence[ReviewItem],
        paper_ids: Sequence[str],
        max_candidates: int,
    ) -> Dict[str, Any]:
        ordered_ids: List[str] = []
        seen = set()
        for paper_id in paper_ids:
            normalized = str(paper_id or "").strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            ordered_ids.append(normalized)
        ranked_candidates: List[Dict[str, Any]] = []
        for paper_id in ordered_ids:
            candidate = workspace.paper_candidates.get(paper_id)
            if candidate is None:
                continue
            ranked_candidates.append(
                self._score_screen_candidate(
                    workspace=workspace,
                    candidate=candidate,
                    open_review_items=open_review_items,
                )
            )
        ranked_candidates.sort(
            key=lambda item: (
                0 if str(item.get("decision") or "drop") == "lock" else 1,
                -float(item.get("screen_score") or 0.0),
                -float(item.get("retrieval_score") or 0.0),
            )
        )
        provider, model_name, has_api_key = describe_chat_model_config(self.model_config)
        if not provider or not model_name or not has_api_key:
            payload = {
                "stage": "proposer_screening",
                "cycle_number": cycle_number,
                "message": "candidate screening requires an LLM model configuration",
                "ranked_candidates": ranked_candidates,
                "llm_screening_used": False,
            }
            workspace.store.write_json(
                f"proposer_cycle_{cycle_number}_candidate_screening.json",
                payload,
            )
            self._raise_execution_failure(
                workspace=workspace,
                cycle_number=cycle_number,
                stage="proposer_screening",
                message="Candidate screening requires an enabled LLM configuration.",
                details={
                    "reason": "missing_model_config",
                    "ranked_candidate_count": len(ranked_candidates),
                    "max_candidates": int(max_candidates),
                },
                structured_output=payload,
            )
        try:
            llm_result = self._llm_screen_candidate_papers(
                workspace=workspace,
                cycle_number=cycle_number,
                open_review_items=open_review_items,
                ranked_candidates=ranked_candidates[:8],
                max_candidates=max_candidates,
            )
            if isinstance(llm_result, tuple) and len(llm_result) == 2:
                llm_payload, screening_raw_response = llm_result
            else:
                llm_payload = llm_result
                screening_raw_response = None
        except Exception as exc:
            payload = {
                "stage": "proposer_screening",
                "cycle_number": cycle_number,
                "message": f"candidate screening LLM execution failed: {exc}",
                "ranked_candidates": ranked_candidates,
                "llm_screening_used": False,
            }
            workspace.store.write_json(
                f"proposer_cycle_{cycle_number}_candidate_screening.json",
                payload,
            )
            self._raise_execution_failure(
                workspace=workspace,
                cycle_number=cycle_number,
                stage="proposer_screening",
                message=f"Candidate screening LLM execution failed: {exc}",
                details={
                    "reason": "llm_screening_exception",
                    "ranked_candidate_count": len(ranked_candidates),
                    "max_candidates": int(max_candidates),
                },
                response_content=getattr(exc, "response_content", None),
                structured_output=payload,
            )
        if not isinstance(llm_payload, dict):
            payload = {
                "stage": "proposer_screening",
                "cycle_number": cycle_number,
                "message": "candidate screening produced no valid structured payload",
                "ranked_candidates": ranked_candidates,
                "llm_screening_used": False,
            }
            workspace.store.write_json(
                f"proposer_cycle_{cycle_number}_candidate_screening.json",
                payload,
            )
            self._raise_execution_failure(
                workspace=workspace,
                cycle_number=cycle_number,
                stage="proposer_screening",
                message="Candidate screening produced no valid structured payload.",
                details={
                    "reason": "invalid_screening_payload",
                    "ranked_candidate_count": len(ranked_candidates),
                    "max_candidates": int(max_candidates),
                },
                response_content=screening_raw_response,
                structured_output=payload,
            )
        payload = llm_payload
        workspace.store.write_json(
            f"proposer_cycle_{cycle_number}_candidate_screening.json",
            payload,
        )
        return payload

    def run(
        self,
        *,
        workspace: ReactReviewedWorkspace,
        cycle_number: int,
        open_review_items: Sequence[ReviewItem],
    ) -> Tuple[AnswerSubmission, ReActTrajectory]:
        stage_snapshot = workspace.snapshot_mutable_state()
        try:
            return self._run_with_llm(
                workspace=workspace,
                cycle_number=cycle_number,
                open_review_items=open_review_items,
            )
        except (ReactReviewedStructuredOutputError, ReactReviewedProposerExecutionError):
            workspace.restore_mutable_state(stage_snapshot, write_snapshot=True)
            raise
        except Exception:
            workspace.restore_mutable_state(stage_snapshot, write_snapshot=True)
            raise

    def _run_with_llm(
        self,
        *,
        workspace: ReactReviewedWorkspace,
        cycle_number: int,
        open_review_items: Sequence[ReviewItem],
    ) -> Tuple[AnswerSubmission, ReActTrajectory]:
        StructuredTool = _lazy_structured_tool_import()
        provider, model_name, has_api_key = describe_chat_model_config(self.model_config)
        if StructuredTool is None:
            self._raise_execution_failure(
                workspace=workspace,
                cycle_number=cycle_number,
                stage="proposer_startup",
                message="Proposer cannot start because StructuredTool support is unavailable.",
                details={"reason": "structured_tool_unavailable"},
            )
        if not has_api_key or not provider or not model_name:
            self._raise_execution_failure(
                workspace=workspace,
                cycle_number=cycle_number,
                stage="proposer_startup",
                message="Proposer cannot start because the LLM configuration is incomplete.",
                details={
                    "provider": provider,
                    "model": model_name,
                    "has_api_key": has_api_key,
                    "fallback_mode": self.fallback_mode,
                },
            )

        agent_holder: Dict[str, Any] = {}
        run_state = _ProposerRunState(evidence_policy=self.evidence_policy)

        def _policy_block(message: str, *, code: str) -> ToolResult:
            return ToolResult(
                observation=f"Policy: {message}",
                data={"error": code, "message": message},
            )

        def plan_queries(focus: str = "initial") -> ToolResult:
            """Generate or refresh query plans for the current QA focus."""
            payload = workspace.plan_queries(focus=focus)
            run_state.record_plan_queries(payload)
            return ToolResult(
                observation=_json_preview({"count": len(payload), "query_plan_ids": [item["query_plan_id"] for item in payload]}),
                data={"query_plans": payload},
            )

        def search_papers(
            query_plan_id: Optional[str] = None,
            query_text: Optional[str] = None,
            lane: str = "data",
            reason: str = "",
        ) -> ToolResult:
            """Search external literature providers using a planned or ad hoc query."""
            if not run_state.query_plan_ids:
                return _policy_block(
                    "plan_queries must be called before search_papers.",
                    code="plan_required_before_search",
                )
            if run_state.locked_candidate_paper_ids and not run_state.acquired_paper_ids and not run_state.screening_required():
                return ToolResult(
                    observation=_json_preview(
                        {
                            "error": "acquire_locked_candidates_before_more_search",
                            "message": "screen_papers already locked candidates for this cycle; acquire them before running more search_papers.",
                            "locked_paper_ids": list(run_state.locked_candidate_paper_ids),
                        }
                    ),
                    data={
                        "error": "acquire_locked_candidates_before_more_search",
                        "message": "screen_papers already locked candidates for this cycle; acquire them before running more search_papers.",
                        "locked_paper_ids": list(run_state.locked_candidate_paper_ids),
                    },
                )
            payload = workspace.search_papers(
                query_plan_id=query_plan_id,
                query_text=query_text,
                lane=lane,
                reason=reason,
            )
            run_state.record_search_results(payload)
            return ToolResult(
                observation=_json_preview(
                    {
                        "count": len(payload),
                        "paper_ids": [item.get("paper_id") for item in payload[:5]],
                    }
                ),
                data={"papers": payload},
            )

        def screen_papers(
            paper_ids: Optional[List[str]] = None,
            max_candidates: int = 3,
        ) -> ToolResult:
            """Rank searched papers and lock the strongest candidates before acquisition."""
            if not run_state.searched_paper_ids:
                return _policy_block(
                    "screen_papers requires prior search_papers results in this cycle.",
                    code="screen_requires_search",
                )
            candidate_ids = [
                paper_id
                for paper_id in list(paper_ids or run_state.search_ordered_paper_ids or sorted(run_state.searched_paper_ids))
                if str(paper_id or "").strip() in run_state.searched_paper_ids
            ]
            if not candidate_ids:
                return _policy_block(
                    "screen_papers received no valid paper_ids from this cycle's search results.",
                    code="screen_requires_valid_candidates",
                )
            payload = self._screen_candidate_papers(
                workspace=workspace,
                cycle_number=cycle_number,
                open_review_items=open_review_items,
                paper_ids=candidate_ids,
                max_candidates=max_candidates,
            )
            run_state.record_screening(payload)
            observation_payload = {
                "locked_candidates": [
                    {
                        "paper_id": item.get("paper_id"),
                        "title": item.get("title"),
                        "reason": item.get("reason"),
                    }
                    for item in list(payload.get("ranked_candidates") or [])
                    if str(item.get("decision") or "") == "lock"
                ],
                "dropped_candidates": [
                    {
                        "paper_id": item.get("paper_id"),
                        "title": item.get("title"),
                        "reason": item.get("reason"),
                    }
                    for item in list(payload.get("ranked_candidates") or [])
                    if str(item.get("decision") or "") == "drop"
                ][:5],
            }
            return ToolResult(
                observation=_json_preview(observation_payload),
                data=payload,
            )

        def acquire_document(paper_id: str) -> ToolResult:
            """Acquire the selected paper and build its section index."""
            normalized_paper_id = str(paper_id or "").strip()
            if normalized_paper_id not in run_state.searched_paper_ids:
                return _policy_block(
                    f"acquire_document requires a paper selected from prior search_papers results; unknown paper_id={normalized_paper_id}.",
                    code="paper_not_searched",
                )
            if run_state.screening_required():
                return _policy_block(
                    "screen_papers must be called after the latest search_papers results and before acquire_document.",
                    code="screen_required_before_acquire",
                )
            if run_state.locked_candidate_paper_ids and normalized_paper_id not in set(run_state.locked_candidate_paper_ids):
                locked_payload = [
                    {
                        "paper_id": item.get("paper_id"),
                        "title": item.get("title"),
                    }
                    for item in list(run_state.candidate_screening or [])
                    if str(item.get("paper_id") or "") in set(run_state.locked_candidate_paper_ids)
                ]
                return ToolResult(
                    observation=_json_preview(
                        {
                            "error": "paper_not_screen_locked",
                            "message": (
                                f"acquire_document requires a paper locked by screen_papers; "
                                f"paper_id={normalized_paper_id} was not locked."
                            ),
                            "locked_candidates": locked_payload,
                        }
                    ),
                    data={
                        "error": "paper_not_screen_locked",
                        "message": (
                            f"acquire_document requires a paper locked by screen_papers; "
                            f"paper_id={normalized_paper_id} was not locked."
                        ),
                        "locked_paper_ids": list(run_state.locked_candidate_paper_ids),
                    },
                )
            payload = workspace.acquire_document(paper_id=paper_id)
            run_state.record_acquisition(payload)
            return ToolResult(observation=_json_preview(payload), data=payload)

        def read_sections(
            paper_id: str,
            section_ids: Optional[List[str]] = None,
            preferred_sections: bool = False,
        ) -> ToolResult:
            """Read indexed sections from an acquired paper."""
            normalized_paper_id = str(paper_id or "").strip()
            if normalized_paper_id not in run_state.acquired_paper_ids:
                return _policy_block(
                    f"read_sections requires acquire_document first for paper_id={normalized_paper_id}.",
                    code="paper_not_acquired",
                )
            payload = workspace.read_sections(
                paper_id=paper_id,
                section_ids=section_ids,
                preferred_sections=preferred_sections,
            )
            run_state.record_sections(normalized_paper_id, payload)
            return ToolResult(
                observation=_json_preview(
                    {
                        "count": len(payload),
                        "section_ids": [item.get("section_id") for item in payload],
                    }
                ),
                data={"sections": payload},
            )

        def extract_evidence(
            paper_id: str,
            section_ids: Optional[List[str]] = None,
            preferred_sections: bool = False,
        ) -> ToolResult:
            """Extract stable evidence items from selected paper sections."""
            normalized_paper_id = str(paper_id or "").strip()
            if normalized_paper_id not in run_state.acquired_paper_ids:
                return _policy_block(
                    f"extract_evidence requires acquire_document first for paper_id={normalized_paper_id}.",
                    code="paper_not_acquired",
                )
            payload = workspace.extract_evidence(
                paper_id=paper_id,
                section_ids=section_ids,
                preferred_sections=preferred_sections,
            )
            run_state.record_evidence(normalized_paper_id, payload)
            return ToolResult(
                observation=_json_preview(
                    {
                        "count": len(payload),
                        "evidence_ids": [item.get("evidence_id") for item in payload[:5]],
                    }
                ),
                data={"evidence": payload},
            )

        def fetch_citation_context(
            paper_id: Optional[str] = None,
            section_id: Optional[str] = None,
            evidence_id: Optional[str] = None,
            citation_id: Optional[str] = None,
        ) -> ToolResult:
            """Fetch precise source context for a citation or evidence reference."""
            payload = workspace.fetch_citation_context(
                paper_id=paper_id,
                section_id=section_id,
                evidence_id=evidence_id,
                citation_id=citation_id,
            )
            return ToolResult(observation=_json_preview(payload), data=payload)

        def inspect_entity_cache(
            name: Optional[str] = None,
            entity_type: Optional[str] = None,
            limit: int = 10,
        ) -> ToolResult:
            """Inspect resolved entity cache entries available to the current run."""
            payload = workspace.inspect_entity_cache(name=name, entity_type=entity_type, limit=limit)
            return ToolResult(observation=_json_preview(payload), data=payload)

        def inspect_submission_anchor(
            section_id: Optional[str] = None,
            step_number: Optional[int] = None,
            review_id: Optional[str] = None,
        ) -> ToolResult:
            """Inspect submission, trajectory, or review anchor context by stable reference."""
            payload = workspace.inspect_submission_anchor(
                section_id=section_id,
                step_number=step_number,
                review_id=review_id,
            )
            return ToolResult(observation=_json_preview(payload), data=payload)

        def analyze_submission_gap() -> ToolResult:
            """Analyze open review items and identify structured submission gaps."""
            payload = workspace.analyze_submission_gap()
            return ToolResult(observation=_json_preview(payload), data=payload)

        def conclude(submission: Any) -> ToolResult:
            """Validate and submit the final AnswerSubmission payload for this cycle."""
            if not run_state.has_any_evidence():
                return _policy_block(
                    "conclude is blocked until extract_evidence produces at least one evidence anchor in this cycle.",
                    code="evidence_required_before_conclude",
                )
            try:
                submission = _tool_plain_payload(submission)
                agent = agent_holder.get("agent")
                payload = self._validate_submission_payload(
                    workspace=workspace,
                    submission=submission,
                    cycle_number=cycle_number,
                    open_review_items=open_review_items,
                    agent=agent,
                    run_state=run_state,
                )
            except Exception as exc:
                return ToolResult(
                    observation=f"Invalid AnswerSubmission: {exc}",
                    data={"__conclude_valid__": False, "error": str(exc)},
                )
            return ToolResult(
                observation=json.dumps(payload.model_dump(exclude_none=True), ensure_ascii=False),
                data={"__conclude_valid__": True, "submission": payload.model_dump(exclude_none=True)},
            )

        tools = [
            StructuredTool.from_function(plan_queries, name="plan_queries", args_schema=tool_schemas.PlanQueriesToolInput),
            StructuredTool.from_function(search_papers, name="search_papers", args_schema=tool_schemas.SearchPapersToolInput),
            StructuredTool.from_function(screen_papers, name="screen_papers", args_schema=tool_schemas.ScreenPapersToolInput),
            StructuredTool.from_function(acquire_document, name="acquire_document", args_schema=tool_schemas.AcquireDocumentToolInput),
            StructuredTool.from_function(read_sections, name="read_sections", args_schema=tool_schemas.SectionAccessToolInput),
            StructuredTool.from_function(extract_evidence, name="extract_evidence", args_schema=tool_schemas.SectionAccessToolInput),
            StructuredTool.from_function(
                fetch_citation_context,
                name="fetch_citation_context",
                args_schema=tool_schemas.FetchCitationContextToolInput,
            ),
            StructuredTool.from_function(
                inspect_entity_cache,
                name="inspect_entity_cache",
                args_schema=tool_schemas.InspectEntityCacheToolInput,
            ),
            StructuredTool.from_function(
                inspect_submission_anchor,
                name="inspect_submission_anchor",
                args_schema=tool_schemas.InspectSubmissionAnchorToolInput,
            ),
            StructuredTool.from_function(
                analyze_submission_gap,
                name="analyze_submission_gap",
                args_schema=tool_schemas.EmptyToolInput,
            ),
            StructuredTool.from_function(conclude, name="conclude", args_schema=tool_schemas.ProposerConcludeToolInput),
        ]
        conclude_call_contract = self._build_submission_prompt_contract(
            workspace=workspace,
            cycle_number=cycle_number,
            open_review_items=open_review_items,
        )
        system_prompt = self._build_system_prompt(conclude_call_contract=conclude_call_contract)
        try:
            agent = ReActAgent(
                agent_id="qa_react_proposer",
                name="ProposerAgent",
                model_config=self.model_config,
                system_prompt=system_prompt,
                max_react_steps=self.max_steps_initial if cycle_number == 1 else self.max_steps_revision,
                verbose=False,
                tools=tools,
                thought_phase_instruction=self._thought_instruction(),
                action_phase_instruction=self._action_instruction(
                    PROPOSER_TOOL_NAMES,
                    conclude_call_contract=conclude_call_contract,
                ),
                search_tool_names=[
                    "plan_queries",
                    "search_papers",
                    "screen_papers",
                    "acquire_document",
                    "read_sections",
                    "extract_evidence",
                    "fetch_citation_context",
                    "inspect_entity_cache",
                    "inspect_submission_anchor",
                ],
                analysis_tool_names=["analyze_submission_gap", "conclude"],
                conclude_argument_name="submission",
                conclude_output_kind="submission",
            )
            agent_holder["agent"] = agent
            response, trajectory = agent.generate_response_with_react(
                query=self._build_user_prompt(
                    workspace=workspace,
                    cycle_number=cycle_number,
                    open_review_items=open_review_items,
                    conclude_call_contract=conclude_call_contract,
                ),
                system_prompt_override=system_prompt,
                max_steps_override=self.max_steps_initial if cycle_number == 1 else self.max_steps_revision,
                llm_timeout_seconds=self.llm_timeout_seconds,
            )
        except Exception as exc:
            response_shim = SimpleNamespace(
                content="",
                structured_output=getattr(exc, "structured_output", None),
                response_content=getattr(exc, "response_content", None),
            )
            partial_trajectory = getattr(exc, "trajectory", None) or getattr(agent_holder.get("agent"), "current_trajectory", None)
            salvaged_submission, salvaged_payload, salvaged_error = self._try_validate_salvaged_submission_payload(
                workspace=workspace,
                response=response_shim,
                cycle_number=cycle_number,
                open_review_items=open_review_items,
                trajectory=partial_trajectory,
                run_state=run_state,
                agent=agent_holder.get("agent"),
            )
            if salvaged_submission is not None:
                logger.warning(
                    "react_reviewed_proposer_payload_salvaged cycle=%s source=forced_conclude_exception",
                    cycle_number,
                )
                return salvaged_submission, partial_trajectory
            if salvaged_payload is not None:
                error = ReactReviewedStructuredOutputError(
                    stage="proposer",
                    cycle_number=cycle_number,
                    message=f"invalid proposer structured output: {salvaged_error or exc}",
                    response_content=getattr(exc, "response_content", None),
                    structured_output={"kind": "submission", "payload": salvaged_payload},
                    trajectory=partial_trajectory,
                )
                _store_invalid_llm_output(
                    artifact_store=workspace.store,
                    prefix=f"proposer_cycle_{cycle_number}",
                    error=error,
                )
                logger.warning(
                    "react_reviewed_proposer_llm_failed cycle=%s source=forced_conclude_exception error=%s",
                    cycle_number,
                    salvaged_error or exc,
                )
                if self.repair_attempts > 0:
                    return self._repair_submission_with_llm(
                        workspace=workspace,
                        cycle_number=cycle_number,
                        open_review_items=open_review_items,
                        error=error,
                        trajectory=partial_trajectory,
                        run_state=run_state,
                    )
                raise error
            if self.repair_attempts > 0 and (
                workspace.current_submission is not None
                or run_state.has_any_evidence()
                or bool(workspace.evidence_items)
            ):
                repair_error = ReactReviewedStructuredOutputError(
                    stage="proposer",
                    cycle_number=cycle_number,
                    message=f"forced conclude execution failed before a valid payload was emitted: {salvaged_error or exc}",
                    response_content=getattr(exc, "response_content", None),
                    structured_output=getattr(exc, "structured_output", None),
                    trajectory=partial_trajectory,
                )
                logger.warning(
                    "react_reviewed_proposer_forced_conclude_exception_entering_repair cycle=%s",
                    cycle_number,
                )
                return self._repair_submission_with_llm(
                    workspace=workspace,
                    cycle_number=cycle_number,
                    open_review_items=open_review_items,
                    error=repair_error,
                    trajectory=partial_trajectory or ReActTrajectory(query=workspace.question),
                    run_state=run_state,
                )
            self._raise_execution_failure(
                workspace=workspace,
                cycle_number=cycle_number,
                stage="proposer_execution",
                message=f"Proposer failed during ReAct execution: {salvaged_error or exc}",
                details={
                    "fallback_mode": self.fallback_mode,
                    "salvaged_payload_available": isinstance(salvaged_payload, dict),
                },
                response_content=getattr(exc, "response_content", None),
                structured_output=(
                    {"kind": "submission", "payload": salvaged_payload}
                    if isinstance(salvaged_payload, dict)
                    else getattr(exc, "structured_output", None)
                ),
            )
        try:
            payload = self._parse_submission_response(response=response)
            submission = self._validate_submission_payload(
                workspace=workspace,
                submission=payload,
                cycle_number=cycle_number,
                open_review_items=open_review_items,
                agent=None,
                trajectory=trajectory,
                run_state=run_state,
            )
            return submission, trajectory
        except Exception as exc:
            salvaged_submission, salvaged_payload, salvaged_error = self._try_validate_salvaged_submission_payload(
                workspace=workspace,
                response=response,
                cycle_number=cycle_number,
                open_review_items=open_review_items,
                trajectory=trajectory,
                run_state=run_state,
            )
            if salvaged_submission is not None:
                logger.warning(
                    "react_reviewed_proposer_payload_salvaged cycle=%s source=forced_conclude_diagnostics",
                    cycle_number,
                )
                return salvaged_submission, trajectory
            error = ReactReviewedStructuredOutputError(
                stage="proposer",
                cycle_number=cycle_number,
                message=f"invalid proposer structured output: {salvaged_error or exc}",
                response_content=getattr(response, "response_content", getattr(response, "content", None)),
                structured_output=(
                    {"kind": "submission", "payload": salvaged_payload}
                    if isinstance(salvaged_payload, dict)
                    else getattr(response, "structured_output", None)
                ),
                trajectory=trajectory,
            )
            _store_invalid_llm_output(
                artifact_store=workspace.store,
                prefix=f"proposer_cycle_{cycle_number}",
                error=error,
            )
            logger.warning("react_reviewed_proposer_llm_failed cycle=%s error=%s", cycle_number, exc)
            if self.repair_attempts > 0:
                return self._repair_submission_with_llm(
                    workspace=workspace,
                    cycle_number=cycle_number,
                    open_review_items=open_review_items,
                    error=error,
                    trajectory=trajectory,
                    run_state=run_state,
                )
            raise error

    def _parse_submission_response(self, *, response: Any) -> Dict[str, Any]:
        payload = _extract_stage_payload(response=response, expected_kind="submission")
        if not isinstance(payload, dict):
            raise ValueError("submission payload must be a JSON object.")
        return payload

    def _build_submission_prompt_scaffold(
        self,
        *,
        workspace: ReactReviewedWorkspace,
        cycle_number: int,
        open_review_items: Sequence[ReviewItem],
    ) -> Dict[str, Any]:
        return build_submission_prompt_scaffold(
            question=workspace.question,
            cycle_number=cycle_number,
            answer_sections=[
                {"section_id": section.section_id, "title": section.title}
                for section in workspace.task_spec.answer_sections
            ],
            issue_refs=[item.review_id for item in open_review_items if item.review_id],
        )

    def _build_submission_prompt_contract(
        self,
        *,
        workspace: ReactReviewedWorkspace,
        cycle_number: int,
        open_review_items: Sequence[ReviewItem],
    ) -> Dict[str, Any]:
        return build_submission_prompt_contract(
            question=workspace.question,
            cycle_number=cycle_number,
            answer_sections=[
                {"section_id": section.section_id, "title": section.title}
                for section in workspace.task_spec.answer_sections
            ],
            issue_refs=[item.review_id for item in open_review_items if item.review_id],
        )

    def _patch_submission_from_prior_context(
        self,
        *,
        workspace: ReactReviewedWorkspace,
        raw_payload: Dict[str, Any],
        open_review_items: Sequence[ReviewItem],
    ) -> Dict[str, Any]:
        prior_submission = workspace.current_submission
        if prior_submission is None:
            return raw_payload
        prior_payload = prior_submission.model_dump(exclude_none=True)
        if not _normalize_list_payload(raw_payload.get("citations")):
            raw_payload["citations"] = copy.deepcopy(prior_payload.get("citations") or [])
        if not _normalize_list_payload(raw_payload.get("limitations")):
            raw_payload["limitations"] = copy.deepcopy(prior_payload.get("limitations") or [])
        if not _normalize_list_payload(raw_payload.get("step_refs")):
            raw_payload["step_refs"] = copy.deepcopy(prior_payload.get("step_refs") or [])
        raw_payload["issue_refs"] = _merge_unique_text(
            _normalize_list_payload(raw_payload.get("issue_refs")),
            [item.review_id for item in open_review_items] + list(prior_submission.issue_refs or []),
        )

        raw_sections = [
            copy.deepcopy(item)
            for item in list(_normalize_list_payload(raw_payload.get("sections")) or [])
            if isinstance(item, dict)
        ]
        raw_sections_by_id = {
            str(item.get("section_id") or "").strip(): item
            for item in raw_sections
            if str(item.get("section_id") or "").strip()
        }
        for prior_section in prior_submission.sections:
            current = raw_sections_by_id.get(prior_section.section_id)
            if current is None:
                cloned = prior_section.model_dump(exclude_none=True)
                cloned["issue_refs"] = _merge_unique_text(
                    cloned.get("issue_refs"),
                    [item.review_id for item in open_review_items],
                )
                raw_sections.append(cloned)
                raw_sections_by_id[prior_section.section_id] = cloned
                continue
            if not _compact_text(current.get("content")):
                current["content"] = prior_section.content
            if not _normalize_list_payload(current.get("citation_ids")):
                current["citation_ids"] = list(prior_section.citation_ids or [])
            if not _normalize_list_payload(current.get("step_refs")):
                current["step_refs"] = [item.model_dump(exclude_none=True) for item in prior_section.step_refs]
            if not isinstance(current.get("section_confidence"), dict):
                current["section_confidence"] = prior_section.section_confidence.model_dump(exclude_none=True)
            current["issue_refs"] = _merge_unique_text(
                _normalize_list_payload(current.get("issue_refs")),
                list(prior_section.issue_refs or []) + [item.review_id for item in open_review_items],
            )
        raw_payload["sections"] = raw_sections
        if raw_payload.get("overall_confidence") in (None, "", []):
            raw_payload["overall_confidence"] = prior_payload.get("overall_confidence")
        return raw_payload

    def _normalize_abstract_only_degradation(
        self,
        *,
        raw_payload: Dict[str, Any],
        run_state: Optional[_ProposerRunState],
    ) -> Dict[str, Any]:
        if run_state is None or not run_state.has_any_evidence() or run_state.has_fulltext_evidence():
            return raw_payload
        limitations = _normalize_list_payload(raw_payload.get("limitations"))
        limitation_text = " ".join(str(item or "").strip().lower() for item in limitations)
        if "abstract" not in limitation_text or ("full text" not in limitation_text and "full-text" not in limitation_text):
            limitations = _merge_unique_text(
                limitations,
                [
                    "This submission is degraded because only abstract-backed evidence was available and no usable full text could be recovered in this cycle."
                ],
            )
        raw_payload["limitations"] = limitations
        current_confidence = _normalize_confidence_payload(
            raw_payload.get("overall_confidence"),
            rationale="Overall confidence normalized for degraded abstract-only evidence.",
        )
        if current_confidence.get("score", 0.0) > 0.45 or current_confidence.get("level") == "high":
            current_confidence["level"] = "low"
            current_confidence["score"] = 0.4
            current_confidence["rationale"] = (
                "Only abstract-backed evidence was available in this cycle, so overall confidence was lowered."
            )
        raw_payload["overall_confidence"] = current_confidence
        return raw_payload

    def _build_fallback_submission_citations(
        self,
        *,
        workspace: ReactReviewedWorkspace,
        run_state: Optional[_ProposerRunState],
    ) -> List[Dict[str, Any]]:
        if run_state is None:
            return []
        fallback_citations: List[Dict[str, Any]] = []
        candidate_paper_ids = list(sorted(run_state.acquired_paper_ids or run_state.searched_paper_ids))
        for paper_id in candidate_paper_ids[:4]:
            paper_record = workspace.paper_records.get(paper_id)
            paper_candidate = workspace.paper_candidates.get(paper_id)
            section_ids = list(sorted(run_state.section_ids_by_paper.get(paper_id, set())))
            evidence_ids = list(sorted(run_state.evidence_ids_by_paper.get(paper_id, set())))
            if not section_ids and not evidence_ids:
                fulltext_status = str(run_state.fulltext_status_by_paper.get(paper_id) or "").strip().lower()
                has_useful_abstract = _paper_has_useful_abstract_support(
                    task_spec=workspace.task_spec,
                    entity_pack=workspace.entity_pack,
                    title=(
                        getattr(paper_record, "title", None)
                        if paper_record is not None
                        else getattr(paper_candidate, "title", None)
                    ),
                    abstract=(
                        getattr(paper_record, "abstract", None)
                        if paper_record is not None
                        else getattr(paper_candidate, "abstract", None)
                    ),
                )
                if has_useful_abstract and (
                    fulltext_status == "abstract_only"
                    or (
                        paper_id in run_state.acquired_paper_ids
                        and fulltext_status in {"fulltext_unusable", "binary_only", "missing", "error"}
                    )
                ):
                    section_ids = ["sec_abstract"]
            if not section_ids and not evidence_ids:
                continue
            fallback_citations.append(
                {
                    "citation_id": f"CIT-{len(fallback_citations) + 1}",
                    "paper_id": paper_id,
                    "doi": getattr(paper_record, "doi", None) if paper_record is not None else None,
                    "title": (
                        getattr(paper_record, "title", None)
                        if paper_record is not None
                        else getattr(paper_candidate, "title", None)
                    )
                    or paper_id,
                    "year": getattr(paper_record, "year", None) if paper_record is not None else None,
                    "venue": getattr(paper_record, "venue", None) if paper_record is not None else None,
                    "section_ids": section_ids,
                    "evidence_ids": evidence_ids,
                }
            )
        return fallback_citations

    def _normalize_submission_citations_for_run_state(
        self,
        *,
        workspace: ReactReviewedWorkspace,
        raw_citations: Sequence[Any],
        run_state: Optional[_ProposerRunState],
    ) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        if run_state is None:
            return [copy.deepcopy(item) for item in list(raw_citations or []) if isinstance(item, dict)]
        for index, raw_item in enumerate(list(raw_citations or []), start=1):
            if not isinstance(raw_item, dict):
                continue
            current = copy.deepcopy(raw_item)
            paper_id = str(current.get("paper_id") or "").strip()
            if not paper_id:
                continue
            if paper_id not in run_state.searched_paper_ids and paper_id not in run_state.acquired_paper_ids:
                continue
            valid_section_ids: List[str] = []
            for section_id in list(current.get("section_ids") or []):
                normalized_section_id = str(section_id or "").strip()
                if not normalized_section_id:
                    continue
                if normalized_section_id == "sec_abstract":
                    has_useful_abstract = _paper_has_useful_abstract_support(
                        task_spec=workspace.task_spec,
                        entity_pack=workspace.entity_pack,
                        title=(
                            getattr(workspace.paper_records.get(paper_id), "title", None)
                            or getattr(workspace.paper_candidates.get(paper_id), "title", None)
                        ),
                        abstract=(
                            getattr(workspace.paper_records.get(paper_id), "abstract", None)
                            or getattr(workspace.paper_candidates.get(paper_id), "abstract", None)
                        ),
                    )
                    if paper_id in run_state.acquired_paper_ids and has_useful_abstract:
                        valid_section_ids.append("sec_abstract")
                    continue
                if normalized_section_id in run_state.section_ids_by_paper.get(paper_id, set()):
                    valid_section_ids.append(normalized_section_id)
            valid_evidence_ids: List[str] = []
            valid_evidence_for_paper = run_state.evidence_ids_by_paper.get(paper_id, set())
            for evidence_id in list(current.get("evidence_ids") or []):
                normalized_evidence_id = str(evidence_id or "").strip()
                if not normalized_evidence_id:
                    continue
                if normalized_evidence_id in run_state.evidence_ids and (
                    not valid_evidence_for_paper or normalized_evidence_id in valid_evidence_for_paper
                ):
                    valid_evidence_ids.append(normalized_evidence_id)
            if not valid_section_ids and not valid_evidence_ids:
                fulltext_status = str(run_state.fulltext_status_by_paper.get(paper_id) or "").strip().lower()
                has_useful_abstract = _paper_has_useful_abstract_support(
                    task_spec=workspace.task_spec,
                    entity_pack=workspace.entity_pack,
                    title=(
                        getattr(workspace.paper_records.get(paper_id), "title", None)
                        or getattr(workspace.paper_candidates.get(paper_id), "title", None)
                    ),
                    abstract=(
                        getattr(workspace.paper_records.get(paper_id), "abstract", None)
                        or getattr(workspace.paper_candidates.get(paper_id), "abstract", None)
                    ),
                )
                if paper_id in run_state.acquired_paper_ids and has_useful_abstract and fulltext_status in {
                    "abstract_only",
                    "fulltext_unusable",
                    "binary_only",
                    "missing",
                    "error",
                }:
                    valid_section_ids = ["sec_abstract"]
            current["citation_id"] = str(current.get("citation_id") or f"CIT-{index}").strip() or f"CIT-{index}"
            current["paper_id"] = paper_id
            current["title"] = _compact_text(current.get("title")) or _paper_title_fallback(workspace, paper_id) or paper_id
            current["section_ids"] = _merge_unique_text([], valid_section_ids)
            current["evidence_ids"] = _merge_unique_text([], valid_evidence_ids)
            normalized.append(current)
        return normalized

    def _salvage_submission_payload(
        self,
        *,
        response: Any,
        trajectory: Optional[ReActTrajectory],
    ) -> Optional[Dict[str, Any]]:
        candidate_payloads: List[Any] = []
        response_content = getattr(response, "response_content", None)
        candidate_payloads.extend(_extract_json_candidates(getattr(response, "structured_output", None)))
        candidate_payloads.extend(_extract_json_candidates(getattr(response, "content", None)))
        candidate_payloads.extend(_extract_json_candidates(response_content))

        if isinstance(response_content, dict):
            for key in (
                "forced_conclude_action_response",
                "forced_conclude_structured_json_response",
            ):
                message_payload = response_content.get(key)
                if not isinstance(message_payload, dict):
                    continue
                candidate_payloads.extend(_extract_json_candidates(message_payload.get("content")))
                candidate_payloads.extend(_extract_json_candidates(message_payload.get("additional_kwargs")))
                function_call = message_payload.get("function_call")
                if function_call is not None:
                    candidate_payloads.extend(_extract_json_candidates(function_call))
                    if isinstance(function_call, dict):
                        candidate_payloads.extend(_extract_json_candidates(function_call.get("arguments")))
                for tool_call in list(message_payload.get("tool_calls") or []):
                    if not isinstance(tool_call, dict):
                        continue
                    candidate_payloads.extend(_extract_json_candidates(tool_call))
                    candidate_payloads.extend(_extract_json_candidates(tool_call.get("args")))
                    function_payload = tool_call.get("function")
                    if isinstance(function_payload, dict):
                        candidate_payloads.extend(_extract_json_candidates(function_payload.get("arguments")))

        for step in reversed(list(getattr(trajectory, "steps", []) or [])):
            if getattr(step, "action", "") == "conclude":
                candidate_payloads.extend(_extract_json_candidates(getattr(step, "action_input", None)))
                candidate_payloads.extend(_extract_json_candidates(getattr(step, "observation_data", None)))
                candidate_payloads.extend(_extract_json_candidates(getattr(step, "observation", None)))
            for call in reversed(list(getattr(step, "tool_calls", []) or [])):
                if getattr(call, "tool_name", "") != "conclude":
                    continue
                candidate_payloads.extend(_extract_json_candidates(getattr(call, "tool_args", None)))
                candidate_payloads.extend(_extract_json_candidates(getattr(call, "observation_data", None)))
                candidate_payloads.extend(_extract_json_candidates(getattr(call, "observation", None)))

        seen_payloads = set()
        for payload in candidate_payloads:
            payload_key = _compact_text(
                json.dumps(payload, ensure_ascii=False, default=str) if isinstance(payload, (dict, list)) else str(payload)
            )
            if payload_key in seen_payloads:
                continue
            seen_payloads.add(payload_key)
            submission_payload = _coerce_salvaged_submission_payload(payload)
            if isinstance(submission_payload, dict):
                return submission_payload
        return None

    def _try_validate_salvaged_submission_payload(
        self,
        *,
        workspace: ReactReviewedWorkspace,
        response: Any,
        cycle_number: int,
        open_review_items: Sequence[ReviewItem],
        trajectory: Optional[ReActTrajectory],
        run_state: Optional[_ProposerRunState],
        agent: Optional[ReActAgent] = None,
    ) -> Tuple[Optional[AnswerSubmission], Optional[Dict[str, Any]], Optional[Exception]]:
        salvaged_payload = self._salvage_submission_payload(response=response, trajectory=trajectory)
        if salvaged_payload is None:
            return None, None, None
        try:
            submission = self._validate_submission_payload(
                workspace=workspace,
                submission=salvaged_payload,
                cycle_number=cycle_number,
                open_review_items=open_review_items,
                agent=agent,
                trajectory=trajectory,
                run_state=run_state,
            )
            return submission, salvaged_payload, None
        except Exception as exc:
            return None, salvaged_payload, exc

    def _repair_submission_with_llm(
        self,
        *,
        workspace: ReactReviewedWorkspace,
        cycle_number: int,
        open_review_items: Sequence[ReviewItem],
        error: ReactReviewedStructuredOutputError,
        trajectory: ReActTrajectory,
        run_state: _ProposerRunState,
    ) -> Tuple[AnswerSubmission, ReActTrajectory]:
        last_error: ReactReviewedStructuredOutputError = error
        for attempt_number in range(1, self.repair_attempts + 1):
            try:
                llm = build_chat_model_from_config(self.model_config)
                raw_response = invoke_llm(
                    llm,
                    [
                        {
                            "role": "system",
                            "content": (
                                build_proposer_repair_system_prompt(
                                    conclude_contract=self._build_submission_prompt_contract(
                                        workspace=workspace,
                                        cycle_number=cycle_number,
                                        open_review_items=open_review_items,
                                    )
                                )
                            ),
                        },
                        {
                            "role": "user",
                            "content": _json_preview(
                                {
                                    "cycle_number": cycle_number,
                                    "repair_attempt": attempt_number,
                                    "question": workspace.question,
                                    "context": workspace.context,
                                    "task_spec": workspace.task_spec.model_dump(exclude_none=True),
                                    "entity_pack": workspace.entity_pack.model_dump(exclude_none=True),
                                    "prior_submission": (
                                        workspace.current_submission.model_dump(exclude_none=True)
                                        if workspace.current_submission is not None
                                        else None
                                    ),
                                    "open_review_items": [item.model_dump(exclude_none=True) for item in open_review_items],
                                    "conclude_call_contract": self._build_submission_prompt_contract(
                                        workspace=workspace,
                                        cycle_number=cycle_number,
                                        open_review_items=open_review_items,
                                    ),
                                    "retrieval_state": run_state.prompt_payload(),
                                    "validation_error": str(last_error),
                                    "invalid_submission_payload": (
                                        _coerce_salvaged_submission_payload(last_error.structured_output)
                                        if isinstance(last_error.structured_output, dict)
                                        else None
                                    ),
                                    "invalid_response_content": (
                                        None
                                        if isinstance(last_error.structured_output, dict)
                                        else last_error.response_content
                                    ),
                                },
                                limit=12000,
                            ),
                        },
                    ],
                )
            except Exception as exc:
                self._raise_execution_failure(
                    workspace=workspace,
                    cycle_number=cycle_number,
                    stage="proposer_repair",
                    message=f"Proposer repair attempt failed to execute: {exc}",
                    details={"attempt": attempt_number},
                    trajectory=trajectory,
                )
            try:
                payload = self._parse_submission_response(
                    response=AgentResponse(content=raw_response)
                )
                submission = self._validate_submission_payload(
                    workspace=workspace,
                    submission=payload,
                    cycle_number=cycle_number,
                    open_review_items=open_review_items,
                    agent=None,
                    trajectory=trajectory,
                    run_state=run_state,
                )
                logger.info("react_reviewed_proposer_repair_succeeded cycle=%s attempt=%s", cycle_number, attempt_number)
                return submission, trajectory
            except Exception as exc:
                salvage_payload = None
                salvage_error: Optional[Exception] = None
                salvage_sources = [
                    SimpleNamespace(content=raw_response, structured_output=None, response_content=raw_response),
                    SimpleNamespace(
                        content=raw_response,
                        structured_output=last_error.structured_output,
                        response_content=last_error.response_content,
                    ),
                ]
                for salvage_source in salvage_sources:
                    salvaged_submission, salvaged_payload, salvaged_error = self._try_validate_salvaged_submission_payload(
                        workspace=workspace,
                        response=salvage_source,
                        cycle_number=cycle_number,
                        open_review_items=open_review_items,
                        trajectory=trajectory,
                        run_state=run_state,
                    )
                    if salvaged_submission is not None:
                        logger.warning(
                            "react_reviewed_proposer_repair_salvaged cycle=%s attempt=%s source=invalid_payload",
                            cycle_number,
                            attempt_number,
                        )
                        return salvaged_submission, trajectory
                    if salvaged_payload is not None:
                        salvage_payload = salvaged_payload
                        salvage_error = salvaged_error
                rebuilt_submission = self._try_rebuild_submission_from_workspace(
                    workspace=workspace,
                    cycle_number=cycle_number,
                    open_review_items=open_review_items,
                    trajectory=trajectory,
                    run_state=run_state,
                )
                if rebuilt_submission is not None:
                    logger.warning(
                        "react_reviewed_proposer_repair_rebuilt_from_workspace cycle=%s attempt=%s",
                        cycle_number,
                        attempt_number,
                    )
                    return rebuilt_submission, trajectory
                last_error = ReactReviewedStructuredOutputError(
                    stage="proposer_repair",
                    cycle_number=cycle_number,
                    message=f"invalid proposer repair output: {salvage_error or exc}",
                    response_content=raw_response,
                    structured_output=(
                        {"kind": "submission", "payload": salvage_payload}
                        if isinstance(salvage_payload, dict)
                        else None
                    ),
                    trajectory=trajectory,
                )
                _store_invalid_llm_output(
                    artifact_store=workspace.store,
                    prefix=f"proposer_cycle_{cycle_number}_repair_{attempt_number}",
                    error=last_error,
                )
        self._raise_execution_failure(
            workspace=workspace,
            cycle_number=cycle_number,
            stage="proposer_repair",
            message=f"Proposer repair attempt failed validation: {last_error}",
            details={"attempts_used": self.repair_attempts},
            response_content=last_error.response_content,
            structured_output=last_error.structured_output,
            trajectory=trajectory,
        )

    def _raise_execution_failure(
        self,
        *,
        workspace: ReactReviewedWorkspace,
        cycle_number: int,
        stage: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        response_content: Any = None,
        structured_output: Any = None,
        trajectory: Optional[ReActTrajectory] = None,
    ) -> None:
        error = ReactReviewedProposerExecutionError(
            stage=stage,
            cycle_number=cycle_number,
            message=message,
            details=details,
            response_content=response_content,
            structured_output=structured_output,
            trajectory=trajectory,
        )
        _store_execution_failure(
            artifact_store=workspace.store,
            prefix=f"proposer_cycle_{cycle_number}",
            error=error,
        )
        logger.warning("react_reviewed_proposer_execution_failed cycle=%s stage=%s error=%s", cycle_number, stage, message)
        raise error

    def _validate_submission_payload(
        self,
        *,
        workspace: ReactReviewedWorkspace,
        submission: Any,
        cycle_number: int,
        open_review_items: Sequence[ReviewItem],
        agent: Optional[ReActAgent],
        trajectory: Optional[ReActTrajectory] = None,
        run_state: Optional[_ProposerRunState] = None,
    ) -> AnswerSubmission:
        raw_payload = copy.deepcopy(submission if isinstance(submission, dict) else json.loads(json.dumps(submission)))
        trajectory_id = None
        if trajectory is not None:
            trajectory_id = trajectory.trajectory_id
        elif agent is not None and getattr(agent, "current_trajectory", None) is not None:
            trajectory_id = agent.current_trajectory.trajectory_id
        legacy_answer_sections = raw_payload.pop("answer_sections", None)
        if legacy_answer_sections is not None and "sections" not in raw_payload:
            raw_payload["sections"] = legacy_answer_sections
        raw_payload.pop("cycle_number", None)
        raw_payload.pop("normalized_question", None)
        raw_payload.pop("conditions", None)
        raw_payload.pop("task_spec_version", None)
        if "confidence" in raw_payload and "overall_confidence" not in raw_payload:
            raw_payload["overall_confidence"] = raw_payload.pop("confidence")
        else:
            raw_payload.pop("confidence", None)
        raw_payload = self._patch_submission_from_prior_context(
            workspace=workspace,
            raw_payload=raw_payload,
            open_review_items=open_review_items,
        )
        raw_payload.setdefault("submission_id", f"submission_cycle_{cycle_number}")
        raw_payload.setdefault("question", workspace.question)
        raw_payload.setdefault("version", cycle_number)
        raw_payload.setdefault("trajectory_id", trajectory_id or f"traj_placeholder_{cycle_number}")
        raw_payload = _align_step_refs_to_trajectory(raw_payload, trajectory_id)
        raw_payload.setdefault("citations", [])
        raw_payload["citations"] = self._normalize_submission_citations_for_run_state(
            workspace=workspace,
            raw_citations=list(_normalize_list_payload(raw_payload.get("citations")) or []),
            run_state=run_state,
        )
        raw_citations_by_id = {
            str(item.get("citation_id") or "").strip(): item
            for item in list(raw_payload.get("citations") or [])
            if isinstance(item, dict) and str(item.get("citation_id") or "").strip()
        }
        def _raw_citation_has_fulltext_anchor(citation_payload: Dict[str, Any]) -> bool:
            paper_id = str(citation_payload.get("paper_id") or "").strip()
            evidence_ids = {
                str(item).strip()
                for item in list(citation_payload.get("evidence_ids") or [])
                if str(item).strip()
            }
            if run_state is None or not paper_id:
                return False
            if run_state.fulltext_evidence_ids_for_paper(paper_id).intersection(evidence_ids):
                return True
            if str(run_state.fulltext_status_by_paper.get(paper_id) or "").strip().lower() == "fulltext_indexed":
                for section_id in list(citation_payload.get("section_ids") or []):
                    normalized_section_id = str(section_id or "").strip()
                    if normalized_section_id != "sec_abstract" and normalized_section_id in run_state.section_ids_by_paper.get(paper_id, set()):
                        return True
            return False
        raw_payload.setdefault("limitations", [])
        raw_payload["limitations"] = _normalize_list_payload(raw_payload.get("limitations"))
        raw_payload.setdefault("issue_refs", [item.review_id for item in open_review_items])
        raw_payload["issue_refs"] = _normalize_list_payload(raw_payload.get("issue_refs"))
        raw_payload.setdefault(
            "overall_confidence",
            _confidence(0.65, "LLM proposer confidence normalized by conclude validator.").model_dump(),
        )
        raw_payload["overall_confidence"] = _normalize_confidence_payload(
            raw_payload.get("overall_confidence"),
            rationale="LLM proposer confidence normalized by conclude validator.",
        )
        raw_payload = self._normalize_abstract_only_degradation(
            raw_payload=raw_payload,
            run_state=run_state,
        )
        fallback_citations = self._build_fallback_submission_citations(workspace=workspace, run_state=run_state)
        raw_citations = list(raw_payload.get("citations") or [])
        anchored_input_citation_found = any(
            bool(list(item.get("section_ids") or []) or list(item.get("evidence_ids") or []))
            for item in raw_citations
            if isinstance(item, dict)
        )
        if fallback_citations and (not raw_citations or not anchored_input_citation_found):
            raw_payload["citations"] = fallback_citations
        citation_ids = {
            str(item.get("citation_id") or "").strip()
            for item in list(raw_payload.get("citations") or [])
            if str(item.get("citation_id") or "").strip()
        }
        normalized_sections: List[Dict[str, Any]] = []
        for answer_section in workspace.task_spec.answer_sections:
            section_payload = next(
                (
                    copy.deepcopy(item)
                    for item in list(raw_payload.get("sections") or [])
                    if str(item.get("section_id") or "").strip() == answer_section.section_id
                ),
                None,
            )
            if section_payload is None and answer_section.required:
                raise ValueError(f"Missing required section: {answer_section.section_id}")
            if section_payload is None:
                continue
            section_payload.setdefault("title", answer_section.title)
            if "content" not in section_payload and "text" in section_payload:
                section_payload["content"] = section_payload.pop("text")
            else:
                section_payload.pop("text", None)
            if "confidence" in section_payload and "section_confidence" not in section_payload:
                section_payload["section_confidence"] = section_payload.pop("confidence")
            else:
                section_payload.pop("confidence", None)
            section_payload.setdefault("content", "")
            section_payload.setdefault(
                "section_confidence",
                _confidence(0.6, "Section confidence normalized by conclude validator.").model_dump(),
            )
            section_payload["section_confidence"] = _normalize_confidence_payload(
                section_payload.get("section_confidence"),
                rationale="Section confidence normalized by conclude validator.",
            )
            legacy_section_citations = section_payload.pop("citations", None)
            section_payload.pop("evidence_refs", None)
            step_refs = []
            for item in list(section_payload.get("step_refs") or []):
                if isinstance(item, dict):
                    item.setdefault("trajectory_id", raw_payload["trajectory_id"])
                step_refs.append(item)
            if not step_refs:
                step_refs = [{"trajectory_id": raw_payload["trajectory_id"], "step_number": 1}]
            section_payload["step_refs"] = step_refs
            section_payload["issue_refs"] = _merge_unique_text(section_payload.get("issue_refs"), [item.review_id for item in open_review_items])
            if not section_payload.get("citation_ids") and legacy_section_citations:
                normalized_citation_ids: List[str] = []
                for item in list(legacy_section_citations):
                    if isinstance(item, dict):
                        citation_id = str(item.get("citation_id") or "").strip()
                    else:
                        citation_id = str(item).strip()
                    if citation_id:
                        normalized_citation_ids.append(citation_id)
                section_payload["citation_ids"] = normalized_citation_ids
            section_payload["citation_ids"] = [
                citation_id
                for citation_id in list(section_payload.get("citation_ids") or [])
                if citation_id in citation_ids
            ]
            if (
                run_state is not None
                and self.evidence_policy == "prefer_fulltext"
                and run_state.has_fulltext_evidence()
                and answer_section.section_id not in {"caveats", "causal_limitations", "open_questions"}
            ):
                current_citation_ids = list(section_payload.get("citation_ids") or [])
                has_fulltext_anchor = any(
                    _raw_citation_has_fulltext_anchor(raw_citations_by_id.get(citation_id, {}))
                    for citation_id in current_citation_ids
                )
                if not has_fulltext_anchor:
                    preferred_fulltext_ids = [
                        citation_id
                        for citation_id, citation_payload in raw_citations_by_id.items()
                        if _raw_citation_has_fulltext_anchor(citation_payload)
                    ]
                    if preferred_fulltext_ids:
                        section_payload["citation_ids"] = _merge_unique_text(
                            preferred_fulltext_ids[:1],
                            current_citation_ids,
                        )[:2]
            normalized_sections.append(section_payload)
        raw_payload["sections"] = normalized_sections
        raw_payload["step_refs"] = [
            item
            if not isinstance(item, dict)
            else {**item, "trajectory_id": item.get("trajectory_id") or raw_payload["trajectory_id"]}
            for item in list(_normalize_list_payload(raw_payload.get("step_refs")) or [])
        ] or [{"trajectory_id": raw_payload["trajectory_id"], "step_number": 1}]
        validated = AnswerSubmission.model_validate(raw_payload)
        if run_state is not None:
            self._validate_grounded_submission(
                submission=validated,
                run_state=run_state,
            )
        return validated

    def _validate_grounded_submission(
        self,
        *,
        submission: AnswerSubmission,
        run_state: _ProposerRunState,
    ) -> None:
        errors: List[str] = []
        if not run_state.query_plan_ids:
            errors.append("Submission is invalid because proposer never called plan_queries.")
        if not run_state.searched_paper_ids:
            errors.append("Submission is invalid because proposer never called search_papers.")
        citation_lookup = {citation.citation_id: citation for citation in submission.citations}
        if not citation_lookup:
            errors.append("Submission has no citation catalog.")
        anchored_citation_found = False

        def _citation_has_fulltext_anchor(citation: SubmissionCitation) -> bool:
            if run_state.fulltext_evidence_ids_for_paper(citation.paper_id).intersection(set(citation.evidence_ids or [])):
                return True
            if run_state.fulltext_status_by_paper.get(citation.paper_id) == "fulltext_indexed":
                for section_id in citation.section_ids or []:
                    if section_id != "sec_abstract" and section_id in run_state.section_ids_by_paper.get(citation.paper_id, set()):
                        return True
            return False

        for citation in submission.citations:
            if citation.paper_id not in run_state.searched_paper_ids:
                errors.append(
                    f"Citation '{citation.citation_id}' references paper_id '{citation.paper_id}' that was not returned by search_papers in this cycle."
                )
            if citation.section_ids or citation.evidence_ids:
                if citation.paper_id not in run_state.acquired_paper_ids:
                    errors.append(
                        f"Citation '{citation.citation_id}' references anchors for paper_id '{citation.paper_id}' without acquire_document in this cycle."
                    )
            for section_id in citation.section_ids:
                if section_id == "sec_abstract":
                    continue
                if section_id not in run_state.section_ids_by_paper.get(citation.paper_id, set()):
                    errors.append(
                        f"Citation '{citation.citation_id}' references unknown section_id '{section_id}' for paper_id '{citation.paper_id}'."
                    )
            for evidence_id in citation.evidence_ids:
                if evidence_id not in run_state.evidence_ids:
                    errors.append(
                        f"Citation '{citation.citation_id}' references evidence_id '{evidence_id}' that was not extracted in this cycle."
                    )
            if citation.section_ids or citation.evidence_ids:
                anchored_citation_found = True
        if not anchored_citation_found:
            errors.append("Submission citations exist but none provide section_ids or evidence_ids anchors from this cycle.")

        for section in submission.sections:
            if not section.citation_ids:
                continue
            missing_citations = [citation_id for citation_id in section.citation_ids if citation_id not in citation_lookup]
            if missing_citations:
                errors.append(
                    f"Section '{section.section_id}' references missing citation_ids: {', '.join(missing_citations)}."
                )
                continue
            if self.evidence_policy == "prefer_fulltext" and run_state.has_fulltext_evidence():
                if section.section_id not in {"caveats", "causal_limitations", "open_questions"}:
                    cited = [citation_lookup[citation_id] for citation_id in section.citation_ids]
                    if cited and not any(_citation_has_fulltext_anchor(citation) for citation in cited):
                        errors.append(
                            f"Section '{section.section_id}' must cite at least one fulltext-backed citation because usable fulltext evidence exists in this cycle."
                        )

        if run_state.has_any_evidence() and not run_state.has_fulltext_evidence():
            limitation_text = " ".join(str(item or "").strip().lower() for item in submission.limitations)
            if "abstract" not in limitation_text or ("full text" not in limitation_text and "full-text" not in limitation_text):
                errors.append(
                    "Abstract-only degraded submissions must explicitly disclose missing usable full text in limitations."
                )
            if submission.overall_confidence.level == "high" or submission.overall_confidence.score > 0.6:
                errors.append("Abstract-only degraded submissions must lower overall confidence.")

        if errors:
            raise ValueError(" ; ".join(_merge_unique_text([], errors)))

    def _run_deterministic(
        self,
        *,
        workspace: ReactReviewedWorkspace,
        cycle_number: int,
        open_review_items: Sequence[ReviewItem],
    ) -> Tuple[AnswerSubmission, ReActTrajectory]:
        trajectory = ReActTrajectory(query=workspace.question)

        planned_queries = workspace.plan_queries(focus="revision" if cycle_number > 1 else "initial")
        plan_step = self._add_tool_step(
            trajectory=trajectory,
            thought="Build query plans directly from the fixed TaskSpec and resolved entities.",
            tool_calls=[
                ToolCallRecord(
                    tool_name="plan_queries",
                    tool_call_id=f"tc_{uuid.uuid4().hex[:8]}",
                    tool_args={"focus": "revision" if cycle_number > 1 else "initial"},
                    observation=_json_preview(planned_queries),
                    observation_data=planned_queries,
                )
            ],
        )

        search_plan_ids = [item["query_plan_id"] for item in planned_queries[:2]]
        search_results: List[Dict[str, Any]] = []
        search_tool_calls: List[ToolCallRecord] = []
        for query_plan_id in search_plan_ids:
            result = workspace.search_papers(query_plan_id=query_plan_id, reason="deterministic proposer")
            search_results.extend(result)
            search_tool_calls.append(
                ToolCallRecord(
                    tool_name="search_papers",
                    tool_call_id=f"tc_{uuid.uuid4().hex[:8]}",
                    tool_args={"query_plan_id": query_plan_id, "reason": "deterministic proposer"},
                    observation=_json_preview(result),
                    observation_data=result,
                )
            )
        search_step = self._add_tool_step(
            trajectory=trajectory,
            thought="Search the highest-value planned lanes before forming a submission.",
            tool_calls=search_tool_calls,
        )

        screened_payload = self._screen_candidate_papers(
            workspace=workspace,
            cycle_number=cycle_number,
            open_review_items=open_review_items,
            paper_ids=[item.get("paper_id") for item in search_results if item.get("paper_id")],
            max_candidates=3,
        )
        screen_step = self._add_tool_step(
            trajectory=trajectory,
            thought="Screen the retrieved papers and lock the most relevant candidates before acquisition.",
            tool_calls=[
                ToolCallRecord(
                    tool_name="screen_papers",
                    tool_call_id=f"tc_{uuid.uuid4().hex[:8]}",
                    tool_args={"paper_ids": [item.get("paper_id") for item in search_results if item.get("paper_id")], "max_candidates": 3},
                    observation=_json_preview(screened_payload),
                    observation_data=screened_payload,
                )
            ],
        )

        selected_paper_ids = list(screened_payload.get("locked_paper_ids") or [])
        if not selected_paper_ids:
            self._raise_execution_failure(
                workspace=workspace,
                cycle_number=cycle_number,
                stage="proposer_screening",
                message="Candidate screening did not lock any papers for acquisition.",
                details={"screened_paper_count": len(screened_payload.get("ranked_candidates") or [])},
                trajectory=trajectory,
            )
        acquired_payloads: List[Dict[str, Any]] = []
        acquire_tool_calls: List[ToolCallRecord] = []
        extracted_paper_ids: List[str] = []
        for paper_id in selected_paper_ids:
            payload = workspace.acquire_document(paper_id=str(paper_id))
            acquired_payloads.append(payload)
            acquire_tool_calls.append(
                ToolCallRecord(
                    tool_name="acquire_document",
                    tool_call_id=f"tc_{uuid.uuid4().hex[:8]}",
                    tool_args={"paper_id": paper_id},
                    observation=_json_preview(payload),
                    observation_data=payload,
                )
            )
            if str(payload.get("fulltext_status") or "").strip().lower() == "fulltext_indexed":
                extracted_paper_ids.append(str(paper_id))
            elif len(extracted_paper_ids) < 2 and _paper_has_useful_abstract_support(
                task_spec=workspace.task_spec,
                entity_pack=workspace.entity_pack,
                title=getattr(workspace.paper_candidates.get(str(paper_id)), "title", None),
                abstract=getattr(workspace.paper_candidates.get(str(paper_id)), "abstract", None),
            ):
                extracted_paper_ids.append(str(paper_id))
            if len(extracted_paper_ids) >= 2:
                break
        acquire_step = self._add_tool_step(
            trajectory=trajectory,
            thought="Acquire full text or abstract-backed section indices for the strongest papers.",
            tool_calls=acquire_tool_calls,
        )
        if not extracted_paper_ids:
            self._raise_execution_failure(
                workspace=workspace,
                cycle_number=cycle_number,
                stage="proposer_acquisition",
                message="Candidate acquisition did not yield any papers with usable full text or clearly relevant abstract support.",
                details={"selected_paper_ids": selected_paper_ids},
                trajectory=trajectory,
            )

        extracted_evidence: List[Dict[str, Any]] = []
        extract_tool_calls: List[ToolCallRecord] = []
        for paper_id in extracted_paper_ids:
            payload = workspace.extract_evidence(paper_id=str(paper_id), preferred_sections=True)
            extracted_evidence.extend(payload)
            extract_tool_calls.append(
                ToolCallRecord(
                    tool_name="extract_evidence",
                    tool_call_id=f"tc_{uuid.uuid4().hex[:8]}",
                    tool_args={"paper_id": paper_id, "preferred_sections": True},
                    observation=_json_preview(payload),
                    observation_data=payload,
                )
            )
        if not extract_tool_calls:
            self._raise_execution_failure(
                workspace=workspace,
                cycle_number=cycle_number,
                stage="proposer_evidence",
                message="No evidence extraction calls were possible after acquisition.",
                details={"selected_paper_ids": extracted_paper_ids},
                trajectory=trajectory,
            )
        extract_step = self._add_tool_step(
            trajectory=trajectory,
            thought="Extract evidence snippets and stable evidence references for answer assembly.",
            tool_calls=extract_tool_calls,
        )

        step_refs = [
            _step_ref(trajectory, plan_step),
            _step_ref(trajectory, search_step),
            _step_ref(trajectory, screen_step),
            _step_ref(trajectory, acquire_step),
            _step_ref(trajectory, extract_step),
        ]
        submission = self._build_submission_from_workspace(
            workspace=workspace,
            cycle_number=cycle_number,
            step_refs=step_refs,
            open_review_items=open_review_items,
        )
        conclude_step = self._add_tool_step(
            trajectory=trajectory,
            thought="Assemble the final sectioned submission and preserve reviewer issue links.",
            tool_calls=[
                ToolCallRecord(
                    tool_name="conclude",
                    tool_call_id=f"tc_{uuid.uuid4().hex[:8]}",
                    tool_args={"submission_id": submission.submission_id},
                    observation=json.dumps(submission.model_dump(exclude_none=True), ensure_ascii=False),
                    observation_data=submission.model_dump(exclude_none=True),
                )
            ],
        )
        if not submission.step_refs:
            submission = submission.model_copy(update={"step_refs": [_step_ref(trajectory, conclude_step)]})
        trajectory.finalize(json.dumps({"submission_id": submission.submission_id}, ensure_ascii=False))
        return submission, trajectory

    def _try_rebuild_submission_from_workspace(
        self,
        *,
        workspace: ReactReviewedWorkspace,
        cycle_number: int,
        open_review_items: Sequence[ReviewItem],
        trajectory: ReActTrajectory,
        run_state: _ProposerRunState,
    ) -> Optional[AnswerSubmission]:
        if not run_state.has_any_evidence():
            return None
        try:
            rebuilt = self._build_submission_from_workspace(
                workspace=workspace,
                cycle_number=cycle_number,
                step_refs=[_step_ref(trajectory, max(1, len(trajectory.steps) or 1))],
                open_review_items=open_review_items,
                run_state=run_state,
            )
            return self._validate_submission_payload(
                workspace=workspace,
                submission=rebuilt.model_dump(exclude_none=True),
                cycle_number=cycle_number,
                open_review_items=open_review_items,
                agent=None,
                trajectory=trajectory,
                run_state=run_state,
            )
        except Exception:
            return None

    def _build_submission_from_workspace(
        self,
        *,
        workspace: ReactReviewedWorkspace,
        cycle_number: int,
        step_refs: Sequence[SubmissionStepRef],
        open_review_items: Sequence[ReviewItem],
        run_state: Optional[_ProposerRunState] = None,
    ) -> AnswerSubmission:
        citations = self._build_submission_citations(workspace, run_state=run_state)
        allowed_paper_ids = {citation.paper_id for citation in citations}
        evidence_items = [
            item
            for item in list(workspace.evidence_items.values())
            if not allowed_paper_ids or item.paper_id in allowed_paper_ids
        ]
        evidence_items.sort(key=_evidence_item_priority_score, reverse=True)
        observations = [item for item in evidence_items if item.role in {"observation", "mechanism"}]
        limitations = [item for item in evidence_items if item.role == "limitation"]
        citation_ids = [item.citation_id for item in citations]
        issue_refs = [item.review_id for item in open_review_items]
        sections: List[SubmissionSection] = []
        for answer_section in workspace.task_spec.answer_sections:
            related_issues = [
                item.review_id
                for item in open_review_items
                if item.target_section_id == answer_section.section_id or item.anchor_kind in {"global", "missing_section"}
            ]
            content = self._render_section_content(
                answer_section_id=answer_section.section_id,
                observations=observations,
                limitations=limitations,
                citations=citations,
                workspace=workspace,
                related_issues=related_issues,
            )
            section_citation_ids = self._section_citation_ids(
                answer_section_id=answer_section.section_id,
                citations=citations,
                default_citation_ids=citation_ids,
            )
            confidence_score = 0.75 if section_citation_ids else 0.35
            sections.append(
                SubmissionSection(
                    section_id=answer_section.section_id,
                    title=answer_section.title,
                    content=content,
                    citation_ids=section_citation_ids,
                    step_refs=list(step_refs[-2:] if step_refs else []),
                    issue_refs=related_issues,
                    section_confidence=_confidence(
                        confidence_score,
                        "Deterministic proposer confidence reflects section citation coverage.",
                    ),
                )
            )
        submission_limitations = [
            "Reviewer-raised issues were carried forward into the submission issue_refs list."
            if open_review_items
            else "No blocking review issues were open when this submission was assembled."
        ]
        if not citations:
            submission_limitations.append(
                "No document-level citations were available, so the submission remains conservative."
            )
        if limitations:
            submission_limitations.append(_compact_text(limitations[0].snippet))
        overall_score = 0.8 if citations and observations else 0.4 if citations else 0.25
        return AnswerSubmission(
            submission_id=f"submission_cycle_{cycle_number}",
            question=workspace.question,
            version=cycle_number,
            sections=sections,
            citations=citations,
            limitations=submission_limitations,
            overall_confidence=_confidence(
                overall_score,
                "Deterministic proposer confidence is based on citation coverage and extracted evidence breadth.",
            ),
            trajectory_id=step_refs[0].trajectory_id if step_refs else f"traj_cycle_{cycle_number}",
            step_refs=list(step_refs),
            issue_refs=issue_refs,
        )

    def _build_submission_citations(
        self,
        workspace: ReactReviewedWorkspace,
        *,
        run_state: Optional[_ProposerRunState] = None,
    ) -> List[SubmissionCitation]:
        scored_records: List[Tuple[float, PaperRecord, List[EvidenceItem]]] = []
        if run_state is not None and run_state.acquired_paper_ids:
            paper_ids = [
                paper_id
                for paper_id in list(run_state.locked_candidate_paper_ids or [])
                if paper_id in workspace.paper_records
            ]
            paper_ids.extend(
                paper_id
                for paper_id in sorted(run_state.acquired_paper_ids)
                if paper_id in workspace.paper_records and paper_id not in set(paper_ids)
            )
        else:
            paper_ids = [paper_record.paper_id for paper_record in list(workspace.paper_records.values())]
        for paper_id in paper_ids:
            paper_record = workspace.paper_records.get(paper_id)
            if paper_record is None:
                continue
            paper_evidence = [
                item for item in workspace.evidence_items.values()
                if item.paper_id == paper_record.paper_id
                and (run_state is None or item.evidence_id in run_state.evidence_ids)
            ]
            paper_evidence.sort(key=_evidence_item_priority_score, reverse=True)
            has_useful_abstract = _paper_has_useful_abstract_support(
                task_spec=workspace.task_spec,
                entity_pack=workspace.entity_pack,
                title=paper_record.title,
                abstract=paper_record.abstract,
            )
            if not paper_evidence and not has_useful_abstract:
                continue
            paper_score = (max((_evidence_item_priority_score(item) for item in paper_evidence), default=0.0)) + (
                2.0 if str(paper_record.fulltext_status or "").strip().lower() == "fulltext_indexed" else 0.0
            ) + (
                1.0 if has_useful_abstract else 0.0
            )
            scored_records.append((paper_score, paper_record, paper_evidence))
        scored_records.sort(key=lambda item: item[0], reverse=True)
        citations: List[SubmissionCitation] = []
        for _score, paper_record, paper_evidence in scored_records[:4]:
            useful_evidence = [item for item in paper_evidence if _evidence_item_priority_score(item) > 1.5] or paper_evidence[:2]
            section_ids = _merge_unique_text([], [item.section_id for item in useful_evidence])
            evidence_ids = _merge_unique_text([], [item.evidence_id for item in useful_evidence])
            if (
                not section_ids
                and not evidence_ids
                and _paper_has_useful_abstract_support(
                    task_spec=workspace.task_spec,
                    entity_pack=workspace.entity_pack,
                    title=paper_record.title,
                    abstract=paper_record.abstract,
                )
            ):
                section_ids = ["sec_abstract"]
            citations.append(
                SubmissionCitation(
                    citation_id=f"CIT-{len(citations) + 1}",
                    paper_id=paper_record.paper_id,
                    doi=paper_record.doi,
                    title=paper_record.title,
                    year=paper_record.year,
                    venue=paper_record.venue,
                    section_ids=section_ids,
                    evidence_ids=evidence_ids,
                )
            )
        return citations

    def _section_citation_ids(
        self,
        *,
        answer_section_id: str,
        citations: Sequence[SubmissionCitation],
        default_citation_ids: Sequence[str],
    ) -> List[str]:
        if answer_section_id in {"caveats", "causal_limitations", "open_questions"}:
            limitation_ids = [item.citation_id for item in citations if item.section_ids]
            return limitation_ids[:2]
        return list(default_citation_ids[:2])

    def _render_section_content(
        self,
        *,
        answer_section_id: str,
        observations: Sequence[EvidenceItem],
        limitations: Sequence[EvidenceItem],
        citations: Sequence[SubmissionCitation],
        workspace: ReactReviewedWorkspace,
        related_issues: Sequence[str],
    ) -> str:
        ranked_observations = sorted(list(observations or []), key=_evidence_item_priority_score, reverse=True)
        useful_observations = [item for item in ranked_observations if _evidence_item_priority_score(item) > 1.5]
        ranked_limitations = sorted(list(limitations or []), key=_evidence_item_priority_score, reverse=True)
        if answer_section_id in {"representative_papers"}:
            if not citations:
                return "Representative papers could not be confirmed from the current retrieval set."
            fragments = [
                f"{citation.title} ({citation.year or 'n.d.'})"
                for citation in citations[:3]
            ]
            return "Representative papers in the current run include " + "; ".join(fragments) + "."
        if answer_section_id in {"caveats", "causal_limitations", "open_questions"}:
            if ranked_limitations:
                return _compact_text(ranked_limitations[0].snippet)
            if related_issues:
                return "Open review items remain attached to this section: " + ", ".join(related_issues) + "."
            return "Current evidence remains incomplete, so this section stays conservative."
        if answer_section_id in {"conditions"}:
            conditions: List[str] = []
            condition_items = [
                item for item in useful_observations
                if item.role == "condition" or bool(item.conditions)
            ] or useful_observations[:3]
            for evidence_item in condition_items[:3]:
                for axis_name, axis_value in evidence_item.conditions.items():
                    conditions.append(f"{axis_name}: {axis_value}")
            if conditions:
                return "Material conditions mentioned in the current evidence include " + "; ".join(conditions[:4]) + "."
        if useful_observations:
            fragments = [
                _compact_text(item.snippet)
                for item in useful_observations[:2]
                if _compact_text(item.snippet)
            ]
            if fragments:
                return " ".join(fragments)
        if citations:
            return (
                f"The retrieved literature set for '{workspace.question}' contains usable citations, "
                "but the extracted evidence is still thin for a stronger section-level claim."
            )
        return "Evidence remains insufficient to support a stronger section-level answer."

    def _build_system_prompt(self, *, conclude_call_contract: Dict[str, Any]) -> str:
        return build_proposer_system_prompt(conclude_contract=conclude_call_contract)

    def _build_user_prompt(
        self,
        *,
        workspace: ReactReviewedWorkspace,
        cycle_number: int,
        open_review_items: Sequence[ReviewItem],
        conclude_call_contract: Optional[Dict[str, Any]] = None,
    ) -> str:
        if conclude_call_contract is None:
            conclude_call_contract = self._build_submission_prompt_contract(
                workspace=workspace,
                cycle_number=cycle_number,
                open_review_items=open_review_items,
            )
        return build_proposer_user_prompt(
            cycle_number=cycle_number,
            question=workspace.question,
            context=workspace.context,
            task_spec=workspace.task_spec.model_dump(exclude_none=True),
            entity_pack=workspace.entity_pack.model_dump(exclude_none=True),
            prior_submission=(
                workspace.current_submission.model_dump(exclude_none=True)
                if workspace.current_submission is not None
                else None
            ),
            prior_proposer_trajectory=(
                workspace.current_proposer_trajectory.to_dict()
                if workspace.current_proposer_trajectory is not None
                else None
            ),
            open_review_items=[item.model_dump(exclude_none=True) for item in open_review_items],
            conclude_call_contract=conclude_call_contract,
            retrieval_policy={
                "fallback_mode": self.fallback_mode,
                "repair_attempts": self.repair_attempts,
                "evidence_policy": self.evidence_policy,
                "phase_order": [
                    "plan_queries",
                    "search_papers",
                    "screen_papers",
                    "acquire_document",
                    "read_sections_or_extract_evidence",
                    "conclude",
                ],
                "selection_rubric": [
                    "entity_and_condition_alignment",
                    "question_type_and_lane_fit",
                    "fulltext_availability_signal",
                    "evidence_density_after_extraction",
                    "coverage_diversity",
                ],
            },
        )

    def _thought_instruction(self) -> str:
        return build_proposer_thought_prompt()

    def _action_instruction(self, tool_names: Sequence[str], *, conclude_call_contract: Dict[str, Any]) -> str:
        retrieval_tools = [name for name in tool_names if name not in {"analyze_submission_gap", "conclude"}]
        return build_proposer_action_prompt(
            tool_names=tool_names,
            retrieval_tools=retrieval_tools,
            conclude_contract=conclude_call_contract,
        )

    def _add_tool_step(
        self,
        *,
        trajectory: ReActTrajectory,
        thought: str,
        tool_calls: Sequence[ToolCallRecord],
    ) -> int:
        step_number = len(trajectory.steps) + 1
        observation = "\n\n".join(call.observation for call in tool_calls) or "(no observation)"
        trajectory.add_step(
            ReActStep(
                step_number=step_number,
                thought=thought,
                action=tool_calls[0].tool_name if len(tool_calls) == 1 else "multi_tool",
                action_input={"tool_calls": [{"tool_name": item.tool_name, "tool_args": item.tool_args} for item in tool_calls]},
                observation=observation,
                tool_calls=list(tool_calls),
                tool_call_id=tool_calls[0].tool_call_id if len(tool_calls) == 1 else None,
                observation_data=tool_calls[0].observation_data if len(tool_calls) == 1 else None,
            )
        )
        return step_number


class ReactReviewedReviewerAgent:
    def __init__(
        self,
        *,
        reviewer_role: ReviewerRole,
        model_config: Optional[Dict[str, Any]],
        max_steps: int,
        max_items: int,
        max_retrieval_actions: int,
        llm_timeout_seconds: float,
    ) -> None:
        self.reviewer_role = reviewer_role
        self.model_config = dict(model_config or {})
        self.max_steps = max(1, int(max_steps))
        self.max_items = max(1, int(max_items))
        self.max_retrieval_actions = max(0, int(max_retrieval_actions))
        self.llm_timeout_seconds = float(llm_timeout_seconds)

    def run(
        self,
        *,
        workspace: ReactReviewedWorkspace,
        submission: AnswerSubmission,
        proposer_trajectory: ReActTrajectory,
        cycle_number: int,
        session: ReviewerSession,
    ) -> Tuple[List[ReviewItem], Optional[ReActTrajectory], ReviewerRunStatus]:
        stage_snapshot = workspace.snapshot_mutable_state()
        try:
            agent_output = self._run_with_llm(
                workspace=workspace,
                submission=submission,
                proposer_trajectory=proposer_trajectory,
                cycle_number=cycle_number,
                session=session,
            )
        except ReactReviewedStructuredOutputError as exc:
            workspace.restore_mutable_state(stage_snapshot, write_snapshot=True)
            return [], exc.trajectory, ReviewerRunStatus(
                reviewer_role=self.reviewer_role,
                status="invalid_json",
                message=str(exc),
                cycle_number=cycle_number,
                retrieval_actions_used=session.budget_state.actions_used,
                retrieval_budget_limit=session.budget_state.budget_limit,
                budget_blocked_calls=session.budget_state.blocked_calls,
            )
        except ReactReviewedReviewerExecutionError:
            workspace.restore_mutable_state(stage_snapshot, write_snapshot=True)
            raise
        items, trajectory, completion_status = agent_output
        return items, trajectory, ReviewerRunStatus(
            reviewer_role=self.reviewer_role,
            status=completion_status,
            message="completed" if completion_status == "completed" else "salvaged reviewer output",
            cycle_number=cycle_number,
            retrieval_actions_used=session.budget_state.actions_used,
            retrieval_budget_limit=session.budget_state.budget_limit,
            budget_blocked_calls=session.budget_state.blocked_calls,
        )

    def _run_with_llm(
        self,
        *,
        workspace: ReactReviewedWorkspace,
        submission: AnswerSubmission,
        proposer_trajectory: ReActTrajectory,
        cycle_number: int,
        session: ReviewerSession,
    ) -> Tuple[List[ReviewItem], Optional[ReActTrajectory], str]:
        StructuredTool = _lazy_structured_tool_import()
        provider, model_name, has_api_key = describe_chat_model_config(self.model_config)
        if StructuredTool is None or not has_api_key or not provider or not model_name:
            error = ReactReviewedReviewerExecutionError(
                stage="reviewer_startup",
                cycle_number=cycle_number,
                reviewer_role=self.reviewer_role,
                message="Reviewer requires an enabled LLM configuration and StructuredTool support.",
                details={
                    "structured_tool_available": StructuredTool is not None,
                    "provider": provider,
                    "model": model_name,
                    "has_api_key": has_api_key,
                },
            )
            _store_reviewer_execution_failure(
                artifact_store=session.artifact_store,
                prefix=f"{self.reviewer_role}_cycle_{cycle_number}",
                error=error,
            )
            raise error

        allowed_tool_names = REVIEWER_TOOL_NAMES[self.reviewer_role]
        agent_holder: Dict[str, Any] = {}

        def _budget_safe(call: Callable[[], Any]) -> ToolResult:
            try:
                payload = call()
            except ReviewerBudgetBlocked as exc:
                return ToolResult(observation=_json_preview(exc.payload), data=exc.payload)
            return ToolResult(observation=_json_preview(payload), data=payload)

        def inspect_submission_anchor(
            section_id: Optional[str] = None,
            step_number: Optional[int] = None,
            review_id: Optional[str] = None,
        ) -> ToolResult:
            """Inspect submission, trajectory, or prior review context by stable anchor."""
            payload = workspace.inspect_submission_anchor(
                section_id=section_id,
                step_number=step_number,
                review_id=review_id,
            )
            return ToolResult(observation=_json_preview(payload), data=payload)

        def inspect_entity_cache(
            name: Optional[str] = None,
            entity_type: Optional[str] = None,
            limit: int = 10,
        ) -> ToolResult:
            """Inspect resolved entity cache entries available to the current reviewer."""
            payload = workspace.inspect_entity_cache(name=name, entity_type=entity_type, limit=limit)
            return ToolResult(observation=_json_preview(payload), data=payload)

        def read_sections(
            paper_id: str,
            section_ids: Optional[List[str]] = None,
            preferred_sections: bool = False,
        ) -> ToolResult:
            """Read indexed sections from an acquired paper within reviewer budget rules."""
            return _budget_safe(
                lambda: workspace.read_sections(
                    paper_id=paper_id,
                    section_ids=section_ids,
                    preferred_sections=preferred_sections,
                    artifact_store=session.artifact_store,
                    session=session,
                    charge_budget=True,
                    requested_via="read_sections",
                    write_snapshot=False,
                )
            )

        def search_papers(
            query_plan_id: Optional[str] = None,
            query_text: Optional[str] = None,
            lane: str = "contrarian",
            reason: str = "",
        ) -> ToolResult:
            """Search external literature providers within reviewer role permissions."""
            return _budget_safe(
                lambda: workspace.search_papers(
                    query_plan_id=query_plan_id,
                    query_text=query_text,
                    lane=lane,
                    reason=reason,
                    artifact_store=session.artifact_store,
                    session=session,
                    charge_budget=True,
                    requested_via="search_papers",
                    write_snapshot=False,
                )
            )

        def acquire_document(paper_id: str) -> ToolResult:
            """Acquire a paper and section index within reviewer budget rules."""
            return _budget_safe(
                lambda: workspace.acquire_document(
                    paper_id=paper_id,
                    artifact_store=session.artifact_store,
                    session=session,
                    charge_budget=True,
                    requested_via="acquire_document",
                    write_snapshot=False,
                )
            )

        def fetch_citation_context(
            paper_id: Optional[str] = None,
            section_id: Optional[str] = None,
            evidence_id: Optional[str] = None,
            citation_id: Optional[str] = None,
        ) -> ToolResult:
            """Fetch citation or evidence context within reviewer budget rules."""
            return _budget_safe(
                lambda: workspace.fetch_citation_context(
                    paper_id=paper_id,
                    section_id=section_id,
                    evidence_id=evidence_id,
                    citation_id=citation_id,
                    artifact_store=session.artifact_store,
                    session=session,
                    requested_via="fetch_citation_context",
                    write_snapshot=False,
                )
            )

        def extract_evidence(
            paper_id: str,
            section_ids: Optional[List[str]] = None,
            preferred_sections: bool = False,
        ) -> ToolResult:
            """Extract evidence items from selected paper sections within reviewer budget rules."""
            return _budget_safe(
                lambda: workspace.extract_evidence(
                    paper_id=paper_id,
                    section_ids=section_ids,
                    preferred_sections=preferred_sections,
                    artifact_store=session.artifact_store,
                    session=session,
                    charge_budget=True,
                    requested_via="extract_evidence",
                    write_snapshot=False,
                )
            )

        def conclude(review: Any) -> ToolResult:
            """Validate and submit the reviewer payload for this cycle."""
            try:
                review = _tool_plain_payload(review)
                items = self._validate_review_payload(
                    review=review,
                    submission=submission,
                    proposer_trajectory=proposer_trajectory,
                    agent=agent_holder.get("agent"),
                )
            except Exception as exc:
                return ToolResult(
                    observation=f"Invalid review payload: {exc}",
                    data={"__conclude_valid__": False, "error": str(exc)},
                )
            return ToolResult(
                observation=json.dumps({"review_items": [item.model_dump(exclude_none=True) for item in items]}, ensure_ascii=False),
                data={
                    "__conclude_valid__": True,
                    "review_items": [item.model_dump(exclude_none=True) for item in items],
                },
            )

        tool_builders: Dict[str, Callable[..., ToolResult]] = {
            "inspect_entity_cache": inspect_entity_cache,
            "inspect_submission_anchor": inspect_submission_anchor,
            "read_sections": read_sections,
            "search_papers": search_papers,
            "acquire_document": acquire_document,
            "fetch_citation_context": fetch_citation_context,
            "extract_evidence": extract_evidence,
            "conclude": conclude,
        }
        tool_args_schemas: Dict[str, Any] = {
            "inspect_entity_cache": tool_schemas.InspectEntityCacheToolInput,
            "inspect_submission_anchor": tool_schemas.InspectSubmissionAnchorToolInput,
            "read_sections": tool_schemas.SectionAccessToolInput,
            "search_papers": tool_schemas.ReviewerSearchPapersToolInput,
            "acquire_document": tool_schemas.AcquireDocumentToolInput,
            "fetch_citation_context": tool_schemas.FetchCitationContextToolInput,
            "extract_evidence": tool_schemas.SectionAccessToolInput,
            "conclude": tool_schemas.ReviewerConcludeToolInput,
        }
        tools = [
            StructuredTool.from_function(tool_builders[name], name=name, args_schema=tool_args_schemas[name])
            for name in allowed_tool_names
        ]
        conclude_call_contract = self._build_review_prompt_contract(
            submission=submission,
            proposer_trajectory=proposer_trajectory,
        )
        system_prompt = self._build_system_prompt(conclude_call_contract=conclude_call_contract)
        try:
            agent = ReActAgent(
                agent_id=f"qa_reviewer_{self.reviewer_role}",
                name=f"ReviewerAgent[{self.reviewer_role}]",
                model_config=self.model_config,
                system_prompt=system_prompt,
                max_react_steps=self.max_steps,
                verbose=False,
                tools=tools,
                thought_phase_instruction=build_reviewer_thought_prompt(),
                action_phase_instruction=build_reviewer_action_prompt(
                    tool_names=allowed_tool_names,
                    retrieval_budget=session.budget_state.budget_limit,
                    conclude_contract=conclude_call_contract,
                ),
                search_tool_names=[name for name in allowed_tool_names if name != "conclude"],
                analysis_tool_names=["conclude"],
                conclude_argument_name="review",
                conclude_output_kind="review_items",
            )
            agent_holder["agent"] = agent
            response, trajectory = agent.generate_response_with_react(
                query=self._build_user_prompt(
                    submission=submission,
                    proposer_trajectory=proposer_trajectory,
                    cycle_number=cycle_number,
                    conclude_call_contract=conclude_call_contract,
                ),
                system_prompt_override=system_prompt,
                max_steps_override=self.max_steps,
                llm_timeout_seconds=self.llm_timeout_seconds,
            )
        except Exception as exc:
            response_content = getattr(exc, "response_content", None)
            partial_trajectory = getattr(exc, "trajectory", None) or getattr(agent_holder.get("agent"), "current_trajectory", None)
            response_shim = SimpleNamespace(
                content="",
                structured_output=getattr(exc, "structured_output", None),
                response_content=response_content,
            )
            salvaged_items = self._salvage_review_payload(
                response=response_shim,
                trajectory=partial_trajectory,
                proposer_trajectory=proposer_trajectory,
                submission=submission,
                max_items=self.max_items,
            )
            if salvaged_items:
                logger.warning(
                    "react_reviewed_reviewer_payload_salvaged role=%s cycle=%s source=forced_conclude_exception count=%s",
                    self.reviewer_role,
                    cycle_number,
                    len(salvaged_items),
                )
                return salvaged_items, partial_trajectory, "salvaged"
            error = ReactReviewedStructuredOutputError(
                stage="reviewer",
                cycle_number=cycle_number,
                reviewer_role=self.reviewer_role,
                message=f"reviewer structured conclude failed after LLM start: {exc}",
                response_content=getattr(exc, "response_content", None),
                structured_output=getattr(exc, "structured_output", None),
            )
            _store_invalid_llm_output(
                artifact_store=session.artifact_store,
                prefix=f"{self.reviewer_role}_cycle_{cycle_number}",
                error=error,
            )
            logger.warning(
                "react_reviewed_reviewer_llm_failed role=%s cycle=%s error=%s",
                self.reviewer_role,
                cycle_number,
                exc,
            )
            raise error
        try:
            raw_items = self._parse_review_items_response(response=response)
            items = [ReviewItem.model_validate(item) for item in list(raw_items or [])]
            return items[: self.max_items], trajectory, "completed"
        except Exception as exc:
            salvaged_items = self._salvage_review_payload(
                response=response,
                trajectory=trajectory,
                proposer_trajectory=proposer_trajectory,
                submission=submission,
                max_items=self.max_items,
            )
            if salvaged_items:
                logger.warning(
                    "react_reviewed_reviewer_payload_salvaged role=%s cycle=%s count=%s",
                    self.reviewer_role,
                    cycle_number,
                    len(salvaged_items),
                )
                return salvaged_items, trajectory, "salvaged"
            error = ReactReviewedStructuredOutputError(
                stage="reviewer",
                cycle_number=cycle_number,
                reviewer_role=self.reviewer_role,
                message=f"invalid reviewer structured output: {exc}",
                response_content=getattr(response, "response_content", getattr(response, "content", None)),
                structured_output=getattr(response, "structured_output", None),
                trajectory=trajectory,
            )
            _store_invalid_llm_output(
                artifact_store=session.artifact_store,
                prefix=f"{self.reviewer_role}_cycle_{cycle_number}",
                error=error,
            )
            logger.warning(
                "react_reviewed_reviewer_llm_failed role=%s cycle=%s error=%s",
                self.reviewer_role,
                cycle_number,
                exc,
            )
            raise error

    def _parse_review_items_response(self, *, response: Any) -> List[Dict[str, Any]]:
        payload = _extract_stage_payload(response=response, expected_kind="review_items")
        if not isinstance(payload, list):
            raise ValueError("review_items payload must be a JSON array.")
        return [item for item in payload if isinstance(item, dict)]

    def _validate_review_payload(
        self,
        *,
        review: Any,
        submission: AnswerSubmission,
        proposer_trajectory: ReActTrajectory,
        agent: Optional[ReActAgent],
    ) -> List[ReviewItem]:
        payload = copy.deepcopy(review if isinstance(review, dict) else json.loads(json.dumps(review)))
        raw_items = payload.get("review_items") if isinstance(payload, dict) else payload
        items: List[ReviewItem] = []
        default_section_id = submission.sections[0].section_id if submission.sections else None
        for index, raw_item in enumerate(list(raw_items or [])[: self.max_items], start=1):
            if not isinstance(raw_item, dict):
                continue
            if not _compact_text(raw_item.get("critique")) or not _compact_text(raw_item.get("required_action")):
                continue
            flaw_type = _compact_text(raw_item.get("flaw_type"))
            if not flaw_type or flaw_type == "needs_manual_review":
                continue
            raw_item.setdefault("review_id", f"{self.reviewer_role}_{index}")
            raw_item.setdefault("reviewer_role", self.reviewer_role)
            raw_item.setdefault("severity", "warning")
            raw_item.setdefault("anchor_kind", "global")
            if raw_item.get("anchor_kind") == "step_section":
                raw_item["target_trajectory_id"] = raw_item.get("target_trajectory_id") or proposer_trajectory.trajectory_id
                raw_item["target_step_number"] = raw_item.get("target_step_number") or 1
                raw_item["target_section_id"] = raw_item.get("target_section_id") or default_section_id
            elif raw_item.get("anchor_kind") in {"section_only", "missing_section"}:
                raw_item["target_section_id"] = raw_item.get("target_section_id") or default_section_id
            items.append(ReviewItem.model_validate(raw_item))
        return items

    def _salvage_review_payload(
        self,
        *,
        response: Any,
        trajectory: Optional[ReActTrajectory],
        proposer_trajectory: ReActTrajectory,
        submission: AnswerSubmission,
        max_items: int,
    ) -> List[ReviewItem]:
        default_section_id = submission.sections[0].section_id if submission.sections else None
        candidate_payloads: List[Any] = []
        candidate_payloads.extend(_extract_json_candidates(getattr(response, "structured_output", None)))
        candidate_payloads.extend(_extract_json_candidates(getattr(response, "content", None)))
        response_content = getattr(response, "response_content", None)
        if isinstance(response_content, dict):
            candidate_payloads.extend(_extract_json_candidates(response_content))
            for value in response_content.values():
                candidate_payloads.extend(_extract_json_candidates(value))
                if isinstance(value, dict):
                    candidate_payloads.extend(_extract_json_candidates(value.get("content")))
                    candidate_payloads.extend(_extract_json_candidates(value.get("additional_kwargs")))
        for step in reversed(list(getattr(trajectory, "steps", []) or [])):
            candidate_payloads.extend(_extract_json_candidates(getattr(step, "observation_data", None)))
            candidate_payloads.extend(_extract_json_candidates(getattr(step, "observation", None)))
            for call in reversed(list(getattr(step, "tool_calls", []) or [])):
                if getattr(call, "tool_name", "") != "conclude":
                    continue
                candidate_payloads.extend(_extract_json_candidates(getattr(call, "observation_data", None)))
                candidate_payloads.extend(_extract_json_candidates(getattr(call, "observation", None)))
        seen_payloads = set()
        for payload in candidate_payloads:
            payload_key = _compact_text(json.dumps(payload, ensure_ascii=False, default=str) if isinstance(payload, (dict, list)) else str(payload))
            if payload_key in seen_payloads:
                continue
            seen_payloads.add(payload_key)
            items = self._coerce_salvaged_review_items(
                payload=payload,
                proposer_trajectory=proposer_trajectory,
                default_section_id=default_section_id,
                max_items=max_items,
            )
            if items:
                return items
        return []

    def _coerce_salvaged_review_items(
        self,
        *,
        payload: Any,
        proposer_trajectory: ReActTrajectory,
        default_section_id: Optional[str],
        max_items: int,
    ) -> List[ReviewItem]:
        current = payload
        if isinstance(current, dict) and current.get("kind") == "review_items":
            current = current.get("payload")
        elif isinstance(current, dict) and isinstance(current.get("review"), dict):
            current = current["review"]
        if isinstance(current, dict) and isinstance(current.get("review_items"), list):
            current = current.get("review_items")
        if not isinstance(current, list):
            return []
        items: List[ReviewItem] = []
        severity_map = {"blocker": "blocking", "critical": "blocking", "warn": "warning"}
        for index, raw_item in enumerate(current[: max_items], start=1):
            if not isinstance(raw_item, dict):
                return []
            normalized = copy.deepcopy(raw_item)
            location = normalized.pop("location", None)
            if isinstance(location, dict):
                if location.get("section_id") and not normalized.get("target_section_id"):
                    normalized["target_section_id"] = location.get("section_id")
                if location.get("step_number") and not normalized.get("target_step_number"):
                    normalized["target_step_number"] = location.get("step_number")
            if "review_item_id" in normalized and "review_id" not in normalized:
                normalized["review_id"] = normalized.pop("review_item_id")
            if "issue" in normalized and "critique" not in normalized:
                normalized["critique"] = normalized.pop("issue")
            if "comment" in normalized and "critique" not in normalized:
                normalized["critique"] = normalized.pop("comment")
            if "notes" in normalized and "critique" not in normalized:
                notes = normalized.pop("notes")
                if isinstance(notes, list):
                    notes_text = "; ".join(str(item).strip() for item in notes if str(item).strip())
                else:
                    notes_text = str(notes).strip()
                if notes_text:
                    normalized["critique"] = notes_text
            if "recommendation" in normalized and "required_action" not in normalized:
                normalized["required_action"] = normalized.pop("recommendation")
            if "required_fix" in normalized and "required_action" not in normalized:
                normalized["required_action"] = normalized.pop("required_fix")
            if "category" in normalized and "flaw_type" not in normalized:
                normalized["flaw_type"] = normalized.pop("category")
            severity = str(normalized.get("severity") or "").strip().lower()
            if severity in severity_map:
                normalized["severity"] = severity_map[severity]
            if not _compact_text(normalized.get("critique")) or not _compact_text(normalized.get("required_action")):
                continue
            flaw_type = _compact_text(normalized.get("flaw_type"))
            if not flaw_type or flaw_type == "needs_manual_review":
                continue
            normalized.setdefault("review_id", f"{self.reviewer_role}_{index}")
            normalized.setdefault("reviewer_role", self.reviewer_role)
            normalized.setdefault("severity", "warning")
            if normalized.get("target_section_id"):
                normalized.setdefault("anchor_kind", "section_only")
            else:
                normalized.setdefault("anchor_kind", "global")
            if normalized.get("anchor_kind") == "step_section":
                normalized["target_trajectory_id"] = normalized.get("target_trajectory_id") or proposer_trajectory.trajectory_id
                normalized["target_step_number"] = normalized.get("target_step_number") or 1
                normalized["target_section_id"] = normalized.get("target_section_id") or default_section_id
            elif normalized.get("anchor_kind") in {"section_only", "missing_section"}:
                normalized["target_section_id"] = normalized.get("target_section_id") or default_section_id
            try:
                items.append(ReviewItem.model_validate(normalized))
            except Exception:
                return []
        return items

    def _build_review_prompt_contract(
        self,
        *,
        submission: AnswerSubmission,
        proposer_trajectory: ReActTrajectory,
    ) -> Dict[str, Any]:
        return build_review_prompt_contract(
            reviewer_role=self.reviewer_role,
            target_section_id=submission.sections[0].section_id if submission.sections else None,
            target_trajectory_id=proposer_trajectory.trajectory_id,
        )

    def _build_system_prompt(self, *, conclude_call_contract: Dict[str, Any]) -> str:
        role_notes = {
            "search_coverage": "Check for missing search directions, controls, and missing sections.",
            "evidence_trace": "Check whether citations, evidence refs, and step refs are traceable and real.",
            "reasoning_consistency": "Check for reasoning jumps, scope drift, and inconsistent section claims.",
            "counterevidence": "Use limited new retrieval only to search for counterevidence or boundary conditions.",
        }
        return build_reviewer_system_prompt(
            reviewer_role=self.reviewer_role,
            role_note=role_notes[self.reviewer_role],
            max_retrieval_actions=self.max_retrieval_actions,
            conclude_contract=conclude_call_contract,
        )

    def _build_user_prompt(
        self,
        *,
        submission: AnswerSubmission,
        proposer_trajectory: ReActTrajectory,
        cycle_number: int,
        conclude_call_contract: Optional[Dict[str, Any]] = None,
    ) -> str:
        if conclude_call_contract is None:
            conclude_call_contract = self._build_review_prompt_contract(
                submission=submission,
                proposer_trajectory=proposer_trajectory,
            )
        return build_reviewer_user_prompt(
            cycle_number=cycle_number,
            reviewer_role=self.reviewer_role,
            submission=submission.model_dump(exclude_none=True),
            proposer_trajectory=proposer_trajectory.to_dict(),
            conclude_call_contract=conclude_call_contract,
        )


class SubmissionSynthesizerAgent:
    def run(
        self,
        *,
        task_spec: TaskSpec,
        submission: AnswerSubmission,
        review_items: Sequence[ReviewItem],
        review_completion_status: ReviewCompletionStatus,
        acceptance_decision: AcceptanceDecision,
        retrieval_diagnostics_summary: str,
        execution_warnings: Sequence[str],
    ) -> QAResult:
        if not acceptance_decision.accepted:
            return self._build_rejected_result(
                task_spec=task_spec,
                submission=submission,
                review_items=review_items,
                review_completion_status=review_completion_status,
                acceptance_decision=acceptance_decision,
                retrieval_diagnostics_summary=retrieval_diagnostics_summary,
                execution_warnings=execution_warnings,
            )
        return self._build_accepted_result(
            task_spec=task_spec,
            submission=submission,
            review_items=review_items,
            review_completion_status=review_completion_status,
            retrieval_diagnostics_summary=retrieval_diagnostics_summary,
            execution_warnings=execution_warnings,
        )

    def _build_accepted_result(
        self,
        *,
        task_spec: TaskSpec,
        submission: AnswerSubmission,
        review_items: Sequence[ReviewItem],
        review_completion_status: ReviewCompletionStatus,
        retrieval_diagnostics_summary: str,
        execution_warnings: Sequence[str],
    ) -> QAResult:
        section_outputs: List[AnswerSectionOutput] = []
        section_confidence: List[SectionConfidenceRecord] = []
        section_lookup = {section.section_id: section for section in submission.sections}
        for answer_section in task_spec.answer_sections:
            section = section_lookup.get(answer_section.section_id)
            if section is None:
                continue
            section_conf = _result_confidence(
                section.section_confidence.score,
                section.section_confidence.rationale,
            )
            section_outputs.append(
                AnswerSectionOutput(
                    section_id=section.section_id,
                    title=section.title or answer_section.title,
                    content=_compact_text(section.content) or "Evidence remains insufficient for a stronger section-level answer.",
                    citation_ids=list(section.citation_ids),
                    section_confidence=section_conf,
                )
            )
            section_confidence.append(
                SectionConfidenceRecord(
                    section_id=section.section_id,
                    title=section.title or answer_section.title,
                    confidence=section_conf,
                )
            )
        unresolved_blocking = [item for item in review_items if item.status == "open" and item.severity == "blocking"]
        limitations_parts = list(submission.limitations)
        if unresolved_blocking:
            limitations_parts.append(
                "Unresolved blocking review items remain: "
                + "; ".join(item.required_action for item in unresolved_blocking[:3])
            )
        if review_completion_status == "incomplete":
            limitations_parts.append("Reviewer completion was incomplete, so the answer remains non-accepted.")
        limitations_summary = " ".join(part for part in limitations_parts if _compact_text(part)).strip()
        overall_score = submission.overall_confidence.score
        if unresolved_blocking:
            overall_score = min(overall_score, 0.45)
        if review_completion_status == "incomplete":
            overall_score = min(overall_score, 0.35)
        return QAResult(
            question=submission.question,
            language="en",
            workflow_mode="react_reviewed",
            acceptance_status="accepted",
            final_answer=_assemble_final_answer(section_outputs),
            sections=section_outputs,
            citations=_submission_to_citation_records(submission),
            claim_trace=[],
            submission_trace=_build_submission_trace(submission),
            review_completion_status=review_completion_status,
            overall_confidence=_result_confidence(
                overall_score,
                "Overall confidence is derived from the accepted submission and outstanding review issues.",
            ),
            section_confidence=section_confidence,
            insufficient_evidence=bool(unresolved_blocking) or not submission.citations,
            limitations_summary=limitations_summary,
            retrieval_diagnostics_summary=str(retrieval_diagnostics_summary or "").strip(),
            execution_warnings=list(_merge_unique_text([], execution_warnings)),
            artifact_paths={},
            time_elapsed=0.0,
        )

    def _build_rejected_result(
        self,
        *,
        task_spec: TaskSpec,
        submission: AnswerSubmission,
        review_items: Sequence[ReviewItem],
        review_completion_status: ReviewCompletionStatus,
        acceptance_decision: AcceptanceDecision,
        retrieval_diagnostics_summary: str,
        execution_warnings: Sequence[str],
    ) -> QAResult:
        reason_lines = list(acceptance_decision.blocker_messages)
        if not reason_lines:
            reason_lines = ["Submission was rejected because the current answer packet is not acceptance-ready."]
        section_outputs: List[AnswerSectionOutput] = []
        section_confidence: List[SectionConfidenceRecord] = []
        rejection_text = "Submission rejected: " + " ".join(reason_lines[:2])
        for answer_section in task_spec.answer_sections:
            content = rejection_text if answer_section.required else "Section withheld because the submission was rejected."
            section_outputs.append(
                AnswerSectionOutput(
                    section_id=answer_section.section_id,
                    title=answer_section.title,
                    content=content,
                    citation_ids=[],
                    section_confidence=_result_confidence(
                        0.2,
                        "Section confidence is low because acceptance gates rejected the submission.",
                    ),
                )
            )
            section_confidence.append(
                SectionConfidenceRecord(
                    section_id=answer_section.section_id,
                    title=answer_section.title,
                    confidence=_result_confidence(
                        0.2,
                        "Section confidence is low because acceptance gates rejected the submission.",
                    ),
                )
            )
        limitations_parts = [
            "Acceptance rejected the candidate submission: " + "; ".join(reason_lines),
        ]
        if review_completion_status == "incomplete":
            limitations_parts.append("Reviewer completion was incomplete.")
        return QAResult(
            question=submission.question,
            language="en",
            workflow_mode="react_reviewed",
            acceptance_status="rejected",
            final_answer="## Acceptance Rejected\n" + "\n".join(f"- {line}" for line in reason_lines),
            sections=section_outputs,
            citations=[],
            claim_trace=[],
            submission_trace=_build_submission_trace(submission),
            review_completion_status=review_completion_status,
            overall_confidence=_result_confidence(
                0.2,
                "Overall confidence is low because acceptance gates rejected the submission.",
            ),
            section_confidence=section_confidence,
            insufficient_evidence=True,
            limitations_summary=" ".join(part for part in limitations_parts if _compact_text(part)).strip(),
            retrieval_diagnostics_summary=str(retrieval_diagnostics_summary or "").strip(),
            execution_warnings=list(_merge_unique_text([], execution_warnings)),
            artifact_paths={},
            time_elapsed=0.0,
        )


class ReactReviewedWorkflow:
    def __init__(
        self,
        *,
        qa_config: Dict[str, Any],
        router: RouterNode,
        entity_resolver: EntityResolverNode,
        query_planner: QueryPlannerNode,
        retriever: RetrieverNode,
        document_acquirer: DocumentAcquirerNode,
        handoff: EvidenceExtractorHandoff,
        evidence_extractor: EvidenceExtractor,
        proposer_model_config: Optional[Dict[str, Any]] = None,
        reviewer_model_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        self.qa_config = copy.deepcopy(qa_config)
        react_config = dict(self.qa_config.get("react_reviewed", {}) or {})
        self.router_agent = RouterAgentWrapper(router=router)
        self.entity_agent = EntityResolverAgentWrapper(resolver=entity_resolver)
        self.query_planner = query_planner
        self.retriever = retriever
        self.document_acquirer = document_acquirer
        self.handoff = handoff
        self.evidence_extractor = evidence_extractor
        self.reviewer_role_order = tuple(role for role in DEFAULT_REVIEWER_ROLES if role in REVIEWER_TOOL_NAMES)
        self.stage_watchdog_seconds = max(0.01, float(react_config.get("stage_watchdog_seconds", 120.0)))
        self.reviewer_max_concurrency = max(1, int(react_config.get("reviewer_max_concurrency", len(self.reviewer_role_order))))
        self.reviewer_global_budget = max(0, int(react_config.get("reviewer_max_retrieval_actions", 1)))
        configured_reviewer_budgets = dict(
            react_config.get("reviewer_retrieval_budget_by_role", DEFAULT_REVIEWER_BUDGET_BY_ROLE) or {}
        )
        self.reviewer_budget_by_role: Dict[ReviewerRole, int] = {
            reviewer_role: max(
                0,
                int(configured_reviewer_budgets.get(reviewer_role, self.reviewer_global_budget)),
            )
            for reviewer_role in self.reviewer_role_order
        }
        self.proposer = ReactReviewedProposerAgent(
            model_config=proposer_model_config,
            max_steps_initial=react_config.get("max_propose_steps_initial", 7),
            max_steps_revision=react_config.get("max_propose_steps_revision", 7),
            llm_timeout_seconds=self.qa_config.get("model_timeout_seconds", 45.0),
            fallback_mode=react_config.get("proposer_fallback_mode", "fail_fast_only"),
            repair_attempts=react_config.get("proposer_repair_attempts", 1),
            evidence_policy=react_config.get("proposer_evidence_policy", "prefer_fulltext"),
        )
        reviewer_model_configs = reviewer_model_configs or {}
        self.reviewers = {
            reviewer_role: ReactReviewedReviewerAgent(
                reviewer_role=reviewer_role,
                model_config=reviewer_model_configs.get(reviewer_role),
                max_steps=react_config.get("reviewer_max_steps", 3),
                max_items=react_config.get("max_review_items_per_reviewer", 3),
                max_retrieval_actions=self.reviewer_budget_by_role.get(reviewer_role, self.reviewer_global_budget),
                llm_timeout_seconds=self.qa_config.get("model_timeout_seconds", 45.0),
            )
            for reviewer_role in self.reviewer_role_order
        }
        self.synthesizer = SubmissionSynthesizerAgent()

    def _evaluate_acceptance(
        self,
        *,
        task_spec: TaskSpec,
        entity_pack: EntityPack,
        submission: AnswerSubmission,
        proposer_trajectory: ReActTrajectory,
        review_items: Sequence[ReviewItem],
        reviewer_statuses: Sequence[ReviewerRunStatus],
        require_all_reviewers: bool,
        review_failure_blocks_acceptance: bool,
    ) -> AcceptanceDecision:
        blocker_codes: List[str] = []
        blocker_messages: List[str] = []
        blocking_review_ids = [
            item.review_id
            for item in review_items
            if item.status == "open" and item.severity == "blocking"
        ]
        if blocking_review_ids:
            blocker_codes.append("blocking_review_items")
            blocker_messages.append(
                "Open blocking review items remain: " + ", ".join(blocking_review_ids[:5])
            )
        if require_all_reviewers:
            incomplete_roles = [
                status.reviewer_role for status in reviewer_statuses if not _is_reviewer_llm_completed(status)
            ]
            if incomplete_roles and review_failure_blocks_acceptance:
                blocker_codes.append("reviewer_incomplete")
                blocker_messages.append(
                    "Reviewer completion failed for: " + ", ".join(str(role) for role in incomplete_roles)
                )
        topic_error = self._topic_alignment_failure(
            task_spec=task_spec,
            entity_pack=entity_pack,
            submission=submission,
        )
        if topic_error:
            blocker_codes.append("topic_alignment")
            blocker_messages.append(topic_error)
        evidence_errors = self._evidence_anchor_failures(
            submission=submission,
            proposer_trajectory=proposer_trajectory,
        )
        if evidence_errors:
            blocker_codes.append("evidence_anchor")
            blocker_messages.extend(evidence_errors)
        junk_error = self._junk_content_failure(submission=submission)
        if junk_error:
            blocker_codes.append("junk_content")
            blocker_messages.append(junk_error)
        status = "accepted" if not blocker_codes else "rejected"
        return AcceptanceDecision(
            status=status,
            blocker_codes=_merge_unique_text([], blocker_codes),
            blocker_messages=_merge_unique_text([], blocker_messages),
            blocking_review_ids=_merge_unique_text([], blocking_review_ids),
        )

    def _topic_alignment_failure(
        self,
        *,
        task_spec: TaskSpec,
        entity_pack: EntityPack,
        submission: AnswerSubmission,
    ) -> Optional[str]:
        section_text = " ".join(_compact_text(section.content) for section in submission.sections)
        normalized_text = section_text.lower()
        if not normalized_text:
            return "Submission content is empty."
        entity_anchors = [
            anchor
            for entity in entity_pack.entities
            for anchor in [entity.canonical_name, *list(entity.query_anchors), *list(entity.aliases), entity.formula]
            if _compact_text(anchor)
        ]
        must_terms = [term for term in list(task_spec.query_constraints.must_include_terms or []) if _compact_text(term)]
        question_type = str(task_spec.question_type or "").strip().lower()
        if question_type == "fact":
            anchors = must_terms[:2] or entity_anchors[:2]
            if anchors and not any(str(anchor).strip().lower() in normalized_text for anchor in anchors):
                return "Final answer does not mention the question's main anchor terms."
        elif question_type in {"mechanism", "causal"}:
            reaction_anchor = next((anchor for anchor in entity_anchors if "her" in str(anchor).lower() or "reaction" in str(anchor).lower()), None)
            catalyst_anchor = next((anchor for anchor in entity_anchors if str(anchor).strip() and str(anchor).lower() not in {"her", "hydrogen evolution reaction"}), None)
            missing = []
            if reaction_anchor and str(reaction_anchor).lower() not in normalized_text:
                missing.append(str(reaction_anchor))
            if catalyst_anchor and str(catalyst_anchor).lower() not in normalized_text:
                missing.append(str(catalyst_anchor))
            if missing:
                return "Mechanism answer is missing core task anchors: " + ", ".join(missing)
        elif question_type == "comparison":
            anchors = []
            for entity in entity_pack.entities:
                anchor = _compact_text(entity.canonical_name) or next(( _compact_text(item) for item in entity.query_anchors if _compact_text(item)), "")
                if anchor:
                    anchors.append(anchor)
            anchors = anchors[:2]
            missing = [anchor for anchor in anchors if anchor.lower() not in normalized_text]
            comparison_terms = ("compare", "compared", "versus", "vs", "benchmark", "side-by-side")
            if missing or not any(term in normalized_text for term in comparison_terms):
                return "Comparison answer is missing comparison-object anchors or comparability language."
        return None

    def _evidence_anchor_failures(
        self,
        *,
        submission: AnswerSubmission,
        proposer_trajectory: ReActTrajectory,
    ) -> List[str]:
        errors: List[str] = []
        citation_lookup = {citation.citation_id: citation for citation in submission.citations}
        if not citation_lookup:
            return ["Submission has no citation catalog."]
        real_steps = {
            (step_ref.trajectory_id, int(step_ref.step_number))
            for step_ref in submission.step_refs
        }
        real_steps.update(
            (proposer_trajectory.trajectory_id, int(step.step_number))
            for step in proposer_trajectory.steps
        )
        anchored_citation_found = False
        for section in submission.sections:
            for citation_id in section.citation_ids:
                citation = citation_lookup.get(citation_id)
                if citation is None:
                    errors.append(f"Section '{section.section_id}' references missing citation_id '{citation_id}'.")
                    continue
                if citation.section_ids or citation.evidence_ids:
                    anchored_citation_found = True
            for step_ref in section.step_refs:
                step_key = (step_ref.trajectory_id, int(step_ref.step_number))
                if step_key not in real_steps:
                    errors.append(
                        f"Section '{section.section_id}' references missing step_ref {step_ref.trajectory_id}:{step_ref.step_number}."
                    )
        if not anchored_citation_found:
            errors.append("Submission citations exist but none provide section_ids or evidence_ids anchors.")
        return _merge_unique_text([], errors)

    def _junk_content_failure(self, *, submission: AnswerSubmission) -> Optional[str]:
        patterns = (
            r"editor\(s\)\s+date",
            r"size\s+file type",
            r"location\s+license\s+abstract",
            r"aboriginal and torres strait islander",
        )
        repeated_sections = {}
        for section in submission.sections:
            content = _compact_text(section.content)
            if not content:
                continue
            repeated_sections[content] = repeated_sections.get(content, 0) + 1
            for pattern in patterns:
                if re.search(pattern, content, re.I):
                    return f"Section '{section.section_id}' contains obvious metadata/junk text."
        if any(count >= 2 and len(text.split()) >= 8 for text, count in repeated_sections.items()):
            return "Multiple sections repeat the same long content block, suggesting answer assembly drift."
        return None

    def _build_reviewer_session(
        self,
        *,
        store: QAArtifactStore,
        reviewer_role: ReviewerRole,
        cycle_number: int,
    ) -> ReviewerSession:
        artifact_root = store.path(Path("reviewers") / reviewer_role / f"cycle_{cycle_number}")
        return ReviewerSession(
            reviewer_role=reviewer_role,
            cycle_number=cycle_number,
            artifact_store=QAArtifactStore(base_dir=artifact_root),
            budget_state=ReviewerBudgetState(
                role=reviewer_role,
                budget_limit=self.reviewer_budget_by_role.get(reviewer_role, self.reviewer_global_budget),
            ),
        )

    def _run_reviewer_with_retries(
        self,
        *,
        reviewer_role: ReviewerRole,
        reviewer: ReactReviewedReviewerAgent,
        workspace: ReactReviewedWorkspace,
        submission: AnswerSubmission,
        proposer_trajectory: ReActTrajectory,
        cycle_number: int,
        review_call_retry_limit: int,
        session: ReviewerSession,
    ) -> ReviewerExecutionResult:
        attempt = 0
        last_error: Optional[BaseException] = None
        while attempt <= review_call_retry_limit:
            try:
                items, reviewer_trajectory, reviewer_status = reviewer.run(
                    workspace=workspace,
                    submission=submission,
                    proposer_trajectory=proposer_trajectory,
                    cycle_number=cycle_number,
                    session=session,
                )
                session.write_run_artifacts(
                    review_items=items,
                    reviewer_trajectory=reviewer_trajectory,
                    reviewer_status=reviewer_status,
                )
                return ReviewerExecutionResult(
                    reviewer_role=reviewer_role,
                    review_items=list(items),
                    reviewer_trajectory=reviewer_trajectory,
                    reviewer_status=reviewer_status,
                )
            except Exception as exc:
                last_error = exc
                attempt += 1
                logger.warning(
                    "react_reviewed_reviewer_retry role=%s cycle=%s attempt=%s error=%s",
                    reviewer_role,
                    cycle_number,
                    attempt,
                    exc,
                )
        error_status = ReviewerRunStatus(
            reviewer_role=reviewer_role,
            status="error",
            message=str(last_error or "unknown reviewer error"),
            cycle_number=cycle_number,
            retrieval_actions_used=session.budget_state.actions_used,
            retrieval_budget_limit=session.budget_state.budget_limit,
            budget_blocked_calls=session.budget_state.blocked_calls,
        )
        session.write_run_artifacts(
            review_items=[],
            reviewer_trajectory=None,
            reviewer_status=error_status,
        )
        return ReviewerExecutionResult(
            reviewer_role=reviewer_role,
            review_items=[],
            reviewer_trajectory=None,
            reviewer_status=error_status,
        )

    def _build_failure_submission(
        self,
        *,
        task_spec: TaskSpec,
        question: str,
        cycle_number: int,
        message: str,
        prior_submission: Optional[AnswerSubmission],
        trajectory_id: str,
    ) -> AnswerSubmission:
        if prior_submission is not None:
            limitations = list(prior_submission.limitations)
            failure_note = _compact_text(message)
            if failure_note:
                limitations = _merge_unique_text(limitations, [f"Workflow failure: {failure_note}"])
            failed_submission = prior_submission.model_copy(
                update={
                    "limitations": limitations,
                    "trajectory_id": trajectory_id or prior_submission.trajectory_id,
                }
            )
            aligned_step_refs = [SubmissionStepRef(trajectory_id=failed_submission.trajectory_id, step_number=1)]
            aligned_sections = [
                section.model_copy(update={"step_refs": list(aligned_step_refs)})
                for section in failed_submission.sections
            ]
            return failed_submission.model_copy(
                update={
                    "sections": aligned_sections,
                    "step_refs": aligned_step_refs,
                }
            )
        sections: List[SubmissionSection] = []
        for answer_section in task_spec.answer_sections:
            sections.append(
                SubmissionSection(
                    section_id=answer_section.section_id,
                    title=answer_section.title,
                    content="Workflow failed before a grounded submission could be completed.",
                    citation_ids=[],
                    step_refs=[SubmissionStepRef(trajectory_id=trajectory_id, step_number=1)],
                    issue_refs=[],
                    section_confidence=_confidence(
                        0.05,
                        "Confidence is minimal because the workflow terminated before producing a validated submission.",
                    ),
                )
            )
        return AnswerSubmission(
            submission_id=f"submission_cycle_{max(1, int(cycle_number))}",
            question=question,
            version=max(1, int(cycle_number)),
            sections=sections,
            citations=[],
            limitations=[f"Workflow failure: {_compact_text(message) or 'unknown error'}"],
            overall_confidence=_confidence(
                0.05,
                "Confidence is minimal because the workflow terminated before producing a validated submission.",
            ),
            trajectory_id=trajectory_id,
            step_refs=[SubmissionStepRef(trajectory_id=trajectory_id, step_number=1)],
            issue_refs=[],
        )

    def run(
        self,
        *,
        question: str,
        context: Optional[str] = None,
        artifact_dir: Optional[str] = None,
    ) -> QAResult:
        started_at = time.perf_counter()
        store = QAArtifactStore(base_dir=artifact_dir)
        task_spec, router_artifacts = self.router_agent.run(
            question=question,
            context=context,
            artifact_store=store,
        )
        entity_pack, entity_artifacts, entity_resolution_snapshot = self.entity_agent.run(
            question=question,
            task_spec=task_spec,
            artifact_store=store,
        )
        workspace = ReactReviewedWorkspace(
            question=question,
            context=context,
            task_spec=task_spec,
            entity_pack=entity_pack,
            entity_resolution_snapshot=entity_resolution_snapshot,
            artifact_store=store,
            query_planner=self.query_planner,
            retriever=self.retriever,
            document_acquirer=self.document_acquirer,
            handoff=self.handoff,
            evidence_extractor=self.evidence_extractor,
            stage_watchdog_seconds=self.stage_watchdog_seconds,
        )

        react_config = dict(self.qa_config.get("react_reviewed", {}) or {})
        max_review_cycles = max(1, int(react_config.get("max_review_cycles", 3)))
        require_all_reviewers = bool(react_config.get("require_all_reviewers", True))
        review_failure_blocks_acceptance = bool(react_config.get("review_failure_blocks_acceptance", True))
        review_call_retry_limit = max(0, int(react_config.get("review_call_retry_limit", 1)))
        max_items_per_step_section = max(1, int(react_config.get("max_review_items_per_step_section", 1)))
        stop_on_no_blocking_items = bool(react_config.get("stop_on_no_blocking_items", True))

        cycle_states: List[SubmissionCycleState] = []
        open_review_items: List[ReviewItem] = []
        accepted_submission: Optional[AnswerSubmission] = None
        accepted_proposer_trajectory: Optional[ReActTrajectory] = None
        latest_submission: Optional[AnswerSubmission] = None
        latest_proposer_trajectory: Optional[ReActTrajectory] = None
        review_completion_status: ReviewCompletionStatus = "completed"
        acceptance_decision = AcceptanceDecision(status="rejected", blocker_codes=["not_evaluated"], blocker_messages=["Acceptance not yet evaluated."])
        latest_reviewers: Dict[str, ReActTrajectory] = {}
        latest_statuses: List[ReviewerRunStatus] = []
        workflow_error: Optional[BaseException] = None
        failing_cycle_number = 1
        try:
            for cycle_number in range(1, max_review_cycles + 1):
                failing_cycle_number = cycle_number
                workspace.set_review_context(
                    submission=latest_submission,
                    proposer_trajectory=latest_proposer_trajectory,
                    open_review_items=open_review_items,
                    cycle_number=cycle_number,
                )
                submission, proposer_trajectory = self.proposer.run(
                    workspace=workspace,
                    cycle_number=cycle_number,
                    open_review_items=open_review_items,
                )
                latest_submission = submission
                latest_proposer_trajectory = proposer_trajectory
                workspace.set_review_context(
                    submission=submission,
                    proposer_trajectory=proposer_trajectory,
                    open_review_items=open_review_items,
                    cycle_number=cycle_number,
                )

                reviewer_items: List[ReviewItem] = []
                reviewer_trajectories: Dict[str, ReActTrajectory] = {}
                reviewer_statuses: List[ReviewerRunStatus] = []
                reviewer_roles = [role for role in self.reviewer_role_order if role in self.reviewers]
                reviewer_sessions = {
                    reviewer_role: self._build_reviewer_session(
                        store=store,
                        reviewer_role=reviewer_role,
                        cycle_number=cycle_number,
                    )
                    for reviewer_role in reviewer_roles
                }
                max_workers = min(self.reviewer_max_concurrency, len(reviewer_roles)) if reviewer_roles else 1
                reviewer_results: Dict[str, ReviewerExecutionResult] = {}
                if reviewer_roles:
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        future_map: Dict[Future[ReviewerExecutionResult], ReviewerRole] = {
                            executor.submit(
                                self._run_reviewer_with_retries,
                                reviewer_role=reviewer_role,
                                reviewer=self.reviewers[reviewer_role],
                                workspace=workspace,
                                submission=submission,
                                proposer_trajectory=proposer_trajectory,
                                cycle_number=cycle_number,
                                review_call_retry_limit=review_call_retry_limit,
                                session=reviewer_sessions[reviewer_role],
                            ): reviewer_role
                            for reviewer_role in reviewer_roles
                        }
                        wait(list(future_map.keys()))
                        for future, reviewer_role in future_map.items():
                            reviewer_results[reviewer_role] = future.result()
                workspace.write_shared_snapshot()
                for reviewer_role in reviewer_roles:
                    execution_result = reviewer_results.get(reviewer_role)
                    if execution_result is None:
                        continue
                    reviewer_items.extend(execution_result.review_items)
                    if execution_result.reviewer_trajectory is not None:
                        reviewer_trajectories[reviewer_role] = execution_result.reviewer_trajectory
                    reviewer_statuses.append(execution_result.reviewer_status)

                latest_reviewers = reviewer_trajectories
                latest_statuses = reviewer_statuses
                normalized_items = self._normalize_review_items(
                    items=reviewer_items,
                    proposer_trajectory=proposer_trajectory,
                    max_items_per_step_section=max_items_per_step_section,
                )
                review_responses = self._build_review_responses(
                    prior_review_items=open_review_items,
                    submission=submission,
                )
                cycle_states.append(
                    SubmissionCycleState(
                        cycle_number=cycle_number,
                        current_submission=submission,
                        proposer_trajectory=proposer_trajectory.to_dict(),
                        reviewer_trajectories={
                            role: trajectory.to_dict()
                            for role, trajectory in reviewer_trajectories.items()
                        },
                        open_review_items=normalized_items,
                        review_responses=review_responses,
                        reviewer_statuses=reviewer_statuses,
                    )
                )
                open_review_items = normalized_items

                all_reviewers_completed = all(_is_reviewer_llm_completed(status) for status in reviewer_statuses)
                review_completion_status = "completed" if all_reviewers_completed else "incomplete"
                acceptance_decision = self._evaluate_acceptance(
                    task_spec=task_spec,
                    entity_pack=entity_pack,
                    submission=submission,
                    proposer_trajectory=proposer_trajectory,
                    review_items=open_review_items,
                    reviewer_statuses=reviewer_statuses,
                    require_all_reviewers=require_all_reviewers,
                    review_failure_blocks_acceptance=review_failure_blocks_acceptance,
                )
                if acceptance_decision.accepted:
                    accepted_submission = submission
                    accepted_proposer_trajectory = proposer_trajectory
                    break
                if review_failure_blocks_acceptance and "reviewer_incomplete" in acceptance_decision.blocker_codes:
                    break
                if stop_on_no_blocking_items and not acceptance_decision.blocking_review_ids:
                    break
        except QueryPlannerExecutionError:
            raise
        except Exception as exc:
            workflow_error = exc
            review_completion_status = "incomplete"
            blocker_message = _compact_text(str(exc)) or "Workflow execution failed."
            acceptance_decision = AcceptanceDecision(
                status="rejected",
                blocker_codes=["workflow_execution_failed"],
                blocker_messages=[blocker_message],
                blocking_review_ids=[
                    item.review_id
                    for item in open_review_items
                    if item.status == "open" and item.severity == "blocking"
                ],
            )
            if latest_proposer_trajectory is None:
                trajectory_id = str(
                    getattr(exc, "trajectory", None).trajectory_id
                    if getattr(exc, "trajectory", None) is not None
                    else f"traj_failure_cycle_{max(1, failing_cycle_number)}"
                )
                latest_proposer_trajectory = ReActTrajectory(query=question, trajectory_id=trajectory_id)
                latest_proposer_trajectory.finalize(json.dumps({"error": blocker_message}, ensure_ascii=False))
            latest_submission = self._build_failure_submission(
                task_spec=task_spec,
                question=question,
                cycle_number=max(1, failing_cycle_number),
                message=blocker_message,
                prior_submission=latest_submission,
                trajectory_id=latest_proposer_trajectory.trajectory_id,
            )
            logger.warning(
                "react_reviewed_workflow_returning_rejected_result error=%s cycle=%s",
                blocker_message,
                failing_cycle_number,
            )

        final_submission = accepted_submission if accepted_submission is not None else latest_submission
        final_proposer_trajectory = accepted_proposer_trajectory if accepted_proposer_trajectory is not None else latest_proposer_trajectory
        if final_submission is None or final_proposer_trajectory is None:
            raise ValueError("react_reviewed workflow did not produce a submission.")

        final_result = self.synthesizer.run(
            task_spec=task_spec,
            submission=final_submission,
            review_items=open_review_items,
            review_completion_status=review_completion_status,
            acceptance_decision=acceptance_decision,
            retrieval_diagnostics_summary=workspace.diagnostics_summary(),
            execution_warnings=workspace.execution_warnings,
        )
        artifact_paths = {
            **router_artifacts,
            **entity_artifacts,
            "candidate_submission": store.write_json(
                "candidate_submission.json",
                final_submission.model_dump(exclude_none=True),
            ),
            "acceptance_decision": store.write_json(
                "acceptance_decision.json",
                acceptance_decision.to_dict(),
            ),
            "submission_trace": store.write_json(
                "submission_trace.json",
                [item.model_dump(exclude_none=True) for item in _build_submission_trace(final_submission)],
            ),
            "submission_cycles": store.write_json(
                "submission_cycles.json",
                [item.model_dump(exclude_none=True) for item in cycle_states],
            ),
            "proposer_trajectory": store.write_json(
                "proposer_trajectory.json",
                final_proposer_trajectory.to_dict(),
            ),
            "reviewer_trajectories": store.write_json(
                "reviewer_trajectories.json",
                {role: trajectory.to_dict() for role, trajectory in latest_reviewers.items()},
            ),
            "review_statuses": store.write_json(
                "review_statuses.json",
                [item.model_dump(exclude_none=True) for item in latest_statuses],
            ),
            "final_review_items": store.write_json(
                "final_review_items.json",
                [item.model_dump(exclude_none=True) for item in open_review_items],
            ),
            "final_answer": store.write_text("final_answer.md", final_result.final_answer),
            "workflow_error": (
                store.write_text("workflow_error.txt", _compact_text(str(workflow_error)))
                if workflow_error is not None
                else None
            ),
            "qa_result": str(store.path("qa_result.json")),
        }
        artifact_paths = {key: value for key, value in artifact_paths.items() if value is not None}
        if acceptance_decision.accepted:
            artifact_paths["final_submission"] = store.write_json(
                "final_submission.json",
                final_submission.model_dump(exclude_none=True),
            )
        elapsed = round(time.perf_counter() - started_at, 3)
        finalized_result = final_result.model_copy(
            update={
                "artifact_paths": artifact_paths,
                "time_elapsed": elapsed,
            }
        )
        store.write_json("qa_result.json", finalized_result.model_dump(exclude_none=True))
        return finalized_result

    def _normalize_review_items(
        self,
        *,
        items: Sequence[ReviewItem],
        proposer_trajectory: ReActTrajectory,
        max_items_per_step_section: int,
    ) -> List[ReviewItem]:
        deduped: List[ReviewItem] = []
        seen = set()
        anchor_counts: Dict[Tuple[str, str, int, str], int] = {}
        placeholder_critiques = {
            "Reviewer output did not provide critique text.",
            "Reviewer identified an issue that required salvage normalization.",
        }
        for item in items:
            current = item
            if item.anchor_kind == "step_section" and item.target_trajectory_id != proposer_trajectory.trajectory_id:
                current = item.model_copy(update={"target_trajectory_id": proposer_trajectory.trajectory_id})
            if current.flaw_type == "needs_manual_review" or str(current.critique or "").strip() in placeholder_critiques:
                continue
            key = (
                current.reviewer_role,
                current.anchor_kind,
                current.severity,
                current.flaw_type,
                current.target_trajectory_id,
                current.target_step_number,
                current.target_section_id,
                current.critique,
            )
            if key in seen:
                continue
            seen.add(key)
            if current.anchor_kind == "step_section":
                anchor_key = (
                    current.target_trajectory_id or "",
                    current.target_section_id or "",
                    int(current.target_step_number or 0),
                    current.reviewer_role,
                )
                anchor_counts[anchor_key] = anchor_counts.get(anchor_key, 0) + 1
                if anchor_counts[anchor_key] > max_items_per_step_section:
                    continue
            deduped.append(current)
        return deduped

    def _build_review_responses(
        self,
        *,
        prior_review_items: Sequence[ReviewItem],
        submission: AnswerSubmission,
    ) -> List[ReviewResponse]:
        section_lookup = {section.section_id: section for section in submission.sections}
        responses: List[ReviewResponse] = []
        for item in prior_review_items:
            section = section_lookup.get(str(item.target_section_id or ""))
            if item.anchor_kind == "missing_section" and section is not None:
                responses.append(
                    ReviewResponse(
                        review_id=item.review_id,
                        response_mode="addressed",
                        response_note="Missing section was added in the revised submission.",
                        new_step_refs=list(section.step_refs),
                        section_patch_refs=[section.section_id],
                    )
                )
                continue
            if section is not None and section.citation_ids:
                responses.append(
                    ReviewResponse(
                        review_id=item.review_id,
                        response_mode="partially_addressed",
                        response_note="Section was revised and still requires reviewer confirmation.",
                        new_step_refs=list(section.step_refs),
                        section_patch_refs=[section.section_id],
                    )
                )
        return responses
