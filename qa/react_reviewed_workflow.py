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
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from agents.chat_models import build_chat_model_from_config, describe_chat_model_config
from agents.react_agent import ReActAgent, ToolResult
from agents.react_reasoning import ReActStep, ReActTrajectory, ToolCallRecord
from qa.artifacts import QAArtifactStore
from qa.evidence import EvidenceExtractor
from qa.handoff import EvidenceExtractorHandoff
from qa.llm_utils import invoke_llm, parse_json_payload
from qa.nodes.document_acquirer import DocumentAcquirerNode
from qa.nodes.entity_resolver import EntityResolverNode
from qa.nodes.query_planner import QueryPlannerNode
from qa.nodes.retriever import RetrieverNode
from qa.nodes.router import RouterNode
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

PROPOSER_TOOL_NAMES = (
    "plan_queries",
    "search_papers",
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


def _confidence(level_score: float, rationale: str) -> SubmissionConfidenceRating:
    score = max(0.0, min(round(float(level_score), 2), 1.0))
    if score >= 0.75:
        level = "high"
    elif score >= 0.45:
        level = "medium"
    else:
        level = "low"
    return SubmissionConfidenceRating(level=level, score=score, rationale=str(rationale).strip() or "No rationale.")


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


@dataclass
class _ProposerRunState:
    evidence_policy: str
    query_plan_ids: List[str] = field(default_factory=list)
    searched_paper_ids: set[str] = field(default_factory=set)
    acquired_paper_ids: set[str] = field(default_factory=set)
    evidence_ids: set[str] = field(default_factory=set)
    section_ids_by_paper: Dict[str, set[str]] = field(default_factory=dict)
    evidence_ids_by_paper: Dict[str, set[str]] = field(default_factory=dict)
    evidence_layers_by_id: Dict[str, str] = field(default_factory=dict)
    fulltext_status_by_paper: Dict[str, str] = field(default_factory=dict)
    fulltext_available_by_paper: Dict[str, bool] = field(default_factory=dict)

    def record_plan_queries(self, payload: Sequence[Dict[str, Any]]) -> None:
        for item in list(payload or []):
            query_plan_id = str(item.get("query_plan_id") or "").strip()
            if query_plan_id and query_plan_id not in self.query_plan_ids:
                self.query_plan_ids.append(query_plan_id)

    def record_search_results(self, payload: Sequence[Dict[str, Any]]) -> None:
        for item in list(payload or []):
            paper_id = str(item.get("paper_id") or "").strip()
            if paper_id:
                self.searched_paper_ids.add(paper_id)

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
            "searched_paper_ids": sorted(self.searched_paper_ids),
            "acquired_paper_ids": sorted(self.acquired_paper_ids),
            "evidence_ids": sorted(self.evidence_ids),
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
        task_spec = self.router.run(question=question, context=context)
        debug_payload = dict(getattr(self.router, "last_run_debug", {}) or {})
        task_spec_path = artifact_store.write_json(
            "router/task_spec.json",
            task_spec.model_dump(exclude_none=True),
        )
        semantic_stage_path = None
        localization_stage_path = None
        fallback_reason_path = None
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
        if isinstance(debug_payload.get("fallback_reason"), dict):
            fallback_reason_path = artifact_store.write_json(
                "router/fallback_reason.json",
                debug_payload["fallback_reason"],
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
        if fallback_reason_path:
            artifacts["router_fallback_reason"] = fallback_reason_path
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
            )
        return self.document_acquirer

    def plan_queries(self, *, focus: str = "initial") -> List[Dict[str, Any]]:
        plans = self.query_planner.run(task_spec=self.task_spec, entity_pack=self.entity_pack)
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
            preferred_sources=["openalex", "crossref", "semantic_scholar"],
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
                candidates = retriever.run(
                    task_spec=self.task_spec,
                    entity_pack=self.entity_pack,
                    query_plans=[query_plan],
                    artifact_store=store,
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
                paper_records, section_indices = acquirer.run(
                    candidates=[candidate],
                    artifact_store=store,
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
            payloads = [
                {
                    "paper_id": paper_record.paper_id,
                    "section_id": "sec_abstract",
                    "section_type": "abstract",
                    "heading": "Abstract",
                    "text": paper_record.abstract,
                }
            ]
        else:
            payloads = [
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
                if not section_ids and not preferred_sections:
                    evidence_items = self.evidence_extractor.run(
                        task_spec=self.task_spec,
                        entity_pack=self.entity_pack,
                        paper_record=paper_record,
                        section_index=section_index,
                    )
                else:
                    evidence_items = []
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
        if citation_id and current_submission is not None:
            citation = next(
                (item for item in current_submission.citations if item.citation_id == str(citation_id)),
                None,
            )
            if citation is not None:
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
        self.llm_timeout_seconds = float(llm_timeout_seconds)
        self.fallback_mode = str(fallback_mode or "fail_fast_only").strip().lower() or "fail_fast_only"
        self.repair_attempts = max(0, int(repair_attempts))
        self.evidence_policy = str(evidence_policy or "prefer_fulltext").strip().lower() or "prefer_fulltext"

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

        def acquire_document(paper_id: str) -> ToolResult:
            """Acquire the selected paper and build its section index."""
            normalized_paper_id = str(paper_id or "").strip()
            if normalized_paper_id not in run_state.searched_paper_ids:
                return _policy_block(
                    f"acquire_document requires a paper selected from prior search_papers results; unknown paper_id={normalized_paper_id}.",
                    code="paper_not_searched",
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
            StructuredTool.from_function(plan_queries, name="plan_queries"),
            StructuredTool.from_function(search_papers, name="search_papers"),
            StructuredTool.from_function(acquire_document, name="acquire_document"),
            StructuredTool.from_function(read_sections, name="read_sections"),
            StructuredTool.from_function(extract_evidence, name="extract_evidence"),
            StructuredTool.from_function(fetch_citation_context, name="fetch_citation_context"),
            StructuredTool.from_function(inspect_entity_cache, name="inspect_entity_cache"),
            StructuredTool.from_function(inspect_submission_anchor, name="inspect_submission_anchor"),
            StructuredTool.from_function(analyze_submission_gap, name="analyze_submission_gap"),
            StructuredTool.from_function(conclude, name="conclude"),
        ]
        system_prompt = self._build_system_prompt()
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
                action_phase_instruction=self._action_instruction(PROPOSER_TOOL_NAMES),
                search_tool_names=[
                    "plan_queries",
                    "search_papers",
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
                query=self._build_user_prompt(workspace=workspace, cycle_number=cycle_number, open_review_items=open_review_items),
                system_prompt_override=system_prompt,
                max_steps_override=self.max_steps_initial if cycle_number == 1 else self.max_steps_revision,
                llm_timeout_seconds=self.llm_timeout_seconds,
            )
        except Exception as exc:
            self._raise_execution_failure(
                workspace=workspace,
                cycle_number=cycle_number,
                stage="proposer_execution",
                message=f"Proposer failed during ReAct execution: {exc}",
                details={"fallback_mode": self.fallback_mode},
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
            error = ReactReviewedStructuredOutputError(
                stage="proposer",
                cycle_number=cycle_number,
                message=f"invalid proposer structured output: {exc}",
                response_content=getattr(response, "content", None),
                structured_output=getattr(response, "structured_output", None),
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
                                "You are repairing an invalid AnswerSubmission for a chemistry QA proposer.\n"
                                "Return STRICT JSON only.\n"
                                "Return either {\"kind\":\"submission\",\"payload\":{...}} or the submission object itself.\n"
                                "Use only the supplied task_spec, entity_pack, review items, retrieval state, and cited IDs.\n"
                                "Do not invent citations, evidence_ids, section_ids, or papers.\n"
                                "If the current run has only abstract-backed evidence, include an explicit degraded-evidence limitation and lower overall confidence.\n"
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
                                    "open_review_items": [item.model_dump(exclude_none=True) for item in open_review_items],
                                    "retrieval_state": run_state.prompt_payload(),
                                    "validation_error": str(last_error),
                                    "invalid_response_content": last_error.response_content,
                                    "invalid_structured_output": last_error.structured_output,
                                },
                                limit=20000,
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
                last_error = ReactReviewedStructuredOutputError(
                    stage="proposer_repair",
                    cycle_number=cycle_number,
                    message=f"invalid proposer repair output: {exc}",
                    response_content=raw_response,
                    structured_output=None,
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
        raw_payload.setdefault("submission_id", f"submission_cycle_{cycle_number}")
        raw_payload.setdefault("question", workspace.question)
        raw_payload.setdefault("version", cycle_number)
        raw_payload.setdefault("trajectory_id", trajectory_id or f"traj_placeholder_{cycle_number}")
        raw_payload.setdefault("citations", [])
        raw_payload.setdefault("limitations", [])
        raw_payload.setdefault("issue_refs", [item.review_id for item in open_review_items])
        raw_payload.setdefault(
            "overall_confidence",
            _confidence(0.65, "LLM proposer confidence normalized by conclude validator.").model_dump(),
        )
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
            section_payload.setdefault("content", "")
            section_payload.setdefault(
                "section_confidence",
                _confidence(0.6, "Section confidence normalized by conclude validator.").model_dump(),
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
            normalized_sections.append(section_payload)
        raw_payload["sections"] = normalized_sections
        raw_payload["step_refs"] = [
            item
            if not isinstance(item, dict)
            else {**item, "trajectory_id": item.get("trajectory_id") or raw_payload["trajectory_id"]}
            for item in list(raw_payload.get("step_refs") or [])
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

        selected_paper_ids = [item.get("paper_id") for item in search_results if item.get("paper_id")][:2]
        acquired_payloads: List[Dict[str, Any]] = []
        acquire_tool_calls: List[ToolCallRecord] = []
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
        acquire_step = self._add_tool_step(
            trajectory=trajectory,
            thought="Acquire full text or abstract-backed section indices for the strongest papers.",
            tool_calls=acquire_tool_calls,
        )

        extracted_evidence: List[Dict[str, Any]] = []
        extract_tool_calls: List[ToolCallRecord] = []
        for paper_id in selected_paper_ids:
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
        extract_step = self._add_tool_step(
            trajectory=trajectory,
            thought="Extract evidence snippets and stable evidence references for answer assembly.",
            tool_calls=extract_tool_calls,
        )

        step_refs = [_step_ref(trajectory, plan_step), _step_ref(trajectory, search_step), _step_ref(trajectory, acquire_step), _step_ref(trajectory, extract_step)]
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

    def _build_submission_from_workspace(
        self,
        *,
        workspace: ReactReviewedWorkspace,
        cycle_number: int,
        step_refs: Sequence[SubmissionStepRef],
        open_review_items: Sequence[ReviewItem],
    ) -> AnswerSubmission:
        citations = self._build_submission_citations(workspace)
        evidence_items = list(workspace.evidence_items.values())
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

    def _build_submission_citations(self, workspace: ReactReviewedWorkspace) -> List[SubmissionCitation]:
        citations: List[SubmissionCitation] = []
        for paper_record in list(workspace.paper_records.values())[:4]:
            paper_evidence = [item for item in workspace.evidence_items.values() if item.paper_id == paper_record.paper_id]
            section_ids = _merge_unique_text([], [item.section_id for item in paper_evidence])
            evidence_ids = _merge_unique_text([], [item.evidence_id for item in paper_evidence])
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
        if answer_section_id in {"representative_papers"}:
            if not citations:
                return "Representative papers could not be confirmed from the current retrieval set."
            fragments = [
                f"{citation.title} ({citation.year or 'n.d.'})"
                for citation in citations[:3]
            ]
            return "Representative papers in the current run include " + "; ".join(fragments) + "."
        if answer_section_id in {"caveats", "causal_limitations", "open_questions"}:
            if limitations:
                return _compact_text(limitations[0].snippet)
            if related_issues:
                return "Open review items remain attached to this section: " + ", ".join(related_issues) + "."
            return "Current evidence remains incomplete, so this section stays conservative."
        if answer_section_id in {"conditions"}:
            conditions: List[str] = []
            for evidence_item in observations[:3]:
                for axis_name, axis_value in evidence_item.conditions.items():
                    conditions.append(f"{axis_name}: {axis_value}")
            if conditions:
                return "Material conditions mentioned in the current evidence include " + "; ".join(conditions[:4]) + "."
        if observations:
            first = observations[0]
            return _compact_text(first.snippet)
        if citations:
            return (
                f"The retrieved literature set for '{workspace.question}' contains usable citations, "
                "but the extracted evidence is still thin for a stronger section-level claim."
            )
        return "Evidence remains insufficient to support a stronger section-level answer."

    def _build_system_prompt(self) -> str:
        return (
            "You are ProposerAgent in a chemistry QA workflow.\n"
            "Fixed upstream inputs are RouterAgent TaskSpec and EntityResolverAgent outputs; do not challenge them.\n"
            "You are the primary retrieval orchestrator for this cycle.\n"
            "You must build an AnswerSubmission already sectioned according to TaskSpec.answer_sections.\n"
            "Required phase order: 1) plan_queries, 2) search_papers, 3) choose candidate papers, 4) acquire_document, 5) read_sections/extract_evidence, 6) conclude.\n"
            "Never skip directly to conclude before evidence extraction succeeds.\n"
            "You may refine retrieval with additional ad hoc searches after plan_queries, but every paper you acquire must come from prior search_papers results in this cycle.\n"
            "Selection rubric, in order: entity/condition alignment with TaskSpec; question-type and lane fit; full-text availability signal; evidence density after extraction; coverage diversity when multiple papers are needed.\n"
            "Prefer full-text evidence. If usable full-text evidence exists, use fulltext-backed citations in substantive sections.\n"
            "If only abstract-backed evidence is available, you may submit a degraded answer only if you explicitly disclose the degradation in limitations and lower overall confidence.\n"
            "Only use citations, section_ids, and evidence_ids returned by tools in this cycle.\n"
            "There is no deterministic fallback. If the run cannot produce a valid evidence-backed submission within budget, fail explicitly.\n"
            "If conclude returns a validation error, fix the submission object rather than inventing unsupported anchors.\n"
            "When finished, call conclude with `submission={...}` only.\n"
            "The only valid conclude argument name is `submission`.\n"
            "Do not output free text instead of the conclude tool call.\n"
        )

    def _build_user_prompt(
        self,
        *,
        workspace: ReactReviewedWorkspace,
        cycle_number: int,
        open_review_items: Sequence[ReviewItem],
    ) -> str:
        return _json_preview(
            {
                "cycle_number": cycle_number,
                "question": workspace.question,
                "context": workspace.context,
                "task_spec": workspace.task_spec.model_dump(exclude_none=True),
                "entity_pack": workspace.entity_pack.model_dump(exclude_none=True),
                "open_review_items": [item.model_dump(exclude_none=True) for item in open_review_items],
                "retrieval_policy": {
                    "fallback_mode": self.fallback_mode,
                    "repair_attempts": self.repair_attempts,
                    "evidence_policy": self.evidence_policy,
                    "phase_order": [
                        "plan_queries",
                        "search_papers",
                        "candidate_locking",
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
            },
            limit=12000,
        )

    def _thought_instruction(self) -> str:
        return (
            "CURRENT PHASE: THOUGHT\n"
            "State the next retrieval or revision intent in 1-2 short sentences.\n"
            "Do not output JSON.\n"
        )

    def _action_instruction(self, tool_names: Sequence[str]) -> str:
        retrieval_tools = [name for name in tool_names if name not in {"analyze_submission_gap", "conclude"}]
        return (
            "CURRENT PHASE: ACTION\n"
            "You must call tools instead of answering in free text.\n"
            f"Allowed tools: {', '.join(tool_names)}.\n"
            f"Treat these as retrieval/inspection tools: {', '.join(retrieval_tools)}.\n"
            "Call plan_queries before any search. Call acquire_document only for papers returned by search_papers in this cycle.\n"
            "Call extract_evidence before conclude. Prefer fulltext-backed evidence when it exists.\n"
            "Do not mix retrieval tools with conclude in the same step.\n"
            "Use conclude only when the submission object is ready.\n"
            "The final step must be `conclude(submission={...})`.\n"
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
        if agent_output is not None:
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
        workspace.restore_mutable_state(stage_snapshot)
        items, trajectory = self._run_deterministic(
            workspace=workspace,
            submission=submission,
            proposer_trajectory=proposer_trajectory,
            cycle_number=cycle_number,
            session=session,
        )
        return items, trajectory, ReviewerRunStatus(
            reviewer_role=self.reviewer_role,
            status="completed",
            message="completed",
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
    ) -> Optional[Tuple[List[ReviewItem], ReActTrajectory, str]]:
        StructuredTool = _lazy_structured_tool_import()
        provider, model_name, has_api_key = describe_chat_model_config(self.model_config)
        if StructuredTool is None or not has_api_key or not provider or not model_name:
            return None

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
        tools = [StructuredTool.from_function(tool_builders[name], name=name) for name in allowed_tool_names]
        system_prompt = self._build_system_prompt()
        try:
            agent = ReActAgent(
                agent_id=f"qa_reviewer_{self.reviewer_role}",
                name=f"ReviewerAgent[{self.reviewer_role}]",
                model_config=self.model_config,
                system_prompt=system_prompt,
                max_react_steps=self.max_steps,
                verbose=False,
                tools=tools,
                thought_phase_instruction="CURRENT PHASE: THOUGHT\nState the next audit target in one short sentence.\n",
                action_phase_instruction=(
                    "CURRENT PHASE: ACTION\n"
                    f"Allowed tools: {', '.join(allowed_tool_names)}.\n"
                    f"Charged retrieval budget: {session.budget_state.budget_limit} cache-miss actions.\n"
                    "Use conclude only when the review item list is ready.\n"
                    "The final step must be `conclude(review={\"review_items\": [...]})`.\n"
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
                ),
                system_prompt_override=system_prompt,
                max_steps_override=self.max_steps,
                llm_timeout_seconds=self.llm_timeout_seconds,
            )
        except Exception as exc:
            error = ReactReviewedStructuredOutputError(
                stage="reviewer",
                cycle_number=cycle_number,
                reviewer_role=self.reviewer_role,
                message=f"reviewer structured conclude failed after LLM start: {exc}",
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
                response_content=getattr(response, "content", None),
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
            raw_item.setdefault("review_id", f"{self.reviewer_role}_{index}")
            raw_item.setdefault("reviewer_role", self.reviewer_role)
            raw_item.setdefault("severity", "warning")
            raw_item.setdefault("anchor_kind", "global")
            raw_item.setdefault("flaw_type", "needs_manual_review")
            raw_item.setdefault("critique", "Reviewer output did not provide critique text.")
            raw_item.setdefault("required_action", "Re-check the anchored section.")
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
            if "recommendation" in normalized and "required_action" not in normalized:
                normalized["required_action"] = normalized.pop("recommendation")
            if "required_fix" in normalized and "required_action" not in normalized:
                normalized["required_action"] = normalized.pop("required_fix")
            if "category" in normalized and "flaw_type" not in normalized:
                normalized["flaw_type"] = normalized.pop("category")
            severity = str(normalized.get("severity") or "").strip().lower()
            if severity in severity_map:
                normalized["severity"] = severity_map[severity]
            normalized.setdefault("review_id", f"{self.reviewer_role}_{index}")
            normalized.setdefault("reviewer_role", self.reviewer_role)
            normalized.setdefault("severity", "warning")
            normalized.setdefault("flaw_type", "needs_manual_review")
            normalized.setdefault("critique", "Reviewer identified an issue that required salvage normalization.")
            normalized.setdefault("required_action", "Re-check the anchored section.")
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

    def _run_deterministic(
        self,
        *,
        workspace: ReactReviewedWorkspace,
        submission: AnswerSubmission,
        proposer_trajectory: ReActTrajectory,
        cycle_number: int,
        session: ReviewerSession,
    ) -> Tuple[List[ReviewItem], ReActTrajectory]:
        trajectory = ReActTrajectory(query=f"{self.reviewer_role} review: {submission.submission_id}")
        inspect_payload = workspace.inspect_submission_anchor()
        inspect_call_id = f"tc_{uuid.uuid4().hex[:8]}"
        trajectory.add_step(
            ReActStep(
                step_number=1,
                thought=f"Inspect the current submission from the {self.reviewer_role} angle.",
                action="inspect_submission_anchor",
                action_input={},
                observation=_json_preview(inspect_payload),
                tool_calls=[
                    ToolCallRecord(
                        tool_name="inspect_submission_anchor",
                        tool_call_id=inspect_call_id,
                        tool_args={},
                        observation=_json_preview(inspect_payload),
                        observation_data=inspect_payload,
                    )
                ],
                tool_call_id=inspect_call_id,
                observation_data=inspect_payload,
            )
        )
        items = self._deterministic_review_items(
            workspace=workspace,
            submission=submission,
            proposer_trajectory=proposer_trajectory,
            session=session,
        )[: self.max_items]
        conclude_call_id = f"tc_{uuid.uuid4().hex[:8]}"
        trajectory.add_step(
            ReActStep(
                step_number=2,
                thought="Return structured review items only.",
                action="conclude",
                action_input={"reviewer_role": self.reviewer_role},
                observation=_json_preview([item.model_dump(exclude_none=True) for item in items]),
                tool_calls=[
                    ToolCallRecord(
                        tool_name="conclude",
                        tool_call_id=conclude_call_id,
                        tool_args={"reviewer_role": self.reviewer_role},
                        observation=_json_preview([item.model_dump(exclude_none=True) for item in items]),
                        observation_data=[item.model_dump(exclude_none=True) for item in items],
                    )
                ],
                tool_call_id=conclude_call_id,
                observation_data=[item.model_dump(exclude_none=True) for item in items],
            )
        )
        trajectory.finalize(json.dumps({"review_item_count": len(items)}, ensure_ascii=False))
        return items, trajectory

    def _deterministic_review_items(
        self,
        *,
        workspace: ReactReviewedWorkspace,
        submission: AnswerSubmission,
        proposer_trajectory: ReActTrajectory,
        session: ReviewerSession,
    ) -> List[ReviewItem]:
        section_lookup = {section.section_id: section for section in submission.sections}
        items: List[ReviewItem] = []
        if self.reviewer_role == "search_coverage":
            for answer_section in workspace.task_spec.answer_sections:
                if answer_section.required and answer_section.section_id not in section_lookup:
                    items.append(
                        ReviewItem(
                            review_id=f"{self.reviewer_role}_{answer_section.section_id}",
                            reviewer_role=self.reviewer_role,
                            anchor_kind="missing_section",
                            severity="blocking",
                            flaw_type="missing_required_section",
                            critique=f"Required TaskSpec section '{answer_section.section_id}' is missing.",
                            required_action="Add the missing required section before acceptance.",
                            target_section_id=answer_section.section_id,
                        )
                    )
            for section in submission.sections:
                if not section.citation_ids:
                    items.append(
                        ReviewItem(
                            review_id=f"{self.reviewer_role}_{section.section_id}",
                            reviewer_role=self.reviewer_role,
                            anchor_kind="section_only",
                            severity="warning",
                            flaw_type="thin_search_coverage",
                            critique="This section has no attached citations from the retrieval pass.",
                            required_action="Either attach evidence-backed citations or narrow the section claim.",
                            target_section_id=section.section_id,
                        )
                    )
        elif self.reviewer_role == "evidence_trace":
            citation_ids = {citation.citation_id for citation in submission.citations}
            for section in submission.sections:
                if any(citation_id not in citation_ids for citation_id in section.citation_ids):
                    items.append(
                        ReviewItem(
                            review_id=f"{self.reviewer_role}_{section.section_id}",
                            reviewer_role=self.reviewer_role,
                            anchor_kind="section_only",
                            severity="blocking",
                            flaw_type="dangling_citation_ref",
                            critique="Section references a citation id that is not present in the submission catalog.",
                            required_action="Fix citation ids so each section maps to a real catalog entry.",
                            target_section_id=section.section_id,
                        )
                    )
                if not section.step_refs:
                    items.append(
                        ReviewItem(
                            review_id=f"{self.reviewer_role}_{section.section_id}_step",
                            reviewer_role=self.reviewer_role,
                            anchor_kind="section_only",
                            severity="blocking",
                            flaw_type="missing_step_ref",
                            critique="Section content is not linked back to a proposer trajectory step.",
                            required_action="Add at least one step reference for this section.",
                            target_section_id=section.section_id,
                        )
                    )
        elif self.reviewer_role == "reasoning_consistency":
            for section in submission.sections:
                if not _compact_text(section.content):
                    items.append(
                        ReviewItem(
                            review_id=f"{self.reviewer_role}_{section.section_id}",
                            reviewer_role=self.reviewer_role,
                            anchor_kind="section_only",
                            severity="blocking",
                            flaw_type="empty_section",
                            critique="Section content is empty after submission assembly.",
                            required_action="Replace the empty section with a conservative evidence-backed statement.",
                            target_section_id=section.section_id,
                        )
                    )
                elif section.citation_ids and "insufficient" in section.content.lower():
                    items.append(
                        ReviewItem(
                            review_id=f"{self.reviewer_role}_{section.section_id}_scope",
                            reviewer_role=self.reviewer_role,
                            anchor_kind="section_only",
                            severity="note",
                            flaw_type="scope_tension",
                            critique="Section cites evidence but still uses strongly under-specified insufficiency language.",
                            required_action="Clarify whether the section is evidence-backed or explicitly unresolved.",
                            target_section_id=section.section_id,
                        )
                    )
        elif self.reviewer_role == "counterevidence":
            with workspace._state_lock:
                query_plan_items = list(workspace.query_plans.items())
            contrarian_plan = next(
                (
                    query_plan_id
                    for query_plan_id, query_plan in query_plan_items
                    if query_plan.lane == "contrarian"
                ),
                None,
            )
            if contrarian_plan:
                try:
                    result = workspace.search_papers(
                        query_plan_id=contrarian_plan,
                        reason="counterevidence reviewer",
                        artifact_store=session.artifact_store,
                        session=session,
                        charge_budget=True,
                        requested_via="search_papers",
                        write_snapshot=False,
                    )
                except ReviewerBudgetBlocked:
                    result = []
                if result and len(submission.citations) <= 1:
                    items.append(
                        ReviewItem(
                            review_id=f"{self.reviewer_role}_global",
                            reviewer_role=self.reviewer_role,
                            anchor_kind="global",
                            severity="warning",
                            flaw_type="counterevidence_not_checked",
                            critique="A contrarian search returned additional papers, but the submission still cites only a narrow support set.",
                            required_action="Check whether contrary or boundary-condition evidence should be reflected in limitations.",
                            evidence_refs=[str(result[0].get("paper_id") or "")],
                        )
                    )
        return items

    def _build_system_prompt(self) -> str:
        role_notes = {
            "search_coverage": "Check for missing search directions, controls, and missing sections.",
            "evidence_trace": "Check whether citations, evidence refs, and step refs are traceable and real.",
            "reasoning_consistency": "Check for reasoning jumps, scope drift, and inconsistent section claims.",
            "counterevidence": "Use limited new retrieval only to search for counterevidence or boundary conditions.",
        }
        return (
            f"You are ReviewerAgent[{self.reviewer_role}] in a chemistry QA workflow.\n"
            f"{role_notes[self.reviewer_role]}\n"
            "Review only the proposer submission and proposer trajectory.\n"
            "Do not challenge RouterAgent or EntityResolverAgent outputs.\n"
            f"You have at most {self.max_retrieval_actions} charged retrieval cache-miss actions.\n"
            "Return ReviewItem objects only by calling conclude.\n"
            "The final tool call must be `conclude(review={\"review_items\": [...]})`.\n"
            "The only valid conclude argument name is `review`.\n"
        )

    def _build_user_prompt(
        self,
        *,
        submission: AnswerSubmission,
        proposer_trajectory: ReActTrajectory,
        cycle_number: int,
    ) -> str:
        return _json_preview(
            {
                "cycle_number": cycle_number,
                "reviewer_role": self.reviewer_role,
                "submission": submission.model_dump(exclude_none=True),
                "proposer_trajectory": proposer_trajectory.to_dict(),
            },
            limit=12000,
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
            max_steps_initial=react_config.get("max_propose_steps_initial", 6),
            max_steps_revision=react_config.get("max_propose_steps_revision", 4),
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

        for cycle_number in range(1, max_review_cycles + 1):
            workspace.set_review_context(
                submission=None,
                proposer_trajectory=None,
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
            "qa_result": str(store.path("qa_result.json")),
        }
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
        for item in items:
            current = item
            if item.anchor_kind == "step_section" and item.target_trajectory_id != proposer_trajectory.trajectory_id:
                current = item.model_copy(update={"target_trajectory_id": proposer_trajectory.trajectory_id})
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
