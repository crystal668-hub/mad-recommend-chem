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
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import parse_qs, unquote, urlparse

from agents.chat_models import build_chat_model_from_config, describe_chat_model_config
from qa.react_reviewed.tools import schemas as tool_schemas
from agents.react_agent import AgentResponse, ReActAgent, ToolResult
from agents.react_reasoning import ReActStep, ReActTrajectory, ToolCallRecord
from pydantic import Field, create_model
from prompts.react_reviewed import (
    build_proposer_action_prompt,
    build_proposer_repair_system_prompt,
    build_proposer_system_prompt,
    build_proposer_thought_prompt,
    build_proposer_user_prompt,
    build_review_prompt_contract,
    build_reviewer_action_prompt,
    build_reviewer_repair_system_prompt,
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
from qa.paper_profiles import GrobidPaperProfileBuilder, extract_profile_xml_segments, write_profile_failure
from qa.nodes.document_acquirer import DocumentAcquirerNode
from qa.nodes.entity_resolver import EntityResolverNode
from qa.nodes.query_planner import QueryPlannerExecutionError, QueryPlannerNode
from qa.nodes.retriever import RetrieverNode
from qa.nodes.router import RouterExecutionError, RouterNode
from qa.providers import PdfUrlProbeClient
from qa.retrieval_utils import normalize_doi
from qa.react_reviewed.models import (
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
    PaperProfile,
    PaperRecord,
    QueryPlan,
    RetrievalDiagnosticRecord,
    Section,
    SectionIndex,
    SectionTextView,
)
from qa.state import EntityPack, TaskSpec
from qa.synthesis_state import AnswerSectionOutput, CitationRecord, ConfidenceRating, QAResult, SectionConfidenceRecord

if TYPE_CHECKING:
    from qa.react_reviewed.memory.workspace import ReactReviewedWorkspace


logger = logging.getLogger("MAD.qa.react_reviewed")

MIN_REACT_REVIEWED_PROPOSER_STEPS = 6
PROPOSER_CANDIDATE_TARGET = 10
PROPOSER_RERANK_TOP_K = 5
PROPOSER_PDF_PROBE_MAX_CANDIDATES = 20

PROPOSER_TOOL_NAMES = (
    "plan_queries",
    "search_papers",
    "download_document",
    "screen_papers",
    "parse_document",
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
        "download_document",
        "parse_document",
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
        "download_document",
        "parse_document",
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


def _paper_title_fallback(workspace: ReactReviewedWorkspace, paper_id: str) -> Optional[str]:
    record = workspace.paper_records.get(str(paper_id or "").strip())
    if record is not None and _compact_text(getattr(record, "title", None)):
        return _compact_text(getattr(record, "title", None))
    candidate = workspace.paper_candidates.get(str(paper_id or "").strip())
    if candidate is not None and _compact_text(getattr(candidate, "title", None)):
        return _compact_text(getattr(candidate, "title", None))
    return None


def _url_has_pdf_signal(value: Any) -> bool:
    text = _compact_text(value)
    if not text:
        return False
    parsed = urlparse(text)
    path = unquote(parsed.path or "").lower()
    if path.endswith(".pdf") or path.endswith("/pdf") or path.endswith("_pdf"):
        return True
    if ".pdf?" in text.lower() or ".pdf#" in text.lower():
        return True
    for key, values in parse_qs(parsed.query, keep_blank_values=True).items():
        lowered_key = str(key or "").strip().lower()
        lowered_values = [str(item or "").strip().lower() for item in list(values or [])]
        if any(".pdf" in item for item in lowered_values):
            return True
        if lowered_key in {"format", "type", "mime"} and any(
            item in {"pdf", "application/pdf"} for item in lowered_values
        ):
            return True
    return False


def _candidate_has_downloadable_pdf_signal(candidate: Optional[PaperCandidate]) -> bool:
    if candidate is None:
        return False
    return bool(
        _url_has_pdf_signal(getattr(candidate, "best_oa_pdf_url", None))
        or _url_has_pdf_signal(getattr(candidate, "oa_url", None))
    )


def _url_is_strict_pdf_download(value: Any) -> bool:
    text = _compact_text(value)
    if not text:
        return False
    parsed = urlparse(text)
    if parsed.query or parsed.fragment:
        return False
    return unquote(parsed.path or "").lower().endswith(".pdf")


def _candidate_has_strict_proposer_pdf_signal(candidate: Optional[PaperCandidate]) -> bool:
    if candidate is None:
        return False
    return bool(
        _compact_text(getattr(candidate, "doi", None))
        and _url_is_strict_pdf_download(getattr(candidate, "open_access_pdf_url", None))
    )


def _candidate_has_proposer_pdf_probe_input(candidate: Optional[PaperCandidate]) -> bool:
    if candidate is None:
        return False
    return bool(
        _compact_text(getattr(candidate, "doi", None))
        and _compact_text(getattr(candidate, "open_access_pdf_url", None))
    )


def _pdf_probe_verdict_rank(verdict: Any) -> int:
    normalized = _compact_text(verdict).lower()
    if normalized == "strong":
        return 2
    if normalized == "weak":
        return 1
    return 0


def _batched_search_stage_timeout(base_timeout_seconds: float, batch_size: int) -> float:
    normalized_base = max(0.01, float(base_timeout_seconds or 0.01))
    normalized_batch_size = max(1, int(batch_size or 1))
    if normalized_batch_size <= 1:
        return normalized_base
    return min(240.0, normalized_base + (20.0 * float(normalized_batch_size - 1)))


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


def _query_hits_in_text(task_spec: TaskSpec, entity_pack: EntityPack, text: Optional[str]) -> List[str]:
    normalized_text = _compact_text(text).lower()
    if not normalized_text:
        return []
    return [term for term in _screen_query_terms(task_spec, entity_pack) if term and term in normalized_text]


def _score_proposer_semantic_scholar_candidate(
    *,
    task_spec: TaskSpec,
    entity_pack: EntityPack,
    candidate: PaperCandidate,
) -> float:
    semantic_summary = _compact_text(
        " ".join(
            [
                str(candidate.abstract or ""),
                str(candidate.tldr or ""),
                " ".join(str(item) for item in list(candidate.fields_of_study or [])),
            ]
        )
    )
    summary_metrics = _paper_relevance_metrics(
        task_spec=task_spec,
        entity_pack=entity_pack,
        title=candidate.title,
        abstract=semantic_summary,
    )
    tldr_hits = len(_query_hits_in_text(task_spec, entity_pack, candidate.tldr))
    field_hits = len(
        _query_hits_in_text(task_spec, entity_pack, " ".join(str(item) for item in list(candidate.fields_of_study or [])))
    )
    citation_count = int(candidate.ranking_features.get("citation_count") or 0)
    citation_bonus = min(2.0, float(citation_count) / 100.0)
    open_access_bonus = 1.0 if bool(candidate.is_open_access) else 0.0
    return round(
        4.0 * float(summary_metrics["title_hits"])
        + 3.0 * float(summary_metrics["abstract_hits"])
        + 1.5 * float(summary_metrics["question_hits"])
        + 2.0 * float(min(tldr_hits, 3))
        + 0.5 * float(min(field_hits, 4))
        + citation_bonus
        + open_access_bonus,
        4,
    )


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


def _extract_unpaywall_pdf_url(payload: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(payload, dict):
        return None
    best_location = dict(payload.get("best_oa_location") or {})
    best_pdf_url = _compact_text(best_location.get("url_for_pdf"))
    if best_pdf_url:
        return best_pdf_url
    for item in list(payload.get("oa_locations") or []):
        if not isinstance(item, dict):
            continue
        pdf_url = _compact_text(item.get("url_for_pdf"))
        if pdf_url:
            return pdf_url
    return None


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
    profile_ready_paper_ids: set[str] = field(default_factory=set)
    evidence_ids: set[str] = field(default_factory=set)
    section_ids_by_paper: Dict[str, set[str]] = field(default_factory=dict)
    evidence_ids_by_paper: Dict[str, set[str]] = field(default_factory=dict)
    evidence_layers_by_id: Dict[str, str] = field(default_factory=dict)
    fulltext_status_by_paper: Dict[str, str] = field(default_factory=dict)
    fulltext_available_by_paper: Dict[str, bool] = field(default_factory=dict)
    search_generation: int = 0
    acquisition_generation: int = 0
    screening_generation: int = 0
    locked_candidate_paper_ids: List[str] = field(default_factory=list)
    dropped_candidate_paper_ids: List[str] = field(default_factory=list)
    candidate_screening: List[Dict[str, Any]] = field(default_factory=list)
    candidate_profiles: List[Dict[str, Any]] = field(default_factory=list)
    latest_screen_status: str = ""
    latest_screen_failure_domain: str = ""
    screening_input_exhaustions: int = 0

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
        self.locked_candidate_paper_ids = []
        self.dropped_candidate_paper_ids = []
        self.candidate_screening = []
        self.candidate_profiles = []
        self.profile_ready_paper_ids = set()

    def record_screening(self, payload: Dict[str, Any]) -> None:
        self.screening_generation = self.acquisition_generation
        self.latest_screen_status = str((payload or {}).get("screen_status") or "").strip()
        self.latest_screen_failure_domain = str((payload or {}).get("failure_domain") or "").strip()
        if self.latest_screen_status == "input_exhausted":
            self.screening_input_exhaustions += 1
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
        self.candidate_profiles = [
            copy.deepcopy(item)
            for item in list((payload or {}).get("paper_profiles") or [])
            if isinstance(item, dict)
        ]
        self.profile_ready_paper_ids = {
            str(item.get("paper_id") or "").strip()
            for item in self.candidate_profiles
            if str(item.get("paper_id") or "").strip() and str(item.get("profile_status") or "").strip().lower() == "ready"
        }
        for item in list((payload or {}).get("indexed_papers") or []):
            if not isinstance(item, dict):
                continue
            paper_id = str(item.get("paper_id") or "").strip()
            if not paper_id:
                continue
            self.fulltext_status_by_paper[paper_id] = str(item.get("fulltext_status") or "").strip()
            self.fulltext_available_by_paper[paper_id] = bool(item.get("fulltext_available"))

    def record_acquisition(self, payload: Dict[str, Any]) -> None:
        payload_dict = dict(payload or {})
        document_payloads = list(payload_dict.get("documents") or [])
        if document_payloads:
            for item in document_payloads:
                self.record_acquisition(dict(item or {}))
            return
        paper_id = str(payload_dict.get("paper_id") or "").strip()
        if not paper_id:
            return
        self.acquired_paper_ids.add(paper_id)
        self.acquisition_generation = self.search_generation
        self.fulltext_status_by_paper[paper_id] = str(payload_dict.get("fulltext_status") or "").strip()
        self.fulltext_available_by_paper[paper_id] = bool(payload_dict.get("fulltext_available"))

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
        return bool(self.acquired_paper_ids) and self.screening_generation < self.acquisition_generation

    def fulltext_evidence_ids_for_paper(self, paper_id: str) -> set[str]:
        return {
            evidence_id
            for evidence_id in self.evidence_ids_by_paper.get(str(paper_id or "").strip(), set())
            if self.evidence_layers_by_id.get(evidence_id) == "fulltext"
        }

    def parsed_paper_ids(self) -> List[str]:
        return sorted(
            paper_id
            for paper_id, status in self.fulltext_status_by_paper.items()
            if str(status or "").strip().lower() == "fulltext_indexed"
        )

    def locked_parsed_paper_ids(self) -> List[str]:
        parsed = set(self.parsed_paper_ids())
        return [paper_id for paper_id in self.locked_candidate_paper_ids if paper_id in parsed]

    def locked_evidence_paper_ids(self) -> List[str]:
        return [
            paper_id
            for paper_id in self.locked_candidate_paper_ids
            if bool(self.evidence_ids_by_paper.get(paper_id))
        ]

    def prompt_payload(self) -> Dict[str, Any]:
        return {
            "evidence_policy": self.evidence_policy,
            "query_plan_ids": list(self.query_plan_ids),
            "search_generation": self.search_generation,
            "acquisition_generation": self.acquisition_generation,
            "screening_generation": self.screening_generation,
            "searched_paper_ids": sorted(self.searched_paper_ids),
            "search_ordered_paper_ids": list(self.search_ordered_paper_ids),
            "profile_ready_paper_ids": sorted(self.profile_ready_paper_ids),
            "locked_candidate_paper_ids": list(self.locked_candidate_paper_ids),
            "dropped_candidate_paper_ids": list(self.dropped_candidate_paper_ids),
            "acquired_paper_ids": sorted(self.acquired_paper_ids),
            "parsed_paper_ids": self.parsed_paper_ids(),
            "locked_parsed_paper_ids": self.locked_parsed_paper_ids(),
            "locked_evidence_paper_ids": self.locked_evidence_paper_ids(),
            "evidence_ids": sorted(self.evidence_ids),
            "candidate_screening": copy.deepcopy(self.candidate_screening),
            "candidate_profiles": copy.deepcopy(self.candidate_profiles),
            "latest_screen_status": self.latest_screen_status,
            "latest_screen_failure_domain": self.latest_screen_failure_domain,
            "screening_input_exhaustions": self.screening_input_exhaustions,
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


__all__ = [name for name in globals() if not name.startswith("__")]
