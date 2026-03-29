import copy
import json
import re
import time
from concurrent.futures import Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from qa.react_reviewed.common import (
    AcceptanceDecision,
    AnswerSubmission,
    DEFAULT_REVIEWER_BUDGET_BY_ROLE,
    DEFAULT_REVIEWER_ROLES,
    DocumentAcquirerNode,
    EntityPack,
    EntityResolverNode,
    EvidenceExtractor,
    EvidenceExtractorHandoff,
    GrobidPaperProfileBuilder,
    PROPOSER_CANDIDATE_TARGET,
    PROPOSER_PDF_PROBE_MAX_CANDIDATES,
    PROPOSER_RERANK_TOP_K,
    QAArtifactStore,
    QAResult,
    QueryPlannerExecutionError,
    QueryPlannerNode,
    REVIEWER_TOOL_NAMES,
    ReActTrajectory,
    RetrieverNode,
    ReviewCompletionStatus,
    ReviewItem,
    ReviewResponse,
    ReviewerBudgetState,
    ReviewerExecutionResult,
    ReviewerRole,
    ReviewerRunStatus,
    ReviewerSession,
    RouterNode,
    SubmissionCycleState,
    SubmissionSection,
    SubmissionStepRef,
    TaskSpec,
    _build_submission_trace,
    _compact_text,
    _confidence,
    _is_reviewer_llm_completed,
    _merge_unique_text,
    logger,
)
from qa.react_reviewed.agents.router_wrapper import RouterAgentWrapper
from qa.react_reviewed.agents.entity_wrapper import EntityResolverAgentWrapper
from qa.react_reviewed.memory.workspace import ReactReviewedWorkspace
from qa.react_reviewed.agents.proposer import ReactReviewedProposerAgent
from qa.react_reviewed.agents.reviewer import ReactReviewedReviewerAgent
from qa.react_reviewed.agents.synthesizer import SubmissionSynthesizerAgent

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
        paper_profile_builder: Optional[GrobidPaperProfileBuilder] = None,
        proposer_model_config: Optional[Dict[str, Any]] = None,
        reviewer_model_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        proposer_candidate_target: int = PROPOSER_CANDIDATE_TARGET,
        proposer_rerank_top_k: int = PROPOSER_RERANK_TOP_K,
        pdf_probe_client: Optional[Any] = None,
        proposer_pdf_probe_enabled: bool = True,
        proposer_pdf_probe_max_candidates: int = PROPOSER_PDF_PROBE_MAX_CANDIDATES,
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
        self.paper_profile_builder = paper_profile_builder or GrobidPaperProfileBuilder()
        self.pdf_probe_client = pdf_probe_client
        self.grobid_preflight_enabled = bool(react_config.get("grobid_preflight_enabled", True))
        self._last_grobid_preflight_artifact_path: Optional[str] = None
        self.proposer_candidate_target = max(1, int(proposer_candidate_target))
        self.proposer_rerank_top_k = max(
            1,
            min(self.proposer_candidate_target, int(proposer_rerank_top_k)),
        )
        self.proposer_pdf_probe_enabled = bool(proposer_pdf_probe_enabled)
        self.proposer_pdf_probe_max_candidates = max(1, int(proposer_pdf_probe_max_candidates))
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
            proposer_candidate_target=self.proposer_candidate_target,
            proposer_rerank_top_k=self.proposer_rerank_top_k,
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
                repair_attempts=react_config.get("reviewer_repair_attempts", 1),
            )
            for reviewer_role in self.reviewer_role_order
        }
        self.synthesizer = SubmissionSynthesizerAgent(
            expose_candidate_submission_when_rejected=react_config.get(
                "expose_candidate_submission_when_rejected",
                False,
            )
        )

    def _run_grobid_preflight(
        self,
        *,
        store: QAArtifactStore,
        workspace: ReactReviewedWorkspace,
    ) -> Optional[str]:
        if not self.grobid_preflight_enabled:
            return None
        ensure_available = getattr(self.paper_profile_builder, "ensure_service_available", None)
        if not callable(ensure_available):
            return None
        try:
            payload = ensure_available()
            normalized_payload = dict(payload or {})
            normalized_payload.setdefault("status", "healthy")
            artifact_path = store.write_json("diagnostics/grobid_preflight.json", normalized_payload)
            self._last_grobid_preflight_artifact_path = artifact_path
            if normalized_payload.get("startup_attempted"):
                workspace._merge_execution_warnings(
                    [
                        "GROBID service was auto-started during react_reviewed preflight."
                    ]
                )
            return artifact_path
        except Exception as exc:
            payload = dict(getattr(self.paper_profile_builder, "last_preflight_payload", {}) or {})
            payload["status"] = "failure"
            payload["error"] = _compact_text(str(exc)) or "unknown grobid preflight failure"
            artifact_path = store.write_json("diagnostics/grobid_preflight.json", payload)
            self._last_grobid_preflight_artifact_path = artifact_path
            raise RuntimeError(payload["error"]) from exc

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
        self._last_grobid_preflight_artifact_path = None
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
            paper_profile_builder=self.paper_profile_builder,
            pdf_probe_client=self.pdf_probe_client,
            stage_watchdog_seconds=self.stage_watchdog_seconds,
            proposer_candidate_target=self.proposer_candidate_target,
            proposer_rerank_top_k=self.proposer_rerank_top_k,
            proposer_pdf_probe_enabled=self.proposer_pdf_probe_enabled,
            proposer_pdf_probe_max_candidates=self.proposer_pdf_probe_max_candidates,
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
        grobid_preflight_artifact_path: Optional[str] = None
        try:
            grobid_preflight_artifact_path = self._run_grobid_preflight(
                store=store,
                workspace=workspace,
            )
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
            "grobid_preflight": grobid_preflight_artifact_path or self._last_grobid_preflight_artifact_path,
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
