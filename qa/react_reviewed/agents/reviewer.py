import copy
import json
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple

from qa.react_reviewed.common import (
    AgentResponse,
    AnswerSubmission,
    REVIEWER_TOOL_NAMES,
    ReActAgent,
    ReActTrajectory,
    ReactReviewedReviewerExecutionError,
    ReactReviewedStructuredOutputError,
    ReviewItem,
    ReviewerBudgetBlocked,
    ReviewerRole,
    ReviewerRunStatus,
    ReviewerSession,
    ToolResult,
    _compact_text,
    _extract_json_candidates,
    _extract_stage_payload,
    _json_preview,
    _lazy_structured_tool_import,
    _store_invalid_llm_output,
    _store_reviewer_execution_failure,
    _tool_plain_payload,
    build_chat_model_from_config,
    build_review_prompt_contract,
    build_reviewer_action_prompt,
    build_reviewer_repair_system_prompt,
    build_reviewer_system_prompt,
    build_reviewer_thought_prompt,
    build_reviewer_user_prompt,
    describe_chat_model_config,
    invoke_llm,
    logger,
    tool_schemas,
)
from qa.react_reviewed.memory.workspace import ReactReviewedWorkspace

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
        repair_attempts: int = 1,
    ) -> None:
        self.reviewer_role = reviewer_role
        self.model_config = dict(model_config or {})
        self.max_steps = max(1, int(max_steps))
        self.max_items = max(1, int(max_items))
        self.max_retrieval_actions = max(0, int(max_retrieval_actions))
        self.llm_timeout_seconds = float(llm_timeout_seconds)
        self.repair_attempts = max(0, int(repair_attempts))

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
            """Read indexed sections from a parsed paper within reviewer budget rules."""
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
            query_plan_ids: Optional[List[str]] = None,
            query_text: Optional[str] = None,
            query_texts: Optional[List[str]] = None,
            lane: str = "contrarian",
            reason: str = "",
        ) -> ToolResult:
            """Search external literature providers within reviewer role permissions."""
            return _budget_safe(
                lambda: workspace.search_papers_batch(
                    query_plan_id=query_plan_id,
                    query_plan_ids=query_plan_ids,
                    query_text=query_text,
                    query_texts=query_texts,
                    lane=lane,
                    reason=reason,
                    artifact_store=session.artifact_store,
                    session=session,
                    charge_budget=True,
                    requested_via="search_papers",
                    write_snapshot=False,
                )
            )

        def download_document(
            paper_id: Optional[str] = None,
            paper_ids: Optional[List[str]] = None,
        ) -> ToolResult:
            """Download paper artifacts within reviewer budget rules."""
            return _budget_safe(
                lambda: workspace.download_documents(
                    paper_id=paper_id,
                    paper_ids=paper_ids,
                    artifact_store=session.artifact_store,
                    session=session,
                    charge_budget=True,
                    requested_via="download_document",
                    write_snapshot=False,
                )
            )

        def parse_document(paper_id: str) -> ToolResult:
            """Parse a downloaded paper into indexed sections within reviewer budget rules."""
            return _budget_safe(
                lambda: workspace.parse_document(
                    paper_id=paper_id,
                    artifact_store=session.artifact_store,
                    session=session,
                    charge_budget=True,
                    requested_via="parse_document",
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
            "download_document": download_document,
            "parse_document": parse_document,
            "fetch_citation_context": fetch_citation_context,
            "extract_evidence": extract_evidence,
            "conclude": conclude,
        }
        tool_args_schemas: Dict[str, Any] = {
            "inspect_entity_cache": tool_schemas.InspectEntityCacheToolInput,
            "inspect_submission_anchor": tool_schemas.InspectSubmissionAnchorToolInput,
            "read_sections": tool_schemas.SectionAccessToolInput,
            "search_papers": tool_schemas.ReviewerSearchPapersToolInput,
            "download_document": tool_schemas.DownloadDocumentToolInput,
            "parse_document": tool_schemas.ParseDocumentToolInput,
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
            if self.repair_attempts > 0:
                repaired_items = self._repair_review_items_with_llm(
                    submission=submission,
                    proposer_trajectory=proposer_trajectory,
                    cycle_number=cycle_number,
                    session=session,
                    error=error,
                    trajectory=partial_trajectory,
                )
                logger.warning(
                    "react_reviewed_reviewer_payload_repaired role=%s cycle=%s count=%s",
                    self.reviewer_role,
                    cycle_number,
                    len(repaired_items),
                )
                return repaired_items, partial_trajectory, "salvaged"
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
            if self.repair_attempts > 0:
                repaired_items = self._repair_review_items_with_llm(
                    submission=submission,
                    proposer_trajectory=proposer_trajectory,
                    cycle_number=cycle_number,
                    session=session,
                    error=error,
                    trajectory=trajectory,
                )
                logger.warning(
                    "react_reviewed_reviewer_payload_repaired role=%s cycle=%s count=%s",
                    self.reviewer_role,
                    cycle_number,
                    len(repaired_items),
                )
                return repaired_items, trajectory, "salvaged"
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

    def _repair_review_items_with_llm(
        self,
        *,
        submission: AnswerSubmission,
        proposer_trajectory: ReActTrajectory,
        cycle_number: int,
        session: ReviewerSession,
        error: ReactReviewedStructuredOutputError,
        trajectory: Optional[ReActTrajectory],
    ) -> List[ReviewItem]:
        conclude_call_contract = self._build_review_prompt_contract(
            submission=submission,
            proposer_trajectory=proposer_trajectory,
        )
        last_error: ReactReviewedStructuredOutputError = error
        for attempt_number in range(1, self.repair_attempts + 1):
            try:
                llm = build_chat_model_from_config(self.model_config)
                raw_response = invoke_llm(
                    llm,
                    [
                        {
                            "role": "system",
                            "content": build_reviewer_repair_system_prompt(
                                conclude_contract=conclude_call_contract,
                            ),
                        },
                        {
                            "role": "user",
                            "content": _json_preview(
                                {
                                    "cycle_number": cycle_number,
                                    "repair_attempt": attempt_number,
                                    "reviewer_role": self.reviewer_role,
                                    "submission": submission.model_dump(exclude_none=True),
                                    "proposer_trajectory": proposer_trajectory.to_dict(),
                                    "conclude_call_contract": conclude_call_contract,
                                    "validation_error": str(last_error),
                                    "invalid_review_payload": last_error.structured_output,
                                    "invalid_response_content": last_error.response_content,
                                },
                                limit=12000,
                            ),
                        },
                    ],
                )
            except Exception as exc:
                execution_error = ReactReviewedReviewerExecutionError(
                    stage="reviewer_repair",
                    cycle_number=cycle_number,
                    reviewer_role=self.reviewer_role,
                    message=f"Reviewer repair attempt failed to execute: {exc}",
                    details={"attempt": attempt_number},
                    response_content=getattr(exc, "response_content", None),
                    structured_output=getattr(exc, "structured_output", None),
                    trajectory=trajectory,
                )
                _store_reviewer_execution_failure(
                    artifact_store=session.artifact_store,
                    prefix=f"{self.reviewer_role}_cycle_{cycle_number}_repair_{attempt_number}",
                    error=execution_error,
                )
                raise execution_error
            try:
                raw_items = self._parse_review_items_response(response=AgentResponse(content=raw_response))
                return [ReviewItem.model_validate(item) for item in list(raw_items or [])][: self.max_items]
            except Exception as exc:
                salvage_sources = [
                    SimpleNamespace(content=raw_response, structured_output=None, response_content=raw_response),
                    SimpleNamespace(
                        content=raw_response,
                        structured_output=last_error.structured_output,
                        response_content=last_error.response_content,
                    ),
                ]
                for salvage_source in salvage_sources:
                    salvaged_items = self._salvage_review_payload(
                        response=salvage_source,
                        trajectory=trajectory,
                        proposer_trajectory=proposer_trajectory,
                        submission=submission,
                        max_items=self.max_items,
                    )
                    if salvaged_items:
                        logger.warning(
                            "react_reviewed_reviewer_repair_salvaged role=%s cycle=%s attempt=%s count=%s",
                            self.reviewer_role,
                            cycle_number,
                            attempt_number,
                            len(salvaged_items),
                        )
                        return salvaged_items
                last_error = ReactReviewedStructuredOutputError(
                    stage="reviewer_repair",
                    cycle_number=cycle_number,
                    reviewer_role=self.reviewer_role,
                    message=f"invalid reviewer repair output: {exc}",
                    response_content=raw_response,
                    structured_output=None,
                    trajectory=trajectory,
                )
                _store_invalid_llm_output(
                    artifact_store=session.artifact_store,
                    prefix=f"{self.reviewer_role}_cycle_{cycle_number}_repair_{attempt_number}",
                    error=last_error,
                )
        raise last_error

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

