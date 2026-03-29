import copy
import json
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import Field, create_model

from qa.react_reviewed.common import (
    AgentResponse,
    AnswerSubmission,
    EvidenceItem,
    MIN_REACT_REVIEWED_PROPOSER_STEPS,
    PROPOSER_CANDIDATE_TARGET,
    PROPOSER_RERANK_TOP_K,
    PROPOSER_TOOL_NAMES,
    PaperCandidate,
    PaperRecord,
    ReActAgent,
    ReActStep,
    ReActTrajectory,
    ReactReviewedProposerExecutionError,
    ReactReviewedStructuredOutputError,
    ReviewItem,
    SubmissionCitation,
    SubmissionSection,
    SubmissionStepRef,
    ToolCallRecord,
    ToolResult,
    _ProposerRunState,
    _align_step_refs_to_trajectory,
    _coerce_salvaged_submission_payload,
    _compact_text,
    _confidence,
    _evidence_item_priority_score,
    _extract_json_candidates,
    _extract_stage_payload,
    _json_preview,
    _lazy_structured_tool_import,
    _merge_unique_text,
    _normalize_confidence_payload,
    _normalize_list_payload,
    _paper_has_useful_abstract_support,
    _paper_title_fallback,
    _step_ref,
    _store_execution_failure,
    _store_invalid_llm_output,
    _tool_plain_payload,
    build_chat_model_from_config,
    build_proposer_action_prompt,
    build_proposer_repair_system_prompt,
    build_proposer_system_prompt,
    build_proposer_thought_prompt,
    build_proposer_user_prompt,
    build_screening_system_prompt,
    build_submission_prompt_contract,
    build_submission_prompt_scaffold,
    describe_chat_model_config,
    extract_profile_xml_segments,
    invoke_llm,
    logger,
    parse_json_payload,
    tool_schemas,
)
from qa.react_reviewed.memory.workspace import ReactReviewedWorkspace

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
        proposer_candidate_target: int = PROPOSER_CANDIDATE_TARGET,
        proposer_rerank_top_k: int = PROPOSER_RERANK_TOP_K,
    ) -> None:
        self.model_config = dict(model_config or {})
        self.max_steps_initial = max(1, int(max_steps_initial))
        self.max_steps_revision = max(1, int(max_steps_revision))
        if self.max_steps_initial < MIN_REACT_REVIEWED_PROPOSER_STEPS:
            raise ValueError(
                "ReactReviewed proposer max_steps_initial must be at least "
                f"{MIN_REACT_REVIEWED_PROPOSER_STEPS} so the required plan/search/download/screen/parse/extract/conclude "
                "tool chain remains feasible under deadline mode."
            )
        if self.max_steps_revision < MIN_REACT_REVIEWED_PROPOSER_STEPS:
            raise ValueError(
                "ReactReviewed proposer max_steps_revision must be at least "
                f"{MIN_REACT_REVIEWED_PROPOSER_STEPS} so revision cycles can still download, parse, extract evidence, and conclude "
                "before deadline-mode retrieval blocking starts."
            )
        self.llm_timeout_seconds = float(llm_timeout_seconds)
        self.fallback_mode = str(fallback_mode or "fail_fast_only").strip().lower() or "fail_fast_only"
        self.repair_attempts = max(0, int(repair_attempts))
        self.evidence_policy = str(evidence_policy or "prefer_fulltext").strip().lower() or "prefer_fulltext"
        self.proposer_candidate_target = max(1, int(proposer_candidate_target))
        self.proposer_rerank_top_k = max(
            1,
            min(self.proposer_candidate_target, int(proposer_rerank_top_k)),
        )

    def _build_screen_candidate_payload(
        self,
        *,
        candidate: PaperCandidate,
        profile_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        profile_xml_artifact_path = str(profile_payload.get("profile_xml_artifact_path") or "").strip()
        xml_segments = extract_profile_xml_segments(profile_xml_artifact_path)
        return {
            "paper_id": candidate.paper_id,
            "title": candidate.title,
            "doi": candidate.doi,
            "year": candidate.year,
            "venue": candidate.venue,
            "retrieval_score": round(float(candidate.retrieval_score or 0.0), 4),
            "profile_status": profile_payload.get("profile_status"),
            "profile_xml_artifact_path": profile_xml_artifact_path,
            "semantic_scholar_metadata": {
                "abstract": candidate.abstract,
                "tldr": candidate.tldr,
                "fields_of_study": list(candidate.fields_of_study or []),
                "is_open_access": candidate.is_open_access,
                "open_access_pdf_url": candidate.open_access_pdf_url,
            },
            "profile_header_text": xml_segments["header_text"],
            "profile_body_text": xml_segments["body_text"],
        }

    def _precheck_downloaded_pdf_candidate(
        self,
        *,
        workspace: ReactReviewedWorkspace,
        paper_id: str,
    ) -> Dict[str, Any]:
        paper_record = workspace.paper_records.get(str(paper_id or "").strip())
        source_artifact_path = str(getattr(paper_record, "source_artifact_path", "") or "").strip()
        if not source_artifact_path or not source_artifact_path.lower().endswith(".pdf") or not Path(source_artifact_path).exists():
            return {
                "status": "failed",
                "reason": "missing_local_pdf_artifact",
                "message": "Downloaded candidate does not have a readable local PDF artifact.",
            }
        pdf_extractor = getattr(workspace.document_acquirer, "pdf_extractor", None)
        if pdf_extractor is None or not all(
            hasattr(pdf_extractor, attr)
            for attr in ("_is_true_pdf", "_extract_with_pymupdf", "_should_salvage_attempt")
        ):
            return {
                "status": "skipped",
                "reason": "precheck_backend_unavailable",
            }
        pdf_bytes = Path(source_artifact_path).read_bytes()
        if not bool(pdf_extractor._is_true_pdf(pdf_bytes)):  # type: ignore[attr-defined]
            return {
                "status": "failed",
                "reason": "invalid_pdf_header",
                "message": "Downloaded content did not match a valid PDF header.",
            }
        attempt = pdf_extractor._extract_with_pymupdf(pdf_bytes=pdf_bytes)  # type: ignore[attr-defined]
        if not getattr(attempt, "succeeded", False):
            return {
                "status": "failed",
                "reason": "pymupdf_open_failed",
                "message": str(getattr(attempt, "failure_reason", "") or "PyMuPDF could not open the downloaded PDF."),
            }
        metrics = dict(getattr(attempt, "metrics", {}) or {})
        if getattr(attempt, "usable", False) or bool(pdf_extractor._should_salvage_attempt(attempt)):  # type: ignore[attr-defined]
            return {
                "status": "passed",
                "reason": "ok",
                "metrics": metrics,
            }
        failure_reasons = list(metrics.get("reasons") or []) or ["pdf_quality_gate_failed"]
        return {
            "status": "failed",
            "reason": str(failure_reasons[0]),
            "message": "PDF precheck rejected the downloaded artifact before profile generation.",
            "metrics": metrics,
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
                        limit=24000,
                    ),
                },
            ],
        )
        parsed = parse_json_payload(raw_response)
        if not isinstance(parsed, dict):
            return None, raw_response
        allowed_ids = {
            str(item.get("paper_id") or "").strip()
            for item in ranked_candidates
            if str(item.get("paper_id") or "").strip()
        }
        decision_map: Dict[str, Dict[str, Any]] = {}
        decision_order: List[str] = []
        for item in list(parsed.get("decisions") or []):
            if not isinstance(item, dict):
                continue
            paper_id = str(item.get("paper_id") or "").strip()
            decision = str(item.get("decision") or "").strip().lower()
            reason = _compact_text(item.get("reason"))
            if paper_id not in allowed_ids or decision not in {"lock", "drop"} or not reason:
                continue
            current = {"paper_id": paper_id, "decision": decision, "reason": reason}
            if paper_id not in decision_map:
                decision_order.append(paper_id)
            decision_map[paper_id] = current
        if not decision_map:
            return None, raw_response
        locked_paper_ids: List[str] = []
        dropped_paper_ids: List[str] = []
        ranked_payload: List[Dict[str, Any]] = []
        candidate_map = {
            str(item.get("paper_id") or "").strip(): item
            for item in ranked_candidates
            if str(item.get("paper_id") or "").strip()
        }
        ordered_rank_ids = decision_order + [
            paper_id
            for paper_id in candidate_map
            if paper_id not in set(decision_order)
        ]
        for paper_id in ordered_rank_ids:
            item = candidate_map[paper_id]
            llm_decision = decision_map.get(paper_id)
            decision = llm_decision["decision"] if llm_decision else "drop"
            reason = llm_decision["reason"] if llm_decision else "Candidate was not selected by listwise reranking."
            current = copy.deepcopy(item)
            current["decision"] = decision
            current["reason"] = reason
            ranked_payload.append(current)
            if decision == "lock" and paper_id not in locked_paper_ids and len(locked_paper_ids) < max_candidates:
                locked_paper_ids.append(paper_id)
            if decision != "lock" and paper_id not in dropped_paper_ids:
                dropped_paper_ids.append(paper_id)
        return (
            {
                "locked_paper_ids": locked_paper_ids,
                "dropped_paper_ids": dropped_paper_ids,
                "ranked_candidates": ranked_payload,
                "screen_status": "ready" if locked_paper_ids else "no_locks",
                "failure_domain": "" if locked_paper_ids else "topic_mismatch",
                "retryable": not bool(locked_paper_ids),
                "message": (
                    "candidate screening selected at least one candidate"
                    if locked_paper_ids
                    else "candidate screening did not lock any candidates"
                ),
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
        fail_on_no_usable: bool = True,
    ) -> Dict[str, Any]:
        ordered_ids: List[str] = []
        seen = set()
        for paper_id in paper_ids:
            normalized = str(paper_id or "").strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            ordered_ids.append(normalized)
        profile_payloads: List[Dict[str, Any]] = []
        dropped_candidates: List[Dict[str, Any]] = []
        ranked_candidates: List[Dict[str, Any]] = []

        for paper_id in ordered_ids:
            paper_record = workspace.paper_records.get(paper_id)
            candidate = workspace.paper_candidates.get(paper_id)
            if candidate is None or paper_record is None:
                continue
            precheck_payload = self._precheck_downloaded_pdf_candidate(
                workspace=workspace,
                paper_id=paper_id,
            )
            if str(precheck_payload.get("status") or "").strip().lower() == "failed":
                dropped_candidates.append(
                    {
                        "paper_id": paper_id,
                        "title": candidate.title,
                        "decision": "drop",
                        "reason": _compact_text(precheck_payload.get("message"))
                        or "PDF precheck rejected the downloaded artifact before profile generation.",
                        "profile_status": "pdf_precheck_failed",
                        "profile_xml_artifact_path": None,
                        "pdf_precheck": copy.deepcopy(precheck_payload),
                    }
                )
                continue
            profile_payload = workspace.build_paper_profile(
                paper_id=paper_id,
                requested_via="screen_papers",
                write_snapshot=False,
            )
            if (
                str(profile_payload.get("profile_status") or "").strip().lower() != "ready"
                or not str(profile_payload.get("profile_xml_artifact_path") or "").strip()
            ):
                dropped_candidates.append(
                    {
                        "paper_id": paper_id,
                        "title": candidate.title,
                        "decision": "drop",
                        "reason": _compact_text(profile_payload.get("error_message"))
                        or "Paper XML profile generation failed after download.",
                        "profile_status": profile_payload.get("profile_status"),
                        "profile_xml_artifact_path": profile_payload.get("profile_xml_artifact_path"),
                    }
                )
                continue
            profile_payloads.append(copy.deepcopy(profile_payload))
            ranked_candidates.append(
                self._build_screen_candidate_payload(
                    candidate=candidate,
                    profile_payload=profile_payload,
                )
            )

        if not ranked_candidates:
            failure_domain = "pdf_input" if any(
                str(item.get("profile_status") or "").strip().lower() == "pdf_precheck_failed"
                for item in dropped_candidates
            ) else "profile_infra"
            payload = {
                "stage": "proposer_screening",
                "cycle_number": cycle_number,
                "message": "candidate screening could not build any usable XML paper profiles after download",
                "screen_status": "input_exhausted" if failure_domain == "pdf_input" else "infra_failure",
                "failure_domain": failure_domain,
                "retryable": failure_domain == "pdf_input",
                "ranked_candidates": dropped_candidates,
                "paper_profiles": profile_payloads,
                "llm_screening_used": False,
            }
            workspace.store.write_json(
                f"proposer_cycle_{cycle_number}_candidate_screening.json",
                payload,
            )
            if not fail_on_no_usable:
                return payload
            self._raise_execution_failure(
                workspace=workspace,
                cycle_number=cycle_number,
                stage="proposer_screening",
                message="Candidate screening could not build any usable XML paper profiles after download.",
                details={
                    "reason": "screening_input_exhausted" if failure_domain == "pdf_input" else "screening_profile_infra_failure",
                    "failure_domain": failure_domain,
                    "requested_candidate_count": len(ordered_ids),
                },
                structured_output=payload,
            )
        provider, model_name, has_api_key = describe_chat_model_config(self.model_config)
        if not provider or not model_name or not has_api_key:
            payload = {
                "stage": "proposer_screening",
                "cycle_number": cycle_number,
                "message": "candidate screening requires an LLM model configuration",
                "ranked_candidates": [*ranked_candidates, *dropped_candidates],
                "paper_profiles": profile_payloads,
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
                ranked_candidates=ranked_candidates,
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
                "ranked_candidates": [*ranked_candidates, *dropped_candidates],
                "paper_profiles": profile_payloads,
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
                "ranked_candidates": [*ranked_candidates, *dropped_candidates],
                "paper_profiles": profile_payloads,
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
        payload = dict(llm_payload)
        payload.setdefault("screen_status", "ready")
        payload.setdefault("failure_domain", "")
        payload.setdefault("retryable", False)
        payload["paper_profiles"] = profile_payloads
        payload["ranked_candidates"] = [
            *list(payload.get("ranked_candidates") or []),
            *dropped_candidates,
        ]
        payload["dropped_paper_ids"] = _merge_unique_text(
            list(payload.get("dropped_paper_ids") or []),
            [item["paper_id"] for item in dropped_candidates if item.get("paper_id")],
        )
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
            query_plan_ids: Optional[List[str]] = None,
            query_text: Optional[str] = None,
            query_texts: Optional[List[str]] = None,
            lane: str = "data",
            reason: str = "",
        ) -> ToolResult:
            """Search Semantic Scholar for direct-PDF proposer candidates using planned or ad hoc queries."""
            if not run_state.query_plan_ids:
                return _policy_block(
                    "plan_queries must be called before search_papers.",
                    code="plan_required_before_search",
                )
            normalized_plan_ids: List[str] = []
            for item in list(query_plan_ids or []):
                normalized_item = str(item or "").strip()
                if normalized_item and normalized_item not in normalized_plan_ids:
                    normalized_plan_ids.append(normalized_item)
            single_plan_id = str(query_plan_id or "").strip()
            if single_plan_id and single_plan_id not in normalized_plan_ids:
                normalized_plan_ids.insert(0, single_plan_id)

            normalized_query_texts: List[str] = []
            single_query_text = str(query_text or "").strip()
            if single_query_text:
                normalized_query_texts.append(single_query_text)
            for item in list(query_texts or []):
                normalized_item = str(item or "").strip()
                if normalized_item and normalized_item not in normalized_query_texts:
                    normalized_query_texts.append(normalized_item)

            batch_payload = workspace.search_papers_batch(
                query_plan_id=query_plan_id,
                query_plan_ids=normalized_plan_ids,
                query_text=query_text,
                query_texts=normalized_query_texts,
                lane=lane,
                reason=reason,
                proposer_only_semantic=True,
            )
            payload = list(batch_payload.get("papers") or [])
            search_warnings = list(batch_payload.get("search_warnings") or [])
            run_state.record_search_results(payload)
            observation_payload = {
                "count": len(payload),
                "paper_ids": [item.get("paper_id") for item in payload[:5]],
                "batch_summary": batch_payload.get("batch_summary"),
            }
            if search_warnings:
                observation_payload["search_warnings"] = search_warnings
            return ToolResult(
                observation=_json_preview(observation_payload),
                data={
                    "papers": payload,
                    "search_warnings": search_warnings,
                    "batch_summary": batch_payload.get("batch_summary"),
                },
            )

        def screen_papers(
            paper_ids: Optional[List[str]] = None,
            max_candidates: Optional[int] = None,
        ) -> ToolResult:
            """Rerank downloaded PDFs using clipped GROBID XML profiles and lock the strongest candidates."""
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
            resolved_max_candidates = self.proposer_rerank_top_k
            if max_candidates is not None:
                resolved_max_candidates = max(
                    1,
                    min(self.proposer_candidate_target, int(max_candidates)),
                )
            missing_acquisitions = [
                paper_id for paper_id in candidate_ids if str(paper_id or "").strip() not in run_state.acquired_paper_ids
            ]
            if missing_acquisitions:
                return ToolResult(
                    observation=_json_preview(
                        {
                            "error": "screen_requires_downloaded_candidates",
                            "message": "screen_papers requires all requested candidates to be downloaded before reranking.",
                            "missing_paper_ids": missing_acquisitions,
                        }
                    ),
                    data={
                        "error": "screen_requires_downloaded_candidates",
                        "message": "screen_papers requires all requested candidates to be downloaded before reranking.",
                        "missing_paper_ids": missing_acquisitions,
                    },
                )
            payload = self._screen_candidate_papers(
                workspace=workspace,
                cycle_number=cycle_number,
                open_review_items=open_review_items,
                paper_ids=candidate_ids,
                max_candidates=resolved_max_candidates,
                fail_on_no_usable=False,
            )
            run_state.record_screening(payload)
            if str(payload.get("failure_domain") or "").strip() == "profile_infra":
                self._raise_execution_failure(
                    workspace=workspace,
                    cycle_number=cycle_number,
                    stage="proposer_screening",
                    message="Candidate screening failed because proposer profile infrastructure is unavailable.",
                    details={
                        "reason": "screening_profile_infra_failure",
                        "failure_domain": "profile_infra",
                    },
                    structured_output=payload,
                )
            if str(payload.get("screen_status") or "").strip() == "input_exhausted":
                payload["recovery_search_remaining"] = max(0, 2 - run_state.screening_input_exhaustions)
                if run_state.screening_input_exhaustions > 1:
                    self._raise_execution_failure(
                        workspace=workspace,
                        cycle_number=cycle_number,
                        stage="proposer_screening",
                        message="Candidate screening exhausted the available PDF-backed inputs after one recovery search.",
                        details={
                            "reason": "screening_input_exhausted",
                            "failure_domain": "pdf_input",
                            "recovery_attempts_used": run_state.screening_input_exhaustions - 1,
                        },
                        structured_output=payload,
                    )
            observation_payload = {
                "screen_status": payload.get("screen_status"),
                "failure_domain": payload.get("failure_domain"),
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

        def download_document(
            paper_id: Optional[str] = None,
            paper_ids: Optional[List[str]] = None,
        ) -> ToolResult:
            """Download and cache selected paper PDFs via Unpaywall first, then Semantic Scholar fallback."""
            requested_paper_ids = workspace._canonical_paper_id_batch(paper_id=paper_id, paper_ids=paper_ids)
            unknown_paper_ids = [
                current_paper_id
                for current_paper_id in requested_paper_ids
                if current_paper_id not in run_state.searched_paper_ids
            ]
            if unknown_paper_ids:
                return _policy_block(
                    "download_document requires papers selected from prior search_papers results; "
                    f"unknown paper_ids={unknown_paper_ids}.",
                    code="paper_not_searched",
                )
            if not requested_paper_ids:
                return _policy_block(
                    "download_document requires at least one searched paper_id.",
                    code="paper_not_searched",
                )
            payload = workspace.download_documents(
                paper_ids=requested_paper_ids,
                proposer_pdf_download=True,
            )
            run_state.record_acquisition(payload)
            return ToolResult(observation=_json_preview(payload), data=payload)

        def _validate_requested_papers(
            *,
            tool_name: str,
            paper_id: Optional[str] = None,
            paper_ids: Optional[List[str]] = None,
            require_locked: bool = False,
            require_parsed: bool = False,
        ) -> Tuple[Optional[ToolResult], List[str]]:
            requested_paper_ids = workspace._canonical_paper_id_batch(paper_id=paper_id, paper_ids=paper_ids)
            if not requested_paper_ids:
                return (
                    _policy_block(
                        f"{tool_name} requires paper_id or paper_ids.",
                        code="paper_id_required",
                    ),
                    [],
                )
            missing_downloads = [
                current_paper_id
                for current_paper_id in requested_paper_ids
                if current_paper_id not in run_state.acquired_paper_ids
            ]
            if missing_downloads:
                return (
                    _policy_block(
                        f"{tool_name} requires download_document first for paper_ids={missing_downloads}.",
                        code="paper_not_downloaded",
                    ),
                    [],
                )
            if require_parsed:
                not_parsed = [
                    current_paper_id
                    for current_paper_id in requested_paper_ids
                    if str(run_state.fulltext_status_by_paper.get(current_paper_id) or "").strip().lower() != "fulltext_indexed"
                ]
                if not_parsed:
                    return (
                        _policy_block(
                            f"{tool_name} requires parse_document first for paper_ids={not_parsed}.",
                            code="paper_not_parsed",
                        ),
                        [],
                    )
            if require_locked and run_state.locked_candidate_paper_ids:
                locked_set = set(run_state.locked_candidate_paper_ids)
                not_locked = [
                    current_paper_id
                    for current_paper_id in requested_paper_ids
                    if current_paper_id not in locked_set
                ]
                if not_locked:
                    return (
                        _policy_block(
                            f"{tool_name} requires papers locked by screen_papers; paper_ids={not_locked} were not locked.",
                            code="paper_not_screen_locked",
                        ),
                        [],
                    )
            return None, requested_paper_ids

        def parse_document(
            paper_id: Optional[str] = None,
            paper_ids: Optional[List[str]] = None,
        ) -> ToolResult:
            """Parse a downloaded paper into indexed sections after reranking selects it."""
            policy_result, requested_paper_ids = _validate_requested_papers(
                tool_name="parse_document",
                paper_id=paper_id,
                paper_ids=paper_ids,
                require_locked=True,
            )
            if policy_result is not None:
                return policy_result
            payload = (
                workspace.parse_documents(paper_ids=requested_paper_ids)
                if len(requested_paper_ids) > 1
                else workspace.parse_document(paper_id=requested_paper_ids[0])
            )
            run_state.record_acquisition(payload)
            return ToolResult(observation=_json_preview(payload), data=payload)

        def read_sections(
            paper_id: Optional[str] = None,
            paper_ids: Optional[List[str]] = None,
            section_ids: Optional[List[str]] = None,
            preferred_sections: bool = False,
        ) -> ToolResult:
            """Read indexed sections from a parsed paper."""
            if paper_ids and section_ids:
                return _policy_block(
                    "read_sections does not support section_ids when batching with paper_ids.",
                    code="batch_section_ids_not_supported",
                )
            policy_result, requested_paper_ids = _validate_requested_papers(
                tool_name="read_sections",
                paper_id=paper_id,
                paper_ids=paper_ids,
                require_parsed=True,
            )
            if policy_result is not None:
                return policy_result
            if len(requested_paper_ids) > 1:
                batch_payload = workspace.read_sections_batch(
                    paper_ids=requested_paper_ids,
                    preferred_sections=preferred_sections,
                )
                payload = list(batch_payload.get("sections") or [])
            else:
                payload = workspace.read_sections(
                    paper_id=requested_paper_ids[0],
                    section_ids=section_ids,
                    preferred_sections=preferred_sections,
                )
            for current_paper_id in requested_paper_ids:
                run_state.record_sections(
                    current_paper_id,
                    [item for item in payload if str(item.get("paper_id") or "").strip() == current_paper_id],
                )
            return ToolResult(
                observation=_json_preview(
                    {
                        "count": len(payload),
                        "paper_ids": requested_paper_ids,
                        "section_ids": [item.get("section_id") for item in payload[:10]],
                    }
                ),
                data={"sections": payload},
            )

        def extract_evidence(
            paper_id: Optional[str] = None,
            paper_ids: Optional[List[str]] = None,
            section_ids: Optional[List[str]] = None,
            preferred_sections: bool = False,
        ) -> ToolResult:
            """Extract stable evidence items from selected paper sections."""
            if paper_ids and section_ids:
                return _policy_block(
                    "extract_evidence does not support section_ids when batching with paper_ids.",
                    code="batch_section_ids_not_supported",
                )
            policy_result, requested_paper_ids = _validate_requested_papers(
                tool_name="extract_evidence",
                paper_id=paper_id,
                paper_ids=paper_ids,
                require_locked=True,
                require_parsed=True,
            )
            if policy_result is not None:
                return policy_result
            if len(requested_paper_ids) > 1:
                batch_payload = workspace.extract_evidence_batch(
                    paper_ids=requested_paper_ids,
                    preferred_sections=preferred_sections,
                )
                payload = list(batch_payload.get("evidence") or [])
            else:
                payload = workspace.extract_evidence(
                    paper_id=requested_paper_ids[0],
                    section_ids=section_ids,
                    preferred_sections=preferred_sections,
                )
            for current_paper_id in requested_paper_ids:
                run_state.record_evidence(
                    current_paper_id,
                    [item for item in payload if str(item.get("paper_id") or "").strip() == current_paper_id],
                )
            return ToolResult(
                observation=_json_preview(
                    {
                        "count": len(payload),
                        "paper_ids": requested_paper_ids,
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

        configured_screen_tool_schema = create_model(
            "ConfiguredScreenPapersToolInput",
            __base__=tool_schemas.ScreenPapersToolInput,
            max_candidates=(
                int,
                Field(
                    default=self.proposer_rerank_top_k,
                    ge=1,
                    le=self.proposer_candidate_target,
                    description="Maximum number of downloaded papers to lock after profile-based reranking.",
                ),
            ),
        )

        tools = [
            StructuredTool.from_function(plan_queries, name="plan_queries", args_schema=tool_schemas.PlanQueriesToolInput),
            StructuredTool.from_function(search_papers, name="search_papers", args_schema=tool_schemas.SearchPapersToolInput),
            StructuredTool.from_function(download_document, name="download_document", args_schema=tool_schemas.DownloadDocumentToolInput),
            StructuredTool.from_function(screen_papers, name="screen_papers", args_schema=configured_screen_tool_schema),
            StructuredTool.from_function(parse_document, name="parse_document", args_schema=tool_schemas.ProposerParseDocumentToolInput),
            StructuredTool.from_function(read_sections, name="read_sections", args_schema=tool_schemas.ProposerSectionAccessToolInput),
            StructuredTool.from_function(extract_evidence, name="extract_evidence", args_schema=tool_schemas.ProposerSectionAccessToolInput),
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
        def _dynamic_action_instruction(**kwargs: Any) -> str:
            runtime_guidance = self._runtime_action_guidance(
                run_state=run_state,
                step_number=int(kwargs.get("step_number") or 1),
                remaining_steps=int(kwargs.get("remaining_steps") or 0),
                max_steps=int(kwargs.get("max_steps") or self.max_steps_initial),
                deadline_mode=bool(kwargs.get("deadline_mode", True)),
            )
            return self._action_instruction(
                PROPOSER_TOOL_NAMES,
                conclude_call_contract=conclude_call_contract,
                runtime_guidance=runtime_guidance,
            )
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
                action_phase_instruction=_dynamic_action_instruction,
                search_tool_names=[
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
                    if not run_state.has_any_evidence():
                        self._raise_missing_evidence_repair_failure(
                            workspace=workspace,
                            cycle_number=cycle_number,
                            error=error,
                            trajectory=partial_trajectory,
                        )
                    return self._repair_submission_with_llm(
                        workspace=workspace,
                        cycle_number=cycle_number,
                        open_review_items=open_review_items,
                        error=error,
                        trajectory=partial_trajectory,
                        run_state=run_state,
                    )
                raise error
            if self.repair_attempts > 0 and run_state.has_any_evidence():
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
            if self.repair_attempts > 0 and not run_state.has_any_evidence():
                repair_error = ReactReviewedStructuredOutputError(
                    stage="proposer",
                    cycle_number=cycle_number,
                    message=f"forced conclude execution failed before a valid payload was emitted: {salvaged_error or exc}",
                    response_content=getattr(exc, "response_content", None),
                    structured_output=getattr(exc, "structured_output", None),
                    trajectory=partial_trajectory,
                )
                self._raise_missing_evidence_repair_failure(
                    workspace=workspace,
                    cycle_number=cycle_number,
                    error=repair_error,
                    trajectory=partial_trajectory,
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
                if not run_state.has_any_evidence():
                    self._raise_missing_evidence_repair_failure(
                        workspace=workspace,
                        cycle_number=cycle_number,
                        error=error,
                        trajectory=trajectory,
                    )
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
        if not run_state.has_any_evidence():
            self._raise_missing_evidence_repair_failure(
                workspace=workspace,
                cycle_number=cycle_number,
                error=error,
                trajectory=trajectory,
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

    def _raise_missing_evidence_repair_failure(
        self,
        *,
        workspace: ReactReviewedWorkspace,
        cycle_number: int,
        error: ReactReviewedStructuredOutputError,
        trajectory: Optional[ReActTrajectory],
    ) -> None:
        structured_output = error.structured_output if isinstance(error.structured_output, dict) else {}
        failure_domain = str(structured_output.get("failure_domain") or "").strip()
        screen_status = str(structured_output.get("screen_status") or "").strip()
        if failure_domain in {"pdf_input", "profile_infra"} or screen_status in {"input_exhausted", "infra_failure"}:
            failure_reason = "screening_input_exhausted" if failure_domain == "pdf_input" else "screening_profile_infra_failure"
            message = (
                "Proposer exhausted PDF-backed screening inputs before extracting current-cycle evidence anchors."
                if failure_domain == "pdf_input"
                else "Proposer could not build screening profiles because the profile infrastructure was unavailable."
            )
            self._raise_execution_failure(
                workspace=workspace,
                cycle_number=cycle_number,
                stage="proposer_screening",
                message=message,
                details={
                    "reason": failure_reason,
                    "failure_domain": failure_domain or "unknown",
                    "upstream_error": str(error),
                },
                response_content=error.response_content,
                structured_output=error.structured_output,
                trajectory=trajectory,
            )
        self._raise_execution_failure(
            workspace=workspace,
            cycle_number=cycle_number,
            stage="proposer_repair",
            message=(
                "Proposer exhausted retrieval budget before extracting current-cycle evidence anchors; "
                "repair was not attempted because submission repair requires evidence from this cycle."
            ),
            details={
                "repair_blocked_reason": "current_cycle_evidence_missing",
                "upstream_error": str(error),
            },
            response_content=error.response_content,
            structured_output=error.structured_output,
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
                        f"Citation '{citation.citation_id}' references anchors for paper_id '{citation.paper_id}' without download_document in this cycle."
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
        batch_search_result = workspace.search_papers_batch(
            query_plan_ids=search_plan_ids,
            reason="deterministic proposer",
            proposer_only_semantic=True,
        )
        search_results = list(batch_search_result.get("papers") or [])
        search_tool_calls: List[ToolCallRecord] = [
            ToolCallRecord(
                tool_name="search_papers",
                tool_call_id=f"tc_{uuid.uuid4().hex[:8]}",
                tool_args={"query_plan_ids": search_plan_ids, "reason": "deterministic proposer"},
                observation=_json_preview(batch_search_result),
                observation_data=batch_search_result,
            )
        ]
        search_step = self._add_tool_step(
            trajectory=trajectory,
            thought="Search the highest-value planned lanes before forming a submission.",
            tool_calls=search_tool_calls,
        )

        candidate_paper_ids = _merge_unique_text(
            [],
            [item.get("paper_id") for item in search_results if item.get("paper_id")],
        )
        batch_download_result = workspace.download_documents(
            paper_ids=candidate_paper_ids,
            proposer_pdf_download=True,
        )
        downloaded_payloads = list(batch_download_result.get("documents") or [])
        download_tool_calls: List[ToolCallRecord] = [
            ToolCallRecord(
                tool_name="download_document",
                tool_call_id=f"tc_{uuid.uuid4().hex[:8]}",
                tool_args={"paper_ids": candidate_paper_ids},
                observation=_json_preview(batch_download_result),
                observation_data=batch_download_result,
            )
        ]
        download_step = self._add_tool_step(
            trajectory=trajectory,
            thought="Download the retrieved OA papers before profile-based reranking.",
            tool_calls=download_tool_calls,
        )
        downloaded_paper_ids = [
            str(payload.get("paper_id") or "").strip()
            for payload in downloaded_payloads
            if str(payload.get("paper_id") or "").strip()
        ]
        if not downloaded_paper_ids:
            self._raise_execution_failure(
                workspace=workspace,
                cycle_number=cycle_number,
                stage="proposer_download",
                message="Candidate download did not yield any papers for post-download screening.",
                details={"candidate_paper_ids": candidate_paper_ids},
                trajectory=trajectory,
            )
        screened_payload = self._screen_candidate_papers(
            workspace=workspace,
            cycle_number=cycle_number,
            open_review_items=open_review_items,
            paper_ids=downloaded_paper_ids,
            max_candidates=self.proposer_rerank_top_k,
        )
        screen_step = self._add_tool_step(
            trajectory=trajectory,
            thought="Build GROBID paper profiles and rerank the downloaded papers before parsing and evidence extraction.",
            tool_calls=[
                ToolCallRecord(
                    tool_name="screen_papers",
                    tool_call_id=f"tc_{uuid.uuid4().hex[:8]}",
                    tool_args={"paper_ids": downloaded_paper_ids, "max_candidates": self.proposer_rerank_top_k},
                    observation=_json_preview(screened_payload),
                    observation_data=screened_payload,
                )
            ],
        )
        parsed_paper_ids = list(screened_payload.get("locked_paper_ids") or [])
        if not parsed_paper_ids:
            self._raise_execution_failure(
                workspace=workspace,
                cycle_number=cycle_number,
                stage="proposer_screening",
                message="Candidate screening did not lock any papers for evidence extraction.",
                details={"screened_paper_count": len(screened_payload.get("ranked_candidates") or [])},
                trajectory=trajectory,
            )
        parsed_payloads: List[Dict[str, Any]] = []
        parse_tool_calls: List[ToolCallRecord] = []
        for paper_id in parsed_paper_ids:
            payload = workspace.parse_document(paper_id=str(paper_id))
            parsed_payloads.append(payload)
            parse_tool_calls.append(
                ToolCallRecord(
                    tool_name="parse_document",
                    tool_call_id=f"tc_{uuid.uuid4().hex[:8]}",
                    tool_args={"paper_id": paper_id},
                    observation=_json_preview(payload),
                    observation_data=payload,
                )
            )
        parse_step = self._add_tool_step(
            trajectory=trajectory,
            thought="Parse only the reranked top-k papers into indexed full-text sections before evidence extraction.",
            tool_calls=parse_tool_calls,
        )

        extracted_evidence: List[Dict[str, Any]] = []
        extract_tool_calls: List[ToolCallRecord] = []
        for paper_id in parsed_paper_ids:
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
                message="No evidence extraction calls were possible after parsing the reranked papers.",
                details={"selected_paper_ids": parsed_paper_ids},
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
            _step_ref(trajectory, download_step),
            _step_ref(trajectory, screen_step),
            _step_ref(trajectory, parse_step),
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
        return build_proposer_system_prompt(
            conclude_contract=conclude_call_contract,
            proposer_candidate_target=self.proposer_candidate_target,
        )

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
                    "download_document",
                    "screen_papers",
                    "parse_document",
                    "read_sections_or_extract_evidence",
                    "conclude",
                ],
                "selection_rubric": [
                    "entity_and_condition_alignment",
                    "question_type_and_lane_fit",
                    "semantic_scholar_strict_pdf_candidate_recall",
                    "clipped_grobid_xml_profile_coverage",
                    "llm_listwise_rerank_for_evidence_likelihood",
                    "coverage_diversity",
                ],
            },
        )

    def _thought_instruction(self) -> str:
        return build_proposer_thought_prompt()

    def _runtime_action_guidance(
        self,
        *,
        run_state: _ProposerRunState,
        step_number: int,
        remaining_steps: int,
        max_steps: int,
        deadline_mode: bool,
    ) -> Dict[str, Any]:
        locked_paper_ids = list(run_state.locked_candidate_paper_ids)
        parsed_locked_paper_ids = run_state.locked_parsed_paper_ids()
        locked_evidence_paper_ids = run_state.locked_evidence_paper_ids()
        recovery_available = bool(
            run_state.latest_screen_status == "input_exhausted"
            and run_state.latest_screen_failure_domain == "pdf_input"
            and run_state.screening_input_exhaustions <= 1
            and not run_state.has_any_evidence()
            and not locked_paper_ids
        )

        if not run_state.query_plan_ids:
            current_stage = "planning"
            exit_criteria = "Leave planning immediately after one plan_queries call returns reusable query_plan_ids."
            recommended_next_tools = ["plan_queries"]
            avoid_actions = ["Do not call search_papers before plan_queries."]
        elif run_state.has_any_evidence() or remaining_steps <= 3:
            current_stage = "closeout"
            exit_criteria = "Stay in closeout once at least one evidence anchor exists or only a few steps remain."
            recommended_next_tools = ["conclude"]
            if locked_paper_ids and not locked_evidence_paper_ids:
                recommended_next_tools.insert(0, "extract_evidence")
            elif parsed_locked_paper_ids and not locked_evidence_paper_ids:
                recommended_next_tools.insert(0, "extract_evidence")
            elif parsed_locked_paper_ids and len(parsed_locked_paper_ids) < len(locked_paper_ids):
                recommended_next_tools.insert(0, "parse_document")
            avoid_actions = ["Do not restart broad discovery unless screen_papers explicitly returned input_exhausted/pdf_input."]
        elif locked_paper_ids and not parsed_locked_paper_ids:
            current_stage = "parsing"
            exit_criteria = "Exit parsing after every currently locked paper is indexed or a parse failure is observed."
            recommended_next_tools = ["parse_document"]
            avoid_actions = ["Do not spend this step on new search/download while locked papers remain unparsed."]
        elif parsed_locked_paper_ids and not locked_evidence_paper_ids:
            current_stage = "evidence_extraction"
            exit_criteria = "Exit evidence extraction after at least one stable evidence anchor is recorded."
            recommended_next_tools = ["extract_evidence", "read_sections"]
            avoid_actions = ["Do not expand search/download until locked parsed papers have been mined for evidence."]
        else:
            current_stage = "acquisition_or_screening"
            exit_criteria = "Leave acquisition/screening only after one search, one download, and one screen pass have either locked papers or explicitly exhausted PDF input."
            if not run_state.searched_paper_ids:
                recommended_next_tools = ["search_papers"]
                avoid_actions = ["Do not spend extra steps on replanning; reuse the current query plans."]
            elif not run_state.acquired_paper_ids:
                recommended_next_tools = ["download_document"]
                avoid_actions = ["Do not run another search before downloading the current cycle's searched papers."]
            elif run_state.screening_required():
                recommended_next_tools = ["screen_papers"]
                avoid_actions = ["Do not parse or extract before screen_papers locks candidates."]
            elif recovery_available:
                recommended_next_tools = ["search_papers", "download_document", "screen_papers"]
                avoid_actions = ["Only use one recovery search/download pair, then stop expanding."]
            else:
                recommended_next_tools = ["screen_papers"]
                avoid_actions = ["Do not keep broad discovery open once the current cycle can be screened."]

        if deadline_mode and remaining_steps <= 2:
            avoid_actions.append("Deadline mode is active; do not call broad discovery tools when only 2 steps remain.")

        return {
            "current_stage": current_stage,
            "exit_criteria": exit_criteria,
            "recommended_next_tools": recommended_next_tools,
            "avoid_actions": avoid_actions,
            "budget_snapshot": {
                "step_number": int(step_number),
                "remaining_steps": int(remaining_steps),
                "max_steps": int(max_steps),
                "query_planned": bool(run_state.query_plan_ids),
                "search_rounds_used": int(run_state.search_generation),
                "download_rounds_used": int(run_state.acquisition_generation),
                "screen_rounds_used": int(run_state.screening_generation),
                "locked_paper_ids": list(locked_paper_ids),
                "parsed_locked_paper_ids": list(parsed_locked_paper_ids),
                "evidence_anchor_count": len(run_state.evidence_ids),
                "screening_required": bool(run_state.screening_required()),
                "recovery_search_download_available": recovery_available,
            },
        }

    def _action_instruction(
        self,
        tool_names: Sequence[str],
        *,
        conclude_call_contract: Dict[str, Any],
        runtime_guidance: Optional[Dict[str, Any]] = None,
    ) -> str:
        retrieval_tools = [name for name in tool_names if name not in {"analyze_submission_gap", "conclude"}]
        return build_proposer_action_prompt(
            tool_names=tool_names,
            retrieval_tools=retrieval_tools,
            conclude_contract=conclude_call_contract,
            proposer_candidate_target=self.proposer_candidate_target,
            runtime_guidance=runtime_guidance,
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

