from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

from qa.live_validation import LiveValidationCase
from qa.live_validation import main as live_validation_main
from qa.live_validation import validate_live_qa
from qa.synthesis_state import QAResult
from utils import ensure_dir, save_json


FIXED_REVIEWER_ROLES = (
    "search_coverage",
    "evidence_trace",
    "reasoning_consistency",
    "counterevidence",
)


def _confidence(score: float = 0.82) -> dict:
    return {
        "level": "high" if score >= 0.75 else "low",
        "score": score,
        "rationale": "validation fixture",
    }


class _BaseFakeValidationSystem:
    def __init__(self) -> None:
        self.qa_config = {
            "providers": {
                "openalex_mailto": "chemqa@example.com",
                "crossref_mailto": "chemqa@example.com",
                "semantic_scholar_api_key": None,
                "unpaywall_email": "chemqa@example.com",
                "http_timeout": 5.0,
            }
        }

    def _runtime_manifest(self) -> dict:
        return {
            "providers": {
                "openalex": {"enabled": True},
                "crossref": {"enabled": True},
                "semantic_scholar": {"enabled": True},
                "unpaywall": {"enabled": True},
            }
        }


class _EvidenceBackedSystem(_BaseFakeValidationSystem):
    def run_qa(self, *, question: str, context=None, artifact_dir=None):
        artifact_root = ensure_dir(Path(artifact_dir))
        paper_id = "paper-1"
        citation_id = "CIT-1"

        save_json(self._runtime_manifest(), artifact_root / "runtime_manifest.json")
        save_json([{"paper_id": paper_id, "title": "Pt/C HER in alkaline media"}], artifact_root / "paper_candidates.json")
        save_json([{"paper_id": paper_id, "title": "Pt/C HER in alkaline media"}], artifact_root / "paper_records.json")
        save_json(
            [{"provider": "openalex", "stage": "search", "lane": "review", "hit_count": 1}],
            artifact_root / "retrieval_diagnostics.json",
        )
        save_json(
            {
                "openalex": {"status": "healthy", "calls": 1, "successes": 1, "retry_exhausted_failures": 0, "skipped_calls": 0, "last_error": None},
                "crossref": {"status": "healthy", "calls": 1, "successes": 1, "retry_exhausted_failures": 0, "skipped_calls": 0, "last_error": None},
            },
            artifact_root / "provider_health.json",
        )
        save_json(
            {
                "section_claims": [
                    {
                        "section_id": "direct_answer",
                        "title": "Direct Answer",
                        "accepted_claim_ids": ["claim-1"],
                        "claim_summaries": ["Pt/C lowers HER overpotential in 1 M KOH."],
                        "core_citation_ids": [citation_id],
                        "section_confidence": _confidence(),
                    }
                ],
                "citation_catalog": [{"citation_id": citation_id, "paper_id": paper_id, "title": "Pt/C HER in alkaline media", "year": 2024}],
                "overall_confidence": _confidence(),
                "section_confidence": [],
                "insufficient_evidence": False,
                "claim_trace": [{"section_id": "direct_answer", "claim_id": "claim-1", "status": "accepted", "citation_ids": [citation_id], "confidence": 0.85}],
                "retrieval_diagnostics_summary": "",
                "execution_warnings": [],
                "question": question,
                "task_spec": {
                    "version": "1.0",
                    "question": question,
                    "normalized_question": question,
                    "question_type": "mechanism",
                    "recency_policy": "none",
                    "answer_sections": [],
                    "required_condition_axes": [],
                    "query_constraints": {"must_include_terms": [], "should_include_terms": [], "exclude_terms": [], "preferred_entity_types": [], "allow_broad_expansion": False},
                    "ambiguity_flags": [],
                    "router_confidence": 0.9,
                },
            },
            artifact_root / "synthesis_input_pack.json",
        )
        save_json({"claims": [{"claim_id": "claim-1", "status": "accepted"}]}, artifact_root / "evidence_ledger_reviewed.json")

        result = QAResult.model_validate(
            {
                "question": question,
                "language": "en",
                "final_answer": "Pt/C improves HER activity in 1 M KOH by lowering overpotential and accelerating interfacial hydrogen evolution. [CIT-1]",
                "sections": [],
                "citations": [{"citation_id": citation_id, "paper_id": paper_id, "title": "Pt/C HER in alkaline media", "year": 2024, "supporting_claim_ids": ["claim-1"]}],
                "claim_trace": [{"section_id": "direct_answer", "claim_id": "claim-1", "status": "accepted", "citation_ids": [citation_id], "confidence": 0.85}],
                "overall_confidence": _confidence(),
                "section_confidence": [],
                "insufficient_evidence": False,
                "limitations_summary": "",
                "retrieval_diagnostics_summary": "",
                "execution_warnings": [],
                "artifact_paths": {
                    "qa_result": str(artifact_root / "qa_result.json"),
                    "runtime_manifest": str(artifact_root / "runtime_manifest.json"),
                    "synthesis_input_pack": str(artifact_root / "synthesis_input_pack.json"),
                    "final_answer": str(artifact_root / "final_answer.md"),
                },
                "time_elapsed": 0.21,
            }
        )
        save_json(result.model_dump(exclude_none=True), artifact_root / "qa_result.json")
        (artifact_root / "final_answer.md").write_text(result.final_answer, encoding="utf-8")
        return result


class _DegradedSystem(_BaseFakeValidationSystem):
    def run_qa(self, *, question: str, context=None, artifact_dir=None):
        artifact_root = ensure_dir(Path(artifact_dir))
        save_json(self._runtime_manifest(), artifact_root / "runtime_manifest.json")
        save_json([], artifact_root / "paper_candidates.json")
        save_json([], artifact_root / "paper_records.json")
        save_json(
            [{"provider": "openalex", "stage": "search", "lane": "review", "failure_count": 1, "skipped_count": 3, "sample_messages": ["retry exhausted; provider unavailable"]}],
            artifact_root / "retrieval_diagnostics.json",
        )
        save_json(
            {
                "openalex": {"status": "unavailable", "calls": 1, "successes": 0, "retry_exhausted_failures": 1, "skipped_calls": 3, "last_error": "retry exhausted"},
                "crossref": {"status": "idle", "calls": 0, "successes": 0, "retry_exhausted_failures": 0, "skipped_calls": 0, "last_error": None},
            },
            artifact_root / "provider_health.json",
        )
        save_json(
            {
                "section_claims": [],
                "citation_catalog": [],
                "overall_confidence": _confidence(0.2),
                "section_confidence": [],
                "insufficient_evidence": True,
                "claim_trace": [],
                "retrieval_diagnostics_summary": "External literature retrieval encountered issues: OpenAlex review search had 1 failure (retry exhausted).",
                "execution_warnings": [],
                "question": question,
                "task_spec": {
                    "version": "1.0",
                    "question": question,
                    "normalized_question": question,
                    "question_type": "mechanism",
                    "recency_policy": "none",
                    "answer_sections": [],
                    "required_condition_axes": [],
                    "query_constraints": {"must_include_terms": [], "should_include_terms": [], "exclude_terms": [], "preferred_entity_types": [], "allow_broad_expansion": False},
                    "ambiguity_flags": [],
                    "router_confidence": 0.9,
                },
            },
            artifact_root / "synthesis_input_pack.json",
        )
        save_json({"claims": []}, artifact_root / "evidence_ledger_reviewed.json")

        result = QAResult.model_validate(
            {
                "question": question,
                "language": "en",
                "final_answer": "Available accepted evidence is limited and does not support a firm conclusion for this section.",
                "sections": [],
                "citations": [],
                "claim_trace": [],
                "overall_confidence": _confidence(0.2),
                "section_confidence": [],
                "insufficient_evidence": True,
                "limitations_summary": "External literature retrieval encountered issues.",
                "retrieval_diagnostics_summary": "External literature retrieval encountered issues: OpenAlex review search had 1 failure (retry exhausted).",
                "execution_warnings": [],
                "artifact_paths": {
                    "qa_result": str(artifact_root / "qa_result.json"),
                    "runtime_manifest": str(artifact_root / "runtime_manifest.json"),
                    "synthesis_input_pack": str(artifact_root / "synthesis_input_pack.json"),
                    "final_answer": str(artifact_root / "final_answer.md"),
                },
                "time_elapsed": 0.19,
            }
        )
        save_json(result.model_dump(exclude_none=True), artifact_root / "qa_result.json")
        (artifact_root / "final_answer.md").write_text(result.final_answer, encoding="utf-8")
        return result


class _BrokenSystem(_BaseFakeValidationSystem):
    def run_qa(self, *, question: str, context=None, artifact_dir=None):
        raise RuntimeError("synthetic pipeline failure")


class _ReactReviewedSystem(_BaseFakeValidationSystem):
    def __init__(
        self,
        *,
        degraded: bool = False,
        missing_top_level_artifact: str | None = None,
        missing_role: str | None = None,
        acceptance_status: str = "accepted",
        review_completion_status: str = "completed",
        reviewer_status_overrides: dict[str, str] | None = None,
        submission_alignment_ok: bool = True,
        anchor_ok: bool = True,
        budget_ok: bool = True,
        budget_artifacts_present: bool = True,
        router_used_fallback: bool = False,
        router_question_type: str = "mechanism",
        router_recency_policy: str = "none",
        entity_resolved_count: int = 1,
        entity_unresolved_count: int = 0,
        pubchem_call_count: int = 0,
        pubchem_error_count: int = 0,
    ) -> None:
        super().__init__()
        self.degraded = degraded
        self.missing_top_level_artifact = missing_top_level_artifact
        self.missing_role = missing_role
        self.acceptance_status = acceptance_status
        self.review_completion_status = review_completion_status
        self.reviewer_status_overrides = dict(reviewer_status_overrides or {})
        self.submission_alignment_ok = submission_alignment_ok
        self.anchor_ok = anchor_ok
        self.budget_ok = budget_ok
        self.budget_artifacts_present = budget_artifacts_present
        self.router_used_fallback = router_used_fallback
        self.router_question_type = router_question_type
        self.router_recency_policy = router_recency_policy
        self.entity_resolved_count = entity_resolved_count
        self.entity_unresolved_count = entity_unresolved_count
        self.pubchem_call_count = pubchem_call_count
        self.pubchem_error_count = pubchem_error_count

    def run_qa(self, *, question: str, context=None, artifact_dir=None):
        artifact_root = ensure_dir(Path(artifact_dir))
        return _write_react_reviewed_fixture(
            artifact_root=artifact_root,
            question=question,
            degraded=self.degraded,
            missing_top_level_artifact=self.missing_top_level_artifact,
            missing_role=self.missing_role,
            acceptance_status=self.acceptance_status,
            review_completion_status=self.review_completion_status,
            reviewer_status_overrides=self.reviewer_status_overrides,
            submission_alignment_ok=self.submission_alignment_ok,
            anchor_ok=self.anchor_ok,
            budget_ok=self.budget_ok,
            budget_artifacts_present=self.budget_artifacts_present,
            router_used_fallback=self.router_used_fallback,
            router_question_type=self.router_question_type,
            router_recency_policy=self.router_recency_policy,
            entity_resolved_count=self.entity_resolved_count,
            entity_unresolved_count=self.entity_unresolved_count,
            pubchem_call_count=self.pubchem_call_count,
            pubchem_error_count=self.pubchem_error_count,
        )


class _DispatchingSuiteSystem(_BaseFakeValidationSystem):
    def __init__(self, scenario_by_question: dict[str, dict]) -> None:
        super().__init__()
        self.scenario_by_question = scenario_by_question

    def run_qa(self, *, question: str, context=None, artifact_dir=None):
        scenario = dict(self.scenario_by_question.get(question) or {})
        return _write_react_reviewed_fixture(
            artifact_root=ensure_dir(Path(artifact_dir)),
            question=question,
            **scenario,
        )


def _react_confidence(score: float = 0.84) -> dict:
    return {
        "level": "high" if score >= 0.75 else "medium",
        "score": score,
        "rationale": "react-reviewed fixture",
    }


def _react_step_ref(step_number: int = 1) -> dict:
    return {"trajectory_id": "traj-proposer", "step_number": step_number}


def _write_react_reviewed_fixture(
    *,
    artifact_root: Path,
    question: str,
    degraded: bool = False,
    missing_top_level_artifact: str | None = None,
    missing_role: str | None = None,
    acceptance_status: str = "accepted",
    review_completion_status: str = "completed",
    reviewer_status_overrides: dict[str, str] | None = None,
    submission_alignment_ok: bool = True,
    anchor_ok: bool = True,
    budget_ok: bool = True,
    budget_artifacts_present: bool = True,
    router_used_fallback: bool = False,
    router_question_type: str = "mechanism",
    router_recency_policy: str = "none",
    entity_resolved_count: int = 1,
    entity_unresolved_count: int = 0,
    pubchem_call_count: int = 0,
    pubchem_error_count: int = 0,
) -> QAResult:
    paper_id = "paper-rr-1"
    citation_id = "CIT-RR-1"
    section_id = "direct_answer"
    trace_section_id = section_id if submission_alignment_ok else "mismatched_section"
    reviewer_status_overrides = dict(reviewer_status_overrides or {})
    final_answer = (
        "Pt/C improves HER activity in alkaline electrolyte by accelerating interfacial hydrogen evolution. [CIT-RR-1]"
        if acceptance_status == "accepted" and not degraded
        else "The workflow completed, but provider degradation limited evidence collection for a firm conclusion."
    )

    save_json(
        {
            "workflow_mode": "react_reviewed",
            "models": {
                "router": {"enabled": True},
                "entity_resolver": {"enabled": True},
            },
            "providers": {
                "pubchem": {"enabled": True},
            },
        },
        artifact_root / "runtime_manifest.json",
    )
    save_json([] if degraded else [{"paper_id": paper_id, "title": "Pt/C HER in alkaline media"}], artifact_root / "paper_candidates.json")
    save_json([] if degraded else [{"paper_id": paper_id, "title": "Pt/C HER in alkaline media"}], artifact_root / "paper_records.json")
    save_json(
        (
            [{"provider": "openalex", "stage": "search", "lane": "review", "failure_count": 1, "skipped_count": 1, "sample_messages": ["retry exhausted"]}]
            if degraded
            else [{"provider": "openalex", "stage": "search", "lane": "review", "hit_count": 1}]
        ),
        artifact_root / "retrieval_diagnostics.json",
    )
    save_json(
        {
            "openalex": {
                "status": "unavailable" if degraded else "healthy",
                "calls": 1,
                "successes": 0 if degraded else 1,
                "retry_exhausted_failures": 1 if degraded else 0,
                "skipped_calls": 1 if degraded else 0,
                "last_error": "retry exhausted" if degraded else None,
            }
        },
        artifact_root / "provider_health.json",
    )

    proposer_trajectory = {
        "query": question,
        "trajectory_id": "traj-proposer",
        "steps": [
            {"step_number": 1, "thought": "Plan search", "action": "plan_queries", "action_input": {"focus": "direct_answer"}, "tool_calls": [], "observation": "planned", "timestamp": "2026-03-18T10:00:00"},
            {"step_number": 2, "thought": "Extract evidence", "action": "extract_evidence", "action_input": {"paper_id": paper_id}, "tool_calls": [], "observation": "evidence extracted", "timestamp": "2026-03-18T10:00:01"},
        ],
        "final_answer": "{}",
        "total_steps": 2,
        "start_time": "2026-03-18T10:00:00",
        "end_time": "2026-03-18T10:00:02",
    }
    final_submission = {
        "submission_id": "submission-1",
        "question": question,
        "version": 1,
        "sections": [
            {
                "section_id": section_id,
                "title": "Direct Answer",
                "content": "Pt/C improves alkaline HER activity.",
                "citation_ids": [] if degraded else [citation_id],
                "step_refs": [_react_step_ref(2)],
                "issue_refs": [],
                "section_confidence": _react_confidence(0.78 if degraded else 0.9),
            }
        ],
        "citations": [] if degraded else [{"citation_id": citation_id, "paper_id": paper_id, "title": "Pt/C HER in alkaline media", "year": 2024, "section_ids": [section_id], "evidence_ids": ["EV-1"]}],
        "limitations": ["Provider degradation limited retrieval breadth."] if degraded else [],
        "overall_confidence": _react_confidence(0.6 if degraded else 0.9),
        "trajectory_id": "traj-proposer",
        "step_refs": [_react_step_ref(2)],
        "issue_refs": [],
    }
    submission_trace = [{"section_id": trace_section_id, "citation_ids": [] if degraded else [citation_id], "step_refs": [_react_step_ref(2)], "issue_refs": []}]

    review_statuses = []
    reviewer_trajectories = {}
    for role in FIXED_REVIEWER_ROLES:
        if role == missing_role:
            continue
        status = {
            "reviewer_role": role,
            "status": reviewer_status_overrides.get(role, "completed"),
            "message": "",
            "cycle_number": 1,
            "retrieval_actions_used": 0 if role != "counterevidence" else (1 if not degraded else 0),
            "retrieval_budget_limit": 1 if role != "counterevidence" else 2,
            "budget_blocked_calls": 0,
        }
        review_statuses.append(status)
        reviewer_trajectories[role] = {
            "query": f"{role} review",
            "trajectory_id": f"traj-{role}",
            "steps": [{"step_number": 1, "thought": f"{role} review thought", "action": "conclude", "action_input": {}, "tool_calls": [], "observation": "done", "timestamp": "2026-03-18T10:01:00"}],
            "final_answer": "{}",
            "total_steps": 1,
            "start_time": "2026-03-18T10:01:00",
            "end_time": "2026-03-18T10:01:01",
        }

    final_review_items = []
    if not anchor_ok:
        final_review_items.append(
            {
                "review_id": "rev-1",
                "reviewer_role": "evidence_trace",
                "anchor_kind": "step_section",
                "severity": "blocking",
                "flaw_type": "bad_anchor",
                "critique": "Anchor is broken.",
                "required_action": "Fix the anchor.",
                "status": "open",
                "target_trajectory_id": "traj-proposer",
                "target_step_number": 99,
                "target_section_id": section_id,
            }
        )

    submission_cycles = [
        {
            "cycle_number": 1,
            "current_submission": final_submission,
            "proposer_trajectory": proposer_trajectory,
            "reviewer_trajectories": reviewer_trajectories,
            "open_review_items": final_review_items,
            "review_responses": [],
            "reviewer_statuses": review_statuses,
        }
    ]

    save_json(final_submission, artifact_root / "candidate_submission.json")
    save_json(
        {
            "status": acceptance_status,
            "blocker_codes": [] if acceptance_status == "accepted" else ["fixture_rejection"],
            "blocker_messages": [] if acceptance_status == "accepted" else ["Fixture rejected the candidate submission."],
            "blocking_review_ids": [],
        },
        artifact_root / "acceptance_decision.json",
    )
    if acceptance_status == "accepted":
        save_json(final_submission, artifact_root / "final_submission.json")
    save_json(submission_trace, artifact_root / "submission_trace.json")
    save_json(submission_cycles, artifact_root / "submission_cycles.json")
    save_json(proposer_trajectory, artifact_root / "proposer_trajectory.json")
    save_json(reviewer_trajectories, artifact_root / "reviewer_trajectories.json")
    save_json(review_statuses, artifact_root / "review_statuses.json")
    save_json(final_review_items, artifact_root / "final_review_items.json")
    router_task_spec = {
        "version": "1.0",
        "question": question,
        "normalized_question": question,
        "question_type": router_question_type,
        "recency_policy": router_recency_policy,
        "year_from": 2022 if router_recency_policy != "none" else None,
        "year_to": 2025 if router_recency_policy != "none" else None,
        "answer_sections": [],
        "required_condition_axes": [],
        "query_constraints": {
            "must_include_terms": [],
            "should_include_terms": [],
            "exclude_terms": [],
            "preferred_entity_types": [],
            "allow_broad_expansion": False,
        },
        "ambiguity_flags": [{"flag_type": "task_ambiguous", "target": "scope", "note": "fixture", "severity": "low"}] if "ambiguous" in question.lower() else [],
        "router_confidence": 0.88,
    }
    save_json(router_task_spec, artifact_root / "router" / "task_spec.json")
    save_json({"primary_question_type": router_question_type, "semantic_confidence": 0.82}, artifact_root / "router" / "semantic_stage.json")
    save_json({"question_type": router_question_type, "recency_policy": router_recency_policy}, artifact_root / "router" / "localization_stage.json")
    if router_used_fallback:
        save_json({"reason": "fixture_router_fallback"}, artifact_root / "router" / "fallback_reason.json")
    save_json(
        {
            "agent": "RouterAgent",
            "input": {"question": question, "context": None},
            "output": router_task_spec,
            "debug": {
                "semantic_stage": {"primary_question_type": router_question_type},
                "localization_stage": {"question_type": router_question_type, "recency_policy": router_recency_policy},
                **({"fallback_reason": {"reason": "fixture_router_fallback"}} if router_used_fallback else {}),
            },
        },
        artifact_root / "router" / "agent_run.json",
    )

    entities = [
        {
            "entity_id": f"entity_{index + 1}",
            "mention": "ethanol" if index == 0 else f"entity-{index + 1}",
            "source_text": "ethanol" if index == 0 else f"entity-{index + 1}",
            "source_span": {"start": index * 2, "end": index * 2 + 1},
            "entity_type": "molecule",
            "canonical_name": "ethanol" if index == 0 else f"entity-{index + 1}",
            "aliases": ["EtOH"] if index == 0 else [],
            "query_anchors": ["ethanol", "EtOH"] if index == 0 else [f"entity-{index + 1}"],
            "resolver_source": "pubchem" if index == 0 and pubchem_call_count > 0 else "seed",
            "resolution_confidence": 0.92,
            "status": "resolved",
        }
        for index in range(entity_resolved_count)
    ]
    unresolved_mentions = [
        {
            "mention": f"unresolved-{index + 1}",
            "candidate_entity_types": ["molecule"],
            "reason": "fixture unresolved",
            "confidence": 0.41,
            "source_text": f"unresolved-{index + 1}",
            "source_span": {"start": 10 + index * 2, "end": 11 + index * 2},
        }
        for index in range(entity_unresolved_count)
    ]
    entity_pack = {
        "version": "1.0",
        "entities": entities,
        "condition_mentions": [],
        "unresolved_mentions": unresolved_mentions,
        "entity_ambiguity_flags": (
            [{"flag_type": "entity_ambiguous", "target": "Pt/C", "note": "fixture ambiguity", "severity": "medium"}]
            if "ambiguous" in question.lower() or entity_unresolved_count > 0
            else []
        ),
    }
    pubchem_calls = []
    for index in range(pubchem_call_count):
        status = "error" if index < pubchem_error_count else "hit"
        pubchem_calls.append(
            {
                "provider": "pubchem",
                "query": "ethanol",
                "status": status,
                "candidate_count": 1 if status != "error" else 0,
                **({"error": "fixture pubchem error"} if status == "error" else {}),
            }
        )
    save_json(entity_pack, artifact_root / "entity_resolver" / "entity_pack.json")
    save_json({"entries": entities, "cache_events": []}, artifact_root / "entity_resolver" / "resolution_index.json")
    save_json(pubchem_calls, artifact_root / "entity_resolver" / "provider_calls.json")
    save_json([], artifact_root / "entity_resolver" / "seed_suggestions.json")
    save_json(
        {
            "agent": "EntityResolverAgent",
            "input": {"question": question, "task_spec": router_task_spec},
            "output": entity_pack,
        },
        artifact_root / "entity_resolver" / "agent_run.json",
    )
    (artifact_root / "final_answer.md").write_text(final_answer, encoding="utf-8")

    for role in FIXED_REVIEWER_ROLES:
        cycle_dir = artifact_root / "reviewers" / role / "cycle_1"
        cycle_dir.mkdir(parents=True, exist_ok=True)
        if role == missing_role:
            continue
        role_status = next(item for item in review_statuses if item["reviewer_role"] == role)
        if budget_artifacts_present:
            actions_used = role_status["retrieval_actions_used"]
            budget_limit = role_status["retrieval_budget_limit"]
            if not budget_ok and role == "counterevidence":
                actions_used = budget_limit + 1
            save_json(
                {"role": role, "budget_limit": budget_limit, "actions_used": actions_used, "cache_hits": 0, "cache_misses": actions_used, "blocked_calls": 0, "charged_tools": {}, "events": []},
                cycle_dir / "budget_usage.json",
            )
            save_json(role_status, cycle_dir / "reviewer_status.json")
            save_json(reviewer_trajectories[role], cycle_dir / "reviewer_trajectory.json")

    result = QAResult.model_validate(
        {
            "question": question,
            "language": "en",
            "workflow_mode": "react_reviewed",
            "acceptance_status": acceptance_status,
            "final_answer": final_answer,
            "sections": [],
            "citations": [] if degraded else [{"citation_id": citation_id, "paper_id": paper_id, "title": "Pt/C HER in alkaline media", "year": 2024, "supporting_claim_ids": []}],
            "claim_trace": [],
            "submission_trace": submission_trace,
            "review_completion_status": review_completion_status,
            "overall_confidence": _react_confidence(0.6 if degraded else 0.9),
            "section_confidence": [],
            "insufficient_evidence": degraded,
            "limitations_summary": "Provider degradation limited retrieval breadth." if degraded else "",
            "retrieval_diagnostics_summary": "External literature retrieval encountered issues." if degraded else "",
            "execution_warnings": [],
            "artifact_paths": {
                "qa_result": str(artifact_root / "qa_result.json"),
                "runtime_manifest": str(artifact_root / "runtime_manifest.json"),
                "candidate_submission": str(artifact_root / "candidate_submission.json"),
                "acceptance_decision": str(artifact_root / "acceptance_decision.json"),
                "submission_trace": str(artifact_root / "submission_trace.json"),
                "submission_cycles": str(artifact_root / "submission_cycles.json"),
                "proposer_trajectory": str(artifact_root / "proposer_trajectory.json"),
                "reviewer_trajectories": str(artifact_root / "reviewer_trajectories.json"),
                "review_statuses": str(artifact_root / "review_statuses.json"),
                "final_review_items": str(artifact_root / "final_review_items.json"),
                "final_answer": str(artifact_root / "final_answer.md"),
                **(
                    {"final_submission": str(artifact_root / "final_submission.json")}
                    if acceptance_status == "accepted"
                    else {}
                ),
            },
            "time_elapsed": 0.33,
        }
    )
    save_json(result.model_dump(exclude_none=True), artifact_root / "qa_result.json")

    if missing_top_level_artifact:
        target = artifact_root / missing_top_level_artifact
        if target.exists():
            target.unlink()

    return result


class QALiveValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(".cache") / f"qa_live_validation_{uuid.uuid4().hex[:8]}"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_validate_live_qa_reports_pass_real_evidence(self):
        artifact_dir = self.temp_dir / "evidence"
        report = validate_live_qa(
            "How does Pt/C affect HER activity in 1 M KOH?",
            system=_ReactReviewedSystem(entity_resolved_count=2),
            artifact_dir=str(artifact_dir),
            perform_network_probe=False,
        )

        self.assertEqual(report.category, "PASS_REAL_EVIDENCE")
        self.assertGreaterEqual(report.citation_count, 1)
        self.assertGreaterEqual(report.accepted_claim_count, 1)
        self.assertTrue(report.citation_paper_record_matches)
        self.assertTrue(Path(report.report_path).exists())

    def test_validate_live_qa_reports_pass_degraded_when_provider_is_blocked(self):
        artifact_dir = self.temp_dir / "degraded"
        report = validate_live_qa(
            "How does Pt/C affect HER activity in 1 M KOH?",
            system=_ReactReviewedSystem(degraded=True),
            artifact_dir=str(artifact_dir),
            perform_network_probe=False,
        )

        self.assertEqual(report.category, "PASS_DEGRADED")
        self.assertTrue(report.provider_failure_detected)
        self.assertTrue(report.insufficient_evidence)
        self.assertEqual(report.citation_count, 0)

    def test_react_reviewed_reports_pass_real_evidence_when_protocol_closes(self):
        artifact_dir = self.temp_dir / "react_real"
        report = validate_live_qa(
            "How does Pt/C affect HER activity in 1 M KOH?",
            system=_ReactReviewedSystem(entity_resolved_count=2),
            artifact_dir=str(artifact_dir),
            perform_network_probe=False,
        )

        self.assertEqual(report.category, "PASS_REAL_EVIDENCE")
        self.assertEqual(report.workflow_mode, "react_reviewed")
        self.assertTrue(report.protocol_ok)
        self.assertEqual(report.review_completion_status, "completed")
        self.assertTrue(report.proposer_step_ref_integrity_ok)
        self.assertTrue(report.review_anchor_integrity_ok)
        self.assertTrue(report.reviewer_budget_integrity_ok)
        self.assertTrue(report.router_ok)
        self.assertTrue(report.entity_ok)
        self.assertEqual(2, report.entity_resolved_count)

    def test_react_reviewed_reports_pass_degraded_when_provider_degrades_but_protocol_closes(self):
        artifact_dir = self.temp_dir / "react_degraded"
        report = validate_live_qa(
            "What are frontier strategies for alkaline HER catalysts?",
            system=_ReactReviewedSystem(degraded=True),
            artifact_dir=str(artifact_dir),
            perform_network_probe=False,
        )

        self.assertEqual(report.category, "PASS_DEGRADED")
        self.assertTrue(report.protocol_ok)
        self.assertTrue(report.provider_failure_detected)
        self.assertEqual(report.review_completion_status, "completed")

    def test_react_reviewed_rejected_path_can_close_without_final_submission(self):
        artifact_dir = self.temp_dir / "react_rejected_protocol"
        report = validate_live_qa(
            "What is the molecular formula of ethanol?",
            system=_ReactReviewedSystem(
                acceptance_status="rejected",
                review_completion_status="completed",
            ),
            artifact_dir=str(artifact_dir),
            perform_network_probe=False,
        )

        self.assertTrue(report.protocol_ok)
        self.assertTrue(report.required_artifacts["candidate_submission.json"])
        self.assertTrue(report.required_artifacts["acceptance_decision.json"])
        self.assertNotIn("final_submission.json", report.missing_artifacts)

    def test_react_reviewed_salvaged_reviewer_counts_as_complete(self):
        artifact_dir = self.temp_dir / "react_salvaged"
        report = validate_live_qa(
            "How does Pt/C affect HER activity in 1 M KOH?",
            system=_ReactReviewedSystem(
                reviewer_status_overrides={"counterevidence": "salvaged"},
                review_completion_status="completed",
            ),
            artifact_dir=str(artifact_dir),
            perform_network_probe=False,
        )

        self.assertTrue(report.protocol_ok)
        self.assertEqual("salvaged", report.reviewer_status_by_role["counterevidence"])
        self.assertEqual("completed", report.review_completion_status)

    def test_react_reviewed_rejected_path_uses_candidate_submission_for_trace_checks(self):
        artifact_dir = self.temp_dir / "react_rejected_mismatch"
        report = validate_live_qa(
            "What is the molecular formula of ethanol?",
            system=_ReactReviewedSystem(
                acceptance_status="rejected",
                review_completion_status="completed",
                submission_alignment_ok=False,
            ),
            artifact_dir=str(artifact_dir),
            perform_network_probe=False,
        )

        self.assertFalse(report.protocol_ok)
        self.assertEqual("submission_trace", report.workflow_protocol_stage)

    def test_react_reviewed_fails_when_required_artifact_is_missing(self):
        artifact_dir = self.temp_dir / "react_missing_artifact"
        report = validate_live_qa(
            "How does Pt/C affect HER activity in 1 M KOH?",
            system=_ReactReviewedSystem(missing_top_level_artifact="final_review_items.json"),
            artifact_dir=str(artifact_dir),
            perform_network_probe=False,
        )

        self.assertEqual(report.category, "FAIL_PIPELINE")
        self.assertIn("final_review_items.json", report.missing_artifacts)

    def test_react_reviewed_fails_when_reviewer_role_is_missing(self):
        artifact_dir = self.temp_dir / "react_missing_role"
        report = validate_live_qa(
            "How does Pt/C affect HER activity in 1 M KOH?",
            system=_ReactReviewedSystem(missing_role="counterevidence"),
            artifact_dir=str(artifact_dir),
            perform_network_probe=False,
        )

        self.assertEqual(report.category, "FAIL_PIPELINE")
        self.assertFalse(report.protocol_ok)
        self.assertEqual(report.reviewer_status_by_role["counterevidence"], "missing")

    def test_react_reviewed_fails_when_review_completion_is_incomplete(self):
        artifact_dir = self.temp_dir / "react_incomplete"
        report = validate_live_qa(
            "How does Pt/C affect HER activity in 1 M KOH?",
            system=_ReactReviewedSystem(review_completion_status="incomplete"),
            artifact_dir=str(artifact_dir),
            perform_network_probe=False,
        )

        self.assertEqual(report.category, "FAIL_PIPELINE")
        self.assertFalse(report.protocol_ok)
        self.assertEqual(report.review_completion_status, "incomplete")

    def test_react_reviewed_fails_when_submission_trace_mismatches_final_submission(self):
        artifact_dir = self.temp_dir / "react_mismatch"
        report = validate_live_qa(
            "How does Pt/C affect HER activity in 1 M KOH?",
            system=_ReactReviewedSystem(submission_alignment_ok=False),
            artifact_dir=str(artifact_dir),
            perform_network_probe=False,
        )

        self.assertEqual(report.category, "FAIL_PIPELINE")
        self.assertFalse(report.protocol_ok)
        self.assertEqual(report.workflow_protocol_stage, "submission_trace")

    def test_react_reviewed_fails_when_review_anchor_cannot_be_resolved(self):
        artifact_dir = self.temp_dir / "react_bad_anchor"
        report = validate_live_qa(
            "How does Pt/C affect HER activity in 1 M KOH?",
            system=_ReactReviewedSystem(anchor_ok=False),
            artifact_dir=str(artifact_dir),
            perform_network_probe=False,
        )

        self.assertEqual(report.category, "FAIL_PIPELINE")
        self.assertFalse(report.protocol_ok)
        self.assertFalse(report.review_anchor_integrity_ok)
        self.assertGreaterEqual(report.open_blocking_review_item_count, 1)

    def test_react_reviewed_fails_when_budget_artifact_exceeds_limit(self):
        artifact_dir = self.temp_dir / "react_bad_budget"
        report = validate_live_qa(
            "How does Pt/C affect HER activity in 1 M KOH?",
            system=_ReactReviewedSystem(budget_ok=False),
            artifact_dir=str(artifact_dir),
            perform_network_probe=False,
        )

        self.assertEqual(report.category, "FAIL_PIPELINE")
        self.assertFalse(report.protocol_ok)
        self.assertFalse(report.reviewer_budget_integrity_ok)

    def test_react_reviewed_case_expectations_mark_extended_degraded_as_nonblocking(self):
        artifact_dir = self.temp_dir / "react_case"
        case = LiveValidationCase.model_validate(
            {
                "case_id": "frontier_case",
                "tier": "extended",
                "question_type": "frontier",
                "question": "What are frontier strategies for alkaline HER catalysts?",
                "expected_category": ["PASS_REAL_EVIDENCE", "PASS_DEGRADED"],
                "require_protocol_ok": True,
                "require_review_completion": True,
                "allow_provider_degraded": True,
                "require_grounding_ok": True,
                "require_router_no_fallback": True,
                "require_router_question_type_match": True,
            }
        )
        report = validate_live_qa(
            case.question,
            system=_ReactReviewedSystem(
                degraded=True,
                router_question_type="frontier",
                router_recency_policy="last_3y",
                entity_resolved_count=1,
            ),
            artifact_dir=str(artifact_dir),
            perform_network_probe=False,
            case=case,
        )

        self.assertEqual(report.category, "PASS_DEGRADED")
        self.assertTrue(report.meets_expectations)

    def test_react_reviewed_core_case_fails_expectations_when_router_uses_fallback(self):
        artifact_dir = self.temp_dir / "react_router_fallback"
        case = LiveValidationCase.model_validate(
            {
                "case_id": "mechanism_ptc_her_koh",
                "tier": "core",
                "question_type": "mechanism",
                "question": "What mechanism is commonly proposed for Pt/C-catalyzed HER in 1 M KOH?",
                "expected_category": "PASS_REAL_EVIDENCE",
                "require_protocol_ok": True,
                "require_review_completion": True,
                "require_real_evidence": True,
                "max_open_blocking_review_items": 0,
                "require_grounding_ok": True,
                "require_router_no_fallback": True,
                "require_router_question_type_match": True,
                "require_entity_not_all_unresolved": True,
            }
        )
        report = validate_live_qa(
            case.question,
            system=_ReactReviewedSystem(router_used_fallback=True, router_question_type="mechanism", entity_resolved_count=1),
            artifact_dir=str(artifact_dir),
            perform_network_probe=False,
            case=case,
        )

        self.assertEqual("PASS_REAL_EVIDENCE", report.category)
        self.assertFalse(report.meets_expectations)
        self.assertTrue(report.router_used_fallback)
        self.assertEqual("router_failure", report.grounding_failure_category)

    def test_react_reviewed_fact_case_requires_pubchem_call(self):
        artifact_dir = self.temp_dir / "react_pubchem_gate"
        case = LiveValidationCase.model_validate(
            {
                "case_id": "fact_ethanol_formula",
                "tier": "core",
                "question_type": "fact",
                "question": "What is the molecular formula of ethanol?",
                "expected_category": "PASS_REAL_EVIDENCE",
                "require_protocol_ok": True,
                "require_review_completion": True,
                "require_real_evidence": True,
                "max_open_blocking_review_items": 0,
                "require_grounding_ok": True,
                "require_router_no_fallback": True,
                "require_router_question_type_match": True,
                "require_entity_not_all_unresolved": True,
                "require_pubchem_call": True,
            }
        )
        report = validate_live_qa(
            case.question,
            system=_ReactReviewedSystem(router_question_type="fact", entity_resolved_count=1, pubchem_call_count=0),
            artifact_dir=str(artifact_dir),
            perform_network_probe=False,
            case=case,
        )

        self.assertFalse(report.meets_expectations)
        self.assertEqual(0, report.pubchem_call_count)
        self.assertEqual("entity_resolution_failure", report.grounding_failure_category)

    def test_react_reviewed_pubchem_error_prevents_pass_real_evidence(self):
        artifact_dir = self.temp_dir / "react_pubchem_error"
        report = validate_live_qa(
            "What is the molecular formula of ethanol?",
            system=_ReactReviewedSystem(
                router_question_type="fact",
                entity_resolved_count=1,
                pubchem_call_count=1,
                pubchem_error_count=1,
            ),
            artifact_dir=str(artifact_dir),
            perform_network_probe=False,
        )

        self.assertNotEqual("PASS_REAL_EVIDENCE", report.category)
        self.assertTrue(report.provider_failure_detected)
        self.assertEqual(1, report.pubchem_error_count)

    def test_react_reviewed_frontier_recent_case_requires_non_none_recency(self):
        artifact_dir = self.temp_dir / "react_frontier_recency"
        case = LiveValidationCase.model_validate(
            {
                "case_id": "frontier_recent_progress_alkaline_her",
                "tier": "extended",
                "question_type": "frontier",
                "question": "What has been the recent progress in alkaline HER catalysts beyond Pt/C?",
                "expected_category": ["PASS_REAL_EVIDENCE", "PASS_DEGRADED"],
                "require_protocol_ok": True,
                "require_review_completion": True,
                "allow_provider_degraded": True,
                "require_grounding_ok": True,
                "require_router_no_fallback": True,
                "require_router_question_type_match": True,
                "require_router_non_none_recency": True,
            }
        )
        report = validate_live_qa(
            case.question,
            system=_ReactReviewedSystem(router_question_type="frontier", router_recency_policy="none", entity_resolved_count=1),
            artifact_dir=str(artifact_dir),
            perform_network_probe=False,
            case=case,
        )

        self.assertFalse(report.meets_expectations)
        self.assertFalse(report.router_recency_policy_ok)

    def test_react_reviewed_question_type_mismatch_breaks_grounding_expectation(self):
        artifact_dir = self.temp_dir / "react_qtype_mismatch"
        case = LiveValidationCase.model_validate(
            {
                "case_id": "comparison_ptc_vs_nimo_her",
                "tier": "core",
                "question_type": "comparison",
                "question": "How does Pt/C compare with NiMo catalysts for HER activity in alkaline media?",
                "expected_category": "PASS_REAL_EVIDENCE",
                "require_protocol_ok": True,
                "require_review_completion": True,
                "require_real_evidence": True,
                "max_open_blocking_review_items": 0,
                "require_grounding_ok": True,
                "require_router_no_fallback": True,
                "require_router_question_type_match": True,
                "require_entity_not_all_unresolved": True,
            }
        )
        report = validate_live_qa(
            case.question,
            system=_ReactReviewedSystem(router_question_type="mechanism", entity_resolved_count=1),
            artifact_dir=str(artifact_dir),
            perform_network_probe=False,
            case=case,
        )

        self.assertFalse(report.meets_expectations)
        self.assertFalse(report.router_question_type_match)
        self.assertEqual("router_failure", report.grounding_failure_category)

    def test_react_reviewed_all_unresolved_entities_break_grounding_expectation(self):
        artifact_dir = self.temp_dir / "react_all_unresolved"
        case = LiveValidationCase.model_validate(
            {
                "case_id": "mechanism_ptc_her_koh",
                "tier": "core",
                "question_type": "mechanism",
                "question": "What mechanism is commonly proposed for Pt/C-catalyzed HER in 1 M KOH?",
                "expected_category": "PASS_REAL_EVIDENCE",
                "require_protocol_ok": True,
                "require_review_completion": True,
                "require_real_evidence": True,
                "max_open_blocking_review_items": 0,
                "require_grounding_ok": True,
                "require_router_no_fallback": True,
                "require_router_question_type_match": True,
                "require_entity_not_all_unresolved": True,
            }
        )
        report = validate_live_qa(
            case.question,
            system=_ReactReviewedSystem(router_question_type="mechanism", entity_resolved_count=0, entity_unresolved_count=3),
            artifact_dir=str(artifact_dir),
            perform_network_probe=False,
            case=case,
        )

        self.assertFalse(report.meets_expectations)
        self.assertTrue(report.entity_all_mentions_unresolved)
        self.assertEqual("entity_resolution_failure", report.grounding_failure_category)

    def test_live_validation_cli_returns_nonzero_for_pipeline_failure(self):
        config_path = self.temp_dir / "config.yaml"
        config_path.write_text(
            "\n".join(
                [
                    "paths:",
                    f"  outputs: \"{self.temp_dir.as_posix()}\"",
                    "qa:",
                    "  save_output: false",
                    "  outputs_dir: \"\"",
                    "  artifact_subdir: \"qa_artifacts\"",
                    "  enable_peer_review: true",
                ]
            ),
            encoding="utf-8",
        )
        artifact_root = self.temp_dir / "suite"

        def _factory(**_kwargs):
            return _BrokenSystem()

        exit_code = live_validation_main(
            [
                "--question",
                "How does Pt/C affect HER activity in 1 M KOH?",
                "--artifact-root",
                str(artifact_root),
                "--config",
                str(config_path),
            ],
            system_factory=_factory,
            configure_logging=False,
            perform_network_probe=False,
        )

        self.assertEqual(exit_code, 1)
        suite_report = artifact_root / "live_validation_suite.json"
        self.assertTrue(suite_report.exists())

    def test_suite_mode_only_blocks_on_core_failures(self):
        config_path = self.temp_dir / "suite_config.yaml"
        config_path.write_text(
            "\n".join(
                [
                    "qa:",
                    "  workflow_mode: react_reviewed",
                    "  save_output: false",
                ]
            ),
            encoding="utf-8",
        )
        suite_path = self.temp_dir / "suite_cases.yaml"
        suite_path.write_text(
            "\n".join(
                [
                    "cases:",
                    "  - case_id: core_fact",
                    "    tier: core",
                    "    question_type: fact",
                    "    question: Core fact question",
                    "    expected_category: PASS_REAL_EVIDENCE",
                    "    require_protocol_ok: true",
                    "    require_review_completion: true",
                    "    require_real_evidence: true",
                    "    max_open_blocking_review_items: 0",
                    "    require_grounding_ok: true",
                    "    require_router_no_fallback: true",
                    "    require_router_question_type_match: true",
                    "    require_entity_not_all_unresolved: true",
                    "  - case_id: frontier_extended",
                    "    tier: extended",
                    "    question_type: frontier",
                    "    question: Frontier question",
                    "    expected_category:",
                    "      - PASS_REAL_EVIDENCE",
                    "      - PASS_DEGRADED",
                    "    require_protocol_ok: true",
                    "    require_review_completion: true",
                    "    allow_provider_degraded: true",
                    "    require_grounding_ok: true",
                    "    require_router_no_fallback: true",
                    "    require_router_question_type_match: true",
                    "    require_router_non_none_recency: true",
                ]
            ),
            encoding="utf-8",
        )

        scenario_by_question = {
            "Core fact question": {"router_question_type": "fact", "entity_resolved_count": 1},
            "Frontier question": {
                "degraded": True,
                "router_question_type": "frontier",
                "router_recency_policy": "last_3y",
                "entity_resolved_count": 1,
            },
        }

        def _factory(**_kwargs):
            return _DispatchingSuiteSystem(scenario_by_question)

        exit_code = live_validation_main(
            [
                "--suite-file",
                str(suite_path),
                "--artifact-root",
                str(self.temp_dir / "suite_run"),
                "--config",
                str(config_path),
            ],
            system_factory=_factory,
            configure_logging=False,
            perform_network_probe=False,
        )

        self.assertEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()
