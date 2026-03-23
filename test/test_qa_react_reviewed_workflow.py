from __future__ import annotations

import json
import shutil
import threading
import time
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agents.react_agent import AgentResponse
from agents.react_reasoning import ReActTrajectory
from pydantic import ValidationError
from qa.artifacts import QAArtifactStore
from qa.facade import QASystem
from qa.handoff import EvidenceExtractorHandoff
from qa.nodes.retriever import RetrieverNode
from qa.react_reviewed_state import AnswerSubmission, ReviewItem, ReviewerRunStatus, SubmissionCitation, SubmissionConfidenceRating, SubmissionSection, SubmissionStepRef
from qa.react_reviewed_workflow import (
    PROPOSER_TOOL_NAMES,
    ReactReviewedProposerExecutionError,
    ReactReviewedProposerAgent,
    ReactReviewedReviewerAgent,
    ReactReviewedStructuredOutputError,
    ReactReviewedWorkflow,
    ReactReviewedWorkspace,
    ReviewerBudgetBlocked,
    ReviewerBudgetState,
    ReviewerSession,
    _ProposerRunState,
)
from qa.retrieval_state import EvidenceItem, PaperCandidate, PaperRecord, QueryPlan, Section, SectionIndex
from qa.runtime import resolve_qa_runtime_config
from qa.state import AnswerSection, EntityPack, SourceSpan, TaskSpec
from qa.synthesis_state import QAResult


def _confidence(score: float = 0.8) -> dict:
    return {
        "level": "high" if score >= 0.75 else "medium" if score >= 0.45 else "low",
        "score": score,
        "rationale": "test fixture",
    }


class _FakeReactReviewedWorkflow:
    def __init__(self) -> None:
        self.calls = []

    def run(self, *, question: str, context=None, artifact_dir=None):
        artifact_root = Path(artifact_dir)
        artifact_root.mkdir(parents=True, exist_ok=True)
        final_submission_path = artifact_root / "final_submission.json"
        final_submission_path.write_text("{}", encoding="utf-8")
        final_answer_path = artifact_root / "final_answer.md"
        final_answer_text = "## Direct Answer\nPt/C improves HER activity under the cited conditions."
        final_answer_path.write_text(final_answer_text, encoding="utf-8")
        result = QAResult.model_validate(
            {
                "question": question,
                "language": "en",
                "workflow_mode": "react_reviewed",
                "final_answer": final_answer_text,
                "sections": [
                    {
                        "section_id": "direct_answer",
                        "title": "Direct Answer",
                        "content": "Pt/C improves HER activity under the cited conditions.",
                        "citation_ids": ["CIT-1"],
                        "section_confidence": _confidence(),
                    }
                ],
                "citations": [
                    {
                        "citation_id": "CIT-1",
                        "paper_id": "paper-1",
                        "title": "Pt/C HER in alkaline media",
                        "year": 2024,
                        "supporting_claim_ids": [],
                    }
                ],
                "claim_trace": [],
                "submission_trace": [
                    {
                        "section_id": "direct_answer",
                        "citation_ids": ["CIT-1"],
                        "step_refs": [{"trajectory_id": "traj_1", "step_number": 1}],
                        "issue_refs": [],
                    }
                ],
                "review_completion_status": "completed",
                "overall_confidence": _confidence(),
                "section_confidence": [
                    {
                        "section_id": "direct_answer",
                        "title": "Direct Answer",
                        "confidence": _confidence(),
                    }
                ],
                "insufficient_evidence": False,
                "limitations_summary": "",
                "retrieval_diagnostics_summary": "",
                "execution_warnings": ["workflow warning"],
                "artifact_paths": {
                    "qa_result": str(artifact_root / "qa_result.json"),
                    "final_answer": str(final_answer_path),
                    "final_submission": str(final_submission_path),
                },
                "time_elapsed": 0.05,
            }
        )
        (artifact_root / "qa_result.json").write_text(
            json.dumps(result.model_dump(exclude_none=True), ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        self.calls.append({"question": question, "context": context, "artifact_dir": str(artifact_root)})
        return result


def _task_spec(question: str = "How does Pt/C affect HER activity?") -> TaskSpec:
    return TaskSpec.model_validate(
        {
            "question": question,
            "normalized_question": question.lower(),
            "question_type": "fact",
            "recency_policy": "none",
            "answer_sections": [
                {
                    "section_id": "direct_answer",
                    "title": "Direct Answer",
                    "required": True,
                    "instruction": "Answer directly.",
                }
            ],
            "router_confidence": 0.9,
        }
    )


def _entity_pack() -> EntityPack:
    return EntityPack.model_validate({})


def _entity_resolution_snapshot() -> dict:
    return {
        "resolution_index": {
            "entries": [
                {
                    "entry_id": "res_1",
                    "entity_type": "solvent",
                    "canonical_name": "ethanol",
                    "aliases": ["EtOH"],
                    "query_anchors": ["ethanol", "EtOH", "C2H6O"],
                    "formula": "C2H6O",
                    "smiles": "CCO",
                    "inchikey": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
                    "pubchem_cid": 702,
                    "resolver_source": "pubchem",
                    "resolution_confidence": 0.94,
                    "status": "resolved",
                    "lookup_keys": ["ethanol", "etoh", "c2h6o"],
                }
            ],
            "cache_events": [],
        },
        "provider_calls": [
            {
                "provider": "pubchem",
                "query": "EtOH",
                "status": "hit",
                "candidate_count": 1,
                "max_candidates": 5,
            }
        ],
        "seed_suggestions": [],
    }


def _submission_confidence(score: float = 0.8) -> SubmissionConfidenceRating:
    return SubmissionConfidenceRating(level="high" if score >= 0.75 else "medium", score=score, rationale="fixture")


def _trajectory(query: str = "fixture") -> ReActTrajectory:
    trajectory = ReActTrajectory(query=query)
    trajectory.finalize("{}")
    return trajectory


def _submission(question: str, *, cycle_number: int = 1, trajectory_id: str = "traj_fixture") -> AnswerSubmission:
    return AnswerSubmission(
        submission_id=f"submission_cycle_{cycle_number}",
        question=question,
        version=cycle_number,
        sections=[
            SubmissionSection(
                section_id="direct_answer",
                title="Direct Answer",
                content="Pt/C improves HER activity under the cited conditions.",
                citation_ids=["CIT-1"],
                step_refs=[SubmissionStepRef(trajectory_id=trajectory_id, step_number=1)],
                issue_refs=[],
                section_confidence=_submission_confidence(),
            )
        ],
        citations=[
            SubmissionCitation(
                citation_id="CIT-1",
                paper_id="paper-1",
                title="Pt/C HER in alkaline media",
                year=2024,
                section_ids=["sec_results"],
                evidence_ids=["ev-1"],
            )
        ],
        limitations=[],
        overall_confidence=_submission_confidence(),
        trajectory_id=trajectory_id,
        step_refs=[SubmissionStepRef(trajectory_id=trajectory_id, step_number=1)],
        issue_refs=[],
    )


def _read_json(path: str) -> object:
    return json.loads(Path(path).read_text(encoding="utf-8"))


class _StaticRouterNode:
    def run(self, *, question: str, context=None):
        return _task_spec(question)


class _StaticEntityResolverNode:
    def run(self, *, question: str, task_spec: TaskSpec):
        return _entity_pack()


class _NoopQueryPlanner:
    def run(self, *, task_spec: TaskSpec, entity_pack: EntityPack):
        return []


class _CountingRetriever:
    def __init__(self, *, delay: float = 0.0) -> None:
        self.delay = delay
        self.calls = 0
        self.lock = threading.Lock()
        self.last_diagnostics = []
        self.last_provider_health = {}

    def run(self, *, task_spec: TaskSpec, entity_pack: EntityPack, query_plans, artifact_store=None):
        with self.lock:
            self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        return [
            PaperCandidate(
                paper_id="paper-1",
                title="Pt/C HER in alkaline media",
                abstract="Pt/C improves HER activity in alkaline media.",
                year=2024,
                provider_hits=["openalex"],
                lane_sources=["contrarian"],
                retrieval_score=0.9,
            )
        ]


class _CountingDocumentAcquirer:
    def __init__(self, *, delay: float = 0.0) -> None:
        self.delay = delay
        self.calls = 0
        self.lock = threading.Lock()
        self.last_diagnostics = []
        self.last_provider_health = {}
        self.last_execution_warnings = []

    def run(self, *, candidates, artifact_store=None):
        with self.lock:
            self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        candidate = candidates[0]
        store = artifact_store or QAArtifactStore()
        fulltext = "Pt/C improves HER activity in alkaline media."
        fulltext_path = store.write_text(f"fulltext/{candidate.paper_id}.txt", fulltext)
        return (
            [
                PaperRecord(
                    paper_id=candidate.paper_id,
                    doi=candidate.doi,
                    title=candidate.title,
                    abstract=candidate.abstract,
                    authors=list(candidate.authors),
                    year=candidate.year,
                    venue=candidate.venue,
                    provider_sources=list(candidate.provider_hits),
                    provider_artifacts=dict(candidate.provider_artifacts),
                    oa_url=candidate.oa_url,
                    fulltext_available=True,
                    fulltext_status="fulltext_indexed",
                    fulltext_artifact_path=fulltext_path,
                )
            ],
            [
                SectionIndex(
                    paper_id=candidate.paper_id,
                    fulltext_status="fulltext_indexed",
                    sections=[
                        Section(
                            section_id="sec_results",
                            section_type="results",
                            heading="Results",
                            fulltext_char_start=0,
                            fulltext_char_end=len(fulltext),
                        )
                    ],
                )
            ],
        )


class _CountingEvidenceExtractor:
    def __init__(self, *, delay: float = 0.0) -> None:
        self.delay = delay
        self.calls = 0
        self.lock = threading.Lock()

    def _make_item(self, paper_id: str, evidence_id: str) -> EvidenceItem:
        return EvidenceItem(
            evidence_id=evidence_id,
            paper_id=paper_id,
            section_id="sec_results",
            section_type="results",
            role="observation",
            snippet="Pt/C improves HER activity.",
            source_span=SourceSpan(start=0, end=12),
            source_layer="fulltext",
            claim_polarity="support",
            extraction_confidence=0.9,
        )

    def run(self, *, task_spec: TaskSpec, entity_pack: EntityPack, paper_record: PaperRecord, section_index: SectionIndex):
        with self.lock:
            self.calls += 1
            evidence_id = f"ev-{self.calls}"
        if self.delay:
            time.sleep(self.delay)
        return [self._make_item(paper_record.paper_id, evidence_id)]

    def _extract_from_section(self, *, task_spec: TaskSpec, entity_pack: EntityPack, paper_record: PaperRecord, section_view):
        return self.run(
            task_spec=task_spec,
            entity_pack=entity_pack,
            paper_record=paper_record,
            section_index=SectionIndex(paper_id=paper_record.paper_id, fulltext_status="fulltext_indexed", sections=[]),
        )


class _FakeProposer:
    def run(self, *, workspace: ReactReviewedWorkspace, cycle_number: int, open_review_items):
        trajectory = _trajectory(f"proposer cycle {cycle_number}")
        return _submission(workspace.question, cycle_number=cycle_number, trajectory_id=trajectory.trajectory_id), trajectory


class _FailingProposer:
    def run(self, *, workspace: ReactReviewedWorkspace, cycle_number: int, open_review_items):
        error = ReactReviewedProposerExecutionError(
            stage="proposer_repair",
            cycle_number=cycle_number,
            message="synthetic proposer failure",
            details={"source": "test"},
            trajectory=_trajectory(f"failing proposer cycle {cycle_number}"),
        )
        raise error


class _ParallelReviewer:
    def __init__(self, reviewer_role: str, barrier: threading.Barrier, delay_seconds: float, events: list[tuple[str, str]]) -> None:
        self.reviewer_role = reviewer_role
        self.barrier = barrier
        self.delay_seconds = delay_seconds
        self.events = events

    def run(self, *, workspace, submission, proposer_trajectory, cycle_number, session):
        self.events.append((self.reviewer_role, threading.current_thread().name))
        self.barrier.wait(timeout=3)
        if self.delay_seconds:
            time.sleep(self.delay_seconds)
        trajectory = _trajectory(f"{self.reviewer_role} review")
        return (
            [],
            trajectory,
            ReviewerRunStatus(
                reviewer_role=self.reviewer_role,
                status="completed",
                message="completed",
                cycle_number=cycle_number,
                retrieval_actions_used=session.budget_state.actions_used,
                retrieval_budget_limit=session.budget_state.budget_limit,
                budget_blocked_calls=session.budget_state.blocked_calls,
            ),
        )


class _SearchingReviewer:
    def __init__(self, reviewer_role: str) -> None:
        self.reviewer_role = reviewer_role

    def run(self, *, workspace, submission, proposer_trajectory, cycle_number, session):
        workspace.search_papers(
            query_text="Pt/C HER alkaline",
            lane="review",
            artifact_store=session.artifact_store,
            session=session,
            charge_budget=False,
            requested_via="search_papers",
            write_snapshot=True,
        )
        trajectory = _trajectory(f"{self.reviewer_role} review")
        return (
            [],
            trajectory,
            ReviewerRunStatus(
                reviewer_role=self.reviewer_role,
                status="completed",
                message="completed",
                cycle_number=cycle_number,
                retrieval_actions_used=session.budget_state.actions_used,
                retrieval_budget_limit=session.budget_state.budget_limit,
                budget_blocked_calls=session.budget_state.blocked_calls,
            ),
        )


class ReactReviewedRuntimeConfigTests(unittest.TestCase):
    def test_resolve_runtime_config_defaults_to_react_reviewed(self):
        resolved = resolve_qa_runtime_config({"qa": {}})

        self.assertEqual("react_reviewed", resolved["workflow_mode"])
        self.assertEqual(7, resolved["react_reviewed"]["max_propose_steps_initial"])
        self.assertEqual(7, resolved["react_reviewed"]["max_propose_steps_revision"])
        self.assertEqual("fail_fast_only", resolved["react_reviewed"]["proposer_fallback_mode"])
        self.assertEqual(1, resolved["react_reviewed"]["proposer_repair_attempts"])
        self.assertEqual("prefer_fulltext", resolved["react_reviewed"]["proposer_evidence_policy"])
        self.assertEqual(3, resolved["react_reviewed"]["max_review_cycles"])
        self.assertEqual(4, resolved["react_reviewed"]["reviewer_max_concurrency"])
        self.assertEqual(
            {
                "search_coverage": 1,
                "evidence_trace": 0,
                "reasoning_consistency": 0,
                "counterevidence": 2,
            },
            resolved["react_reviewed"]["reviewer_retrieval_budget_by_role"],
        )
        self.assertTrue(resolved["react_reviewed"]["review_failure_blocks_acceptance"])

    def test_qa_result_accepts_submission_trace_fields(self):
        result = QAResult.model_validate(
            {
                "question": "How does Pt/C affect HER activity?",
                "language": "en",
                "workflow_mode": "react_reviewed",
                "acceptance_status": "rejected",
                "final_answer": "## Direct Answer\nPt/C improves HER activity.",
                "sections": [],
                "citations": [],
                "claim_trace": [],
                "submission_trace": [
                    {
                        "section_id": "direct_answer",
                        "citation_ids": ["CIT-1"],
                        "step_refs": [{"trajectory_id": "traj_1", "step_number": 1}],
                        "issue_refs": [],
                    }
                ],
                "review_completion_status": "incomplete",
                "overall_confidence": _confidence(0.4),
                "section_confidence": [],
                "insufficient_evidence": True,
                "limitations_summary": "Reviewer completion was incomplete.",
                "retrieval_diagnostics_summary": "",
                "execution_warnings": [],
                "artifact_paths": {},
                "time_elapsed": 0.1,
            }
        )

        self.assertEqual("react_reviewed", result.workflow_mode)
        self.assertEqual("incomplete", result.review_completion_status)
        self.assertEqual("rejected", result.acceptance_status)
        self.assertEqual(1, len(result.submission_trace))


class ReactReviewedDispatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(".cache") / f"qa_react_reviewed_{uuid.uuid4().hex[:8]}"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_qasystem_dispatches_to_react_reviewed_workflow(self):
        fake_workflow = _FakeReactReviewedWorkflow()
        system = QASystem(
            config={
                "paths": {"outputs": str(self.temp_dir)},
                "qa": {
                    "workflow_mode": "react_reviewed",
                    "save_output": False,
                    "outputs_dir": str(self.temp_dir),
                    "artifact_subdir": "qa_artifacts",
                },
                "llm": {},
            },
            react_reviewed_workflow=fake_workflow,
        )

        artifact_dir = self.temp_dir / "artifacts"
        result = system.run_qa(
            question="How does Pt/C affect HER activity in 1 M KOH?",
            artifact_dir=str(artifact_dir),
        )

        self.assertEqual(1, len(fake_workflow.calls))
        self.assertEqual("react_reviewed", result.workflow_mode)
        self.assertIn("runtime_manifest", result.artifact_paths)
        self.assertEqual(str(artifact_dir), fake_workflow.calls[0]["artifact_dir"])

    def test_react_reviewed_public_result_is_written_when_save_output_enabled(self):
        fake_workflow = _FakeReactReviewedWorkflow()
        system = QASystem(
            config={
                "paths": {"outputs": str(self.temp_dir)},
                "qa": {
                    "workflow_mode": "react_reviewed",
                    "save_output": True,
                    "outputs_dir": str(self.temp_dir / "outputs"),
                    "artifact_subdir": "qa_artifacts",
                },
                "llm": {},
            },
            react_reviewed_workflow=fake_workflow,
        )

        result = system.run_qa(
            question="How does Pt/C affect HER activity in 1 M KOH?",
            artifact_dir=str(self.temp_dir / "artifacts_public"),
        )

        self.assertIn("public_result", result.artifact_paths)
        public_payload = _read_json(result.artifact_paths["public_result"])
        self.assertIn("runtime_manifest", public_payload["artifact_paths"])
        self.assertIn("final_submission", public_payload["artifact_paths"])

    def test_react_reviewed_runtime_warnings_are_merged_into_final_result(self):
        fake_workflow = _FakeReactReviewedWorkflow()
        system = QASystem(
            config={
                "paths": {"outputs": str(self.temp_dir)},
                "qa": {
                    "workflow_mode": "react_reviewed",
                    "save_output": False,
                    "outputs_dir": str(self.temp_dir),
                    "artifact_subdir": "qa_artifacts",
                },
                "llm": {},
            },
            react_reviewed_workflow=fake_workflow,
        )

        result = system.run_qa(
            question="How does Pt/C affect HER activity in 1 M KOH?",
            artifact_dir=str(self.temp_dir / "artifacts_warning"),
        )

        self.assertIn("workflow warning", result.execution_warnings)
        self.assertTrue(any("semantic_scholar_api_key" in warning for warning in result.execution_warnings))
        self.assertTrue(any("unpaywall_email" in warning for warning in result.execution_warnings))
        qa_result_payload = _read_json(result.artifact_paths["qa_result"])
        self.assertEqual(result.execution_warnings, qa_result_payload["execution_warnings"])

    def test_save_output_false_skips_public_result_but_keeps_internal_artifacts(self):
        fake_workflow = _FakeReactReviewedWorkflow()
        system = QASystem(
            config={
                "paths": {"outputs": str(self.temp_dir)},
                "qa": {
                    "workflow_mode": "react_reviewed",
                    "save_output": False,
                    "outputs_dir": str(self.temp_dir),
                    "artifact_subdir": "qa_artifacts",
                },
                "llm": {},
            },
            react_reviewed_workflow=fake_workflow,
        )

        result = system.run_qa(
            question="How does Pt/C affect HER activity in 1 M KOH?",
            artifact_dir=str(self.temp_dir / "artifacts_no_public"),
        )

        self.assertNotIn("public_result", result.artifact_paths)
        self.assertTrue(Path(result.artifact_paths["qa_result"]).exists())
        self.assertTrue(Path(result.artifact_paths["final_answer"]).exists())


class ReactReviewedWorkflowExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(".cache") / f"qa_reviewer_parallel_{uuid.uuid4().hex[:8]}"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _make_workspace(
        self,
        *,
        retriever: _CountingRetriever | None = None,
        document_acquirer: _CountingDocumentAcquirer | None = None,
        evidence_extractor: _CountingEvidenceExtractor | None = None,
        entity_resolution_snapshot: dict | None = None,
        stage_watchdog_seconds: float = 120.0,
    ) -> ReactReviewedWorkspace:
        artifact_store = QAArtifactStore(base_dir=self.temp_dir / "workspace")
        return ReactReviewedWorkspace(
            question="How does Pt/C affect HER activity?",
            context=None,
            task_spec=_task_spec(),
            entity_pack=_entity_pack(),
            entity_resolution_snapshot=entity_resolution_snapshot or {},
            artifact_store=artifact_store,
            query_planner=_NoopQueryPlanner(),
            retriever=retriever or _CountingRetriever(),
            document_acquirer=document_acquirer or _CountingDocumentAcquirer(),
            handoff=EvidenceExtractorHandoff(),
            evidence_extractor=evidence_extractor or _CountingEvidenceExtractor(),
            stage_watchdog_seconds=stage_watchdog_seconds,
        )

    def _make_session(self, role: str, budget_limit: int) -> ReviewerSession:
        return ReviewerSession(
            reviewer_role=role,
            cycle_number=1,
            artifact_store=QAArtifactStore(base_dir=self.temp_dir / "sessions" / role),
            budget_state=ReviewerBudgetState(role=role, budget_limit=budget_limit),
        )

    def _make_workflow(self) -> ReactReviewedWorkflow:
        workflow = ReactReviewedWorkflow(
            qa_config={"react_reviewed": {"reviewer_max_concurrency": 4, "reviewer_retrieval_budget_by_role": {
                "search_coverage": 1,
                "evidence_trace": 0,
                "reasoning_consistency": 0,
                "counterevidence": 2,
            }}},
            router=_StaticRouterNode(),
            entity_resolver=_StaticEntityResolverNode(),
            query_planner=_NoopQueryPlanner(),
            retriever=_CountingRetriever(),
            document_acquirer=_CountingDocumentAcquirer(),
            handoff=EvidenceExtractorHandoff(),
            evidence_extractor=_CountingEvidenceExtractor(),
        )
        workflow.proposer = _FakeProposer()
        return workflow

    def test_proposer_prefers_structured_output_over_response_content(self):
        proposer = ReactReviewedProposerAgent(
            model_config={},
            max_steps_initial=6,
            max_steps_revision=6,
            llm_timeout_seconds=45.0,
        )
        payload = _submission("How does Pt/C affect HER activity?").model_dump(exclude_none=True)
        response = AgentResponse(
            content="this is not valid json",
            structured_output={"kind": "submission", "payload": payload},
        )

        parsed = proposer._parse_submission_response(response=response)

        self.assertEqual(payload["submission_id"], parsed["submission_id"])
        self.assertEqual(payload["trajectory_id"], parsed["trajectory_id"])

    def test_validate_submission_payload_normalizes_known_legacy_fields(self):
        proposer = ReactReviewedProposerAgent(
            model_config={},
            max_steps_initial=6,
            max_steps_revision=6,
            llm_timeout_seconds=45.0,
        )
        workspace = self._make_workspace()

        submission = {
            "cycle_number": 1,
            "normalized_question": workspace.question.lower(),
            "conditions": {"electrolyte": "1 M KOH", "catalyst": "Pt/C"},
            "citations": [
                {
                    "citation_id": "CIT-1",
                    "paper_id": "paper-1",
                    "title": "Pt/C HER in alkaline media",
                    "year": 2024,
                    "section_ids": ["sec_results"],
                    "evidence_ids": ["ev-1"],
                }
            ],
            "sections": [
                {
                    "section_id": "direct_answer",
                    "title": "Direct Answer",
                    "content": "Pt/C improves HER activity under the cited conditions.",
                    "citations": ["CIT-1"],
                    "evidence_refs": ["ev-1"],
                }
            ],
        }

        normalized = proposer._validate_submission_payload(
            workspace=workspace,
            submission=submission,
            cycle_number=1,
            open_review_items=[],
            agent=None,
        )

        self.assertEqual(workspace.question, normalized.question)
        self.assertEqual(["CIT-1"], normalized.sections[0].citation_ids)
        self.assertEqual(["CIT-1"], [item.citation_id for item in normalized.citations])

    def test_validate_submission_payload_accepts_answer_sections_and_text_aliases(self):
        proposer = ReactReviewedProposerAgent(
            model_config={},
            max_steps_initial=6,
            max_steps_revision=6,
            llm_timeout_seconds=45.0,
        )
        workspace = self._make_workspace()

        normalized = proposer._validate_submission_payload(
            workspace=workspace,
            submission={
                "answer_sections": [
                    {
                        "section_id": "direct_answer",
                        "title": "Direct Answer",
                        "text": "Pt/C follows an alkaline Volmer-first HER pathway.",
                    }
                ]
            },
            cycle_number=1,
            open_review_items=[],
            agent=None,
        )

        self.assertEqual(1, len(normalized.sections))
        self.assertEqual("Pt/C follows an alkaline Volmer-first HER pathway.", normalized.sections[0].content)

    def test_screen_candidate_papers_prefers_relevant_fulltext_friendly_candidates(self):
        proposer = ReactReviewedProposerAgent(
            model_config={},
            max_steps_initial=6,
            max_steps_revision=6,
            llm_timeout_seconds=45.0,
        )
        workspace = self._make_workspace()
        workspace.entity_pack = EntityPack.model_validate(
            {
                "entities": [
                    {
                        "entity_id": "ent_1",
                        "mention": "Pt/C",
                        "canonical_name": "platinum on carbon",
                        "entity_type": "catalyst",
                        "aliases": ["Pt/C", "platinum on carbon", "pt on carbon"],
                        "query_anchors": ["Pt/C", "platinum on carbon", "pt on carbon"],
                        "resolver_source": "fixture",
                        "resolution_confidence": 0.98,
                        "status": "resolved",
                        "source_text": "Pt/C",
                        "source_span": {"start": 0, "end": 4},
                    }
                ],
                "condition_mentions": [
                    {
                        "condition_id": "cond_1",
                        "axis": "catalyst",
                        "raw_value": "Pt/C",
                        "normalized_value": "platinum on carbon",
                        "confidence": 0.98,
                        "source_text": "Pt/C",
                        "source_span": {"start": 0, "end": 4},
                    }
                ],
            }
        )
        workspace.paper_candidates = {
            "paper-full": PaperCandidate(
                paper_id="paper-full",
                title="Pt/C improves HER activity in 1 M KOH",
                abstract="Pt/C improves HER activity in alkaline electrolyte and reports catalyst-dependent kinetics.",
                doi="10.1000/full",
                provider_hits=["openalex", "semantic_scholar"],
                lane_sources=["data"],
                retrieval_score=8.5,
                oa_url="https://example.org/fulltext",
            ),
            "paper-generic": PaperCandidate(
                paper_id="paper-generic",
                title="Broad review of battery interfaces",
                abstract="A broad review of ethanol oxidation, fuel cells, and battery interfaces.",
                provider_hits=["openalex"],
                lane_sources=["review"],
                retrieval_score=7.9,
            ),
        }

        screened = proposer._screen_candidate_papers(
            workspace=workspace,
            cycle_number=1,
            open_review_items=[],
            paper_ids=["paper-generic", "paper-full"],
            max_candidates=2,
        )

        self.assertEqual("paper-full", screened["locked_paper_ids"][0])
        self.assertIn("paper-generic", screened["dropped_paper_ids"])

    def test_screen_candidate_papers_drops_comparator_only_primary_entity_reference(self):
        proposer = ReactReviewedProposerAgent(
            model_config={},
            max_steps_initial=6,
            max_steps_revision=6,
            llm_timeout_seconds=45.0,
        )
        workspace = self._make_workspace()
        workspace.entity_pack = EntityPack.model_validate(
            {
                "entities": [
                    {
                        "entity_id": "ent_1",
                        "mention": "Pt/C",
                        "canonical_name": "platinum on carbon",
                        "entity_type": "catalyst",
                        "aliases": ["Pt/C", "platinum on carbon", "pt on carbon"],
                        "query_anchors": ["Pt/C", "platinum on carbon", "pt on carbon"],
                        "resolver_source": "fixture",
                        "resolution_confidence": 0.98,
                        "status": "resolved",
                        "source_text": "Pt/C",
                        "source_span": {"start": 0, "end": 4},
                    }
                ],
                "condition_mentions": [
                    {
                        "condition_id": "cond_1",
                        "axis": "catalyst",
                        "raw_value": "Pt/C",
                        "normalized_value": "platinum on carbon",
                        "confidence": 0.98,
                        "source_text": "Pt/C",
                        "source_span": {"start": 0, "end": 4},
                    }
                ],
            }
        )
        workspace.paper_candidates = {
            "paper-target": PaperCandidate(
                paper_id="paper-target",
                title="Pt/C catalyst instability in alkaline medium",
                abstract="Pt/C in alkaline medium loses ECSA rapidly under accelerated cycling.",
                doi="10.1000/target",
                provider_hits=["openalex", "semantic_scholar"],
                lane_sources=["data"],
                retrieval_score=8.2,
                oa_url="https://example.org/target",
            ),
            "paper-comparator": PaperCandidate(
                paper_id="paper-comparator",
                title="Subnanometric Ru clusters improve alkaline hydrogen evolution",
                abstract="Ru clusters show a turnover frequency 36-fold larger than commercial Pt/C in alkaline HER.",
                doi="10.1000/comparator",
                provider_hits=["openalex", "semantic_scholar"],
                lane_sources=["data"],
                retrieval_score=8.0,
                oa_url="https://example.org/comparator",
            ),
        }

        screened = proposer._screen_candidate_papers(
            workspace=workspace,
            cycle_number=1,
            open_review_items=[],
            paper_ids=["paper-comparator", "paper-target"],
            max_candidates=2,
        )

        self.assertEqual(["paper-target"], screened["locked_paper_ids"])
        self.assertIn("paper-comparator", screened["dropped_paper_ids"])

    def test_normalize_submission_citations_does_not_backfill_irrelevant_abstract(self):
        proposer = ReactReviewedProposerAgent(
            model_config={},
            max_steps_initial=6,
            max_steps_revision=6,
            llm_timeout_seconds=45.0,
        )
        workspace = self._make_workspace()
        workspace.paper_candidates["paper-generic"] = PaperCandidate(
            paper_id="paper-generic",
            title="Broad review of battery interfaces",
            abstract="A broad review of ethanol oxidation, fuel cells, and battery interfaces.",
            provider_hits=["openalex"],
            lane_sources=["review"],
            retrieval_score=7.9,
        )
        run_state = _ProposerRunState(evidence_policy="prefer_fulltext")
        run_state.searched_paper_ids.add("paper-generic")
        run_state.acquired_paper_ids.add("paper-generic")
        run_state.fulltext_status_by_paper["paper-generic"] = "fulltext_unusable"

        normalized = proposer._normalize_submission_citations_for_run_state(
            workspace=workspace,
            raw_citations=[
                {
                    "citation_id": "CIT-1",
                    "paper_id": "paper-generic",
                    "title": "Broad review of battery interfaces",
                }
            ],
            run_state=run_state,
        )

        self.assertEqual([], normalized[0]["section_ids"])
        self.assertEqual([], normalized[0]["evidence_ids"])

    def test_validate_submission_payload_backfills_citations_from_run_state(self):
        proposer = ReactReviewedProposerAgent(
            model_config={},
            max_steps_initial=6,
            max_steps_revision=6,
            llm_timeout_seconds=45.0,
        )
        workspace = self._make_workspace()
        workspace.paper_records["paper-1"] = PaperRecord(
            paper_id="paper-1",
            title="Recovered paper",
            doi="10.1000/example",
            year=2024,
            venue="Journal",
            abstract="abstract",
            fulltext_available=True,
            fulltext_status="fulltext_indexed",
        )
        run_state = _ProposerRunState(evidence_policy="prefer_fulltext")
        run_state.query_plan_ids.append("qp_1")
        run_state.searched_paper_ids.add("paper-1")
        run_state.acquired_paper_ids.add("paper-1")
        run_state.fulltext_status_by_paper["paper-1"] = "fulltext_indexed"
        run_state.section_ids_by_paper["paper-1"] = {"sec_results"}

        normalized = proposer._validate_submission_payload(
            workspace=workspace,
            submission={
                "sections": [
                    {
                        "section_id": "direct_answer",
                        "title": "Direct Answer",
                        "content": "Pt/C remains the benchmark in alkaline HER.",
                    },
                ],
                "citations": [],
                "limitations": ["evidence is degraded"],
                "overall_confidence": _confidence(0.3),
            },
            cycle_number=1,
            open_review_items=[],
            agent=None,
            run_state=run_state,
        )

        self.assertEqual(1, len(normalized.citations))
        self.assertEqual("paper-1", normalized.citations[0].paper_id)
        self.assertEqual(["sec_results"], normalized.citations[0].section_ids)

    def test_validate_submission_payload_backfills_abstract_anchor_for_acquired_unusable_fulltext(self):
        proposer = ReactReviewedProposerAgent(
            model_config={},
            max_steps_initial=6,
            max_steps_revision=6,
            llm_timeout_seconds=45.0,
        )
        workspace = self._make_workspace()
        workspace.paper_candidates["paper-1"] = PaperCandidate(
            paper_id="paper-1",
            title="Pt/C HER in alkaline media",
            abstract="Abstract confirms Pt/C improves HER activity in alkaline media.",
            year=2024,
            provider_hits=["openalex"],
            lane_sources=["review"],
            retrieval_score=0.8,
        )
        run_state = _ProposerRunState(evidence_policy="prefer_fulltext")
        run_state.query_plan_ids.append("qp_1")
        run_state.searched_paper_ids.add("paper-1")
        run_state.acquired_paper_ids.add("paper-1")
        run_state.fulltext_status_by_paper["paper-1"] = "fulltext_unusable"

        normalized = proposer._validate_submission_payload(
            workspace=workspace,
            submission={
                "sections": [
                    {
                        "section_id": "direct_answer",
                        "title": "Direct Answer",
                        "content": "Evidence remains limited but the catalyst is active.",
                    },
                ],
                "citations": [],
                "limitations": ["Missing usable full text; abstract-only evidence."],
                "overall_confidence": _confidence(0.2),
            },
            cycle_number=1,
            open_review_items=[],
            agent=None,
            run_state=run_state,
        )

        self.assertEqual(1, len(normalized.citations))
        self.assertEqual("paper-1", normalized.citations[0].paper_id)
        self.assertEqual(["sec_abstract"], normalized.citations[0].section_ids)
        self.assertEqual("Pt/C HER in alkaline media", normalized.citations[0].title)

    def test_validate_submission_payload_inherits_prior_submission_scaffold_for_revision(self):
        proposer = ReactReviewedProposerAgent(
            model_config={},
            max_steps_initial=6,
            max_steps_revision=6,
            llm_timeout_seconds=45.0,
        )
        workspace = self._make_workspace()
        prior_submission = _submission(workspace.question, cycle_number=1, trajectory_id="traj_prior")
        workspace.set_review_context(
            submission=prior_submission,
            proposer_trajectory=_trajectory("prior proposer"),
            open_review_items=[],
            cycle_number=2,
        )
        run_state = _ProposerRunState(evidence_policy="prefer_fulltext")
        run_state.query_plan_ids.append("qp_1")
        run_state.searched_paper_ids.add("paper-1")
        run_state.acquired_paper_ids.add("paper-1")
        run_state.section_ids_by_paper["paper-1"] = {"sec_results"}
        run_state.evidence_ids.add("ev-1")
        run_state.evidence_ids_by_paper["paper-1"] = {"ev-1"}
        run_state.evidence_layers_by_id["ev-1"] = "fulltext"
        run_state.fulltext_status_by_paper["paper-1"] = "fulltext_indexed"

        normalized = proposer._validate_submission_payload(
            workspace=workspace,
            submission={
                "sections": [
                    {
                        "section_id": "direct_answer",
                        "title": "Direct Answer",
                        "content": "Updated answer after review.",
                    }
                ],
                "citations": [],
                "limitations": [],
                "overall_confidence": _confidence(0.5),
            },
            cycle_number=2,
            open_review_items=[],
            agent=None,
            run_state=run_state,
        )

        self.assertEqual(1, len(normalized.citations))
        self.assertEqual("paper-1", normalized.citations[0].paper_id)
        self.assertEqual(["CIT-1"], normalized.sections[0].citation_ids)
        self.assertEqual("traj_prior", normalized.step_refs[0].trajectory_id)

    def test_validate_submission_payload_realigns_step_refs_to_current_trajectory_when_available(self):
        proposer = ReactReviewedProposerAgent(
            model_config={},
            max_steps_initial=6,
            max_steps_revision=6,
            llm_timeout_seconds=45.0,
        )
        workspace = self._make_workspace()
        prior_submission = _submission(workspace.question, cycle_number=1, trajectory_id="traj_prior")
        revision_trajectory = _trajectory("revision proposer")
        workspace.set_review_context(
            submission=prior_submission,
            proposer_trajectory=revision_trajectory,
            open_review_items=[],
            cycle_number=2,
        )
        run_state = _ProposerRunState(evidence_policy="prefer_fulltext")
        run_state.query_plan_ids.append("qp_1")
        run_state.searched_paper_ids.add("paper-1")
        run_state.acquired_paper_ids.add("paper-1")
        run_state.section_ids_by_paper["paper-1"] = {"sec_results"}
        run_state.evidence_ids.add("ev-1")
        run_state.evidence_ids_by_paper["paper-1"] = {"ev-1"}
        run_state.evidence_layers_by_id["ev-1"] = "fulltext"
        run_state.fulltext_status_by_paper["paper-1"] = "fulltext_indexed"

        normalized = proposer._validate_submission_payload(
            workspace=workspace,
            submission={
                "sections": [
                    {
                        "section_id": "direct_answer",
                        "title": "Direct Answer",
                        "content": "Updated answer after review.",
                    }
                ],
                "citations": [],
                "limitations": [],
                "overall_confidence": _confidence(0.5),
            },
            cycle_number=2,
            open_review_items=[],
            agent=None,
            trajectory=revision_trajectory,
            run_state=run_state,
        )

        self.assertEqual(revision_trajectory.trajectory_id, normalized.trajectory_id)
        self.assertEqual(revision_trajectory.trajectory_id, normalized.step_refs[0].trajectory_id)
        self.assertEqual(revision_trajectory.trajectory_id, normalized.sections[0].step_refs[0].trajectory_id)

    def test_validate_submission_payload_auto_normalizes_abstract_only_degradation(self):
        proposer = ReactReviewedProposerAgent(
            model_config={},
            max_steps_initial=6,
            max_steps_revision=6,
            llm_timeout_seconds=45.0,
        )
        workspace = self._make_workspace()
        run_state = _ProposerRunState(evidence_policy="prefer_fulltext")
        run_state.query_plan_ids.append("qp_1")
        run_state.searched_paper_ids.add("paper-abs")
        run_state.acquired_paper_ids.add("paper-abs")
        run_state.fulltext_status_by_paper["paper-abs"] = "abstract_only"
        run_state.section_ids_by_paper["paper-abs"] = {"sec_abstract"}
        run_state.record_evidence(
            "paper-abs",
            [{"evidence_id": "ev-abs", "section_id": "sec_abstract", "source_layer": "abstract"}],
        )

        normalized = proposer._validate_submission_payload(
            workspace=workspace,
            submission={
                "submission_id": "submission_cycle_1",
                "question": workspace.question,
                "version": 1,
                "sections": [
                    {
                        "section_id": "direct_answer",
                        "title": "Direct Answer",
                        "content": "Pt/C likely improves HER activity.",
                        "citation_ids": ["CIT-1"],
                        "step_refs": [{"trajectory_id": "traj_1", "step_number": 1}],
                        "issue_refs": [],
                        "section_confidence": _confidence(0.7),
                    }
                ],
                "citations": [
                    {
                        "citation_id": "CIT-1",
                        "paper_id": "paper-abs",
                        "title": "Abstract paper",
                        "year": 2024,
                        "section_ids": ["sec_abstract"],
                        "evidence_ids": ["ev-abs"],
                    }
                ],
                "limitations": [],
                "overall_confidence": _confidence(0.8),
                "trajectory_id": "traj_1",
                "step_refs": [{"trajectory_id": "traj_1", "step_number": 1}],
                "issue_refs": [],
            },
            cycle_number=1,
            open_review_items=[],
            agent=None,
            trajectory=_trajectory("abstract degradation auto normalize"),
            run_state=run_state,
        )

        self.assertTrue(any("abstract-backed evidence" in item for item in normalized.limitations))
        self.assertEqual("low", normalized.overall_confidence.level)
        self.assertLessEqual(normalized.overall_confidence.score, 0.45)

    def test_validate_submission_payload_replaces_invalid_citations_with_current_cycle_fallback(self):
        proposer = ReactReviewedProposerAgent(
            model_config={},
            max_steps_initial=6,
            max_steps_revision=6,
            llm_timeout_seconds=45.0,
        )
        workspace = self._make_workspace()
        workspace.paper_records["paper-1"] = PaperRecord(
            paper_id="paper-1",
            title="NiFe LDH OER paper",
            doi="10.1000/nife",
            year=2024,
            venue="Journal",
            abstract="Abstract",
            fulltext_available=True,
            fulltext_status="fulltext_indexed",
        )
        run_state = _ProposerRunState(evidence_policy="prefer_fulltext")
        run_state.query_plan_ids.append("qp_1")
        run_state.searched_paper_ids.update({"paper-1", "paper-2"})
        run_state.acquired_paper_ids.add("paper-1")
        run_state.evidence_ids.add("ev-1")
        run_state.fulltext_status_by_paper["paper-1"] = "fulltext_indexed"
        run_state.section_ids_by_paper["paper-1"] = {"sec_results"}
        run_state.evidence_ids_by_paper["paper-1"] = {"ev-1"}
        run_state.evidence_layers_by_id["ev-1"] = "fulltext"

        normalized = proposer._validate_submission_payload(
            workspace=workspace,
            submission={
                "sections": [
                    {
                        "section_id": "direct_answer",
                        "title": "Direct Answer",
                        "content": "NiFe LDH is active for OER in alkaline electrolyte.",
                        "citation_ids": ["CIT-1"],
                    },
                ],
                "citations": [
                    {
                        "citation_id": "CIT-1",
                        "paper_id": "paper-2",
                        "title": "Wrong paper",
                        "section_ids": ["sec_abstract"],
                        "evidence_ids": ["ev-paper-2"],
                    }
                ],
                "limitations": [],
                "overall_confidence": _confidence(0.4),
            },
            cycle_number=1,
            open_review_items=[],
            agent=None,
            run_state=run_state,
        )

        self.assertEqual(1, len(normalized.citations))
        self.assertEqual("paper-1", normalized.citations[0].paper_id)
        self.assertEqual(["sec_results"], normalized.citations[0].section_ids)
        self.assertEqual(["ev-1"], normalized.citations[0].evidence_ids)

    def test_validate_submission_payload_normalizes_scalar_confidence_fields(self):
        proposer = ReactReviewedProposerAgent(
            model_config={},
            max_steps_initial=6,
            max_steps_revision=6,
            llm_timeout_seconds=45.0,
        )
        workspace = self._make_workspace()

        normalized = proposer._validate_submission_payload(
            workspace=workspace,
            submission={
                "sections": [
                    {
                        "section_id": "direct_answer",
                        "title": "Direct Answer",
                        "content": "Pt/C remains the benchmark in alkaline HER.",
                        "section_confidence": 0.2,
                    }
                ],
                "citations": [],
                "limitations": ["No document-level citations were available, so the submission remains conservative."],
                "overall_confidence": 0.15,
            },
            cycle_number=1,
            open_review_items=[],
            agent=None,
        )

        self.assertEqual("low", normalized.overall_confidence.level)
        self.assertEqual(0.15, normalized.overall_confidence.score)
        self.assertEqual("low", normalized.sections[0].section_confidence.level)
        self.assertEqual(0.2, normalized.sections[0].section_confidence.score)

    def test_validate_submission_payload_ignores_task_spec_version_and_maps_confidence_aliases(self):
        proposer = ReactReviewedProposerAgent(
            model_config={},
            max_steps_initial=6,
            max_steps_revision=6,
            llm_timeout_seconds=45.0,
        )
        workspace = self._make_workspace()

        normalized = proposer._validate_submission_payload(
            workspace=workspace,
            submission={
                "task_spec_version": "1.0",
                "sections": [
                    {
                        "section_id": "direct_answer",
                        "title": "Direct Answer",
                        "content": "Pt/C remains active in alkaline HER.",
                        "confidence": 0.25,
                    }
                ],
                "citations": [],
                "limitations": ["No document-level citations were available, so the submission remains conservative."],
                "confidence": 0.15,
            },
            cycle_number=1,
            open_review_items=[],
            agent=None,
        )

        self.assertEqual("low", normalized.overall_confidence.level)
        self.assertEqual(0.15, normalized.overall_confidence.score)
        self.assertEqual("low", normalized.sections[0].section_confidence.level)
        self.assertEqual(0.25, normalized.sections[0].section_confidence.score)

    def test_validate_submission_payload_normalizes_dict_list_fields(self):
        proposer = ReactReviewedProposerAgent(
            model_config={},
            max_steps_initial=6,
            max_steps_revision=6,
            llm_timeout_seconds=45.0,
        )
        workspace = self._make_workspace()

        normalized = proposer._validate_submission_payload(
            workspace=workspace,
            submission={
                "sections": [
                    {
                        "section_id": "direct_answer",
                        "title": "Direct Answer",
                        "content": "Pt/C remains the benchmark in alkaline HER.",
                    }
                ],
                "citations": {},
                "limitations": {},
                "issue_refs": {},
                "step_refs": {},
                "overall_confidence": _confidence(0.2),
            },
            cycle_number=1,
            open_review_items=[],
            agent=None,
        )

        self.assertEqual([], normalized.citations)
        self.assertEqual([], normalized.limitations)
        self.assertEqual([], normalized.issue_refs)
        self.assertEqual(1, len(normalized.step_refs))

    def test_proposer_salvages_submission_from_forced_conclude_diagnostics(self):
        proposer = ReactReviewedProposerAgent(
            model_config={},
            max_steps_initial=6,
            max_steps_revision=6,
            llm_timeout_seconds=45.0,
        )
        response = AgentResponse(content="")
        setattr(
            response,
            "response_content",
            {
                "forced_conclude_structured_json_response": {
                    "content": json.dumps(
                        {
                            "kind": "submission",
                            "payload": {
                                "submission_id": "submission_cycle_1",
                                "question": "How does Pt/C affect HER activity?",
                                "version": 1,
                                "sections": [
                                    {
                                        "section_id": "direct_answer",
                                        "title": "Direct Answer",
                                        "content": "Pt/C remains active in alkaline HER.",
                                        "citation_ids": ["CIT-1"],
                                        "step_refs": [{"trajectory_id": "traj_1", "step_number": 1}],
                                        "issue_refs": [],
                                        "section_confidence": _confidence(0.4),
                                    }
                                ],
                                "citations": [
                                    {
                                        "citation_id": "CIT-1",
                                        "paper_id": "paper-1",
                                        "title": "Pt/C HER in alkaline media",
                                        "section_ids": ["sec_results"],
                                        "evidence_ids": ["ev-1"],
                                    }
                                ],
                                "limitations": [],
                                "overall_confidence": _confidence(0.4),
                                "trajectory_id": "traj_1",
                                "step_refs": [{"trajectory_id": "traj_1", "step_number": 1}],
                                "issue_refs": [],
                            },
                        }
                    )
                }
            },
        )

        salvaged = proposer._salvage_submission_payload(response=response, trajectory=None)

        self.assertIsNotNone(salvaged)
        self.assertEqual("submission_cycle_1", salvaged["submission_id"])
        self.assertEqual("direct_answer", salvaged["sections"][0]["section_id"])

    def test_validate_submission_payload_still_rejects_unknown_fields(self):
        proposer = ReactReviewedProposerAgent(
            model_config={},
            max_steps_initial=6,
            max_steps_revision=6,
            llm_timeout_seconds=45.0,
        )
        workspace = self._make_workspace()

        with self.assertRaises(ValidationError):
            proposer._validate_submission_payload(
                workspace=workspace,
                submission={
                    "sections": [
                        {
                            "section_id": "direct_answer",
                            "title": "Direct Answer",
                            "content": "Pt/C improves HER activity under the cited conditions.",
                        }
                    ],
                    "unexpected_field": "should fail",
                },
                cycle_number=1,
                open_review_items=[],
                agent=None,
            )

    def test_reviewer_rejects_wrong_top_level_key(self):
        reviewer = ReactReviewedReviewerAgent(
            reviewer_role="search_coverage",
            model_config={},
            max_steps=3,
            max_items=3,
            max_retrieval_actions=1,
            llm_timeout_seconds=45.0,
        )
        response = AgentResponse(content=json.dumps({"review": [{"severity": "blocking"}]}))

        with self.assertRaises(ValueError):
            reviewer._parse_review_items_response(response=response)

    def test_reviewer_salvages_nested_review_payload_with_alias_fields(self):
        reviewer = ReactReviewedReviewerAgent(
            reviewer_role="reasoning_consistency",
            model_config={},
            max_steps=3,
            max_items=3,
            max_retrieval_actions=0,
            llm_timeout_seconds=45.0,
        )
        trajectory = _trajectory("review salvage")
        submission = _submission("What is the molecular formula of ethanol?")
        response = AgentResponse(
            content=json.dumps(
                {
                    "review": {
                        "review_items": [
                            {
                                "review_item_id": "RI-1",
                                "severity": "blocker",
                                "category": "scope_drift",
                                "issue": "Direct answer is off-topic.",
                                "required_fix": "Replace with the ethanol formula.",
                                "location": {"section_id": "direct_answer"},
                            }
                        ]
                    }
                }
            )
        )

        items = reviewer._salvage_review_payload(
            response=response,
            trajectory=trajectory,
            proposer_trajectory=trajectory,
            submission=submission,
            max_items=3,
        )

        self.assertEqual(1, len(items))
        self.assertEqual("RI-1", items[0].review_id)
        self.assertEqual("blocking", items[0].severity)
        self.assertEqual("scope_drift", items[0].flaw_type)
        self.assertEqual("direct_answer", items[0].target_section_id)

    def test_reviewer_salvages_forced_conclude_diagnostics_with_notes_payload(self):
        reviewer = ReactReviewedReviewerAgent(
            reviewer_role="reasoning_consistency",
            model_config={},
            max_steps=3,
            max_items=3,
            max_retrieval_actions=0,
            llm_timeout_seconds=45.0,
        )
        trajectory = _trajectory("review salvage from diagnostics")
        submission = _submission("What is the molecular formula of ethanol?")
        response = AgentResponse(content="")
        setattr(
            response,
            "response_content",
            {
                "forced_conclude_structured_json_response": {
                    "content": json.dumps(
                        {
                            "kind": "review_items",
                            "payload": [
                                {
                                    "reviewer_role": "reasoning_consistency",
                                    "category": "reasoning_gap",
                                    "required_fix": "Rewrite the direct answer so it follows the cited evidence.",
                                    "notes": [
                                        "Direct answer is off-topic.",
                                        "Supporting evidence does not justify the claim.",
                                    ],
                                }
                            ],
                        }
                    )
                }
            },
        )

        items = reviewer._salvage_review_payload(
            response=response,
            trajectory=None,
            proposer_trajectory=trajectory,
            submission=submission,
            max_items=3,
        )

        self.assertEqual(1, len(items))
        self.assertEqual("reasoning_consistency_1", items[0].review_id)
        self.assertEqual("reasoning_consistency", items[0].reviewer_role)
        self.assertIn("Direct answer is off-topic.", items[0].critique)
        self.assertIn("Supporting evidence does not justify the claim.", items[0].critique)

    def test_proposer_invalid_output_raises_and_restores_workspace_state(self):
        workflow = self._make_workflow()
        workflow.proposer = ReactReviewedProposerAgent(
            model_config={"provider": "openai", "model": "fake", "api_key": "test"},
            max_steps_initial=6,
            max_steps_revision=6,
            llm_timeout_seconds=45.0,
        )
        workspace = self._make_workspace()

        def _invalid_run_with_llm(**kwargs):
            kwargs["workspace"].paper_candidates["paper-1"] = PaperCandidate(
                paper_id="paper-1",
                title="stale candidate",
                year=2024,
                provider_hits=["openalex"],
                lane_sources=["review"],
                retrieval_score=0.5,
            )
            raise ReactReviewedStructuredOutputError(
                stage="proposer",
                cycle_number=kwargs["cycle_number"],
                message="invalid proposer structured output",
            )

        workflow.proposer._run_with_llm = _invalid_run_with_llm

        with self.assertRaises(ReactReviewedStructuredOutputError):
            workflow.proposer.run(
                workspace=workspace,
                cycle_number=1,
                open_review_items=[],
            )

        self.assertEqual({}, workspace.paper_candidates)

    def test_proposer_invalid_tool_arguments_response_enters_repair(self):
        proposer = ReactReviewedProposerAgent(
            model_config={"provider": "openai", "model": "fake", "api_key": "test"},
            max_steps_initial=6,
            max_steps_revision=6,
            llm_timeout_seconds=45.0,
            repair_attempts=1,
        )
        workspace = self._make_workspace()
        repaired_submission = _submission(workspace.question)
        repaired_trajectory = _trajectory("repaired proposer output")
        repair_calls: list[dict[str, object]] = []

        class _StubStructuredTool:
            @staticmethod
            def from_function(func, name=None, args_schema=None):
                class _Tool:
                    def __init__(self, func, name, args_schema):
                        self.func = func
                        self.name = name or getattr(func, "__name__", "tool")
                        self.args_schema = args_schema

                    def invoke(self, payload):
                        if isinstance(payload, dict):
                            return self.func(**payload)
                        return self.func(payload)

                return _Tool(func, name, args_schema)

        class _RepairableReActAgent:
            def __init__(self, *args, **kwargs):
                pass

            def generate_response_with_react(self, *args, **kwargs):
                response = AgentResponse(content="Invalid tool arguments for `conclude`: submission.overall_confidence: Field required")
                response.response_content = {
                    "forced_conclude_action_response": {
                        "message_type": "AIMessage",
                        "content": "",
                        "additional_kwargs": {},
                        "tool_calls": [
                            {
                                "name": "conclude",
                                "args": {
                                    "submission": {
                                        "answer_sections": [],
                                        "overall_confidence": 0.05,
                                        "evidence_items": [],
                                    }
                                },
                                "id": "call_1",
                            }
                        ],
                        "function_call": None,
                    },
                    "forced_conclude_structured_json_response": None,
                }
                return response, _trajectory("invalid conclude output")

        def _repair_submission_with_llm(**kwargs):
            repair_calls.append(kwargs)
            return repaired_submission, repaired_trajectory

        proposer._repair_submission_with_llm = _repair_submission_with_llm

        with patch("qa.react_reviewed_workflow._lazy_structured_tool_import", return_value=_StubStructuredTool), patch(
            "qa.react_reviewed_workflow.ReActAgent",
            _RepairableReActAgent,
        ):
            submission, trajectory = proposer.run(
                workspace=workspace,
                cycle_number=1,
                open_review_items=[],
            )

        self.assertEqual(repaired_submission.submission_id, submission.submission_id)
        self.assertEqual(repaired_trajectory.trajectory_id, trajectory.trajectory_id)
        self.assertEqual(1, len(repair_calls))
        self.assertIsInstance(repair_calls[0]["error"].response_content, dict)
        self.assertEqual(
            "",
            repair_calls[0]["error"].response_content["forced_conclude_action_response"]["content"],
        )

    def test_parse_submission_response_salvages_forced_conclude_tool_args(self):
        proposer = ReactReviewedProposerAgent(
            model_config={},
            max_steps_initial=6,
            max_steps_revision=6,
            llm_timeout_seconds=45.0,
        )
        payload = _submission("How does Pt/C affect HER activity?").model_dump(exclude_none=True)
        response = AgentResponse(content="not valid json")
        response.response_content = {
            "forced_conclude_action_response": {
                "message_type": "AIMessage",
                "content": "",
                "additional_kwargs": {},
                "tool_calls": [
                    {
                        "name": "conclude",
                        "args": {"submission": payload},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
                "function_call": None,
                "tool_call_id": None,
            }
        }

        salvaged = proposer._salvage_submission_payload(response=response, trajectory=None)

        self.assertIsInstance(salvaged, dict)
        self.assertEqual(payload["submission_id"], salvaged["submission_id"])
        self.assertEqual(payload["trajectory_id"], salvaged["trajectory_id"])

    def test_build_user_prompt_includes_conclude_contract(self):
        proposer = ReactReviewedProposerAgent(
            model_config={},
            max_steps_initial=6,
            max_steps_revision=6,
            llm_timeout_seconds=45.0,
        )
        workspace = self._make_workspace()

        prompt = proposer._build_user_prompt(
            workspace=workspace,
            cycle_number=2,
            open_review_items=[],
        )
        payload = json.loads(prompt)

        self.assertIn("conclude_call_contract", payload)
        self.assertEqual(
            "Call conclude with exactly {\"submission\": {...}}. Do not send a bare payload and do not use alternate top-level keys such as payload, answer_sections, or review.",
            payload["conclude_call_contract"]["tool_call_rule"],
        )
        self.assertEqual(["direct_answer"], payload["conclude_call_contract"]["required_section_ids"])
        self.assertIn("submission", payload["conclude_call_contract"]["tool_call_example"])

    def test_try_validate_salvaged_submission_payload_normalizes_invalid_citations_with_run_state(self):
        proposer = ReactReviewedProposerAgent(
            model_config={},
            max_steps_initial=6,
            max_steps_revision=6,
            llm_timeout_seconds=45.0,
        )
        workspace = self._make_workspace()
        trajectory = _trajectory("salvaged payload validation error")
        invalid_payload = _submission(workspace.question, trajectory_id=trajectory.trajectory_id).model_dump(exclude_none=True)
        invalid_payload["citations"][0]["paper_id"] = "<paper_id_from_search_papers>"
        response = SimpleNamespace(
            content="",
            structured_output=None,
            response_content={
                "forced_conclude_action_response": {
                    "message_type": "AIMessage",
                    "content": "",
                    "additional_kwargs": {},
                    "tool_calls": [
                        {
                            "name": "conclude",
                            "args": {"submission": invalid_payload},
                            "id": "call_1",
                            "type": "tool_call",
                        }
                    ],
                    "function_call": None,
                    "tool_call_id": None,
                }
            },
        )
        run_state = _ProposerRunState(evidence_policy="prefer_fulltext")
        run_state.query_plan_ids = ["qp_1"]
        run_state.searched_paper_ids = {"paper-1"}
        run_state.acquired_paper_ids = {"paper-1"}
        run_state.evidence_ids = {"ev-1"}
        run_state.section_ids_by_paper = {"paper-1": {"sec_results"}}
        run_state.evidence_ids_by_paper = {"paper-1": {"ev-1"}}
        run_state.evidence_layers_by_id = {"ev-1": "fulltext"}
        run_state.fulltext_status_by_paper = {"paper-1": "fulltext_indexed"}
        run_state.fulltext_available_by_paper = {"paper-1": True}

        submission, salvaged_payload, validation_error = proposer._try_validate_salvaged_submission_payload(
            workspace=workspace,
            response=response,
            cycle_number=1,
            open_review_items=[],
            trajectory=trajectory,
            run_state=run_state,
        )

        self.assertIsNotNone(submission)
        self.assertEqual(invalid_payload["submission_id"], salvaged_payload["submission_id"])
        self.assertIsNone(validation_error)
        self.assertEqual("paper-1", submission.citations[0].paper_id)
        self.assertEqual(["ev-1"], submission.citations[0].evidence_ids)

    def test_action_instruction_requires_submission_wrapper(self):
        proposer = ReactReviewedProposerAgent(
            model_config={},
            max_steps_initial=6,
            max_steps_revision=6,
            llm_timeout_seconds=45.0,
        )

        instruction = proposer._action_instruction(PROPOSER_TOOL_NAMES)

        self.assertIn('For conclude, the tool args object must be exactly {"submission": {...}}.', instruction)
        self.assertIn("Do not pass a bare submission object", instruction)

    def test_proposer_repair_parses_valid_json_response(self):
        proposer = ReactReviewedProposerAgent(
            model_config={"provider": "openai", "model": "fake", "api_key": "test"},
            max_steps_initial=6,
            max_steps_revision=6,
            llm_timeout_seconds=45.0,
            repair_attempts=1,
        )
        workspace = self._make_workspace()
        trajectory = _trajectory("repair proposer output")
        invalid_error = ReactReviewedStructuredOutputError(
            stage="proposer",
            cycle_number=1,
            message="invalid proposer structured output",
            response_content="not json",
            structured_output=None,
            trajectory=trajectory,
        )
        run_state = _ProposerRunState(evidence_policy="strict")
        repaired_submission = _submission(workspace.question, trajectory_id=trajectory.trajectory_id)

        with patch("qa.react_reviewed_workflow.build_chat_model_from_config", return_value=object()), patch(
            "qa.react_reviewed_workflow.invoke_llm",
            return_value=json.dumps({"kind": "submission", "payload": repaired_submission.model_dump(exclude_none=True)}),
        ), patch.object(proposer, "_validate_submission_payload", return_value=repaired_submission):
            submission, repaired_trajectory = proposer._repair_submission_with_llm(
                workspace=workspace,
                cycle_number=1,
                open_review_items=[],
                error=invalid_error,
                trajectory=trajectory,
                run_state=run_state,
            )

        self.assertEqual(repaired_submission.submission_id, submission.submission_id)
        self.assertEqual(trajectory.trajectory_id, repaired_trajectory.trajectory_id)

    def test_proposer_repair_prompt_requests_exact_submission_envelope(self):
        proposer = ReactReviewedProposerAgent(
            model_config={"provider": "openai", "model": "fake", "api_key": "test"},
            max_steps_initial=6,
            max_steps_revision=6,
            llm_timeout_seconds=45.0,
            repair_attempts=1,
        )
        workspace = self._make_workspace()
        trajectory = _trajectory("repair prompt proposer output")
        invalid_error = ReactReviewedStructuredOutputError(
            stage="proposer",
            cycle_number=1,
            message="invalid proposer structured output",
            response_content="not json",
            structured_output=None,
            trajectory=trajectory,
        )
        run_state = _ProposerRunState(evidence_policy="strict")
        repaired_submission = _submission(workspace.question, trajectory_id=trajectory.trajectory_id)

        def _inspect_invoke(_llm, messages):
            self.assertIn('Return EXACTLY {"kind":"submission","payload":{...}}.', messages[0]["content"])
            user_payload = json.loads(messages[1]["content"])
            self.assertIn("conclude_call_contract", user_payload)
            self.assertIn("repair_json_example", user_payload["conclude_call_contract"])
            self.assertIn("tool_call_example", user_payload["conclude_call_contract"])
            return json.dumps({"kind": "submission", "payload": repaired_submission.model_dump(exclude_none=True)})

        with patch("qa.react_reviewed_workflow.build_chat_model_from_config", return_value=object()), patch(
            "qa.react_reviewed_workflow.invoke_llm",
            side_effect=_inspect_invoke,
        ), patch.object(proposer, "_validate_submission_payload", return_value=repaired_submission):
            submission, repaired_trajectory = proposer._repair_submission_with_llm(
                workspace=workspace,
                cycle_number=1,
                open_review_items=[],
                error=invalid_error,
                trajectory=trajectory,
                run_state=run_state,
            )

        self.assertEqual(repaired_submission.submission_id, submission.submission_id)
        self.assertEqual(trajectory.trajectory_id, repaired_trajectory.trajectory_id)

    def test_proposer_repair_prompt_prefers_invalid_submission_payload_over_raw_response_noise(self):
        proposer = ReactReviewedProposerAgent(
            model_config={"provider": "openai", "model": "fake", "api_key": "test"},
            max_steps_initial=6,
            max_steps_revision=6,
            llm_timeout_seconds=45.0,
            repair_attempts=1,
        )
        workspace = self._make_workspace()
        trajectory = _trajectory("repair prompt invalid payload proposer output")
        invalid_payload = {
            "submission_id": "submission_cycle_1",
            "question": workspace.question,
            "version": 1,
            "sections": [
                {
                    "section_id": "direct_answer",
                    "title": "Direct Answer",
                    "content": "<grounded section content>",
                    "citation_ids": ["CIT-1"],
                    "step_refs": [{"trajectory_id": "<trajectory_id>", "step_number": 1}],
                    "issue_refs": [],
                    "section_confidence": _confidence(0.3),
                }
            ],
            "citations": [
                {
                    "citation_id": "CIT-1",
                    "paper_id": "<paper_id_from_search_papers>",
                    "title": "<paper_title_from_tools>",
                    "section_ids": ["<section_id_from_tools_or_sec_abstract>"],
                    "evidence_ids": ["<evidence_id_from_tools>"],
                }
            ],
            "limitations": ["<explicit limitation grounded in the current run>"],
            "overall_confidence": _confidence(0.2),
            "trajectory_id": "<trajectory_id>",
            "step_refs": [{"trajectory_id": "<trajectory_id>", "step_number": 1}],
            "issue_refs": [],
        }
        invalid_error = ReactReviewedStructuredOutputError(
            stage="proposer",
            cycle_number=1,
            message="invalid proposer structured output: placeholder payload",
            response_content={"forced_conclude_action_response": {"content": "provider noise"}},
            structured_output={"kind": "submission", "payload": invalid_payload},
            trajectory=trajectory,
        )
        run_state = _ProposerRunState(evidence_policy="strict")
        repaired_submission = _submission(workspace.question, trajectory_id=trajectory.trajectory_id)

        def _inspect_invoke(_llm, messages):
            self.assertIn("Do not copy angle-bracket placeholders", messages[0]["content"])
            user_payload = json.loads(messages[1]["content"])
            self.assertEqual(invalid_payload["submission_id"], user_payload["invalid_submission_payload"]["submission_id"])
            self.assertEqual(
                "<paper_id_from_search_papers>",
                user_payload["invalid_submission_payload"]["citations"][0]["paper_id"],
            )
            self.assertIsNone(user_payload["invalid_response_content"])
            return json.dumps({"kind": "submission", "payload": repaired_submission.model_dump(exclude_none=True)})

        with patch("qa.react_reviewed_workflow.build_chat_model_from_config", return_value=object()), patch(
            "qa.react_reviewed_workflow.invoke_llm",
            side_effect=_inspect_invoke,
        ), patch.object(proposer, "_validate_submission_payload", return_value=repaired_submission):
            submission, repaired_trajectory = proposer._repair_submission_with_llm(
                workspace=workspace,
                cycle_number=1,
                open_review_items=[],
                error=invalid_error,
                trajectory=trajectory,
                run_state=run_state,
            )

        self.assertEqual(repaired_submission.submission_id, submission.submission_id)
        self.assertEqual(trajectory.trajectory_id, repaired_trajectory.trajectory_id)

    def test_proposer_repair_salvages_prior_invalid_payload_when_repair_response_is_not_json(self):
        proposer = ReactReviewedProposerAgent(
            model_config={"provider": "openai", "model": "fake", "api_key": "test"},
            max_steps_initial=6,
            max_steps_revision=6,
            llm_timeout_seconds=45.0,
            repair_attempts=1,
        )
        workspace = self._make_workspace()
        trajectory = _trajectory("repair salvage proposer output")
        salvaged_submission = _submission(workspace.question, trajectory_id=trajectory.trajectory_id)
        invalid_error = ReactReviewedStructuredOutputError(
            stage="proposer",
            cycle_number=1,
            message="invalid proposer structured output",
            response_content={
                "forced_conclude_action_response": {
                    "message_type": "AIMessage",
                    "content": "",
                    "additional_kwargs": {},
                    "tool_calls": [
                        {
                            "name": "conclude",
                            "args": {
                                "submission": salvaged_submission.model_dump(exclude_none=True),
                            },
                            "id": "call_1",
                            "type": "tool_call",
                        }
                    ],
                    "function_call": None,
                    "tool_call_id": None,
                }
            },
            structured_output=None,
            trajectory=trajectory,
        )
        run_state = _ProposerRunState(evidence_policy="strict")

        def _validate_submission_payload(*, submission, **kwargs):
            self.assertEqual(
                salvaged_submission.model_dump(exclude_none=True)["submission_id"],
                submission["submission_id"],
            )
            return salvaged_submission

        with patch("qa.react_reviewed_workflow.build_chat_model_from_config", return_value=object()), patch(
            "qa.react_reviewed_workflow.invoke_llm",
            return_value="not json",
        ), patch.object(proposer, "_validate_submission_payload", side_effect=_validate_submission_payload):
            submission, repaired_trajectory = proposer._repair_submission_with_llm(
                workspace=workspace,
                cycle_number=1,
                open_review_items=[],
                error=invalid_error,
                trajectory=trajectory,
                run_state=run_state,
            )

        self.assertEqual(salvaged_submission.submission_id, submission.submission_id)
        self.assertEqual(trajectory.trajectory_id, repaired_trajectory.trajectory_id)

    def test_proposer_repair_rebuilds_from_workspace_when_response_is_truncated(self):
        proposer = ReactReviewedProposerAgent(
            model_config={"provider": "openai", "model": "fake", "api_key": "test"},
            max_steps_initial=6,
            max_steps_revision=6,
            llm_timeout_seconds=45.0,
            repair_attempts=1,
        )
        workspace = self._make_workspace()
        trajectory = _trajectory("repair workspace rebuild")
        invalid_error = ReactReviewedStructuredOutputError(
            stage="proposer",
            cycle_number=1,
            message="invalid proposer structured output",
            response_content="not json",
            structured_output=None,
            trajectory=trajectory,
        )
        workspace.paper_records["paper-1"] = PaperRecord(
            paper_id="paper-1",
            title="Pt/C improves HER activity in alkaline media",
            doi="10.1000/example",
            year=2024,
            venue="Journal",
            abstract="Pt/C improves HER activity in alkaline media with lower overpotential.",
            fulltext_available=True,
            fulltext_status="fulltext_indexed",
        )
        workspace.evidence_items["ev-1"] = EvidenceItem(
            evidence_id="ev-1",
            paper_id="paper-1",
            doi="10.1000/example",
            section_id="sec_results",
            section_type="results",
            role="observation",
            snippet="Compared to Pt/NTC at 178 mV, Pt/TC reached 58 mV at 10 mA cm-2 in alkaline HER.",
            source_span={"start": 0, "end": 87},
            source_layer="fulltext",
            claim_polarity="support",
            conditions={"electrolyte": "1 m koh"},
            metric_mentions=["178 mV", "58 mV", "10 mA cm-2"],
            extraction_confidence=0.92,
        )
        run_state = _ProposerRunState(evidence_policy="prefer_fulltext")
        run_state.query_plan_ids = ["qp_1"]
        run_state.searched_paper_ids = {"paper-1"}
        run_state.acquired_paper_ids = {"paper-1"}
        run_state.locked_candidate_paper_ids = ["paper-1"]
        run_state.section_ids_by_paper = {"paper-1": {"sec_results"}}
        run_state.evidence_ids_by_paper = {"paper-1": {"ev-1"}}
        run_state.evidence_ids = {"ev-1"}
        run_state.evidence_layers_by_id = {"ev-1": "fulltext"}
        run_state.fulltext_status_by_paper = {"paper-1": "fulltext_indexed"}
        run_state.fulltext_available_by_paper = {"paper-1": True}

        with patch("qa.react_reviewed_workflow.build_chat_model_from_config", return_value=object()), patch(
            "qa.react_reviewed_workflow.invoke_llm",
            return_value='{"kind":"submission","payload":{"submission_id":"submission_cycle_1"',
        ):
            submission, repaired_trajectory = proposer._repair_submission_with_llm(
                workspace=workspace,
                cycle_number=1,
                open_review_items=[],
                error=invalid_error,
                trajectory=trajectory,
                run_state=run_state,
            )

        self.assertEqual("submission_cycle_1", submission.submission_id)
        self.assertEqual("paper-1", submission.citations[0].paper_id)
        self.assertIn("CIT-1", submission.sections[0].citation_ids)
        self.assertEqual(trajectory.trajectory_id, repaired_trajectory.trajectory_id)

    def test_proposer_requires_revision_budget_large_enough_for_grounded_cycle(self):
        with self.assertRaisesRegex(ValueError, "max_steps_revision must be at least 6"):
            ReactReviewedProposerAgent(
                model_config={},
                max_steps_initial=6,
                max_steps_revision=4,
                llm_timeout_seconds=45.0,
            )

    def test_proposer_fail_fast_without_fallback_when_model_unavailable(self):
        proposer = ReactReviewedProposerAgent(
            model_config={},
            max_steps_initial=6,
            max_steps_revision=6,
            llm_timeout_seconds=45.0,
        )
        workspace = self._make_workspace()

        with self.assertRaises(ReactReviewedProposerExecutionError):
            proposer.run(
                workspace=workspace,
                cycle_number=1,
                open_review_items=[],
            )

        self.assertEqual({}, workspace.paper_candidates)
        failure_path = self.temp_dir / "workspace" / "diagnostics" / "proposer_cycle_1_failure.json"
        self.assertTrue(failure_path.exists(), str(failure_path))

    def test_proposer_execution_failure_persists_forced_conclude_raw_responses(self):
        proposer = ReactReviewedProposerAgent(
            model_config={"provider": "openai", "model": "fake", "api_key": "test"},
            max_steps_initial=6,
            max_steps_revision=6,
            llm_timeout_seconds=45.0,
        )
        workspace = self._make_workspace()

        class _StubStructuredTool:
            @staticmethod
            def from_function(func, name=None, args_schema=None):
                class _Tool:
                    def __init__(self, func, name, args_schema):
                        self.func = func
                        self.name = name or getattr(func, "__name__", "tool")
                        self.args_schema = args_schema

                    def invoke(self, payload):
                        if isinstance(payload, dict):
                            return self.func(**payload)
                        return self.func(payload)

                return _Tool(func, name, args_schema)

        class _RaisingReActAgent:
            def __init__(self, *args, **kwargs):
                pass

            def generate_response_with_react(self, *args, **kwargs):
                error = RuntimeError("Forced conclude failed to emit a recognized `conclude` tool call.")
                error.response_content = {
                    "forced_conclude_action_response": {
                        "message_type": "AIMessage",
                        "content": "first raw return",
                        "additional_kwargs": {"provider": "test"},
                        "tool_calls": [],
                        "function_call": None,
                    },
                    "forced_conclude_structured_json_response": {
                        "message_type": "AIMessage",
                        "content": "not json",
                        "additional_kwargs": {},
                        "tool_calls": [],
                        "function_call": None,
                    },
                }
                error.structured_output = None
                raise error

        with patch("qa.react_reviewed_workflow._lazy_structured_tool_import", return_value=_StubStructuredTool), patch(
            "qa.react_reviewed_workflow.ReActAgent",
            _RaisingReActAgent,
        ):
            with self.assertRaises(ReactReviewedProposerExecutionError):
                proposer.run(
                    workspace=workspace,
                    cycle_number=1,
                    open_review_items=[],
                )

        failure_path = self.temp_dir / "workspace" / "diagnostics" / "proposer_cycle_1_failure.json"
        payload = _read_json(str(failure_path))
        self.assertEqual(
            "first raw return",
            payload["response_content"]["forced_conclude_action_response"]["content"],
        )
        self.assertEqual(
            "not json",
            payload["response_content"]["forced_conclude_structured_json_response"]["content"],
        )

    def test_proposer_forced_conclude_exception_with_invalid_salvage_enters_repair(self):
        proposer = ReactReviewedProposerAgent(
            model_config={"provider": "openai", "model": "fake", "api_key": "test"},
            max_steps_initial=6,
            max_steps_revision=6,
            llm_timeout_seconds=45.0,
            repair_attempts=1,
        )
        workspace = self._make_workspace()
        repaired_submission = _submission(workspace.question, trajectory_id="traj_repaired")
        repaired_trajectory = _trajectory("repaired after forced conclude exception")
        invalid_payload = _submission(workspace.question, trajectory_id="traj_bad").model_dump(exclude_none=True)
        invalid_payload["citations"][0]["paper_id"] = "paper_missing"

        class _StubStructuredTool:
            @staticmethod
            def from_function(func, name=None, args_schema=None):
                class _Tool:
                    def __init__(self, func, name, args_schema):
                        self.func = func
                        self.name = name or getattr(func, "__name__", "tool")
                        self.args_schema = args_schema

                    def invoke(self, payload):
                        if isinstance(payload, dict):
                            return self.func(**payload)
                        return self.func(payload)

                return _Tool(func, name, args_schema)

        class _RaisingReActAgent:
            def __init__(self, *args, **kwargs):
                self.current_trajectory = _trajectory("forced conclude exception")

            def generate_response_with_react(self, *args, **kwargs):
                error = RuntimeError("Forced conclude failed to emit a recognized `conclude` tool call.")
                error.response_content = {
                    "forced_conclude_structured_json_response": {
                        "message_type": "AIMessage",
                        "content": json.dumps({"submission": invalid_payload}),
                        "additional_kwargs": {},
                        "tool_calls": [],
                        "function_call": None,
                    }
                }
                error.structured_output = None
                error.trajectory = self.current_trajectory
                raise error

        with patch("qa.react_reviewed_workflow._lazy_structured_tool_import", return_value=_StubStructuredTool), patch(
            "qa.react_reviewed_workflow.ReActAgent",
            _RaisingReActAgent,
        ), patch.object(
            proposer,
            "_repair_submission_with_llm",
            return_value=(repaired_submission, repaired_trajectory),
        ) as repair_mock:
            submission, trajectory = proposer.run(
                workspace=workspace,
                cycle_number=1,
                open_review_items=[],
            )

        self.assertEqual(repaired_submission.submission_id, submission.submission_id)
        self.assertEqual(repaired_trajectory.trajectory_id, trajectory.trajectory_id)
        repair_error = repair_mock.call_args.kwargs["error"]
        self.assertIsInstance(repair_error, ReactReviewedStructuredOutputError)
        self.assertEqual("proposer", repair_error.stage)
        self.assertIn("Submission has no citation catalog", str(repair_error))
        self.assertEqual("submission_cycle_1", repair_error.structured_output["payload"]["submission_id"])
        invalid_path = self.temp_dir / "workspace" / "diagnostics" / "proposer_cycle_1_invalid_response.json"
        self.assertTrue(invalid_path.exists(), str(invalid_path))

    def test_revision_forced_conclude_exception_without_salvage_enters_repair_from_prior_submission(self):
        proposer = ReactReviewedProposerAgent(
            model_config={"provider": "openai", "model": "fake", "api_key": "test"},
            max_steps_initial=6,
            max_steps_revision=6,
            llm_timeout_seconds=45.0,
            repair_attempts=1,
        )
        workspace = self._make_workspace()
        prior_submission = _submission(workspace.question, cycle_number=1, trajectory_id="traj_prior")
        prior_trajectory = _trajectory("prior cycle")
        workspace.set_review_context(
            submission=prior_submission,
            proposer_trajectory=prior_trajectory,
            open_review_items=[],
            cycle_number=2,
        )
        repaired_submission = _submission(workspace.question, cycle_number=2, trajectory_id="traj_repaired")
        repaired_trajectory = _trajectory("repaired revision output")

        class _StubStructuredTool:
            @staticmethod
            def from_function(func, name=None, args_schema=None):
                class _Tool:
                    def __init__(self, func, name, args_schema):
                        self.func = func
                        self.name = name or getattr(func, "__name__", "tool")
                        self.args_schema = args_schema

                    def invoke(self, payload):
                        if isinstance(payload, dict):
                            return self.func(**payload)
                        return self.func(payload)

                return _Tool(func, name, args_schema)

        class _RaisingReActAgent:
            def __init__(self, *args, **kwargs):
                self.current_trajectory = _trajectory("revision forced conclude exception")

            def generate_response_with_react(self, *args, **kwargs):
                error = RuntimeError("Forced conclude failed to emit a recognized `conclude` tool call.")
                error.response_content = {
                    "forced_conclude_action_response": {
                        "message_type": "AIMessage",
                        "content": "",
                        "additional_kwargs": {},
                        "tool_calls": [],
                        "function_call": None,
                    }
                }
                error.structured_output = None
                error.trajectory = self.current_trajectory
                raise error

        with patch("qa.react_reviewed_workflow._lazy_structured_tool_import", return_value=_StubStructuredTool), patch(
            "qa.react_reviewed_workflow.ReActAgent",
            _RaisingReActAgent,
        ), patch.object(
            proposer,
            "_repair_submission_with_llm",
            return_value=(repaired_submission, repaired_trajectory),
        ) as repair_mock:
            submission, trajectory = proposer.run(
                workspace=workspace,
                cycle_number=2,
                open_review_items=[],
            )

        self.assertEqual(repaired_submission.submission_id, submission.submission_id)
        self.assertEqual(repaired_trajectory.trajectory_id, trajectory.trajectory_id)
        repair_error = repair_mock.call_args.kwargs["error"]
        self.assertIsInstance(repair_error, ReactReviewedStructuredOutputError)
        self.assertIn("forced conclude execution failed before a valid payload was emitted", str(repair_error))

    def test_initial_forced_conclude_exception_without_salvage_enters_repair_when_evidence_exists(self):
        proposer = ReactReviewedProposerAgent(
            model_config={"provider": "openai", "model": "fake", "api_key": "test"},
            max_steps_initial=6,
            max_steps_revision=6,
            llm_timeout_seconds=45.0,
            repair_attempts=1,
        )
        workspace = self._make_workspace()
        workspace.evidence_items["ev-1"] = EvidenceItem.model_validate(
            {
                "evidence_id": "ev-1",
                "paper_id": "paper-1",
                "doi": "10.1000/example",
                "section_id": "sec_abstract",
                "section_type": "abstract",
                "role": "observation",
                "snippet": "Pt/C is active for alkaline HER.",
                "source_span": {"start": 0, "end": 31},
                "source_layer": "abstract",
                "claim_polarity": "support",
                "conditions": {},
                "condition_source_refs": [],
                "metric_mentions": [],
                "entity_mentions": ["Pt/C"],
                "extraction_confidence": 0.7,
            }
        )
        repaired_submission = _submission(workspace.question, cycle_number=1, trajectory_id="traj_repaired")
        repaired_trajectory = _trajectory("repaired initial output")

        class _StubStructuredTool:
            @staticmethod
            def from_function(func, name=None, args_schema=None):
                class _Tool:
                    def __init__(self, func, name, args_schema):
                        self.func = func
                        self.name = name or getattr(func, "__name__", "tool")
                        self.args_schema = args_schema

                    def invoke(self, payload):
                        if isinstance(payload, dict):
                            return self.func(**payload)
                        return self.func(payload)

                return _Tool(func, name, args_schema)

        class _RaisingReActAgent:
            def __init__(self, *args, **kwargs):
                self.current_trajectory = _trajectory("initial forced conclude exception")

            def generate_response_with_react(self, *args, **kwargs):
                error = RuntimeError("Forced conclude failed to emit a recognized `conclude` tool call.")
                error.response_content = {
                    "forced_conclude_action_response": {
                        "message_type": "AIMessage",
                        "content": "",
                        "additional_kwargs": {},
                        "tool_calls": [],
                        "function_call": None,
                    }
                }
                error.structured_output = None
                error.trajectory = self.current_trajectory
                raise error

        with patch("qa.react_reviewed_workflow._lazy_structured_tool_import", return_value=_StubStructuredTool), patch(
            "qa.react_reviewed_workflow.ReActAgent",
            _RaisingReActAgent,
        ), patch.object(
            proposer,
            "_repair_submission_with_llm",
            return_value=(repaired_submission, repaired_trajectory),
        ) as repair_mock:
            submission, trajectory = proposer.run(
                workspace=workspace,
                cycle_number=1,
                open_review_items=[],
            )

        self.assertEqual(repaired_submission.submission_id, submission.submission_id)
        self.assertEqual(repaired_trajectory.trajectory_id, trajectory.trajectory_id)
        repair_error = repair_mock.call_args.kwargs["error"]
        self.assertIsInstance(repair_error, ReactReviewedStructuredOutputError)
        self.assertIn("forced conclude execution failed before a valid payload was emitted", str(repair_error))

    def test_ad_hoc_query_plan_ids_are_stable_across_repeated_searches(self):
        workspace = self._make_workspace(retriever=_CountingRetriever())

        first = workspace.search_papers(
            query_text="Pt/C HER alkaline",
            lane="review",
            reason="first",
            write_snapshot=False,
        )
        second = workspace.search_papers(
            query_text="Pt/C HER alkaline",
            lane="review",
            reason="second",
            write_snapshot=False,
        )

        self.assertTrue(first)
        self.assertTrue(second)
        self.assertEqual(first[0]["query_plan_id"], second[0]["query_plan_id"])
        self.assertEqual(1, len(workspace.query_plans))

    def test_plan_queries_reuses_existing_lane_plans_across_repeated_calls(self):
        class _FixedPlanner:
            def run(self, *, task_spec: TaskSpec, entity_pack: EntityPack):
                return [
                    QueryPlan(
                        lane="review",
                        query_text="Pt/C HER alkaline review",
                        must_terms=["Pt/C", "HER"],
                        exclude_terms=[],
                        year_from=None,
                        year_to=None,
                        preferred_sources=["openalex"],
                    ),
                    QueryPlan(
                        lane="contrarian",
                        query_text="Pt/C HER alkaline limitations",
                        must_terms=["Pt/C", "HER"],
                        exclude_terms=["hydrazine"],
                        year_from=None,
                        year_to=None,
                        preferred_sources=["openalex"],
                    ),
                ]

        workspace = ReactReviewedWorkspace(
            question="How does Pt/C affect HER activity?",
            context=None,
            task_spec=_task_spec(),
            entity_pack=_entity_pack(),
            entity_resolution_snapshot={},
            artifact_store=QAArtifactStore(base_dir=self.temp_dir / "workspace_plan_dedupe"),
            query_planner=_FixedPlanner(),
            retriever=_CountingRetriever(),
            document_acquirer=_CountingDocumentAcquirer(),
            handoff=EvidenceExtractorHandoff(),
            evidence_extractor=_CountingEvidenceExtractor(),
        )

        first = workspace.plan_queries(focus="initial")
        second = workspace.plan_queries(focus="revision")

        self.assertEqual(["qp_1", "qp_2"], [item["query_plan_id"] for item in first])
        self.assertEqual(["qp_1", "qp_2"], [item["query_plan_id"] for item in second])
        self.assertEqual(2, len(workspace.query_plans))

    def test_validate_submission_auto_prefers_fulltext_backed_citation_when_available(self):
        proposer = ReactReviewedProposerAgent(
            model_config={},
            max_steps_initial=6,
            max_steps_revision=6,
            llm_timeout_seconds=45.0,
        )
        workspace = self._make_workspace()
        run_state = _ProposerRunState(evidence_policy="prefer_fulltext")
        run_state.query_plan_ids = ["qp_1"]
        run_state.searched_paper_ids.update({"paper-full", "paper-abs"})
        run_state.acquired_paper_ids.update({"paper-full", "paper-abs"})
        run_state.fulltext_status_by_paper.update({"paper-full": "fulltext_indexed", "paper-abs": "abstract_only"})
        run_state.section_ids_by_paper.update({"paper-full": {"sec_results"}, "paper-abs": {"sec_abstract"}})
        run_state.record_evidence(
            "paper-full",
            [{"evidence_id": "ev-full", "section_id": "sec_results", "source_layer": "fulltext"}],
        )
        run_state.record_evidence(
            "paper-abs",
            [{"evidence_id": "ev-abs", "section_id": "sec_abstract", "source_layer": "abstract"}],
        )

        raw_submission = {
            "submission_id": "submission_cycle_1",
            "question": workspace.question,
            "version": 1,
            "sections": [
                {
                    "section_id": "direct_answer",
                    "title": "Direct Answer",
                    "content": "Pt/C improves HER activity.",
                    "citation_ids": ["CIT-ABS"],
                    "step_refs": [{"trajectory_id": "traj_1", "step_number": 1}],
                    "issue_refs": [],
                    "section_confidence": _confidence(0.6),
                }
            ],
            "citations": [
                {
                    "citation_id": "CIT-FULL",
                    "paper_id": "paper-full",
                    "title": "Fulltext paper",
                    "year": 2024,
                    "section_ids": ["sec_results"],
                    "evidence_ids": ["ev-full"],
                },
                {
                    "citation_id": "CIT-ABS",
                    "paper_id": "paper-abs",
                    "title": "Abstract paper",
                    "year": 2024,
                    "section_ids": ["sec_abstract"],
                    "evidence_ids": ["ev-abs"],
                },
            ],
            "limitations": [],
            "overall_confidence": _confidence(0.5),
            "trajectory_id": "traj_1",
            "step_refs": [{"trajectory_id": "traj_1", "step_number": 1}],
            "issue_refs": [],
        }

        normalized = proposer._validate_submission_payload(
            workspace=workspace,
            submission=raw_submission,
            cycle_number=1,
            open_review_items=[],
            agent=None,
            trajectory=_trajectory("fulltext validation"),
            run_state=run_state,
        )

        self.assertEqual(["CIT-FULL", "CIT-ABS"], normalized.sections[0].citation_ids)

    def test_validate_submission_allows_explicit_abstract_only_degradation(self):
        proposer = ReactReviewedProposerAgent(
            model_config={},
            max_steps_initial=6,
            max_steps_revision=6,
            llm_timeout_seconds=45.0,
        )
        workspace = self._make_workspace()
        run_state = _ProposerRunState(evidence_policy="prefer_fulltext")
        run_state.query_plan_ids = ["qp_1"]
        run_state.searched_paper_ids.add("paper-abs")
        run_state.acquired_paper_ids.add("paper-abs")
        run_state.fulltext_status_by_paper["paper-abs"] = "abstract_only"
        run_state.section_ids_by_paper["paper-abs"] = {"sec_abstract"}
        run_state.record_evidence(
            "paper-abs",
            [{"evidence_id": "ev-abs", "section_id": "sec_abstract", "source_layer": "abstract"}],
        )

        submission = proposer._validate_submission_payload(
            workspace=workspace,
            submission={
                "submission_id": "submission_cycle_1",
                "question": workspace.question,
                "version": 1,
                "sections": [
                    {
                        "section_id": "direct_answer",
                        "title": "Direct Answer",
                        "content": "Pt/C likely improves HER activity, but this run only recovered abstract-backed support.",
                        "citation_ids": ["CIT-1"],
                        "step_refs": [{"trajectory_id": "traj_1", "step_number": 1}],
                        "issue_refs": [],
                        "section_confidence": _confidence(0.45),
                    }
                ],
                "citations": [
                    {
                        "citation_id": "CIT-1",
                        "paper_id": "paper-abs",
                        "title": "Abstract paper",
                        "year": 2024,
                        "section_ids": ["sec_abstract"],
                        "evidence_ids": ["ev-abs"],
                    }
                ],
                "limitations": [
                    "This submission is degraded because only abstract-backed evidence was available and no usable full text could be recovered in this cycle."
                ],
                "overall_confidence": _confidence(0.4),
                "trajectory_id": "traj_1",
                "step_refs": [{"trajectory_id": "traj_1", "step_number": 1}],
                "issue_refs": [],
            },
            cycle_number=1,
            open_review_items=[],
            agent=None,
            trajectory=_trajectory("abstract degradation"),
            run_state=run_state,
        )

        self.assertEqual("submission_cycle_1", submission.submission_id)
        self.assertEqual(["CIT-1"], submission.sections[0].citation_ids)

    def test_reviewer_invalid_json_status_restores_workspace_state(self):
        workflow = self._make_workflow()
        reviewer = workflow.reviewers["search_coverage"]
        workspace = self._make_workspace()
        session = self._make_session("search_coverage", 1)
        submission = _submission(workspace.question)
        proposer_trajectory = _trajectory("proposer invalid-json")
        invalid_trajectory = _trajectory("reviewer invalid-json")

        def _invalid_run_with_llm(**kwargs):
            kwargs["workspace"].paper_records["paper-1"] = PaperRecord(
                paper_id="paper-1",
                title="stale record",
                year=2024,
            )
            raise ReactReviewedStructuredOutputError(
                stage="reviewer",
                cycle_number=kwargs["cycle_number"],
                reviewer_role="search_coverage",
                message="invalid reviewer structured output",
                trajectory=invalid_trajectory,
            )

        reviewer._run_with_llm = _invalid_run_with_llm

        items, trajectory, status = reviewer.run(
            workspace=workspace,
            submission=submission,
            proposer_trajectory=proposer_trajectory,
            cycle_number=1,
            session=session,
        )

        self.assertEqual([], items)
        self.assertEqual("invalid_json", status.status)
        self.assertEqual(invalid_trajectory.trajectory_id, trajectory.trajectory_id)
        self.assertEqual({}, workspace.paper_records)

    def test_workflow_marks_incomplete_when_reviewer_returns_invalid_json(self):
        workflow = self._make_workflow()
        failing_reviewer = workflow.reviewers["search_coverage"]
        invalid_trajectory = _trajectory("search_coverage invalid")

        def _invalid_run_with_llm(**kwargs):
            raise ReactReviewedStructuredOutputError(
                stage="reviewer",
                cycle_number=kwargs["cycle_number"],
                reviewer_role="search_coverage",
                message="invalid reviewer structured output",
                trajectory=invalid_trajectory,
            )

        failing_reviewer._run_with_llm = _invalid_run_with_llm
        workflow.reviewers = {
            "search_coverage": failing_reviewer,
            "evidence_trace": _ParallelReviewer("evidence_trace", threading.Barrier(1), 0.0, []),
            "reasoning_consistency": _ParallelReviewer("reasoning_consistency", threading.Barrier(1), 0.0, []),
            "counterevidence": _ParallelReviewer("counterevidence", threading.Barrier(1), 0.0, []),
        }

        result = workflow.run(
            question="How does Pt/C affect HER activity?",
            artifact_dir=str(self.temp_dir / "artifacts_invalid_json"),
        )

        review_statuses = _read_json(result.artifact_paths["review_statuses"])
        self.assertEqual("incomplete", result.review_completion_status)
        self.assertEqual("rejected", result.acceptance_status)
        self.assertEqual("invalid_json", review_statuses[0]["status"])
        self.assertNotIn("final_submission", result.artifact_paths)
        self.assertIn("acceptance_decision", result.artifact_paths)

    def test_workflow_treats_salvaged_reviewer_as_completed_for_review_completion(self):
        workflow = self._make_workflow()

        class _SalvagedReviewer:
            def __init__(self, reviewer_role: str) -> None:
                self.reviewer_role = reviewer_role

            def run(self, *, workspace, submission, proposer_trajectory, cycle_number, session):
                trajectory = _trajectory(f"{self.reviewer_role} salvaged")
                return (
                    [],
                    trajectory,
                    ReviewerRunStatus(
                        reviewer_role=self.reviewer_role,
                        status="salvaged",
                        message="salvaged reviewer output",
                        cycle_number=cycle_number,
                        retrieval_actions_used=session.budget_state.actions_used,
                        retrieval_budget_limit=session.budget_state.budget_limit,
                        budget_blocked_calls=session.budget_state.blocked_calls,
                    ),
                )

        workflow.reviewers = {
            "search_coverage": _SalvagedReviewer("search_coverage"),
            "evidence_trace": _ParallelReviewer("evidence_trace", threading.Barrier(1), 0.0, []),
            "reasoning_consistency": _ParallelReviewer("reasoning_consistency", threading.Barrier(1), 0.0, []),
            "counterevidence": _ParallelReviewer("counterevidence", threading.Barrier(1), 0.0, []),
        }

        result = workflow.run(
            question="How does Pt/C affect HER activity?",
            artifact_dir=str(self.temp_dir / "artifacts_salvaged"),
        )

        review_statuses = _read_json(result.artifact_paths["review_statuses"])
        self.assertEqual("completed", result.review_completion_status)
        self.assertEqual("accepted", result.acceptance_status)
        self.assertEqual("salvaged", review_statuses[0]["status"])

    def test_workflow_rejects_off_topic_submission_even_when_reviewers_complete(self):
        workflow = self._make_workflow()

        class _EntityResolverWithEthanol:
            def run(self, *, question: str, task_spec: TaskSpec):
                return EntityPack.model_validate(
                    {
                        "entities": [
                            {
                                "entity_id": "ent_1",
                                "mention": "ethanol",
                                "canonical_name": "ethanol",
                                "entity_type": "molecule",
                                "aliases": ["EtOH"],
                                "query_anchors": ["ethanol", "C2H6O"],
                                "resolver_source": "fixture",
                                "resolution_confidence": 0.9,
                                "status": "resolved",
                                "source_text": question,
                                "source_span": {"start": 0, "end": len(question)},
                            }
                        ]
                    }
                )

        class _BadFactProposer:
            def run(self, *, workspace: ReactReviewedWorkspace, cycle_number: int, open_review_items):
                trajectory = _trajectory("off-topic proposer")
                submission = _submission(
                    workspace.question,
                    cycle_number=cycle_number,
                    trajectory_id=trajectory.trajectory_id,
                ).model_copy(
                    update={
                        "sections": [
                            SubmissionSection(
                                section_id="direct_answer",
                                title="Direct Answer",
                                content="Flexible printed zinc-air battery is an energy storage technology.",
                                citation_ids=["CIT-1"],
                                step_refs=[SubmissionStepRef(trajectory_id=trajectory.trajectory_id, step_number=1)],
                                issue_refs=[],
                                section_confidence=_submission_confidence(),
                            )
                        ]
                    }
                )
                return submission, trajectory

        workflow.entity_agent = workflow.entity_agent.__class__(resolver=_EntityResolverWithEthanol())
        workflow.proposer = _BadFactProposer()
        workflow.reviewers = {
            "search_coverage": _ParallelReviewer("search_coverage", threading.Barrier(1), 0.0, []),
            "evidence_trace": _ParallelReviewer("evidence_trace", threading.Barrier(1), 0.0, []),
            "reasoning_consistency": _ParallelReviewer("reasoning_consistency", threading.Barrier(1), 0.0, []),
            "counterevidence": _ParallelReviewer("counterevidence", threading.Barrier(1), 0.0, []),
        }

        result = workflow.run(
            question="What is the molecular formula of ethanol?",
            artifact_dir=str(self.temp_dir / "artifacts_off_topic"),
        )

        self.assertEqual("completed", result.review_completion_status)
        self.assertEqual("rejected", result.acceptance_status)
        self.assertTrue(result.insufficient_evidence)
        self.assertIn("Acceptance Rejected", result.final_answer)
        self.assertNotIn("Flexible printed zinc-air battery", result.final_answer)
        self.assertNotIn("final_submission", result.artifact_paths)

    def test_workflow_rejects_submission_without_evidence_anchors(self):
        workflow = self._make_workflow()

        class _AnchorlessProposer:
            def run(self, *, workspace: ReactReviewedWorkspace, cycle_number: int, open_review_items):
                trajectory = _trajectory("anchorless proposer")
                submission = _submission(
                    workspace.question,
                    cycle_number=cycle_number,
                    trajectory_id=trajectory.trajectory_id,
                ).model_copy(
                    update={
                        "citations": [
                            SubmissionCitation(
                                citation_id="CIT-1",
                                paper_id="paper-1",
                                title="Pt/C HER in alkaline media",
                                year=2024,
                                section_ids=[],
                                evidence_ids=[],
                            )
                        ]
                    }
                )
                return submission, trajectory

        workflow.proposer = _AnchorlessProposer()
        workflow.reviewers = {
            "search_coverage": _ParallelReviewer("search_coverage", threading.Barrier(1), 0.0, []),
            "evidence_trace": _ParallelReviewer("evidence_trace", threading.Barrier(1), 0.0, []),
            "reasoning_consistency": _ParallelReviewer("reasoning_consistency", threading.Barrier(1), 0.0, []),
            "counterevidence": _ParallelReviewer("counterevidence", threading.Barrier(1), 0.0, []),
        }

        result = workflow.run(
            question="How does Pt/C affect HER activity?",
            artifact_dir=str(self.temp_dir / "artifacts_anchorless"),
        )

        decision = _read_json(result.artifact_paths["acceptance_decision"])
        self.assertEqual("rejected", result.acceptance_status)
        self.assertIn("evidence_anchor", decision["blocker_codes"])
        self.assertNotIn("final_submission", result.artifact_paths)

    def test_reviewers_run_in_parallel_and_merge_in_fixed_role_order(self):
        workflow = self._make_workflow()
        start_events: list[tuple[str, str]] = []
        barrier = threading.Barrier(4)
        workflow.reviewers = {
            "search_coverage": _ParallelReviewer("search_coverage", barrier, 0.30, start_events),
            "evidence_trace": _ParallelReviewer("evidence_trace", barrier, 0.20, start_events),
            "reasoning_consistency": _ParallelReviewer("reasoning_consistency", barrier, 0.10, start_events),
            "counterevidence": _ParallelReviewer("counterevidence", barrier, 0.00, start_events),
        }

        result = workflow.run(
            question="How does Pt/C affect HER activity?",
            artifact_dir=str(self.temp_dir / "artifacts"),
        )

        expected_roles = [
            "search_coverage",
            "evidence_trace",
            "reasoning_consistency",
            "counterevidence",
        ]
        self.assertEqual(4, len(start_events))
        self.assertGreaterEqual(len({thread_name for _, thread_name in start_events}), 2)

        cycle_states = _read_json(result.artifact_paths["submission_cycles"])
        self.assertEqual(
            expected_roles,
            [item["reviewer_role"] for item in cycle_states[0]["reviewer_statuses"]],
        )
        review_statuses = _read_json(result.artifact_paths["review_statuses"])
        self.assertEqual(expected_roles, [item["reviewer_role"] for item in review_statuses])
        reviewer_trajectory_keys = list(_read_json(result.artifact_paths["reviewer_trajectories"]).keys())
        self.assertEqual(expected_roles, reviewer_trajectory_keys)
        for reviewer_role in expected_roles:
            budget_usage_path = self.temp_dir / "artifacts" / "reviewers" / reviewer_role / "cycle_1" / "budget_usage.json"
            self.assertTrue(budget_usage_path.exists(), str(budget_usage_path))

    def test_search_cache_dedupes_and_only_one_session_is_charged(self):
        retriever = _CountingRetriever(delay=0.2)
        workspace = self._make_workspace(retriever=retriever)
        sessions = [self._make_session("search_coverage", 1), self._make_session("counterevidence", 1)]
        barrier = threading.Barrier(2)
        errors: list[BaseException] = []

        def _worker(session: ReviewerSession) -> None:
            try:
                barrier.wait(timeout=3)
                workspace.search_papers(
                    query_text="Pt/C HER alkaline",
                    lane="contrarian",
                    artifact_store=session.artifact_store,
                    session=session,
                    charge_budget=True,
                    requested_via="search_papers",
                    write_snapshot=False,
                )
            except BaseException as exc:  # pragma: no cover - test plumbing
                errors.append(exc)

        threads = [threading.Thread(target=_worker, args=(session,)) for session in sessions]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if errors:
            raise errors[0]

        self.assertEqual(1, retriever.calls)
        self.assertEqual([0, 1], sorted(session.budget_state.actions_used for session in sessions))

    def test_acquire_document_cache_dedupes(self):
        document_acquirer = _CountingDocumentAcquirer(delay=0.2)
        workspace = self._make_workspace(document_acquirer=document_acquirer)
        workspace.paper_candidates["paper-1"] = PaperCandidate(
            paper_id="paper-1",
            title="Pt/C HER in alkaline media",
            year=2024,
            provider_hits=["openalex"],
            lane_sources=["contrarian"],
            retrieval_score=0.9,
        )
        sessions = [self._make_session("search_coverage", 1), self._make_session("counterevidence", 1)]
        barrier = threading.Barrier(2)
        errors: list[BaseException] = []

        def _worker(session: ReviewerSession) -> None:
            try:
                barrier.wait(timeout=3)
                workspace.acquire_document(
                    paper_id="paper-1",
                    artifact_store=session.artifact_store,
                    session=session,
                    charge_budget=True,
                    requested_via="acquire_document",
                    write_snapshot=False,
                )
            except BaseException as exc:  # pragma: no cover - test plumbing
                errors.append(exc)

        threads = [threading.Thread(target=_worker, args=(session,)) for session in sessions]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if errors:
            raise errors[0]

        self.assertEqual(1, document_acquirer.calls)
        self.assertEqual([0, 1], sorted(session.budget_state.actions_used for session in sessions))

    def test_extract_evidence_cache_dedupes_without_double_charging_acquire(self):
        document_acquirer = _CountingDocumentAcquirer(delay=0.2)
        evidence_extractor = _CountingEvidenceExtractor(delay=0.2)
        workspace = self._make_workspace(document_acquirer=document_acquirer, evidence_extractor=evidence_extractor)
        workspace.paper_candidates["paper-1"] = PaperCandidate(
            paper_id="paper-1",
            title="Pt/C HER in alkaline media",
            year=2024,
            provider_hits=["openalex"],
            lane_sources=["contrarian"],
            retrieval_score=0.9,
        )
        sessions = [self._make_session("search_coverage", 1), self._make_session("counterevidence", 1)]
        barrier = threading.Barrier(2)
        errors: list[BaseException] = []

        def _worker(session: ReviewerSession) -> None:
            try:
                barrier.wait(timeout=3)
                workspace.extract_evidence(
                    paper_id="paper-1",
                    artifact_store=session.artifact_store,
                    session=session,
                    charge_budget=True,
                    requested_via="extract_evidence",
                    write_snapshot=False,
                )
            except BaseException as exc:  # pragma: no cover - test plumbing
                errors.append(exc)

        threads = [threading.Thread(target=_worker, args=(session,)) for session in sessions]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if errors:
            raise errors[0]

        self.assertEqual(1, document_acquirer.calls)
        self.assertEqual(1, evidence_extractor.calls)
        self.assertEqual([0, 1], sorted(session.budget_state.actions_used for session in sessions))

    def test_workspace_stage_watchdog_records_timeout_for_acquire_document(self):
        document_acquirer = _CountingDocumentAcquirer(delay=0.2)
        workspace = self._make_workspace(document_acquirer=document_acquirer, stage_watchdog_seconds=0.05)
        workspace.paper_candidates["paper-1"] = PaperCandidate(
            paper_id="paper-1",
            title="Pt/C HER in alkaline media",
            year=2024,
            provider_hits=["openalex"],
            lane_sources=["contrarian"],
            retrieval_score=0.9,
        )

        started_at = time.perf_counter()
        with self.assertRaises(TimeoutError):
            workspace.acquire_document(
                paper_id="paper-1",
                artifact_store=workspace.store,
                write_snapshot=False,
            )
        elapsed = time.perf_counter() - started_at

        stage_status = _read_json(str(self.temp_dir / "workspace" / "diagnostics" / "runtime_stage_status.json"))
        self.assertEqual("acquire_document", stage_status["stage"])
        self.assertEqual("timeout", stage_status["status"])
        self.assertLess(elapsed, 0.15)

    def test_budget_block_prevents_state_mutation(self):
        retriever = _CountingRetriever()
        workspace = self._make_workspace(retriever=retriever)
        session = self._make_session("evidence_trace", 0)

        with self.assertRaises(ReviewerBudgetBlocked) as ctx:
            workspace.search_papers(
                query_text="Pt/C HER alkaline",
                lane="contrarian",
                artifact_store=session.artifact_store,
                session=session,
                charge_budget=True,
                requested_via="search_papers",
                write_snapshot=False,
            )

        self.assertTrue(ctx.exception.payload["__budget_blocked__"])
        self.assertEqual(0, retriever.calls)
        self.assertEqual(1, session.budget_state.blocked_calls)
        self.assertEqual({}, workspace.paper_candidates)

    def test_role_budget_limits_search_coverage_and_allows_counterevidence_two_step_flow(self):
        search_workspace = self._make_workspace(retriever=_CountingRetriever(), document_acquirer=_CountingDocumentAcquirer())
        search_session = self._make_session("search_coverage", 1)
        search_workspace.search_papers(
            query_text="Pt/C HER alkaline",
            lane="contrarian",
            artifact_store=search_session.artifact_store,
            session=search_session,
            charge_budget=True,
            requested_via="search_papers",
            write_snapshot=False,
        )
        with self.assertRaises(ReviewerBudgetBlocked):
            search_workspace.acquire_document(
                paper_id="paper-1",
                artifact_store=search_session.artifact_store,
                session=search_session,
                charge_budget=True,
                requested_via="acquire_document",
                write_snapshot=False,
            )
        self.assertEqual(1, search_session.budget_state.actions_used)

        counter_workspace = self._make_workspace(
            retriever=_CountingRetriever(),
            document_acquirer=_CountingDocumentAcquirer(),
            evidence_extractor=_CountingEvidenceExtractor(),
        )
        counter_session = self._make_session("counterevidence", 2)
        counter_workspace.search_papers(
            query_text="Pt/C HER alkaline",
            lane="contrarian",
            artifact_store=counter_session.artifact_store,
            session=counter_session,
            charge_budget=True,
            requested_via="search_papers",
            write_snapshot=False,
        )
        payload = counter_workspace.extract_evidence(
            paper_id="paper-1",
            artifact_store=counter_session.artifact_store,
            session=counter_session,
            charge_budget=True,
            requested_via="extract_evidence",
            write_snapshot=False,
        )
        self.assertTrue(payload)
        self.assertEqual(2, counter_session.budget_state.actions_used)

    def test_inspect_entity_cache_reads_resolution_snapshot_without_retrieval_side_effects(self):
        retriever = _CountingRetriever()
        workspace = self._make_workspace(
            retriever=retriever,
            entity_resolution_snapshot=_entity_resolution_snapshot(),
        )

        payload = workspace.inspect_entity_cache(name="EtOH", entity_type="solvent", limit=5)

        self.assertEqual(1, payload["count"])
        self.assertEqual("ethanol", payload["entries"][0]["canonical_name"])
        self.assertEqual("pubchem", payload["entries"][0]["resolver_source"])
        self.assertEqual("EtOH", payload["entries"][0]["aliases"][0])
        self.assertEqual(1, len(payload["provider_calls"]))
        self.assertEqual(0, retriever.calls)
        self.assertEqual({}, workspace.paper_candidates)

    def test_fetch_citation_context_prefers_explicit_citation_evidence_items(self):
        workspace = self._make_workspace()
        workspace.evidence_items["ev-1"] = EvidenceItem.model_validate(
            {
                "evidence_id": "ev-1",
                "paper_id": "paper-1",
                "doi": "10.1000/example",
                "section_id": "sec_results",
                "section_type": "results",
                "role": "observation",
                "snippet": "Pt/C reaches 10 mA cm-2 at 45 mV in 1.0 M KOH.",
                "source_span": {"start": 0, "end": 48},
                "source_layer": "fulltext",
                "claim_polarity": "support",
                "conditions": {"electrolyte": "1.0 M KOH"},
                "condition_source_refs": [],
                "metric_mentions": ["10 mA cm-2", "45 mV"],
                "entity_mentions": ["Pt/C"],
                "extraction_confidence": 0.8,
            }
        )
        workspace.set_review_context(
            submission=_submission(workspace.question),
            proposer_trajectory=_trajectory("citation context"),
            open_review_items=[],
            cycle_number=1,
        )

        payload = workspace.fetch_citation_context(citation_id="CIT-1")

        self.assertEqual("CIT-1", payload["citation_id"])
        self.assertEqual(["ev-1"], payload["evidence_ids"])
        self.assertEqual("ev-1", payload["evidence"][0]["evidence_id"])
        self.assertEqual("fulltext", payload["evidence"][0]["source_layer"])

    def test_retriever_search_honors_openalex_semantic_scholar_crossref_priority(self):
        call_order: list[str] = []

        class _OpenAlexClient:
            def search(self, query_plan, limit=8):
                call_order.append("openalex")
                return [
                    {
                        "display_name": "Pt/C alkaline HER benchmark",
                        "doi": "10.1000/openalex",
                        "publication_year": 2024,
                        "abstract": "Pt/C benchmark for alkaline HER in KOH.",
                        "authorships": [],
                        "best_oa_location": {"landing_page_url": "https://example.org/openalex"},
                    }
                ]

        class _SemanticScholarClient:
            def search(self, query_plan, limit=8):
                call_order.append("semantic_scholar")
                return [
                    {
                        "title": "Commercial Pt/C in alkaline HER",
                        "abstract": "Commercial Pt/C benchmark with overpotential and Tafel metrics.",
                        "year": 2023,
                        "venue": "Journal",
                        "authors": [],
                        "externalIds": {"DOI": "10.1000/sem"},
                        "openAccessPdf": {"url": "https://example.org/sem.pdf"},
                    }
                ]

            def enrich(self, candidate):
                return None

        class _CrossrefClient:
            def search(self, query_plan, limit=8):
                call_order.append("crossref")
                return [
                    {
                        "title": ["Pt foil and Pt/C side-by-side alkaline HER"],
                        "DOI": "10.1000/crossref",
                        "issued": {"date-parts": [[2022]]},
                        "container-title": ["Journal"],
                        "author": [],
                    }
                ]

            def enrich(self, candidate):
                return None

        retriever = RetrieverNode(
            openalex_client=_OpenAlexClient(),
            semantic_scholar_client=_SemanticScholarClient(),
            crossref_client=_CrossrefClient(),
            per_lane_limit=4,
            final_top_k=6,
        )
        candidates = retriever.run(
            task_spec=_task_spec("Does Pt/C improve HER activity in alkaline media?"),
            entity_pack=_entity_pack(),
            query_plans=[
                QueryPlan(
                    lane="data",
                    query_text="Pt/C HER alkaline benchmark overpotential",
                    must_terms=["Pt/C", "HER", "alkaline"],
                    exclude_terms=[],
                    preferred_sources=["openalex", "semantic_scholar", "crossref"],
                )
            ],
            artifact_store=QAArtifactStore(base_dir=self.temp_dir / "retriever_priority"),
        )

        self.assertEqual(["openalex", "semantic_scholar", "crossref"], call_order[:3])
        self.assertEqual(3, len(candidates))
        provider_sets = [set(candidate.provider_hits) for candidate in candidates]
        self.assertTrue(any("semantic_scholar" in provider_hits for provider_hits in provider_sets))
        self.assertTrue(any("crossref" in provider_hits for provider_hits in provider_sets))

    def test_react_reviewed_run_writes_all_top_level_and_nested_artifacts(self):
        workflow = self._make_workflow()
        workflow.reviewers = {
            "search_coverage": _SearchingReviewer("search_coverage"),
            "evidence_trace": _ParallelReviewer("evidence_trace", threading.Barrier(1), 0.0, []),
            "reasoning_consistency": _ParallelReviewer("reasoning_consistency", threading.Barrier(1), 0.0, []),
            "counterevidence": _ParallelReviewer("counterevidence", threading.Barrier(1), 0.0, []),
        }

        result = workflow.run(
            question="How does Pt/C affect HER activity?",
            artifact_dir=str(self.temp_dir / "artifacts"),
        )

        artifact_root = self.temp_dir / "artifacts"
        for name in (
            "candidate_submission.json",
            "acceptance_decision.json",
            "final_submission.json",
            "submission_trace.json",
            "submission_cycles.json",
            "proposer_trajectory.json",
            "reviewer_trajectories.json",
            "review_statuses.json",
            "final_review_items.json",
            "final_answer.md",
            "qa_result.json",
            "retrieval_diagnostics.json",
            "provider_health.json",
        ):
            self.assertTrue((artifact_root / name).exists(), name)
        for reviewer_role in ("search_coverage", "evidence_trace", "reasoning_consistency", "counterevidence"):
            cycle_root = artifact_root / "reviewers" / reviewer_role / "cycle_1"
            self.assertTrue((cycle_root / "budget_usage.json").exists())
            self.assertTrue((cycle_root / "reviewer_status.json").exists())
            self.assertTrue((cycle_root / "reviewer_trajectory.json").exists())
        self.assertEqual("react_reviewed", _read_json(result.artifact_paths["qa_result"])["workflow_mode"])

    def test_react_reviewed_entity_and_router_artifacts_are_written(self):
        workflow = self._make_workflow()
        workflow.reviewers = {
            "search_coverage": _SearchingReviewer("search_coverage"),
            "evidence_trace": _ParallelReviewer("evidence_trace", threading.Barrier(1), 0.0, []),
            "reasoning_consistency": _ParallelReviewer("reasoning_consistency", threading.Barrier(1), 0.0, []),
            "counterevidence": _ParallelReviewer("counterevidence", threading.Barrier(1), 0.0, []),
        }

        workflow.run(
            question="How does Pt/C affect HER activity?",
            artifact_dir=str(self.temp_dir / "artifacts"),
        )

        artifact_root = self.temp_dir / "artifacts"
        for path in (
            artifact_root / "router" / "task_spec.json",
            artifact_root / "router" / "agent_run.json",
            artifact_root / "entity_resolver" / "entity_pack.json",
            artifact_root / "entity_resolver" / "resolution_index.json",
            artifact_root / "entity_resolver" / "provider_calls.json",
            artifact_root / "entity_resolver" / "seed_suggestions.json",
            artifact_root / "entity_resolver" / "agent_run.json",
        ):
            self.assertTrue(path.exists(), str(path))

    def test_react_reviewed_qa_result_matches_final_submission_and_trace(self):
        workflow = self._make_workflow()
        workflow.reviewers = {
            "search_coverage": _SearchingReviewer("search_coverage"),
            "evidence_trace": _ParallelReviewer("evidence_trace", threading.Barrier(1), 0.0, []),
            "reasoning_consistency": _ParallelReviewer("reasoning_consistency", threading.Barrier(1), 0.0, []),
            "counterevidence": _ParallelReviewer("counterevidence", threading.Barrier(1), 0.0, []),
        }

        result = workflow.run(
            question="How does Pt/C affect HER activity?",
            artifact_dir=str(self.temp_dir / "artifacts"),
        )

        qa_result_payload = _read_json(result.artifact_paths["qa_result"])
        review_statuses = _read_json(result.artifact_paths["review_statuses"])
        self.assertEqual(
            [item.model_dump(exclude_none=True) for item in result.submission_trace],
            _read_json(result.artifact_paths["submission_trace"]),
        )
        self.assertEqual(
            result.final_answer,
            Path(result.artifact_paths["final_answer"]).read_text(encoding="utf-8"),
        )
        self.assertEqual("react_reviewed", qa_result_payload["workflow_mode"])
        self.assertEqual("accepted", qa_result_payload["acceptance_status"])
        self.assertEqual(result.review_completion_status, qa_result_payload["review_completion_status"])
        self.assertEqual(
            "completed" if all(item["status"] == "completed" for item in review_statuses) else "incomplete",
            qa_result_payload["review_completion_status"],
        )

    def test_qa_result_file_matches_returned_model_dump(self):
        workflow = self._make_workflow()
        workflow.reviewers = {
            "search_coverage": _SearchingReviewer("search_coverage"),
            "evidence_trace": _ParallelReviewer("evidence_trace", threading.Barrier(1), 0.0, []),
            "reasoning_consistency": _ParallelReviewer("reasoning_consistency", threading.Barrier(1), 0.0, []),
            "counterevidence": _ParallelReviewer("counterevidence", threading.Barrier(1), 0.0, []),
        }

        result = workflow.run(
            question="How does Pt/C affect HER activity?",
            artifact_dir=str(self.temp_dir / "artifacts"),
        )

        self.assertEqual(result.model_dump(exclude_none=True), _read_json(result.artifact_paths["qa_result"]))

    def test_final_answer_markdown_matches_result_final_answer(self):
        workflow = self._make_workflow()
        workflow.reviewers = {
            "search_coverage": _SearchingReviewer("search_coverage"),
            "evidence_trace": _ParallelReviewer("evidence_trace", threading.Barrier(1), 0.0, []),
            "reasoning_consistency": _ParallelReviewer("reasoning_consistency", threading.Barrier(1), 0.0, []),
            "counterevidence": _ParallelReviewer("counterevidence", threading.Barrier(1), 0.0, []),
        }

        result = workflow.run(
            question="How does Pt/C affect HER activity?",
            artifact_dir=str(self.temp_dir / "artifacts"),
        )

        self.assertEqual(
            result.final_answer,
            Path(result.artifact_paths["final_answer"]).read_text(encoding="utf-8"),
        )

    def test_artifact_paths_only_reference_existing_files(self):
        workflow = self._make_workflow()
        workflow.reviewers = {
            "search_coverage": _SearchingReviewer("search_coverage"),
            "evidence_trace": _ParallelReviewer("evidence_trace", threading.Barrier(1), 0.0, []),
            "reasoning_consistency": _ParallelReviewer("reasoning_consistency", threading.Barrier(1), 0.0, []),
            "counterevidence": _ParallelReviewer("counterevidence", threading.Barrier(1), 0.0, []),
        }

        result = workflow.run(
            question="How does Pt/C affect HER activity?",
            artifact_dir=str(self.temp_dir / "artifacts"),
        )

        for path in result.artifact_paths.values():
            self.assertTrue(Path(path).exists(), path)

    def test_provider_health_and_retrieval_diagnostics_survive_to_final_report(self):
        workflow = self._make_workflow()
        workflow.reviewers = {
            "search_coverage": _SearchingReviewer("search_coverage"),
            "evidence_trace": _ParallelReviewer("evidence_trace", threading.Barrier(1), 0.0, []),
            "reasoning_consistency": _ParallelReviewer("reasoning_consistency", threading.Barrier(1), 0.0, []),
            "counterevidence": _ParallelReviewer("counterevidence", threading.Barrier(1), 0.0, []),
        }

        result = workflow.run(
            question="How does Pt/C affect HER activity?",
            artifact_dir=str(self.temp_dir / "artifacts"),
        )

        self.assertTrue((self.temp_dir / "artifacts" / "provider_health.json").exists())
        self.assertTrue((self.temp_dir / "artifacts" / "retrieval_diagnostics.json").exists())
        self.assertEqual("", result.retrieval_diagnostics_summary)

    def test_validate_review_payload_discards_placeholder_items_without_targeted_feedback(self):
        workflow = self._make_workflow()
        reviewer = workflow.reviewers["search_coverage"]
        submission = _submission("How does Pt/C affect HER activity?")
        proposer_trajectory = _trajectory("review placeholder validation")

        items = reviewer._validate_review_payload(
            review={"review_items": [{"anchor_kind": "global", "severity": "warning"}]},
            submission=submission,
            proposer_trajectory=proposer_trajectory,
            agent=None,
        )

        self.assertEqual([], items)

    def test_normalize_review_items_drops_placeholder_manual_review_item(self):
        workflow = self._make_workflow()
        proposer_trajectory = _trajectory("normalize reviewer placeholders")
        items = workflow._normalize_review_items(
            items=[
                ReviewItem(
                    review_id="search_coverage_1",
                    reviewer_role="search_coverage",
                    anchor_kind="global",
                    severity="blocking",
                    flaw_type="needs_manual_review",
                    critique="Reviewer output did not provide critique text.",
                    required_action="Re-check the anchored section.",
                )
            ],
            proposer_trajectory=proposer_trajectory,
            max_items_per_step_section=1,
        )

        self.assertEqual([], items)

    def test_workflow_returns_rejected_qa_result_when_proposer_execution_fails(self):
        workflow = self._make_workflow()
        workflow.proposer = _FailingProposer()

        result = workflow.run(
            question="How does Pt/C affect HER activity?",
            artifact_dir=str(self.temp_dir / "artifacts"),
        )

        self.assertEqual("rejected", result.acceptance_status)
        self.assertEqual("incomplete", result.review_completion_status)
        self.assertTrue(Path(result.artifact_paths["qa_result"]).exists())
        self.assertTrue(Path(result.artifact_paths["workflow_error"]).exists())
        self.assertIn("synthetic proposer failure", Path(result.artifact_paths["workflow_error"]).read_text(encoding="utf-8"))
        self.assertIn("synthetic proposer failure", result.final_answer)


if __name__ == "__main__":
    unittest.main()
