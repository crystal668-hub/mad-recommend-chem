from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

from qa.facade import QASystem
from qa.nodes.query_planner import QueryPlannerExecutionError
from qa.nodes.router import RouterExecutionError
from qa.peer_review_errors import PeerReviewExecutionError
from qa.retrieval_pipeline import HeterogeneousRetrievalPipeline
from qa.review_pipeline import StructuredPeerReviewPipeline

from test.qa_test_utils import (
    StaticClaimMiner,
    StaticDocumentAcquirer,
    StaticEvidenceExtractor,
    StaticRetriever,
    NullLedgerRetrievalPipeline,
    StaticGroundingPipeline,
    build_ledger_system,
    make_base_config,
    read_json,
)
from qa.evidence import EvidenceLedgerBuilder


class FailingGroundingPipeline:
    def run_detailed(self, question: str, context: str | None = None):
        raise RouterExecutionError(
            stage="semantic",
            reason="semantic stage returned unusable output",
            question=question,
            normalized_question=question.lower(),
            context=context,
            debug_payload={
                "input": {"question": question, "context": context},
                "normalized_question": question.lower(),
                "failure": {
                    "error": "router_execution_failed",
                    "stage": "semantic",
                    "reason": "semantic stage returned unusable output",
                },
            },
        )


class FailingQueryPlanner:
    def run(self, *, task_spec, entity_pack):
        raise QueryPlannerExecutionError(
            stage="planning",
            reason="query planner returned unusable output",
            task_spec=task_spec,
            debug_payload={
                "input": {
                    "task_spec": task_spec.model_dump(exclude_none=True),
                    "entity_pack": entity_pack.model_dump(exclude_none=True),
                },
                "failure": {
                    "error": "query_planner_execution_failed",
                    "stage": "planning",
                    "reason": "query planner returned unusable output",
                },
            },
        )


class LedgerWorkflowSystemTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(".cache") / f"qa_ledger_{uuid.uuid4().hex[:8]}"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_qasystem_dispatches_to_ledger_pipeline_and_writes_full_artifacts(self):
        system = build_ledger_system(self.temp_dir, save_output=True)
        artifact_dir = self.temp_dir / "artifacts"

        result = system.run_qa(
            question="How does Pt/C affect HER activity in 1 M KOH?",
            artifact_dir=str(artifact_dir),
        )

        self.assertEqual("ledger", result.workflow_mode)
        self.assertTrue(result.final_answer.strip())
        required_files = [
            artifact_dir / "runtime_manifest.json",
            artifact_dir / "qa_result.json",
            artifact_dir / "final_answer.md",
            artifact_dir / "synthesis_input_pack.json",
            artifact_dir / "retrieval_diagnostics.json",
            artifact_dir / "provider_health.json",
            artifact_dir / "evidence_ledger_reviewed.json",
            artifact_dir / "review_summaries.json",
        ]
        for path in required_files:
            self.assertTrue(path.exists(), str(path))
        self.assertIn("qa_result", result.artifact_paths)
        self.assertIn("final_answer", result.artifact_paths)
        self.assertIn("runtime_manifest", result.artifact_paths)
        self.assertIn("review_summaries", result.artifact_paths)
        self.assertIn("reviewed_evidence_ledger", result.artifact_paths)

    def test_ledger_entity_resolution_artifacts_are_written(self):
        system = build_ledger_system(self.temp_dir, save_output=False)
        artifact_dir = self.temp_dir / "artifacts"

        result = system.run_qa(
            question="How does Pt/C affect HER activity in 1 M KOH?",
            artifact_dir=str(artifact_dir),
        )

        entity_files = {
            "entity_pack": artifact_dir / "entity_resolver" / "entity_pack.json",
            "resolution_index": artifact_dir / "entity_resolver" / "resolution_index.json",
            "provider_calls": artifact_dir / "entity_resolver" / "provider_calls.json",
            "seed_suggestions": artifact_dir / "entity_resolver" / "seed_suggestions.json",
        }
        for key, path in entity_files.items():
            self.assertTrue(path.exists(), str(path))
            self.assertEqual(str(path), result.artifact_paths[key])

    def test_ledger_public_result_and_qa_result_are_consistent(self):
        system = build_ledger_system(self.temp_dir, save_output=True)
        artifact_dir = self.temp_dir / "artifacts"

        result = system.run_qa(
            question="How does Pt/C affect HER activity in 1 M KOH?",
            artifact_dir=str(artifact_dir),
        )

        public_payload = read_json(result.artifact_paths["public_result"])
        qa_result_payload = read_json(result.artifact_paths["qa_result"])
        self.assertEqual("ledger", public_payload["workflow_mode"])
        self.assertEqual(public_payload["workflow_mode"], qa_result_payload["workflow_mode"])
        self.assertEqual(public_payload["question"], qa_result_payload["question"])
        self.assertEqual(public_payload["final_answer"], qa_result_payload["final_answer"])
        self.assertEqual(
            public_payload["artifact_paths"]["runtime_manifest"],
            qa_result_payload["artifact_paths"]["runtime_manifest"],
        )

    def test_ledger_review_outputs_are_propagated_to_final_result(self):
        system = build_ledger_system(
            self.temp_dir,
            save_output=False,
            document_warnings=["document acquisition warning"],
            review_warnings=["peer review warning"],
        )
        artifact_dir = self.temp_dir / "artifacts"

        result = system.run_qa(
            question="How does Pt/C affect HER activity in 1 M KOH?",
            artifact_dir=str(artifact_dir),
        )

        reviewed_ledger = read_json(artifact_dir / "evidence_ledger_reviewed.json")
        review_summaries = read_json(artifact_dir / "review_summaries.json")
        self.assertEqual(len(review_summaries), len(reviewed_ledger["review_summaries"]))
        self.assertIn("document acquisition warning", result.execution_warnings)
        self.assertIn("peer review warning", result.execution_warnings)

    def test_ledger_fails_fast_when_retrieval_returns_no_ledger(self):
        config = make_base_config(self.temp_dir, workflow_mode="ledger", save_output=False)
        system = QASystem(
            config=config,
            grounding_pipeline=StaticGroundingPipeline(),
            retrieval_pipeline=NullLedgerRetrievalPipeline(),
        )
        artifact_dir = self.temp_dir / "artifacts"

        with self.assertRaises(ValueError):
            system.run_qa(
                question="How does Pt/C affect HER activity in 1 M KOH?",
                artifact_dir=str(artifact_dir),
            )

        self.assertFalse((artifact_dir / "qa_result.json").exists())

    def test_ledger_writes_router_failure_artifacts_and_stops_before_qa_result(self):
        config = make_base_config(self.temp_dir, workflow_mode="ledger", save_output=False)
        system = QASystem(
            config=config,
            grounding_pipeline=FailingGroundingPipeline(),
            retrieval_pipeline=NullLedgerRetrievalPipeline(),
        )
        artifact_dir = self.temp_dir / "artifacts"

        with self.assertRaises(RouterExecutionError):
            system.run_qa(
                question="How does Pt/C affect HER activity in 1 M KOH?",
                artifact_dir=str(artifact_dir),
            )

        self.assertTrue((artifact_dir / "runtime_manifest.json").exists())
        self.assertTrue((artifact_dir / "router" / "failure.json").exists())
        self.assertTrue((artifact_dir / "router" / "agent_run.json").exists())
        self.assertFalse((artifact_dir / "router" / "task_spec.json").exists())
        self.assertFalse((artifact_dir / "qa_result.json").exists())
        failure_payload = read_json(artifact_dir / "router" / "failure.json")
        self.assertEqual("router_execution_failed", failure_payload["error"])
        self.assertEqual("semantic", failure_payload["stage"])

    def test_ledger_writes_query_planner_failure_artifacts_and_stops_before_qa_result(self):
        system = build_ledger_system(self.temp_dir, save_output=False)
        system.retrieval_pipeline = HeterogeneousRetrievalPipeline(
            query_planner=FailingQueryPlanner(),
            retriever=StaticRetriever(),
            document_acquirer=StaticDocumentAcquirer(),
            evidence_extractor=StaticEvidenceExtractor(),
            claim_miner=StaticClaimMiner(),
            ledger_builder=EvidenceLedgerBuilder(),
            peer_review_pipeline=None,
        )
        artifact_dir = self.temp_dir / "artifacts"

        with self.assertRaises(QueryPlannerExecutionError):
            system.run_qa(
                question="How does Pt/C affect HER activity in 1 M KOH?",
                artifact_dir=str(artifact_dir),
            )

        self.assertTrue((artifact_dir / "runtime_manifest.json").exists())
        self.assertTrue((artifact_dir / "query_planner" / "failure.json").exists())
        self.assertTrue((artifact_dir / "query_planner" / "agent_run.json").exists())
        self.assertFalse((artifact_dir / "query_plans.json").exists())
        self.assertFalse((artifact_dir / "qa_result.json").exists())
        failure_payload = read_json(artifact_dir / "query_planner" / "failure.json")
        self.assertEqual("query_planner_execution_failed", failure_payload["error"])
        self.assertEqual("planning", failure_payload["stage"])

    def test_ledger_writes_peer_review_failure_artifacts_and_stops_before_qa_result(self):
        system = build_ledger_system(self.temp_dir, save_output=False)
        system.peer_review_pipeline = StructuredPeerReviewPipeline()
        artifact_dir = self.temp_dir / "artifacts"

        with self.assertRaises(PeerReviewExecutionError) as ctx:
            system.run_qa(
                question="How does Pt/C affect HER activity in 1 M KOH?",
                artifact_dir=str(artifact_dir),
            )

        self.assertEqual("peer_review_startup", ctx.exception.stage)
        self.assertTrue((artifact_dir / "runtime_manifest.json").exists())
        self.assertTrue((artifact_dir / "peer_review" / "failure.json").exists())
        self.assertTrue((artifact_dir / "peer_review" / "agent_run.json").exists())
        self.assertFalse((artifact_dir / "evidence_ledger_reviewed.json").exists())
        self.assertFalse((artifact_dir / "review_summaries.json").exists())
        self.assertFalse((artifact_dir / "qa_result.json").exists())
        failure_payload = read_json(artifact_dir / "peer_review" / "failure.json")
        self.assertEqual("peer_review_execution_failed", failure_payload["error"])
        self.assertEqual("peer_review_startup", failure_payload["stage"])
        agent_run_payload = read_json(artifact_dir / "peer_review" / "agent_run.json")
        self.assertEqual("StructuredPeerReviewPipeline", agent_run_payload["agent"])
        self.assertEqual(1, agent_run_payload["input"]["claim_count"])

    def test_qa_result_file_matches_returned_model_dump(self):
        system = build_ledger_system(self.temp_dir, save_output=False)
        artifact_dir = self.temp_dir / "artifacts"

        result = system.run_qa(
            question="How does Pt/C affect HER activity in 1 M KOH?",
            artifact_dir=str(artifact_dir),
        )

        qa_result_payload = read_json(result.artifact_paths["qa_result"])
        self.assertEqual(
            result.model_dump(exclude_none=True),
            qa_result_payload,
        )

    def test_final_answer_markdown_matches_result_final_answer(self):
        system = build_ledger_system(self.temp_dir, save_output=False)
        artifact_dir = self.temp_dir / "artifacts"

        result = system.run_qa(
            question="How does Pt/C affect HER activity in 1 M KOH?",
            artifact_dir=str(artifact_dir),
        )

        markdown = Path(result.artifact_paths["final_answer"]).read_text(encoding="utf-8")
        self.assertEqual(result.final_answer, markdown)

    def test_artifact_paths_only_reference_existing_files(self):
        system = build_ledger_system(self.temp_dir, save_output=True)
        artifact_dir = self.temp_dir / "artifacts"

        result = system.run_qa(
            question="How does Pt/C affect HER activity in 1 M KOH?",
            artifact_dir=str(artifact_dir),
        )

        for path in result.artifact_paths.values():
            self.assertTrue(Path(path).exists(), path)

    def test_provider_health_and_retrieval_diagnostics_survive_to_final_report(self):
        system = build_ledger_system(self.temp_dir, save_output=False)
        artifact_dir = self.temp_dir / "artifacts"

        result = system.run_qa(
            question="How does Pt/C affect HER activity in 1 M KOH?",
            artifact_dir=str(artifact_dir),
        )

        provider_health = read_json(artifact_dir / "provider_health.json")
        retrieval_diagnostics = read_json(artifact_dir / "retrieval_diagnostics.json")
        qa_result_payload = read_json(result.artifact_paths["qa_result"])
        self.assertIn("openalex", provider_health)
        self.assertGreaterEqual(len(retrieval_diagnostics), 1)
        self.assertEqual(result.retrieval_diagnostics_summary, qa_result_payload["retrieval_diagnostics_summary"])

    def test_save_output_false_skips_public_result_but_keeps_internal_artifacts(self):
        system = build_ledger_system(self.temp_dir, save_output=False)
        artifact_dir = self.temp_dir / "artifacts"

        result = system.run_qa(
            question="How does Pt/C affect HER activity in 1 M KOH?",
            artifact_dir=str(artifact_dir),
        )

        self.assertNotIn("public_result", result.artifact_paths)
        self.assertTrue((artifact_dir / "qa_result.json").exists())
        self.assertTrue((artifact_dir / "final_answer.md").exists())


if __name__ == "__main__":
    unittest.main()
