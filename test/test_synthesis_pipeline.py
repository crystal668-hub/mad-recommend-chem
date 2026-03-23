from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

from qa.nodes.answer_validator import AnswerValidationError, AnswerValidator
from qa.nodes.synthesizer import SynthesizerExecutionError, SynthesizerNode
from qa.retrieval_state import EvidenceLedger
from qa.synthesis_pipeline import VerifiedSynthesisPipeline
from qa.synthesis_state import (
    AnswerSectionOutput,
    CitationRecord,
    ConfidenceRating,
    QAResult,
    SectionClaimPack,
    SectionConfidenceRecord,
    SynthesisInputPack,
)
from test.qa_test_utils import make_task_spec, read_json


def _confidence(level: str = "high", score: float = 0.82, rationale: str = "fixture") -> ConfidenceRating:
    return ConfidenceRating(level=level, score=score, rationale=rationale)


def _input_pack(*, insufficient_evidence: bool = False) -> SynthesisInputPack:
    task_spec = make_task_spec()
    section_confidence = _confidence("high", 0.84)
    limitations_confidence = _confidence("low", 0.25)
    return SynthesisInputPack(
        question=task_spec.question,
        task_spec=task_spec,
        section_claims=[
            SectionClaimPack(
                section_id="direct_answer",
                title="Direct Answer",
                accepted_claim_ids=["claim-1"],
                claim_summaries=["Pt/C improves HER activity in 1 M KOH."],
                core_citation_ids=["CIT-1"],
                section_confidence=section_confidence,
            )
        ],
        citation_catalog=[
            CitationRecord(
                citation_id="CIT-1",
                paper_id="paper-1",
                title="Pt/C HER in alkaline media",
                year=2024,
                supporting_claim_ids=["claim-1"],
            )
        ],
        overall_confidence=section_confidence,
        section_confidence=[
            SectionConfidenceRecord(
                section_id="direct_answer",
                title="Direct Answer",
                confidence=section_confidence,
            ),
            SectionConfidenceRecord(
                section_id="limitations_controversies",
                title="Limitations / Controversies",
                confidence=limitations_confidence,
            ),
        ],
        insufficient_evidence=insufficient_evidence,
        claim_trace=[
            {
                "section_id": "direct_answer",
                "claim_id": "claim-1",
                "status": "accepted",
                "citation_ids": ["CIT-1"],
                "confidence": 0.84,
            }
        ],
        retrieval_diagnostics_summary="",
        execution_warnings=[],
    )


def _valid_result(input_pack: SynthesisInputPack) -> QAResult:
    return QAResult(
        question=input_pack.question,
        language="en",
        final_answer="## Direct Answer\nPt/C improves HER activity in 1 M KOH.",
        sections=[
            AnswerSectionOutput(
                section_id="direct_answer",
                title="Direct Answer",
                content="Pt/C improves HER activity in 1 M KOH.",
                citation_ids=["CIT-1"],
                section_confidence=input_pack.section_claims[0].section_confidence,
            )
        ],
        citations=list(input_pack.citation_catalog),
        claim_trace=list(input_pack.claim_trace),
        overall_confidence=input_pack.overall_confidence,
        section_confidence=[input_pack.section_confidence[0]],
        insufficient_evidence=input_pack.insufficient_evidence,
        limitations_summary="",
        retrieval_diagnostics_summary=input_pack.retrieval_diagnostics_summary,
        execution_warnings=list(input_pack.execution_warnings),
        artifact_paths={},
        time_elapsed=0.0,
    )


class _FakeLLM:
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)

    def invoke(self, messages):
        if not self.responses:
            raise AssertionError("LLM invoked more times than expected.")
        return self.responses.pop(0)


class _StaticPackBuilder:
    def __init__(self, input_pack: SynthesisInputPack) -> None:
        self.input_pack = input_pack

    def run(self, **kwargs):
        return self.input_pack


class _SuccessfulSynthesizer:
    def run(self, input_pack: SynthesisInputPack) -> QAResult:
        return _valid_result(input_pack)


class _InvalidSynthesizer:
    def run(self, input_pack: SynthesisInputPack) -> QAResult:
        return QAResult(
            question=input_pack.question,
            language="en",
            final_answer="No structured answer.",
            sections=[],
            citations=[],
            claim_trace=[],
            overall_confidence=input_pack.overall_confidence,
            section_confidence=[],
            insufficient_evidence=input_pack.insufficient_evidence,
            limitations_summary="",
            retrieval_diagnostics_summary=input_pack.retrieval_diagnostics_summary,
            execution_warnings=[],
            artifact_paths={},
            time_elapsed=0.0,
        )


class SynthesizerNodeTests(unittest.TestCase):
    def test_missing_llm_raises_typed_error(self):
        node = SynthesizerNode(llm=None)

        with self.assertRaises(SynthesizerExecutionError) as ctx:
            node.run(_input_pack())

        self.assertEqual("startup", ctx.exception.stage)
        self.assertIn("LLM is unavailable", ctx.exception.reason)
        self.assertEqual("synthesizer_execution_failed", node.last_run_debug["failure"]["error"])

    def test_invalid_output_raises_typed_error(self):
        node = SynthesizerNode(llm=_FakeLLM(["not json"]))

        with self.assertRaises(SynthesizerExecutionError) as ctx:
            node.run(_input_pack())

        self.assertEqual("synthesis", ctx.exception.stage)
        self.assertIn("returned unusable output", ctx.exception.reason)


class AnswerValidatorTests(unittest.TestCase):
    def test_structural_issue_raises_typed_error(self):
        validator = AnswerValidator()
        input_pack = _input_pack()
        invalid_result = QAResult(
            question=input_pack.question,
            language="en",
            final_answer="No structured answer.",
            sections=[],
            citations=[],
            claim_trace=[],
            overall_confidence=input_pack.overall_confidence,
            section_confidence=[],
            insufficient_evidence=input_pack.insufficient_evidence,
            limitations_summary="",
            retrieval_diagnostics_summary="",
            execution_warnings=[],
            artifact_paths={},
            time_elapsed=0.0,
        )

        with self.assertRaises(AnswerValidationError) as ctx:
            validator.run(input_pack=input_pack, draft_result=invalid_result)

        self.assertEqual("validation", ctx.exception.stage)
        self.assertIn("did not contain any sections", ctx.exception.reason)
        self.assertEqual("answer_validation_failed", validator.last_run_debug["failure"]["error"])


class VerifiedSynthesisPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(".cache") / f"qa_synthesis_{uuid.uuid4().hex[:8]}"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_pipeline_writes_synthesizer_failure_artifacts_and_stops_before_qa_result(self):
        input_pack = _input_pack()
        pipeline = VerifiedSynthesisPipeline(
            pack_builder=_StaticPackBuilder(input_pack),
            synthesizer=SynthesizerNode(llm=None),
            answer_validator=AnswerValidator(),
        )
        artifact_dir = self.temp_dir / "synthesizer_failure"

        with self.assertRaises(SynthesizerExecutionError):
            pipeline.run(
                task_spec=input_pack.task_spec,
                evidence_ledger=EvidenceLedger(),
                paper_records=[],
                artifact_dir=str(artifact_dir),
            )

        self.assertTrue((artifact_dir / "synthesis_input_pack.json").exists())
        self.assertTrue((artifact_dir / "synthesizer" / "failure.json").exists())
        self.assertTrue((artifact_dir / "synthesizer" / "agent_run.json").exists())
        self.assertFalse((artifact_dir / "qa_result.json").exists())
        failure_payload = read_json(artifact_dir / "synthesizer" / "failure.json")
        self.assertEqual("synthesizer_execution_failed", failure_payload["error"])

    def test_pipeline_writes_answer_validator_failure_artifacts_and_stops_before_qa_result(self):
        input_pack = _input_pack()
        pipeline = VerifiedSynthesisPipeline(
            pack_builder=_StaticPackBuilder(input_pack),
            synthesizer=_InvalidSynthesizer(),
            answer_validator=AnswerValidator(),
        )
        artifact_dir = self.temp_dir / "answer_validator_failure"

        with self.assertRaises(AnswerValidationError):
            pipeline.run(
                task_spec=input_pack.task_spec,
                evidence_ledger=EvidenceLedger(),
                paper_records=[],
                artifact_dir=str(artifact_dir),
            )

        self.assertTrue((artifact_dir / "synthesis_input_pack.json").exists())
        self.assertTrue((artifact_dir / "answer_validator" / "failure.json").exists())
        self.assertTrue((artifact_dir / "answer_validator" / "agent_run.json").exists())
        self.assertFalse((artifact_dir / "qa_result.json").exists())
        failure_payload = read_json(artifact_dir / "answer_validator" / "failure.json")
        self.assertEqual("answer_validation_failed", failure_payload["error"])

    def test_pipeline_still_produces_qa_result_with_valid_synthesizer_output(self):
        input_pack = _input_pack()
        pipeline = VerifiedSynthesisPipeline(
            pack_builder=_StaticPackBuilder(input_pack),
            synthesizer=_SuccessfulSynthesizer(),
            answer_validator=AnswerValidator(),
        )
        artifact_dir = self.temp_dir / "synthesis_success"

        result = pipeline.run(
            task_spec=input_pack.task_spec,
            evidence_ledger=EvidenceLedger(),
            paper_records=[],
            artifact_dir=str(artifact_dir),
        )

        self.assertTrue((artifact_dir / "qa_result.json").exists())
        self.assertEqual("## Direct Answer\nPt/C improves HER activity in 1 M KOH.", result.final_answer)


if __name__ == "__main__":
    unittest.main()
