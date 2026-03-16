from __future__ import annotations

import time
from typing import Optional, Sequence

from qa.artifacts import QAArtifactStore
from qa.nodes.answer_validator import AnswerValidator
from qa.nodes.synthesis_pack_builder import SynthesisPackBuilder
from qa.nodes.synthesizer import SynthesizerNode
from qa.retrieval_state import EvidenceLedger, PaperRecord, RetrievalDiagnosticRecord, RetrievalState, ReviewSummary
from qa.state import TaskSpec
from qa.synthesis_state import QAResult


class VerifiedSynthesisPipeline:
    def __init__(
        self,
        *,
        pack_builder: Optional[SynthesisPackBuilder] = None,
        synthesizer: Optional[SynthesizerNode] = None,
        answer_validator: Optional[AnswerValidator] = None,
    ) -> None:
        self.pack_builder = pack_builder or SynthesisPackBuilder()
        self.synthesizer = synthesizer or SynthesizerNode()
        self.answer_validator = answer_validator or AnswerValidator()

    def run(
        self,
        *,
        task_spec: TaskSpec,
        evidence_ledger: EvidenceLedger,
        paper_records: Sequence[PaperRecord],
        review_summaries: Optional[Sequence[ReviewSummary]] = None,
        retrieval_diagnostics: Optional[Sequence[RetrievalDiagnosticRecord]] = None,
        execution_warnings: Optional[Sequence[str]] = None,
        artifact_dir: Optional[str] = None,
    ) -> QAResult:
        started_at = time.perf_counter()
        store = QAArtifactStore(base_dir=artifact_dir)

        input_pack = self.pack_builder.run(
            task_spec=task_spec,
            evidence_ledger=evidence_ledger,
            review_summaries=review_summaries,
            paper_records=paper_records,
            retrieval_diagnostics=retrieval_diagnostics,
            execution_warnings=execution_warnings,
        )
        synthesis_pack_path = store.write_json("synthesis_input_pack.json", input_pack.model_dump(exclude_none=True))

        fallback_result = self.synthesizer.build_deterministic_result(input_pack)
        draft_result = self.synthesizer.run(input_pack)
        validated_result = self.answer_validator.run(
            input_pack=input_pack,
            draft_result=draft_result,
            fallback_result=fallback_result,
        )

        final_answer_path = store.write_text("final_answer.md", validated_result.final_answer)
        qa_result_path = str(store.path("qa_result.json"))
        artifact_paths = {
            "synthesis_input_pack": synthesis_pack_path,
            "final_answer": final_answer_path,
            "qa_result": qa_result_path,
        }
        elapsed = round(time.perf_counter() - started_at, 3)
        finalized_result = validated_result.model_copy(
            update={
                "artifact_paths": artifact_paths,
                "time_elapsed": elapsed,
            }
        )
        store.write_json("qa_result.json", finalized_result.model_dump(exclude_none=True))
        return finalized_result

    def run_from_retrieval(
        self,
        retrieval_state: RetrievalState,
        *,
        artifact_dir: Optional[str] = None,
        execution_warnings: Optional[Sequence[str]] = None,
    ) -> QAResult:
        if retrieval_state.evidence_ledger is None:
            raise ValueError("RetrievalState must contain an evidence ledger before synthesis can run.")
        return self.run(
            task_spec=retrieval_state.task_spec,
            evidence_ledger=retrieval_state.evidence_ledger,
            paper_records=retrieval_state.paper_records,
            review_summaries=retrieval_state.evidence_ledger.review_summaries,
            retrieval_diagnostics=retrieval_state.retrieval_diagnostics,
            execution_warnings=execution_warnings,
            artifact_dir=artifact_dir or retrieval_state.artifact_dir,
        )

    __call__ = run
