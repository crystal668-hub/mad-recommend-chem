from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Sequence

from qa.artifacts import QAArtifactStore
from qa.nodes.answer_validator import AnswerValidationError, AnswerValidator
from qa.nodes.synthesis_pack_builder import SynthesisPackBuilder
from qa.nodes.synthesizer import SynthesizerExecutionError, SynthesizerNode
from qa.retrieval_state import EvidenceLedger, PaperRecord, RetrievalDiagnosticRecord, RetrievalState, ReviewSummary
from qa.state import TaskSpec
from qa.synthesis_state import QAResult


logger = logging.getLogger("MAD.qa.synthesis")


class VerifiedSynthesisPipeline:
    def __init__(
        self,
        *,
        pack_builder: Optional[SynthesisPackBuilder] = None,
        synthesizer: Optional[SynthesizerNode] = None,
        answer_validator: Optional[AnswerValidator] = None,
        progress_log_every: int = 10,
    ) -> None:
        self.pack_builder = pack_builder or SynthesisPackBuilder()
        self.synthesizer = synthesizer or SynthesizerNode()
        self.answer_validator = answer_validator or AnswerValidator()
        self.progress_log_every = max(1, int(progress_log_every))

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
        logger.info(
            "qa_synthesis_start accepted_claims=%s contested_claims=%s papers=%s",
            sum(1 for claim in evidence_ledger.claims if claim.status == "accepted"),
            sum(1 for claim in evidence_ledger.claims if claim.status == "contested"),
            len(paper_records),
        )

        input_pack = self.pack_builder.run(
            task_spec=task_spec,
            evidence_ledger=evidence_ledger,
            review_summaries=review_summaries,
            paper_records=paper_records,
            retrieval_diagnostics=retrieval_diagnostics,
            execution_warnings=execution_warnings,
        )
        synthesis_pack_path = store.write_json("synthesis_input_pack.json", input_pack.model_dump(exclude_none=True))
        logger.info(
            "qa_synthesis_pack_complete sections=%s citations=%s warnings=%s",
            len(input_pack.section_claims),
            len(input_pack.citation_catalog),
            len(input_pack.execution_warnings),
        )

        try:
            draft_result = self.synthesizer.run(input_pack)
        except SynthesizerExecutionError as exc:
            self._write_failure_artifacts(
                store=store,
                node_name="synthesizer",
                input_payload=input_pack.model_dump(exclude_none=True),
                error=exc,
            )
            raise
        logger.info("qa_synthesis_draft_complete sections=%s", len(draft_result.sections))
        try:
            validated_result = self.answer_validator.run(
                input_pack=input_pack,
                draft_result=draft_result,
            )
        except AnswerValidationError as exc:
            self._write_failure_artifacts(
                store=store,
                node_name="answer_validator",
                input_payload={
                    "input_pack": input_pack.model_dump(exclude_none=True),
                    "draft_result": draft_result.model_dump(exclude_none=True),
                },
                error=exc,
            )
            raise

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
        logger.info(
            "qa_synthesis_complete sections=%s citations=%s elapsed=%.3f",
            len(finalized_result.sections),
            len(finalized_result.citations),
            elapsed,
        )
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
        merged_warnings: list[str] = []
        seen = set()
        for warning in [*(retrieval_state.execution_warnings or []), *(execution_warnings or [])]:
            text = str(warning or "").strip()
            key = text.lower()
            if not text or key in seen:
                continue
            seen.add(key)
            merged_warnings.append(text)
        return self.run(
            task_spec=retrieval_state.task_spec,
            evidence_ledger=retrieval_state.evidence_ledger,
            paper_records=retrieval_state.paper_records,
            review_summaries=retrieval_state.evidence_ledger.review_summaries,
            retrieval_diagnostics=retrieval_state.retrieval_diagnostics,
            execution_warnings=merged_warnings,
            artifact_dir=artifact_dir or retrieval_state.artifact_dir,
        )

    __call__ = run

    def _write_failure_artifacts(
        self,
        *,
        store: QAArtifactStore,
        node_name: str,
        input_payload: Dict[str, Any],
        error: Any,
    ) -> None:
        debug_payload = dict(getattr(error, "debug_payload", {}) or {})
        payload = error.to_payload() if hasattr(error, "to_payload") else {
            "error": f"{node_name}_execution_failed",
            "stage": "unknown",
            "reason": str(error),
        }
        store.write_json(f"{node_name}/failure.json", payload)
        store.write_json(
            f"{node_name}/agent_run.json",
            {
                "agent": node_name,
                "input": input_payload,
                "error": payload,
                "debug": debug_payload,
            },
        )
