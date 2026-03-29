from typing import List, Sequence, Tuple

from qa.react_reviewed.common import (
    AcceptanceDecision,
    AnswerSectionOutput,
    AnswerSubmission,
    QAResult,
    ReviewCompletionStatus,
    ReviewItem,
    SectionConfidenceRecord,
    TaskSpec,
    _assemble_final_answer,
    _build_submission_trace,
    _compact_text,
    _merge_unique_text,
    _result_confidence,
    _submission_to_citation_records,
)

class SubmissionSynthesizerAgent:
    def __init__(self, *, expose_candidate_submission_when_rejected: bool = False) -> None:
        self.expose_candidate_submission_when_rejected = bool(expose_candidate_submission_when_rejected)

    def run(
        self,
        *,
        task_spec: TaskSpec,
        submission: AnswerSubmission,
        review_items: Sequence[ReviewItem],
        review_completion_status: ReviewCompletionStatus,
        acceptance_decision: AcceptanceDecision,
        retrieval_diagnostics_summary: str,
        execution_warnings: Sequence[str],
    ) -> QAResult:
        if not acceptance_decision.accepted:
            if self.expose_candidate_submission_when_rejected:
                return self._build_rejected_candidate_result(
                    task_spec=task_spec,
                    submission=submission,
                    review_items=review_items,
                    review_completion_status=review_completion_status,
                    acceptance_decision=acceptance_decision,
                    retrieval_diagnostics_summary=retrieval_diagnostics_summary,
                    execution_warnings=execution_warnings,
                )
            return self._build_rejected_result(
                task_spec=task_spec,
                submission=submission,
                review_items=review_items,
                review_completion_status=review_completion_status,
                acceptance_decision=acceptance_decision,
                retrieval_diagnostics_summary=retrieval_diagnostics_summary,
                execution_warnings=execution_warnings,
            )
        return self._build_accepted_result(
            task_spec=task_spec,
            submission=submission,
            review_items=review_items,
            review_completion_status=review_completion_status,
            retrieval_diagnostics_summary=retrieval_diagnostics_summary,
            execution_warnings=execution_warnings,
        )

    def _build_accepted_result(
        self,
        *,
        task_spec: TaskSpec,
        submission: AnswerSubmission,
        review_items: Sequence[ReviewItem],
        review_completion_status: ReviewCompletionStatus,
        retrieval_diagnostics_summary: str,
        execution_warnings: Sequence[str],
    ) -> QAResult:
        section_outputs: List[AnswerSectionOutput] = []
        section_confidence: List[SectionConfidenceRecord] = []
        section_lookup = {section.section_id: section for section in submission.sections}
        for answer_section in task_spec.answer_sections:
            section = section_lookup.get(answer_section.section_id)
            if section is None:
                continue
            section_conf = _result_confidence(
                section.section_confidence.score,
                section.section_confidence.rationale,
            )
            section_outputs.append(
                AnswerSectionOutput(
                    section_id=section.section_id,
                    title=section.title or answer_section.title,
                    content=_compact_text(section.content) or "Evidence remains insufficient for a stronger section-level answer.",
                    citation_ids=list(section.citation_ids),
                    section_confidence=section_conf,
                )
            )
            section_confidence.append(
                SectionConfidenceRecord(
                    section_id=section.section_id,
                    title=section.title or answer_section.title,
                    confidence=section_conf,
                )
            )
        unresolved_blocking = [item for item in review_items if item.status == "open" and item.severity == "blocking"]
        limitations_parts = list(submission.limitations)
        if unresolved_blocking:
            limitations_parts.append(
                "Unresolved blocking review items remain: "
                + "; ".join(item.required_action for item in unresolved_blocking[:3])
            )
        if review_completion_status == "incomplete":
            limitations_parts.append("Reviewer completion was incomplete, so the answer remains non-accepted.")
        limitations_summary = " ".join(part for part in limitations_parts if _compact_text(part)).strip()
        overall_score = submission.overall_confidence.score
        if unresolved_blocking:
            overall_score = min(overall_score, 0.45)
        if review_completion_status == "incomplete":
            overall_score = min(overall_score, 0.35)
        return QAResult(
            question=submission.question,
            language="en",
            workflow_mode="react_reviewed",
            acceptance_status="accepted",
            final_answer=_assemble_final_answer(section_outputs),
            sections=section_outputs,
            citations=_submission_to_citation_records(submission),
            claim_trace=[],
            submission_trace=_build_submission_trace(submission),
            review_completion_status=review_completion_status,
            overall_confidence=_result_confidence(
                overall_score,
                "Overall confidence is derived from the accepted submission and outstanding review issues.",
            ),
            section_confidence=section_confidence,
            insufficient_evidence=bool(unresolved_blocking) or not submission.citations,
            limitations_summary=limitations_summary,
            retrieval_diagnostics_summary=str(retrieval_diagnostics_summary or "").strip(),
            execution_warnings=list(_merge_unique_text([], execution_warnings)),
            artifact_paths={},
            time_elapsed=0.0,
        )

    def _build_submission_section_outputs(
        self,
        *,
        task_spec: TaskSpec,
        submission: AnswerSubmission,
    ) -> Tuple[List[AnswerSectionOutput], List[SectionConfidenceRecord]]:
        section_outputs: List[AnswerSectionOutput] = []
        section_confidence: List[SectionConfidenceRecord] = []
        section_lookup = {section.section_id: section for section in submission.sections}
        for answer_section in task_spec.answer_sections:
            section = section_lookup.get(answer_section.section_id)
            if section is None:
                continue
            section_conf = _result_confidence(
                section.section_confidence.score,
                section.section_confidence.rationale,
            )
            section_outputs.append(
                AnswerSectionOutput(
                    section_id=section.section_id,
                    title=section.title or answer_section.title,
                    content=_compact_text(section.content) or "Evidence remains insufficient for a stronger section-level answer.",
                    citation_ids=list(section.citation_ids),
                    section_confidence=section_conf,
                )
            )
            section_confidence.append(
                SectionConfidenceRecord(
                    section_id=section.section_id,
                    title=section.title or answer_section.title,
                    confidence=section_conf,
                )
            )
        return section_outputs, section_confidence

    def _build_rejected_result(
        self,
        *,
        task_spec: TaskSpec,
        submission: AnswerSubmission,
        review_items: Sequence[ReviewItem],
        review_completion_status: ReviewCompletionStatus,
        acceptance_decision: AcceptanceDecision,
        retrieval_diagnostics_summary: str,
        execution_warnings: Sequence[str],
    ) -> QAResult:
        reason_lines = list(acceptance_decision.blocker_messages)
        if not reason_lines:
            reason_lines = ["Submission was rejected because the current answer packet is not acceptance-ready."]
        section_outputs: List[AnswerSectionOutput] = []
        section_confidence: List[SectionConfidenceRecord] = []
        rejection_text = "Submission rejected: " + " ".join(reason_lines[:2])
        for answer_section in task_spec.answer_sections:
            content = rejection_text if answer_section.required else "Section withheld because the submission was rejected."
            section_outputs.append(
                AnswerSectionOutput(
                    section_id=answer_section.section_id,
                    title=answer_section.title,
                    content=content,
                    citation_ids=[],
                    section_confidence=_result_confidence(
                        0.2,
                        "Section confidence is low because acceptance gates rejected the submission.",
                    ),
                )
            )
            section_confidence.append(
                SectionConfidenceRecord(
                    section_id=answer_section.section_id,
                    title=answer_section.title,
                    confidence=_result_confidence(
                        0.2,
                        "Section confidence is low because acceptance gates rejected the submission.",
                    ),
                )
            )
        limitations_parts = [
            "Acceptance rejected the candidate submission: " + "; ".join(reason_lines),
        ]
        if review_completion_status == "incomplete":
            limitations_parts.append("Reviewer completion was incomplete.")
        return QAResult(
            question=submission.question,
            language="en",
            workflow_mode="react_reviewed",
            acceptance_status="rejected",
            final_answer="## Acceptance Rejected\n" + "\n".join(f"- {line}" for line in reason_lines),
            sections=section_outputs,
            citations=[],
            claim_trace=[],
            submission_trace=_build_submission_trace(submission),
            review_completion_status=review_completion_status,
            overall_confidence=_result_confidence(
                0.2,
                "Overall confidence is low because acceptance gates rejected the submission.",
            ),
            section_confidence=section_confidence,
            insufficient_evidence=True,
            limitations_summary=" ".join(part for part in limitations_parts if _compact_text(part)).strip(),
            retrieval_diagnostics_summary=str(retrieval_diagnostics_summary or "").strip(),
            execution_warnings=list(_merge_unique_text([], execution_warnings)),
            artifact_paths={},
            time_elapsed=0.0,
        )

    def _build_rejected_candidate_result(
        self,
        *,
        task_spec: TaskSpec,
        submission: AnswerSubmission,
        review_items: Sequence[ReviewItem],
        review_completion_status: ReviewCompletionStatus,
        acceptance_decision: AcceptanceDecision,
        retrieval_diagnostics_summary: str,
        execution_warnings: Sequence[str],
    ) -> QAResult:
        section_outputs, section_confidence = self._build_submission_section_outputs(
            task_spec=task_spec,
            submission=submission,
        )
        blocker_summary = "; ".join(acceptance_decision.blocker_messages) or "Submission remains not acceptance-ready."
        limitations_parts = list(submission.limitations)
        limitations_parts.append(
            "Candidate submission was surfaced despite rejection; outstanding blockers: " + blocker_summary
        )
        unresolved_blocking = [item for item in review_items if item.status == "open" and item.severity == "blocking"]
        if unresolved_blocking:
            limitations_parts.append(
                "Unresolved blocking review items remain: "
                + "; ".join(item.required_action for item in unresolved_blocking[:3])
            )
        if review_completion_status == "incomplete":
            limitations_parts.append("Reviewer completion was incomplete.")
        overall_score = submission.overall_confidence.score
        if unresolved_blocking:
            overall_score = min(overall_score, 0.45)
        if review_completion_status == "incomplete":
            overall_score = min(overall_score, 0.35)
        return QAResult(
            question=submission.question,
            language="en",
            workflow_mode="react_reviewed",
            acceptance_status="rejected",
            final_answer=_assemble_final_answer(section_outputs),
            sections=section_outputs,
            citations=_submission_to_citation_records(submission),
            claim_trace=[],
            submission_trace=_build_submission_trace(submission),
            review_completion_status=review_completion_status,
            overall_confidence=_result_confidence(
                overall_score,
                "Overall confidence is derived from the surfaced candidate submission and outstanding review issues.",
            ),
            section_confidence=section_confidence,
            insufficient_evidence=bool(unresolved_blocking) or not submission.citations,
            limitations_summary=" ".join(part for part in limitations_parts if _compact_text(part)).strip(),
            retrieval_diagnostics_summary=str(retrieval_diagnostics_summary or "").strip(),
            execution_warnings=list(_merge_unique_text([], execution_warnings)),
            artifact_paths={},
            time_elapsed=0.0,
        )
