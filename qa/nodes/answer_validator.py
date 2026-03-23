from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Set

from qa.synthesis_state import AnswerSectionOutput, QAResult, SynthesisInputPack


LOW_CONFIDENCE_STRONG_LANGUAGE = (
    re.compile(r"\bconsistently supports\b", re.I),
    re.compile(r"\bclearly (?:shows|demonstrates|establishes)\b", re.I),
    re.compile(r"\bdefinitive(?:ly)?\b", re.I),
    re.compile(r"\bproves?\b", re.I),
    re.compile(r"\bfirmly establish(?:es|ed)\b", re.I),
)


class AnswerValidationError(RuntimeError):
    def __init__(
        self,
        *,
        stage: str,
        reason: str,
        input_pack: SynthesisInputPack,
        debug_payload: Dict[str, Any],
    ) -> None:
        super().__init__(reason)
        self.stage = str(stage or "").strip() or "unknown"
        self.reason = str(reason or "").strip() or "answer validation failed"
        self.question = str(input_pack.question or "")
        self.question_type = str(input_pack.task_spec.question_type or "")
        self.debug_payload = dict(debug_payload or {})

    def to_payload(self) -> Dict[str, Any]:
        return {
            "error": "answer_validation_failed",
            "stage": self.stage,
            "reason": self.reason,
            "question": self.question,
            "question_type": self.question_type,
        }


class AnswerValidator:
    def __init__(self) -> None:
        self.last_run_debug: Dict[str, Any] = {}

    def run(
        self,
        *,
        input_pack: SynthesisInputPack,
        draft_result: QAResult,
    ) -> QAResult:
        self.last_run_debug = {
            "input_pack": input_pack.model_dump(exclude_none=True),
            "draft_result": draft_result.model_dump(exclude_none=True),
        }
        sanitized = self._sanitize_result(input_pack=input_pack, draft_result=draft_result)
        self.last_run_debug["sanitized_result"] = sanitized.model_dump(exclude_none=True)
        structural_issues = self._collect_structural_issues(input_pack=input_pack, result=sanitized)
        if structural_issues:
            self._raise_failure(
                stage="validation",
                reason=" ; ".join(structural_issues),
                input_pack=input_pack,
                failure_type="structural_issue",
                details=structural_issues,
            )
        confidence_issues = self._collect_confidence_language_issues(input_pack=input_pack, result=sanitized)
        if confidence_issues:
            self._raise_failure(
                stage="validation",
                reason=" ; ".join(confidence_issues),
                input_pack=input_pack,
                failure_type="confidence_language_mismatch",
                details=confidence_issues,
            )
        return sanitized

    __call__ = run

    def _raise_failure(
        self,
        *,
        stage: str,
        reason: str,
        input_pack: SynthesisInputPack,
        failure_type: str,
        details: Sequence[str],
    ) -> None:
        failure_payload = {
            "error": "answer_validation_failed",
            "stage": str(stage or "").strip() or "unknown",
            "reason": str(reason or "").strip() or "answer validation failed",
            "failure_type": str(failure_type or "").strip() or "unknown",
            "details": [str(item).strip() for item in details if str(item).strip()],
        }
        self.last_run_debug["failure"] = failure_payload
        raise AnswerValidationError(
            stage=failure_payload["stage"],
            reason=failure_payload["reason"],
            input_pack=input_pack,
            debug_payload=self.last_run_debug,
        )

    def _sanitize_result(self, *, input_pack: SynthesisInputPack, draft_result: QAResult) -> QAResult:
        citation_lookup = {citation.citation_id: citation for citation in input_pack.citation_catalog}
        allowed_section_ids = {pack.section_id for pack in input_pack.section_claims}
        allowed_section_ids.update(item.section_id for item in input_pack.section_confidence)
        title_lookup = {pack.section_id: pack.title for pack in input_pack.section_claims}
        title_lookup.update({item.section_id: item.title for item in input_pack.section_confidence})
        confidence_lookup = {item.section_id: item.confidence for item in input_pack.section_confidence}
        allowed_citations_by_section = {
            pack.section_id: set(pack.core_citation_ids)
            for pack in input_pack.section_claims
        }
        limitations_citations = {
            citation_id
            for claim in input_pack.contested_claims
            for citation_id in claim.citation_ids
        }
        for item in input_pack.section_confidence:
            if item.section_id not in allowed_citations_by_section and item.title == "Limitations / Controversies":
                allowed_citations_by_section[item.section_id] = limitations_citations

        sections: List[AnswerSectionOutput] = []
        seen_ids: Set[str] = set()
        for section in draft_result.sections:
            if section.section_id not in allowed_section_ids or section.section_id in seen_ids:
                continue
            cleaned_content = str(section.content or "").strip()
            if not cleaned_content:
                continue
            seen_ids.add(section.section_id)
            allowed_citations = allowed_citations_by_section.get(section.section_id, set())
            citation_ids = [
                citation_id
                for citation_id in section.citation_ids
                if citation_id in citation_lookup and citation_id in allowed_citations
            ]
            sections.append(
                AnswerSectionOutput(
                    section_id=section.section_id,
                    title=title_lookup.get(section.section_id, section.title),
                    content=cleaned_content,
                    citation_ids=citation_ids,
                    section_confidence=confidence_lookup[section.section_id],
                )
            )

        referenced_ids: List[str] = []
        for section in sections:
            for citation_id in section.citation_ids:
                if citation_id not in referenced_ids:
                    referenced_ids.append(citation_id)
        included_section_ids = {section.section_id for section in sections}
        claim_trace = [item for item in input_pack.claim_trace if item.section_id in included_section_ids]
        final_answer = str(draft_result.final_answer or "").strip() or self._assemble_final_answer(sections)
        limitations_summary = str(draft_result.limitations_summary or "").strip()
        if not limitations_summary:
            limitations_section = next(
                (section for section in sections if section.title == "Limitations / Controversies"),
                None,
            )
            limitations_summary = limitations_section.content if limitations_section is not None else ""

        return QAResult(
            question=input_pack.question,
            language="en",
            final_answer=final_answer,
            sections=sections,
            citations=[citation_lookup[citation_id] for citation_id in referenced_ids if citation_id in citation_lookup],
            claim_trace=claim_trace,
            overall_confidence=input_pack.overall_confidence,
            section_confidence=[
                item for item in input_pack.section_confidence if item.section_id in included_section_ids
            ],
            insufficient_evidence=input_pack.insufficient_evidence,
            limitations_summary=limitations_summary,
            retrieval_diagnostics_summary=input_pack.retrieval_diagnostics_summary,
            execution_warnings=list(input_pack.execution_warnings),
            artifact_paths=draft_result.artifact_paths,
            time_elapsed=draft_result.time_elapsed,
        )

    def _collect_structural_issues(self, *, input_pack: SynthesisInputPack, result: QAResult) -> List[str]:
        issues: List[str] = []
        if not result.sections:
            issues.append("Validated answer did not contain any sections.")
        primary_section_id = next((pack.section_id for pack in input_pack.section_claims), None)
        if primary_section_id is not None and primary_section_id not in {section.section_id for section in result.sections}:
            issues.append(f"Validated answer is missing required primary section '{primary_section_id}'.")

        limitations_required = bool(
            input_pack.contested_claims
            or input_pack.insufficient_evidence
            or input_pack.overall_confidence.level == "low"
            or input_pack.retrieval_diagnostics_summary
        )
        if limitations_required and not any(section.title == "Limitations / Controversies" for section in result.sections):
            issues.append("Validated answer is missing the required Limitations / Controversies section.")

        allowed_citations_by_section = {
            pack.section_id: set(pack.core_citation_ids)
            for pack in input_pack.section_claims
        }
        limitations_citations = {
            citation_id
            for claim in input_pack.contested_claims
            for citation_id in claim.citation_ids
        }
        for item in input_pack.section_confidence:
            if item.title == "Limitations / Controversies":
                allowed_citations_by_section[item.section_id] = limitations_citations

        for section in result.sections:
            allowed = allowed_citations_by_section.get(section.section_id, set())
            if allowed and not section.citation_ids and section.section_id != primary_section_id:
                issues.append(f"Section '{section.section_id}' lost all allowed citations during validation.")
            if any(citation_id not in allowed for citation_id in section.citation_ids):
                issues.append(f"Section '{section.section_id}' cites IDs outside its allowed citation set.")
        return issues

    def _collect_confidence_language_issues(self, *, input_pack: SynthesisInputPack, result: QAResult) -> List[str]:
        issues: List[str] = []
        texts = [result.final_answer]
        if input_pack.overall_confidence.level == "low":
            texts.extend(section.content for section in result.sections)
            if any(pattern.search(text) for text in texts for pattern in LOW_CONFIDENCE_STRONG_LANGUAGE):
                issues.append("Low-confidence answer uses overly strong language in final_answer or section text.")
            return issues

        low_sections = {
            item.section_id
            for item in input_pack.section_confidence
            if item.confidence.level == "low"
        }
        for section in result.sections:
            if section.section_id not in low_sections:
                continue
            if any(pattern.search(section.content) for pattern in LOW_CONFIDENCE_STRONG_LANGUAGE):
                issues.append(f"Low-confidence section '{section.section_id}' uses overly strong language.")
        return issues

    def _assemble_final_answer(self, sections: Sequence[AnswerSectionOutput]) -> str:
        return "\n\n".join(f"## {section.title}\n{section.content}" for section in sections).strip()
