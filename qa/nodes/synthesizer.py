from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from prompts.qa_prompts import SYNTHESIS_SYSTEM_PROMPT, build_synthesizer_user_prompt
from qa.llm_utils import invoke_llm, parse_json_object
from qa.synthesis_state import (
    AnswerSectionOutput,
    QAResult,
    SectionClaimPack,
    SynthesisInputPack,
)


class SynthesizerExecutionError(RuntimeError):
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
        self.reason = str(reason or "").strip() or "synthesizer execution failed"
        self.question = str(input_pack.question or "")
        self.question_type = str(input_pack.task_spec.question_type or "")
        self.debug_payload = dict(debug_payload or {})

    def to_payload(self) -> Dict[str, Any]:
        return {
            "error": "synthesizer_execution_failed",
            "stage": self.stage,
            "reason": self.reason,
            "question": self.question,
            "question_type": self.question_type,
        }


class SynthesizerNode:
    def __init__(self, llm: Any = None) -> None:
        self.llm = llm
        self.last_run_debug: Dict[str, Any] = {}

    def run(self, input_pack: SynthesisInputPack) -> QAResult:
        self.last_run_debug = {
            "input_pack": input_pack.model_dump(exclude_none=True),
        }
        if self.llm is None:
            self._raise_failure(
                stage="startup",
                reason="synthesizer LLM is unavailable",
                input_pack=input_pack,
            )

        candidate = self._synthesize_with_llm(input_pack)
        if candidate is None:
            self._raise_failure(
                stage="synthesis",
                reason="synthesizer returned unusable output",
                input_pack=input_pack,
            )
        self.last_run_debug["output"] = candidate.model_dump(exclude_none=True)
        return candidate

    __call__ = run

    def _raise_failure(
        self,
        *,
        stage: str,
        reason: str,
        input_pack: SynthesisInputPack,
    ) -> None:
        failure_payload = {
            "error": "synthesizer_execution_failed",
            "stage": str(stage or "").strip() or "unknown",
            "reason": str(reason or "").strip() or "synthesizer execution failed",
        }
        self.last_run_debug["failure"] = failure_payload
        self.last_run_debug.pop("output", None)
        raise SynthesizerExecutionError(
            stage=failure_payload["stage"],
            reason=failure_payload["reason"],
            input_pack=input_pack,
            debug_payload=self.last_run_debug,
        )

    def build_deterministic_result(self, input_pack: SynthesisInputPack) -> QAResult:
        sections: List[AnswerSectionOutput] = []
        citation_lookup = {citation.citation_id: citation for citation in input_pack.citation_catalog}
        section_conf_lookup = {
            item.section_id: item.confidence
            for item in input_pack.section_confidence
        }
        for section_pack in input_pack.section_claims:
            sections.append(
                AnswerSectionOutput(
                    section_id=section_pack.section_id,
                    title=section_pack.title,
                    content=self._render_section_content(section_pack),
                    citation_ids=[citation_id for citation_id in section_pack.core_citation_ids if citation_id in citation_lookup],
                    section_confidence=section_pack.section_confidence,
                )
            )

        limitations_section = self._render_limitations_section(input_pack, section_conf_lookup=section_conf_lookup)
        if limitations_section is not None:
            sections.append(limitations_section)

        referenced_citation_ids = []
        for section in sections:
            for citation_id in section.citation_ids:
                if citation_id not in referenced_citation_ids:
                    referenced_citation_ids.append(citation_id)
        citations = [citation_lookup[citation_id] for citation_id in referenced_citation_ids if citation_id in citation_lookup]

        included_section_ids = {section.section_id for section in sections}
        claim_trace = [item for item in input_pack.claim_trace if item.section_id in included_section_ids]
        final_answer = self._assemble_final_answer(sections)
        limitations_summary = limitations_section.content if limitations_section is not None else ""

        return QAResult(
            question=input_pack.question,
            language="en",
            final_answer=final_answer,
            sections=sections,
            citations=citations,
            claim_trace=claim_trace,
            overall_confidence=input_pack.overall_confidence,
            section_confidence=[
                section_conf
                for section_conf in input_pack.section_confidence
                if section_conf.section_id in included_section_ids
            ],
            insufficient_evidence=input_pack.insufficient_evidence,
            limitations_summary=limitations_summary,
            retrieval_diagnostics_summary=input_pack.retrieval_diagnostics_summary,
            execution_warnings=list(input_pack.execution_warnings),
            artifact_paths={},
            time_elapsed=0.0,
        )

    def _synthesize_with_llm(self, input_pack: SynthesisInputPack) -> Optional[QAResult]:
        messages = [
            {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
            {"role": "user", "content": build_synthesizer_user_prompt(input_pack.model_dump(exclude_none=True))},
        ]
        try:
            raw_output = invoke_llm(self.llm, messages)
        except Exception:
            return None
        self.last_run_debug["raw_output"] = raw_output
        parsed = parse_json_object(raw_output)
        if parsed is None:
            return None

        allowed_section_ids = [pack.section_id for pack in input_pack.section_claims]
        if any(item.section_id == "limitations_controversies" for item in input_pack.section_confidence):
            allowed_section_ids.append("limitations_controversies")
        for item in input_pack.section_confidence:
            if item.section_id not in allowed_section_ids:
                allowed_section_ids.append(item.section_id)

        section_conf_lookup = {item.section_id: item.confidence for item in input_pack.section_confidence}
        title_lookup = {pack.section_id: pack.title for pack in input_pack.section_claims}
        for item in input_pack.section_confidence:
            title_lookup.setdefault(item.section_id, item.title)

        sections: List[AnswerSectionOutput] = []
        raw_sections = parsed.get("sections")
        if not isinstance(raw_sections, list):
            return None
        for raw_section in raw_sections:
            if not isinstance(raw_section, dict):
                continue
            section_id = str(raw_section.get("section_id") or "").strip()
            if section_id not in allowed_section_ids:
                continue
            content = str(raw_section.get("content") or "").strip()
            if not content:
                continue
            raw_citation_ids = raw_section.get("citation_ids")
            citation_ids = []
            if isinstance(raw_citation_ids, list):
                citation_ids = [str(item).strip() for item in raw_citation_ids if str(item).strip()]
            sections.append(
                AnswerSectionOutput(
                    section_id=section_id,
                    title=title_lookup.get(section_id, section_id.replace("_", " ").title()),
                    content=content,
                    citation_ids=citation_ids,
                    section_confidence=section_conf_lookup[section_id],
                )
            )
        if not sections:
            return None

        final_answer = str(parsed.get("final_answer") or "").strip() or self._assemble_final_answer(sections)
        limitations_summary = str(parsed.get("limitations_summary") or "").strip()
        citation_lookup = {citation.citation_id: citation for citation in input_pack.citation_catalog}
        referenced_ids = []
        for section in sections:
            for citation_id in section.citation_ids:
                if citation_id in citation_lookup and citation_id not in referenced_ids:
                    referenced_ids.append(citation_id)

        included_section_ids = {section.section_id for section in sections}
        claim_trace = [item for item in input_pack.claim_trace if item.section_id in included_section_ids]
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
            artifact_paths={},
            time_elapsed=0.0,
        )

    def _render_section_content(self, section_pack: SectionClaimPack) -> str:
        confidence_level = section_pack.section_confidence.level
        if not section_pack.claim_summaries:
            return "Available accepted evidence is limited and does not support a firm conclusion for this section."

        opener = self._section_opener(section_pack=section_pack, confidence_level=confidence_level)
        primary_claim = section_pack.claim_summaries[0]
        remaining_claims = section_pack.claim_summaries[1:]
        sentences = [f"{opener} {primary_claim}"]
        if remaining_claims:
            if "condition" in section_pack.section_id.lower():
                sentences.append(
                    "The accepted record further narrows the answer through the following condition-bound observations: "
                    + self._join_claims(remaining_claims)
                )
            elif "mechanism" in section_pack.section_id.lower():
                sentences.append(
                    "Additional accepted mechanistic support indicates that " + self._join_claims(remaining_claims, lowercase_first=True)
                )
            else:
                sentences.append(
                    "Additional accepted evidence indicates that " + self._join_claims(remaining_claims, lowercase_first=True)
                )
        return " ".join(sentences)

    def _render_limitations_section(
        self,
        input_pack: SynthesisInputPack,
        *,
        section_conf_lookup: Dict[str, Any],
    ) -> Optional[AnswerSectionOutput]:
        if (
            not input_pack.contested_claims
            and not input_pack.insufficient_evidence
            and input_pack.overall_confidence.level != "low"
            and not input_pack.retrieval_diagnostics_summary
        ):
            return None
        section_id, section_confidence = self._limitations_slot(
            input_pack=input_pack,
            section_conf_lookup=section_conf_lookup,
        )

        sentences: List[str] = []
        citation_ids: List[str] = []
        if input_pack.contested_claims:
            sentences.append(
                "Some ledger claims remain contested and are therefore excluded from the main conclusion."
            )
            for item in input_pack.contested_claims[:3]:
                sentences.append(f"{item.claim_summary} This point remains contested because {self._lowercase_first(item.rationale)}")
                for citation_id in item.citation_ids:
                    if citation_id not in citation_ids:
                        citation_ids.append(citation_id)
        if input_pack.insufficient_evidence:
            sentences.append(
                "Accepted evidence is too sparse to support a complete answer across every requested section, so the output remains partial and conservative."
            )
        if input_pack.retrieval_diagnostics_summary:
            sentences.append(input_pack.retrieval_diagnostics_summary)
        if input_pack.overall_confidence.level == "low" and not input_pack.insufficient_evidence:
            sentences.append(
                "Overall confidence remains low, so the wording intentionally avoids a firm synthesis beyond the accepted record."
            )
        content = " ".join(sentences) if sentences else "Residual uncertainty remains and is preserved explicitly."
        return AnswerSectionOutput(
            section_id=section_id,
            title="Limitations / Controversies",
            content=content,
            citation_ids=citation_ids,
            section_confidence=section_confidence,
        )

    def _limitations_slot(
        self,
        *,
        input_pack: SynthesisInputPack,
        section_conf_lookup: Dict[str, Any],
    ) -> tuple[str, Any]:
        non_main_section_ids = {
            item.section_id
            for item in input_pack.section_confidence
            if item.section_id not in {pack.section_id for pack in input_pack.section_claims}
        }
        for item in input_pack.section_confidence:
            lower_title = item.title.lower()
            lower_id = item.section_id.lower()
            if (
                item.title == "Limitations / Controversies"
                or "limit" in lower_title
                or "controvers" in lower_title
                or "caveat" in lower_title
                or "open question" in lower_title
                or "limit" in lower_id
                or "controvers" in lower_id
                or "caveat" in lower_id
                or item.section_id in non_main_section_ids
            ):
                return item.section_id, item.confidence
        return "limitations_controversies", section_conf_lookup.get("limitations_controversies") or input_pack.overall_confidence

    def _section_opener(self, *, section_pack: SectionClaimPack, confidence_level: str) -> str:
        lower_id = section_pack.section_id.lower()
        if confidence_level == "high":
            base = "The accepted literature consistently supports"
        elif confidence_level == "medium":
            base = "Current accepted evidence suggests"
        else:
            base = "Available accepted evidence is limited but indicates"

        if "direct" in lower_id or "summary" in lower_id or "conclusion" in lower_id or "recent" in lower_id:
            return base + " that"
        if "condition" in lower_id:
            return "The answer depends on the following accepted condition-bound findings:"
        if "mechanism" in lower_id or "path" in lower_id:
            return base + " that"
        return base + " that"

    def _join_claims(self, claims: Sequence[str], *, lowercase_first: bool = False) -> str:
        normalized = []
        for claim in claims:
            text = claim.strip()
            if lowercase_first:
                text = self._lowercase_first(text)
            normalized.append(text)
        return " ".join(normalized)

    def _lowercase_first(self, text: str) -> str:
        cleaned = text.strip()
        if not cleaned:
            return cleaned
        return cleaned[:1].lower() + cleaned[1:]

    def _assemble_final_answer(self, sections: Sequence[AnswerSectionOutput]) -> str:
        blocks = [f"## {section.title}\n{section.content}" for section in sections]
        return "\n\n".join(blocks).strip()
