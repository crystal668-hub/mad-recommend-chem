from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

from qa.retrieval_state import PaperRecord, Section, SectionIndex, SectionTextView
from qa.retrieval_utils import looks_like_garbled_text, looks_like_placeholder_text
from qa.state import SourceSpan, TaskSpec


FULLTEXT_SECTION_POLICY = {
    "fact": (),
    "frontier": (),
    "causal": ("results", "discussion", "methods"),
    "mechanism": ("results", "discussion", "methods"),
    "comparison": ("results", "discussion", "methods"),
}
FULLTEXT_FALLBACK_SECTIONS = ("results", "discussion", "methods")
UNKNOWN_FULLTEXT_MIN_CHARS = 800
UNKNOWN_FULLTEXT_MIN_ALPHA_TOKENS = 80


class EvidenceExtractorHandoff:
    def get_primary_text(self, paper_record: PaperRecord) -> str:
        return paper_record.abstract or ""

    def should_read_fulltext(
        self,
        task_spec: TaskSpec,
        *,
        evidence_is_weak: bool = False,
        missing_conditions: bool = False,
    ) -> bool:
        if task_spec.question_type in {"causal", "mechanism", "comparison"}:
            return True
        return evidence_is_weak or missing_conditions

    def preferred_section_types(
        self,
        task_spec: TaskSpec,
        *,
        evidence_is_weak: bool = False,
        missing_conditions: bool = False,
    ) -> Sequence[str]:
        section_types = FULLTEXT_SECTION_POLICY.get(task_spec.question_type, ())
        if section_types:
            return section_types
        if evidence_is_weak or missing_conditions:
            return FULLTEXT_FALLBACK_SECTIONS
        return ()

    def read_section_text(
        self,
        paper_record: PaperRecord,
        section_index: SectionIndex,
        section_id: str,
    ) -> Optional[SectionTextView]:
        if not paper_record.fulltext_artifact_path:
            return None
        section = next((item for item in section_index.sections if item.section_id == section_id), None)
        if section is None:
            return None
        fulltext = Path(paper_record.fulltext_artifact_path).read_text(encoding="utf-8")
        text = fulltext[section.fulltext_char_start : section.fulltext_char_end]
        return SectionTextView(
            paper_id=paper_record.paper_id,
            section_id=section.section_id,
            section_type=section.section_type,
            heading=section.heading,
            text=text,
            page_start=section.page_start,
            page_end=section.page_end,
            fulltext_char_start=section.fulltext_char_start,
            fulltext_char_end=section.fulltext_char_end,
        )

    def read_preferred_sections(
        self,
        paper_record: PaperRecord,
        section_index: SectionIndex,
        task_spec: TaskSpec,
        *,
        evidence_is_weak: bool = False,
        missing_conditions: bool = False,
    ) -> List[SectionTextView]:
        if not self.should_read_fulltext(
            task_spec,
            evidence_is_weak=evidence_is_weak,
            missing_conditions=missing_conditions,
        ):
            return []
        allowed_types = set(
            self.preferred_section_types(
                task_spec,
                evidence_is_weak=evidence_is_weak,
                missing_conditions=missing_conditions,
            )
        )
        if not allowed_types:
            return []
        section_views: List[SectionTextView] = []
        for section in section_index.sections:
            if section.section_type not in allowed_types:
                continue
            view = self.read_section_text(paper_record=paper_record, section_index=section_index, section_id=section.section_id)
            if view is not None:
                section_views.append(view)
        if section_views:
            return section_views
        for section in section_index.sections:
            if section.section_type != "unknown":
                continue
            view = self.read_section_text(paper_record=paper_record, section_index=section_index, section_id=section.section_id)
            if view is None or not self._allow_unknown_fulltext_view(view.text):
                continue
            section_views.append(view)
        return section_views

    def fulltext_span_to_section_span(self, section: Section, fulltext_span: SourceSpan) -> SourceSpan:
        start = max(section.fulltext_char_start, fulltext_span.start)
        end = min(section.fulltext_char_end, fulltext_span.end)
        return SourceSpan(
            start=start - section.fulltext_char_start,
            end=end - section.fulltext_char_start,
        )

    def _allow_unknown_fulltext_view(self, text: str) -> bool:
        cleaned = str(text or "")
        if len(cleaned) < UNKNOWN_FULLTEXT_MIN_CHARS:
            return False
        if looks_like_placeholder_text(cleaned) or looks_like_garbled_text(cleaned):
            return False
        alpha_tokens = sum(1 for token in cleaned.split() if any(char.isalpha() for char in token))
        return alpha_tokens >= UNKNOWN_FULLTEXT_MIN_ALPHA_TOKENS
