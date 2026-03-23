from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pymupdf as fitz

from qa.artifacts import QAArtifactStore
from qa.retrieval_utils import (
    looks_like_garbled_text,
    looks_like_placeholder_text,
    normalize_text,
    printable_text_ratio,
)


logger = logging.getLogger("MAD.qa.pdf_extraction")

SECTION_HEADING_PATTERN = re.compile(
    r"(?mi)^(?:\d+(?:\.\d+)*\s+)?(abstract|introduction|background|materials?\s+and\s+methods|methods?|experimental|results?|discussion|conclusions?|limitations?)\s*$"
)
SECTION_TYPE_MAP = {
    "abstract": "abstract",
    "introduction": "introduction",
    "background": "introduction",
    "materials and methods": "methods",
    "material and methods": "methods",
    "methods": "methods",
    "method": "methods",
    "experimental": "methods",
    "result": "results",
    "results": "results",
    "discussion": "discussion",
    "conclusion": "conclusion",
    "conclusions": "conclusion",
    "limitation": "limitations",
    "limitations": "limitations",
}
DOCLING_SKIP_LABELS = {"page_header", "page_footer"}
SYMBOL_RUN_PATTERN = re.compile(r"(?:[^A-Za-z0-9\s]){8,}")
UNKNOWN_SECTION_TITLE = "Body"
UNKNOWN_SECTION_TYPE = "unknown"
REPEATED_LINE_RATIO_THRESHOLD = 0.35
MAX_UNKNOWN_SNIPPETS = 3


@dataclass
class PDFExtractionConfig:
    enabled: bool = True
    primary_backend: str = "pymupdf"
    secondary_backend: str = "none"
    min_total_chars: int = 800
    min_chars_per_text_page: int = 80
    min_text_page_ratio: float = 0.5
    min_printable_ratio: float = 0.95
    snippet_target_chars: int = 1000
    snippet_overlap_chars: int = 120
    preserve_page_blocks: bool = True

    @classmethod
    def from_dict(cls, payload: Optional[dict[str, Any]]) -> "PDFExtractionConfig":
        raw = dict(payload or {})
        config = cls(
            enabled=bool(raw.get("enabled", True)),
            primary_backend=str(raw.get("primary_backend") or "pymupdf").strip().lower(),
            secondary_backend=str(raw.get("secondary_backend") or "none").strip().lower(),
            min_total_chars=max(0, int(raw.get("min_total_chars", 800) or 800)),
            min_chars_per_text_page=max(1, int(raw.get("min_chars_per_text_page", 80) or 80)),
            min_text_page_ratio=float(raw.get("min_text_page_ratio", 0.5) or 0.5),
            min_printable_ratio=float(raw.get("min_printable_ratio", 0.95) or 0.95),
            snippet_target_chars=max(200, int(raw.get("snippet_target_chars", 1000) or 1000)),
            snippet_overlap_chars=max(0, int(raw.get("snippet_overlap_chars", 120) or 120)),
            preserve_page_blocks=bool(raw.get("preserve_page_blocks", True)),
        )
        if config.snippet_overlap_chars >= config.snippet_target_chars:
            config.snippet_overlap_chars = max(0, config.snippet_target_chars // 4)
        config.min_text_page_ratio = min(1.0, max(0.0, config.min_text_page_ratio))
        config.min_printable_ratio = min(1.0, max(0.0, config.min_printable_ratio))
        return config


@dataclass
class ExtractedBlock:
    page_no: int
    text: str


@dataclass
class ExtractedSection:
    heading: str
    section_type: str
    text: str
    page_start: int
    page_end: int
    fulltext_char_start: int = 0
    fulltext_char_end: int = 0


@dataclass
class ExtractionAttempt:
    extractor: str
    succeeded: bool
    fulltext: str = ""
    page_texts: list[str] = field(default_factory=list)
    blocks: list[ExtractedBlock] = field(default_factory=list)
    sections: list[ExtractedSection] = field(default_factory=list)
    page_count: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    usable: bool = False
    failure_reason: Optional[str] = None
    ocr_applied: bool = False


@dataclass
class PDFExtractionResult:
    fulltext_status: str
    source_artifact_path: str
    fulltext_artifact_path: Optional[str] = None
    sections_artifact_path: Optional[str] = None
    snippets_artifact_path: Optional[str] = None
    extraction_report_path: Optional[str] = None
    sections: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    extractor: Optional[str] = None
    ocr_applied: bool = False
    report: dict[str, Any] = field(default_factory=dict)


class PDFExtractionPipeline:
    def __init__(
        self,
        *,
        config: Optional[PDFExtractionConfig | dict[str, Any]] = None,
    ) -> None:
        if isinstance(config, PDFExtractionConfig):
            self.config = config
        else:
            self.config = PDFExtractionConfig.from_dict(config)
        self._docling_converter: Any = None

    def process(
        self,
        *,
        paper_id: str,
        pdf_bytes: bytes,
        artifact_store: QAArtifactStore,
    ) -> PDFExtractionResult:
        source_artifact_path = artifact_store.write_bytes(f"fulltext/{paper_id}.pdf", pdf_bytes)
        warnings: list[str] = []
        attempts: list[ExtractionAttempt] = []

        if not self.config.enabled:
            report = {
                "paper_id": paper_id,
                "status": "binary_only",
                "selected_extractor": None,
                "ocr_applied": False,
                "attempts": [],
                "warnings": [],
            }
            report_path = artifact_store.write_json(f"fulltext/{paper_id}.extraction_report.json", report)
            return PDFExtractionResult(
                fulltext_status="binary_only",
                source_artifact_path=source_artifact_path,
                fulltext_artifact_path=source_artifact_path,
                extraction_report_path=report_path,
                report=report,
            )

        if not self._is_true_pdf(pdf_bytes):
            warning = (
                f"{paper_id}: downloaded content was labeled as PDF but the file header did not match %PDF-; "
                "fulltext extraction skipped."
            )
            warnings.append(warning)
            report = {
                "paper_id": paper_id,
                "status": "fulltext_unusable",
                "selected_extractor": None,
                "ocr_applied": False,
                "attempts": [],
                "warnings": warnings,
                "failure_reason": "invalid_pdf_header",
            }
            report_path = artifact_store.write_json(f"fulltext/{paper_id}.extraction_report.json", report)
            logger.warning("fulltext_unusable paper_id=%s reason=invalid_pdf_header", paper_id)
            return PDFExtractionResult(
                fulltext_status="fulltext_unusable",
                source_artifact_path=source_artifact_path,
                fulltext_artifact_path=source_artifact_path,
                extraction_report_path=report_path,
                warnings=warnings,
                report=report,
            )

        logger.info("pdf_detected paper_id=%s", paper_id)
        primary_attempt = self._extract_with_pymupdf(pdf_bytes=pdf_bytes)
        attempts.append(primary_attempt)
        if primary_attempt.usable:
            return self._finalize_success(
                paper_id=paper_id,
                artifact_store=artifact_store,
                source_artifact_path=source_artifact_path,
                attempt=primary_attempt,
                attempts=attempts,
                warnings=warnings,
            )

        secondary_attempt: Optional[ExtractionAttempt] = None
        if self.config.secondary_backend == "docling":
            logger.info("docling_fallback_triggered paper_id=%s", paper_id)
            secondary_attempt = self._extract_with_docling(pdf_path=Path(source_artifact_path))
            attempts.append(secondary_attempt)
            if secondary_attempt.usable:
                return self._finalize_success(
                    paper_id=paper_id,
                    artifact_store=artifact_store,
                    source_artifact_path=source_artifact_path,
                    attempt=secondary_attempt,
                    attempts=attempts,
                    warnings=warnings,
                )

        if primary_attempt.failure_reason:
            warnings.append(f"{paper_id}: PyMuPDF extraction failed: {primary_attempt.failure_reason}")
        elif primary_attempt.metrics.get("reasons"):
            warnings.append(
                f"{paper_id}: PyMuPDF extraction was rejected by quality gates: "
                + ", ".join(primary_attempt.metrics["reasons"])
                + "."
            )
        if secondary_attempt is not None:
            if secondary_attempt.failure_reason:
                warnings.append(f"{paper_id}: Docling fallback failed: {secondary_attempt.failure_reason}")
            elif secondary_attempt.metrics.get("reasons"):
                warnings.append(
                    f"{paper_id}: Docling output was rejected by quality gates: "
                    + ", ".join(secondary_attempt.metrics["reasons"])
                    + "."
                )

        report = {
            "paper_id": paper_id,
            "status": "fulltext_unusable",
            "selected_extractor": None,
            "ocr_applied": False,
            "attempts": [self._attempt_payload(item) for item in attempts],
            "warnings": warnings,
        }
        report_path = artifact_store.write_json(f"fulltext/{paper_id}.extraction_report.json", report)
        logger.warning("fulltext_unusable paper_id=%s", paper_id)
        return PDFExtractionResult(
            fulltext_status="fulltext_unusable",
            source_artifact_path=source_artifact_path,
            fulltext_artifact_path=source_artifact_path,
            extraction_report_path=report_path,
            warnings=warnings,
            report=report,
        )

    def _finalize_success(
        self,
        *,
        paper_id: str,
        artifact_store: QAArtifactStore,
        source_artifact_path: str,
        attempt: ExtractionAttempt,
        attempts: list[ExtractionAttempt],
        warnings: list[str],
    ) -> PDFExtractionResult:
        sections = [self._section_payload(section) for section in attempt.sections]
        snippets = self._build_snippets(paper_id=paper_id, attempt=attempt)
        fulltext_path = artifact_store.write_text(f"fulltext/{paper_id}.fulltext.txt", attempt.fulltext)
        sections_path = artifact_store.write_json(f"fulltext/{paper_id}.sections.json", sections)
        snippets_path = artifact_store.write_text(
            f"fulltext/{paper_id}.snippets.jsonl",
            self._jsonl_payload(snippets),
        )
        report = {
            "paper_id": paper_id,
            "status": "fulltext_indexed",
            "selected_extractor": attempt.extractor,
            "ocr_applied": attempt.ocr_applied,
            "attempts": [self._attempt_payload(item) for item in attempts],
            "warnings": warnings,
            "snippet_count": len(snippets),
            "section_count": len(sections),
        }
        report_path = artifact_store.write_json(f"fulltext/{paper_id}.extraction_report.json", report)
        logger.info(
            "fulltext_usable paper_id=%s extractor=%s snippets=%s",
            paper_id,
            attempt.extractor,
            len(snippets),
        )
        return PDFExtractionResult(
            fulltext_status="fulltext_indexed",
            source_artifact_path=source_artifact_path,
            fulltext_artifact_path=fulltext_path,
            sections_artifact_path=sections_path,
            snippets_artifact_path=snippets_path,
            extraction_report_path=report_path,
            sections=sections,
            warnings=warnings,
            extractor=attempt.extractor,
            ocr_applied=attempt.ocr_applied,
            report=report,
        )

    def _extract_with_pymupdf(self, *, pdf_bytes: bytes, ocr_applied: bool = False) -> ExtractionAttempt:
        try:
            document = fitz.open(stream=pdf_bytes, filetype="pdf")
        except Exception as exc:
            logger.warning("pymupdf_extraction_failed error=%s", exc)
            return ExtractionAttempt(
                extractor="pymupdf",
                succeeded=False,
                failure_reason=str(exc),
                ocr_applied=ocr_applied,
            )

        page_texts: list[str] = []
        blocks: list[ExtractedBlock] = []
        try:
            for page_index in range(document.page_count):
                page = document.load_page(page_index)
                page_blocks: list[str] = []
                for block in page.get_text("blocks", sort=True):
                    if len(block) < 5:
                        continue
                    text = self._normalize_extracted_text(block[4])
                    if not text:
                        continue
                    page_blocks.append(text)
                    if self.config.preserve_page_blocks:
                        blocks.append(ExtractedBlock(page_no=page_index + 1, text=text))
                page_text = "\n\n".join(page_blocks)
                if not page_text:
                    page_text = self._normalize_extracted_text(page.get_text("text", sort=True))
                page_texts.append(page_text)
        finally:
            document.close()

        fulltext, page_spans = self._join_page_texts(page_texts)
        sections = self._sections_from_fulltext(fulltext=fulltext, page_spans=page_spans)
        metrics = self._evaluate_quality(fulltext=fulltext, page_texts=page_texts)
        usable = not metrics["reasons"]
        logger.info(
            "pymupdf_extraction_succeeded pages=%s usable=%s total_chars=%s",
            len(page_texts),
            usable,
            metrics["total_chars"],
        )
        return ExtractionAttempt(
            extractor="pymupdf",
            succeeded=True,
            fulltext=fulltext,
            page_texts=page_texts,
            blocks=blocks,
            sections=sections,
            page_count=len(page_texts),
            metrics=metrics,
            usable=usable,
            ocr_applied=ocr_applied,
        )

    def _extract_with_docling(self, *, pdf_path: Path) -> ExtractionAttempt:
        try:
            converter = self._get_docling_converter()
            conversion = converter.convert(pdf_path, raises_on_error=False)
            document = getattr(conversion, "document", None)
            if document is None:
                status = getattr(conversion, "status", "missing_document")
                return ExtractionAttempt(
                    extractor="docling",
                    succeeded=False,
                    failure_reason=str(status),
                )
        except Exception as exc:
            return ExtractionAttempt(
                extractor="docling",
                succeeded=False,
                failure_reason=str(exc),
            )

        page_buckets: dict[int, list[str]] = defaultdict(list)
        raw_sections: list[dict[str, Any]] = []
        current_section: Optional[dict[str, Any]] = None

        for item, _level in document.iterate_items(with_groups=False, traverse_pictures=False):
            label = str(getattr(item, "label", "")).lower()
            if label in DOCLING_SKIP_LABELS:
                continue
            page_no = self._item_page_no(item)
            text = self._normalize_extracted_text(getattr(item, "text", ""))
            if not text and hasattr(item, "export_to_markdown"):
                try:
                    text = self._normalize_extracted_text(item.export_to_markdown(document))
                except Exception:
                    text = ""
            if not text:
                continue
            if label == "section_header":
                current_section = {
                    "heading": text,
                    "section_type": self._section_type_for_heading(text),
                    "page_start": page_no or 1,
                    "page_end": page_no or 1,
                    "parts": [],
                }
                raw_sections.append(current_section)
                continue
            if current_section is None:
                current_section = {
                    "heading": UNKNOWN_SECTION_TITLE,
                    "section_type": UNKNOWN_SECTION_TYPE,
                    "page_start": page_no or 1,
                    "page_end": page_no or 1,
                    "parts": [],
                }
                raw_sections.append(current_section)
            current_section["parts"].append(text)
            if page_no:
                current_section["page_end"] = max(int(current_section["page_end"]), int(page_no))
                page_buckets[page_no].append(text)

        page_count = len(getattr(document, "pages", {}) or {}) or max(page_buckets.keys(), default=0)
        page_texts = [
            "\n\n".join(page_buckets.get(page_no, []))
            for page_no in range(1, page_count + 1)
        ]
        sections = self._sections_from_docling_entries(raw_sections)
        fulltext = self._fulltext_from_sections(sections)
        if not fulltext:
            try:
                fulltext = self._normalize_extracted_text(document.export_to_markdown(page_break_placeholder="\n"))
            except Exception:
                fulltext = ""
            if fulltext:
                fulltext, page_spans = self._join_page_texts(page_texts or [fulltext])
                sections = self._sections_from_fulltext(fulltext=fulltext, page_spans=page_spans)

        metrics = self._evaluate_quality(fulltext=fulltext, page_texts=page_texts or ([fulltext] if fulltext else []))
        usable = not metrics["reasons"]
        return ExtractionAttempt(
            extractor="docling",
            succeeded=True,
            fulltext=fulltext,
            page_texts=page_texts,
            sections=sections,
            page_count=page_count,
            metrics=metrics,
            usable=usable,
        )

    def _evaluate_quality(self, *, fulltext: str, page_texts: list[str]) -> dict[str, Any]:
        normalized_fulltext = self._normalize_extracted_text(fulltext)
        page_count = len(page_texts)
        text_lengths = [len(normalize_text(page_text)) for page_text in page_texts]
        total_chars = len(normalize_text(normalized_fulltext))
        text_pages = sum(1 for length in text_lengths if length >= self.config.min_chars_per_text_page)
        text_page_ratio = (text_pages / float(page_count)) if page_count else 0.0
        printable_ratio = printable_text_ratio(normalized_fulltext)
        repeated_line_ratio = self._repeated_line_ratio(normalized_fulltext)
        garbled = (
            looks_like_garbled_text(normalized_fulltext)
            or "\ufffd" in normalized_fulltext
            or bool(SYMBOL_RUN_PATTERN.search(normalized_fulltext))
        )
        reasons: list[str] = []
        if not normalized_fulltext:
            reasons.append("empty_text")
        if looks_like_placeholder_text(normalized_fulltext):
            reasons.append("placeholder_text")
        if page_count > 2 and total_chars < self.config.min_total_chars:
            reasons.append("total_chars_below_threshold")
        if page_count and text_page_ratio < self.config.min_text_page_ratio:
            reasons.append("text_page_ratio_below_threshold")
        if printable_ratio < self.config.min_printable_ratio:
            reasons.append("printable_ratio_below_threshold")
        if garbled:
            reasons.append("garbled_text_detected")
        if repeated_line_ratio > REPEATED_LINE_RATIO_THRESHOLD:
            reasons.append("repeated_line_noise")

        base_score = 1.0
        if page_count > 2 and total_chars < self.config.min_total_chars:
            base_score -= 0.25
        if page_count and text_page_ratio < self.config.min_text_page_ratio:
            base_score -= 0.25
        if printable_ratio < self.config.min_printable_ratio:
            base_score -= 0.2
        if garbled:
            base_score -= 0.35
        if repeated_line_ratio > REPEATED_LINE_RATIO_THRESHOLD:
            base_score -= 0.15
        return {
            "total_chars": total_chars,
            "page_count": page_count,
            "text_page_ratio": round(text_page_ratio, 3),
            "printable_ratio": round(printable_ratio, 3),
            "repeated_line_ratio": round(repeated_line_ratio, 3),
            "quality_score": round(max(0.0, min(base_score, 1.0)), 2),
            "reasons": reasons,
        }

    def _build_snippets(self, *, paper_id: str, attempt: ExtractionAttempt) -> list[dict[str, Any]]:
        if len(attempt.sections) == 1 and attempt.sections[0].section_type == UNKNOWN_SECTION_TYPE and attempt.blocks:
            snippets = self._snippets_from_blocks(paper_id=paper_id, attempt=attempt)
        else:
            snippets = self._snippets_from_sections(paper_id=paper_id, attempt=attempt)
        if len(attempt.sections) == 1 and attempt.sections[0].section_type == UNKNOWN_SECTION_TYPE:
            snippets = snippets[:MAX_UNKNOWN_SNIPPETS]
        return snippets

    def _snippets_from_sections(self, *, paper_id: str, attempt: ExtractionAttempt) -> list[dict[str, Any]]:
        snippets: list[dict[str, Any]] = []
        for section in attempt.sections:
            for text in self._iter_windows(section.text):
                payload = self._snippet_payload(
                    paper_id=paper_id,
                    snippet_index=len(snippets),
                    section_title=section.heading,
                    page_start=section.page_start,
                    page_end=section.page_end,
                    text=text,
                    extractor=attempt.extractor,
                    ocr_applied=attempt.ocr_applied,
                    base_quality=float(attempt.metrics.get("quality_score") or 0.0),
                )
                if payload is not None:
                    snippets.append(payload)
        return snippets

    def _snippets_from_blocks(self, *, paper_id: str, attempt: ExtractionAttempt) -> list[dict[str, Any]]:
        snippets: list[dict[str, Any]] = []
        blocks = [block for block in attempt.blocks if normalize_text(block.text)]
        if not blocks:
            return snippets

        start_index = 0
        while start_index < len(blocks):
            end_index = start_index
            char_count = 0
            while end_index < len(blocks) and char_count < self.config.snippet_target_chars:
                char_count += len(blocks[end_index].text) + 2
                end_index += 1
            selected = blocks[start_index:end_index]
            text = "\n\n".join(block.text for block in selected)
            payload = self._snippet_payload(
                paper_id=paper_id,
                snippet_index=len(snippets),
                section_title=UNKNOWN_SECTION_TITLE,
                page_start=selected[0].page_no,
                page_end=selected[-1].page_no,
                text=text,
                extractor=attempt.extractor,
                ocr_applied=attempt.ocr_applied,
                base_quality=float(attempt.metrics.get("quality_score") or 0.0),
            )
            if payload is not None:
                snippets.append(payload)

            if end_index >= len(blocks):
                break

            overlap_chars = 0
            next_start = end_index - 1
            while next_start > start_index and overlap_chars < self.config.snippet_overlap_chars:
                overlap_chars += len(blocks[next_start].text) + 2
                next_start -= 1
            start_index = max(start_index + 1, next_start + 1)
        return snippets

    def _snippet_payload(
        self,
        *,
        paper_id: str,
        snippet_index: int,
        section_title: str,
        page_start: int,
        page_end: int,
        text: str,
        extractor: str,
        ocr_applied: bool,
        base_quality: float,
    ) -> Optional[dict[str, Any]]:
        cleaned_text = self._normalize_extracted_text(text)
        if len(cleaned_text) < 40:
            return None
        if looks_like_placeholder_text(cleaned_text) or looks_like_garbled_text(cleaned_text):
            return None
        normalized = normalize_text(cleaned_text).lower()
        symbol_count = sum(1 for char in cleaned_text if not char.isalnum() and not char.isspace())
        symbol_ratio = symbol_count / float(max(1, len(cleaned_text)))
        quality_score = max(0.0, min(base_quality, 1.0))
        if len(normalized) < 120:
            quality_score -= 0.15
        if printable_text_ratio(cleaned_text) < 0.97:
            quality_score -= 0.15
        if symbol_ratio > 0.25:
            quality_score -= 0.15
        quality_score = round(max(0.0, min(quality_score, 1.0)), 2)
        if quality_score <= 0.2:
            return None
        return {
            "snippet_id": f"{paper_id}:snippet:{snippet_index + 1}",
            "paper_id": paper_id,
            "page_start": page_start,
            "page_end": page_end,
            "section_title": section_title,
            "text": cleaned_text,
            "normalized_text": normalized,
            "extractor": extractor,
            "ocr_applied": ocr_applied,
            "quality_score": quality_score,
        }

    def _sections_from_fulltext(
        self,
        *,
        fulltext: str,
        page_spans: list[tuple[int, int, int]],
    ) -> list[ExtractedSection]:
        if not fulltext:
            return []
        matches = list(SECTION_HEADING_PATTERN.finditer(fulltext))
        if not matches:
            return [
                ExtractedSection(
                    heading=UNKNOWN_SECTION_TITLE,
                    section_type=UNKNOWN_SECTION_TYPE,
                    text=fulltext,
                    page_start=page_spans[0][2] if page_spans else 1,
                    page_end=page_spans[-1][2] if page_spans else 1,
                    fulltext_char_start=0,
                    fulltext_char_end=len(fulltext),
                )
            ]

        sections: list[ExtractedSection] = []
        for index, match in enumerate(matches):
            heading = match.group(1).strip()
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(fulltext)
            text = fulltext[start:end].strip()
            if not text:
                continue
            page_start = self._page_for_char(page_spans=page_spans, char_index=start)
            page_end = self._page_for_char(page_spans=page_spans, char_index=max(start, end - 1))
            sections.append(
                ExtractedSection(
                    heading=heading,
                    section_type=self._section_type_for_heading(heading),
                    text=text,
                    page_start=page_start,
                    page_end=page_end,
                    fulltext_char_start=start,
                    fulltext_char_end=end,
                )
            )
        if sections:
            return sections
        return [
            ExtractedSection(
                heading=UNKNOWN_SECTION_TITLE,
                section_type=UNKNOWN_SECTION_TYPE,
                text=fulltext,
                page_start=page_spans[0][2] if page_spans else 1,
                page_end=page_spans[-1][2] if page_spans else 1,
                fulltext_char_start=0,
                fulltext_char_end=len(fulltext),
            )
        ]

    def _sections_from_docling_entries(self, raw_sections: list[dict[str, Any]]) -> list[ExtractedSection]:
        sections: list[ExtractedSection] = []
        for raw_section in raw_sections:
            text = self._normalize_extracted_text("\n\n".join(raw_section.get("parts") or []))
            if not text:
                continue
            sections.append(
                ExtractedSection(
                    heading=normalize_text(raw_section.get("heading")) or UNKNOWN_SECTION_TITLE,
                    section_type=str(raw_section.get("section_type") or UNKNOWN_SECTION_TYPE),
                    text=text,
                    page_start=max(1, int(raw_section.get("page_start") or 1)),
                    page_end=max(1, int(raw_section.get("page_end") or raw_section.get("page_start") or 1)),
                )
            )
        fulltext = self._fulltext_from_sections(sections)
        if not fulltext:
            return []
        return self._apply_section_offsets(sections)

    def _fulltext_from_sections(self, sections: list[ExtractedSection]) -> str:
        if not sections:
            return ""
        parts: list[str] = []
        for index, section in enumerate(self._apply_section_offsets(sections)):
            if index:
                parts.append("\n\n")
            heading = normalize_text(section.heading) or UNKNOWN_SECTION_TITLE
            parts.append(heading)
            parts.append("\n")
            parts.append(section.text)
        return "".join(parts)

    def _apply_section_offsets(self, sections: list[ExtractedSection]) -> list[ExtractedSection]:
        cursor = 0
        normalized_sections: list[ExtractedSection] = []
        for index, section in enumerate(sections):
            heading = normalize_text(section.heading) or UNKNOWN_SECTION_TITLE
            text = self._normalize_extracted_text(section.text)
            if not text:
                continue
            if index:
                cursor += 2
            cursor += len(heading) + 1
            start = cursor
            cursor += len(text)
            end = cursor
            normalized_sections.append(
                ExtractedSection(
                    heading=heading,
                    section_type=section.section_type,
                    text=text,
                    page_start=section.page_start,
                    page_end=section.page_end,
                    fulltext_char_start=start,
                    fulltext_char_end=end,
                )
            )
        return normalized_sections

    def _attempt_payload(self, attempt: ExtractionAttempt) -> dict[str, Any]:
        return {
            "extractor": attempt.extractor,
            "succeeded": attempt.succeeded,
            "usable": attempt.usable,
            "ocr_applied": attempt.ocr_applied,
            "page_count": attempt.page_count,
            "metrics": attempt.metrics,
            "failure_reason": attempt.failure_reason,
        }

    def _section_payload(self, section: ExtractedSection) -> dict[str, Any]:
        return {
            "section_id": f"sec_{section.section_type}_{section.fulltext_char_start}",
            "section_type": section.section_type,
            "heading": section.heading,
            "page_start": section.page_start,
            "page_end": section.page_end,
            "fulltext_char_start": section.fulltext_char_start,
            "fulltext_char_end": section.fulltext_char_end,
            "text": section.text,
        }

    def _jsonl_payload(self, records: list[dict[str, Any]]) -> str:
        if not records:
            return ""
        return "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n"

    def _join_page_texts(self, page_texts: list[str]) -> tuple[str, list[tuple[int, int, int]]]:
        parts: list[str] = []
        page_spans: list[tuple[int, int, int]] = []
        cursor = 0
        for page_index, page_text in enumerate(page_texts):
            if page_index:
                parts.append("\n\n")
                cursor += 2
            normalized_page = self._normalize_extracted_text(page_text)
            start = cursor
            parts.append(normalized_page)
            cursor += len(normalized_page)
            page_spans.append((start, cursor, page_index + 1))
        return "".join(parts).strip(), page_spans

    def _page_for_char(self, *, page_spans: list[tuple[int, int, int]], char_index: int) -> int:
        for start, end, page_no in page_spans:
            if start <= char_index <= max(start, end):
                return page_no
        if page_spans:
            return page_spans[-1][2]
        return 1

    def _iter_windows(self, text: str) -> list[str]:
        cleaned = self._normalize_extracted_text(text)
        if not cleaned:
            return []
        if len(cleaned) <= self.config.snippet_target_chars:
            return [cleaned]
        windows: list[str] = []
        start = 0
        step = max(1, self.config.snippet_target_chars - self.config.snippet_overlap_chars)
        while start < len(cleaned):
            end = min(len(cleaned), start + self.config.snippet_target_chars)
            snippet = cleaned[start:end].strip()
            if snippet:
                windows.append(snippet)
            if end >= len(cleaned):
                break
            start += step
        return windows

    def _repeated_line_ratio(self, text: str) -> float:
        lines = [normalize_text(line).lower() for line in text.splitlines() if len(normalize_text(line)) >= 15]
        if len(lines) < 4:
            return 0.0
        counts = Counter(lines)
        repeated = sum(count for count in counts.values() if count > 1)
        return repeated / float(len(lines))

    def _normalize_extracted_text(self, text: Optional[str]) -> str:
        raw = str(text or "").replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
        lines = []
        previous_blank = False
        for raw_line in raw.split("\n"):
            line = re.sub(r"[ \t]+", " ", raw_line).strip()
            if not line:
                if previous_blank:
                    continue
                previous_blank = True
                lines.append("")
                continue
            previous_blank = False
            lines.append(line)
        return "\n".join(lines).strip()

    def _is_true_pdf(self, content: bytes) -> bool:
        return bytes(content or b"").lstrip().startswith(b"%PDF-")

    def _section_type_for_heading(self, heading: str) -> str:
        return SECTION_TYPE_MAP.get(normalize_text(heading).lower(), UNKNOWN_SECTION_TYPE)

    def _item_page_no(self, item: Any) -> Optional[int]:
        providers = getattr(item, "prov", None) or []
        for provider in providers:
            page_no = getattr(provider, "page_no", None)
            if page_no:
                return int(page_no)
        return None

    def _get_docling_converter(self) -> Any:
        if self._docling_converter is not None:
            return self._docling_converter
        from docling.document_converter import DocumentConverter

        self._docling_converter = DocumentConverter()
        return self._docling_converter
