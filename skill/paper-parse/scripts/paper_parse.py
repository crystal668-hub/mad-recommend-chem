from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import pymupdf as fitz


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
UNKNOWN_SECTION_TITLE = "Body"
UNKNOWN_SECTION_TYPE = "unknown"


def _compact_text(value: Any) -> str:
    return " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split()).strip()


def _normalize_text(value: Any) -> str:
    text = str(value or "").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _printable_text_ratio(text: str) -> float:
    cleaned = str(text or "")
    if not cleaned:
        return 0.0
    printable = sum(1 for char in cleaned if char.isprintable() or char in "\n\t")
    return printable / max(1, len(cleaned))


def _repeated_line_ratio(text: str) -> float:
    lines = [_compact_text(line) for line in str(text or "").splitlines() if _compact_text(line)]
    if not lines:
        return 0.0
    counts = Counter(lines)
    repeated = sum(count for count in counts.values() if count > 1)
    return repeated / max(1, len(lines))


@dataclass
class ParserConfig:
    enabled: bool = True
    primary_backend: str = "pymupdf"
    secondary_backend: str = "docling"
    min_total_chars: int = 800
    min_chars_per_text_page: int = 80
    min_text_page_ratio: float = 0.5
    min_printable_ratio: float = 0.95
    snippet_target_chars: int = 1000
    snippet_overlap_chars: int = 120
    preserve_page_blocks: bool = True

    @classmethod
    def from_dict(cls, payload: Optional[dict[str, Any]]) -> "ParserConfig":
        raw = dict(payload or {})
        config = cls(
            enabled=bool(raw.get("enabled", True)),
            primary_backend=str(raw.get("primary_backend") or "pymupdf").strip().lower(),
            secondary_backend=str(raw.get("secondary_backend") or "docling").strip().lower(),
            min_total_chars=max(0, int(raw.get("min_total_chars", 800) or 800)),
            min_chars_per_text_page=max(1, int(raw.get("min_chars_per_text_page", 80) or 80)),
            min_text_page_ratio=min(1.0, max(0.0, float(raw.get("min_text_page_ratio", 0.5) or 0.5))),
            min_printable_ratio=min(1.0, max(0.0, float(raw.get("min_printable_ratio", 0.95) or 0.95))),
            snippet_target_chars=max(200, int(raw.get("snippet_target_chars", 1000) or 1000)),
            snippet_overlap_chars=max(0, int(raw.get("snippet_overlap_chars", 120) or 120)),
            preserve_page_blocks=bool(raw.get("preserve_page_blocks", True)),
        )
        if config.snippet_overlap_chars >= config.snippet_target_chars:
            config.snippet_overlap_chars = max(0, config.snippet_target_chars // 4)
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


class PaperParseEngine:
    def __init__(self, *, config: Optional[ParserConfig | dict[str, Any]] = None) -> None:
        if isinstance(config, ParserConfig):
            self.config = config
        else:
            self.config = ParserConfig.from_dict(config)
        self._docling_converter: Any = None

    def process_document(self, *, input_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
        path = Path(input_path)
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        document_id = path.stem or "document"
        if path.suffix.lower() == ".pdf":
            return self.process_pdf_bytes(document_id=document_id, pdf_bytes=path.read_bytes(), output_dir=destination)
        text = path.read_text(encoding="utf-8")
        return self.process_text(document_id=document_id, text=text, output_dir=destination, source_extension=path.suffix.lower())

    def process_text(
        self,
        *,
        document_id: str,
        text: str,
        output_dir: str | Path,
        source_extension: str = ".txt",
    ) -> dict[str, Any]:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        source_path = destination / f"{document_id}{source_extension or '.txt'}"
        source_path.write_text(text, encoding="utf-8")
        normalized = _normalize_text(text)
        sections = self._apply_section_offsets(self._sections_from_fulltext(fulltext=normalized, page_spans=[]))
        snippets = self._build_snippets(normalized)
        fulltext_path = destination / f"{document_id}.fulltext.txt"
        sections_path = destination / f"{document_id}.sections.json"
        snippets_path = destination / f"{document_id}.snippets.json"
        report_path = destination / f"{document_id}.extraction_report.json"
        fulltext_path.write_text(normalized, encoding="utf-8")
        sections_payload = [self._section_payload(section) for section in sections]
        snippets_payload = snippets
        sections_path.write_text(json.dumps(sections_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        snippets_path.write_text(json.dumps(snippets_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        report = {
            "document_id": document_id,
            "status": "fulltext_indexed",
            "selected_extractor": "text",
            "attempts": [],
            "warnings": [],
        }
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return {
            "document_id": document_id,
            "fulltext_status": "fulltext_indexed",
            "source_artifact_path": str(source_path),
            "fulltext_artifact_path": str(fulltext_path),
            "sections_artifact_path": str(sections_path),
            "snippets_artifact_path": str(snippets_path),
            "extraction_report_path": str(report_path),
            "sections": sections_payload,
            "warnings": [],
            "extractor": "text",
            "ocr_applied": False,
            "report": report,
        }

    def process_pdf_bytes(
        self,
        *,
        document_id: str,
        pdf_bytes: bytes,
        output_dir: str | Path,
    ) -> dict[str, Any]:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        source_path = destination / f"{document_id}.pdf"
        source_path.write_bytes(pdf_bytes)
        warnings: list[str] = []
        attempts: list[ExtractionAttempt] = []

        if not self.config.enabled:
            report = {
                "document_id": document_id,
                "status": "binary_only",
                "selected_extractor": None,
                "attempts": [],
                "warnings": [],
            }
            report_path = destination / f"{document_id}.extraction_report.json"
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            return {
                "document_id": document_id,
                "fulltext_status": "binary_only",
                "source_artifact_path": str(source_path),
                "fulltext_artifact_path": str(source_path),
                "sections_artifact_path": None,
                "snippets_artifact_path": None,
                "extraction_report_path": str(report_path),
                "sections": [],
                "warnings": [],
                "extractor": None,
                "ocr_applied": False,
                "report": report,
            }

        if not self._is_true_pdf(pdf_bytes):
            warning = f"{document_id}: invalid PDF header; parsing skipped."
            warnings.append(warning)
            report = {
                "document_id": document_id,
                "status": "fulltext_unusable",
                "selected_extractor": None,
                "attempts": [],
                "warnings": warnings,
                "failure_reason": "invalid_pdf_header",
            }
            report_path = destination / f"{document_id}.extraction_report.json"
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            return {
                "document_id": document_id,
                "fulltext_status": "fulltext_unusable",
                "source_artifact_path": str(source_path),
                "fulltext_artifact_path": str(source_path),
                "sections_artifact_path": None,
                "snippets_artifact_path": None,
                "extraction_report_path": str(report_path),
                "sections": [],
                "warnings": warnings,
                "extractor": None,
                "ocr_applied": False,
                "report": report,
            }

        primary_attempt = self._extract_with_pymupdf(pdf_bytes)
        attempts.append(primary_attempt)
        if primary_attempt.usable:
            return self._finalize_success(
                document_id=document_id,
                source_path=source_path,
                destination=destination,
                attempt=primary_attempt,
                attempts=attempts,
                warnings=warnings,
            )

        if self.config.secondary_backend == "docling":
            secondary_attempt = self._extract_with_docling(source_path)
            attempts.append(secondary_attempt)
            if secondary_attempt.usable:
                return self._finalize_success(
                    document_id=document_id,
                    source_path=source_path,
                    destination=destination,
                    attempt=secondary_attempt,
                    attempts=attempts,
                    warnings=warnings,
                )

        if primary_attempt.failure_reason:
            warnings.append(f"{document_id}: PyMuPDF extraction failed: {primary_attempt.failure_reason}")
        elif primary_attempt.metrics.get("reasons"):
            warnings.append(
                f"{document_id}: PyMuPDF extraction was rejected by quality gates: "
                + ", ".join(primary_attempt.metrics.get("reasons", []))
                + "."
            )
        if len(attempts) > 1:
            secondary_attempt = attempts[-1]
            if secondary_attempt.failure_reason:
                warnings.append(f"{document_id}: Docling fallback failed: {secondary_attempt.failure_reason}")
            elif secondary_attempt.metrics.get("reasons"):
                warnings.append(
                    f"{document_id}: Docling output was rejected by quality gates: "
                    + ", ".join(secondary_attempt.metrics.get("reasons", []))
                    + "."
                )
        report = {
            "document_id": document_id,
            "status": "fulltext_unusable",
            "selected_extractor": None,
            "attempts": [self._attempt_payload(item) for item in attempts],
            "warnings": warnings,
        }
        report_path = destination / f"{document_id}.extraction_report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return {
            "document_id": document_id,
            "fulltext_status": "fulltext_unusable",
            "source_artifact_path": str(source_path),
            "fulltext_artifact_path": str(source_path),
            "sections_artifact_path": None,
            "snippets_artifact_path": None,
            "extraction_report_path": str(report_path),
            "sections": [],
            "warnings": warnings,
            "extractor": None,
            "ocr_applied": False,
            "report": report,
        }

    def _finalize_success(
        self,
        *,
        document_id: str,
        source_path: Path,
        destination: Path,
        attempt: ExtractionAttempt,
        attempts: list[ExtractionAttempt],
        warnings: list[str],
    ) -> dict[str, Any]:
        sections = list(attempt.sections or [])
        if not sections:
            sections = self._apply_section_offsets(self._sections_from_fulltext(fulltext=attempt.fulltext, page_spans=[]))
        fulltext = attempt.fulltext or self._fulltext_from_sections(sections)
        fulltext_path = destination / f"{document_id}.fulltext.txt"
        sections_path = destination / f"{document_id}.sections.json"
        snippets_path = destination / f"{document_id}.snippets.json"
        report_path = destination / f"{document_id}.extraction_report.json"
        fulltext_path.write_text(fulltext, encoding="utf-8")
        sections_payload = [self._section_payload(section) for section in sections]
        snippets_payload = self._build_snippets(fulltext)
        sections_path.write_text(json.dumps(sections_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        snippets_path.write_text(json.dumps(snippets_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        report = {
            "document_id": document_id,
            "status": "fulltext_indexed",
            "selected_extractor": attempt.extractor,
            "attempts": [self._attempt_payload(item) for item in attempts],
            "warnings": warnings,
        }
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return {
            "document_id": document_id,
            "fulltext_status": "fulltext_indexed",
            "source_artifact_path": str(source_path),
            "fulltext_artifact_path": str(fulltext_path),
            "sections_artifact_path": str(sections_path),
            "snippets_artifact_path": str(snippets_path),
            "extraction_report_path": str(report_path),
            "sections": sections_payload,
            "warnings": warnings,
            "extractor": attempt.extractor,
            "ocr_applied": bool(attempt.ocr_applied),
            "report": report,
        }

    def _extract_with_pymupdf(self, pdf_bytes: bytes) -> ExtractionAttempt:
        try:
            document = fitz.open(stream=pdf_bytes, filetype="pdf")
        except Exception as exc:
            return ExtractionAttempt(extractor="pymupdf", succeeded=False, failure_reason=str(exc))
        page_texts: list[str] = []
        blocks: list[ExtractedBlock] = []
        try:
            for page_index in range(document.page_count):
                page = document.load_page(page_index)
                page_blocks: list[str] = []
                for block in page.get_text("blocks", sort=True):
                    if len(block) < 5:
                        continue
                    text = _normalize_text(block[4])
                    if not text:
                        continue
                    page_blocks.append(text)
                    if self.config.preserve_page_blocks:
                        blocks.append(ExtractedBlock(page_no=page_index + 1, text=text))
                page_text = "\n\n".join(page_blocks)
                if not page_text:
                    page_text = _normalize_text(page.get_text("text", sort=True))
                page_texts.append(page_text)
        finally:
            document.close()
        fulltext, page_spans = self._join_page_texts(page_texts)
        sections = self._apply_section_offsets(self._sections_from_fulltext(fulltext=fulltext, page_spans=page_spans))
        metrics = self._evaluate_quality(fulltext=fulltext, page_texts=page_texts)
        return ExtractionAttempt(
            extractor="pymupdf",
            succeeded=True,
            fulltext=fulltext,
            page_texts=page_texts,
            blocks=blocks,
            sections=sections,
            page_count=len(page_texts),
            metrics=metrics,
            usable=not metrics["reasons"],
        )

    def _extract_with_docling(self, pdf_path: Path) -> ExtractionAttempt:
        try:
            converter = self._get_docling_converter()
        except Exception as exc:
            return ExtractionAttempt(extractor="docling", succeeded=False, failure_reason=str(exc))
        try:
            conversion = converter.convert(pdf_path, raises_on_error=False)
            document = getattr(conversion, "document", None)
            if document is None:
                status = getattr(conversion, "status", "missing_document")
                return ExtractionAttempt(extractor="docling", succeeded=False, failure_reason=str(status))
        except Exception as exc:
            return ExtractionAttempt(extractor="docling", succeeded=False, failure_reason=str(exc))

        page_buckets: dict[int, list[str]] = defaultdict(list)
        for item, _level in document.iterate_items(with_groups=False, traverse_pictures=False):
            label = str(getattr(item, "label", "")).lower()
            if label in DOCLING_SKIP_LABELS:
                continue
            provenance = list(getattr(item, "prov", []) or [])
            page_no = int(getattr(provenance[0], "page_no", 1) or 1) if provenance else 1
            text = _normalize_text(getattr(item, "text", "") or "")
            if text:
                page_buckets[page_no].append(text)

        page_texts = ["\n\n".join(page_buckets[index]) for index in sorted(page_buckets)]
        fulltext, page_spans = self._join_page_texts(page_texts)
        sections = self._apply_section_offsets(self._sections_from_fulltext(fulltext=fulltext, page_spans=page_spans))
        metrics = self._evaluate_quality(fulltext=fulltext, page_texts=page_texts)
        return ExtractionAttempt(
            extractor="docling",
            succeeded=True,
            fulltext=fulltext,
            page_texts=page_texts,
            sections=sections,
            page_count=len(page_texts),
            metrics=metrics,
            usable=not metrics["reasons"],
        )

    def _get_docling_converter(self) -> Any:
        if self._docling_converter is None:
            from docling.document_converter import DocumentConverter

            self._docling_converter = DocumentConverter()
        return self._docling_converter

    def _join_page_texts(self, page_texts: list[str]) -> tuple[str, list[dict[str, int]]]:
        fulltext_parts: list[str] = []
        page_spans: list[dict[str, int]] = []
        cursor = 0
        for index, page_text in enumerate(page_texts):
            normalized = _normalize_text(page_text)
            if fulltext_parts:
                fulltext_parts.append("\n\n")
                cursor += 2
            start = cursor
            fulltext_parts.append(normalized)
            cursor += len(normalized)
            page_spans.append({"page_no": index + 1, "start": start, "end": cursor})
        return "".join(fulltext_parts), page_spans

    def _sections_from_fulltext(self, *, fulltext: str, page_spans: list[dict[str, int]]) -> list[ExtractedSection]:
        normalized = _normalize_text(fulltext)
        if not normalized:
            return []
        matches = list(SECTION_HEADING_PATTERN.finditer(normalized))
        if not matches:
            return [
                ExtractedSection(
                    heading=UNKNOWN_SECTION_TITLE,
                    section_type=UNKNOWN_SECTION_TYPE,
                    text=normalized,
                    page_start=self._page_for_offset(0, page_spans),
                    page_end=self._page_for_offset(len(normalized), page_spans),
                )
            ]
        sections: list[ExtractedSection] = []
        if matches[0].start() > 0:
            prefix = normalized[: matches[0].start()].strip()
            if prefix:
                sections.append(
                    ExtractedSection(
                        heading=UNKNOWN_SECTION_TITLE,
                        section_type=UNKNOWN_SECTION_TYPE,
                        text=prefix,
                        page_start=self._page_for_offset(0, page_spans),
                        page_end=self._page_for_offset(matches[0].start(), page_spans),
                    )
                )
        for index, match in enumerate(matches):
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(normalized)
            heading = _compact_text(match.group(1)).title()
            text = normalized[start:end].strip()
            if not text:
                continue
            key = _compact_text(match.group(1)).lower()
            sections.append(
                ExtractedSection(
                    heading=heading,
                    section_type=SECTION_TYPE_MAP.get(key, UNKNOWN_SECTION_TYPE),
                    text=text,
                    page_start=self._page_for_offset(match.start(), page_spans),
                    page_end=self._page_for_offset(end, page_spans),
                )
            )
        return sections

    def _page_for_offset(self, offset: int, page_spans: list[dict[str, int]]) -> int:
        if not page_spans:
            return 1
        for page_span in page_spans:
            if int(page_span["start"]) <= offset <= int(page_span["end"]):
                return int(page_span["page_no"])
        return int(page_spans[-1]["page_no"])

    def _apply_section_offsets(self, sections: list[ExtractedSection]) -> list[ExtractedSection]:
        cursor = 0
        applied: list[ExtractedSection] = []
        for section in sections:
            text = _normalize_text(section.text)
            start = cursor
            end = start + len(text)
            applied.append(
                ExtractedSection(
                    heading=section.heading,
                    section_type=section.section_type,
                    text=text,
                    page_start=max(1, int(section.page_start or 1)),
                    page_end=max(int(section.page_start or 1), int(section.page_end or section.page_start or 1)),
                    fulltext_char_start=start,
                    fulltext_char_end=end,
                )
            )
            cursor = end + 2
        return applied

    def _fulltext_from_sections(self, sections: list[ExtractedSection]) -> str:
        return "\n\n".join(section.text for section in sections if _normalize_text(section.text))

    def _build_snippets(self, fulltext: str) -> list[dict[str, Any]]:
        text = _normalize_text(fulltext)
        if not text:
            return []
        snippets: list[dict[str, Any]] = []
        step = max(1, self.config.snippet_target_chars - self.config.snippet_overlap_chars)
        start = 0
        snippet_index = 1
        while start < len(text):
            end = min(len(text), start + self.config.snippet_target_chars)
            snippet_text = text[start:end].strip()
            if snippet_text:
                snippets.append(
                    {
                        "snippet_id": f"snippet-{snippet_index}",
                        "char_start": start,
                        "char_end": end,
                        "text": snippet_text,
                    }
                )
                snippet_index += 1
            if end >= len(text):
                break
            start += step
        return snippets

    def _section_payload(self, section: ExtractedSection) -> dict[str, Any]:
        return {
            "section_id": f"section-{section.fulltext_char_start}-{section.fulltext_char_end}",
            "section_type": section.section_type,
            "heading": section.heading,
            "page_start": section.page_start,
            "page_end": max(section.page_start, section.page_end),
            "fulltext_char_start": section.fulltext_char_start,
            "fulltext_char_end": section.fulltext_char_end,
            "text": section.text,
        }

    def _evaluate_quality(self, *, fulltext: str, page_texts: list[str]) -> dict[str, Any]:
        total_chars = len(_compact_text(fulltext))
        page_count = len(page_texts)
        text_pages = [page for page in page_texts if len(_compact_text(page)) >= self.config.min_chars_per_text_page]
        text_page_ratio = len(text_pages) / max(1, page_count)
        printable_ratio = _printable_text_ratio(fulltext)
        repeated_ratio = _repeated_line_ratio(fulltext)
        reasons: list[str] = []
        if total_chars < self.config.min_total_chars:
            reasons.append("total_chars_below_threshold")
        if text_page_ratio < self.config.min_text_page_ratio:
            reasons.append("text_page_ratio_below_threshold")
        if printable_ratio < self.config.min_printable_ratio:
            reasons.append("printable_ratio_below_threshold")
        if repeated_ratio > 0.35:
            reasons.append("repeated_line_ratio_above_threshold")
        return {
            "total_chars": total_chars,
            "page_count": page_count,
            "text_page_ratio": round(text_page_ratio, 4),
            "printable_ratio": round(printable_ratio, 4),
            "repeated_line_ratio": round(repeated_ratio, 4),
            "reasons": reasons,
        }

    def _attempt_payload(self, attempt: ExtractionAttempt) -> dict[str, Any]:
        return {
            "extractor": attempt.extractor,
            "succeeded": attempt.succeeded,
            "usable": attempt.usable,
            "failure_reason": attempt.failure_reason,
            "metrics": dict(attempt.metrics or {}),
            "page_count": attempt.page_count,
        }

    def _is_true_pdf(self, content: bytes) -> bool:
        return bytes(content[:5]) == b"%PDF-"


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Portable paper parsing skill")
    parser.add_argument("--input", required=True, help="Local PDF/text path to parse")
    parser.add_argument("--output-dir", required=True, help="Directory for emitted artifacts")
    parser.add_argument("--config-json", default=None, help="Optional JSON object overriding parser config")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    config = ParserConfig.from_dict(json.loads(args.config_json) if args.config_json else None)
    engine = PaperParseEngine(config=config)
    result = engine.process_document(input_path=args.input, output_dir=args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
