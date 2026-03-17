from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

import pymupdf as fitz

from qa.artifacts import QAArtifactStore
from qa.pdf_extraction import ExtractedSection, ExtractionAttempt, PDFExtractionPipeline


def _workspace_tmpdir() -> Path:
    root = Path("test") / "_tmp" / f"pdf_extraction_{uuid.uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _make_pdf_bytes(pages: list[str]) -> bytes:
    document = fitz.open()
    try:
        for page_text in pages:
            page = document.new_page()
            page.insert_textbox(fitz.Rect(40, 40, 555, 800), page_text, fontsize=11)
        return document.tobytes()
    finally:
        document.close()


def _build_usable_attempt(
    pipeline: PDFExtractionPipeline,
    *,
    extractor: str,
    ocr_applied: bool = False,
) -> ExtractionAttempt:
    raw_sections = [
        ExtractedSection(
            heading="Results",
            section_type="results",
            text=" ".join(["Pt/C improved HER activity in 1 M KOH and preserved stability."] * 35),
            page_start=1,
            page_end=2,
        ),
        ExtractedSection(
            heading="Discussion",
            section_type="discussion",
            text=" ".join(["The performance gain was attributed to faster charge transfer."] * 25),
            page_start=2,
            page_end=3,
        ),
    ]
    sections = pipeline._apply_section_offsets(raw_sections)
    fulltext = pipeline._fulltext_from_sections(sections)
    return ExtractionAttempt(
        extractor=extractor,
        succeeded=True,
        fulltext=fulltext,
        page_texts=["alpha " * 120, "beta " * 120, "gamma " * 120],
        sections=sections,
        page_count=3,
        metrics={
            "total_chars": len(fulltext),
            "page_count": 3,
            "text_page_ratio": 1.0,
            "printable_ratio": 1.0,
            "repeated_line_ratio": 0.0,
            "quality_score": 0.92,
            "reasons": [],
        },
        usable=True,
        ocr_applied=ocr_applied,
    )


class PDFExtractionPipelineTests(unittest.TestCase):
    def test_process_uses_docling_when_pymupdf_quality_is_rejected(self):
        pipeline = PDFExtractionPipeline()
        pipeline._extract_with_docling = lambda pdf_path: _build_usable_attempt(pipeline, extractor="docling")  # type: ignore[method-assign]

        tmpdir = _workspace_tmpdir()
        try:
            result = pipeline.process(
                paper_id="paper-1",
                pdf_bytes=_make_pdf_bytes(["1", "2", "3"]),
                artifact_store=QAArtifactStore(base_dir=tmpdir),
            )
            self.assertEqual("fulltext_indexed", result.fulltext_status)
            self.assertEqual("docling", result.extractor)
            self.assertFalse(result.ocr_applied)
            self.assertTrue(Path(result.fulltext_artifact_path).exists())
            self.assertTrue(Path(result.sections_artifact_path).exists())
            self.assertTrue(Path(result.snippets_artifact_path).exists())
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_process_accepts_ocr_output_when_fallback_succeeds(self):
        pipeline = PDFExtractionPipeline()
        pipeline._extract_with_docling = lambda pdf_path: ExtractionAttempt(  # type: ignore[method-assign]
            extractor="docling",
            succeeded=False,
            failure_reason="docling unavailable",
        )
        pipeline._extract_with_ocr = lambda pdf_path: _build_usable_attempt(  # type: ignore[method-assign]
            pipeline,
            extractor="pymupdf",
            ocr_applied=True,
        )

        tmpdir = _workspace_tmpdir()
        try:
            result = pipeline.process(
                paper_id="paper-1",
                pdf_bytes=_make_pdf_bytes(["1", "2", "3"]),
                artifact_store=QAArtifactStore(base_dir=tmpdir),
            )
            self.assertEqual("fulltext_indexed", result.fulltext_status)
            self.assertTrue(result.ocr_applied)
            self.assertEqual("pymupdf", result.extractor)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_process_surfaces_ocr_unavailable_as_warning(self):
        pipeline = PDFExtractionPipeline()
        pipeline._extract_with_docling = lambda pdf_path: ExtractionAttempt(  # type: ignore[method-assign]
            extractor="docling",
            succeeded=False,
            failure_reason="docling unavailable",
        )
        pipeline._extract_with_ocr = lambda pdf_path: ExtractionAttempt(  # type: ignore[method-assign]
            extractor="ocrmypdf",
            succeeded=False,
            failure_reason="ocr_unavailable",
        )

        tmpdir = _workspace_tmpdir()
        try:
            result = pipeline.process(
                paper_id="paper-1",
                pdf_bytes=_make_pdf_bytes(["1", "2", "3"]),
                artifact_store=QAArtifactStore(base_dir=tmpdir),
            )
            self.assertEqual("fulltext_unusable", result.fulltext_status)
            self.assertTrue(any("ocr fallback" in warning.lower() for warning in result.warnings))
            self.assertTrue(Path(result.extraction_report_path).exists())
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
