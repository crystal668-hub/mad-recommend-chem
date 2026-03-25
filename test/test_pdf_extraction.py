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
    def test_process_indexes_born_digital_pdf_with_pymupdf_by_default(self):
        pipeline = PDFExtractionPipeline()
        pipeline._extract_with_docling = lambda pdf_path: self.fail("docling should be disabled by default")  # type: ignore[method-assign]

        tmpdir = _workspace_tmpdir()
        try:
            result = pipeline.process(
                paper_id="paper-1",
                pdf_bytes=_make_pdf_bytes(
                    [
                        (
                            "Introduction\nPt/C catalysts remain a common HER benchmark in alkaline electrolyte. "
                            "Recent reports compare carbon support morphology, metal dispersion, and interfacial water structure. "
                            "Several studies note that activity trends depend on catalyst loading, KOH concentration, and current-density regime."
                        ),
                        (
                            "Results\nIn 1 M KOH, Pt/C delivered lower overpotential at 10 mA cm-2 than bare carbon and retained stable current "
                            "over repeated sweeps. Electrochemical impedance suggested faster charge transfer, while Tafel analysis indicated "
                            "improved apparent kinetics under matched loading and ink formulation."
                        ),
                        (
                            "Discussion\nThe advantage appears to arise from better utilization of exposed Pt sites together with support-enabled "
                            "wetting and gas release. However, comparisons across papers remain sensitive to normalization choice, catalyst layer "
                            "thickness, and uncompensated resistance treatment."
                        ),
                    ]
                ),
                artifact_store=QAArtifactStore(base_dir=tmpdir),
            )
            self.assertEqual("fulltext_indexed", result.fulltext_status)
            self.assertEqual("pymupdf", result.extractor)
            self.assertFalse(result.ocr_applied)
            self.assertTrue(Path(result.fulltext_artifact_path).exists())
            self.assertTrue(Path(result.sections_artifact_path).exists())
            self.assertTrue(Path(result.snippets_artifact_path).exists())
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_process_uses_docling_when_explicitly_enabled_and_pymupdf_quality_is_rejected(self):
        pipeline = PDFExtractionPipeline(config={"secondary_backend": "docling"})
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

    def test_process_reports_pymupdf_quality_failure_when_docling_is_disabled(self):
        pipeline = PDFExtractionPipeline()
        pipeline._extract_with_docling = lambda pdf_path: self.fail("docling should be disabled by default")  # type: ignore[method-assign]

        tmpdir = _workspace_tmpdir()
        try:
            result = pipeline.process(
                paper_id="paper-1",
                pdf_bytes=_make_pdf_bytes(["1", "2", "3"]),
                artifact_store=QAArtifactStore(base_dir=tmpdir),
            )
            self.assertEqual("fulltext_unusable", result.fulltext_status)
            self.assertEqual([], result.report["attempts"][1:])
            self.assertTrue(any("pymupdf extraction was rejected by quality gates" in warning.lower() for warning in result.warnings))
            self.assertTrue(Path(result.extraction_report_path).exists())
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_process_salvages_high_quality_text_when_only_garbled_flag_is_triggered(self):
        pipeline = PDFExtractionPipeline()
        salvaged_attempt = _build_usable_attempt(pipeline, extractor="pymupdf")
        salvaged_attempt.usable = False
        salvaged_attempt.metrics = {
            **salvaged_attempt.metrics,
            "total_chars": 6000,
            "quality_score": 0.65,
            "reasons": ["garbled_text_detected"],
        }
        pipeline._extract_with_pymupdf = lambda pdf_bytes, ocr_applied=False: salvaged_attempt  # type: ignore[method-assign]

        tmpdir = _workspace_tmpdir()
        try:
            result = pipeline.process(
                paper_id="paper-2",
                pdf_bytes=_make_pdf_bytes(["Results\nUseful extracted text."] * 3),
                artifact_store=QAArtifactStore(base_dir=tmpdir),
            )
            self.assertEqual("fulltext_indexed", result.fulltext_status)
            self.assertTrue(any("retained for fallback indexing" in warning.lower() for warning in result.warnings))
            self.assertTrue(Path(result.fulltext_artifact_path).exists())
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_section_payload_normalizes_page_end_when_offsets_are_reversed(self):
        pipeline = PDFExtractionPipeline()
        payload = pipeline._section_payload(
            ExtractedSection(
                heading="Results",
                section_type="results",
                text="Useful text.",
                page_start=7,
                page_end=3,
                fulltext_char_start=0,
                fulltext_char_end=12,
            )
        )

        self.assertEqual(7, payload["page_start"])
        self.assertEqual(7, payload["page_end"])


if __name__ == "__main__":
    unittest.main()
