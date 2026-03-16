from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

from qa.artifacts import QAArtifactStore
from qa.handoff import EvidenceExtractorHandoff
from qa.nodes.document_acquirer import DocumentAcquirerNode
from qa.providers import FetchedDocument, HttpTextFetcher
from qa.retrieval_state import PaperCandidate, PaperRecord, Section, SectionIndex
from qa.state import QueryConstraints, TaskSpec


class _StaticFetcher:
    def __init__(self, fetched: FetchedDocument) -> None:
        self._fetched = fetched

    def fetch(self, url: str) -> FetchedDocument:
        return self._fetched


def _candidate() -> PaperCandidate:
    return PaperCandidate.model_validate(
        {
            "paper_id": "paper-1",
            "doi": "10.1000/test",
            "title": "Pt/C in alkaline HER",
            "abstract": "Pt/C improves HER in 1 M KOH.",
            "authors": ["A. Author"],
            "year": 2024,
            "venue": "Journal",
            "provider_hits": ["openalex"],
            "lane_sources": ["review"],
            "retrieval_score": 0.9,
            "ranking_features": {},
            "provider_artifacts": {},
            "oa_url": "https://example.test/fulltext",
        }
    )


def _task_spec(question_type: str = "mechanism") -> TaskSpec:
    return TaskSpec.model_validate(
        {
            "question": "Does Pt/C improve HER activity in 1 M KOH?",
            "normalized_question": "does pt/c improve her activity in 1 m koh",
            "question_type": question_type,
            "recency_policy": "none",
            "answer_sections": [],
            "required_condition_axes": [],
            "query_constraints": QueryConstraints(),
            "ambiguity_flags": [],
            "router_confidence": 0.9,
        }
    )


def _workspace_tmpdir() -> Path:
    root = Path("test") / "_tmp" / f"qa_controls_{uuid.uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    return root


class DocumentAcquisitionTests(unittest.TestCase):
    def test_http_text_fetcher_keeps_image_payload_binary(self):
        class _Response:
            status_code = 200
            headers = {"content-type": "image/jpeg"}
            content = b"\xff\xd8\xff"
            text = "should-not-be-used"

            def raise_for_status(self) -> None:
                return None

        fetcher = HttpTextFetcher(request_get=lambda *args, **kwargs: _Response())
        fetched = fetcher.fetch("https://example.test/figure.jpg")

        self.assertEqual("image/jpeg", fetched.content_type)
        self.assertEqual(b"\xff\xd8\xff", fetched.binary)
        self.assertIsNone(fetched.text)

    def test_document_acquirer_ignores_short_redirect_pages(self):
        node = DocumentAcquirerNode(
            unpaywall_client=None,
            fetcher=_StaticFetcher(
                FetchedDocument(
                    url="https://example.test/landing",
                    content_type="text/html",
                    text="<html><body>Redirecting</body></html>",
                )
            ),
        )

        tmpdir = _workspace_tmpdir()
        try:
            records, indices = node.run([_candidate()], artifact_store=QAArtifactStore(base_dir=tmpdir))
            self.assertFalse(records[0].fulltext_available)
            self.assertIsNone(records[0].fulltext_artifact_path)
            self.assertEqual("abstract_only", indices[0].fulltext_status)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_document_acquirer_preserves_binary_non_text_artifacts(self):
        node = DocumentAcquirerNode(
            unpaywall_client=None,
            fetcher=_StaticFetcher(
                FetchedDocument(
                    url="https://example.test/figure.jpg",
                    content_type="image/jpeg",
                    binary=b"\xff\xd8\xff",
                )
            ),
        )

        tmpdir = _workspace_tmpdir()
        try:
            records, indices = node.run([_candidate()], artifact_store=QAArtifactStore(base_dir=tmpdir))
            self.assertTrue(records[0].fulltext_available)
            self.assertTrue(str(records[0].fulltext_artifact_path).endswith(".jpg"))
            self.assertTrue(Path(records[0].fulltext_artifact_path).exists())
            self.assertEqual("binary_only", indices[0].fulltext_status)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class HandoffFallbackTests(unittest.TestCase):
    def test_unknown_fulltext_section_is_allowed_when_text_is_long_and_textual(self):
        handoff = EvidenceExtractorHandoff()
        long_text = " ".join(["Pt/C improves HER activity in 1 M KOH."] * 120)

        tmpdir = _workspace_tmpdir()
        try:
            fulltext_path = tmpdir / "paper.txt"
            fulltext_path.write_text(long_text, encoding="utf-8")
            paper_record = PaperRecord.model_validate(
                {
                    "paper_id": "paper-1",
                    "title": "Pt/C in alkaline HER",
                    "fulltext_available": True,
                    "fulltext_format": "text/plain",
                    "fulltext_artifact_path": str(fulltext_path),
                }
            )
            section_index = SectionIndex(
                paper_id="paper-1",
                fulltext_status="fulltext_indexed",
                sections=[
                    Section(
                        section_id="sec_0_unknown",
                        section_type="unknown",
                        heading="Body",
                        fulltext_char_start=0,
                        fulltext_char_end=len(long_text),
                    )
                ],
            )

            views = handoff.read_preferred_sections(
                paper_record=paper_record,
                section_index=section_index,
                task_spec=_task_spec("mechanism"),
                evidence_is_weak=True,
                missing_conditions=True,
            )
            self.assertEqual(1, len(views))
            self.assertEqual("unknown", views[0].section_type)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_unknown_fulltext_section_is_rejected_when_garbled(self):
        handoff = EvidenceExtractorHandoff()
        garbage_text = "JFIF ICC_PROFILE 8BIM " * 100

        tmpdir = _workspace_tmpdir()
        try:
            fulltext_path = tmpdir / "paper.txt"
            fulltext_path.write_text(garbage_text, encoding="utf-8")
            paper_record = PaperRecord.model_validate(
                {
                    "paper_id": "paper-1",
                    "title": "Pt/C in alkaline HER",
                    "fulltext_available": True,
                    "fulltext_format": "text/plain",
                    "fulltext_artifact_path": str(fulltext_path),
                }
            )
            section_index = SectionIndex(
                paper_id="paper-1",
                fulltext_status="fulltext_indexed",
                sections=[
                    Section(
                        section_id="sec_0_unknown",
                        section_type="unknown",
                        heading="Body",
                        fulltext_char_start=0,
                        fulltext_char_end=len(garbage_text),
                    )
                ],
            )

            views = handoff.read_preferred_sections(
                paper_record=paper_record,
                section_index=section_index,
                task_spec=_task_spec("mechanism"),
                evidence_is_weak=True,
                missing_conditions=True,
            )
            self.assertEqual([], views)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
