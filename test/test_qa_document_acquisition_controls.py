from __future__ import annotations

import json
import shutil
import time
import unittest
import uuid
from pathlib import Path

import pymupdf as fitz

from qa.artifacts import QAArtifactStore
from qa.handoff import EvidenceExtractorHandoff
from qa.nodes.document_acquirer import DocumentAcquirerNode
from qa.providers import DEFAULT_BROWSER_USER_AGENT, FetchedDocument, HttpTextFetcher, ProviderRequestError
from qa.retrieval_state import PaperCandidate, PaperRecord, Section, SectionIndex
from qa.state import QueryConstraints, TaskSpec


class _StaticFetcher:
    def __init__(self, fetched: FetchedDocument) -> None:
        self._fetched = fetched

    def fetch(self, url: str) -> FetchedDocument:
        return self._fetched


class _SlowFetcher:
    def __init__(self, *, delay_seconds: float) -> None:
        self.delay_seconds = float(delay_seconds)

    def fetch(self, url: str) -> FetchedDocument:
        time.sleep(self.delay_seconds)
        return FetchedDocument(url=url, content_type="text/plain", text="slow response")


class _LookupClient:
    def __init__(self, payload: dict) -> None:
        self.payload = dict(payload)
        self.calls: list[str] = []

    def lookup(self, doi: str):
        self.calls.append(doi)
        return dict(self.payload)


class _RecordedFetcher:
    def __init__(self, responses: dict[str, object]) -> None:
        self.responses = dict(responses)
        self.urls: list[str] = []

    def fetch(self, url: str) -> FetchedDocument:
        self.urls.append(url)
        response = self.responses[url]
        if isinstance(response, Exception):
            raise response
        return response


def _make_pdf_bytes(pages: list[str]) -> bytes:
    document = fitz.open()
    try:
        for page_text in pages:
            page = document.new_page()
            page.insert_textbox(fitz.Rect(40, 40, 555, 800), page_text, fontsize=11)
        return document.tobytes()
    finally:
        document.close()


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
    def test_http_text_fetcher_uses_browser_user_agent_by_default(self):
        seen_headers: list[dict] = []

        class _Response:
            status_code = 200
            headers = {"content-type": "application/pdf"}
            content = b"%PDF-1.4\n"
            text = ""
            url = "https://example.test/paper.pdf"
            history = []

            def raise_for_status(self) -> None:
                return None

        def _request_get(url, *, headers=None, timeout=None, params=None):
            del url, timeout, params
            seen_headers.append(dict(headers or {}))
            return _Response()

        fetcher = HttpTextFetcher(request_get=_request_get)
        fetched = fetcher.fetch("https://example.test/paper.pdf")

        self.assertEqual("application/pdf", fetched.content_type)
        self.assertEqual(DEFAULT_BROWSER_USER_AGENT, seen_headers[0]["User-Agent"])

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

    def test_http_text_fetcher_rejects_excessive_redirects(self):
        class _Response:
            status_code = 200
            headers = {"content-type": "text/html"}
            content = b"<html></html>"
            text = "<html></html>"
            url = "https://example.test/final"
            history = [object(), object(), object()]

            def raise_for_status(self) -> None:
                return None

        fetcher = HttpTextFetcher(
            request_get=lambda *args, **kwargs: _Response(),
            max_redirects=1,
        )

        with self.assertRaises(ProviderRequestError) as ctx:
            fetcher.fetch("https://example.test/start")

        self.assertIn("redirect limit exceeded", str(ctx.exception))

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

    def test_document_acquirer_extracts_real_pdf_into_fulltext_artifacts(self):
        pdf_bytes = _make_pdf_bytes(
            [
                (
                    "Abstract\nPt/C remains a standard HER benchmark in alkaline electrolyte, and recent studies "
                    "compare dispersion, support morphology, local water structure, and catalyst-layer transport under "
                    "matched catalyst loading, electrolyte composition, and normalization conventions."
                ),
                (
                    "Results\nIn 1 M KOH, Pt/C delivered lower overpotential at 10 mA cm-2 than bare carbon, "
                    "retained stable current during repeated sweeps, and showed reduced charge-transfer resistance "
                    "in impedance measurements under comparable ink formulation, loading, and support surface area."
                ),
                (
                    "Discussion\nThe observed advantage is consistent with improved utilization of exposed Pt sites, "
                    "faster interfacial charge transfer, and more favorable bubble release, while comparisons remain "
                    "sensitive to normalization choice, uncompensated resistance treatment, catalyst-layer thickness, "
                    "and the specific current-density regime used for benchmarking."
                ),
            ]
        )
        node = DocumentAcquirerNode(
            unpaywall_client=None,
            fetcher=_StaticFetcher(
                FetchedDocument(
                    url="https://example.test/paper.pdf",
                    content_type="application/pdf",
                    binary=pdf_bytes,
                )
            ),
        )

        tmpdir = _workspace_tmpdir()
        try:
            records, indices = node.run([_candidate()], artifact_store=QAArtifactStore(base_dir=tmpdir))
            record = records[0]
            self.assertTrue(record.fulltext_available)
            self.assertEqual("fulltext_indexed", record.fulltext_status)
            self.assertEqual("application/pdf", record.fulltext_format)
            self.assertEqual("pymupdf", record.fulltext_extractor)
            self.assertTrue(str(record.source_artifact_path).endswith(".pdf"))
            self.assertTrue(str(record.fulltext_artifact_path).endswith(".fulltext.txt"))
            self.assertTrue(str(record.sections_artifact_path).endswith(".sections.json"))
            self.assertTrue(str(record.snippets_artifact_path).endswith(".snippets.jsonl"))
            self.assertTrue(Path(record.fulltext_artifact_path).exists())
            self.assertTrue(Path(record.snippets_artifact_path).exists())
            self.assertEqual("fulltext_indexed", indices[0].fulltext_status)
            self.assertTrue(indices[0].sections)

            snippet_lines = [
                json.loads(line)
                for line in Path(record.snippets_artifact_path).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertTrue(snippet_lines)
            self.assertEqual("paper-1", snippet_lines[0]["paper_id"])
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_document_acquirer_marks_invalid_pdf_payload_unusable(self):
        node = DocumentAcquirerNode(
            unpaywall_client=None,
            fetcher=_StaticFetcher(
                FetchedDocument(
                    url="https://example.test/paper.pdf",
                    content_type="application/pdf",
                    binary=b"<html><body>Redirecting</body></html>",
                )
            ),
        )

        tmpdir = _workspace_tmpdir()
        try:
            records, indices = node.run([_candidate()], artifact_store=QAArtifactStore(base_dir=tmpdir))
            record = records[0]
            self.assertTrue(record.fulltext_available)
            self.assertEqual("fulltext_unusable", record.fulltext_status)
            self.assertTrue(record.extraction_warnings)
            self.assertTrue(Path(record.extraction_report_path).exists())
            self.assertEqual("fulltext_unusable", indices[0].fulltext_status)
            self.assertEqual([], indices[0].sections)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_document_acquirer_times_out_single_paper_and_records_runtime_diagnostic(self):
        node = DocumentAcquirerNode(
            unpaywall_client=None,
            fetcher=_SlowFetcher(delay_seconds=0.2),
            document_fetch_timeout_seconds=0.05,
            document_fetch_total_timeout_seconds=0.1,
        )

        tmpdir = _workspace_tmpdir()
        try:
            store = QAArtifactStore(base_dir=tmpdir)
            records, indices = node.run([_candidate()], artifact_store=store)
            record = records[0]
            self.assertFalse(record.fulltext_available)
            self.assertEqual("abstract_only", record.fulltext_status)
            self.assertEqual("abstract_only", indices[0].fulltext_status)
            self.assertTrue(any("timed out" in warning for warning in node.last_execution_warnings))
            runtime_payload = json.loads((tmpdir / "diagnostics" / "document_acquirer_runtime.json").read_text(encoding="utf-8"))
            self.assertEqual("paper-1", runtime_payload["paper_id"])
            self.assertIn(runtime_payload["status"], {"timeout", "success"})
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_download_pdf_only_with_fallback_uses_unpaywall_pdf_first(self):
        unpaywall_url = "https://oa.example.test/paper.pdf"
        fallback_url = "https://ss.example.test/paper.pdf"
        fetcher = _RecordedFetcher(
            {
                unpaywall_url: FetchedDocument(
                    url=unpaywall_url,
                    content_type="application/pdf",
                    binary=b"%PDF-1.4\nunpaywall\n",
                )
            }
        )
        node = DocumentAcquirerNode(
            unpaywall_client=_LookupClient(
                {
                    "best_oa_location": {
                        "url_for_pdf": unpaywall_url,
                    }
                }
            ),
            fetcher=fetcher,
        )
        candidate = _candidate().model_copy(update={"open_access_pdf_url": fallback_url})

        tmpdir = _workspace_tmpdir()
        try:
            record, index = node.download_pdf_only_with_fallback(
                candidate=candidate,
                artifact_store=QAArtifactStore(base_dir=tmpdir),
            )

            self.assertEqual([unpaywall_url], fetcher.urls)
            self.assertEqual("binary_only", record.fulltext_status)
            self.assertEqual("binary_only", index.fulltext_status)
            self.assertTrue(str(record.fulltext_artifact_path).endswith(".pdf"))
            self.assertIn("unpaywall_pdf", record.provider_sources)
            self.assertNotIn("semantic_scholar_pdf_fallback", record.provider_sources)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_download_pdf_only_with_fallback_retries_semantic_scholar_once(self):
        unpaywall_url = "https://oa.example.test/paper.pdf"
        fallback_url = "https://ss.example.test/paper.pdf"
        fetcher = _RecordedFetcher(
            {
                unpaywall_url: RuntimeError("upstream 403"),
                fallback_url: FetchedDocument(
                    url=fallback_url,
                    content_type="application/pdf",
                    binary=b"%PDF-1.4\nsemantic\n",
                ),
            }
        )
        node = DocumentAcquirerNode(
            unpaywall_client=_LookupClient(
                {
                    "best_oa_location": {
                        "url_for_pdf": unpaywall_url,
                    }
                }
            ),
            fetcher=fetcher,
        )
        candidate = _candidate().model_copy(update={"open_access_pdf_url": fallback_url})

        tmpdir = _workspace_tmpdir()
        try:
            record, index = node.download_pdf_only_with_fallback(
                candidate=candidate,
                artifact_store=QAArtifactStore(base_dir=tmpdir),
            )

            self.assertEqual([unpaywall_url, fallback_url], fetcher.urls)
            self.assertEqual("binary_only", record.fulltext_status)
            self.assertEqual("binary_only", index.fulltext_status)
            self.assertIn("semantic_scholar_pdf_fallback", record.provider_sources)
            self.assertNotIn("unpaywall_pdf", record.provider_sources[-1:])
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_download_pdf_only_with_fallback_uses_semantic_scholar_when_unpaywall_pdf_url_missing(self):
        fallback_url = "https://ss.example.test/paper.pdf"
        fetcher = _RecordedFetcher(
            {
                fallback_url: FetchedDocument(
                    url=fallback_url,
                    content_type="application/pdf",
                    binary=b"%PDF-1.4\nsemantic\n",
                ),
            }
        )
        node = DocumentAcquirerNode(
            unpaywall_client=_LookupClient(
                {
                    "best_oa_location": {},
                }
            ),
            fetcher=fetcher,
        )
        candidate = _candidate().model_copy(update={"open_access_pdf_url": fallback_url})

        tmpdir = _workspace_tmpdir()
        try:
            record, index = node.download_pdf_only_with_fallback(
                candidate=candidate,
                artifact_store=QAArtifactStore(base_dir=tmpdir),
            )

            self.assertEqual([fallback_url], fetcher.urls)
            self.assertEqual("binary_only", record.fulltext_status)
            self.assertEqual("binary_only", index.fulltext_status)
            self.assertIn("semantic_scholar_pdf_fallback", record.provider_sources)
            runtime_payload = json.loads((tmpdir / "diagnostics" / "document_acquirer_runtime.json").read_text(encoding="utf-8"))
            self.assertEqual("semantic_scholar_pdf_fallback", runtime_payload["download_source"])
            self.assertEqual("missing_unpaywall_url_for_pdf", runtime_payload["fallback_reason"])
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_download_pdf_only_with_fallback_fails_when_both_sources_fail(self):
        unpaywall_url = "https://oa.example.test/paper.pdf"
        fallback_url = "https://ss.example.test/paper.pdf"
        fetcher = _RecordedFetcher(
            {
                unpaywall_url: RuntimeError("upstream 403"),
                fallback_url: RuntimeError("fallback 404"),
            }
        )
        node = DocumentAcquirerNode(
            unpaywall_client=_LookupClient(
                {
                    "best_oa_location": {
                        "url_for_pdf": unpaywall_url,
                    }
                }
            ),
            fetcher=fetcher,
        )
        candidate = _candidate().model_copy(update={"open_access_pdf_url": fallback_url})

        tmpdir = _workspace_tmpdir()
        try:
            with self.assertRaises(RuntimeError):
                node.download_pdf_only_with_fallback(
                    candidate=candidate,
                    artifact_store=QAArtifactStore(base_dir=tmpdir),
                )
            self.assertEqual([unpaywall_url, fallback_url], fetcher.urls)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_download_pdf_only_with_fallback_skips_duplicate_semantic_retry_url(self):
        unpaywall_url = "https://oa.example.test/paper.pdf"
        fetcher = _RecordedFetcher(
            {
                unpaywall_url: RuntimeError("upstream 403"),
            }
        )
        node = DocumentAcquirerNode(
            unpaywall_client=_LookupClient(
                {
                    "best_oa_location": {
                        "url_for_pdf": unpaywall_url,
                    }
                }
            ),
            fetcher=fetcher,
        )
        candidate = _candidate().model_copy(update={"open_access_pdf_url": unpaywall_url})

        tmpdir = _workspace_tmpdir()
        try:
            with self.assertRaises(RuntimeError):
                node.download_pdf_only_with_fallback(
                    candidate=candidate,
                    artifact_store=QAArtifactStore(base_dir=tmpdir),
                )
            self.assertEqual([unpaywall_url], fetcher.urls)
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
