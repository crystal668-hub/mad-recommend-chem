from __future__ import annotations

import html
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

from qa.artifacts import QAArtifactStore
from qa.pdf_extraction import PDFExtractionPipeline
from qa.providers import HttpTextFetcher, ProviderRequestError, ProviderUnavailableError
from qa.retrieval_state import PaperCandidate, PaperRecord, RetrievalDiagnosticRecord, Section, SectionIndex
from qa.retrieval_utils import (
    guess_binary_extension,
    looks_like_garbled_text,
    looks_like_placeholder_text,
    normalize_text,
)


logger = logging.getLogger("MAD.qa.document_acquirer")

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


class DocumentAcquirerNode:
    def __init__(
        self,
        unpaywall_client: Optional[Any] = None,
        fetcher: Optional[Any] = None,
        pdf_extractor: Optional[PDFExtractionPipeline] = None,
        document_fetch_timeout_seconds: float = 45.0,
        document_fetch_total_timeout_seconds: float = 300.0,
    ) -> None:
        self.unpaywall_client = unpaywall_client
        self.fetcher = fetcher or HttpTextFetcher()
        self.pdf_extractor = pdf_extractor or PDFExtractionPipeline()
        self.document_fetch_timeout_seconds = max(0.01, float(document_fetch_timeout_seconds))
        self.document_fetch_total_timeout_seconds = max(
            self.document_fetch_timeout_seconds,
            float(document_fetch_total_timeout_seconds),
        )
        self.last_diagnostics: List[RetrievalDiagnosticRecord] = []
        self.last_provider_health: dict[str, dict[str, Any]] = {}
        self.last_execution_warnings: list[str] = []
        self.last_runtime_status: dict[str, Any] = {}
        self._diagnostic_map: dict[tuple[str, str, str], dict[str, Any]] = {}
        self._provider_health_fallback: dict[str, dict[str, Any]] = {}

    def run(
        self,
        candidates: Sequence[PaperCandidate],
        artifact_store: Optional[QAArtifactStore] = None,
        parse_fulltext: bool = True,
    ) -> Tuple[List[PaperRecord], List[SectionIndex]]:
        store = artifact_store or QAArtifactStore()
        paper_records: List[PaperRecord] = []
        section_indices: List[SectionIndex] = []
        self._diagnostic_map = {}
        self._provider_health_fallback = self._init_provider_health()
        self.last_execution_warnings = []

        if self.unpaywall_client is None:
            skipped_candidates = sum(1 for candidate in candidates if candidate.doi)
            if skipped_candidates:
                self._record_diagnostic(
                    provider="unpaywall",
                    stage="lookup",
                    outcome="skipped",
                    count=skipped_candidates,
                    message="Unpaywall lookup skipped because the client is not configured.",
                )
                self._provider_health_fallback["unpaywall"]["status"] = "disabled"
                self._provider_health_fallback["unpaywall"]["skipped_calls"] += skipped_candidates
                self._provider_health_fallback["unpaywall"]["last_error"] = (
                    "Unpaywall lookup skipped because the client is not configured."
                )

        for candidate in candidates:
            paper_record, section_index = self._acquire_one(
                candidate=candidate,
                store=store,
                parse_fulltext=parse_fulltext,
            )
            paper_records.append(paper_record)
            section_indices.append(section_index)

        self.last_diagnostics = self._finalize_diagnostics()
        self.last_provider_health = self._collect_provider_health()
        return paper_records, section_indices

    def download_documents(
        self,
        candidates: Sequence[PaperCandidate],
        artifact_store: Optional[QAArtifactStore] = None,
    ) -> Tuple[List[PaperRecord], List[SectionIndex]]:
        return self.run(
            candidates=candidates,
            artifact_store=artifact_store,
            parse_fulltext=False,
        )

    def download_pdf_only_with_fallback(
        self,
        *,
        candidate: PaperCandidate,
        artifact_store: Optional[QAArtifactStore] = None,
    ) -> Tuple[PaperRecord, SectionIndex]:
        store = artifact_store or QAArtifactStore()
        self._diagnostic_map = {}
        self._provider_health_fallback = self._init_provider_health()
        self.last_execution_warnings = []
        paper_record, section_index = self._download_one_with_fallback(
            candidate=candidate,
            store=store,
        )
        self.last_diagnostics = self._finalize_diagnostics()
        self.last_provider_health = self._collect_provider_health()
        return paper_record, section_index

    def parse_documents(
        self,
        paper_records: Sequence[PaperRecord],
        artifact_store: Optional[QAArtifactStore] = None,
    ) -> Tuple[List[PaperRecord], List[SectionIndex]]:
        store = artifact_store or QAArtifactStore()
        parsed_records: List[PaperRecord] = []
        section_indices: List[SectionIndex] = []
        for paper_record in paper_records:
            parsed_record, section_index = self._parse_one(
                paper_record=paper_record,
                store=store,
            )
            parsed_records.append(parsed_record)
            section_indices.append(section_index)
        return parsed_records, section_indices

    def _download_one_with_fallback(
        self,
        *,
        candidate: PaperCandidate,
        store: QAArtifactStore,
    ) -> Tuple[PaperRecord, SectionIndex]:
        provider_artifacts = dict(candidate.provider_artifacts)
        provider_sources = list(candidate.provider_hits)
        oa_url = None
        download_source = None
        fallback_reason: Optional[str] = None
        acquire_started_at = time.perf_counter()

        self._write_runtime_status(
            store=store,
            stage="download_document",
            status="start",
            paper_id=candidate.paper_id,
            provider="document_acquirer",
            metadata={
                "doi": candidate.doi,
                "document_fetch_timeout_seconds": self.document_fetch_timeout_seconds,
                "document_fetch_total_timeout_seconds": self.document_fetch_total_timeout_seconds,
            },
        )

        try:
            if not str(candidate.doi or "").strip():
                raise ValueError(f"paper_id={candidate.paper_id} has no DOI for Unpaywall lookup.")
            if self.unpaywall_client is None:
                raise RuntimeError("Unpaywall lookup is required for proposer PDF download but the client is not configured.")

            self._check_total_timeout(
                started_at=acquire_started_at,
                paper_id=candidate.paper_id,
                stage="unpaywall_lookup",
                store=store,
                provider="unpaywall",
            )
            self._write_runtime_status(
                store=store,
                stage="unpaywall_lookup",
                status="start",
                paper_id=candidate.paper_id,
                provider="unpaywall",
                metadata={"doi": candidate.doi},
                started_at=acquire_started_at,
            )
            try:
                unpaywall_payload = self._run_with_timeout(
                    lambda: self.unpaywall_client.lookup(candidate.doi),
                    timeout_seconds=self.document_fetch_timeout_seconds,
                )
                self._record_provider_health_success("unpaywall")
                self._record_diagnostic(provider="unpaywall", stage="lookup", outcome="hit")
                self._write_runtime_status(
                    store=store,
                    stage="unpaywall_lookup",
                    status="success",
                    paper_id=candidate.paper_id,
                    provider="unpaywall",
                    metadata={"doi": candidate.doi},
                    started_at=acquire_started_at,
                )
            except Exception as exc:
                self._record_provider_health_failure("unpaywall", exc)
                self._record_diagnostic(
                    provider="unpaywall",
                    stage="lookup",
                    outcome=self._classify_error(exc),
                    message=f"paper_id={candidate.paper_id}: {exc}",
                )
                self._write_runtime_status(
                    store=store,
                    stage="unpaywall_lookup",
                    status=self._classify_error(exc),
                    paper_id=candidate.paper_id,
                    provider="unpaywall",
                    metadata={"error": str(exc), "doi": candidate.doi},
                    started_at=acquire_started_at,
                )
                raise
            provider_artifacts["unpaywall"] = store.write_json(
                f"provider_raw/unpaywall/{candidate.paper_id}.json",
                unpaywall_payload,
            )
            provider_sources = list(dict.fromkeys([*provider_sources, "unpaywall"]))
            best_location = dict((unpaywall_payload or {}).get("best_oa_location") or {})
            unpaywall_pdf_url = str(best_location.get("url_for_pdf") or "").strip()
            fallback_url = str(
                candidate.open_access_pdf_url or candidate.best_oa_pdf_url or ""
            ).strip()
            if not unpaywall_pdf_url:
                if not fallback_url:
                    raise RuntimeError(
                        f"paper_id={candidate.paper_id} did not expose best_oa_location.url_for_pdf from Unpaywall."
                    )
                fallback_reason = "missing_unpaywall_url_for_pdf"
                pdf_artifact_path, resolved_oa_url = self._fetch_pdf_artifact(
                    candidate=candidate,
                    url=fallback_url,
                    store=store,
                    acquire_started_at=acquire_started_at,
                    download_source="semantic_scholar_pdf_fallback",
                )
                provider_sources = list(dict.fromkeys([*provider_sources, "semantic_scholar_pdf_fallback"]))
                oa_url = resolved_oa_url
                download_source = "semantic_scholar_pdf_fallback"
            else:
                try:
                    pdf_artifact_path, resolved_oa_url = self._fetch_pdf_artifact(
                        candidate=candidate,
                        url=unpaywall_pdf_url,
                        store=store,
                        acquire_started_at=acquire_started_at,
                        download_source="unpaywall_pdf",
                    )
                    provider_sources = list(dict.fromkeys([*provider_sources, "unpaywall_pdf"]))
                    oa_url = resolved_oa_url
                    download_source = "unpaywall_pdf"
                except Exception as first_exc:
                    if not fallback_url or fallback_url == unpaywall_pdf_url:
                        raise RuntimeError(
                            f"paper_id={candidate.paper_id} could not be downloaded from Unpaywall PDF URL: {first_exc}"
                        ) from first_exc
                    fallback_reason = "unpaywall_pdf_download_failed"
                    pdf_artifact_path, resolved_oa_url = self._fetch_pdf_artifact(
                        candidate=candidate,
                        url=fallback_url,
                        store=store,
                        acquire_started_at=acquire_started_at,
                        download_source="semantic_scholar_pdf_fallback",
                    )
                    provider_sources = list(dict.fromkeys([*provider_sources, "semantic_scholar_pdf_fallback"]))
                    oa_url = resolved_oa_url
                    download_source = "semantic_scholar_pdf_fallback"

            section_index = SectionIndex(
                paper_id=candidate.paper_id,
                fulltext_status="binary_only",
                sections=[],
            )
            index_artifact_path = store.write_json(
                f"indices/{candidate.paper_id}.json",
                section_index.model_dump(exclude_none=True),
            )
            paper_record = PaperRecord(
                paper_id=candidate.paper_id,
                doi=candidate.doi,
                title=candidate.title,
                abstract=candidate.abstract,
                authors=list(candidate.authors),
                year=candidate.year,
                venue=candidate.venue,
                provider_sources=provider_sources,
                provider_artifacts=provider_artifacts,
                oa_url=oa_url,
                fulltext_available=True,
                fulltext_status="binary_only",
                fulltext_format="application/pdf",
                fulltext_artifact_path=pdf_artifact_path,
                source_artifact_path=pdf_artifact_path,
                index_artifact_path=index_artifact_path,
            )
            self._write_runtime_status(
                store=store,
                stage="download_document",
                status="success",
                paper_id=candidate.paper_id,
                provider="document_acquirer",
                url=oa_url,
                metadata={
                    "download_source": download_source,
                    "fallback_reason": fallback_reason,
                    "fulltext_status": "binary_only",
                    "fulltext_available": True,
                    "section_count": 0,
                },
                started_at=acquire_started_at,
            )
            return paper_record, section_index
        except Exception as exc:
            self._write_runtime_status(
                store=store,
                stage="download_document",
                status=self._classify_error(exc),
                paper_id=candidate.paper_id,
                provider="document_acquirer",
                url=oa_url,
                metadata={
                    "error": str(exc),
                    "download_source": download_source,
                    "fallback_reason": fallback_reason,
                },
                started_at=acquire_started_at,
            )
            raise

    def _fetch_pdf_artifact(
        self,
        *,
        candidate: PaperCandidate,
        url: str,
        store: QAArtifactStore,
        acquire_started_at: float,
        download_source: str,
    ) -> Tuple[str, str]:
        self._check_total_timeout(
            started_at=acquire_started_at,
            paper_id=candidate.paper_id,
            stage="oa_fetch",
            store=store,
            provider="oa_fetch",
            url=url,
        )
        self._write_runtime_status(
            store=store,
            stage="oa_fetch",
            status="start",
            paper_id=candidate.paper_id,
            provider="oa_fetch",
            url=url,
            metadata={"download_source": download_source},
            started_at=acquire_started_at,
        )
        try:
            fetched = self._run_with_timeout(
                lambda: self.fetcher.fetch(url),
                timeout_seconds=self.document_fetch_timeout_seconds,
            )
            binary = getattr(fetched, "binary", None)
            content_type = str(getattr(fetched, "content_type", "") or "").strip().lower()
            if not binary or not (
                content_type == "application/pdf" or bytes(binary[:5]) == b"%PDF-"
            ):
                raise ValueError(
                    f"paper_id={candidate.paper_id} did not return a usable PDF from {download_source}."
                )
            artifact_path = store.write_bytes(f"fulltext/{candidate.paper_id}.pdf", binary)
            self._record_provider_health_success("oa_fetch")
            self._record_diagnostic(provider="oa_fetch", stage="fetch", outcome="hit")
            resolved_url = str(getattr(fetched, "final_url", None) or getattr(fetched, "url", None) or url)
            self._write_runtime_status(
                store=store,
                stage="oa_fetch",
                status="success",
                paper_id=candidate.paper_id,
                provider="oa_fetch",
                url=url,
                metadata={
                    "download_source": download_source,
                    "final_url": resolved_url,
                    "redirect_count": int(getattr(fetched, "redirect_count", 0) or 0),
                    "content_type": content_type,
                },
                started_at=acquire_started_at,
            )
            return artifact_path, resolved_url
        except Exception as exc:
            self._record_provider_health_failure("oa_fetch", exc)
            if self._classify_error(exc) == "timeout":
                self.last_execution_warnings.append(
                    f"oa_fetch timed out for paper_id={candidate.paper_id} url={url}: {exc}"
                )
            self._record_diagnostic(
                provider="oa_fetch",
                stage="fetch",
                outcome=self._classify_error(exc),
                message=f"paper_id={candidate.paper_id}: {exc}",
            )
            self._write_runtime_status(
                store=store,
                stage="oa_fetch",
                status=self._classify_error(exc),
                paper_id=candidate.paper_id,
                provider="oa_fetch",
                url=url,
                metadata={"error": str(exc), "download_source": download_source},
                started_at=acquire_started_at,
            )
            raise

    __call__ = run

    def _parse_one(
        self,
        *,
        paper_record: PaperRecord,
        store: QAArtifactStore,
    ) -> Tuple[PaperRecord, SectionIndex]:
        source_artifact_path = str(paper_record.source_artifact_path or "").strip()
        fulltext_artifact_path = str(paper_record.fulltext_artifact_path or "").strip()
        fulltext_status = str(paper_record.fulltext_status or "missing").strip() or "missing"

        if fulltext_status == "fulltext_indexed" and fulltext_artifact_path:
            section_index = self._build_section_index(
                paper_id=paper_record.paper_id,
                fulltext_status=fulltext_status,
                fulltext_artifact_path=fulltext_artifact_path,
            )
            index_artifact_path = store.write_json(
                f"indices/{paper_record.paper_id}.json",
                section_index.model_dump(exclude_none=True),
            )
            return paper_record.model_copy(update={"index_artifact_path": index_artifact_path}), section_index

        if source_artifact_path.lower().endswith(".pdf") and Path(source_artifact_path).exists():
            pdf_result = self.pdf_extractor.process(
                paper_id=paper_record.paper_id,
                pdf_bytes=Path(source_artifact_path).read_bytes(),
                artifact_store=store,
            )
            indexed_sections = [
                Section.model_validate(
                    {
                        "section_id": section_payload.get("section_id"),
                        "section_type": section_payload.get("section_type"),
                        "heading": section_payload.get("heading"),
                        "page_start": section_payload.get("page_start"),
                        "page_end": section_payload.get("page_end"),
                        "fulltext_char_start": section_payload.get("fulltext_char_start"),
                        "fulltext_char_end": section_payload.get("fulltext_char_end"),
                    }
                )
                for section_payload in list(pdf_result.sections or [])
            ]
            section_index = self._build_section_index(
                paper_id=paper_record.paper_id,
                fulltext_status=pdf_result.fulltext_status,
                fulltext_artifact_path=pdf_result.fulltext_artifact_path,
                sections=indexed_sections,
            )
            index_artifact_path = store.write_json(
                f"indices/{paper_record.paper_id}.json",
                section_index.model_dump(exclude_none=True),
            )
            self.last_execution_warnings.extend(list(pdf_result.warnings or []))
            return (
                paper_record.model_copy(
                    update={
                        "fulltext_available": True,
                        "fulltext_status": pdf_result.fulltext_status,
                        "fulltext_format": paper_record.fulltext_format or "application/pdf",
                        "fulltext_artifact_path": pdf_result.fulltext_artifact_path,
                        "source_artifact_path": pdf_result.source_artifact_path,
                        "index_artifact_path": index_artifact_path,
                        "extraction_report_path": pdf_result.extraction_report_path,
                        "sections_artifact_path": pdf_result.sections_artifact_path,
                        "snippets_artifact_path": pdf_result.snippets_artifact_path,
                        "extraction_warnings": list(pdf_result.warnings or []),
                        "fulltext_extractor": pdf_result.extractor,
                        "ocr_applied": bool(pdf_result.ocr_applied),
                    }
                ),
                section_index,
            )

        section_index = self._build_section_index(
            paper_id=paper_record.paper_id,
            fulltext_status=fulltext_status,
            fulltext_artifact_path=paper_record.fulltext_artifact_path,
        )
        index_artifact_path = store.write_json(
            f"indices/{paper_record.paper_id}.json",
            section_index.model_dump(exclude_none=True),
        )
        return paper_record.model_copy(update={"index_artifact_path": index_artifact_path}), section_index

    def _acquire_one(
        self,
        candidate: PaperCandidate,
        store: QAArtifactStore,
        *,
        parse_fulltext: bool = True,
    ) -> Tuple[PaperRecord, SectionIndex]:
        provider_artifacts = dict(candidate.provider_artifacts)
        provider_sources = list(candidate.provider_hits)
        oa_url = candidate.oa_url
        fulltext_available = False
        fulltext_format: Optional[str] = None
        fulltext_artifact_path: Optional[str] = None
        source_artifact_path: Optional[str] = None
        extraction_report_path: Optional[str] = None
        sections_artifact_path: Optional[str] = None
        snippets_artifact_path: Optional[str] = None
        extraction_warnings: list[str] = []
        fulltext_extractor: Optional[str] = None
        ocr_applied = False
        fulltext_status = "abstract_only"
        indexed_sections: Optional[list[Section]] = None
        timed_out = False
        acquire_started_at = time.perf_counter()

        self._write_runtime_status(
            store=store,
            stage="download_document",
            status="start",
            paper_id=candidate.paper_id,
            provider="document_acquirer",
            metadata={
                "doi": candidate.doi,
                "oa_url": oa_url,
                "document_fetch_timeout_seconds": self.document_fetch_timeout_seconds,
                "document_fetch_total_timeout_seconds": self.document_fetch_total_timeout_seconds,
            },
        )
        logger.info("document_acquirer_stage_start paper_id=%s stage=%s", candidate.paper_id, "download_document")

        try:
            if self.unpaywall_client and candidate.doi:
                self._check_total_timeout(
                    started_at=acquire_started_at,
                    paper_id=candidate.paper_id,
                    stage="unpaywall_lookup",
                    store=store,
                    provider="unpaywall",
                )
                self._write_runtime_status(
                    store=store,
                    stage="unpaywall_lookup",
                    status="start",
                    paper_id=candidate.paper_id,
                    provider="unpaywall",
                    metadata={"doi": candidate.doi},
                    started_at=acquire_started_at,
                )
                logger.info("document_acquirer_stage_start paper_id=%s stage=%s", candidate.paper_id, "unpaywall_lookup")
                try:
                    unpaywall_payload = self._run_with_timeout(
                        lambda: self.unpaywall_client.lookup(candidate.doi),
                        timeout_seconds=self.document_fetch_timeout_seconds,
                    )
                    self._record_provider_health_success("unpaywall")
                    self._write_runtime_status(
                        store=store,
                        stage="unpaywall_lookup",
                        status="success",
                        paper_id=candidate.paper_id,
                        provider="unpaywall",
                        metadata={"doi": candidate.doi},
                        started_at=acquire_started_at,
                    )
                    logger.info("document_acquirer_stage_success paper_id=%s stage=%s", candidate.paper_id, "unpaywall_lookup")
                except ProviderUnavailableError as exc:
                    logger.warning("unpaywall_lookup_skipped paper_id=%s error=%s", candidate.paper_id, exc)
                    self._record_provider_health_skipped("unpaywall", str(exc))
                    self._record_diagnostic(
                        provider="unpaywall",
                        stage="lookup",
                        outcome="skipped",
                        message=f"paper_id={candidate.paper_id}: {exc}",
                    )
                    self._write_runtime_status(
                        store=store,
                        stage="unpaywall_lookup",
                        status="skipped",
                        paper_id=candidate.paper_id,
                        provider="unpaywall",
                        metadata={"error": str(exc), "doi": candidate.doi},
                        started_at=acquire_started_at,
                    )
                    unpaywall_payload = None
                except Exception as exc:
                    logger.warning("unpaywall_lookup_failed paper_id=%s error=%s", candidate.paper_id, exc)
                    self._record_provider_health_failure("unpaywall", exc)
                    self._record_diagnostic(
                        provider="unpaywall",
                        stage="lookup",
                        outcome=self._classify_error(exc),
                        message=f"paper_id={candidate.paper_id}: {exc}",
                    )
                    self._write_runtime_status(
                        store=store,
                        stage="unpaywall_lookup",
                        status=self._classify_error(exc),
                        paper_id=candidate.paper_id,
                        provider="unpaywall",
                        metadata={"error": str(exc), "doi": candidate.doi},
                        started_at=acquire_started_at,
                    )
                    unpaywall_payload = None
                if unpaywall_payload:
                    self._record_diagnostic(provider="unpaywall", stage="lookup", outcome="hit")
                    provider_sources = list(dict.fromkeys([*provider_sources, "unpaywall"]))
                    provider_artifacts["unpaywall"] = store.write_json(
                        f"provider_raw/unpaywall/{candidate.paper_id}.json",
                        unpaywall_payload,
                    )
                    best_location = unpaywall_payload.get("best_oa_location") or {}
                    oa_url = (
                        best_location.get("url_for_pdf")
                        or best_location.get("url_for_landing_page")
                        or best_location.get("url")
                        or oa_url
                    )
                else:
                    self._record_diagnostic(provider="unpaywall", stage="lookup", outcome="empty")

            if oa_url:
                self._check_total_timeout(
                    started_at=acquire_started_at,
                    paper_id=candidate.paper_id,
                    stage="oa_fetch",
                    store=store,
                    provider="oa_fetch",
                    url=oa_url,
                )
                self._write_runtime_status(
                    store=store,
                    stage="oa_fetch",
                    status="start",
                    paper_id=candidate.paper_id,
                    provider="oa_fetch",
                    url=oa_url,
                    started_at=acquire_started_at,
                )
                logger.info("document_acquirer_stage_start paper_id=%s stage=%s", candidate.paper_id, "oa_fetch")
                try:
                    fetched = self._run_with_timeout(
                        lambda: self.fetcher.fetch(oa_url),
                        timeout_seconds=self.document_fetch_timeout_seconds,
                    )
                    self._record_provider_health_success("oa_fetch")
                    self._write_runtime_status(
                        store=store,
                        stage="oa_fetch",
                        status="success",
                        paper_id=candidate.paper_id,
                        provider="oa_fetch",
                        url=oa_url,
                        metadata={
                            "final_url": getattr(fetched, "final_url", None) or getattr(fetched, "url", None),
                            "redirect_count": int(getattr(fetched, "redirect_count", 0) or 0),
                            "content_type": fetched.content_type,
                        },
                        started_at=acquire_started_at,
                    )
                    logger.info("document_acquirer_stage_success paper_id=%s stage=%s", candidate.paper_id, "oa_fetch")
                except Exception as exc:
                    logger.warning("oa_fetch_failed paper_id=%s url=%s error=%s", candidate.paper_id, oa_url, exc)
                    self._record_provider_health_failure("oa_fetch", exc)
                    if self._classify_error(exc) == "timeout":
                        self.last_execution_warnings.append(
                            f"oa_fetch timed out for paper_id={candidate.paper_id} url={oa_url}: {exc}"
                        )
                    self._record_diagnostic(
                        provider="oa_fetch",
                        stage="fetch",
                        outcome=self._classify_error(exc),
                        message=f"paper_id={candidate.paper_id}: {exc}",
                    )
                    self._write_runtime_status(
                        store=store,
                        stage="oa_fetch",
                        status=self._classify_error(exc),
                        paper_id=candidate.paper_id,
                        provider="oa_fetch",
                        url=oa_url,
                        metadata={
                            "error": str(exc),
                            "attempts": getattr(exc, "attempts", None),
                            "status_code": getattr(exc, "status_code", None),
                        },
                        started_at=acquire_started_at,
                    )
                    fetched = None
                if fetched is not None:
                    self._record_diagnostic(provider="oa_fetch", stage="fetch", outcome="hit")
                    fulltext_format = fetched.content_type
                    if fetched.content_type == "application/pdf" and fetched.binary is not None:
                        if not parse_fulltext:
                            source_artifact_path = store.write_bytes(f"fulltext/{candidate.paper_id}.pdf", fetched.binary)
                            fulltext_artifact_path = source_artifact_path
                            fulltext_available = True
                            fulltext_status = "binary_only"
                            fulltext_format = fetched.content_type
                            indexed_sections = []
                        else:
                            self._check_total_timeout(
                                started_at=acquire_started_at,
                                paper_id=candidate.paper_id,
                                stage="pdf_extraction",
                                store=store,
                                provider="pdf_extraction",
                                url=getattr(fetched, "final_url", None) or fetched.url,
                            )
                            self._write_runtime_status(
                                store=store,
                                stage="pdf_extraction",
                                status="start",
                                paper_id=candidate.paper_id,
                                provider="pdf_extraction",
                                url=getattr(fetched, "final_url", None) or fetched.url,
                                metadata={"content_type": fetched.content_type},
                                started_at=acquire_started_at,
                            )
                            logger.info("document_acquirer_stage_start paper_id=%s stage=%s", candidate.paper_id, "pdf_extraction")
                            pdf_result = self._run_with_timeout(
                                lambda: self.pdf_extractor.process(
                                    paper_id=candidate.paper_id,
                                    pdf_bytes=fetched.binary,
                                    artifact_store=store,
                                ),
                                timeout_seconds=self.document_fetch_total_timeout_seconds,
                            )
                            self._write_runtime_status(
                                store=store,
                                stage="pdf_extraction",
                                status="success",
                                paper_id=candidate.paper_id,
                                provider="pdf_extraction",
                                url=getattr(fetched, "final_url", None) or fetched.url,
                                metadata={"fulltext_status": pdf_result.fulltext_status, "extractor": pdf_result.extractor},
                                started_at=acquire_started_at,
                            )
                            logger.info("document_acquirer_stage_success paper_id=%s stage=%s", candidate.paper_id, "pdf_extraction")
                            source_artifact_path = pdf_result.source_artifact_path
                            fulltext_artifact_path = pdf_result.fulltext_artifact_path
                            extraction_report_path = pdf_result.extraction_report_path
                            sections_artifact_path = pdf_result.sections_artifact_path
                            snippets_artifact_path = pdf_result.snippets_artifact_path
                            extraction_warnings = list(pdf_result.warnings)
                            fulltext_extractor = pdf_result.extractor
                            ocr_applied = bool(pdf_result.ocr_applied)
                            fulltext_available = True
                            fulltext_status = pdf_result.fulltext_status
                            if pdf_result.sections:
                                indexed_sections = [
                                    Section.model_validate(
                                        {
                                            "section_id": section_payload.get("section_id"),
                                            "section_type": section_payload.get("section_type"),
                                            "heading": section_payload.get("heading"),
                                            "page_start": section_payload.get("page_start"),
                                            "page_end": section_payload.get("page_end"),
                                            "fulltext_char_start": section_payload.get("fulltext_char_start"),
                                            "fulltext_char_end": section_payload.get("fulltext_char_end"),
                                        }
                                    )
                                    for section_payload in pdf_result.sections
                                ]
                            self.last_execution_warnings.extend(extraction_warnings)
                    elif fetched.binary is not None:
                        fulltext_artifact_path = store.write_bytes(
                            f"fulltext/{candidate.paper_id}.{guess_binary_extension(fetched.content_type)}",
                            fetched.binary,
                        )
                        source_artifact_path = fulltext_artifact_path
                        fulltext_available = True
                        fulltext_status = "binary_only"
                    else:
                        raw_text = fetched.text or ""
                        fulltext = self._normalize_fulltext(raw_text, fetched.content_type)
                        if fulltext:
                            fulltext_artifact_path = store.write_text(
                                f"fulltext/{candidate.paper_id}.txt",
                                fulltext,
                            )
                            source_artifact_path = fulltext_artifact_path
                            fulltext_available = True
                            fulltext_status = "fulltext_indexed"
        except TimeoutError as exc:
            timed_out = True
            warning_text = f"document acquisition timed out for paper_id={candidate.paper_id}: {exc}"
            logger.warning("document_acquirer_timeout paper_id=%s error=%s", candidate.paper_id, exc)
            self.last_execution_warnings.append(warning_text)
            self._record_diagnostic(
                provider="document_acquirer",
                stage="acquire",
                outcome="timeout",
                message=warning_text,
            )
            self._write_runtime_status(
                store=store,
                stage="download_document",
                status="timeout",
                paper_id=candidate.paper_id,
                provider="document_acquirer",
                url=oa_url,
                metadata={"error": str(exc)},
                started_at=acquire_started_at,
            )
        except Exception as exc:
            self._write_runtime_status(
                store=store,
                stage="download_document",
                status="failure",
                paper_id=candidate.paper_id,
                provider="document_acquirer",
                url=oa_url,
                metadata={"error": str(exc)},
                started_at=acquire_started_at,
            )
            raise

        if not timed_out:
            self._write_runtime_status(
                store=store,
                stage="section_indexing",
                status="start",
                paper_id=candidate.paper_id,
                provider="section_indexing",
                metadata={"fulltext_status": fulltext_status},
                started_at=acquire_started_at,
            )
        section_index = self._build_section_index(
            paper_id=candidate.paper_id,
            fulltext_status=fulltext_status,
            fulltext_artifact_path=fulltext_artifact_path,
            sections=indexed_sections,
        )
        if not timed_out:
            self._write_runtime_status(
                store=store,
                stage="section_indexing",
                status="success",
                paper_id=candidate.paper_id,
                provider="section_indexing",
                metadata={"fulltext_status": fulltext_status, "section_count": len(section_index.sections)},
                started_at=acquire_started_at,
            )
        index_artifact_path = store.write_json(
            f"indices/{candidate.paper_id}.json",
            section_index.model_dump(exclude_none=True),
        )

        paper_record = PaperRecord(
            paper_id=candidate.paper_id,
            doi=candidate.doi,
            title=candidate.title,
            abstract=candidate.abstract,
            authors=candidate.authors,
            year=candidate.year,
            venue=candidate.venue,
            provider_sources=provider_sources,
            provider_artifacts=provider_artifacts,
            oa_url=oa_url,
            fulltext_available=fulltext_available,
            fulltext_status=fulltext_status,
            fulltext_format=fulltext_format,
            fulltext_artifact_path=fulltext_artifact_path,
            source_artifact_path=source_artifact_path,
            index_artifact_path=index_artifact_path,
            extraction_report_path=extraction_report_path,
            sections_artifact_path=sections_artifact_path,
            snippets_artifact_path=snippets_artifact_path,
            extraction_warnings=extraction_warnings,
            fulltext_extractor=fulltext_extractor,
            ocr_applied=ocr_applied,
        )
        if not timed_out:
            self._write_runtime_status(
                store=store,
                stage="download_document",
                status="success",
                paper_id=candidate.paper_id,
                provider="document_acquirer",
                url=oa_url,
                metadata={
                    "fulltext_status": fulltext_status,
                    "fulltext_available": fulltext_available,
                    "section_count": len(section_index.sections),
                },
                started_at=acquire_started_at,
            )
            logger.info("document_acquirer_stage_success paper_id=%s stage=%s", candidate.paper_id, "download_document")
        return paper_record, section_index

    def _normalize_fulltext(self, raw_text: str, content_type: Optional[str]) -> str:
        if not raw_text:
            return ""
        if (content_type or "").lower() == "text/html":
            raw_text = self._html_to_text(raw_text)
        normalized = raw_text.replace("\r\n", "\n").replace("\r", "\n")
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        normalized = normalized.strip()
        if not normalized:
            return ""
        if looks_like_placeholder_text(normalized):
            return ""
        if looks_like_garbled_text(normalized):
            return ""
        return normalized

    def _html_to_text(self, raw_html: str) -> str:
        without_scripts = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", raw_html)
        without_tags = re.sub(r"(?s)<[^>]+>", " ", without_scripts)
        unescaped = html.unescape(without_tags)
        return re.sub(r"[ \t]+", " ", unescaped)

    def _build_section_index(
        self,
        paper_id: str,
        fulltext_status: str,
        fulltext_artifact_path: Optional[str],
        sections: Optional[Sequence[Section]] = None,
    ) -> SectionIndex:
        if fulltext_status != "fulltext_indexed" or not fulltext_artifact_path:
            return SectionIndex(paper_id=paper_id, fulltext_status=fulltext_status, sections=[])
        if sections is not None:
            return SectionIndex(paper_id=paper_id, fulltext_status=fulltext_status, sections=list(sections))

        fulltext = Path(fulltext_artifact_path).read_text(encoding="utf-8")
        matches = list(SECTION_HEADING_PATTERN.finditer(fulltext))
        if not matches:
            return SectionIndex(
                paper_id=paper_id,
                fulltext_status=fulltext_status,
                sections=[
                    Section(
                        section_id="sec_0_unknown",
                        section_type="unknown",
                        heading="Body",
                        fulltext_char_start=0,
                        fulltext_char_end=len(fulltext),
                    )
                ],
            )

        sections: List[Section] = []
        for index, match in enumerate(matches):
            heading = match.group(1)
            section_type = SECTION_TYPE_MAP.get(heading.strip().lower(), "unknown")
            content_start = match.end()
            content_end = matches[index + 1].start() if index + 1 < len(matches) else len(fulltext)
            sections.append(
                Section(
                    section_id=f"sec_{index}_{section_type}",
                    section_type=section_type,
                    heading=heading.strip(),
                    fulltext_char_start=content_start,
                    fulltext_char_end=content_end,
                )
            )
        return SectionIndex(paper_id=paper_id, fulltext_status=fulltext_status, sections=sections)

    def _record_diagnostic(
        self,
        *,
        provider: str,
        stage: str,
        outcome: str,
        count: int = 1,
        message: Optional[str] = None,
    ) -> None:
        key = (provider, stage, "")
        record = self._diagnostic_map.setdefault(
            key,
            {
                "provider": provider,
                "stage": stage,
                "lane": None,
                "hit_count": 0,
                "failure_count": 0,
                "timeout_count": 0,
                "skipped_count": 0,
                "empty_count": 0,
                "sample_messages": [],
            },
        )
        field_name = f"{outcome}_count"
        if field_name in record:
            record[field_name] += max(1, int(count))
        if message:
            cleaned = normalize_text(message)
            if cleaned and cleaned not in record["sample_messages"] and len(record["sample_messages"]) < 3:
                record["sample_messages"].append(cleaned[:240])

    def _init_provider_health(self) -> dict[str, dict[str, Any]]:
        return {
            "unpaywall": {
                "status": "idle" if self.unpaywall_client is not None else "disabled",
                "calls": 0,
                "successes": 0,
                "retry_exhausted_failures": 0,
                "skipped_calls": 0,
                "last_error": None,
            },
            "oa_fetch": {
                "status": "idle" if self.fetcher is not None else "disabled",
                "calls": 0,
                "successes": 0,
                "retry_exhausted_failures": 0,
                "skipped_calls": 0,
                "last_error": None,
            },
        }

    def _record_provider_health_success(self, provider: str) -> None:
        record = self._provider_health_fallback[provider]
        record["calls"] += 1
        record["successes"] += 1
        record["status"] = "healthy"
        record["last_error"] = None

    def _record_provider_health_failure(self, provider: str, exc: Exception) -> None:
        record = self._provider_health_fallback[provider]
        record["calls"] += 1
        record["last_error"] = str(exc)
        if isinstance(exc, ProviderRequestError) and exc.retry_exhausted:
            record["retry_exhausted_failures"] += 1
            record["status"] = "unavailable" if exc.provider_unavailable else "degraded"
            return
        if record["status"] == "idle":
            record["status"] = "degraded"

    def _record_provider_health_skipped(self, provider: str, message: str) -> None:
        record = self._provider_health_fallback[provider]
        record["skipped_calls"] += 1
        record["status"] = "unavailable"
        record["last_error"] = message

    def _collect_provider_health(self) -> dict[str, dict[str, Any]]:
        collected = {
            provider_name: dict(payload)
            for provider_name, payload in self._provider_health_fallback.items()
        }
        for provider_name, client in (
            ("unpaywall", self.unpaywall_client),
            ("oa_fetch", self.fetcher),
        ):
            snapshot = self._client_health_snapshot(client)
            if snapshot is not None:
                collected[provider_name] = snapshot
        return collected

    def _client_health_snapshot(self, client: Any) -> Optional[dict[str, Any]]:
        if client is None:
            return None
        snapshot_fn = getattr(client, "health_snapshot", None)
        if callable(snapshot_fn):
            snapshot = snapshot_fn()
            if isinstance(snapshot, dict):
                return dict(snapshot)
        return None

    def _finalize_diagnostics(self) -> List[RetrievalDiagnosticRecord]:
        diagnostics: List[RetrievalDiagnosticRecord] = []
        for key in sorted(self._diagnostic_map):
            diagnostics.append(RetrievalDiagnosticRecord.model_validate(self._diagnostic_map[key]))
        return diagnostics

    def _classify_error(self, exc: Exception) -> str:
        if isinstance(exc, ProviderUnavailableError):
            return "skipped"
        if isinstance(exc, ProviderRequestError):
            return exc.failure_kind
        message = str(exc or "").lower()
        if "timeout" in message or "timed out" in message:
            return "timeout"
        return "failure"

    def _run_with_timeout(self, operation: Any, *, timeout_seconds: float) -> Any:
        if timeout_seconds <= 0:
            return operation()
        result_holder: dict[str, Any] = {}
        error_holder: dict[str, BaseException] = {}
        done = threading.Event()

        def _target() -> None:
            try:
                result_holder["value"] = operation()
            except BaseException as exc:  # pragma: no cover - thread plumbing
                error_holder["error"] = exc
            finally:
                done.set()

        thread = threading.Thread(target=_target, daemon=True)
        thread.start()
        if not done.wait(timeout_seconds):
            raise TimeoutError(f"timed out after {round(float(timeout_seconds), 2)}s")
        if "error" in error_holder:
            raise error_holder["error"]
        return result_holder.get("value")

    def _write_runtime_status(
        self,
        *,
        store: QAArtifactStore,
        stage: str,
        status: str,
        paper_id: str,
        provider: str,
        url: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        started_at: Optional[float] = None,
    ) -> None:
        payload = {
            "paper_id": str(paper_id),
            "stage": str(stage),
            "status": str(status),
            "provider": str(provider),
            "url": str(url or "").strip() or None,
            "timestamp": round(time.time(), 6),
        }
        if started_at is not None:
            payload["elapsed_total_seconds"] = round(max(0.0, time.perf_counter() - float(started_at)), 3)
        if metadata:
            payload.update({key: value for key, value in dict(metadata).items() if value is not None})
        self.last_runtime_status = dict(payload)
        store.write_json("diagnostics/document_acquirer_runtime.json", payload)

    def _check_total_timeout(
        self,
        *,
        started_at: float,
        paper_id: str,
        stage: str,
        store: QAArtifactStore,
        provider: str,
        url: Optional[str] = None,
    ) -> None:
        elapsed = max(0.0, time.perf_counter() - float(started_at))
        if elapsed <= self.document_fetch_total_timeout_seconds:
            return
        self._write_runtime_status(
            store=store,
            stage=stage,
            status="timeout",
            paper_id=paper_id,
            provider=provider,
            url=url,
            metadata={
                "error": (
                    f"document acquisition exceeded total timeout "
                    f"after {round(elapsed, 3)}s (limit {self.document_fetch_total_timeout_seconds}s)"
                )
            },
            started_at=started_at,
        )
        raise TimeoutError(
            f"document acquisition exceeded total timeout after {round(elapsed, 3)}s "
            f"(limit {self.document_fetch_total_timeout_seconds}s)"
        )
