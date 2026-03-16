from __future__ import annotations

import html
import logging
import re
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

from qa.artifacts import QAArtifactStore
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
    ) -> None:
        self.unpaywall_client = unpaywall_client
        self.fetcher = fetcher or HttpTextFetcher()
        self.last_diagnostics: List[RetrievalDiagnosticRecord] = []
        self.last_provider_health: dict[str, dict[str, Any]] = {}
        self._diagnostic_map: dict[tuple[str, str, str], dict[str, Any]] = {}
        self._provider_health_fallback: dict[str, dict[str, Any]] = {}

    def run(
        self,
        candidates: Sequence[PaperCandidate],
        artifact_store: Optional[QAArtifactStore] = None,
    ) -> Tuple[List[PaperRecord], List[SectionIndex]]:
        store = artifact_store or QAArtifactStore()
        paper_records: List[PaperRecord] = []
        section_indices: List[SectionIndex] = []
        self._diagnostic_map = {}
        self._provider_health_fallback = self._init_provider_health()

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
            paper_record, section_index = self._acquire_one(candidate=candidate, store=store)
            paper_records.append(paper_record)
            section_indices.append(section_index)

        self.last_diagnostics = self._finalize_diagnostics()
        self.last_provider_health = self._collect_provider_health()
        return paper_records, section_indices

    __call__ = run

    def _acquire_one(self, candidate: PaperCandidate, store: QAArtifactStore) -> Tuple[PaperRecord, SectionIndex]:
        provider_artifacts = dict(candidate.provider_artifacts)
        provider_sources = list(candidate.provider_hits)
        oa_url = candidate.oa_url
        fulltext_available = False
        fulltext_format: Optional[str] = None
        fulltext_artifact_path: Optional[str] = None
        fulltext_status = "abstract_only"

        if self.unpaywall_client and candidate.doi:
            try:
                unpaywall_payload = self.unpaywall_client.lookup(candidate.doi)
                self._record_provider_health_success("unpaywall")
            except ProviderUnavailableError as exc:
                logger.warning("unpaywall_lookup_skipped paper_id=%s error=%s", candidate.paper_id, exc)
                self._record_provider_health_skipped("unpaywall", str(exc))
                self._record_diagnostic(
                    provider="unpaywall",
                    stage="lookup",
                    outcome="skipped",
                    message=f"paper_id={candidate.paper_id}: {exc}",
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
            try:
                fetched = self.fetcher.fetch(oa_url)
                self._record_provider_health_success("oa_fetch")
            except Exception as exc:
                logger.warning("oa_fetch_failed paper_id=%s url=%s error=%s", candidate.paper_id, oa_url, exc)
                self._record_provider_health_failure("oa_fetch", exc)
                self._record_diagnostic(
                    provider="oa_fetch",
                    stage="fetch",
                    outcome=self._classify_error(exc),
                    message=f"paper_id={candidate.paper_id}: {exc}",
                )
                fetched = None
            if fetched is not None:
                self._record_diagnostic(provider="oa_fetch", stage="fetch", outcome="hit")
                fulltext_format = fetched.content_type
                if fetched.content_type == "application/pdf" and fetched.binary is not None:
                    fulltext_artifact_path = store.write_bytes(
                        f"fulltext/{candidate.paper_id}.pdf",
                        fetched.binary,
                    )
                    fulltext_available = True
                    fulltext_status = "binary_only"
                elif fetched.binary is not None:
                    fulltext_artifact_path = store.write_bytes(
                        f"fulltext/{candidate.paper_id}.{guess_binary_extension(fetched.content_type)}",
                        fetched.binary,
                    )
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
                        fulltext_available = True
                        fulltext_status = "fulltext_indexed"

        section_index = self._build_section_index(
            paper_id=candidate.paper_id,
            fulltext_status=fulltext_status,
            fulltext_artifact_path=fulltext_artifact_path,
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
            fulltext_format=fulltext_format,
            fulltext_artifact_path=fulltext_artifact_path,
            index_artifact_path=index_artifact_path,
        )
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
    ) -> SectionIndex:
        if fulltext_status != "fulltext_indexed" or not fulltext_artifact_path:
            return SectionIndex(paper_id=paper_id, fulltext_status=fulltext_status, sections=[])

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
