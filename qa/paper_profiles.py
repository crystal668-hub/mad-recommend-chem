from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence
from urllib.error import URLError
from urllib.request import urlopen

from qa.artifacts import QAArtifactStore
from qa.retrieval_state import PaperProfile, PaperRecord


SECTION_HEADING_PATTERN = re.compile(
    r"(?mi)^(?:\d+(?:\.\d+)*\s+)?(abstract|introduction|background|materials?\s+and\s+methods|methods?|experimental|results?|discussion|conclusions?|limitations?)\s*$"
)
METRIC_PATTERN = re.compile(
    r"\b(?:overpotential|tafel|faradaic efficiency|current density|exchange current|yield|selectivity|stability|durability)\b",
    re.I,
)
ENTITY_PATTERN = re.compile(r"\b(?:Pt/C|Pt|NiMo|Ru|IrO2|CO2RR|HER|OER|ORR|KOH|NaOH|electrolyte)\b", re.I)


def _compact_text(value: Any) -> str:
    return " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split()).strip()


def _lazy_grobid_loader_imports():
    try:
        from langchain_community.document_loaders.generic import GenericLoader
        from langchain_community.document_loaders.parsers import GrobidParser
    except Exception as exc:  # pragma: no cover - exercised through integration environments
        raise RuntimeError(
            "GROBID profile extraction requires langchain-community with GenericLoader and GrobidParser."
        ) from exc
    return GenericLoader, GrobidParser


class GrobidPaperProfileBuilder:
    def __init__(
        self,
        *,
        grobid_url: str = "http://localhost:8070",
        loader_factory: Optional[Callable[[Path], Any]] = None,
        max_summary_chars: int = 1200,
    ) -> None:
        self.grobid_url = str(grobid_url or "http://localhost:8070").strip() or "http://localhost:8070"
        self.loader_factory = loader_factory
        self.max_summary_chars = max(200, int(max_summary_chars or 1200))

    def build(self, *, paper_record: PaperRecord, artifact_store: Optional[QAArtifactStore] = None) -> PaperProfile:
        store = artifact_store or QAArtifactStore()
        source_pdf_path = self._resolve_local_pdf_artifact_path(paper_record)
        local_text_path = self._resolve_local_text_artifact_path(paper_record)
        grobid_error: Optional[Exception] = None
        if source_pdf_path is not None:
            try:
                grobid_profile = self._build_from_grobid(
                    paper_record=paper_record,
                    source_path=source_pdf_path,
                    artifact_store=store,
                )
                if local_text_path is None:
                    return grobid_profile
                local_profile = self._build_from_local_text(
                    paper_record=paper_record,
                    text_path=local_text_path,
                    artifact_store=store,
                )
                return self._merge_profiles(
                    primary=grobid_profile,
                    supplement=local_profile,
                    artifact_store=store,
                )
            except Exception as exc:
                grobid_error = exc

        if local_text_path is not None:
            return self._build_from_local_text(
                paper_record=paper_record,
                text_path=local_text_path,
                artifact_store=store,
            )

        if grobid_error is not None:
            raise grobid_error

        source_artifact_path = str(paper_record.source_artifact_path or "").strip()
        if not source_artifact_path:
            raise ValueError(f"paper_id={paper_record.paper_id} has no source_artifact_path for GROBID parsing.")
        raise ValueError(f"paper_id={paper_record.paper_id} does not have a PDF source artifact for GROBID parsing.")

    def _resolve_local_pdf_artifact_path(self, paper_record: PaperRecord) -> Optional[Path]:
        candidates = [
            str(paper_record.source_artifact_path or "").strip(),
            str(paper_record.fulltext_artifact_path or "").strip(),
        ]
        for candidate_path in candidates:
            if not candidate_path:
                continue
            path = Path(candidate_path)
            if not path.exists():
                continue
            if path.suffix.lower() == ".pdf":
                return path
        return None

    def _build_from_grobid(
        self,
        *,
        paper_record: PaperRecord,
        source_path: Path,
        artifact_store: QAArtifactStore,
    ) -> PaperProfile:
        self._assert_grobid_available()
        loader = self._build_loader(source_path)
        documents = list(loader.load() or [])
        if not documents:
            raise ValueError(f"paper_id={paper_record.paper_id} produced no GROBID documents.")

        raw_payload = []
        for index, document in enumerate(documents):
            raw_payload.append(
                {
                    "index": index,
                    "page_content": getattr(document, "page_content", ""),
                    "metadata": dict(getattr(document, "metadata", {}) or {}),
                }
            )
        return self._build_profile_from_payload(
            paper_record=paper_record,
            raw_payload=raw_payload,
            artifact_store=artifact_store,
            parser_name="langchain_community.GenericLoader+GrobidParser",
            raw_artifact_name=f"proposer_profiles/{paper_record.paper_id}.grobid_raw.json",
        )

    def _resolve_local_text_artifact_path(self, paper_record: PaperRecord) -> Optional[Path]:
        candidates = [
            str(paper_record.fulltext_artifact_path or "").strip(),
            str(paper_record.source_artifact_path or "").strip(),
        ]
        for candidate_path in candidates:
            if not candidate_path:
                continue
            path = Path(candidate_path)
            if not path.exists():
                continue
            if path.suffix.lower() in {".txt", ".md", ".text"}:
                return path
        return None

    def _build_from_local_text(
        self,
        *,
        paper_record: PaperRecord,
        text_path: Path,
        artifact_store: QAArtifactStore,
    ) -> PaperProfile:
        combined_text = text_path.read_text(encoding="utf-8", errors="ignore").strip()
        if not combined_text:
            raise ValueError(f"paper_id={paper_record.paper_id} produced no usable indexed full-text.")
        raw_payload = [
            {
                "index": 0,
                "page_content": combined_text,
                "metadata": {
                    "source_path": str(text_path),
                    "source_kind": "local_fulltext_text",
                },
            }
        ]
        return self._build_profile_from_payload(
            paper_record=paper_record,
            raw_payload=raw_payload,
            artifact_store=artifact_store,
            parser_name="local_fulltext_text",
            raw_artifact_name=f"proposer_profiles/{paper_record.paper_id}.local_raw.json",
        )

    def _merge_profiles(
        self,
        *,
        primary: PaperProfile,
        supplement: PaperProfile,
        artifact_store: QAArtifactStore,
    ) -> PaperProfile:
        merged_section_headings = self._merge_text_lists(primary.section_headings, supplement.section_headings)
        merged_entities = self._merge_text_lists(primary.materials_or_entities, supplement.materials_or_entities)
        merged_metrics = self._merge_text_lists(primary.reported_metrics, supplement.reported_metrics)
        merged_evidence_sections = self._merge_text_lists(
            primary.evidence_rich_sections,
            supplement.evidence_rich_sections,
        ) or self._evidence_rich_sections(merged_section_headings)
        merged_methods = self._merge_methods(primary=primary, supplement=supplement)
        merged_summary = self._prefer_text(primary.abstract_or_summary, supplement.abstract_or_summary)
        merged_problem = self._prefer_text(primary.problem_or_task, supplement.problem_or_task)
        merged_limitations = self._profile_limitations(merged_section_headings, merged_methods, merged_metrics)
        merged_parser_name = f"{primary.parser_name}+{supplement.parser_name}"
        merged_profile = primary.model_copy(
            update={
                "parser_name": merged_parser_name,
                "abstract_or_summary": merged_summary,
                "section_headings": merged_section_headings,
                "problem_or_task": merged_problem,
                "materials_or_entities": merged_entities,
                "methods_or_experimental_setup": merged_methods,
                "reported_metrics": merged_metrics,
                "evidence_rich_sections": merged_evidence_sections,
                "citation_readiness_summary": self._citation_readiness_summary(
                    merged_section_headings,
                    merged_metrics,
                    merged_evidence_sections,
                ),
                "limitations": merged_limitations,
            }
        )
        parser_artifact_path = artifact_store.write_json(
            f"proposer_profiles/{primary.paper_id}.profile.json",
            merged_profile.model_dump(exclude_none=True),
        )
        return merged_profile.model_copy(update={"parser_artifact_path": parser_artifact_path})

    def _prefer_text(self, primary_value: Optional[str], supplement_value: Optional[str]) -> Optional[str]:
        primary_text = _compact_text(primary_value)
        if primary_text:
            return primary_text
        supplement_text = _compact_text(supplement_value)
        return supplement_text or None

    def _merge_text_lists(self, primary_values: Sequence[str], supplement_values: Sequence[str]) -> List[str]:
        merged: List[str] = []
        seen = set()
        for value in [*list(primary_values or []), *list(supplement_values or [])]:
            cleaned = _compact_text(value)
            if not cleaned:
                continue
            normalized = cleaned.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            merged.append(cleaned)
        return merged

    def _merge_methods(self, *, primary: PaperProfile, supplement: PaperProfile) -> Optional[str]:
        primary_has_method_heading = any(
            term in heading.lower()
            for heading in list(primary.section_headings or [])
            for term in ("method", "experimental", "materials and methods")
        )
        if primary_has_method_heading:
            return self._prefer_text(primary.methods_or_experimental_setup, supplement.methods_or_experimental_setup)
        return self._prefer_text(supplement.methods_or_experimental_setup, primary.methods_or_experimental_setup)

    def _assert_grobid_available(self) -> None:
        if self.loader_factory is not None:
            return
        grobid_health_url = f"{self.grobid_url.rstrip('/')}/api/isalive"
        try:
            with urlopen(grobid_health_url, timeout=2.0) as response:
                if getattr(response, "status", 200) >= 500:
                    raise RuntimeError(f"GROBID health check returned HTTP {response.status}.")
        except (URLError, TimeoutError, RuntimeError) as exc:
            raise RuntimeError(
                f"GROBID server unavailable at {self.grobid_url}; cannot parse PDF-only paper profiles."
            ) from exc

    def _build_profile_from_payload(
        self,
        *,
        paper_record: PaperRecord,
        raw_payload: Sequence[Dict[str, Any]],
        artifact_store: QAArtifactStore,
        parser_name: str,
        raw_artifact_name: str,
    ) -> PaperProfile:
        text_fragments = [
            str(item.get("page_content") or "").replace("\r\n", "\n").replace("\r", "\n").strip()
            for item in raw_payload
            if str(item.get("page_content") or "").strip()
        ]
        combined_text = "\n\n".join(fragment for fragment in text_fragments if fragment).strip()
        if not combined_text:
            raise ValueError(f"paper_id={paper_record.paper_id} produced no usable parsed text.")

        raw_artifact_path = artifact_store.write_json(raw_artifact_name, list(raw_payload))
        source_artifact_path = _compact_text(paper_record.source_artifact_path) or _compact_text(paper_record.fulltext_artifact_path) or None
        section_headings = self._section_headings(raw_payload, combined_text)
        summary = self._abstract_or_summary(raw_payload, combined_text, fallback=paper_record.abstract)
        methods = self._section_summary(raw_payload, combined_text, target_terms=("method", "experimental"))
        problem = self._problem_or_task(raw_payload, combined_text, title=paper_record.title)
        metrics = self._reported_metrics(combined_text)
        entities = self._materials_or_entities(combined_text, title=paper_record.title)
        evidence_rich_sections = self._evidence_rich_sections(section_headings)
        limitations = self._profile_limitations(section_headings, methods, metrics)
        profile = PaperProfile(
            paper_id=paper_record.paper_id,
            title=paper_record.title,
            doi=paper_record.doi,
            year=paper_record.year,
            venue=paper_record.venue,
            oa_source_url=_compact_text(paper_record.oa_url) or None,
            source_artifact_path=source_artifact_path,
            parser_name=parser_name,
            profile_status="ready",
            abstract_or_summary=summary,
            section_headings=section_headings,
            problem_or_task=problem,
            materials_or_entities=entities,
            methods_or_experimental_setup=methods,
            reported_metrics=metrics,
            evidence_rich_sections=evidence_rich_sections,
            citation_readiness_summary=self._citation_readiness_summary(section_headings, metrics, evidence_rich_sections),
            limitations=limitations,
            raw_artifact_path=raw_artifact_path,
        )
        parser_artifact_path = artifact_store.write_json(
            f"proposer_profiles/{paper_record.paper_id}.profile.json",
            profile.model_dump(exclude_none=True),
        )
        return profile.model_copy(update={"parser_artifact_path": parser_artifact_path})

    def _build_loader(self, source_path: Path) -> Any:
        if self.loader_factory is not None:
            return self.loader_factory(source_path)
        GenericLoader, GrobidParser = _lazy_grobid_loader_imports()
        parser = None
        parser_attempts = (
            {"segment_sentences": False, "grobid_url": self.grobid_url},
            {"segment_sentences": False, "grobid_server": self.grobid_url},
            {"segment_sentences": False},
            {},
        )
        for kwargs in parser_attempts:
            try:
                parser = GrobidParser(**kwargs)
                break
            except TypeError:
                continue
        if parser is None:
            parser = GrobidParser()
        from_filesystem = getattr(GenericLoader, "from_filesystem", None)
        if not callable(from_filesystem):
            raise RuntimeError("GenericLoader.from_filesystem is unavailable for GROBID profile extraction.")
        return from_filesystem(str(source_path.parent), glob=source_path.name, parser=parser)

    def _section_headings(self, raw_payload: Sequence[Dict[str, Any]], combined_text: str) -> List[str]:
        headings: List[str] = []
        for item in list(raw_payload or []):
            metadata = dict(item.get("metadata") or {})
            for key in ("section_title", "section", "header", "title"):
                candidate = _compact_text(metadata.get(key))
                if candidate and candidate.lower() not in {value.lower() for value in headings}:
                    headings.append(candidate)
        for match in SECTION_HEADING_PATTERN.finditer(combined_text):
            heading = _compact_text(match.group(1)).title()
            if heading and heading.lower() not in {value.lower() for value in headings}:
                headings.append(heading)
        return headings[:12]

    def _abstract_or_summary(
        self,
        raw_payload: Sequence[Dict[str, Any]],
        combined_text: str,
        *,
        fallback: Optional[str],
    ) -> str:
        for item in list(raw_payload or []):
            metadata = dict(item.get("metadata") or {})
            if "abstract" in _compact_text(metadata.get("section_title") or metadata.get("section")).lower():
                text = _compact_text(item.get("page_content"))
                if text:
                    return text[: self.max_summary_chars]
        if _compact_text(fallback):
            return _compact_text(fallback)[: self.max_summary_chars]
        return combined_text[: self.max_summary_chars]

    def _section_summary(
        self,
        raw_payload: Sequence[Dict[str, Any]],
        combined_text: str,
        *,
        target_terms: Sequence[str],
    ) -> Optional[str]:
        lowered_terms = {str(term).strip().lower() for term in list(target_terms or []) if str(term).strip()}
        for item in list(raw_payload or []):
            metadata = dict(item.get("metadata") or {})
            heading = _compact_text(metadata.get("section_title") or metadata.get("section") or metadata.get("title")).lower()
            if not heading:
                continue
            if any(term in heading for term in lowered_terms):
                text = _compact_text(item.get("page_content"))
                if text:
                    return text[: self.max_summary_chars]
        return combined_text[: min(600, self.max_summary_chars)] if combined_text else None

    def _problem_or_task(self, raw_payload: Sequence[Dict[str, Any]], combined_text: str, *, title: str) -> str:
        for item in list(raw_payload or []):
            text = _compact_text(item.get("page_content"))
            if not text:
                continue
            metadata = dict(item.get("metadata") or {})
            heading = _compact_text(metadata.get("section_title") or metadata.get("section")).lower()
            if heading in {"abstract", "introduction", "background"}:
                return text[: self.max_summary_chars]
        if _compact_text(title):
            return f"{_compact_text(title)}. {combined_text[: min(500, self.max_summary_chars)]}".strip()
        return combined_text[: self.max_summary_chars]

    def _reported_metrics(self, combined_text: str) -> List[str]:
        metrics: List[str] = []
        for match in METRIC_PATTERN.finditer(combined_text):
            metric = _compact_text(match.group(0))
            if metric and metric.lower() not in {value.lower() for value in metrics}:
                metrics.append(metric)
        return metrics[:10]

    def _materials_or_entities(self, combined_text: str, *, title: str) -> List[str]:
        entities: List[str] = []
        corpus = f"{_compact_text(title)} {_compact_text(combined_text)}"
        for match in ENTITY_PATTERN.finditer(corpus):
            entity = _compact_text(match.group(0))
            if entity and entity.lower() not in {value.lower() for value in entities}:
                entities.append(entity)
        return entities[:12]

    def _evidence_rich_sections(self, headings: Sequence[str]) -> List[str]:
        preferred = [
            heading
            for heading in list(headings or [])
            if any(term in heading.lower() for term in ("result", "discussion", "conclusion", "abstract"))
        ]
        if preferred:
            return preferred[:6]
        return list(headings or [])[:4]

    def _profile_limitations(
        self,
        headings: Sequence[str],
        methods: Optional[str],
        metrics: Sequence[str],
    ) -> List[str]:
        limitations: List[str] = []
        normalized_headings = {heading.lower() for heading in list(headings or [])}
        if "results" not in normalized_headings and "result" not in normalized_headings:
            limitations.append("GROBID profile did not recover an explicit Results section heading.")
        if not _compact_text(methods):
            limitations.append("Methods summary is weak or missing from the parsed profile.")
        if not list(metrics or []):
            limitations.append("No canonical metric terms were recovered from the parsed profile.")
        return limitations

    def _citation_readiness_summary(
        self,
        headings: Sequence[str],
        metrics: Sequence[str],
        evidence_rich_sections: Sequence[str],
    ) -> str:
        if list(evidence_rich_sections or []) and list(metrics or []):
            return "Profile indicates explicit evidence-bearing sections and metric language for downstream citation extraction."
        if list(evidence_rich_sections or []):
            return "Profile includes likely evidence-bearing sections, but reported metric coverage is limited."
        if list(headings or []):
            return "Profile recovered section structure, but evidence-bearing signals are weak."
        return "Profile quality is limited and may not support reliable evidence extraction."


def write_profile_failure(
    *,
    store: QAArtifactStore,
    paper_record: PaperRecord,
    reason: str,
) -> str:
    payload = {
        "paper_id": paper_record.paper_id,
        "title": paper_record.title,
        "doi": paper_record.doi,
        "year": paper_record.year,
        "venue": paper_record.venue,
        "source_artifact_path": paper_record.source_artifact_path,
        "reason": _compact_text(reason) or "unknown profile extraction error",
    }
    return store.write_json(f"proposer_profiles/{paper_record.paper_id}.profile_failure.json", payload)
