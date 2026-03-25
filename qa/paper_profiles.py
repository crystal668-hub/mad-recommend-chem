from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

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
        source_artifact_path = str(paper_record.source_artifact_path or "").strip()
        if not source_artifact_path:
            raise ValueError(f"paper_id={paper_record.paper_id} has no source_artifact_path for GROBID parsing.")
        source_path = Path(source_artifact_path)
        if source_path.suffix.lower() != ".pdf":
            raise ValueError(f"paper_id={paper_record.paper_id} does not have a PDF source artifact for GROBID parsing.")

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
        raw_artifact_path = store.write_json(
            f"proposer_profiles/{paper_record.paper_id}.grobid_raw.json",
            raw_payload,
        )

        text_fragments = [_compact_text(item["page_content"]) for item in raw_payload if _compact_text(item["page_content"])]
        combined_text = "\n".join(text_fragments).strip()
        if not combined_text:
            raise ValueError(f"paper_id={paper_record.paper_id} produced no usable GROBID text.")

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
            parser_name="langchain_community.GenericLoader+GrobidParser",
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
        parser_artifact_path = store.write_json(
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
