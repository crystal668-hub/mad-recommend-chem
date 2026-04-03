from __future__ import annotations

import argparse
import json
import random
import re
import time
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

import requests


RETRYABLE_STATUS_CODES = {408, 429}


def _compact_text(value: Any) -> str:
    return " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split()).strip()


def _normalize_text(value: Any) -> str:
    return _compact_text(value)


def normalize_doi(value: Optional[str]) -> Optional[str]:
    text = _compact_text(value)
    if not text:
        return None
    text = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", text, flags=re.I)
    return text.lower() or None


def stable_paper_id(doi: Optional[str], title: str, year: Optional[int] = None) -> str:
    normalized_doi = normalize_doi(doi)
    if normalized_doi:
        token = normalized_doi
        prefix = "doi"
    else:
        token = f"{_normalize_text(title).lower()}|{year or ''}"
        prefix = "title"
    slug = re.sub(r"[^a-z0-9]+", "-", token.lower()).strip("-")
    return f"{prefix}-{slug[:80] or 'paper'}"


def title_similarity(left: str, right: str) -> float:
    return SequenceMatcher(a=_normalize_text(left).lower(), b=_normalize_text(right).lower()).ratio()


@dataclass
class QueryPlan:
    query_text: str
    must_terms: list[str] = field(default_factory=list)
    exclude_terms: list[str] = field(default_factory=list)
    year_from: Optional[int] = None
    year_to: Optional[int] = None
    preferred_sources: list[str] = field(default_factory=list)
    limit: int = 8


@dataclass
class PaperCandidate:
    paper_id: str
    title: str
    doi: Optional[str] = None
    abstract: Optional[str] = None
    authors: list[str] = field(default_factory=list)
    year: Optional[int] = None
    venue: Optional[str] = None
    provider_hits: list[str] = field(default_factory=list)
    retrieval_score: float = 0.0
    oa_url: Optional[str] = None
    open_access_pdf_url: Optional[str] = None


@dataclass
class ProviderHealthRecord:
    status: str = "idle"
    calls: int = 0
    successes: int = 0
    skipped_calls: int = 0
    retry_exhausted_failures: int = 0
    last_error: Optional[str] = None


class ProviderRequestError(RuntimeError):
    pass


class _HttpClient:
    def __init__(self, *, timeout: float = 10.0, retry_attempts: int = 2) -> None:
        self.timeout = float(timeout)
        self.retry_attempts = max(0, int(retry_attempts))
        self.health = ProviderHealthRecord()

    def get(self, url: str, *, params: Optional[dict[str, Any]] = None, headers: Optional[dict[str, str]] = None) -> Any:
        self.health.calls += 1
        attempts = self.retry_attempts + 1
        for attempt in range(1, attempts + 1):
            try:
                response = requests.get(url, params=params, headers=headers, timeout=self.timeout)
                status_code = int(getattr(response, "status_code", 200))
                if status_code in RETRYABLE_STATUS_CODES or status_code >= 500:
                    raise requests.exceptions.HTTPError(f"HTTP {status_code}", response=response)
                response.raise_for_status()
                self.health.successes += 1
                self.health.status = "healthy"
                self.health.last_error = None
                return response
            except requests.exceptions.RequestException as exc:
                self.health.last_error = str(exc)
                self.health.status = "degraded"
                if attempt >= attempts:
                    self.health.retry_exhausted_failures += 1
                    raise ProviderRequestError(str(exc)) from exc
                time.sleep(min(1.0 * (2 ** (attempt - 1)) + random.random() * 0.1, 4.0))
        raise ProviderRequestError("request failed")


class RetrievalEngine:
    def __init__(self, *, openalex_mailto: Optional[str] = None, crossref_mailto: Optional[str] = None, semantic_scholar_api_key: Optional[str] = None, timeout: float = 10.0) -> None:
        self.openalex_mailto = _compact_text(openalex_mailto) or None
        self.crossref_mailto = _compact_text(crossref_mailto) or None
        self.semantic_scholar_api_key = _compact_text(semantic_scholar_api_key) or None
        self.http = _HttpClient(timeout=timeout)

    def search(self, plan: QueryPlan) -> dict[str, Any]:
        papers: list[PaperCandidate] = []
        diagnostics: list[dict[str, Any]] = []
        provider_health: dict[str, Any] = {}
        for provider_name in self._provider_order(plan):
            try:
                payloads = self._search_provider(provider_name, plan)
            except Exception as exc:
                diagnostics.append({"provider": provider_name, "stage": "search", "outcome": "failure", "message": str(exc)})
                provider_health[provider_name] = asdict(self.http.health)
                continue
            diagnostics.append({"provider": provider_name, "stage": "search", "outcome": "hit" if payloads else "empty", "count": len(payloads)})
            provider_health[provider_name] = asdict(self.http.health)
            for raw_item in payloads:
                candidate = self._candidate_from_provider(provider_name, raw_item)
                if candidate is None:
                    continue
                existing = self._find_duplicate(papers, candidate)
                if existing is None:
                    papers.append(candidate)
                else:
                    self._merge(existing, candidate)
        for candidate in papers:
            candidate.retrieval_score = self._score(candidate, plan)
        ranked = sorted(papers, key=lambda item: item.retrieval_score, reverse=True)[: max(1, plan.limit)]
        return {
            "request": asdict(plan),
            "papers": [asdict(item) for item in ranked],
            "diagnostics": diagnostics,
            "provider_health": provider_health,
        }

    def _provider_order(self, plan: QueryPlan) -> list[str]:
        ordered: list[str] = []
        for provider_name in [*plan.preferred_sources, "openalex", "semantic_scholar", "crossref"]:
            normalized = _compact_text(provider_name).lower()
            if normalized in {"openalex", "semantic_scholar", "crossref"} and normalized not in ordered:
                ordered.append(normalized)
        return ordered

    def _search_provider(self, provider_name: str, plan: QueryPlan) -> list[dict[str, Any]]:
        if provider_name == "openalex":
            params: dict[str, Any] = {"search": plan.query_text, "per-page": plan.limit}
            if plan.year_from and plan.year_to:
                params["filter"] = f"publication_year:{plan.year_from}-{plan.year_to}"
            if self.openalex_mailto:
                params["mailto"] = self.openalex_mailto
            response = self.http.get("https://api.openalex.org/works", params=params)
            return list(response.json().get("results") or [])
        if provider_name == "semantic_scholar":
            params = {
                "query": plan.query_text,
                "limit": plan.limit,
                "fields": "title,abstract,year,venue,authors,externalIds,openAccessPdf,url",
            }
            if plan.year_from:
                params["year"] = f"{plan.year_from}:{plan.year_to or plan.year_from}"
            headers = {"x-api-key": self.semantic_scholar_api_key} if self.semantic_scholar_api_key else None
            response = self.http.get("https://api.semanticscholar.org/graph/v1/paper/search", params=params, headers=headers)
            return list(response.json().get("data") or [])
        params = {"rows": plan.limit, "query.bibliographic": plan.query_text}
        if plan.year_from and plan.year_to:
            params["filter"] = f"from-pub-date:{plan.year_from}-01-01,until-pub-date:{plan.year_to}-12-31"
        if self.crossref_mailto:
            params["mailto"] = self.crossref_mailto
        response = self.http.get("https://api.crossref.org/works", params=params)
        return list(response.json().get("message", {}).get("items") or [])

    def _candidate_from_provider(self, provider_name: str, raw_item: dict[str, Any]) -> Optional[PaperCandidate]:
        if provider_name == "openalex":
            title = _normalize_text(raw_item.get("display_name") or raw_item.get("title"))
            if not title:
                return None
            doi = normalize_doi(raw_item.get("doi") or (raw_item.get("ids") or {}).get("doi"))
            best_oa_location = raw_item.get("best_oa_location") or {}
            return PaperCandidate(
                paper_id=stable_paper_id(doi, title, raw_item.get("publication_year")),
                title=title,
                doi=doi,
                abstract=_normalize_text(raw_item.get("abstract") or "") or None,
                authors=[_normalize_text((item.get("author") or {}).get("display_name")) for item in list(raw_item.get("authorships") or []) if _normalize_text((item.get("author") or {}).get("display_name"))],
                year=raw_item.get("publication_year"),
                venue=_normalize_text((((raw_item.get("primary_location") or {}).get("source") or {}).get("display_name")) or ""),
                provider_hits=["openalex"],
                oa_url=_compact_text(best_oa_location.get("landing_page_url")) or None,
                open_access_pdf_url=_compact_text(best_oa_location.get("pdf_url")) or None,
            )
        if provider_name == "semantic_scholar":
            title = _normalize_text(raw_item.get("title"))
            if not title:
                return None
            doi = normalize_doi(((raw_item.get("externalIds") or {}).get("DOI")) or raw_item.get("doi"))
            return PaperCandidate(
                paper_id=stable_paper_id(doi, title, raw_item.get("year")),
                title=title,
                doi=doi,
                abstract=_normalize_text(raw_item.get("abstract")) or None,
                authors=[_normalize_text(item.get("name")) for item in list(raw_item.get("authors") or []) if _normalize_text(item.get("name"))],
                year=raw_item.get("year"),
                venue=_normalize_text(raw_item.get("venue")) or None,
                provider_hits=["semantic_scholar"],
                oa_url=_compact_text(raw_item.get("url")) or None,
                open_access_pdf_url=_compact_text((raw_item.get("openAccessPdf") or {}).get("url")) or None,
            )
        titles = raw_item.get("title") or []
        title = _normalize_text(titles[0] if isinstance(titles, list) and titles else titles)
        if not title:
            return None
        year = None
        for key in ("published-print", "published-online", "issued"):
            date_parts = (raw_item.get(key) or {}).get("date-parts") if isinstance(raw_item.get(key), dict) else None
            if date_parts and date_parts[0]:
                year = int(date_parts[0][0])
                break
        return PaperCandidate(
            paper_id=stable_paper_id(normalize_doi(raw_item.get("DOI")), title, year),
            title=title,
            doi=normalize_doi(raw_item.get("DOI") or raw_item.get("doi")),
            abstract=_normalize_text(raw_item.get("abstract")) or None,
            authors=[_normalize_text(" ".join([item.get("given", ""), item.get("family", "")])) for item in list(raw_item.get("author") or []) if _normalize_text(" ".join([item.get("given", ""), item.get("family", "")]))],
            year=year,
            venue=_normalize_text((raw_item.get("container-title") or [""])[0]) or None,
            provider_hits=["crossref"],
        )

    def _find_duplicate(self, papers: list[PaperCandidate], candidate: PaperCandidate) -> Optional[PaperCandidate]:
        for existing in papers:
            if normalize_doi(existing.doi) and normalize_doi(existing.doi) == normalize_doi(candidate.doi):
                return existing
            if title_similarity(existing.title, candidate.title) >= 0.94:
                return existing
        return None

    def _merge(self, target: PaperCandidate, incoming: PaperCandidate) -> None:
        if not target.doi and incoming.doi:
            target.doi = incoming.doi
        if len(incoming.abstract or "") > len(target.abstract or ""):
            target.abstract = incoming.abstract
        if not target.authors and incoming.authors:
            target.authors = list(incoming.authors)
        if target.year is None and incoming.year is not None:
            target.year = incoming.year
        if not target.venue and incoming.venue:
            target.venue = incoming.venue
        if not target.oa_url and incoming.oa_url:
            target.oa_url = incoming.oa_url
        if not target.open_access_pdf_url and incoming.open_access_pdf_url:
            target.open_access_pdf_url = incoming.open_access_pdf_url
        target.provider_hits = list(dict.fromkeys([*target.provider_hits, *incoming.provider_hits]))

    def _score(self, candidate: PaperCandidate, plan: QueryPlan) -> float:
        corpus = _normalize_text(f"{candidate.title} {candidate.abstract or ''}").lower()
        score = 0.0
        for term in [_compact_text(item).lower() for item in plan.must_terms if _compact_text(item)]:
            if term in corpus:
                score += 3.0
        for token in re.findall(r"[a-z0-9]+", plan.query_text.lower()):
            if token in corpus:
                score += 0.5
        for term in [_compact_text(item).lower() for item in plan.exclude_terms if _compact_text(item)]:
            if term in corpus:
                score -= 4.0
        if candidate.doi:
            score += 1.0
        if candidate.abstract:
            score += 1.0
        if candidate.open_access_pdf_url:
            score += 0.5
        return round(score, 4)


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Portable paper retrieval skill")
    parser.add_argument("--query", required=True, help="Query text")
    parser.add_argument("--output-dir", required=True, help="Directory to write search results")
    parser.add_argument("--must-terms", default="", help="Comma-separated must terms")
    parser.add_argument("--exclude-terms", default="", help="Comma-separated exclude terms")
    parser.add_argument("--preferred-sources", default="", help="Comma-separated provider order")
    parser.add_argument("--year-from", type=int, default=None)
    parser.add_argument("--year-to", type=int, default=None)
    parser.add_argument("--limit", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = QueryPlan(
        query_text=args.query,
        must_terms=[item.strip() for item in args.must_terms.split(",") if item.strip()],
        exclude_terms=[item.strip() for item in args.exclude_terms.split(",") if item.strip()],
        year_from=args.year_from,
        year_to=args.year_to,
        preferred_sources=[item.strip() for item in args.preferred_sources.split(",") if item.strip()],
        limit=max(1, int(args.limit)),
    )
    engine = RetrievalEngine()
    result = engine.search(plan)
    result_path = output_dir / "retrieval_result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
