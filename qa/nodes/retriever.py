from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence

from qa.artifacts import QAArtifactStore
from qa.providers import (
    CrossrefClient,
    OpenAlexClient,
    ProviderRequestError,
    ProviderUnavailableError,
    SemanticScholarClient,
)
from qa.retrieval_state import PaperCandidate, QueryPlan, RetrievalDiagnosticRecord
from qa.retrieval_utils import (
    first_author_key,
    flatten_author_names,
    normalize_doi,
    normalize_text,
    stable_paper_id,
    title_similarity,
    year_in_window,
)
from qa.state import EntityPack, TaskSpec


logger = logging.getLogger("MAD.qa.retriever")

QUESTION_TYPE_MATCH_TERMS: Dict[str, Sequence[str]] = {
    "fact": ("evidence",),
    "causal": ("effect", "impact", "increase", "decrease"),
    "mechanism": ("mechanism", "pathway", "intermediate"),
    "comparison": ("compare", "versus", "better"),
    "frontier": ("recent", "latest", "advance"),
}
LANE_FEATURE_TERMS: Dict[str, Sequence[str]] = {
    "review": ("review", "perspective", "survey"),
    "frontier": ("recent", "latest", "advance"),
    "data": ("benchmark", "yield", "selectivity", "current density", "overpotential"),
    "contrarian": ("limitation", "challenge", "negative", "null", "controvers"),
}
METHOD_SIGNAL_TERMS: Sequence[str] = (
    "icp",
    "spectrom",
    "optical emission",
    "mass spectrom",
    "voltammetry",
    "chromatograph",
    "sensor",
    "determination",
    "detection",
    "quantification",
    "quantify",
    "assay",
)
RETRIEVAL_SIGNAL_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "to",
    "using",
    "what",
    "when",
    "where",
    "which",
    "with",
}


class RetrieverNode:
    def __init__(
        self,
        openalex_client: Optional[Any] = None,
        crossref_client: Optional[Any] = None,
        semantic_scholar_client: Optional[Any] = None,
        per_lane_limit: int = 8,
        final_top_k: int = 12,
        lane_reserve: int = 1,
        title_similarity_threshold: float = 0.94,
        max_enrichment_candidates: int = 8,
    ) -> None:
        self.openalex_client = openalex_client or OpenAlexClient()
        self.crossref_client = crossref_client or CrossrefClient()
        self.semantic_scholar_client = semantic_scholar_client or SemanticScholarClient()
        self.per_lane_limit = per_lane_limit
        self.final_top_k = final_top_k
        self.lane_reserve = lane_reserve
        self.title_similarity_threshold = title_similarity_threshold
        self.max_enrichment_candidates = max(1, int(max_enrichment_candidates))
        self.last_diagnostics: List[RetrievalDiagnosticRecord] = []
        self.last_provider_health: Dict[str, Dict[str, Any]] = {}
        self._diagnostic_map: Dict[tuple[str, str, str], Dict[str, Any]] = {}
        self._provider_health_fallback: Dict[str, Dict[str, Any]] = {}

    def run(
        self,
        task_spec: TaskSpec,
        entity_pack: EntityPack,
        query_plans: Sequence[QueryPlan],
        artifact_store: Optional[QAArtifactStore] = None,
        max_runtime_seconds: Optional[float] = None,
    ) -> List[PaperCandidate]:
        store = artifact_store or QAArtifactStore()
        candidates: List[PaperCandidate] = []
        self._diagnostic_map = {}
        self._provider_health_fallback = self._init_provider_health()
        deadline = None
        if max_runtime_seconds is not None:
            resolved_runtime_budget = max(0.0, float(max_runtime_seconds))
            if resolved_runtime_budget > 0:
                deadline = time.perf_counter() + resolved_runtime_budget

        def _remaining_runtime_seconds() -> Optional[float]:
            if deadline is None:
                return None
            return max(0.0, deadline - time.perf_counter())

        def _runtime_budget_allows(provider_name: str, stage: str, *, minimum_seconds: float = 0.01) -> bool:
            remaining = _remaining_runtime_seconds()
            if remaining is None or remaining >= max(0.0, float(minimum_seconds)):
                return True
            self._record_diagnostic(
                provider=provider_name,
                stage=stage,
                outcome="skipped",
                message=(
                    f"runtime budget exhausted before {stage}; "
                    f"remaining_seconds={round(remaining, 3)}"
                ),
            )
            return False

        for query_plan in query_plans:
            for provider_name in self._search_provider_order(query_plan):
                search_client = {
                    "openalex": self.openalex_client,
                    "semantic_scholar": self.semantic_scholar_client,
                    "crossref": self.crossref_client,
                }.get(provider_name)
                if not _runtime_budget_allows(
                    provider_name,
                    "search",
                    minimum_seconds=0.01,
                ):
                    break
                results = self._run_provider_search(provider_name=provider_name, query_plan=query_plan)
                if results is None:
                    continue
                store.write_json(f"provider_raw/{provider_name}/search_{query_plan.lane}.json", results)
                for raw_item in results:
                    normalized = self._candidate_from_provider(
                        provider_name=provider_name,
                        raw_item=raw_item,
                        lane=query_plan.lane,
                        store=store,
                    )
                    if not normalized:
                        continue
                    existing = self._find_duplicate(candidates, normalized)
                    if existing is None:
                        candidates.append(normalized)
                    else:
                        self._merge_candidates(existing, normalized)
            remaining = _remaining_runtime_seconds()
            if remaining is not None and remaining <= 0:
                break

        shortlist = self._shortlist_candidates_for_enrichment(
            candidates=candidates,
            task_spec=task_spec,
            entity_pack=entity_pack,
            query_plans=query_plans,
        )

        for candidate in shortlist[: self.max_enrichment_candidates]:
            remaining = _remaining_runtime_seconds()
            if remaining is not None and remaining <= 0:
                break
            self._enrich_with_crossref(
                candidate=candidate,
                store=store,
                runtime_remaining_seconds=remaining,
            )
            remaining = _remaining_runtime_seconds()
            if remaining is not None and remaining <= 0:
                break
            self._enrich_with_semantic_scholar(
                candidate=candidate,
                store=store,
                runtime_remaining_seconds=remaining,
            )
            candidate.retrieval_score = self._score_candidate(candidate, task_spec, entity_pack)

        self.last_diagnostics = self._finalize_diagnostics()
        self.last_provider_health = self._collect_provider_health()
        return self._select_diverse_candidates(shortlist, query_plans)

    __call__ = run

    def _search_provider_order(self, query_plan: QueryPlan) -> List[str]:
        preferred = [str(item or "").strip().lower() for item in list(query_plan.preferred_sources or [])]
        ordered: List[str] = []
        for provider_name in preferred + ["openalex", "semantic_scholar", "crossref"]:
            if provider_name in {"openalex", "semantic_scholar", "crossref"} and provider_name not in ordered:
                ordered.append(provider_name)
        return ordered

    def _run_provider_search(self, *, provider_name: str, query_plan: QueryPlan) -> Optional[List[Dict[str, Any]]]:
        search_client = {
            "openalex": self.openalex_client,
            "semantic_scholar": self.semantic_scholar_client,
            "crossref": self.crossref_client,
        }.get(provider_name)
        if search_client is None:
            return None
        try:
            results = list(search_client.search(query_plan, limit=self.per_lane_limit) or [])
            self._record_provider_health_success(provider_name)
        except ProviderUnavailableError as exc:
            logger.warning("%s_search_skipped lane=%s error=%s", provider_name, query_plan.lane, exc)
            self._record_provider_health_skipped(provider_name, str(exc))
            self._record_diagnostic(
                provider=provider_name,
                stage="search",
                lane=query_plan.lane,
                outcome="skipped",
                message=str(exc),
            )
            return None
        except Exception as exc:
            logger.warning("%s_search_failed lane=%s error=%s", provider_name, query_plan.lane, exc)
            self._record_provider_health_failure(provider_name, exc)
            self._record_diagnostic(
                provider=provider_name,
                stage="search",
                lane=query_plan.lane,
                outcome=self._classify_error(exc),
                message=str(exc),
            )
            return None
        if results:
            self._record_diagnostic(
                provider=provider_name,
                stage="search",
                lane=query_plan.lane,
                outcome="hit",
                count=len(results),
            )
        else:
            self._record_diagnostic(
                provider=provider_name,
                stage="search",
                lane=query_plan.lane,
                outcome="empty",
            )
        return results

    def _candidate_from_provider(
        self,
        *,
        provider_name: str,
        raw_item: Dict[str, Any],
        lane: str,
        store: QAArtifactStore,
    ) -> Optional[PaperCandidate]:
        if provider_name == "openalex":
            return self._candidate_from_openalex(raw_item=raw_item, lane=lane, store=store)
        if provider_name == "semantic_scholar":
            return self._candidate_from_semantic_scholar(raw_item=raw_item, lane=lane, store=store)
        if provider_name == "crossref":
            return self._candidate_from_crossref(raw_item=raw_item, lane=lane, store=store)
        return None

    def _candidate_from_openalex(
        self,
        raw_item: Dict[str, Any],
        lane: str,
        store: QAArtifactStore,
    ) -> Optional[PaperCandidate]:
        title = normalize_text(raw_item.get("display_name") or raw_item.get("title"))
        if not title:
            return None
        doi = normalize_doi(raw_item.get("doi") or (raw_item.get("ids") or {}).get("doi"))
        year = raw_item.get("publication_year")
        paper_id = stable_paper_id(doi=doi, title=title, year=year)
        abstract = self._extract_openalex_abstract(raw_item)
        authors = flatten_author_names(raw_item.get("authorships"))
        venue = normalize_text(
            ((raw_item.get("primary_location") or {}).get("source") or {}).get("display_name")
            or ((raw_item.get("host_venue") or {}).get("display_name"))
        ) or None
        best_oa_location = raw_item.get("best_oa_location") or {}
        best_oa_pdf_url = best_oa_location.get("pdf_url")
        best_oa_landing_page_url = best_oa_location.get("landing_page_url")
        oa_url = best_oa_pdf_url or best_oa_landing_page_url
        open_access = raw_item.get("open_access") or {}
        oa_eligible = bool(oa_url) or bool(open_access.get("is_oa")) or bool(best_oa_location)
        oa_signal_reason = (
            "openalex_best_oa_pdf"
            if best_oa_pdf_url
            else "openalex_best_oa_landing_page"
            if best_oa_landing_page_url
            else "openalex_open_access_flag"
            if oa_eligible
            else None
        )

        artifact_path = store.write_json(f"provider_raw/openalex/{paper_id}.json", raw_item)
        return PaperCandidate(
            paper_id=paper_id,
            doi=doi,
            title=title,
            abstract=abstract,
            authors=authors,
            year=year,
            venue=venue,
            provider_hits=["openalex"],
            lane_sources=[lane],
            ranking_features={},
            provider_artifacts={"openalex": artifact_path},
            oa_url=oa_url,
            openalex_id=normalize_text(raw_item.get("id")) or None,
            best_oa_pdf_url=best_oa_pdf_url,
            best_oa_landing_page_url=best_oa_landing_page_url,
            oa_eligible=oa_eligible,
            oa_source="openalex" if oa_eligible else None,
            oa_signal_reason=oa_signal_reason,
        )

    def _candidate_from_semantic_scholar(
        self,
        raw_item: Dict[str, Any],
        lane: str,
        store: QAArtifactStore,
    ) -> Optional[PaperCandidate]:
        title = normalize_text(raw_item.get("title"))
        if not title:
            return None
        doi = normalize_doi(((raw_item.get("externalIds") or {}).get("DOI")) or raw_item.get("doi"))
        year = raw_item.get("year")
        abstract = normalize_text(raw_item.get("abstract")) or None
        authors = flatten_author_names(raw_item.get("authors"))
        venue = normalize_text(raw_item.get("venue")) or None
        best_oa_pdf_url = ((raw_item.get("openAccessPdf") or {}).get("url")) or None
        oa_url = best_oa_pdf_url or raw_item.get("url")
        paper_id = stable_paper_id(doi=doi, title=title, year=year)
        artifact_path = store.write_json(f"provider_raw/semantic_scholar/{paper_id}.json", raw_item)
        candidate = PaperCandidate(
            paper_id=paper_id,
            doi=doi,
            title=title,
            abstract=abstract,
            authors=authors,
            year=year,
            venue=venue,
            provider_hits=["semantic_scholar"],
            lane_sources=[lane],
            ranking_features={},
            provider_artifacts={"semantic_scholar": artifact_path},
            oa_url=oa_url,
            best_oa_pdf_url=best_oa_pdf_url,
            oa_eligible=bool(best_oa_pdf_url),
            oa_source="semantic_scholar" if best_oa_pdf_url else None,
            oa_signal_reason="semantic_scholar_open_access_pdf" if best_oa_pdf_url else None,
        )
        citation_count = raw_item.get("citationCount")
        if citation_count is not None:
            candidate.ranking_features["citation_count"] = int(citation_count or 0)
        return candidate

    def _candidate_from_crossref(
        self,
        raw_item: Dict[str, Any],
        lane: str,
        store: QAArtifactStore,
    ) -> Optional[PaperCandidate]:
        titles = raw_item.get("title") or []
        title = normalize_text(titles[0] if isinstance(titles, list) and titles else titles)
        if not title:
            return None
        doi = normalize_doi(raw_item.get("DOI") or raw_item.get("doi"))
        year = self._extract_crossref_year(raw_item)
        authors = flatten_author_names(raw_item.get("author"))
        venue_titles = raw_item.get("container-title") or []
        venue = normalize_text(venue_titles[0] if isinstance(venue_titles, list) and venue_titles else venue_titles) or None
        abstract = normalize_text(raw_item.get("abstract")) or None
        paper_id = stable_paper_id(doi=doi, title=title, year=year)
        artifact_path = store.write_json(f"provider_raw/crossref/{paper_id}.json", raw_item)
        return PaperCandidate(
            paper_id=paper_id,
            doi=doi,
            title=title,
            abstract=abstract,
            authors=authors,
            year=year,
            venue=venue,
            provider_hits=["crossref"],
            lane_sources=[lane],
            ranking_features={},
            provider_artifacts={"crossref": artifact_path},
        )

    def _extract_openalex_abstract(self, raw_item: Dict[str, Any]) -> Optional[str]:
        direct_abstract = normalize_text(raw_item.get("abstract"))
        if direct_abstract:
            return direct_abstract
        inverted_index = raw_item.get("abstract_inverted_index") or {}
        if not inverted_index:
            return None
        ordered_tokens: List[tuple[int, str]] = []
        for token, positions in inverted_index.items():
            for position in positions or []:
                ordered_tokens.append((int(position), str(token)))
        ordered_tokens.sort(key=lambda item: item[0])
        if not ordered_tokens:
            return None
        return " ".join(token for _, token in ordered_tokens)

    def _find_duplicate(
        self,
        existing_candidates: Iterable[PaperCandidate],
        candidate: PaperCandidate,
    ) -> Optional[PaperCandidate]:
        candidate_doi = normalize_doi(candidate.doi)
        candidate_first_author = first_author_key(candidate.authors)
        for existing in existing_candidates:
            existing_doi = normalize_doi(existing.doi)
            if candidate_doi and existing_doi and candidate_doi == existing_doi:
                return existing

            similarity = title_similarity(existing.title, candidate.title)
            if similarity < self.title_similarity_threshold:
                continue

            if existing.year is not None and candidate.year is not None and abs(existing.year - candidate.year) > 1:
                continue

            existing_first_author = first_author_key(existing.authors)
            if existing_first_author and candidate_first_author and existing_first_author != candidate_first_author:
                continue
            return existing
        return None

    def _merge_candidates(self, target: PaperCandidate, incoming: PaperCandidate) -> None:
        if not target.doi and incoming.doi:
            target.doi = incoming.doi
        if len(incoming.abstract or "") > len(target.abstract or ""):
            target.abstract = incoming.abstract
        if not target.authors and incoming.authors:
            target.authors = incoming.authors
        if target.year is None and incoming.year is not None:
            target.year = incoming.year
        if not target.venue and incoming.venue:
            target.venue = incoming.venue
        if not target.oa_url and incoming.oa_url:
            target.oa_url = incoming.oa_url
        if not target.openalex_id and incoming.openalex_id:
            target.openalex_id = incoming.openalex_id
        if not target.best_oa_pdf_url and incoming.best_oa_pdf_url:
            target.best_oa_pdf_url = incoming.best_oa_pdf_url
        if not target.best_oa_landing_page_url and incoming.best_oa_landing_page_url:
            target.best_oa_landing_page_url = incoming.best_oa_landing_page_url
        if not target.oa_eligible and incoming.oa_eligible:
            target.oa_eligible = True
        if not target.oa_source and incoming.oa_source:
            target.oa_source = incoming.oa_source
        if not target.oa_signal_reason and incoming.oa_signal_reason:
            target.oa_signal_reason = incoming.oa_signal_reason
        target.provider_hits = list(dict.fromkeys([*target.provider_hits, *incoming.provider_hits]))
        target.lane_sources = list(dict.fromkeys([*target.lane_sources, *incoming.lane_sources]))
        merged_artifacts = dict(target.provider_artifacts)
        merged_artifacts.update(incoming.provider_artifacts)
        target.provider_artifacts = merged_artifacts

    def _enrich_with_crossref(
        self,
        candidate: PaperCandidate,
        store: QAArtifactStore,
        *,
        runtime_remaining_seconds: Optional[float] = None,
    ) -> None:
        if not self._candidate_needs_crossref_enrichment(candidate):
            self._record_diagnostic(
                provider="crossref",
                stage="enrichment",
                outcome="skipped",
                message=f"paper_id={candidate.paper_id}: skipped because bibliography fields are already present",
            )
            return
        if runtime_remaining_seconds is not None and runtime_remaining_seconds < max(
            1.0,
            float(getattr(self.crossref_client, "timeout", 1.0) or 1.0),
        ):
            self._record_diagnostic(
                provider="crossref",
                stage="enrichment",
                outcome="skipped",
                message=(
                    f"paper_id={candidate.paper_id}: skipped because runtime budget is exhausted "
                    f"(remaining_seconds={round(runtime_remaining_seconds, 3)})"
                ),
            )
            return
        try:
            raw = self.crossref_client.enrich(candidate.model_dump()) if self.crossref_client else None
            if self.crossref_client:
                self._record_provider_health_success("crossref")
        except ProviderUnavailableError as exc:
            logger.warning("crossref_enrichment_skipped paper_id=%s error=%s", candidate.paper_id, exc)
            self._record_provider_health_skipped("crossref", str(exc))
            self._record_diagnostic(
                provider="crossref",
                stage="enrichment",
                outcome="skipped",
                message=f"paper_id={candidate.paper_id}: {exc}",
            )
            return
        except Exception as exc:
            logger.warning("crossref_enrichment_failed paper_id=%s error=%s", candidate.paper_id, exc)
            self._record_provider_health_failure("crossref", exc)
            self._record_diagnostic(
                provider="crossref",
                stage="enrichment",
                outcome=self._classify_error(exc),
                message=f"paper_id={candidate.paper_id}: {exc}",
            )
            return
        if not raw:
            self._record_diagnostic(provider="crossref", stage="enrichment", outcome="empty")
            return

        artifact_path = store.write_json(f"provider_raw/crossref/{candidate.paper_id}.json", raw)
        doi = normalize_doi(raw.get("DOI") or raw.get("doi"))
        titles = raw.get("title") or []
        title = normalize_text(titles[0] if isinstance(titles, list) and titles else titles)
        authors = flatten_author_names(raw.get("author"))
        year = self._extract_crossref_year(raw)
        venue_titles = raw.get("container-title") or []
        venue = normalize_text(venue_titles[0] if isinstance(venue_titles, list) and venue_titles else venue_titles) or None

        candidate.provider_hits = list(dict.fromkeys([*candidate.provider_hits, "crossref"]))
        candidate.provider_artifacts = {**candidate.provider_artifacts, "crossref": artifact_path}
        if doi and not candidate.doi:
            candidate.doi = doi
        if title and len(title) >= len(candidate.title):
            candidate.title = title
        if authors and not candidate.authors:
            candidate.authors = authors
        if year and candidate.year is None:
            candidate.year = year
        if venue and not candidate.venue:
            candidate.venue = venue
        self._record_diagnostic(provider="crossref", stage="enrichment", outcome="hit")

    def _extract_crossref_year(self, raw: Dict[str, Any]) -> Optional[int]:
        for key in ("published-print", "published-online", "issued"):
            parts = raw.get(key, {}).get("date-parts") if isinstance(raw.get(key), dict) else None
            if parts and parts[0]:
                try:
                    return int(parts[0][0])
                except (TypeError, ValueError):
                    continue
        return None

    def _enrich_with_semantic_scholar(
        self,
        candidate: PaperCandidate,
        store: QAArtifactStore,
        *,
        runtime_remaining_seconds: Optional[float] = None,
    ) -> None:
        if not self._candidate_needs_semantic_scholar_enrichment(candidate):
            self._record_diagnostic(
                provider="semantic_scholar",
                stage="enrichment",
                outcome="skipped",
                message=f"paper_id={candidate.paper_id}: skipped because abstract and citation metadata are already present",
            )
            return
        if runtime_remaining_seconds is not None and runtime_remaining_seconds < max(
            1.0,
            float(getattr(self.semantic_scholar_client, "timeout", 1.0) or 1.0),
        ):
            self._record_diagnostic(
                provider="semantic_scholar",
                stage="enrichment",
                outcome="skipped",
                message=(
                    f"paper_id={candidate.paper_id}: skipped because runtime budget is exhausted "
                    f"(remaining_seconds={round(runtime_remaining_seconds, 3)})"
                ),
            )
            return
        try:
            raw = self.semantic_scholar_client.enrich(candidate.model_dump()) if self.semantic_scholar_client else None
            if self.semantic_scholar_client:
                self._record_provider_health_success("semantic_scholar")
        except ProviderUnavailableError as exc:
            logger.warning("semantic_scholar_enrichment_skipped paper_id=%s error=%s", candidate.paper_id, exc)
            self._record_provider_health_skipped("semantic_scholar", str(exc))
            self._record_diagnostic(
                provider="semantic_scholar",
                stage="enrichment",
                outcome="skipped",
                message=f"paper_id={candidate.paper_id}: {exc}",
            )
            return
        except Exception as exc:
            logger.warning("semantic_scholar_enrichment_failed paper_id=%s error=%s", candidate.paper_id, exc)
            self._record_provider_health_failure("semantic_scholar", exc)
            self._record_diagnostic(
                provider="semantic_scholar",
                stage="enrichment",
                outcome=self._classify_error(exc),
                message=f"paper_id={candidate.paper_id}: {exc}",
            )
            return
        if not raw:
            self._record_diagnostic(provider="semantic_scholar", stage="enrichment", outcome="empty")
            return

        artifact_path = store.write_json(f"provider_raw/semantic_scholar/{candidate.paper_id}.json", raw)
        doi = normalize_doi(((raw.get("externalIds") or {}).get("DOI")) or raw.get("doi"))
        abstract = normalize_text(raw.get("abstract")) or None
        authors = flatten_author_names(raw.get("authors"))
        venue = normalize_text(raw.get("venue")) or None
        year = raw.get("year")
        citation_count = raw.get("citationCount")

        candidate.provider_hits = list(dict.fromkeys([*candidate.provider_hits, "semantic_scholar"]))
        candidate.provider_artifacts = {**candidate.provider_artifacts, "semantic_scholar": artifact_path}
        if doi and not candidate.doi:
            candidate.doi = doi
        if abstract and len(abstract) > len(candidate.abstract or ""):
            candidate.abstract = abstract
        if authors and not candidate.authors:
            candidate.authors = authors
        if venue and not candidate.venue:
            candidate.venue = venue
        if year and candidate.year is None:
            candidate.year = year
        candidate.ranking_features["citation_count"] = int(citation_count or 0)
        self._record_diagnostic(provider="semantic_scholar", stage="enrichment", outcome="hit")

    def _candidate_needs_crossref_enrichment(self, candidate: PaperCandidate) -> bool:
        return not bool(
            normalize_doi(candidate.doi)
            and candidate.year is not None
            and normalize_text(candidate.venue)
            and list(candidate.authors or [])
        )

    def _candidate_needs_semantic_scholar_enrichment(self, candidate: PaperCandidate) -> bool:
        return not bool(candidate.abstract and candidate.ranking_features.get("citation_count") is not None)

    def _score_candidate(self, candidate: PaperCandidate, task_spec: TaskSpec, entity_pack: EntityPack) -> float:
        corpus = normalize_text(f"{candidate.title} {candidate.abstract or ''}").lower()
        anchor_terms = self._collect_anchor_terms(task_spec=task_spec, entity_pack=entity_pack)
        entity_hits = sum(1 for term in anchor_terms if term and term in corpus)
        required_terms = self._required_query_terms(task_spec)
        required_phrase_hits = sum(1 for term in required_terms if term and term in corpus)
        should_phrase_hits = sum(1 for term in self._suggested_query_terms(task_spec) if term and term in corpus)
        corpus_tokens = self._signal_tokens(corpus)
        required_token_hits = sum(1 for token in self._signal_tokens(" ".join(required_terms)) if token in corpus_tokens)
        question_token_hits = sum(
            1 for token in self._signal_tokens(task_spec.normalized_question) if token in corpus_tokens
        )
        question_type_hits = sum(
            1 for term in QUESTION_TYPE_MATCH_TERMS.get(task_spec.question_type, ()) if term in corpus
        )
        method_signal_hits = sum(1 for term in METHOD_SIGNAL_TERMS if term in corpus)
        lane_feature_hits = sum(
            1
            for lane in candidate.lane_sources
            for term in LANE_FEATURE_TERMS.get(lane, ())
            if term in corpus
        )
        time_window_hit = 1.0 if year_in_window(candidate.year, task_spec.year_from, task_spec.year_to) else 0.0
        abstract_available = 1.0 if candidate.abstract else 0.0
        doi_complete = 1.0 if candidate.doi else 0.0

        candidate.ranking_features.update(
            {
                "entity_anchor_hits": entity_hits,
                "required_phrase_hits": required_phrase_hits,
                "required_token_hits": required_token_hits,
                "question_token_hits": question_token_hits,
                "should_phrase_hits": should_phrase_hits,
                "question_type_hits": question_type_hits,
                "method_signal_hits": method_signal_hits,
                "time_window_hit": time_window_hit,
                "lane_feature_hits": lane_feature_hits,
                "abstract_available": abstract_available,
                "doi_complete": doi_complete,
            }
        )

        score = (
            4.0 * min(entity_hits, 3)
            + 3.0 * min(required_phrase_hits, 3)
            + 0.75 * min(required_token_hits, 4)
            + 0.75 * min(should_phrase_hits, 2)
            + 0.5 * min(question_token_hits, 4)
            + 2.0 * min(question_type_hits, 2)
            + 1.25 * min(method_signal_hits, 2)
            + 1.5 * time_window_hit
            + 1.0 * min(lane_feature_hits, 3)
            + 0.75 * abstract_available
            + 0.5 * doi_complete
        )
        return round(score, 4)

    def _shortlist_candidates_for_enrichment(
        self,
        *,
        candidates: Sequence[PaperCandidate],
        task_spec: TaskSpec,
        entity_pack: EntityPack,
        query_plans: Sequence[QueryPlan],
    ) -> List[PaperCandidate]:
        scored_candidates: List[PaperCandidate] = []
        for candidate in candidates:
            candidate.retrieval_score = self._score_candidate(candidate, task_spec, entity_pack)
            scored_candidates.append(candidate)

        gated_candidates = [
            candidate for candidate in scored_candidates if self._passes_required_term_gate(candidate, task_spec)
        ]
        if gated_candidates:
            effective_candidates = gated_candidates
        else:
            method_signal_candidates = [
                candidate
                for candidate in scored_candidates
                if int(candidate.ranking_features.get("method_signal_hits") or 0) > 0
            ]
            effective_candidates = method_signal_candidates or scored_candidates
        shortlist_limit = max(self.final_top_k + max(1, len(query_plans)), self.per_lane_limit)
        return sorted(effective_candidates, key=lambda item: item.retrieval_score, reverse=True)[:shortlist_limit]

    def _collect_anchor_terms(self, task_spec: TaskSpec, entity_pack: EntityPack) -> List[str]:
        terms: List[str] = []
        for entity in entity_pack.entities:
            terms.extend(anchor.lower() for anchor in entity.query_anchors if normalize_text(anchor))
            terms.append(entity.canonical_name.lower())
        for term in task_spec.query_constraints.must_include_terms:
            cleaned = normalize_text(term).lower()
            if cleaned:
                terms.append(cleaned)
        return list(dict.fromkeys(terms))

    def _required_query_terms(self, task_spec: TaskSpec) -> List[str]:
        terms: List[str] = []
        for term in list(task_spec.query_constraints.must_include_terms or []):
            cleaned = normalize_text(term).lower()
            if cleaned:
                terms.append(cleaned)
        return list(dict.fromkeys(terms))

    def _suggested_query_terms(self, task_spec: TaskSpec) -> List[str]:
        terms: List[str] = []
        for term in list(task_spec.query_constraints.should_include_terms or []):
            cleaned = normalize_text(term).lower()
            if cleaned:
                terms.append(cleaned)
        return list(dict.fromkeys(terms))

    def _signal_tokens(self, text: Optional[str]) -> List[str]:
        tokens: List[str] = []
        for token in re.findall(r"[a-z0-9][a-z0-9/+\-.]*", normalize_text(text).lower()):
            if token in RETRIEVAL_SIGNAL_STOPWORDS:
                continue
            if len(token) < 3 and not any(char.isdigit() for char in token):
                continue
            tokens.append(token)
        return list(dict.fromkeys(tokens))

    def _passes_required_term_gate(self, candidate: PaperCandidate, task_spec: TaskSpec) -> bool:
        required_terms = self._required_query_terms(task_spec)
        if not required_terms:
            return True

        corpus = normalize_text(f"{candidate.title} {candidate.abstract or ''}").lower()
        corpus_tokens = set(self._signal_tokens(corpus))
        required_phrase_hits = int(candidate.ranking_features.get("required_phrase_hits") or 0)
        required_token_hits = int(candidate.ranking_features.get("required_token_hits") or 0)
        question_token_hits = int(candidate.ranking_features.get("question_token_hits") or 0)
        should_phrase_hits = int(candidate.ranking_features.get("should_phrase_hits") or 0)
        method_signal_hits = int(candidate.ranking_features.get("method_signal_hits") or 0)
        required_tokens = self._signal_tokens(" ".join(required_terms))
        should_terms = self._suggested_query_terms(task_spec)
        normalized_question = normalize_text(task_spec.normalized_question).lower()
        multiword_required_terms = [term for term in required_terms if len(self._signal_tokens(term)) >= 2]
        phrase_target = 1 if len(required_terms) <= 2 else 2
        token_target = min(max(2, len(required_tokens) // 2), 5) if required_tokens else 0
        question_target = min(max(2, len(self._signal_tokens(task_spec.normalized_question)) // 3), 4)
        requires_method_signal = bool(should_terms) and any(
            token in normalized_question for token in ("method", "analytical", "technique", "instrument")
        )
        requires_context_phrase = bool(multiword_required_terms)

        if requires_context_phrase:
            context_phrase_hit = any(term in corpus for term in multiword_required_terms)
            context_token_hit = any(
                all(token in corpus_tokens for token in self._signal_tokens(term))
                for term in multiword_required_terms
            )
            if not context_phrase_hit and not context_token_hit:
                return False

        if requires_method_signal and should_phrase_hits <= 0 and method_signal_hits <= 0:
            return False

        if required_phrase_hits >= phrase_target:
            return True
        if required_token_hits >= token_target and question_token_hits >= question_target:
            return True
        return False

    def _select_diverse_candidates(
        self,
        candidates: Sequence[PaperCandidate],
        query_plans: Sequence[QueryPlan],
    ) -> List[PaperCandidate]:
        sorted_candidates = sorted(candidates, key=lambda item: item.retrieval_score, reverse=True)
        selected: List[PaperCandidate] = []
        selected_ids = set()

        for query_plan in query_plans:
            lane_candidates = [
                candidate for candidate in sorted_candidates if query_plan.lane in candidate.lane_sources
            ]
            for candidate in lane_candidates[: self.lane_reserve]:
                if candidate.paper_id in selected_ids:
                    continue
                selected.append(candidate)
                selected_ids.add(candidate.paper_id)

        for candidate in sorted_candidates:
            if len(selected) >= self.final_top_k:
                break
            if candidate.paper_id in selected_ids:
                continue
            selected.append(candidate)
            selected_ids.add(candidate.paper_id)

        return selected[: self.final_top_k]

    def _record_diagnostic(
        self,
        *,
        provider: str,
        stage: str,
        outcome: str,
        lane: Optional[str] = None,
        count: int = 1,
        message: Optional[str] = None,
    ) -> None:
        key = (provider, stage, lane or "")
        record = self._diagnostic_map.setdefault(
            key,
            {
                "provider": provider,
                "stage": stage,
                "lane": lane,
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

    def _init_provider_health(self) -> Dict[str, Dict[str, Any]]:
        health: Dict[str, Dict[str, Any]] = {}
        for provider_name, client in (
            ("openalex", self.openalex_client),
            ("crossref", self.crossref_client),
            ("semantic_scholar", self.semantic_scholar_client),
        ):
            health[provider_name] = {
                "status": "idle" if client is not None else "disabled",
                "calls": 0,
                "successes": 0,
                "retry_exhausted_failures": 0,
                "skipped_calls": 0,
                "last_error": None,
            }
        return health

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

    def _collect_provider_health(self) -> Dict[str, Dict[str, Any]]:
        collected = {
            provider_name: dict(payload)
            for provider_name, payload in self._provider_health_fallback.items()
        }
        for provider_name, client in (
            ("openalex", self.openalex_client),
            ("crossref", self.crossref_client),
            ("semantic_scholar", self.semantic_scholar_client),
        ):
            snapshot = self._client_health_snapshot(client)
            if snapshot is not None:
                collected[provider_name] = snapshot
        return collected

    def _client_health_snapshot(self, client: Any) -> Optional[Dict[str, Any]]:
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
