from __future__ import annotations

import logging
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
    ) -> None:
        self.openalex_client = openalex_client or OpenAlexClient()
        self.crossref_client = crossref_client or CrossrefClient()
        self.semantic_scholar_client = semantic_scholar_client or SemanticScholarClient()
        self.per_lane_limit = per_lane_limit
        self.final_top_k = final_top_k
        self.lane_reserve = lane_reserve
        self.title_similarity_threshold = title_similarity_threshold
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
    ) -> List[PaperCandidate]:
        store = artifact_store or QAArtifactStore()
        candidates: List[PaperCandidate] = []
        self._diagnostic_map = {}
        self._provider_health_fallback = self._init_provider_health()

        for query_plan in query_plans:
            try:
                results = list(self.openalex_client.search(query_plan, limit=self.per_lane_limit) or [])
                self._record_provider_health_success("openalex")
            except ProviderUnavailableError as exc:
                logger.warning("openalex_search_skipped lane=%s error=%s", query_plan.lane, exc)
                self._record_provider_health_skipped("openalex", str(exc))
                self._record_diagnostic(
                    provider="openalex",
                    stage="search",
                    lane=query_plan.lane,
                    outcome="skipped",
                    message=str(exc),
                )
                continue
            except Exception as exc:
                logger.warning("openalex_search_failed lane=%s error=%s", query_plan.lane, exc)
                self._record_provider_health_failure("openalex", exc)
                self._record_diagnostic(
                    provider="openalex",
                    stage="search",
                    lane=query_plan.lane,
                    outcome=self._classify_error(exc),
                    message=str(exc),
                )
                continue
            if results:
                self._record_diagnostic(
                    provider="openalex",
                    stage="search",
                    lane=query_plan.lane,
                    outcome="hit",
                    count=len(results),
                )
            else:
                self._record_diagnostic(
                    provider="openalex",
                    stage="search",
                    lane=query_plan.lane,
                    outcome="empty",
                )

            store.write_json(f"provider_raw/openalex/search_{query_plan.lane}.json", results)
            for raw_item in results:
                normalized = self._candidate_from_openalex(raw_item=raw_item, lane=query_plan.lane, store=store)
                if not normalized:
                    continue
                existing = self._find_duplicate(candidates, normalized)
                if existing is None:
                    candidates.append(normalized)
                else:
                    self._merge_candidates(existing, normalized)

        for candidate in candidates:
            self._enrich_with_crossref(candidate=candidate, store=store)
            self._enrich_with_semantic_scholar(candidate=candidate, store=store)
            candidate.retrieval_score = self._score_candidate(candidate, task_spec, entity_pack)

        self.last_diagnostics = self._finalize_diagnostics()
        self.last_provider_health = self._collect_provider_health()
        return self._select_diverse_candidates(candidates, query_plans)

    __call__ = run

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
        oa_url = best_oa_location.get("pdf_url") or best_oa_location.get("landing_page_url")

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
        target.provider_hits = list(dict.fromkeys([*target.provider_hits, *incoming.provider_hits]))
        target.lane_sources = list(dict.fromkeys([*target.lane_sources, *incoming.lane_sources]))
        merged_artifacts = dict(target.provider_artifacts)
        merged_artifacts.update(incoming.provider_artifacts)
        target.provider_artifacts = merged_artifacts

    def _enrich_with_crossref(self, candidate: PaperCandidate, store: QAArtifactStore) -> None:
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

    def _enrich_with_semantic_scholar(self, candidate: PaperCandidate, store: QAArtifactStore) -> None:
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

    def _score_candidate(self, candidate: PaperCandidate, task_spec: TaskSpec, entity_pack: EntityPack) -> float:
        corpus = normalize_text(f"{candidate.title} {candidate.abstract or ''}").lower()
        anchor_terms = self._collect_anchor_terms(task_spec=task_spec, entity_pack=entity_pack)
        entity_hits = sum(1 for term in anchor_terms if term and term in corpus)
        question_type_hits = sum(
            1 for term in QUESTION_TYPE_MATCH_TERMS.get(task_spec.question_type, ()) if term in corpus
        )
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
                "question_type_hits": question_type_hits,
                "time_window_hit": time_window_hit,
                "lane_feature_hits": lane_feature_hits,
                "abstract_available": abstract_available,
                "doi_complete": doi_complete,
            }
        )

        score = (
            4.0 * min(entity_hits, 3)
            + 2.0 * min(question_type_hits, 2)
            + 1.5 * time_window_hit
            + 1.0 * min(lane_feature_hits, 3)
            + 0.75 * abstract_available
            + 0.5 * doi_complete
        )
        return round(score, 4)

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
