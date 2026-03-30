from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from qa.react_reviewed.common import (
    PROPOSER_CANDIDATE_TARGET,
    PROPOSER_PDF_PROBE_MAX_CANDIDATES,
    PROPOSER_RERANK_TOP_K,
    AnswerSubmission,
    DocumentAcquirerNode,
    EntityPack,
    EvidenceExtractor,
    EvidenceExtractorHandoff,
    EvidenceItem,
    GrobidPaperProfileBuilder,
    PaperCandidate,
    PaperProfile,
    PaperRecord,
    PdfUrlProbeClient,
    QAArtifactStore,
    QueryPlan,
    QueryPlannerExecutionError,
    QueryPlannerNode,
    ReActTrajectory,
    RetrievalDiagnosticRecord,
    RetrieverNode,
    ReviewItem,
    ReviewerBudgetBlocked,
    ReviewerSession,
    Section,
    SectionIndex,
    SectionTextView,
    TaskSpec,
    ThreadPoolExecutor,
    _InflightOperation,
    _batched_search_stage_timeout,
    _candidate_has_downloadable_pdf_signal,
    _compact_text,
    _extract_unpaywall_pdf_url,
    _merge_unique_text,
    _pdf_probe_verdict_rank,
    _review_item_priority_terms,
    _score_proposer_semantic_scholar_candidate,
    copy,
    logger,
    normalize_doi,
    threading,
    time,
    write_profile_failure,
)

class ReactReviewedWorkspace:
    def __init__(
        self,
        *,
        question: str,
        context: Optional[str],
        task_spec: TaskSpec,
        entity_pack: EntityPack,
        entity_resolution_snapshot: Optional[Dict[str, Any]],
        artifact_store: QAArtifactStore,
        query_planner: QueryPlannerNode,
        retriever: RetrieverNode,
        document_acquirer: DocumentAcquirerNode,
        handoff: EvidenceExtractorHandoff,
        evidence_extractor: EvidenceExtractor,
        paper_profile_builder: Optional[GrobidPaperProfileBuilder] = None,
        pdf_probe_client: Optional[Any] = None,
        stage_watchdog_seconds: float = 120.0,
        proposer_candidate_target: int = PROPOSER_CANDIDATE_TARGET,
        proposer_rerank_top_k: int = PROPOSER_RERANK_TOP_K,
        proposer_pdf_probe_enabled: bool = True,
        proposer_pdf_probe_max_candidates: int = PROPOSER_PDF_PROBE_MAX_CANDIDATES,
    ) -> None:
        self.question = question
        self.context = context
        self.task_spec = task_spec
        self.entity_pack = entity_pack
        self.entity_resolution_snapshot = copy.deepcopy(entity_resolution_snapshot or {})
        self.store = artifact_store
        self.query_planner = query_planner
        self.retriever = retriever
        self.document_acquirer = document_acquirer
        self.handoff = handoff
        self.evidence_extractor = evidence_extractor
        self.paper_profile_builder = paper_profile_builder or GrobidPaperProfileBuilder()
        self.pdf_probe_client = pdf_probe_client
        self.stage_watchdog_seconds = max(0.01, float(stage_watchdog_seconds))
        self.proposer_candidate_target = max(1, int(proposer_candidate_target))
        self.proposer_rerank_top_k = max(
            1,
            min(self.proposer_candidate_target, int(proposer_rerank_top_k)),
        )
        self.proposer_pdf_probe_enabled = bool(proposer_pdf_probe_enabled)
        self.proposer_pdf_probe_max_candidates = max(1, int(proposer_pdf_probe_max_candidates))

        self._state_lock = threading.RLock()
        self.query_plans: Dict[str, QueryPlan] = {}
        self.paper_candidates: Dict[str, PaperCandidate] = {}
        self.paper_records: Dict[str, PaperRecord] = {}
        self.paper_profiles: Dict[str, PaperProfile] = {}
        self.section_indices: Dict[str, SectionIndex] = {}
        self.evidence_items: Dict[str, EvidenceItem] = {}
        self.retrieval_diagnostics: List[RetrievalDiagnosticRecord] = []
        self.provider_health: Dict[str, Dict[str, Any]] = {}
        self.execution_warnings: List[str] = []
        self.current_submission: Optional[AnswerSubmission] = None
        self.current_proposer_trajectory: Optional[ReActTrajectory] = None
        self.current_review_items: List[ReviewItem] = []
        self.current_cycle_number: int = 1
        self._ad_hoc_query_plan_ids: Dict[Tuple[str, str], str] = {}
        self._search_result_cache: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
        self._search_warning_cache: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
        self._search_batch_result_cache: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        self._oa_lookup_cache: Dict[str, Dict[str, Any]] = {}
        self._pdf_probe_cache: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        self._acquire_result_cache: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        self._acquire_batch_result_cache: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        self._index_result_cache: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        self._section_read_cache: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
        self._profile_result_cache: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        self._extract_result_cache: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
        self._citation_context_cache: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        self._search_inflight: Dict[Tuple[Any, ...], _InflightOperation] = {}
        self._search_batch_inflight: Dict[Tuple[Any, ...], _InflightOperation] = {}
        self._pdf_probe_inflight: Dict[Tuple[Any, ...], _InflightOperation] = {}
        self._acquire_inflight: Dict[Tuple[Any, ...], _InflightOperation] = {}
        self._acquire_batch_inflight: Dict[Tuple[Any, ...], _InflightOperation] = {}
        self._index_inflight: Dict[Tuple[Any, ...], _InflightOperation] = {}
        self._profile_inflight: Dict[Tuple[Any, ...], _InflightOperation] = {}
        self._extract_inflight: Dict[Tuple[Any, ...], _InflightOperation] = {}
        self._stage_events: List[Dict[str, Any]] = []

    def _register_query_plan(self, query_plan: QueryPlan, *, prefix: str = "qp") -> str:
        with self._state_lock:
            query_plan_id = f"{prefix}_{len(self.query_plans) + 1}"
            self.query_plans[query_plan_id] = query_plan
        return query_plan_id

    def _query_plan_signature(self, query_plan: QueryPlan) -> Tuple[Any, ...]:
        def _normalize_terms(values: Optional[Sequence[Any]]) -> Tuple[str, ...]:
            normalized = {
                self._normalize_cache_text(item)
                for item in list(values or [])
                if self._normalize_cache_text(item)
            }
            return tuple(sorted(normalized))

        return (
            str(query_plan.lane or "").strip().lower(),
            self._normalize_cache_text(query_plan.query_text),
            _normalize_terms(query_plan.must_terms),
            _normalize_terms(query_plan.exclude_terms),
            int(query_plan.year_from) if query_plan.year_from is not None else None,
            int(query_plan.year_to) if query_plan.year_to is not None else None,
        )

    def _find_registered_query_plan_id(self, query_plan: QueryPlan, *, prefix: Optional[str] = None) -> Optional[str]:
        target_signature = self._query_plan_signature(query_plan)
        with self._state_lock:
            for query_plan_id, existing in self.query_plans.items():
                if prefix and not str(query_plan_id).startswith(f"{prefix}_"):
                    continue
                if self._query_plan_signature(existing) == target_signature:
                    return query_plan_id
        return None

    def _merge_diagnostics(self, diagnostics: Sequence[RetrievalDiagnosticRecord]) -> None:
        with self._state_lock:
            self.retrieval_diagnostics.extend(list(diagnostics or []))

    def _merge_provider_health(self, payload: Optional[Dict[str, Dict[str, Any]]]) -> None:
        with self._state_lock:
            for provider_name, provider_payload in dict(payload or {}).items():
                current = dict(self.provider_health.get(provider_name) or {})
                merged = dict(current)
                merged.update(dict(provider_payload or {}))
                self.provider_health[provider_name] = merged

    def _merge_execution_warnings(self, warnings: Optional[Sequence[str]]) -> None:
        with self._state_lock:
            self.execution_warnings = _merge_unique_text(self.execution_warnings, warnings)

    def _record_stage_event(
        self,
        *,
        stage: str,
        status: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload = {
            "timestamp": round(time.time(), 6),
            "stage": str(stage),
            "status": str(status),
            "details": copy.deepcopy(details or {}),
        }
        with self._state_lock:
            self._stage_events.append(payload)
            events_payload = copy.deepcopy(self._stage_events)
        self.store.write_json("diagnostics/runtime_stage_events.json", events_payload)
        self.store.write_json("diagnostics/runtime_stage_status.json", payload)

    def _run_stage(
        self,
        *,
        stage: str,
        operation: Callable[[], Any],
        details: Optional[Dict[str, Any]] = None,
        timeout_seconds: Optional[float] = None,
    ) -> Any:
        start_details = copy.deepcopy(details or {})
        effective_timeout_seconds = max(0.01, float(timeout_seconds or self.stage_watchdog_seconds))
        start_details["watchdog_seconds"] = effective_timeout_seconds
        self._record_stage_event(stage=stage, status="start", details=start_details)
        started_at = time.perf_counter()
        result_holder: Dict[str, Any] = {}
        error_holder: Dict[str, BaseException] = {}
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
        if not done.wait(effective_timeout_seconds):
            elapsed = round(max(0.0, time.perf_counter() - started_at), 3)
            timeout_exc = TimeoutError(
                f"{stage} exceeded stage_watchdog_seconds={effective_timeout_seconds}s "
                f"(elapsed {elapsed}s)"
            )
            timeout_details = copy.deepcopy(details or {})
            timeout_details.update({"elapsed_seconds": elapsed, "error": str(timeout_exc)})
            self._record_stage_event(stage=stage, status="timeout", details=timeout_details)
            raise timeout_exc
        try:
            if "error" in error_holder:
                raise error_holder["error"]
            result = result_holder.get("value")
        except Exception as exc:
            elapsed = round(max(0.0, time.perf_counter() - started_at), 3)
            failure_status = "timeout" if isinstance(exc, TimeoutError) else "failure"
            failure_details = copy.deepcopy(details or {})
            failure_details.update(
                {
                    "elapsed_seconds": elapsed,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            self._record_stage_event(stage=stage, status=failure_status, details=failure_details)
            raise
        elapsed = round(max(0.0, time.perf_counter() - started_at), 3)
        success_details = copy.deepcopy(details or {})
        success_details["elapsed_seconds"] = elapsed
        self._record_stage_event(stage=stage, status="success", details=success_details)
        return result

    def set_review_context(
        self,
        *,
        submission: Optional[AnswerSubmission],
        proposer_trajectory: Optional[ReActTrajectory],
        open_review_items: Optional[Sequence[ReviewItem]],
        cycle_number: int,
    ) -> None:
        with self._state_lock:
            self.current_submission = submission
            self.current_proposer_trajectory = proposer_trajectory
            self.current_review_items = list(open_review_items or [])
            self.current_cycle_number = int(cycle_number)

    def snapshot_mutable_state(self) -> Dict[str, Any]:
        with self._state_lock:
            return {
                "paper_candidates": copy.deepcopy(self.paper_candidates),
                "paper_records": copy.deepcopy(self.paper_records),
                "paper_profiles": copy.deepcopy(self.paper_profiles),
                "section_indices": copy.deepcopy(self.section_indices),
                "evidence_items": copy.deepcopy(self.evidence_items),
                "retrieval_diagnostics": copy.deepcopy(self.retrieval_diagnostics),
                "provider_health": copy.deepcopy(self.provider_health),
                "execution_warnings": list(self.execution_warnings),
                "search_result_cache": copy.deepcopy(self._search_result_cache),
                "search_warning_cache": copy.deepcopy(self._search_warning_cache),
                "search_batch_result_cache": copy.deepcopy(self._search_batch_result_cache),
                "oa_lookup_cache": copy.deepcopy(self._oa_lookup_cache),
                "pdf_probe_cache": copy.deepcopy(self._pdf_probe_cache),
                "acquire_result_cache": copy.deepcopy(self._acquire_result_cache),
                "acquire_batch_result_cache": copy.deepcopy(self._acquire_batch_result_cache),
                "index_result_cache": copy.deepcopy(self._index_result_cache),
                "section_read_cache": copy.deepcopy(self._section_read_cache),
                "profile_result_cache": copy.deepcopy(self._profile_result_cache),
                "extract_result_cache": copy.deepcopy(self._extract_result_cache),
                "citation_context_cache": copy.deepcopy(self._citation_context_cache),
            }

    def restore_mutable_state(self, snapshot: Dict[str, Any], *, write_snapshot: bool = False) -> None:
        with self._state_lock:
            self.paper_candidates = copy.deepcopy(snapshot.get("paper_candidates", {}))
            self.paper_records = copy.deepcopy(snapshot.get("paper_records", {}))
            self.paper_profiles = copy.deepcopy(snapshot.get("paper_profiles", {}))
            self.section_indices = copy.deepcopy(snapshot.get("section_indices", {}))
            self.evidence_items = copy.deepcopy(snapshot.get("evidence_items", {}))
            self.retrieval_diagnostics = copy.deepcopy(snapshot.get("retrieval_diagnostics", []))
            self.provider_health = copy.deepcopy(snapshot.get("provider_health", {}))
            self.execution_warnings = list(snapshot.get("execution_warnings", []))
            self._search_result_cache = copy.deepcopy(snapshot.get("search_result_cache", {}))
            self._search_warning_cache = copy.deepcopy(snapshot.get("search_warning_cache", {}))
            self._search_batch_result_cache = copy.deepcopy(snapshot.get("search_batch_result_cache", {}))
            self._oa_lookup_cache = copy.deepcopy(snapshot.get("oa_lookup_cache", {}))
            self._pdf_probe_cache = copy.deepcopy(snapshot.get("pdf_probe_cache", {}))
            self._acquire_result_cache = copy.deepcopy(snapshot.get("acquire_result_cache", {}))
            self._acquire_batch_result_cache = copy.deepcopy(snapshot.get("acquire_batch_result_cache", {}))
            self._index_result_cache = copy.deepcopy(snapshot.get("index_result_cache", {}))
            self._section_read_cache = copy.deepcopy(snapshot.get("section_read_cache", {}))
            self._profile_result_cache = copy.deepcopy(snapshot.get("profile_result_cache", {}))
            self._extract_result_cache = copy.deepcopy(snapshot.get("extract_result_cache", {}))
            self._citation_context_cache = copy.deepcopy(snapshot.get("citation_context_cache", {}))
            self._search_inflight = {}
            self._search_batch_inflight = {}
            self._pdf_probe_inflight = {}
            self._acquire_inflight = {}
            self._acquire_batch_inflight = {}
            self._index_inflight = {}
            self._profile_inflight = {}
            self._extract_inflight = {}
        if write_snapshot:
            self._write_retrieval_snapshot()

    def write_shared_snapshot(self) -> None:
        self._write_retrieval_snapshot()

    def _normalize_cache_text(self, value: Any) -> str:
        return _compact_text(value).lower()

    def _canonical_section_ids(self, section_ids: Optional[Sequence[str]]) -> Tuple[str, ...]:
        return tuple(str(item).strip() for item in (section_ids or []) if str(item).strip())

    def _canonical_query_batch(
        self,
        *,
        query_plan_id: Optional[str] = None,
        query_plan_ids: Optional[Sequence[str]] = None,
        query_text: Optional[str] = None,
        query_texts: Optional[Sequence[str]] = None,
    ) -> Tuple[List[str], List[str]]:
        normalized_plan_ids: List[str] = []
        single_plan_id = str(query_plan_id or "").strip()
        if single_plan_id:
            normalized_plan_ids.append(single_plan_id)
        for item in list(query_plan_ids or []):
            normalized_item = str(item or "").strip()
            if normalized_item and normalized_item not in normalized_plan_ids:
                normalized_plan_ids.append(normalized_item)

        normalized_query_texts: List[str] = []
        single_query_text = str(query_text or "").strip()
        if single_query_text:
            normalized_query_texts.append(single_query_text)
        for item in list(query_texts or []):
            normalized_item = str(item or "").strip()
            if normalized_item and normalized_item not in normalized_query_texts:
                normalized_query_texts.append(normalized_item)
        return normalized_plan_ids, normalized_query_texts

    def _canonical_paper_id_batch(
        self,
        *,
        paper_id: Optional[str] = None,
        paper_ids: Optional[Sequence[str]] = None,
    ) -> List[str]:
        normalized: List[str] = []
        single_paper_id = str(paper_id or "").strip()
        if single_paper_id:
            normalized.append(single_paper_id)
        for item in list(paper_ids or []):
            normalized_item = str(item or "").strip()
            if normalized_item and normalized_item not in normalized:
                normalized.append(normalized_item)
        return normalized

    def _prepare_cached_operation(
        self,
        *,
        cache: Dict[Tuple[Any, ...], Any],
        inflight_map: Optional[Dict[Tuple[Any, ...], _InflightOperation]],
        cache_key: Tuple[Any, ...],
        session: Optional[ReviewerSession],
        charge_budget: bool,
        tool_name: str,
        requested_via: Optional[str],
    ) -> Tuple[str, Any]:
        with self._state_lock:
            if cache_key in cache:
                return "hit", copy.deepcopy(cache[cache_key])
            if inflight_map is not None:
                inflight = inflight_map.get(cache_key)
                if inflight is not None:
                    return "wait", inflight
            if charge_budget and session is not None and not session.try_charge(
                tool_name=tool_name,
                cache_key=cache_key,
                requested_via=requested_via,
            ):
                return "blocked", session.blocked_payload(
                    tool_name=tool_name,
                    cache_key=cache_key,
                    requested_via=requested_via,
                )
            elif session is not None and not charge_budget:
                session.record_miss(
                    tool_name=tool_name,
                    cache_key=cache_key,
                    requested_via=requested_via,
                )
            if inflight_map is None:
                return "owner", None
            inflight = _InflightOperation()
            inflight_map[cache_key] = inflight
            return "owner", inflight

    def _wait_for_cached_operation(self, inflight: _InflightOperation) -> Any:
        inflight.event.wait()
        if inflight.error is not None:
            raise inflight.error
        return copy.deepcopy(inflight.result)

    def _finalize_cached_operation(
        self,
        *,
        inflight_map: Dict[Tuple[Any, ...], _InflightOperation],
        cache_key: Tuple[Any, ...],
        result: Any = None,
        error: Optional[BaseException] = None,
    ) -> None:
        with self._state_lock:
            inflight = inflight_map.pop(cache_key, None)
        if inflight is None:
            return
        inflight.result = copy.deepcopy(result)
        inflight.error = error
        inflight.event.set()

    def _build_retriever_clone(self) -> Any:
        if isinstance(self.retriever, RetrieverNode):
            return RetrieverNode(
                openalex_client=self.retriever.openalex_client,
                crossref_client=self.retriever.crossref_client,
                semantic_scholar_client=self.retriever.semantic_scholar_client,
                per_lane_limit=max(int(self.retriever.per_lane_limit), self.proposer_candidate_target),
                final_top_k=max(int(self.retriever.final_top_k), self.proposer_candidate_target),
                lane_reserve=self.retriever.lane_reserve,
                title_similarity_threshold=self.retriever.title_similarity_threshold,
                max_enrichment_candidates=self.retriever.max_enrichment_candidates,
            )
        return self.retriever

    def _build_document_acquirer_clone(self) -> Any:
        if isinstance(self.document_acquirer, DocumentAcquirerNode):
            pdf_extractor = self.document_acquirer.pdf_extractor
            pdf_extractor_clone = pdf_extractor
            if hasattr(pdf_extractor, "config"):
                try:
                    pdf_extractor_clone = pdf_extractor.__class__(config=pdf_extractor.config)
                except Exception:
                    pdf_extractor_clone = pdf_extractor
            return DocumentAcquirerNode(
                unpaywall_client=self.document_acquirer.unpaywall_client,
                fetcher=self.document_acquirer.fetcher,
                pdf_extractor=pdf_extractor_clone,
                document_fetch_timeout_seconds=self.document_acquirer.document_fetch_timeout_seconds,
                document_fetch_total_timeout_seconds=self.document_acquirer.document_fetch_total_timeout_seconds,
            )
        return self.document_acquirer

    def _build_pdf_probe_clone(self) -> Optional[Any]:
        if isinstance(self.pdf_probe_client, PdfUrlProbeClient):
            return PdfUrlProbeClient(
                timeout=self.pdf_probe_client.timeout,
                max_redirects=self.pdf_probe_client.max_redirects,
                retry_attempts=self.pdf_probe_client.retry_attempts,
                backoff_base_seconds=self.pdf_probe_client.backoff_base_seconds,
                backoff_max_seconds=self.pdf_probe_client.backoff_max_seconds,
                browser_headers=self.pdf_probe_client.browser_headers,
                range_bytes=self.pdf_probe_client.range_bytes,
            )
        return self.pdf_probe_client

    def _probe_proposer_pdf_url(
        self,
        *,
        candidate: PaperCandidate,
        probe_client: Optional[Any],
        url: Optional[str] = None,
    ) -> Dict[str, Any]:
        resolved_url = _compact_text(url) or _compact_text(getattr(candidate, "open_access_pdf_url", None))
        if not resolved_url or probe_client is None:
            raise RuntimeError(f"paper_id={candidate.paper_id} has no configured PDF probe client or URL.")
        cache_key = ("pdf_probe", self._normalize_cache_text(resolved_url))
        state, payload = self._prepare_cached_operation(
            cache=self._pdf_probe_cache,
            inflight_map=self._pdf_probe_inflight,
            cache_key=cache_key,
            session=None,
            charge_budget=False,
            tool_name="search_papers",
            requested_via="proposer_pdf_probe",
        )
        if state == "wait":
            return dict(self._wait_for_cached_operation(payload) or {})
        if state == "hit":
            return dict(payload or {})

        try:
            result = probe_client.probe(resolved_url)
            payload = {
                "paper_id": candidate.paper_id,
                "url": resolved_url,
                "verdict": str(getattr(result, "verdict", "") or "").strip().lower(),
                "method": str(getattr(result, "method", "") or "").strip().lower(),
                "final_url": _compact_text(getattr(result, "final_url", None)) or resolved_url,
                "status_code": int(getattr(result, "status_code", 0) or 0),
                "content_type": _compact_text(getattr(result, "content_type", None)).lower(),
                "content_disposition": _compact_text(getattr(result, "content_disposition", None)) or None,
                "redirect_count": int(getattr(result, "redirect_count", 0) or 0),
            }
            with self._state_lock:
                self._pdf_probe_cache[cache_key] = copy.deepcopy(payload)
            self._finalize_cached_operation(
                inflight_map=self._pdf_probe_inflight,
                cache_key=cache_key,
                result=payload,
            )
            return payload
        except Exception as exc:
            self._finalize_cached_operation(
                inflight_map=self._pdf_probe_inflight,
                cache_key=cache_key,
                error=exc,
            )
            raise

    def _lookup_unpaywall_payload(
        self,
        *,
        candidate: PaperCandidate,
        artifact_store: QAArtifactStore,
    ) -> Optional[Dict[str, Any]]:
        doi = _compact_text(getattr(candidate, "doi", None))
        if not doi:
            return None
        unpaywall_client = getattr(self.document_acquirer, "unpaywall_client", None)
        if unpaywall_client is None:
            return None
        cache_key = normalize_doi(doi) or doi.lower()
        with self._state_lock:
            cached = copy.deepcopy(self._oa_lookup_cache.get(cache_key))
        if cached is not None:
            payload = cached
        else:
            try:
                payload = unpaywall_client.lookup(doi)
            except Exception:
                logger.debug("proposer_unpaywall_lookup_failed paper_id=%s doi=%s", candidate.paper_id, doi, exc_info=True)
                return None
            if not isinstance(payload, dict):
                return None
            with self._state_lock:
                self._oa_lookup_cache[cache_key] = copy.deepcopy(payload)
        artifact_path = artifact_store.write_json(f"provider_raw/unpaywall/{candidate.paper_id}.json", payload)
        candidate.provider_artifacts = {**candidate.provider_artifacts, "unpaywall": artifact_path}
        candidate.provider_hits = list(dict.fromkeys([*candidate.provider_hits, "unpaywall"]))
        return payload

    def _resolve_proposer_probe_targets(
        self,
        *,
        candidate: PaperCandidate,
        artifact_store: QAArtifactStore,
    ) -> List[Dict[str, str]]:
        targets: List[Dict[str, str]] = []
        seen_urls: set[str] = set()

        def _append_target(url: Optional[str], *, source: str) -> None:
            normalized_url = _compact_text(url)
            if not normalized_url:
                return
            cache_key = self._normalize_cache_text(normalized_url)
            if cache_key in seen_urls:
                return
            seen_urls.add(cache_key)
            targets.append({"source": source, "url": normalized_url})

        _append_target(
            _compact_text(getattr(candidate, "open_access_pdf_url", None))
            or _compact_text(getattr(candidate, "best_oa_pdf_url", None)),
            source=_compact_text(getattr(candidate, "oa_source", None)) or "semantic_scholar",
        )
        unpaywall_payload = self._lookup_unpaywall_payload(candidate=candidate, artifact_store=artifact_store)
        unpaywall_pdf_url = _extract_unpaywall_pdf_url(unpaywall_payload)
        if unpaywall_pdf_url:
            _append_target(unpaywall_pdf_url, source="unpaywall")
            if not _compact_text(getattr(candidate, "open_access_pdf_url", None)):
                candidate.open_access_pdf_url = unpaywall_pdf_url
                candidate.best_oa_pdf_url = unpaywall_pdf_url
                candidate.oa_url = unpaywall_pdf_url
                candidate.oa_eligible = True
                candidate.oa_source = "unpaywall"
                candidate.oa_signal_reason = "unpaywall_best_oa_pdf"
        return targets

    def _search_papers_with_semantic_scholar_only(
        self,
        *,
        retriever: RetrieverNode,
        query_plan: QueryPlan,
        artifact_store: QAArtifactStore,
    ) -> List[PaperCandidate]:
        retriever._diagnostic_map = {}
        retriever._provider_health_fallback = retriever._init_provider_health()
        setattr(retriever, "last_search_warnings", [])
        raw_results = retriever._run_provider_search(
            provider_name="semantic_scholar",
            query_plan=query_plan,
        ) or []
        artifact_store.write_json(
            f"provider_raw/semantic_scholar/search_{query_plan.lane}.json",
            raw_results,
        )

        candidates: List[PaperCandidate] = []
        for raw_item in raw_results:
            candidate = retriever._candidate_from_semantic_scholar(
                raw_item=raw_item,
                lane=query_plan.lane,
                store=artifact_store,
            )
            if candidate is None:
                continue
            existing = retriever._find_duplicate(candidates, candidate)
            if existing is None:
                candidates.append(candidate)
            else:
                retriever._merge_candidates(existing, candidate)

        scored_candidates: List[PaperCandidate] = []
        for candidate in candidates:
            if not _compact_text(getattr(candidate, "doi", None)):
                continue
            candidate.retrieval_score = _score_proposer_semantic_scholar_candidate(
                task_spec=self.task_spec,
                entity_pack=self.entity_pack,
                candidate=candidate,
            )
            scored_candidates.append(candidate)

        scored_candidates.sort(
            key=lambda item: (
                -float(item.retrieval_score or 0.0),
                -int(item.ranking_features.get("citation_count") or 0),
                str(item.paper_id),
            )
        )

        probe_warnings: List[Dict[str, Any]] = []
        pdf_probe_client = self._build_pdf_probe_clone()
        if self.proposer_pdf_probe_enabled and pdf_probe_client is not None:
            probe_candidates = list(scored_candidates[: self.proposer_pdf_probe_max_candidates])
            accepted_candidates: List[PaperCandidate] = []

            with ThreadPoolExecutor(max_workers=max(1, min(len(probe_candidates), 4))) as executor:
                future_map = {
                    executor.submit(
                        self._resolve_proposer_probe_targets,
                        candidate=candidate,
                        artifact_store=artifact_store,
                    ): candidate
                    for candidate in probe_candidates
                }
                for future, candidate in list(future_map.items()):
                    try:
                        probe_targets = list(future.result() or [])
                    except Exception as exc:
                        probe_warnings.append(
                            {
                                "paper_id": candidate.paper_id,
                                "url": _compact_text(getattr(candidate, "open_access_pdf_url", None)),
                                "reason": type(exc).__name__,
                                "message": str(exc),
                            }
                        )
                        continue
                    if not probe_targets:
                        probe_warnings.append(
                            {
                                "paper_id": candidate.paper_id,
                                "url": _compact_text(getattr(candidate, "open_access_pdf_url", None)),
                                "reason": "missing_pdf_url",
                                "message": "candidate had no probeable OA PDF URL after Semantic Scholar and Unpaywall resolution",
                            }
                        )
                        continue
                    accepted = False
                    last_warning: Optional[Dict[str, Any]] = None
                    for target in probe_targets:
                        try:
                            probe_payload = dict(
                                self._probe_proposer_pdf_url(
                                    candidate=candidate,
                                    probe_client=pdf_probe_client,
                                    url=target.get("url"),
                                )
                                or {}
                            )
                        except Exception as exc:
                            last_warning = {
                                "paper_id": candidate.paper_id,
                                "url": target.get("url") or _compact_text(getattr(candidate, "open_access_pdf_url", None)),
                                "reason": type(exc).__name__,
                                "message": str(exc),
                            }
                            continue
                        verdict = _compact_text(probe_payload.get("verdict")).lower()
                        if verdict in {"strong", "weak"}:
                            candidate.pdf_probe_verdict = verdict
                            candidate.pdf_probe_method = _compact_text(probe_payload.get("method")).lower() or None
                            candidate.pdf_probe_final_url = _compact_text(probe_payload.get("final_url")) or None
                            candidate.open_access_pdf_url = _compact_text(target.get("url")) or candidate.open_access_pdf_url
                            candidate.best_oa_pdf_url = candidate.open_access_pdf_url or candidate.best_oa_pdf_url
                            if _compact_text(target.get("source")) == "unpaywall":
                                candidate.oa_source = "unpaywall"
                                candidate.oa_signal_reason = "unpaywall_best_oa_pdf"
                                candidate.oa_eligible = True
                            accepted_candidates.append(candidate)
                            accepted = True
                            break
                        last_warning = {
                            "paper_id": candidate.paper_id,
                            "url": probe_payload.get("url") or target.get("url") or _compact_text(getattr(candidate, "open_access_pdf_url", None)),
                            "reason": "non_pdf",
                            "message": (
                                f"response did not validate as PDF "
                                f"(content_type={probe_payload.get('content_type') or 'unknown'}, "
                                f"status_code={probe_payload.get('status_code') or 'unknown'})"
                            ),
                        }
                    if not accepted and last_warning is not None:
                        probe_warnings.append(last_warning)

            filtered_candidates = accepted_candidates
            retriever.last_diagnostics = retriever._finalize_diagnostics() + [
                RetrievalDiagnosticRecord(
                    provider="pdf_probe",
                    stage="fetch",
                    lane=query_plan.lane,
                    hit_count=sum(1 for item in filtered_candidates if _pdf_probe_verdict_rank(item.pdf_probe_verdict) > 0),
                    failure_count=sum(1 for item in probe_warnings if str(item.get("reason") or "").strip() != "non_pdf"),
                    empty_count=sum(1 for item in probe_warnings if str(item.get("reason") or "").strip() == "non_pdf"),
                    sample_messages=[
                        str(item.get("message") or "").strip()
                        for item in probe_warnings[:3]
                        if str(item.get("message") or "").strip()
                    ],
                )
            ]
            provider_health = retriever._collect_provider_health()
            if hasattr(pdf_probe_client, "health_snapshot"):
                provider_health["pdf_probe"] = dict(pdf_probe_client.health_snapshot() or {})
            retriever.last_provider_health = provider_health
        else:
            filtered_candidates = []
            for candidate in scored_candidates:
                probe_targets = self._resolve_proposer_probe_targets(
                    candidate=candidate,
                    artifact_store=artifact_store,
                )
                if not probe_targets:
                    continue
                candidate.open_access_pdf_url = _compact_text(probe_targets[0].get("url")) or candidate.open_access_pdf_url
                candidate.best_oa_pdf_url = candidate.open_access_pdf_url or candidate.best_oa_pdf_url
                filtered_candidates.append(candidate)
            retriever.last_diagnostics = retriever._finalize_diagnostics()
            retriever.last_provider_health = retriever._collect_provider_health()

        filtered_candidates.sort(
            key=lambda item: (
                -_pdf_probe_verdict_rank(getattr(item, "pdf_probe_verdict", None)),
                -float(item.retrieval_score or 0.0),
                -int(item.ranking_features.get("citation_count") or 0),
                str(item.paper_id),
            )
        )
        setattr(retriever, "last_search_warnings", probe_warnings)
        return filtered_candidates[: self.proposer_candidate_target]

    def _index_downloaded_document(
        self,
        *,
        paper_id: str,
        artifact_store: Optional[QAArtifactStore] = None,
        session: Optional[ReviewerSession] = None,
        charge_budget: bool = False,
        requested_via: Optional[str] = None,
        write_snapshot: bool = True,
    ) -> Dict[str, Any]:
        store = artifact_store or self.store
        normalized_paper_id = str(paper_id or "").strip()
        cache_key = ("index_document", normalized_paper_id)
        state, payload = self._prepare_cached_operation(
            cache=self._index_result_cache,
            inflight_map=self._index_inflight,
            cache_key=cache_key,
            session=session,
            charge_budget=charge_budget,
            tool_name="parse_document",
            requested_via=requested_via,
        )
        if state == "blocked":
            raise ReviewerBudgetBlocked(payload)
        if state == "wait":
            result = self._wait_for_cached_operation(payload)
            if session is not None:
                session.record_hit(tool_name="parse_document", cache_key=cache_key, requested_via=requested_via)
        elif state == "hit":
            result = payload
            if session is not None:
                session.record_hit(tool_name="parse_document", cache_key=cache_key, requested_via=requested_via)
        else:
            with self._state_lock:
                paper_record = self.paper_records.get(normalized_paper_id)
                candidate = self.paper_candidates.get(normalized_paper_id)
                current_index = self.section_indices.get(normalized_paper_id)
            if paper_record is None:
                error = ValueError(f"Unknown paper_id for indexing: {normalized_paper_id}")
                self._finalize_cached_operation(
                    inflight_map=self._index_inflight,
                    cache_key=cache_key,
                    error=error,
                )
                raise error
            if str(paper_record.fulltext_status or "").strip().lower() == "fulltext_indexed":
                result = {
                    "paper_id": paper_record.paper_id,
                    "fulltext_available": paper_record.fulltext_available,
                    "fulltext_status": paper_record.fulltext_status,
                    "section_count": len((current_index.sections if current_index is not None else [])),
                    "artifact_path": paper_record.fulltext_artifact_path,
                }
                with self._state_lock:
                    self._index_result_cache[cache_key] = copy.deepcopy(result)
                self._finalize_cached_operation(
                    inflight_map=self._index_inflight,
                    cache_key=cache_key,
                    result=result,
                )
            else:
                source_artifact_path = str(paper_record.source_artifact_path or "").strip()
                can_index_from_local_pdf = bool(source_artifact_path.lower().endswith(".pdf") and Path(source_artifact_path).exists())
                try:
                    parse_documents = getattr(self.document_acquirer, "parse_documents", None)
                    if callable(parse_documents):
                        updated_records, section_indices = self._run_stage(
                            stage="index_document",
                            details={"paper_id": normalized_paper_id, "requested_via": requested_via},
                            operation=lambda: parse_documents([paper_record], artifact_store=store),
                        )
                        updated_record = updated_records[0]
                        section_index = section_indices[0]
                    else:
                        pdf_extractor = getattr(self.document_acquirer, "pdf_extractor", None)
                        if can_index_from_local_pdf and pdf_extractor is not None:
                            def _index_operation() -> Tuple[PaperRecord, SectionIndex]:
                                pdf_result = pdf_extractor.process(
                                    paper_id=paper_record.paper_id,
                                    pdf_bytes=Path(source_artifact_path).read_bytes(),
                                    artifact_store=store,
                                )
                                section_index = SectionIndex(
                                    paper_id=paper_record.paper_id,
                                    fulltext_status=pdf_result.fulltext_status,
                                    sections=[
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
                                    ],
                                )
                                index_artifact_path = store.write_json(
                                    f"indices/{paper_record.paper_id}.json",
                                    section_index.model_dump(exclude_none=True),
                                )
                                updated_record = paper_record.model_copy(
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
                                        "extraction_warnings": list(pdf_result.warnings),
                                        "fulltext_extractor": pdf_result.extractor,
                                        "ocr_applied": bool(pdf_result.ocr_applied),
                                    }
                                )
                                return updated_record, section_index

                            updated_record, section_index = self._run_stage(
                                stage="index_document",
                                details={"paper_id": normalized_paper_id, "requested_via": requested_via},
                                operation=_index_operation,
                            )
                        else:
                            if candidate is None:
                                raise ValueError(f"Missing candidate for paper_id={normalized_paper_id} re-index fallback.")
                            acquirer = self._build_document_acquirer_clone()
                            def _reindex_operation() -> Tuple[List[PaperRecord], List[SectionIndex]]:
                                try:
                                    return acquirer.run(
                                        candidates=[candidate],
                                        artifact_store=store,
                                        parse_fulltext=True,
                                    )
                                except TypeError:
                                    return acquirer.run(
                                        candidates=[candidate],
                                        artifact_store=store,
                                    )
                            updated_records, section_indices = self._run_stage(
                                stage="index_document",
                                details={"paper_id": normalized_paper_id, "requested_via": requested_via},
                                operation=_reindex_operation,
                            )
                            self._merge_diagnostics(getattr(acquirer, "last_diagnostics", []) or [])
                            self._merge_provider_health(getattr(acquirer, "last_provider_health", {}) or {})
                            self._merge_execution_warnings(getattr(acquirer, "last_execution_warnings", []) or [])
                            updated_record = updated_records[0]
                            section_index = section_indices[0]
                    with self._state_lock:
                        self.paper_records[updated_record.paper_id] = updated_record
                        self.section_indices[section_index.paper_id] = section_index
                        self._section_read_cache = {
                            key: value
                            for key, value in self._section_read_cache.items()
                            if len(key) < 2 or key[1] != normalized_paper_id
                        }
                        result = {
                            "paper_id": updated_record.paper_id,
                            "fulltext_available": updated_record.fulltext_available,
                            "fulltext_status": updated_record.fulltext_status,
                            "section_count": len(section_index.sections),
                            "artifact_path": updated_record.fulltext_artifact_path,
                        }
                        self._acquire_result_cache[("paper", normalized_paper_id)] = copy.deepcopy(result)
                        self._index_result_cache[cache_key] = copy.deepcopy(result)
                    self._finalize_cached_operation(
                        inflight_map=self._index_inflight,
                        cache_key=cache_key,
                        result=result,
                    )
                except Exception as exc:
                    self._finalize_cached_operation(
                        inflight_map=self._index_inflight,
                        cache_key=cache_key,
                        error=exc,
                    )
                    raise
        if write_snapshot:
            self._write_retrieval_snapshot()
        return result

    def plan_queries(self, *, focus: str = "initial") -> List[Dict[str, Any]]:
        try:
            plans = self.query_planner.run(task_spec=self.task_spec, entity_pack=self.entity_pack)
        except QueryPlannerExecutionError as exc:
            self._write_query_planner_failure_artifacts(error=exc)
            raise
        if str(focus or "").strip().lower() == "revision":
            with self._state_lock:
                review_items = [item for item in self.current_review_items if item.status == "open"]
            priority_terms = _review_item_priority_terms(review_items)
            if priority_terms:
                augmented_plans: List[QueryPlan] = []
                for plan in plans:
                    augmented_plans.append(
                        QueryPlan(
                            lane=plan.lane,
                            query_text=" ".join(_merge_unique_text([plan.query_text], priority_terms)),
                            must_terms=_merge_unique_text(list(plan.must_terms or []), priority_terms),
                            exclude_terms=list(plan.exclude_terms or []),
                            year_from=plan.year_from,
                            year_to=plan.year_to,
                            preferred_sources=list(plan.preferred_sources or []),
                        )
                    )
                plans = augmented_plans
        payloads: List[Dict[str, Any]] = []
        for plan in plans:
            query_plan_id = self._find_registered_query_plan_id(plan, prefix="qp")
            if not query_plan_id:
                query_plan_id = self._register_query_plan(plan, prefix="qp")
            payloads.append(
                {
                    "query_plan_id": query_plan_id,
                    "focus": focus,
                    **plan.model_dump(exclude_none=True),
                }
            )
        self._write_retrieval_snapshot()
        return payloads

    def _write_query_planner_failure_artifacts(self, *, error: QueryPlannerExecutionError) -> None:
        debug_payload = dict(error.debug_payload or {})
        self.store.write_json(
            "query_planner/failure.json",
            error.to_payload(),
        )
        self.store.write_json(
            "query_planner/agent_run.json",
            {
                "agent": "QueryPlannerNode",
                "input": {
                    "task_spec": self.task_spec.model_dump(exclude_none=True),
                    "entity_pack": self.entity_pack.model_dump(exclude_none=True),
                },
                "error": error.to_payload(),
                "debug": debug_payload,
            },
        )

    def _ensure_query_plan(self, query_plan_id: Optional[str], query_text: Optional[str], lane: str) -> Tuple[str, QueryPlan]:
        normalized_lane = lane if lane in {"review", "frontier", "data", "contrarian"} else "data"
        if query_plan_id:
            with self._state_lock:
                query_plan = self.query_plans.get(str(query_plan_id))
            if query_plan is None:
                raise ValueError(f"Unknown query_plan_id: {query_plan_id}")
            return str(query_plan_id), query_plan
        cleaned_query = _compact_text(query_text)
        if not cleaned_query:
            raise ValueError("query_text is required when query_plan_id is omitted.")
        ad_hoc_key = (normalized_lane, self._normalize_cache_text(cleaned_query))
        with self._state_lock:
            existing_id = self._ad_hoc_query_plan_ids.get(ad_hoc_key)
            if existing_id:
                existing_plan = self.query_plans.get(existing_id)
                if existing_plan is not None:
                    return existing_id, existing_plan
        query_plan = QueryPlan(
            lane=normalized_lane,
            query_text=cleaned_query,
            must_terms=[],
            exclude_terms=[],
            year_from=self.task_spec.year_from,
            year_to=self.task_spec.year_to,
            preferred_sources=["openalex", "semantic_scholar", "crossref"],
        )
        ad_hoc_id = self._register_query_plan(query_plan, prefix="ad_hoc")
        with self._state_lock:
            self._ad_hoc_query_plan_ids[ad_hoc_key] = ad_hoc_id
        return ad_hoc_id, query_plan

    def search_papers(
        self,
        *,
        query_plan_id: Optional[str] = None,
        query_text: Optional[str] = None,
        lane: str = "data",
        reason: str = "",
        artifact_store: Optional[QAArtifactStore] = None,
        session: Optional[ReviewerSession] = None,
        charge_budget: bool = False,
        requested_via: Optional[str] = None,
        write_snapshot: bool = True,
        stage_watchdog_seconds: Optional[float] = None,
        proposer_only_semantic: bool = False,
    ) -> List[Dict[str, Any]]:
        store = artifact_store or self.store
        resolved_id, query_plan = self._ensure_query_plan(query_plan_id, query_text, lane)
        effective_stage_watchdog_seconds = max(0.01, float(stage_watchdog_seconds or self.stage_watchdog_seconds))
        cache_key = ("query_plan", resolved_id, bool(proposer_only_semantic))
        state, payload = self._prepare_cached_operation(
            cache=self._search_result_cache,
            inflight_map=self._search_inflight,
            cache_key=cache_key,
            session=session,
            charge_budget=charge_budget,
            tool_name="search_papers",
            requested_via=requested_via,
        )
        if state == "blocked":
            raise ReviewerBudgetBlocked(payload)
        if state == "wait":
            candidate_payloads = self._wait_for_cached_operation(payload)
            search_warnings = list(self._search_warning_cache.get(cache_key, []) or [])
            if session is not None:
                session.record_hit(tool_name="search_papers", cache_key=cache_key, requested_via=requested_via)
        elif state == "hit":
            candidate_payloads = payload
            search_warnings = list(self._search_warning_cache.get(cache_key, []) or [])
            if session is not None:
                session.record_hit(tool_name="search_papers", cache_key=cache_key, requested_via=requested_via)
        else:
            retriever = self._build_retriever_clone()
            try:
                def _retriever_operation() -> List[Dict[str, Any]]:
                    if proposer_only_semantic and isinstance(retriever, RetrieverNode):
                        return self._search_papers_with_semantic_scholar_only(
                            retriever=retriever,
                            query_plan=query_plan,
                            artifact_store=store,
                        )
                    retriever_kwargs = {
                        "task_spec": self.task_spec,
                        "entity_pack": self.entity_pack,
                        "query_plans": [query_plan],
                        "artifact_store": store,
                    }
                    if isinstance(retriever, RetrieverNode):
                        retriever_kwargs["max_runtime_seconds"] = max(1.0, effective_stage_watchdog_seconds - 5.0)
                    return retriever.run(**retriever_kwargs)

                candidates = self._run_stage(
                    stage="search_papers",
                    details={
                        "query_plan_id": resolved_id,
                        "lane": query_plan.lane,
                        "requested_via": requested_via,
                    },
                    operation=_retriever_operation,
                    timeout_seconds=effective_stage_watchdog_seconds,
                )
                self._merge_diagnostics(getattr(retriever, "last_diagnostics", []) or [])
                self._merge_provider_health(getattr(retriever, "last_provider_health", {}) or {})
                search_warnings = list(getattr(retriever, "last_search_warnings", []) or [])
                filtered_candidates = list(candidates or [])
                if not proposer_only_semantic:
                    filtered_candidates = [
                        candidate
                        for candidate in filtered_candidates
                        if _candidate_has_downloadable_pdf_signal(candidate)
                    ]
                with self._state_lock:
                    for candidate in filtered_candidates:
                        self.paper_candidates[candidate.paper_id] = candidate
                    candidate_payloads = [
                        candidate.model_dump(exclude_none=True)
                        for candidate in filtered_candidates[: self.proposer_candidate_target]
                    ]
                    self._search_result_cache[cache_key] = copy.deepcopy(candidate_payloads)
                    self._search_warning_cache[cache_key] = copy.deepcopy(search_warnings)
                self._finalize_cached_operation(
                    inflight_map=self._search_inflight,
                    cache_key=cache_key,
                    result=candidate_payloads,
                )
            except Exception as exc:
                self._finalize_cached_operation(
                    inflight_map=self._search_inflight,
                    cache_key=cache_key,
                    error=exc,
                )
                raise
        if write_snapshot:
            self._write_retrieval_snapshot()
        return [
            {
                "query_plan_id": resolved_id,
                "reason": str(reason or "").strip(),
                **item,
            }
            for item in candidate_payloads
        ]

    def search_papers_batch(
        self,
        *,
        query_plan_id: Optional[str] = None,
        query_plan_ids: Optional[Sequence[str]] = None,
        query_text: Optional[str] = None,
        query_texts: Optional[Sequence[str]] = None,
        lane: str = "data",
        reason: str = "",
        artifact_store: Optional[QAArtifactStore] = None,
        session: Optional[ReviewerSession] = None,
        charge_budget: bool = False,
        requested_via: Optional[str] = None,
        write_snapshot: bool = True,
        proposer_only_semantic: bool = False,
    ) -> Dict[str, Any]:
        store = artifact_store or self.store
        normalized_plan_ids, normalized_query_texts = self._canonical_query_batch(
            query_plan_id=query_plan_id,
            query_plan_ids=query_plan_ids,
            query_text=query_text,
            query_texts=query_texts,
        )
        input_count = len(normalized_plan_ids) or len(normalized_query_texts) or 1
        batch_search_timeout = _batched_search_stage_timeout(self.stage_watchdog_seconds, input_count)
        cache_key = (
            "batch_search",
            tuple(normalized_plan_ids),
            tuple(self._normalize_cache_text(item) for item in normalized_query_texts),
            str(lane or "").strip().lower(),
            bool(proposer_only_semantic),
        )
        state, payload = self._prepare_cached_operation(
            cache=self._search_batch_result_cache,
            inflight_map=self._search_batch_inflight,
            cache_key=cache_key,
            session=session,
            charge_budget=charge_budget,
            tool_name="search_papers",
            requested_via=requested_via,
        )
        if state == "blocked":
            raise ReviewerBudgetBlocked(payload)
        if state == "wait":
            result = self._wait_for_cached_operation(payload)
            if session is not None:
                session.record_hit(tool_name="search_papers", cache_key=cache_key, requested_via=requested_via)
        elif state == "hit":
            result = payload
            if session is not None:
                session.record_hit(tool_name="search_papers", cache_key=cache_key, requested_via=requested_via)
        else:
            tasks: List[Tuple[str, str, str]] = []
            if normalized_plan_ids:
                tasks = [("query_plan_id", item, str(item or "").strip()) for item in normalized_plan_ids]
            elif normalized_query_texts:
                tasks = []
                for item in normalized_query_texts:
                    resolved_id, _query_plan = self._ensure_query_plan(None, item, lane)
                    tasks.append(("query_text", item, resolved_id))
            else:
                fallback_id, _query_plan = self._ensure_query_plan(query_plan_id, query_text, lane)
                tasks = [("query_plan_id", str(query_plan_id or "").strip(), fallback_id)]

            papers_by_input: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
            search_warnings: List[Dict[str, Any]] = []
            query_failures: List[Dict[str, Any]] = []

            with ThreadPoolExecutor(max_workers=max(1, min(len(tasks), 4))) as executor:
                future_map = {
                    executor.submit(
                        self.search_papers,
                        query_plan_id=resolved_id,
                        query_text=None,
                        lane=lane,
                        reason=reason,
                        artifact_store=store,
                        session=None,
                        charge_budget=False,
                        requested_via=requested_via,
                        write_snapshot=False,
                        stage_watchdog_seconds=batch_search_timeout,
                        proposer_only_semantic=proposer_only_semantic,
                    ): (kind, value, resolved_id)
                    for kind, value, resolved_id in tasks
                }
                for future, task in list(future_map.items()):
                    kind, value, resolved_id = task
                    try:
                        papers_by_input[task] = list(future.result() or [])
                        search_warnings.extend(
                            copy.deepcopy(
                                self._search_warning_cache.get(
                                    ("query_plan", resolved_id, bool(proposer_only_semantic)),
                                    [],
                                )
                                or []
                            )
                        )
                    except Exception as exc:
                        warning = {
                            kind: value,
                            "reason": type(exc).__name__,
                            "message": str(exc),
                        }
                        search_warnings.append(warning)
                        query_failures.append(warning)
                        papers_by_input[task] = []

            deduped_payloads: List[Dict[str, Any]] = []
            seen_paper_ids: set[str] = set()
            for task in tasks:
                for payload_item in list(papers_by_input.get(task) or []):
                    if not isinstance(payload_item, dict):
                        continue
                    paper_id = str(payload_item.get("paper_id") or "").strip()
                    if paper_id and paper_id in seen_paper_ids:
                        continue
                    if paper_id:
                        seen_paper_ids.add(paper_id)
                    deduped_payloads.append(payload_item)

            if not deduped_payloads and query_failures and len(query_failures) == len(tasks):
                error = RuntimeError(
                    "batch search failed for all queries: "
                    + "; ".join(str(item.get("message") or item.get("reason") or "unknown") for item in query_failures)
                )
                self._finalize_cached_operation(
                    inflight_map=self._search_batch_inflight,
                    cache_key=cache_key,
                    error=error,
                )
                raise error

            result = {
                "papers": deduped_payloads,
                "search_warnings": search_warnings,
                "batch_summary": {
                    "input_query_count": len(tasks),
                    "successful_query_count": len(tasks) - len(query_failures),
                    "failed_query_count": len(query_failures),
                    "deduped_paper_count": len(deduped_payloads),
                },
            }
            with self._state_lock:
                self._search_batch_result_cache[cache_key] = copy.deepcopy(result)
            self._finalize_cached_operation(
                inflight_map=self._search_batch_inflight,
                cache_key=cache_key,
                result=result,
            )
        if write_snapshot:
            self._write_retrieval_snapshot()
        return result

    def download_documents(
        self,
        *,
        paper_id: Optional[str] = None,
        paper_ids: Optional[Sequence[str]] = None,
        artifact_store: Optional[QAArtifactStore] = None,
        session: Optional[ReviewerSession] = None,
        charge_budget: bool = False,
        requested_via: Optional[str] = None,
        write_snapshot: bool = True,
        proposer_pdf_download: bool = False,
    ) -> Dict[str, Any]:
        store = artifact_store or self.store
        normalized_paper_ids = self._canonical_paper_id_batch(paper_id=paper_id, paper_ids=paper_ids)
        cache_key = (
            "batch_download",
            tuple(normalized_paper_ids),
            bool(proposer_pdf_download),
        )
        state, payload = self._prepare_cached_operation(
            cache=self._acquire_batch_result_cache,
            inflight_map=self._acquire_batch_inflight,
            cache_key=cache_key,
            session=session,
            charge_budget=charge_budget,
            tool_name="download_document",
            requested_via=requested_via,
        )
        if state == "blocked":
            raise ReviewerBudgetBlocked(payload)
        if state == "wait":
            result = self._wait_for_cached_operation(payload)
            if session is not None:
                session.record_hit(tool_name="download_document", cache_key=cache_key, requested_via=requested_via)
        elif state == "hit":
            result = payload
            if session is not None:
                session.record_hit(tool_name="download_document", cache_key=cache_key, requested_via=requested_via)
        else:
            if not normalized_paper_ids:
                error = ValueError("download_document requires paper_id or paper_ids.")
                self._finalize_cached_operation(
                    inflight_map=self._acquire_batch_inflight,
                    cache_key=cache_key,
                    error=error,
                )
                raise error
            cached_hit_count = 0
            with self._state_lock:
                for normalized_paper_id in normalized_paper_ids:
                    single_key = ("paper_id", normalized_paper_id)
                    if (
                        single_key in self._acquire_result_cache
                        or (normalized_paper_id in self.paper_records and normalized_paper_id in self.section_indices)
                    ):
                        cached_hit_count += 1

            documents_by_paper: Dict[str, Dict[str, Any]] = {}
            download_warnings: List[Dict[str, Any]] = []
            with ThreadPoolExecutor(max_workers=max(1, min(len(normalized_paper_ids), 4))) as executor:
                future_map = {
                    executor.submit(
                        self.download_document,
                        paper_id=current_paper_id,
                        artifact_store=store,
                        session=None,
                        charge_budget=False,
                        requested_via=requested_via,
                        write_snapshot=False,
                        proposer_pdf_download=proposer_pdf_download,
                    ): current_paper_id
                    for current_paper_id in normalized_paper_ids
                }
                for future, current_paper_id in list(future_map.items()):
                    try:
                        documents_by_paper[current_paper_id] = dict(future.result() or {})
                    except Exception as exc:
                        download_warnings.append(
                            {
                                "paper_id": current_paper_id,
                                "status": "failure",
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            }
                        )

            documents = [
                documents_by_paper[paper_id]
                for paper_id in normalized_paper_ids
                if paper_id in documents_by_paper
            ]
            if not documents and download_warnings and len(download_warnings) == len(normalized_paper_ids):
                error = RuntimeError(
                    "batch download failed for all papers: "
                    + "; ".join(
                        f"{item.get('paper_id')}: {item.get('error') or item.get('error_type') or 'unknown'}"
                        for item in download_warnings
                    )
                )
                self._finalize_cached_operation(
                    inflight_map=self._acquire_batch_inflight,
                    cache_key=cache_key,
                    error=error,
                )
                raise error
            result = {
                "documents": documents,
                "download_warnings": download_warnings,
                "batch_summary": {
                    "requested_count": len(normalized_paper_ids),
                    "successful_count": len(documents),
                    "failed_count": len(download_warnings),
                    "cached_hit_count": cached_hit_count,
                },
            }
            with self._state_lock:
                self._acquire_batch_result_cache[cache_key] = copy.deepcopy(result)
            self._finalize_cached_operation(
                inflight_map=self._acquire_batch_inflight,
                cache_key=cache_key,
                result=result,
            )
        if write_snapshot:
            self._write_retrieval_snapshot()
        return result

    def download_document(
        self,
        *,
        paper_id: str,
        artifact_store: Optional[QAArtifactStore] = None,
        session: Optional[ReviewerSession] = None,
        charge_budget: bool = False,
        requested_via: Optional[str] = None,
        write_snapshot: bool = True,
        proposer_pdf_download: bool = False,
    ) -> Dict[str, Any]:
        store = artifact_store or self.store
        normalized_paper_id = str(paper_id)
        cache_key = ("paper_id", normalized_paper_id)
        with self._state_lock:
            if cache_key not in self._acquire_result_cache and normalized_paper_id in self.paper_records and normalized_paper_id in self.section_indices:
                paper_record = self.paper_records[normalized_paper_id]
                section_index = self.section_indices[normalized_paper_id]
                self._acquire_result_cache[cache_key] = {
                    "paper_id": paper_record.paper_id,
                    "fulltext_available": paper_record.fulltext_available,
                    "fulltext_status": paper_record.fulltext_status,
                    "section_count": len(section_index.sections),
                    "artifact_path": paper_record.fulltext_artifact_path,
                }
        state, payload = self._prepare_cached_operation(
            cache=self._acquire_result_cache,
            inflight_map=self._acquire_inflight,
            cache_key=cache_key,
            session=session,
            charge_budget=charge_budget,
            tool_name="download_document",
            requested_via=requested_via,
        )
        if state == "blocked":
            raise ReviewerBudgetBlocked(payload)
        if state == "wait":
            result = self._wait_for_cached_operation(payload)
            if session is not None:
                session.record_hit(tool_name="download_document", cache_key=cache_key, requested_via=requested_via)
        elif state == "hit":
            result = payload
            if session is not None:
                session.record_hit(tool_name="download_document", cache_key=cache_key, requested_via=requested_via)
        else:
            with self._state_lock:
                candidate = self.paper_candidates.get(normalized_paper_id)
            if candidate is None:
                error = ValueError(f"Unknown paper_id: {paper_id}")
                self._finalize_cached_operation(
                    inflight_map=self._acquire_inflight,
                    cache_key=cache_key,
                    error=error,
                )
                raise error
            acquirer = self._build_document_acquirer_clone()
            try:
                def _download_only_operation() -> Tuple[List[PaperRecord], List[SectionIndex]]:
                    if proposer_pdf_download and hasattr(acquirer, "download_pdf_only_with_fallback"):
                        paper_record, section_index = acquirer.download_pdf_only_with_fallback(
                            candidate=candidate,
                            artifact_store=store,
                        )
                        return [paper_record], [section_index]
                    try:
                        return acquirer.run(
                            candidates=[candidate],
                            artifact_store=store,
                            parse_fulltext=False,
                        )
                    except TypeError:
                        return acquirer.run(
                            candidates=[candidate],
                            artifact_store=store,
                        )

                paper_records, section_indices = self._run_stage(
                    stage="download_document",
                    details={
                        "paper_id": normalized_paper_id,
                        "requested_via": requested_via,
                    },
                    operation=_download_only_operation,
                )
                self._merge_diagnostics(getattr(acquirer, "last_diagnostics", []) or [])
                self._merge_provider_health(getattr(acquirer, "last_provider_health", {}) or {})
                self._merge_execution_warnings(getattr(acquirer, "last_execution_warnings", []) or [])
                with self._state_lock:
                    for paper_record, section_index in zip(paper_records, section_indices):
                        self.paper_records[paper_record.paper_id] = paper_record
                        self.section_indices[section_index.paper_id] = section_index
                    stored_record = self.paper_records[candidate.paper_id]
                    stored_index = self.section_indices[candidate.paper_id]
                    result = {
                        "paper_id": stored_record.paper_id,
                        "fulltext_available": stored_record.fulltext_available,
                        "fulltext_status": stored_record.fulltext_status,
                        "section_count": len(stored_index.sections),
                        "artifact_path": stored_record.fulltext_artifact_path,
                    }
                    self._acquire_result_cache[cache_key] = copy.deepcopy(result)
                self._finalize_cached_operation(
                    inflight_map=self._acquire_inflight,
                    cache_key=cache_key,
                    result=result,
                )
            except Exception as exc:
                self._finalize_cached_operation(
                    inflight_map=self._acquire_inflight,
                    cache_key=cache_key,
                    error=exc,
                )
                raise
        if write_snapshot:
            self._write_retrieval_snapshot()
        return result

    def acquire_document(
        self,
        *,
        paper_id: str,
        artifact_store: Optional[QAArtifactStore] = None,
        session: Optional[ReviewerSession] = None,
        charge_budget: bool = False,
        requested_via: Optional[str] = None,
        write_snapshot: bool = True,
    ) -> Dict[str, Any]:
        return self.download_document(
            paper_id=paper_id,
            artifact_store=artifact_store,
            session=session,
            charge_budget=charge_budget,
            requested_via=requested_via,
            write_snapshot=write_snapshot,
        )

    def parse_document(
        self,
        *,
        paper_id: str,
        artifact_store: Optional[QAArtifactStore] = None,
        session: Optional[ReviewerSession] = None,
        charge_budget: bool = False,
        requested_via: Optional[str] = None,
        write_snapshot: bool = True,
    ) -> Dict[str, Any]:
        return self._index_downloaded_document(
            paper_id=str(paper_id),
            artifact_store=artifact_store,
            session=session,
            charge_budget=charge_budget,
            requested_via=requested_via or "parse_document",
            write_snapshot=write_snapshot,
        )

    def parse_documents(
        self,
        *,
        paper_id: Optional[str] = None,
        paper_ids: Optional[Sequence[str]] = None,
        artifact_store: Optional[QAArtifactStore] = None,
        requested_via: Optional[str] = None,
        write_snapshot: bool = True,
    ) -> Dict[str, Any]:
        normalized_paper_ids = self._canonical_paper_id_batch(paper_id=paper_id, paper_ids=paper_ids)
        if not normalized_paper_ids:
            raise ValueError("parse_document requires paper_id or paper_ids.")
        documents_by_paper: Dict[str, Dict[str, Any]] = {}
        parse_warnings: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max(1, min(len(normalized_paper_ids), 4))) as executor:
            future_map = {
                executor.submit(
                    self.parse_document,
                    paper_id=current_paper_id,
                    artifact_store=artifact_store,
                    session=None,
                    charge_budget=False,
                    requested_via=requested_via or "parse_document",
                    write_snapshot=False,
                ): current_paper_id
                for current_paper_id in normalized_paper_ids
            }
            for future, current_paper_id in list(future_map.items()):
                try:
                    documents_by_paper[current_paper_id] = dict(future.result() or {})
                except Exception as exc:
                    parse_warnings.append(
                        {
                            "paper_id": current_paper_id,
                            "status": "failure",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
        documents = [
            documents_by_paper[current_paper_id]
            for current_paper_id in normalized_paper_ids
            if current_paper_id in documents_by_paper
        ]
        if not documents and parse_warnings and len(parse_warnings) == len(normalized_paper_ids):
            raise RuntimeError(
                "batch parse failed for all papers: "
                + "; ".join(
                    f"{item.get('paper_id')}: {item.get('error') or item.get('error_type') or 'unknown'}"
                    for item in parse_warnings
                )
            )
        result = {
            "documents": documents,
            "parse_warnings": parse_warnings,
            "batch_summary": {
                "requested_count": len(normalized_paper_ids),
                "successful_count": len(documents),
                "failed_count": len(parse_warnings),
            },
        }
        if write_snapshot:
            self._write_retrieval_snapshot()
        return result

    def _ensure_document(
        self,
        paper_id: str,
        *,
        require_indexed: bool = False,
        artifact_store: Optional[QAArtifactStore] = None,
        session: Optional[ReviewerSession] = None,
        charge_budget: bool = False,
        requested_via: Optional[str] = None,
        write_snapshot: bool = True,
    ) -> Tuple[PaperRecord, SectionIndex]:
        with self._state_lock:
            has_document = paper_id in self.paper_records and paper_id in self.section_indices
        if not has_document:
            self.download_document(
                paper_id=paper_id,
                artifact_store=artifact_store,
                session=session,
                charge_budget=charge_budget,
                requested_via=requested_via,
                write_snapshot=write_snapshot,
            )
        with self._state_lock:
            paper_record = self.paper_records.get(paper_id)
            section_index = self.section_indices.get(paper_id)
        if require_indexed and paper_record is not None:
            fulltext_status = str(paper_record.fulltext_status or "").strip().lower()
            if fulltext_status != "fulltext_indexed":
                self.parse_document(
                    paper_id=paper_id,
                    artifact_store=artifact_store,
                    session=session,
                    charge_budget=False,
                    requested_via=requested_via or "parse_document",
                    write_snapshot=write_snapshot,
                )
                with self._state_lock:
                    paper_record = self.paper_records.get(paper_id)
                    section_index = self.section_indices.get(paper_id)
        if paper_record is None or section_index is None:
            raise ValueError(f"Failed to download or parse paper_id={paper_id}")
        return paper_record, section_index

    def build_paper_profile(
        self,
        *,
        paper_id: str,
        artifact_store: Optional[QAArtifactStore] = None,
        requested_via: Optional[str] = None,
        write_snapshot: bool = True,
    ) -> Dict[str, Any]:
        store = artifact_store or self.store
        normalized_paper_id = str(paper_id or "").strip()
        paper_record, _ = self._ensure_document(
            normalized_paper_id,
            require_indexed=False,
            artifact_store=artifact_store,
            requested_via=requested_via or "screen_papers",
            write_snapshot=write_snapshot,
        )
        cache_key = ("paper_profile", normalized_paper_id)
        state, payload = self._prepare_cached_operation(
            cache=self._profile_result_cache,
            inflight_map=self._profile_inflight,
            cache_key=cache_key,
            session=None,
            charge_budget=False,
            tool_name="screen_papers",
            requested_via=requested_via,
        )
        if state == "wait":
            result = self._wait_for_cached_operation(payload)
        elif state == "hit":
            result = payload
        else:
            try:
                profile = self._run_stage(
                    stage="build_paper_profile",
                    details={
                        "paper_id": normalized_paper_id,
                        "requested_via": requested_via,
                    },
                    operation=lambda: self.paper_profile_builder.build(
                        paper_record=paper_record,
                        artifact_store=store,
                    ),
                )
                result = profile.model_dump(exclude_none=True)
                with self._state_lock:
                    self.paper_profiles[normalized_paper_id] = profile
                    self._profile_result_cache[cache_key] = copy.deepcopy(result)
                self._finalize_cached_operation(
                    inflight_map=self._profile_inflight,
                    cache_key=cache_key,
                    result=result,
                )
            except Exception as exc:
                failure_artifact_path = write_profile_failure(
                    store=store,
                    paper_record=paper_record,
                    reason=str(exc),
                )
                result = {
                    "paper_id": normalized_paper_id,
                    "title": paper_record.title,
                    "doi": paper_record.doi,
                    "year": paper_record.year,
                    "venue": paper_record.venue,
                    "source_artifact_path": paper_record.source_artifact_path,
                    "profile_status": "error",
                    "error_message": str(exc),
                    "profile_xml_artifact_path": None,
                    "failure_artifact_path": failure_artifact_path,
                }
                with self._state_lock:
                    self._profile_result_cache[cache_key] = copy.deepcopy(result)
                self._finalize_cached_operation(
                    inflight_map=self._profile_inflight,
                    cache_key=cache_key,
                    result=result,
                )
        if write_snapshot:
            self._write_retrieval_snapshot()
        return result

    def read_sections(
        self,
        *,
        paper_id: str,
        section_ids: Optional[Sequence[str]] = None,
        preferred_sections: bool = False,
        artifact_store: Optional[QAArtifactStore] = None,
        session: Optional[ReviewerSession] = None,
        charge_budget: bool = False,
        requested_via: Optional[str] = None,
        write_snapshot: bool = True,
    ) -> List[Dict[str, Any]]:
        paper_record, section_index = self._ensure_document(
            str(paper_id),
            require_indexed=True,
            artifact_store=artifact_store,
            session=session,
            charge_budget=charge_budget,
            requested_via=requested_via or "read_sections",
            write_snapshot=write_snapshot,
        )
        selected_ids = self._canonical_section_ids(section_ids)
        if not preferred_sections and not selected_ids and section_index.sections:
            selected_ids = (section_index.sections[0].section_id,)
        cache_key = ("sections", str(paper_id), selected_ids, bool(preferred_sections))
        state, payload = self._prepare_cached_operation(
            cache=self._section_read_cache,
            inflight_map=None,
            cache_key=cache_key,
            session=session,
            charge_budget=False,
            tool_name="read_sections",
            requested_via=requested_via,
        )
        if state == "hit":
            if session is not None:
                session.record_hit(tool_name="read_sections", cache_key=cache_key, requested_via=requested_via)
            return payload
        def _read_sections_operation() -> List[Dict[str, Any]]:
            if preferred_sections:
                section_views = self.handoff.read_preferred_sections(
                    paper_record=paper_record,
                    section_index=section_index,
                    task_spec=self.task_spec,
                    evidence_is_weak=False,
                    missing_conditions=False,
                )
            else:
                section_views = [
                    self.handoff.read_section_text(
                        paper_record=paper_record,
                        section_index=section_index,
                        section_id=section_id,
                    )
                    for section_id in list(selected_ids)
                ]
                section_views = [view for view in section_views if view is not None]
            if not section_views and paper_record.abstract:
                return [
                    {
                        "paper_id": paper_record.paper_id,
                        "section_id": "sec_abstract",
                        "section_type": "abstract",
                        "heading": "Abstract",
                        "text": paper_record.abstract,
                    }
                ]
            return [
                {
                    "paper_id": view.paper_id,
                    "section_id": view.section_id,
                    "section_type": view.section_type,
                    "heading": view.heading,
                    "text": view.text,
                    "page_start": view.page_start,
                    "page_end": view.page_end,
                }
                for view in section_views
            ]

        payloads = self._run_stage(
            stage="read_sections",
            details={
                "paper_id": str(paper_id),
                "requested_via": requested_via,
                "preferred_sections": bool(preferred_sections),
                "section_ids": list(selected_ids),
            },
            operation=_read_sections_operation,
        )
        with self._state_lock:
            self._section_read_cache[cache_key] = copy.deepcopy(payloads)
        return payloads

    def read_sections_batch(
        self,
        *,
        paper_id: Optional[str] = None,
        paper_ids: Optional[Sequence[str]] = None,
        section_ids: Optional[Sequence[str]] = None,
        preferred_sections: bool = False,
        artifact_store: Optional[QAArtifactStore] = None,
        requested_via: Optional[str] = None,
        write_snapshot: bool = True,
    ) -> Dict[str, Any]:
        normalized_paper_ids = self._canonical_paper_id_batch(paper_id=paper_id, paper_ids=paper_ids)
        if not normalized_paper_ids:
            raise ValueError("read_sections requires paper_id or paper_ids.")
        if len(normalized_paper_ids) > 1 and list(section_ids or []):
            raise ValueError("Batch read_sections does not support section_ids; omit them when using paper_ids.")
        sections_by_paper: Dict[str, List[Dict[str, Any]]] = {}
        section_warnings: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max(1, min(len(normalized_paper_ids), 4))) as executor:
            future_map = {
                executor.submit(
                    self.read_sections,
                    paper_id=current_paper_id,
                    section_ids=section_ids,
                    preferred_sections=preferred_sections,
                    artifact_store=artifact_store,
                    session=None,
                    charge_budget=False,
                    requested_via=requested_via or "read_sections",
                    write_snapshot=False,
                ): current_paper_id
                for current_paper_id in normalized_paper_ids
            }
            for future, current_paper_id in list(future_map.items()):
                try:
                    sections_by_paper[current_paper_id] = list(future.result() or [])
                except Exception as exc:
                    section_warnings.append(
                        {
                            "paper_id": current_paper_id,
                            "status": "failure",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
        sections: List[Dict[str, Any]] = []
        for current_paper_id in normalized_paper_ids:
            sections.extend(sections_by_paper.get(current_paper_id, []))
        if not sections and section_warnings and len(section_warnings) == len(normalized_paper_ids):
            raise RuntimeError(
                "batch read_sections failed for all papers: "
                + "; ".join(
                    f"{item.get('paper_id')}: {item.get('error') or item.get('error_type') or 'unknown'}"
                    for item in section_warnings
                )
            )
        result = {
            "sections": sections,
            "section_warnings": section_warnings,
            "batch_summary": {
                "requested_count": len(normalized_paper_ids),
                "successful_count": len(sections_by_paper),
                "failed_count": len(section_warnings),
            },
        }
        if write_snapshot:
            self._write_retrieval_snapshot()
        return result

    def extract_evidence(
        self,
        *,
        paper_id: str,
        section_ids: Optional[Sequence[str]] = None,
        preferred_sections: bool = False,
        artifact_store: Optional[QAArtifactStore] = None,
        session: Optional[ReviewerSession] = None,
        charge_budget: bool = False,
        requested_via: Optional[str] = None,
        write_snapshot: bool = True,
    ) -> List[Dict[str, Any]]:
        cache_key = ("evidence", str(paper_id), self._canonical_section_ids(section_ids), bool(preferred_sections))
        state, payload = self._prepare_cached_operation(
            cache=self._extract_result_cache,
            inflight_map=self._extract_inflight,
            cache_key=cache_key,
            session=session,
            charge_budget=charge_budget,
            tool_name="extract_evidence",
            requested_via=requested_via,
        )
        if state == "blocked":
            raise ReviewerBudgetBlocked(payload)
        if state == "wait":
            result = self._wait_for_cached_operation(payload)
            if session is not None:
                session.record_hit(tool_name="extract_evidence", cache_key=cache_key, requested_via=requested_via)
        elif state == "hit":
            result = payload
            if session is not None:
                session.record_hit(tool_name="extract_evidence", cache_key=cache_key, requested_via=requested_via)
        else:
            paper_record, section_index = self._ensure_document(
                str(paper_id),
                require_indexed=True,
                artifact_store=artifact_store,
                session=session,
                charge_budget=False,
                requested_via="extract_evidence",
                write_snapshot=write_snapshot,
            )
            try:
                def _extract_evidence_operation() -> List[EvidenceItem]:
                    if not section_ids and not preferred_sections:
                        return self.evidence_extractor.run(
                            task_spec=self.task_spec,
                            entity_pack=self.entity_pack,
                            paper_record=paper_record,
                            section_index=section_index,
                        )
                    evidence_items: List[EvidenceItem] = []
                    for section_payload in self.read_sections(
                        paper_id=paper_id,
                        section_ids=section_ids,
                        preferred_sections=preferred_sections,
                        artifact_store=artifact_store,
                        session=session,
                        charge_budget=False,
                        requested_via="extract_evidence",
                        write_snapshot=False,
                    ):
                        if section_payload.get("section_id") == "sec_abstract":
                            fulltext_end = len(str(section_payload.get("text") or ""))
                            section_view = SectionTextView(
                                paper_id=paper_id,
                                section_id="sec_abstract",
                                section_type=section_payload.get("section_type", "abstract"),
                                heading=section_payload.get("heading", "Abstract"),
                                text=section_payload.get("text", ""),
                                page_start=None,
                                page_end=None,
                                fulltext_char_start=0,
                                fulltext_char_end=fulltext_end,
                            )
                        else:
                            view = self.handoff.read_section_text(
                                paper_record=paper_record,
                                section_index=section_index,
                                section_id=str(section_payload["section_id"]),
                            )
                            if view is None:
                                continue
                            section_view = view
                        evidence_items.extend(
                            self.evidence_extractor._extract_from_section(
                                task_spec=self.task_spec,
                                entity_pack=self.entity_pack,
                                paper_record=paper_record,
                                section_view=section_view,
                            )
                        )
                    return evidence_items

                evidence_items = self._run_stage(
                    stage="extract_evidence",
                    details={
                        "paper_id": str(paper_id),
                        "requested_via": requested_via,
                        "preferred_sections": bool(preferred_sections),
                        "section_ids": list(self._canonical_section_ids(section_ids)),
                    },
                    operation=_extract_evidence_operation,
                )
                with self._state_lock:
                    for evidence_item in evidence_items:
                        self.evidence_items[evidence_item.evidence_id] = evidence_item
                    result = [item.model_dump(exclude_none=True) for item in evidence_items]
                    self._extract_result_cache[cache_key] = copy.deepcopy(result)
                self._finalize_cached_operation(
                    inflight_map=self._extract_inflight,
                    cache_key=cache_key,
                    result=result,
                )
            except Exception as exc:
                self._finalize_cached_operation(
                    inflight_map=self._extract_inflight,
                    cache_key=cache_key,
                    error=exc,
                )
                raise
        if write_snapshot:
            self._write_retrieval_snapshot()
        return result

    def extract_evidence_batch(
        self,
        *,
        paper_id: Optional[str] = None,
        paper_ids: Optional[Sequence[str]] = None,
        section_ids: Optional[Sequence[str]] = None,
        preferred_sections: bool = False,
        artifact_store: Optional[QAArtifactStore] = None,
        requested_via: Optional[str] = None,
        write_snapshot: bool = True,
    ) -> Dict[str, Any]:
        normalized_paper_ids = self._canonical_paper_id_batch(paper_id=paper_id, paper_ids=paper_ids)
        if not normalized_paper_ids:
            raise ValueError("extract_evidence requires paper_id or paper_ids.")
        if len(normalized_paper_ids) > 1 and list(section_ids or []):
            raise ValueError("Batch extract_evidence does not support section_ids; omit them when using paper_ids.")
        evidence_by_paper: Dict[str, List[Dict[str, Any]]] = {}
        evidence_warnings: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max(1, min(len(normalized_paper_ids), 4))) as executor:
            future_map = {
                executor.submit(
                    self.extract_evidence,
                    paper_id=current_paper_id,
                    section_ids=section_ids,
                    preferred_sections=preferred_sections,
                    artifact_store=artifact_store,
                    session=None,
                    charge_budget=False,
                    requested_via=requested_via or "extract_evidence",
                    write_snapshot=False,
                ): current_paper_id
                for current_paper_id in normalized_paper_ids
            }
            for future, current_paper_id in list(future_map.items()):
                try:
                    evidence_by_paper[current_paper_id] = list(future.result() or [])
                except Exception as exc:
                    evidence_warnings.append(
                        {
                            "paper_id": current_paper_id,
                            "status": "failure",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
        evidence: List[Dict[str, Any]] = []
        for current_paper_id in normalized_paper_ids:
            evidence.extend(evidence_by_paper.get(current_paper_id, []))
        if not evidence and evidence_warnings and len(evidence_warnings) == len(normalized_paper_ids):
            raise RuntimeError(
                "batch extract_evidence failed for all papers: "
                + "; ".join(
                    f"{item.get('paper_id')}: {item.get('error') or item.get('error_type') or 'unknown'}"
                    for item in evidence_warnings
                )
            )
        result = {
            "evidence": evidence,
            "evidence_warnings": evidence_warnings,
            "batch_summary": {
                "requested_count": len(normalized_paper_ids),
                "successful_count": len(evidence_by_paper),
                "failed_count": len(evidence_warnings),
            },
        }
        if write_snapshot:
            self._write_retrieval_snapshot()
        return result

    def fetch_citation_context(
        self,
        *,
        paper_id: Optional[str] = None,
        section_id: Optional[str] = None,
        evidence_id: Optional[str] = None,
        citation_id: Optional[str] = None,
        artifact_store: Optional[QAArtifactStore] = None,
        session: Optional[ReviewerSession] = None,
        charge_budget: bool = False,
        requested_via: Optional[str] = None,
        write_snapshot: bool = True,
    ) -> Dict[str, Any]:
        del charge_budget
        if evidence_id:
            with self._state_lock:
                evidence_item = self.evidence_items.get(str(evidence_id))
            if evidence_item is None:
                raise ValueError(f"Unknown evidence_id: {evidence_id}")
            return evidence_item.model_dump(exclude_none=True)
        with self._state_lock:
            current_submission = self.current_submission
            evidence_items = dict(self.evidence_items)
        if citation_id and current_submission is not None:
            citation = next(
                (item for item in current_submission.citations if item.citation_id == str(citation_id)),
                None,
            )
            if citation is not None:
                if evidence_id is None and citation.evidence_ids:
                    evidence_payloads = [
                        evidence_items[evidence_ref].model_dump(exclude_none=True)
                        for evidence_ref in list(citation.evidence_ids)
                        if evidence_ref in evidence_items
                    ]
                    if evidence_payloads:
                        return {
                            "citation_id": citation.citation_id,
                            "paper_id": citation.paper_id,
                            "section_ids": list(citation.section_ids or []),
                            "evidence_ids": list(citation.evidence_ids or []),
                            "evidence": evidence_payloads,
                        }
                paper_id = citation.paper_id
                if section_id is None and citation.section_ids:
                    section_id = citation.section_ids[0]
        if not paper_id:
            raise ValueError("paper_id or evidence_id or citation_id is required.")
        cache_key = ("citation_context", str(paper_id), str(section_id or ""), str(evidence_id or ""), str(citation_id or ""))
        state, payload = self._prepare_cached_operation(
            cache=self._citation_context_cache,
            inflight_map=None,
            cache_key=cache_key,
            session=session,
            charge_budget=False,
            tool_name="fetch_citation_context",
            requested_via=requested_via,
        )
        if state == "hit":
            if session is not None:
                session.record_hit(tool_name="fetch_citation_context", cache_key=cache_key, requested_via=requested_via)
            return payload
        sections = self.read_sections(
            paper_id=str(paper_id),
            section_ids=[section_id] if section_id else None,
            preferred_sections=section_id is None,
            artifact_store=artifact_store,
            session=session,
            charge_budget=True,
            requested_via=requested_via or "fetch_citation_context",
            write_snapshot=write_snapshot,
        )
        if not sections:
            raise ValueError(f"No section text available for paper_id={paper_id}")
        first_section = dict(sections[0])
        first_section["paper_id"] = paper_id
        with self._state_lock:
            self._citation_context_cache[cache_key] = copy.deepcopy(first_section)
        return first_section

    def inspect_submission_anchor(
        self,
        *,
        section_id: Optional[str] = None,
        step_number: Optional[int] = None,
        review_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        with self._state_lock:
            current_submission = self.current_submission
            current_proposer_trajectory = self.current_proposer_trajectory
            current_review_items = list(self.current_review_items)
            current_cycle_number = self.current_cycle_number
        payload: Dict[str, Any] = {
            "cycle_number": current_cycle_number,
            "submission": None,
            "trajectory_step": None,
            "review_item": None,
        }
        if current_submission is not None:
            if section_id:
                payload["submission"] = next(
                    (
                        item.model_dump(exclude_none=True)
                        for item in current_submission.sections
                        if item.section_id == str(section_id)
                    ),
                    None,
                )
            else:
                payload["submission"] = current_submission.model_dump(exclude_none=True)
        if current_proposer_trajectory is not None and step_number is not None:
            payload["trajectory_step"] = next(
                (
                    step.to_dict()
                    for step in current_proposer_trajectory.steps
                    if step.step_number == int(step_number)
                ),
                None,
            )
        if review_id:
            payload["review_item"] = next(
                (
                    item.model_dump(exclude_none=True)
                    for item in current_review_items
                    if item.review_id == str(review_id)
                ),
                None,
            )
        return payload

    def analyze_submission_gap(self) -> Dict[str, Any]:
        with self._state_lock:
            current_submission = self.current_submission
            current_review_items = list(self.current_review_items)
        missing_section_ids = [
            section.section_id
            for section in self.task_spec.answer_sections
            if current_submission is None
            or not any(item.section_id == section.section_id for item in current_submission.sections)
        ]
        open_blocking_items = [
            item.model_dump(exclude_none=True)
            for item in current_review_items
            if item.status == "open" and item.severity == "blocking"
        ]
        return {
            "missing_section_ids": missing_section_ids,
            "open_blocking_items": open_blocking_items,
            "open_review_item_count": len([item for item in current_review_items if item.status == "open"]),
        }

    def inspect_entity_cache(
        self,
        *,
        name: Optional[str] = None,
        entity_type: Optional[str] = None,
        limit: int = 10,
    ) -> Dict[str, Any]:
        resolution_index = dict(self.entity_resolution_snapshot.get("resolution_index") or {})
        entries = list(resolution_index.get("entries") or [])
        filtered: List[Dict[str, Any]] = []
        normalized_name = _compact_text(name).lower()
        normalized_type = _compact_text(entity_type).lower()
        for entry in entries:
            if normalized_type and str(entry.get("entity_type") or "").strip().lower() != normalized_type:
                continue
            if normalized_name:
                candidate_texts = [
                    str(entry.get("canonical_name") or ""),
                    str(entry.get("formula") or ""),
                    *(str(value) for value in list(entry.get("aliases") or [])),
                    *(str(value) for value in list(entry.get("query_anchors") or [])),
                ]
                normalized_candidates = {_compact_text(value).lower() for value in candidate_texts if _compact_text(value)}
                if normalized_name not in normalized_candidates:
                    continue
            filtered.append(dict(entry))
        filtered = filtered[: max(1, int(limit or 1))]
        return {
            "count": len(filtered),
            "entries": filtered,
            "provider_calls": list(self.entity_resolution_snapshot.get("provider_calls") or []),
        }

    def diagnostics_summary(self) -> str:
        with self._state_lock:
            diagnostics = list(self.retrieval_diagnostics)
        messages: List[str] = []
        for record in diagnostics:
            if any(getattr(record, field) > 0 for field in ("failure_count", "timeout_count", "skipped_count")):
                parts: List[str] = []
                if record.failure_count:
                    parts.append(f"{record.failure_count} failure")
                if record.timeout_count:
                    parts.append(f"{record.timeout_count} timeout")
                if record.skipped_count:
                    parts.append(f"{record.skipped_count} skipped")
                label = record.lane or record.stage
                messages.append(f"{record.provider} {label} had {', '.join(parts)}")
        if not messages:
            return ""
        return "External literature retrieval encountered issues: " + "; ".join(messages) + "."

    def _write_retrieval_snapshot(self) -> None:
        with self._state_lock:
            query_plans = [
                {"query_plan_id": query_plan_id, **query_plan.model_dump(exclude_none=True)}
                for query_plan_id, query_plan in self.query_plans.items()
            ]
            paper_candidates = [item.model_dump(exclude_none=True) for item in self.paper_candidates.values()]
            paper_records = [item.model_dump(exclude_none=True) for item in self.paper_records.values()]
            section_indices = [item.model_dump(exclude_none=True) for item in self.section_indices.values()]
            retrieval_diagnostics = [item.model_dump(exclude_none=True) for item in self.retrieval_diagnostics]
            provider_health = copy.deepcopy(self.provider_health)
            evidence_items = [item.model_dump(exclude_none=True) for item in self.evidence_items.values()]
            execution_warnings = list(self.execution_warnings)
        self.store.write_json("query_plans.json", query_plans)
        self.store.write_json("paper_candidates.json", paper_candidates)
        self.store.write_json("paper_records.json", paper_records)
        self.store.write_json("section_indices.json", section_indices)
        self.store.write_json("retrieval_diagnostics.json", retrieval_diagnostics)
        self.store.write_json("provider_health.json", provider_health)
        self.store.write_json("evidence_items.json", evidence_items)
        self.store.write_json("execution_warnings.json", execution_warnings)
