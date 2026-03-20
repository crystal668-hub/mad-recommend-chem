from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence

import requests
from dotenv import load_dotenv
from pydantic import Field, field_validator

from qa.artifacts import QAArtifactStore
from qa.facade import QASystem
from qa.runtime import resolve_qa_runtime_config
from qa.state import StrictModel
from qa.synthesis_state import QAResult
from utils import ensure_dir, generate_timestamp, load_config, save_json, setup_logging


ValidationCategory = Literal["PASS_REAL_EVIDENCE", "PASS_DEGRADED", "FAIL_PIPELINE"]
ReachabilityStatus = Literal["reachable", "blocked", "skipped"]
SuiteTier = Literal["core", "extended"]

DEFAULT_LIVE_QA_QUESTIONS = (
    "How does Pt/C affect HER activity in 1 M KOH?",
)
DEFAULT_REACT_REVIEWED_SUITE_FILE = "./qa/resources/live_validation_react_reviewed_suite.yaml"
FIXED_REVIEWER_ROLES = (
    "search_coverage",
    "evidence_trace",
    "reasoning_consistency",
    "counterevidence",
)
COMPLETE_REVIEWER_STATUSES = {"completed", "salvaged"}

REQUIRED_LEDGER_ARTIFACT_FILES = (
    "qa_result.json",
    "runtime_manifest.json",
    "retrieval_diagnostics.json",
    "provider_health.json",
    "evidence_ledger_reviewed.json",
    "synthesis_input_pack.json",
)
REQUIRED_REACT_REVIEWED_ARTIFACT_FILES = (
    "qa_result.json",
    "runtime_manifest.json",
    "retrieval_diagnostics.json",
    "provider_health.json",
    "candidate_submission.json",
    "acceptance_decision.json",
    "submission_trace.json",
    "submission_cycles.json",
    "proposer_trajectory.json",
    "reviewer_trajectories.json",
    "review_statuses.json",
    "final_review_items.json",
    "router/task_spec.json",
    "router/agent_run.json",
    "entity_resolver/entity_pack.json",
    "entity_resolver/resolution_index.json",
    "entity_resolver/provider_calls.json",
    "entity_resolver/seed_suggestions.json",
    "entity_resolver/agent_run.json",
)

GENERIC_LIMITATION_MARKERS = (
    "insufficient evidence",
    "limited evidence",
    "does not support a firm conclusion",
    "accepted evidence is too sparse",
    "output remains partial and conservative",
)

PROVIDER_PREFLIGHT_SPECS: Dict[str, Dict[str, Any]] = {
    "openalex": {
        "config_key": "openalex_mailto",
        "credential_name": "OPENALEX_MAILTO",
        "url": "https://api.openalex.org/works",
        "params_builder": lambda providers: {
            "search": "Pt/C HER 1 M KOH",
            "per-page": 1,
            **({"mailto": providers["openalex_mailto"]} if providers.get("openalex_mailto") else {}),
        },
    },
    "crossref": {
        "config_key": "crossref_mailto",
        "credential_name": "CROSSREF_MAILTO",
        "url": "https://api.crossref.org/works",
        "params_builder": lambda providers: {
            "rows": 1,
            "query.title": "Pt/C HER 1 M KOH",
            **({"mailto": providers["crossref_mailto"]} if providers.get("crossref_mailto") else {}),
        },
    },
    "semantic_scholar": {
        "config_key": "semantic_scholar_api_key",
        "credential_name": "SEMANTIC_SCHOLAR_API_KEY",
        "url": "https://api.semanticscholar.org/graph/v1/paper/search",
        "params_builder": lambda _providers: {
            "query": "Pt/C HER 1 M KOH",
            "limit": 1,
            "fields": "title",
        },
        "headers_builder": lambda providers: (
            {"x-api-key": providers["semantic_scholar_api_key"]}
            if providers.get("semantic_scholar_api_key")
            else None
        ),
    },
    "unpaywall": {
        "config_key": "unpaywall_email",
        "credential_name": "UNPAYWALL_EMAIL",
        "url": "https://api.unpaywall.org/v2/10.1038/nature12373",
        "params_builder": lambda providers: {"email": providers["unpaywall_email"]},
        "requires_credential_for_probe": True,
    },
}


class ProviderPreflightRecord(StrictModel):
    provider: str
    credential_name: Optional[str] = None
    credential_configured: bool = False
    reachability: ReachabilityStatus = "skipped"
    detail: str = ""


class LiveValidationCase(StrictModel):
    case_id: str
    question: str
    question_type: str
    tier: SuiteTier = "core"
    context: Optional[str] = None
    expected_category: List[ValidationCategory] = Field(default_factory=list)
    require_protocol_ok: bool = False
    require_review_completion: bool = False
    allow_provider_degraded: bool = False
    require_real_evidence: bool = False
    max_open_blocking_review_items: Optional[int] = Field(default=None, ge=0)
    require_grounding_ok: bool = False
    require_router_no_fallback: bool = False
    require_router_question_type_match: bool = False
    require_router_non_none_recency: bool = False
    require_entity_not_all_unresolved: bool = False
    require_pubchem_call: bool = False

    @field_validator("expected_category", mode="before")
    @classmethod
    def coerce_expected_category(cls, value):
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        text = str(value).strip()
        return [text] if text else []


class LiveValidationReport(StrictModel):
    question: str
    category: ValidationCategory
    artifact_dir: str
    case_id: Optional[str] = None
    question_type: Optional[str] = None
    tier: Optional[SuiteTier] = None
    workflow_mode: str = "ledger"
    report_path: Optional[str] = None
    qa_result_path: Optional[str] = None
    provider_preflight: Dict[str, ProviderPreflightRecord] = Field(default_factory=dict)
    provider_runtime_health: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    retrieval_diagnostics_summary: str = ""
    citation_count: int = 0
    accepted_claim_count: int = 0
    paper_candidate_count: int = 0
    paper_record_count: int = 0
    insufficient_evidence: bool = False
    generic_answer_only: bool = False
    provider_failure_detected: bool = False
    citation_paper_record_matches: bool = False
    unmatched_citation_ids: List[str] = Field(default_factory=list)
    required_artifacts: Dict[str, bool] = Field(default_factory=dict)
    missing_artifacts: List[str] = Field(default_factory=list)
    execution_warnings: List[str] = Field(default_factory=list)
    final_answer_preview: str = ""
    validation_notes: List[str] = Field(default_factory=list)
    expected_category: List[ValidationCategory] = Field(default_factory=list)
    require_protocol_ok: bool = False
    require_review_completion: bool = False
    allow_provider_degraded: bool = False
    require_real_evidence: bool = False
    max_open_blocking_review_items: Optional[int] = None
    require_grounding_ok: bool = False
    require_router_no_fallback: bool = False
    require_router_question_type_match: bool = False
    require_router_non_none_recency: bool = False
    require_entity_not_all_unresolved: bool = False
    require_pubchem_call: bool = False
    meets_expectations: bool = True
    expectation_failures: List[str] = Field(default_factory=list)
    protocol_ok: bool = True
    protocol_failures: List[str] = Field(default_factory=list)
    workflow_protocol_stage: str = "not_applicable"
    review_completion_status: str = "completed"
    cycle_count: int = 0
    reviewer_status_by_role: Dict[str, str] = Field(default_factory=dict)
    open_blocking_review_item_count: int = 0
    submission_section_count: int = 0
    submission_trace_count: int = 0
    proposer_step_ref_integrity_ok: bool = True
    review_anchor_integrity_ok: bool = True
    reviewer_budget_integrity_ok: bool = True
    grounding_ok: bool = True
    router_ok: bool = True
    entity_ok: bool = True
    provider_ok: bool = True
    router_used_fallback: bool = False
    router_expected_question_type: Optional[str] = None
    router_actual_question_type: Optional[str] = None
    router_question_type_match: bool = True
    router_recency_policy: Optional[str] = None
    router_recency_policy_ok: bool = True
    router_ambiguity_flag_count: int = 0
    entity_resolved_count: int = 0
    entity_unresolved_count: int = 0
    entity_all_mentions_unresolved: bool = False
    entity_ambiguity_flag_count: int = 0
    pubchem_call_count: int = 0
    pubchem_error_count: int = 0
    grounding_failure_category: Optional[str] = None
    error_message: Optional[str] = None


class LiveValidationSuiteReport(StrictModel):
    generated_at: str
    config_path: str
    suite_file: Optional[str] = None
    shadow_config_path: Optional[str] = None
    selected_tier: Optional[str] = None
    selected_case: Optional[str] = None
    questions: List[LiveValidationReport] = Field(default_factory=list)
    summary: Dict[str, int] = Field(default_factory=dict)
    expectation_summary: Dict[str, int] = Field(default_factory=dict)


def validate_live_qa(
    question: str,
    *,
    context: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    config_path: str = "./config/config.yaml",
    artifact_dir: Optional[str] = None,
    system: Optional[Any] = None,
    system_factory: Optional[Callable[..., Any]] = None,
    perform_network_probe: bool = True,
    probe_request_get: Optional[Callable[..., Any]] = None,
    case: Optional[LiveValidationCase] = None,
) -> LiveValidationReport:
    resolved_config = copy.deepcopy(config) if config is not None else load_config(config_path)
    resolved_artifact_dir = str(Path(artifact_dir or _default_artifact_dir(question)))
    store = QAArtifactStore(base_dir=resolved_artifact_dir)
    provider_config = _resolve_provider_config(config=resolved_config, system=system)
    provider_preflight = _run_provider_preflight(
        provider_config,
        perform_network_probe=perform_network_probe,
        request_get=probe_request_get,
    )
    runtime_config = resolve_qa_runtime_config(resolved_config)
    workflow_mode = str(runtime_config.get("workflow_mode") or "ledger").strip() or "ledger"

    try:
        qa_system = system or _build_system(
            config=resolved_config,
            config_path=config_path,
            system_factory=system_factory,
        )
        result = qa_system.run_qa(
            question=question,
            context=context,
            artifact_dir=resolved_artifact_dir,
        )
    except Exception as exc:
        report = LiveValidationReport(
            question=question,
            category="FAIL_PIPELINE",
            artifact_dir=resolved_artifact_dir,
            case_id=case.case_id if case else None,
            question_type=case.question_type if case else None,
            tier=case.tier if case else None,
            workflow_mode=workflow_mode,
            provider_preflight=provider_preflight,
            required_artifacts=_required_artifact_map(Path(resolved_artifact_dir), workflow_mode=workflow_mode),
            missing_artifacts=_missing_artifacts(Path(resolved_artifact_dir), workflow_mode=workflow_mode),
            validation_notes=[
                "QA live validation failed before a complete artifact set was produced.",
            ],
            error_message=str(exc),
        )
        report = _apply_case_expectations(report, case=case)
        report_path = store.write_json("live_validation_report.json", report.model_dump(exclude_none=True))
        return report.model_copy(update={"report_path": report_path})

    artifact_root = _resolve_artifact_root(result=result, requested_artifact_dir=resolved_artifact_dir)
    payloads = _load_validation_payloads(artifact_root)
    report = _build_validation_report(
        question=question,
        artifact_root=artifact_root,
        result=result,
        provider_preflight=provider_preflight,
        payloads=payloads,
        case=case,
    )
    report = _apply_case_expectations(report, case=case)
    report_path = QAArtifactStore(base_dir=artifact_root).write_json(
        "live_validation_report.json",
        report.model_dump(exclude_none=True),
    )
    return report.model_copy(update={"report_path": report_path})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ChemQA live QA validation")
    parser.add_argument(
        "--question",
        action="append",
        help="Validation question. Repeat the flag to validate multiple questions.",
    )
    parser.add_argument("--context", default=None, help="Optional user context or constraints.")
    parser.add_argument("--artifact-dir", default=None, help="Artifact directory for a single-question validation.")
    parser.add_argument(
        "--artifact-root",
        default=None,
        help="Root directory for multi-question validation artifacts.",
    )
    parser.add_argument(
        "--config",
        default="./config/config.yaml",
        help="Path to configuration file.",
    )
    parser.add_argument(
        "--suite-file",
        default=None,
        help="Path to a live-validation suite YAML/JSON file.",
    )
    parser.add_argument(
        "--tier",
        choices=("core", "extended", "all"),
        default="all",
        help="When using a suite, select core, extended, or all cases.",
    )
    parser.add_argument(
        "--case",
        default=None,
        help="Run a single suite case by case_id.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first failed expectation.",
    )
    parser.add_argument(
        "--shadow-config",
        default=None,
        help="Optional overlay config merged on top of --config for this validation run.",
    )
    parser.add_argument(
        "--skip-preflight-probe",
        action="store_true",
        help="Skip provider reachability probes and only report credential configuration.",
    )
    return parser


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    system_factory: Optional[Callable[..., Any]] = None,
    configure_logging: bool = True,
    perform_network_probe: Optional[bool] = None,
) -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)

    manual_question_mode = bool(args.question)
    suite_mode = not manual_question_mode
    if manual_question_mode and (args.suite_file or args.case or args.tier != "all"):
        parser.error("--question cannot be combined with --suite-file, --case, or --tier.")
    if args.artifact_dir and suite_mode:
        parser.error("--artifact-dir only supports --question mode.")

    config = _load_runtime_config(args.config, shadow_config_path=args.shadow_config)
    if configure_logging:
        setup_logging(config)

    probe_enabled = perform_network_probe if perform_network_probe is not None else (not args.skip_preflight_probe)

    reports: List[LiveValidationReport] = []
    if suite_mode:
        suite_file = args.suite_file or DEFAULT_REACT_REVIEWED_SUITE_FILE
        cases = _load_suite_cases(suite_file)
        cases = _filter_suite_cases(cases, tier=args.tier, case_id=args.case)
        if not cases:
            parser.error("No live-validation suite cases matched the requested filters.")

        suite_root = _resolve_suite_root(args=args, item_count=len(cases))
        for index, case in enumerate(cases, start=1):
            question_artifact_dir = str(suite_root / f"{index:02d}_{_slugify(case.case_id)}")
            report = validate_live_qa(
                case.question,
                context=args.context or case.context,
                config=config,
                config_path=args.config,
                artifact_dir=question_artifact_dir,
                system_factory=system_factory,
                perform_network_probe=probe_enabled,
                case=case,
            )
            reports.append(report)
            _print_report(report)
            if args.fail_fast and not report.meets_expectations:
                break

        suite_report = LiveValidationSuiteReport(
            generated_at=generate_timestamp(),
            config_path=args.config,
            suite_file=suite_file,
            shadow_config_path=args.shadow_config,
            selected_tier=args.tier,
            selected_case=args.case,
            questions=reports,
            summary=_summarize_reports(reports),
            expectation_summary=_summarize_expectations(reports),
        )
        suite_report_path = suite_root / "live_validation_suite.json"
        save_json(suite_report.model_dump(exclude_none=True), suite_report_path)
        print(f"Suite report: {suite_report_path}")
        expectation_summary = suite_report.expectation_summary
        return 1 if int(expectation_summary.get("blocking_failures") or 0) > 0 else 0

    questions = list(args.question or DEFAULT_LIVE_QA_QUESTIONS)
    if args.artifact_dir and len(questions) != 1:
        parser.error("--artifact-dir only supports a single --question.")

    suite_root = _resolve_suite_root(args=args, item_count=len(questions))
    for index, question in enumerate(questions, start=1):
        if args.artifact_dir:
            question_artifact_dir = str(Path(args.artifact_dir))
        else:
            question_artifact_dir = str(suite_root / f"{index:02d}_{_slugify(question)}")
        report = validate_live_qa(
            question,
            context=args.context,
            config=config,
            config_path=args.config,
            artifact_dir=question_artifact_dir,
            system_factory=system_factory,
            perform_network_probe=probe_enabled,
        )
        reports.append(report)
        _print_report(report)
        if args.fail_fast and report.category == "FAIL_PIPELINE":
            break

    suite_report = LiveValidationSuiteReport(
        generated_at=generate_timestamp(),
        config_path=args.config,
        shadow_config_path=args.shadow_config,
        questions=reports,
        summary=_summarize_reports(reports),
        expectation_summary=_summarize_expectations(reports),
    )
    suite_report_path = suite_root / "live_validation_suite.json"
    save_json(suite_report.model_dump(exclude_none=True), suite_report_path)
    print(f"Suite report: {suite_report_path}")
    return 1 if any(report.category == "FAIL_PIPELINE" for report in reports) else 0


def _build_system(
    *,
    config: Dict[str, Any],
    config_path: str,
    system_factory: Optional[Callable[..., Any]],
) -> Any:
    factory = system_factory or QASystem
    return factory(config=config, config_path=config_path)


def _load_runtime_config(config_path: str, *, shadow_config_path: Optional[str]) -> Dict[str, Any]:
    config = copy.deepcopy(load_config(config_path))
    if not shadow_config_path:
        return config
    shadow_config = copy.deepcopy(load_config(shadow_config_path))
    return _deep_merge_dicts(config, shadow_config)


def _load_suite_cases(suite_file: str) -> List[LiveValidationCase]:
    payload = load_config(suite_file)
    raw_cases = payload.get("cases") if isinstance(payload, dict) else payload
    if not isinstance(raw_cases, list):
        raise ValueError(f"Live validation suite must contain a top-level 'cases' list: {suite_file}")
    return [LiveValidationCase.model_validate(item) for item in raw_cases]


def _filter_suite_cases(
    cases: Sequence[LiveValidationCase],
    *,
    tier: str,
    case_id: Optional[str],
) -> List[LiveValidationCase]:
    filtered = list(cases)
    if tier in {"core", "extended"}:
        filtered = [case for case in filtered if case.tier == tier]
    if case_id:
        filtered = [case for case in filtered if case.case_id == case_id]
    return filtered


def _deep_merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _resolve_provider_config(*, config: Dict[str, Any], system: Optional[Any]) -> Dict[str, Any]:
    if system is not None:
        qa_config = dict(getattr(system, "qa_config", {}) or {})
        providers = dict(qa_config.get("providers", {}) or {})
        if providers:
            return providers
    return dict(resolve_qa_runtime_config(config).get("providers", {}) or {})


def _run_provider_preflight(
    providers: Dict[str, Any],
    *,
    perform_network_probe: bool,
    request_get: Optional[Callable[..., Any]] = None,
) -> Dict[str, ProviderPreflightRecord]:
    request_fn = request_get or requests.get
    timeout = min(5.0, float(providers.get("http_timeout") or 5.0))
    checks: Dict[str, ProviderPreflightRecord] = {}
    for provider_name, spec in PROVIDER_PREFLIGHT_SPECS.items():
        credential_name = str(spec["credential_name"])
        credential_value = str(providers.get(str(spec["config_key"])) or "").strip()
        credential_configured = bool(credential_value)
        record = ProviderPreflightRecord(
            provider=provider_name,
            credential_name=credential_name,
            credential_configured=credential_configured,
            reachability="skipped",
            detail="Network preflight was skipped.",
        )

        requires_credential = bool(spec.get("requires_credential_for_probe"))
        if requires_credential and not credential_configured:
            record.detail = f"{credential_name} is missing; reachability probe skipped."
            checks[provider_name] = record
            continue

        if not perform_network_probe:
            if not credential_configured:
                record.detail = f"{credential_name} is missing."
            checks[provider_name] = record
            continue

        try:
            response = request_fn(
                spec["url"],
                params=spec.get("params_builder", lambda _providers: None)(providers),
                headers=spec.get("headers_builder", lambda _providers: None)(providers),
                timeout=timeout,
            )
            status_code = int(getattr(response, "status_code", 0) or 0)
            record.reachability = "reachable" if status_code and status_code < 500 else "blocked"
            record.detail = f"HTTP {status_code}" if status_code else "Request completed without a status code."
        except requests.exceptions.RequestException as exc:
            record.reachability = "blocked"
            record.detail = str(exc)
        checks[provider_name] = record
    return checks


def _resolve_artifact_root(*, result: QAResult, requested_artifact_dir: str) -> Path:
    qa_result_path = str(result.artifact_paths.get("qa_result") or "").strip()
    if qa_result_path:
        return Path(qa_result_path).resolve().parent
    return Path(requested_artifact_dir).resolve()


def _load_validation_payloads(artifact_root: Path) -> Dict[str, Any]:
    return {
        "runtime_manifest": _load_json_file(artifact_root / "runtime_manifest.json", default={}),
        "paper_candidates": _load_json_file(artifact_root / "paper_candidates.json", default=[]),
        "paper_records": _load_json_file(artifact_root / "paper_records.json", default=[]),
        "provider_health": _load_json_file(artifact_root / "provider_health.json", default={}),
        "retrieval_diagnostics": _load_json_file(artifact_root / "retrieval_diagnostics.json", default=[]),
        "synthesis_input_pack": _load_json_file(artifact_root / "synthesis_input_pack.json", default={}),
        "candidate_submission": _load_json_file(artifact_root / "candidate_submission.json", default={}),
        "final_submission": _load_json_file(artifact_root / "final_submission.json", default={}),
        "acceptance_decision": _load_json_file(artifact_root / "acceptance_decision.json", default={}),
        "submission_trace": _load_json_file(artifact_root / "submission_trace.json", default=[]),
        "submission_cycles": _load_json_file(artifact_root / "submission_cycles.json", default=[]),
        "proposer_trajectory": _load_json_file(artifact_root / "proposer_trajectory.json", default={}),
        "reviewer_trajectories": _load_json_file(artifact_root / "reviewer_trajectories.json", default={}),
        "review_statuses": _load_json_file(artifact_root / "review_statuses.json", default=[]),
        "final_review_items": _load_json_file(artifact_root / "final_review_items.json", default=[]),
        "router_task_spec": _load_json_file(artifact_root / "router" / "task_spec.json", default={}),
        "router_agent_run": _load_json_file(artifact_root / "router" / "agent_run.json", default={}),
        "router_semantic_stage": _load_json_file(artifact_root / "router" / "semantic_stage.json", default={}),
        "router_localization_stage": _load_json_file(artifact_root / "router" / "localization_stage.json", default={}),
        "router_fallback_reason": _load_json_file(artifact_root / "router" / "fallback_reason.json", default={}),
        "entity_pack": _load_json_file(artifact_root / "entity_resolver" / "entity_pack.json", default={}),
        "entity_resolution_index": _load_json_file(artifact_root / "entity_resolver" / "resolution_index.json", default={}),
        "entity_provider_calls": _load_json_file(artifact_root / "entity_resolver" / "provider_calls.json", default=[]),
        "entity_seed_suggestions": _load_json_file(artifact_root / "entity_resolver" / "seed_suggestions.json", default=[]),
        "entity_agent_run": _load_json_file(artifact_root / "entity_resolver" / "agent_run.json", default={}),
        "evidence_ledger": _load_json_file(
            artifact_root / "evidence_ledger_reviewed.json",
            default=_load_json_file(artifact_root / "evidence_ledger.json", default={}),
        ),
    }


def _select_react_reviewed_submission(*, payloads: Dict[str, Any], result: QAResult) -> tuple[str, Dict[str, Any]]:
    acceptance_status = str(result.acceptance_status or "accepted").strip().lower() or "accepted"
    if str(result.workflow_mode or "").strip() != "react_reviewed":
        return "final_submission", dict(payloads.get("final_submission") or {})
    if acceptance_status == "rejected":
        return "candidate_submission", dict(payloads.get("candidate_submission") or {})
    return "final_submission", dict(payloads.get("final_submission") or {})


def _build_validation_report(
    *,
    question: str,
    artifact_root: Path,
    result: QAResult,
    provider_preflight: Dict[str, ProviderPreflightRecord],
    payloads: Dict[str, Any],
    case: Optional[LiveValidationCase],
) -> LiveValidationReport:
    paper_candidates = list(payloads.get("paper_candidates") or [])
    paper_records = list(payloads.get("paper_records") or [])
    provider_health = dict(payloads.get("provider_health") or {})
    retrieval_diagnostics = list(payloads.get("retrieval_diagnostics") or [])
    synthesis_input_pack = dict(payloads.get("synthesis_input_pack") or {})
    evidence_ledger = dict(payloads.get("evidence_ledger") or {})
    submission_artifact_name, validation_submission = _select_react_reviewed_submission(payloads=payloads, result=result)

    citation_count = len(result.citations)
    accepted_claim_count = _count_accepted_claims(
        evidence_ledger=evidence_ledger,
        synthesis_input_pack=synthesis_input_pack,
        submission_payload=validation_submission,
        result=result,
    )
    unmatched_citation_ids = _find_unmatched_citations(result=result, paper_records=paper_records)
    generic_answer_only = _is_generic_answer(
        final_answer=result.final_answer,
        citation_count=citation_count,
        accepted_claim_count=accepted_claim_count,
    )
    provider_failure_detected = _has_provider_failure(
        provider_preflight=provider_preflight,
        provider_health=provider_health,
        retrieval_diagnostics=retrieval_diagnostics,
        retrieval_summary=result.retrieval_diagnostics_summary,
        execution_warnings=result.execution_warnings,
    )
    required_artifacts = _required_artifact_map(
        artifact_root,
        workflow_mode=result.workflow_mode,
        acceptance_status=result.acceptance_status,
    )
    missing_artifacts = [name for name, present in required_artifacts.items() if not present]
    grounding_state = _build_grounding_state(
        payloads=payloads,
        provider_failure_detected=provider_failure_detected,
        case=case,
    )
    provider_failure_detected = provider_failure_detected or int(grounding_state["pubchem_error_count"]) > 0
    protocol_state = _build_protocol_state(
        artifact_root=artifact_root,
        payloads=payloads,
        result=result,
    )

    validation_notes: List[str] = []
    if missing_artifacts:
        validation_notes.append(f"Missing required artifacts: {', '.join(missing_artifacts)}.")
    if unmatched_citation_ids:
        validation_notes.append("One or more citations do not map to a real paper record.")
    if provider_failure_detected:
        validation_notes.append("Provider/network degradation was detected in the live run.")
    if generic_answer_only:
        validation_notes.append("The final answer is still dominated by fallback limitation language.")
    if grounding_state["grounding_failures"]:
        validation_notes.append(f"Grounding failures: {' | '.join(grounding_state['grounding_failures'])}.")
    if protocol_state["protocol_failures"]:
        validation_notes.append(f"Protocol failures: {' | '.join(protocol_state['protocol_failures'])}.")
    if str(result.workflow_mode or "").strip() == "react_reviewed" and submission_artifact_name == "candidate_submission":
        validation_notes.append("Protocol validation used candidate_submission because acceptance_status=rejected.")

    category = _classify_validation_result(
        result=result,
        paper_candidate_count=len(paper_candidates),
        paper_record_count=len(paper_records),
        citation_count=citation_count,
        accepted_claim_count=accepted_claim_count,
        citation_matches=not unmatched_citation_ids and citation_count > 0,
        generic_answer_only=generic_answer_only,
        provider_failure_detected=provider_failure_detected,
        entity_provider_error_detected=int(grounding_state["pubchem_error_count"]) > 0,
        missing_artifacts=missing_artifacts,
        protocol_ok=bool(protocol_state["protocol_ok"]),
        review_completion_status=str(protocol_state["review_completion_status"]),
        reviewer_status_by_role=dict(protocol_state["reviewer_status_by_role"]),
        final_submission=validation_submission,
        submission_trace_items=list(protocol_state["submission_trace_items"]),
    )
    grounding_failure_category = grounding_state["grounding_failure_category"]
    if grounding_failure_category is None and category == "FAIL_PIPELINE":
        grounding_failure_category = (
            "retrieval_provider_failure" if provider_failure_detected else "review_or_synthesis_failure"
        )

    return LiveValidationReport(
        question=question,
        category=category,
        artifact_dir=str(artifact_root),
        case_id=case.case_id if case else None,
        question_type=case.question_type if case else None,
        tier=case.tier if case else None,
        workflow_mode=str(result.workflow_mode or "ledger"),
        qa_result_path=str(result.artifact_paths.get("qa_result") or artifact_root / "qa_result.json"),
        provider_preflight=provider_preflight,
        provider_runtime_health=provider_health,
        retrieval_diagnostics_summary=str(result.retrieval_diagnostics_summary or "").strip(),
        citation_count=citation_count,
        accepted_claim_count=accepted_claim_count,
        paper_candidate_count=len(paper_candidates),
        paper_record_count=len(paper_records),
        insufficient_evidence=bool(result.insufficient_evidence),
        generic_answer_only=generic_answer_only,
        provider_failure_detected=provider_failure_detected,
        citation_paper_record_matches=not unmatched_citation_ids and citation_count > 0,
        unmatched_citation_ids=unmatched_citation_ids,
        required_artifacts=required_artifacts,
        missing_artifacts=missing_artifacts,
        execution_warnings=list(result.execution_warnings or []),
        final_answer_preview=str(result.final_answer or "").strip()[:280],
        validation_notes=validation_notes,
        protocol_ok=bool(protocol_state["protocol_ok"]),
        protocol_failures=list(protocol_state["protocol_failures"]),
        workflow_protocol_stage=str(protocol_state["workflow_protocol_stage"]),
        review_completion_status=str(protocol_state["review_completion_status"]),
        cycle_count=int(protocol_state["cycle_count"]),
        reviewer_status_by_role=dict(protocol_state["reviewer_status_by_role"]),
        open_blocking_review_item_count=int(protocol_state["open_blocking_review_item_count"]),
        submission_section_count=int(protocol_state["submission_section_count"]),
        submission_trace_count=int(protocol_state["submission_trace_count"]),
        proposer_step_ref_integrity_ok=bool(protocol_state["proposer_step_ref_integrity_ok"]),
        review_anchor_integrity_ok=bool(protocol_state["review_anchor_integrity_ok"]),
        reviewer_budget_integrity_ok=bool(protocol_state["reviewer_budget_integrity_ok"]),
        grounding_ok=bool(grounding_state["grounding_ok"]),
        router_ok=bool(grounding_state["router_ok"]),
        entity_ok=bool(grounding_state["entity_ok"]),
        provider_ok=bool(grounding_state["provider_ok"]),
        router_used_fallback=bool(grounding_state["router_used_fallback"]),
        router_expected_question_type=grounding_state["router_expected_question_type"],
        router_actual_question_type=grounding_state["router_actual_question_type"],
        router_question_type_match=bool(grounding_state["router_question_type_match"]),
        router_recency_policy=grounding_state["router_recency_policy"],
        router_recency_policy_ok=bool(grounding_state["router_recency_policy_ok"]),
        router_ambiguity_flag_count=int(grounding_state["router_ambiguity_flag_count"]),
        entity_resolved_count=int(grounding_state["entity_resolved_count"]),
        entity_unresolved_count=int(grounding_state["entity_unresolved_count"]),
        entity_all_mentions_unresolved=bool(grounding_state["entity_all_mentions_unresolved"]),
        entity_ambiguity_flag_count=int(grounding_state["entity_ambiguity_flag_count"]),
        pubchem_call_count=int(grounding_state["pubchem_call_count"]),
        pubchem_error_count=int(grounding_state["pubchem_error_count"]),
        grounding_failure_category=grounding_failure_category,
    )


def _build_grounding_state(
    *,
    payloads: Dict[str, Any],
    provider_failure_detected: bool,
    case: Optional[LiveValidationCase],
) -> Dict[str, Any]:
    runtime_manifest = dict(payloads.get("runtime_manifest") or {})
    models_manifest = dict(runtime_manifest.get("models") or {})
    router_model_enabled = bool((models_manifest.get("router") or {}).get("enabled"))

    router_task_spec = dict(payloads.get("router_task_spec") or {})
    router_agent_run = dict(payloads.get("router_agent_run") or {})
    router_semantic_stage = dict(payloads.get("router_semantic_stage") or {})
    router_localization_stage = dict(payloads.get("router_localization_stage") or {})
    router_fallback_reason = dict(payloads.get("router_fallback_reason") or {})

    entity_pack = dict(payloads.get("entity_pack") or {})
    entity_provider_calls = list(payloads.get("entity_provider_calls") or [])
    entity_agent_run = dict(payloads.get("entity_agent_run") or {})

    router_expected_question_type = str(case.question_type or "").strip() or None if case else None
    router_actual_question_type = str(router_task_spec.get("question_type") or "").strip() or None
    router_question_type_match = (
        router_expected_question_type is None
        or (router_actual_question_type is not None and router_actual_question_type == router_expected_question_type)
    )
    router_recency_policy = str(router_task_spec.get("recency_policy") or "").strip() or None
    router_recency_policy_ok = router_recency_policy is not None and router_recency_policy != "none"
    router_used_fallback = bool(router_fallback_reason)
    router_ambiguity_flag_count = len(list(router_task_spec.get("ambiguity_flags") or []))

    router_failures: List[str] = []
    if not router_task_spec:
        router_failures.append("router/task_spec.json is missing or unreadable.")
    if not router_agent_run:
        router_failures.append("router/agent_run.json is missing or unreadable.")
    if router_model_enabled:
        if not router_semantic_stage:
            router_failures.append("router/semantic_stage.json is required when router LLM is enabled.")
        if not router_localization_stage:
            router_failures.append("router/localization_stage.json is required when router LLM is enabled.")
    if router_actual_question_type is None:
        router_failures.append("Router did not emit a question_type.")
    if router_used_fallback:
        router_failures.append("Router used fallback output instead of a clean semantic/localization run.")
    if router_expected_question_type and not router_question_type_match:
        router_failures.append(
            f"Router question_type mismatch: expected {router_expected_question_type}, got {router_actual_question_type or 'missing'}."
        )
    if case and case.require_router_non_none_recency and not router_recency_policy_ok:
        router_failures.append("This case requires a non-none router recency policy.")
    router_ok = not router_failures

    entity_records = list(entity_pack.get("entities") or [])
    unresolved_mentions = list(entity_pack.get("unresolved_mentions") or [])
    entity_ambiguity_flags = list(entity_pack.get("entity_ambiguity_flags") or [])
    entity_resolved_count = len(entity_records)
    entity_unresolved_count = len(unresolved_mentions)
    entity_all_mentions_unresolved = entity_resolved_count == 0 and entity_unresolved_count > 0
    entity_ambiguity_flag_count = len(entity_ambiguity_flags)
    pubchem_calls = [
        dict(item or {})
        for item in entity_provider_calls
        if str((item or {}).get("provider") or "").strip().lower() == "pubchem"
    ]
    pubchem_call_count = len(pubchem_calls)
    pubchem_error_count = sum(1 for item in pubchem_calls if str(item.get("status") or "").strip().lower() == "error")

    entity_failures: List[str] = []
    if not entity_pack:
        entity_failures.append("entity_resolver/entity_pack.json is missing or unreadable.")
    if not entity_agent_run:
        entity_failures.append("entity_resolver/agent_run.json is missing or unreadable.")
    if not isinstance(payloads.get("entity_resolution_index"), dict):
        entity_failures.append("entity_resolver/resolution_index.json is missing or unreadable.")
    if not isinstance(payloads.get("entity_provider_calls"), list):
        entity_failures.append("entity_resolver/provider_calls.json is missing or unreadable.")
    if not isinstance(payloads.get("entity_seed_suggestions"), list):
        entity_failures.append("entity_resolver/seed_suggestions.json is missing or unreadable.")
    if case and case.require_entity_not_all_unresolved and entity_all_mentions_unresolved:
        entity_failures.append("All extracted entity mentions remained unresolved in a core case.")
    if case and case.require_pubchem_call and pubchem_call_count < 1:
        entity_failures.append("This case requires at least one real PubChem provider call.")
    entity_ok = not entity_failures

    provider_ok = (not provider_failure_detected) and pubchem_error_count == 0
    grounding_ok = router_ok and entity_ok

    grounding_failure_category = None
    if not router_ok:
        grounding_failure_category = "router_failure"
    elif not entity_ok:
        grounding_failure_category = "entity_resolution_failure"
    elif not provider_ok:
        grounding_failure_category = "retrieval_provider_failure"

    return {
        "grounding_ok": grounding_ok,
        "router_ok": router_ok,
        "entity_ok": entity_ok,
        "provider_ok": provider_ok,
        "router_used_fallback": router_used_fallback,
        "router_expected_question_type": router_expected_question_type,
        "router_actual_question_type": router_actual_question_type,
        "router_question_type_match": router_question_type_match,
        "router_recency_policy": router_recency_policy,
        "router_recency_policy_ok": router_recency_policy_ok,
        "router_ambiguity_flag_count": router_ambiguity_flag_count,
        "entity_resolved_count": entity_resolved_count,
        "entity_unresolved_count": entity_unresolved_count,
        "entity_all_mentions_unresolved": entity_all_mentions_unresolved,
        "entity_ambiguity_flag_count": entity_ambiguity_flag_count,
        "pubchem_call_count": pubchem_call_count,
        "pubchem_error_count": pubchem_error_count,
        "grounding_failure_category": grounding_failure_category,
        "grounding_failures": [*router_failures, *entity_failures],
    }


def _build_protocol_state(
    *,
    artifact_root: Path,
    payloads: Dict[str, Any],
    result: QAResult,
) -> Dict[str, Any]:
    workflow_mode = str(result.workflow_mode or "ledger").strip() or "ledger"
    if workflow_mode != "react_reviewed":
        return {
            "protocol_ok": True,
            "protocol_failures": [],
            "workflow_protocol_stage": "not_applicable",
            "review_completion_status": str(result.review_completion_status or "completed"),
            "cycle_count": 0,
            "reviewer_status_by_role": {},
            "open_blocking_review_item_count": 0,
            "submission_section_count": 0,
            "submission_trace_count": len(list(result.submission_trace or [])),
            "proposer_step_ref_integrity_ok": True,
            "review_anchor_integrity_ok": True,
            "reviewer_budget_integrity_ok": True,
            "submission_trace_items": [item.model_dump(exclude_none=True) for item in list(result.submission_trace or [])],
        }

    protocol_failures: List[str] = []
    workflow_protocol_stage = "completed"

    def record_failure(stage: str, message: str) -> None:
        nonlocal workflow_protocol_stage
        if not protocol_failures:
            workflow_protocol_stage = stage
        protocol_failures.append(message)

    submission_artifact_name, effective_submission = _select_react_reviewed_submission(payloads=payloads, result=result)
    submission_trace_items = _coerce_submission_trace_items(payloads=payloads, result=result)
    submission_cycles = list(payloads.get("submission_cycles") or [])
    proposer_trajectory = dict(payloads.get("proposer_trajectory") or {})
    reviewer_trajectories = dict(payloads.get("reviewer_trajectories") or {})
    review_statuses = list(payloads.get("review_statuses") or [])
    final_review_items = list(payloads.get("final_review_items") or [])

    if workflow_mode != "react_reviewed":
        record_failure("workflow_mode", f"workflow_mode must be react_reviewed, got {workflow_mode!r}.")
    if list(result.claim_trace or []):
        record_failure("claim_trace", "react_reviewed runs must leave claim_trace empty.")

    effective_sections = list(effective_submission.get("sections") or [])
    effective_section_ids = [
        str(section.get("section_id") or "").strip()
        for section in effective_sections
        if str(section.get("section_id") or "").strip()
    ]
    trace_section_ids = [
        str(item.get("section_id") or "").strip()
        for item in submission_trace_items
        if str(item.get("section_id") or "").strip()
    ]
    if not submission_trace_items:
        record_failure("submission_trace", "submission_trace must be present and non-empty.")
    if set(effective_section_ids) != set(trace_section_ids):
        record_failure("submission_trace", f"{submission_artifact_name} sections must align with submission_trace section_ids.")

    proposer_trajectory_id = str(proposer_trajectory.get("trajectory_id") or "").strip()
    proposer_step_numbers = {
        int(step.get("step_number") or 0)
        for step in list(proposer_trajectory.get("steps") or [])
        if int(step.get("step_number") or 0) >= 1
    }
    proposer_step_ref_integrity_ok = True
    all_step_ref_groups: List[List[Dict[str, Any]]] = []
    for item in submission_trace_items:
        all_step_ref_groups.append(list(item.get("step_refs") or []))
    for section in effective_sections:
        all_step_ref_groups.append(list(section.get("step_refs") or []))
    for step_refs in all_step_ref_groups:
        for step_ref in step_refs:
            trajectory_id = str(step_ref.get("trajectory_id") or "").strip()
            step_number = int(step_ref.get("step_number") or 0)
            if not trajectory_id or step_number < 1:
                proposer_step_ref_integrity_ok = False
                continue
            if trajectory_id != proposer_trajectory_id or step_number not in proposer_step_numbers:
                proposer_step_ref_integrity_ok = False
    if not proposer_trajectory_id or not proposer_step_numbers:
        proposer_step_ref_integrity_ok = False
    if not proposer_step_ref_integrity_ok:
        record_failure("proposer_trajectory", "submission step_refs must resolve to proposer_trajectory steps.")

    reviewer_status_by_role: Dict[str, str] = {}
    duplicate_roles: List[str] = []
    for payload in review_statuses:
        role = str(payload.get("reviewer_role") or "").strip()
        status = str(payload.get("status") or "").strip()
        if not role:
            continue
        if role in reviewer_status_by_role:
            duplicate_roles.append(role)
        reviewer_status_by_role[role] = status
    for role in FIXED_REVIEWER_ROLES:
        reviewer_status_by_role.setdefault(role, "missing")
    if duplicate_roles:
        record_failure("reviewers", f"review_statuses contain duplicate reviewer roles: {', '.join(sorted(set(duplicate_roles)))}.")
    actual_roles = {str(payload.get("reviewer_role") or "").strip() for payload in review_statuses if str(payload.get("reviewer_role") or "").strip()}
    if actual_roles != set(FIXED_REVIEWER_ROLES):
        record_failure("reviewers", "review_statuses must cover exactly the 4 fixed reviewer roles.")
    if str(result.review_completion_status or "").strip() != "completed":
        record_failure("review_completion", "review_completion_status must be completed for protocol closure.")
    incomplete_roles = [
        role
        for role, status in reviewer_status_by_role.items()
        if role in FIXED_REVIEWER_ROLES and status not in COMPLETE_REVIEWER_STATUSES
    ]
    if incomplete_roles:
        record_failure(
            "reviewers",
            "all reviewer roles must finish completed or salvaged; non-completed roles: "
            + f"{', '.join(sorted(incomplete_roles))}.",
        )

    successful_roles = {
        role
        for role, status in reviewer_status_by_role.items()
        if status in COMPLETE_REVIEWER_STATUSES
    }
    reviewer_trajectory_keys = {str(role).strip() for role in reviewer_trajectories.keys() if str(role).strip()}
    if reviewer_trajectory_keys != successful_roles:
        record_failure("reviewers", "reviewer_trajectories must align exactly with successful reviewer roles.")

    review_anchor_integrity_ok = True
    effective_section_id_set = set(effective_section_ids)
    for item in final_review_items:
        anchor_kind = str(item.get("anchor_kind") or "").strip()
        target_section_id = str(item.get("target_section_id") or "").strip()
        if anchor_kind == "global":
            continue
        if anchor_kind == "step_section":
            target_trajectory_id = str(item.get("target_trajectory_id") or "").strip()
            target_step_number = int(item.get("target_step_number") or 0)
            if (
                not target_section_id
                or target_section_id not in effective_section_id_set
                or target_trajectory_id != proposer_trajectory_id
                or target_step_number not in proposer_step_numbers
            ):
                review_anchor_integrity_ok = False
        elif anchor_kind == "section_only":
            if not target_section_id or target_section_id not in effective_section_id_set:
                review_anchor_integrity_ok = False
        elif anchor_kind == "missing_section":
            if not target_section_id:
                review_anchor_integrity_ok = False
        else:
            review_anchor_integrity_ok = False
    if not review_anchor_integrity_ok:
        record_failure("review_items", "final_review_items contain unresolved or invalid anchors.")

    cycle_count = len(submission_cycles)
    if cycle_count < 1:
        record_failure("submission_cycles", "submission_cycles must be present and non-empty.")
        final_cycle_number = 0
    else:
        last_cycle = dict(submission_cycles[-1] or {})
        final_cycle_number = int(last_cycle.get("cycle_number") or cycle_count)
        current_submission = dict(last_cycle.get("current_submission") or {})
        if str(current_submission.get("submission_id") or "").strip() != str(effective_submission.get("submission_id") or "").strip():
            record_failure(
                "submission_cycles",
                f"{submission_artifact_name} submission_id must match submission_cycles[-1].current_submission.",
            )

    reviewer_budget_integrity_ok = True
    if final_cycle_number >= 1:
        for role in FIXED_REVIEWER_ROLES:
            cycle_dir = artifact_root / "reviewers" / role / f"cycle_{final_cycle_number}"
            budget_payload = _load_json_file(cycle_dir / "budget_usage.json", default=None)
            status_payload = _load_json_file(cycle_dir / "reviewer_status.json", default=None)
            if not isinstance(budget_payload, dict):
                reviewer_budget_integrity_ok = False
                record_failure("reviewer_budget", f"Missing or unreadable budget artifact for reviewer {role}.")
                continue
            if not isinstance(status_payload, dict):
                reviewer_budget_integrity_ok = False
                record_failure("reviewer_budget", f"Missing or unreadable reviewer_status artifact for reviewer {role}.")
                continue
            actions_used = int(budget_payload.get("actions_used") or 0)
            budget_limit = int(budget_payload.get("budget_limit") or 0)
            if actions_used > budget_limit:
                reviewer_budget_integrity_ok = False
                record_failure("reviewer_budget", f"Reviewer {role} exceeded retrieval budget ({actions_used}>{budget_limit}).")
            if str(status_payload.get("reviewer_role") or "").strip() != role:
                reviewer_budget_integrity_ok = False
                record_failure("reviewer_budget", f"Reviewer {role} local status artifact has mismatched reviewer_role.")
            if reviewer_status_by_role.get(role) in COMPLETE_REVIEWER_STATUSES:
                reviewer_trajectory_payload = _load_json_file(cycle_dir / "reviewer_trajectory.json", default=None)
                if not isinstance(reviewer_trajectory_payload, dict) or not str(reviewer_trajectory_payload.get("trajectory_id") or "").strip():
                    reviewer_budget_integrity_ok = False
                    record_failure(
                        "reviewer_budget",
                        f"Reviewer {role} is {reviewer_status_by_role.get(role)} but reviewer_trajectory artifact is missing.",
                    )
    else:
        reviewer_budget_integrity_ok = False
        record_failure("reviewer_budget", "Final reviewer cycle number could not be resolved.")

    open_blocking_review_item_count = sum(
        1
        for item in final_review_items
        if str(item.get("severity") or "").strip() == "blocking"
        and str(item.get("status") or "").strip() == "open"
    )

    return {
        "protocol_ok": not protocol_failures,
        "protocol_failures": protocol_failures,
        "workflow_protocol_stage": workflow_protocol_stage,
        "review_completion_status": str(result.review_completion_status or ""),
        "cycle_count": cycle_count,
        "reviewer_status_by_role": reviewer_status_by_role,
        "open_blocking_review_item_count": open_blocking_review_item_count,
        "submission_section_count": len(effective_sections),
        "submission_trace_count": len(submission_trace_items),
        "proposer_step_ref_integrity_ok": proposer_step_ref_integrity_ok,
        "review_anchor_integrity_ok": review_anchor_integrity_ok,
        "reviewer_budget_integrity_ok": reviewer_budget_integrity_ok,
        "submission_trace_items": submission_trace_items,
    }


def _coerce_submission_trace_items(*, payloads: Dict[str, Any], result: QAResult) -> List[Dict[str, Any]]:
    payload_trace = payloads.get("submission_trace")
    if isinstance(payload_trace, list) and payload_trace:
        return [dict(item or {}) for item in payload_trace]
    return [item.model_dump(exclude_none=True) for item in list(result.submission_trace or [])]


def _classify_validation_result(
    *,
    result: QAResult,
    paper_candidate_count: int,
    paper_record_count: int,
    citation_count: int,
    accepted_claim_count: int,
    citation_matches: bool,
    generic_answer_only: bool,
    provider_failure_detected: bool,
    entity_provider_error_detected: bool,
    missing_artifacts: Sequence[str],
    protocol_ok: bool,
    review_completion_status: str,
    reviewer_status_by_role: Dict[str, str],
    final_submission: Dict[str, Any],
    submission_trace_items: Sequence[Dict[str, Any]],
) -> ValidationCategory:
    if missing_artifacts:
        return "FAIL_PIPELINE"

    workflow_mode = str(result.workflow_mode or "").strip()
    if workflow_mode == "react_reviewed":
        if not protocol_ok:
            return "FAIL_PIPELINE"
        if review_completion_status != "completed":
            return "FAIL_PIPELINE"
        if any(reviewer_status_by_role.get(role) not in COMPLETE_REVIEWER_STATUSES for role in FIXED_REVIEWER_ROLES):
            return "FAIL_PIPELINE"

        submission_has_cited_section = any(
            list(section.get("citation_ids") or [])
            for section in list(final_submission.get("sections") or [])
        )
        submission_trace_has_cited_section = any(
            list(item.get("citation_ids") or [])
            for item in submission_trace_items
        )
        if (
            paper_candidate_count >= 1
            and paper_record_count >= 1
            and citation_count >= 1
            and accepted_claim_count >= 1
            and citation_matches
            and submission_has_cited_section
            and submission_trace_has_cited_section
            and not result.insufficient_evidence
            and not generic_answer_only
            and not entity_provider_error_detected
        ):
            return "PASS_REAL_EVIDENCE"
        if provider_failure_detected and str(result.final_answer or "").strip():
            return "PASS_DEGRADED"
        return "FAIL_PIPELINE"

    if (
        paper_candidate_count >= 1
        and paper_record_count >= 1
        and citation_count >= 1
        and accepted_claim_count >= 1
        and citation_matches
        and not result.insufficient_evidence
        and not generic_answer_only
    ):
        return "PASS_REAL_EVIDENCE"
    if provider_failure_detected and str(result.final_answer or "").strip():
        return "PASS_DEGRADED"
    return "FAIL_PIPELINE"


def _count_accepted_claims(
    *,
    evidence_ledger: Dict[str, Any],
    synthesis_input_pack: Dict[str, Any],
    submission_payload: Dict[str, Any],
    result: QAResult,
) -> int:
    if str(result.workflow_mode or "").strip() == "react_reviewed":
        accepted_sections = 0
        for section in list(submission_payload.get("sections") or []):
            citation_ids = [str(item).strip() for item in list(section.get("citation_ids") or []) if str(item).strip()]
            if citation_ids:
                accepted_sections += 1
        if accepted_sections:
            return accepted_sections
        return sum(1 for item in list(result.submission_trace or []) if list(item.citation_ids or []))

    accepted_ids = {
        str(claim.get("claim_id") or "").strip()
        for claim in list(evidence_ledger.get("claims") or [])
        if str(claim.get("status") or "").strip() == "accepted"
    }
    for section in list(synthesis_input_pack.get("section_claims") or []):
        for claim_id in list(section.get("accepted_claim_ids") or []):
            cleaned = str(claim_id or "").strip()
            if cleaned:
                accepted_ids.add(cleaned)
    for trace_item in list(result.claim_trace or []):
        if str(trace_item.status or "").strip() == "accepted":
            accepted_ids.add(str(trace_item.claim_id or "").strip())
    return len(accepted_ids)


def _find_unmatched_citations(*, result: QAResult, paper_records: Sequence[Dict[str, Any]]) -> List[str]:
    paper_ids = {str(record.get("paper_id") or "").strip() for record in paper_records if record.get("paper_id")}
    unmatched: List[str] = []
    for citation in list(result.citations or []):
        paper_id = str(citation.paper_id or "").strip()
        if paper_id and paper_id in paper_ids:
            continue
        unmatched.append(str(citation.citation_id or "").strip())
    return unmatched


def _is_generic_answer(*, final_answer: str, citation_count: int, accepted_claim_count: int) -> bool:
    normalized = re.sub(r"\s+", " ", str(final_answer or "").strip().lower())
    if not normalized:
        return True
    if citation_count > 0 or accepted_claim_count > 0:
        return False
    return any(marker in normalized for marker in GENERIC_LIMITATION_MARKERS)


def _has_provider_failure(
    *,
    provider_preflight: Dict[str, ProviderPreflightRecord],
    provider_health: Dict[str, Dict[str, Any]],
    retrieval_diagnostics: Sequence[Dict[str, Any]],
    retrieval_summary: str,
    execution_warnings: Sequence[str],
) -> bool:
    if any(record.reachability == "blocked" for record in provider_preflight.values()):
        return True

    for payload in provider_health.values():
        status = str(payload.get("status") or "").strip().lower()
        if status in {"degraded", "unavailable"}:
            return True

    for diagnostic in retrieval_diagnostics:
        if any(int(diagnostic.get(field) or 0) > 0 for field in ("failure_count", "timeout_count", "skipped_count")):
            return True

    summary_text = " ".join([str(retrieval_summary or ""), *(str(item or "") for item in execution_warnings)]).lower()
    return any(
        token in summary_text
        for token in (
            "external literature retrieval encountered issues",
            "retry exhausted",
            "provider unavailable",
            "network",
            "timeout",
        )
    )


def _apply_case_expectations(
    report: LiveValidationReport,
    *,
    case: Optional[LiveValidationCase],
) -> LiveValidationReport:
    if case is None:
        return report

    failures: List[str] = []
    category_ok = report.category in case.expected_category
    if (
        not category_ok
        and case.allow_provider_degraded
        and report.category == "PASS_DEGRADED"
        and report.provider_failure_detected
    ):
        category_ok = True
    if case.expected_category and not category_ok:
        failures.append(
            f"category {report.category} does not satisfy expected_category={case.expected_category!r}."
        )
    if case.require_protocol_ok and not report.protocol_ok:
        failures.append("protocol_ok is required but the react_reviewed protocol did not close cleanly.")
    if case.require_review_completion and report.review_completion_status != "completed":
        failures.append("review_completion_status must be completed.")
    if case.require_real_evidence and report.category != "PASS_REAL_EVIDENCE":
        failures.append("real-evidence threshold was required but PASS_REAL_EVIDENCE was not reached.")
    if (
        case.max_open_blocking_review_items is not None
        and report.open_blocking_review_item_count > case.max_open_blocking_review_items
    ):
        failures.append(
            "open blocking review items exceeded limit "
            f"({report.open_blocking_review_item_count}>{case.max_open_blocking_review_items})."
        )
    if case.require_grounding_ok and not report.grounding_ok:
        failures.append("grounding_ok is required but router/entity grounding checks failed.")
    if case.require_router_no_fallback and report.router_used_fallback:
        failures.append("router fallback is not allowed for this case.")
    if case.require_router_question_type_match and not report.router_question_type_match:
        failures.append(
            "router question_type did not match the expected case type "
            f"({report.router_actual_question_type or 'missing'} != {case.question_type})."
        )
    if case.require_router_non_none_recency and not report.router_recency_policy_ok:
        failures.append("router recency policy must be non-none for this case.")
    if case.require_entity_not_all_unresolved and report.entity_all_mentions_unresolved:
        failures.append("all extracted entity mentions remained unresolved.")
    if case.require_pubchem_call and report.pubchem_call_count < 1:
        failures.append("at least one real PubChem provider call is required for this case.")

    return report.model_copy(
        update={
            "expected_category": list(case.expected_category),
            "require_protocol_ok": case.require_protocol_ok,
            "require_review_completion": case.require_review_completion,
            "allow_provider_degraded": case.allow_provider_degraded,
            "require_real_evidence": case.require_real_evidence,
            "max_open_blocking_review_items": case.max_open_blocking_review_items,
            "require_grounding_ok": case.require_grounding_ok,
            "require_router_no_fallback": case.require_router_no_fallback,
            "require_router_question_type_match": case.require_router_question_type_match,
            "require_router_non_none_recency": case.require_router_non_none_recency,
            "require_entity_not_all_unresolved": case.require_entity_not_all_unresolved,
            "require_pubchem_call": case.require_pubchem_call,
            "meets_expectations": not failures,
            "expectation_failures": failures,
        }
    )


def _required_artifact_map(
    artifact_root: Path,
    *,
    workflow_mode: str = "ledger",
    acceptance_status: str = "accepted",
) -> Dict[str, bool]:
    if str(workflow_mode or "").strip() == "react_reviewed":
        required_files = list(REQUIRED_REACT_REVIEWED_ARTIFACT_FILES)
        if str(acceptance_status or "accepted").strip() == "accepted":
            required_files.append("final_submission.json")
    else:
        required_files = list(REQUIRED_LEDGER_ARTIFACT_FILES)
    return {name: (artifact_root / name).exists() for name in required_files}


def _missing_artifacts(
    artifact_root: Path,
    *,
    workflow_mode: str = "ledger",
    acceptance_status: str = "accepted",
) -> List[str]:
    return [
        name
        for name, present in _required_artifact_map(
            artifact_root,
            workflow_mode=workflow_mode,
            acceptance_status=acceptance_status,
        ).items()
        if not present
    ]


def _load_json_file(path: Path, *, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return default


def _resolve_suite_root(*, args: argparse.Namespace, item_count: int) -> Path:
    if args.artifact_dir:
        return ensure_dir(Path(args.artifact_dir).parent)
    if args.artifact_root:
        return ensure_dir(Path(args.artifact_root))
    if item_count == 1:
        return ensure_dir(Path("./logs/runs") / generate_timestamp() / "qa_live_validation")
    return ensure_dir(Path("./logs/runs") / generate_timestamp() / "qa_live_validation_suite")


def _default_artifact_dir(question: str) -> Path:
    return ensure_dir(Path("./logs/runs") / generate_timestamp() / "qa_live_validation" / _slugify(question))


def _slugify(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", str(text or "").strip()).strip("_").lower()
    return cleaned[:64] or "live_qa"


def _summarize_reports(reports: Sequence[LiveValidationReport]) -> Dict[str, int]:
    summary = {
        "PASS_REAL_EVIDENCE": 0,
        "PASS_DEGRADED": 0,
        "FAIL_PIPELINE": 0,
    }
    for report in reports:
        summary[report.category] = summary.get(report.category, 0) + 1
    return summary


def _summarize_expectations(reports: Sequence[LiveValidationReport]) -> Dict[str, int]:
    summary = {
        "met_expectations": 0,
        "failed_expectations": 0,
        "blocking_failures": 0,
        "nonblocking_failures": 0,
    }
    for report in reports:
        if not report.expected_category and report.tier is None:
            continue
        if report.meets_expectations:
            summary["met_expectations"] += 1
            continue
        summary["failed_expectations"] += 1
        if report.tier == "extended":
            summary["nonblocking_failures"] += 1
        else:
            summary["blocking_failures"] += 1
    return summary


def _print_report(report: LiveValidationReport) -> None:
    prefix = f"{report.case_id}: " if report.case_id else ""
    print(f"[{report.category}] {prefix}{report.question}")
    print(f"Artifact dir: {report.artifact_dir}")
    if report.question_type:
        print(f"Question type: {report.question_type}")
    print(f"Workflow mode: {report.workflow_mode}")
    print("Provider reachability:")
    for provider_name in ("openalex", "crossref", "semantic_scholar", "unpaywall"):
        record = report.provider_preflight.get(provider_name)
        if record is None:
            continue
        credential_state = "configured" if record.credential_configured else "missing"
        detail = f"{record.reachability}; credential {credential_state}"
        if record.detail:
            detail = f"{detail}; {record.detail}"
        print(f"  - {provider_name}: {detail}")
    print(f"Retrieval diagnostics: {report.retrieval_diagnostics_summary or 'n/a'}")
    print(f"Citations: {report.citation_count}")
    print(f"Accepted claims: {report.accepted_claim_count}")
    if report.workflow_mode == "react_reviewed":
        print(f"Protocol ok: {report.protocol_ok}")
        print(f"Review completion: {report.review_completion_status}")
        print(f"Open blocking review items: {report.open_blocking_review_item_count}")
        print(
            "Grounding:"
            f" ok={report.grounding_ok}"
            f" router_ok={report.router_ok}"
            f" entity_ok={report.entity_ok}"
            f" provider_ok={report.provider_ok}"
        )
        print(
            "Grounding details:"
            f" router_fallback={report.router_used_fallback}"
            f" qtype={report.router_actual_question_type or 'n/a'}"
            f" pubchem_calls={report.pubchem_call_count}"
            f" pubchem_errors={report.pubchem_error_count}"
            f" resolved={report.entity_resolved_count}"
            f" unresolved={report.entity_unresolved_count}"
        )
    if report.expected_category:
        print(f"Expectations met: {report.meets_expectations}")
    if report.validation_notes:
        print(f"Notes: {' '.join(report.validation_notes)}")
    if report.expectation_failures:
        print(f"Expectation failures: {' '.join(report.expectation_failures)}")
    if report.report_path:
        print(f"Report: {report.report_path}")
    print("")


if __name__ == "__main__":
    raise SystemExit(main())
