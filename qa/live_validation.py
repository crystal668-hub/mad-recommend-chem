from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence

import requests
from dotenv import load_dotenv
from pydantic import Field

from qa.artifacts import QAArtifactStore
from qa.facade import QASystem
from qa.runtime import resolve_qa_runtime_config
from qa.state import StrictModel
from qa.synthesis_state import QAResult
from utils import ensure_dir, generate_timestamp, load_config, save_json, setup_logging


ValidationCategory = Literal["PASS_REAL_EVIDENCE", "PASS_DEGRADED", "FAIL_PIPELINE"]
ReachabilityStatus = Literal["reachable", "blocked", "skipped"]

DEFAULT_LIVE_QA_QUESTIONS = (
    "How does Pt/C affect HER activity in 1 M KOH?",
)

REQUIRED_ARTIFACT_FILES = (
    "qa_result.json",
    "runtime_manifest.json",
    "retrieval_diagnostics.json",
    "provider_health.json",
    "synthesis_input_pack.json",
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


class LiveValidationReport(StrictModel):
    question: str
    category: ValidationCategory
    artifact_dir: str
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
    error_message: Optional[str] = None


class LiveValidationSuiteReport(StrictModel):
    generated_at: str
    config_path: str
    questions: List[LiveValidationReport] = Field(default_factory=list)
    summary: Dict[str, int] = Field(default_factory=dict)


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
            provider_preflight=provider_preflight,
            required_artifacts=_required_artifact_map(Path(resolved_artifact_dir)),
            missing_artifacts=_missing_artifacts(Path(resolved_artifact_dir)),
            validation_notes=[
                "QA live validation failed before a complete artifact set was produced.",
            ],
            error_message=str(exc),
        )
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
    )
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

    questions = list(args.question or DEFAULT_LIVE_QA_QUESTIONS)
    if args.artifact_dir and len(questions) != 1:
        parser.error("--artifact-dir only supports a single --question.")

    config = copy.deepcopy(load_config(args.config))
    if configure_logging:
        setup_logging(config)

    probe_enabled = perform_network_probe if perform_network_probe is not None else (not args.skip_preflight_probe)
    suite_root = _resolve_suite_root(args=args, questions=questions)

    reports: List[LiveValidationReport] = []
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

    suite_report = LiveValidationSuiteReport(
        generated_at=generate_timestamp(),
        config_path=args.config,
        questions=reports,
        summary=_summarize_reports(reports),
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
        config_key = str(spec["config_key"])
        credential_name = str(spec["credential_name"])
        credential_value = str(providers.get(config_key) or "").strip()
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
        "paper_candidates": _load_json_file(artifact_root / "paper_candidates.json", default=[]),
        "paper_records": _load_json_file(artifact_root / "paper_records.json", default=[]),
        "provider_health": _load_json_file(artifact_root / "provider_health.json", default={}),
        "retrieval_diagnostics": _load_json_file(artifact_root / "retrieval_diagnostics.json", default=[]),
        "synthesis_input_pack": _load_json_file(artifact_root / "synthesis_input_pack.json", default={}),
        "evidence_ledger": _load_json_file(
            artifact_root / "evidence_ledger_reviewed.json",
            default=_load_json_file(artifact_root / "evidence_ledger.json", default={}),
        ),
    }


def _build_validation_report(
    *,
    question: str,
    artifact_root: Path,
    result: QAResult,
    provider_preflight: Dict[str, ProviderPreflightRecord],
    payloads: Dict[str, Any],
) -> LiveValidationReport:
    paper_candidates = list(payloads.get("paper_candidates") or [])
    paper_records = list(payloads.get("paper_records") or [])
    provider_health = dict(payloads.get("provider_health") or {})
    retrieval_diagnostics = list(payloads.get("retrieval_diagnostics") or [])
    synthesis_input_pack = dict(payloads.get("synthesis_input_pack") or {})
    evidence_ledger = dict(payloads.get("evidence_ledger") or {})

    citation_count = len(result.citations)
    accepted_claim_count = _count_accepted_claims(
        evidence_ledger=evidence_ledger,
        synthesis_input_pack=synthesis_input_pack,
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
    required_artifacts = _required_artifact_map(artifact_root)
    missing_artifacts = [name for name, present in required_artifacts.items() if not present]

    validation_notes: List[str] = []
    if missing_artifacts:
        validation_notes.append(f"Missing required artifacts: {', '.join(missing_artifacts)}.")
    if unmatched_citation_ids:
        validation_notes.append("One or more citations do not map to a real paper record.")
    if provider_failure_detected:
        validation_notes.append("Provider/network degradation was detected in the live run.")
    if generic_answer_only:
        validation_notes.append("The final answer is still dominated by fallback limitation language.")

    category = _classify_validation_result(
        result=result,
        paper_candidate_count=len(paper_candidates),
        paper_record_count=len(paper_records),
        citation_count=citation_count,
        accepted_claim_count=accepted_claim_count,
        citation_matches=not unmatched_citation_ids and citation_count > 0,
        generic_answer_only=generic_answer_only,
        provider_failure_detected=provider_failure_detected,
        missing_artifacts=missing_artifacts,
    )

    return LiveValidationReport(
        question=question,
        category=category,
        artifact_dir=str(artifact_root),
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
    )


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
    missing_artifacts: Sequence[str],
) -> ValidationCategory:
    if missing_artifacts:
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
    result: QAResult,
) -> int:
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


def _required_artifact_map(artifact_root: Path) -> Dict[str, bool]:
    return {name: (artifact_root / name).exists() for name in REQUIRED_ARTIFACT_FILES}


def _missing_artifacts(artifact_root: Path) -> List[str]:
    return [name for name, present in _required_artifact_map(artifact_root).items() if not present]


def _load_json_file(path: Path, *, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return default


def _resolve_suite_root(*, args: argparse.Namespace, questions: Sequence[str]) -> Path:
    if args.artifact_dir:
        return ensure_dir(Path(args.artifact_dir).parent)
    if args.artifact_root:
        return ensure_dir(Path(args.artifact_root))
    if len(questions) == 1:
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


def _print_report(report: LiveValidationReport) -> None:
    print(f"[{report.category}] {report.question}")
    print(f"Artifact dir: {report.artifact_dir}")
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
    if report.validation_notes:
        print(f"Notes: {' '.join(report.validation_notes)}")
    if report.report_path:
        print(f"Report: {report.report_path}")
    print("")


if __name__ == "__main__":
    raise SystemExit(main())
