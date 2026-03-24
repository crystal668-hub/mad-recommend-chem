from __future__ import annotations

import argparse
import copy
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence

from dotenv import load_dotenv
from pydantic import Field, field_validator

from qa.facade import QASystem
from qa.state import StrictModel
from qa.synthesis_state import QAResult
from utils import ensure_dir, generate_timestamp, load_config, save_json, setup_logging


SmokeAnswerType = Literal["choice_text", "integer"]

DEFAULT_CASES_FILE = "./evals/chembench_smoke_cases.yaml"
DEFAULT_OUTPUT_ROOT = "./outputs/chembench_smoke"
SHORT_ANSWER_INSTRUCTION = (
    "ChemBench smoke test protocol:\n"
    "- Answer the chemistry question normally.\n"
    "- On the last line, output exactly: FINAL_SHORT_ANSWER: <short answer>\n"
    "- Keep <short answer> concise. For numeric questions, output only the final number when possible."
)


class ChembenchSmokeCase(StrictModel):
    subset: str
    name: str
    uuid: str
    question: str
    gold: str
    type: SmokeAnswerType
    keywords: List[str] = Field(default_factory=list)
    source_url: str

    @field_validator("keywords", mode="before")
    @classmethod
    def coerce_keywords(cls, value):
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        text = str(value).strip()
        return [text] if text else []


class ChembenchSmokeCaseReport(StrictModel):
    subset: str
    name: str
    uuid: str
    artifact_dir: str
    workflow_mode: Optional[str] = None
    pipeline_ok: bool
    extracted_ok: bool
    correct: bool = False
    predicted_answer: str = ""
    normalized_prediction: str = ""
    normalized_gold: str = ""
    prediction_source: str = ""
    citation_count: int = 0
    review_completion_status: Optional[str] = None
    reasoning_reviewer_status: Optional[str] = None
    error_message: Optional[str] = None
    notes: List[str] = Field(default_factory=list)


class ChembenchSmokeSuiteReport(StrictModel):
    generated_at: str
    config_path: str
    cases_file: str
    shadow_config_path: Optional[str] = None
    selected_cases: List[str] = Field(default_factory=list)
    report_path: Optional[str] = None
    cases: List[ChembenchSmokeCaseReport] = Field(default_factory=list)
    summary: Dict[str, int] = Field(default_factory=dict)


def load_smoke_cases(cases_file: str) -> List[ChembenchSmokeCase]:
    payload = load_config(cases_file)
    raw_cases = payload.get("cases") if isinstance(payload, dict) else payload
    if not isinstance(raw_cases, list):
        raise ValueError(f"ChemBench smoke cases file must contain a top-level 'cases' list: {cases_file}")
    return [ChembenchSmokeCase.model_validate(item) for item in raw_cases]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ChemQA ChemBench smoke runner")
    parser.add_argument(
        "--cases-file",
        default=DEFAULT_CASES_FILE,
        help="Path to the ChemBench smoke case YAML/JSON file.",
    )
    parser.add_argument(
        "--case",
        action="append",
        help="Run only the selected case name, uuid, or subset/name. Repeat to select multiple cases.",
    )
    parser.add_argument(
        "--config",
        default="./config/config.yaml",
        help="Path to configuration file.",
    )
    parser.add_argument(
        "--shadow-config",
        default=None,
        help="Optional overlay config merged on top of --config for this run.",
    )
    parser.add_argument(
        "--artifact-root",
        default=None,
        help="Root directory for per-case artifacts and the suite report.",
    )
    parser.add_argument(
        "--context",
        default=None,
        help="Optional extra context appended after the smoke-test instruction.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first incorrect or failed case.",
    )
    return parser


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    system_factory: Optional[Callable[..., Any]] = None,
    configure_logging: bool = True,
) -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)

    config = _load_runtime_config(args.config, shadow_config_path=args.shadow_config)
    if configure_logging:
        setup_logging(config)

    cases = load_smoke_cases(args.cases_file)
    selectors = list(args.case or [])
    if selectors:
        cases = [case for case in cases if any(_case_matches_selector(case, selector) for selector in selectors)]
    if not cases:
        parser.error("No ChemBench smoke cases matched the requested filters.")

    suite_root = _resolve_suite_root(args.artifact_root)
    reports: List[ChembenchSmokeCaseReport] = []

    for index, case in enumerate(cases, start=1):
        artifact_dir = suite_root / f"{index:02d}_{_slugify(case.name)}"
        report = run_smoke_case(
            case,
            config=config,
            config_path=args.config,
            artifact_dir=str(artifact_dir),
            extra_context=args.context,
            system_factory=system_factory,
        )
        reports.append(report)
        _print_case_report(report)
        if args.fail_fast and (not report.pipeline_ok or not report.correct):
            break

    suite_report = ChembenchSmokeSuiteReport(
        generated_at=generate_timestamp(),
        config_path=args.config,
        cases_file=args.cases_file,
        shadow_config_path=args.shadow_config,
        selected_cases=selectors,
        cases=reports,
        summary=_summarize_reports(reports),
    )
    report_path = suite_root / "chembench_smoke_report.json"
    save_json(suite_report.model_dump(exclude_none=True), report_path)
    print(f"ChemBench smoke report: {report_path}")
    return 1 if int(suite_report.summary.get("incorrect_or_failed") or 0) > 0 else 0


def run_smoke_case(
    case: ChembenchSmokeCase,
    *,
    config: Optional[Dict[str, Any]] = None,
    config_path: str = "./config/config.yaml",
    artifact_dir: Optional[str] = None,
    extra_context: Optional[str] = None,
    system: Optional[Any] = None,
    system_factory: Optional[Callable[..., Any]] = None,
) -> ChembenchSmokeCaseReport:
    resolved_config = copy.deepcopy(config) if config is not None else load_config(config_path)
    resolved_artifact_dir = str(Path(artifact_dir or _default_case_artifact_dir(case)))
    ensure_dir(resolved_artifact_dir)

    try:
        qa_system = system or _build_system(
            config=resolved_config,
            config_path=config_path,
            system_factory=system_factory,
        )
        result = qa_system.run_qa(
            question=case.question,
            context=_build_case_context(extra_context),
            artifact_dir=resolved_artifact_dir,
        )
    except Exception as exc:
        report = ChembenchSmokeCaseReport(
            subset=case.subset,
            name=case.name,
            uuid=case.uuid,
            artifact_dir=resolved_artifact_dir,
            pipeline_ok=False,
            extracted_ok=False,
            correct=False,
            prediction_source="unavailable",
            error_message=str(exc),
            notes=["Pipeline execution failed before answer extraction."],
        )
        _write_case_report(report)
        return report

    artifact_root = _resolve_artifact_root(result=result, requested_artifact_dir=resolved_artifact_dir)
    predicted_answer, extracted_ok, prediction_source = extract_short_answer(result.final_answer)
    correct, normalized_prediction, normalized_gold = score_case_answer(case=case, predicted_answer=predicted_answer)
    reasoning_status = _load_reasoning_reviewer_status(artifact_root)

    notes: List[str] = []
    if not extracted_ok:
        notes.append("FINAL_SHORT_ANSWER marker was missing; fell back to the last non-empty answer line.")
    if not predicted_answer.strip():
        notes.append("No non-empty short answer could be extracted from final_answer.")
    if result.insufficient_evidence:
        notes.append("Run was marked insufficient_evidence by the workflow.")

    report = ChembenchSmokeCaseReport(
        subset=case.subset,
        name=case.name,
        uuid=case.uuid,
        artifact_dir=str(artifact_root),
        workflow_mode=str(result.workflow_mode or "").strip() or None,
        pipeline_ok=True,
        extracted_ok=extracted_ok,
        correct=correct,
        predicted_answer=predicted_answer,
        normalized_prediction=normalized_prediction,
        normalized_gold=normalized_gold,
        prediction_source=prediction_source,
        citation_count=len(list(result.citations or [])),
        review_completion_status=str(result.review_completion_status or "").strip() or None,
        reasoning_reviewer_status=reasoning_status,
        notes=notes,
    )
    _write_case_report(report)
    return report


def extract_short_answer(final_answer: str) -> tuple[str, bool, str]:
    text = str(final_answer or "")
    matches = re.findall(r"(?im)^\s*FINAL_SHORT_ANSWER\s*:\s*(.+?)\s*$", text)
    if matches:
        return matches[-1].strip(), True, "final_short_answer_marker"
    fallback = _last_non_empty_line(text)
    return fallback, False, "final_answer_last_line"


def score_case_answer(case: ChembenchSmokeCase, predicted_answer: str) -> tuple[bool, str, str]:
    normalized_prediction = _normalize_text(predicted_answer)
    normalized_gold = _normalize_text(case.gold)
    if case.type == "integer":
        predicted_value = _extract_last_integer(predicted_answer)
        gold_value = _extract_last_integer(case.gold)
        if predicted_value is None or gold_value is None:
            return False, normalized_prediction, normalized_gold
        return predicted_value == gold_value, str(predicted_value), str(gold_value)

    accepted_forms = [_normalize_text(case.gold), *[_normalize_text(keyword) for keyword in case.keywords]]
    accepted_forms = [item for item in accepted_forms if item]
    is_correct = any(
        normalized_prediction == accepted
        or (accepted and accepted in normalized_prediction)
        or (normalized_prediction and normalized_prediction in accepted)
        for accepted in accepted_forms
    )
    return is_correct, normalized_prediction, normalized_gold


def _build_system(
    *,
    config: Dict[str, Any],
    config_path: str,
    system_factory: Optional[Callable[..., Any]],
) -> Any:
    factory = system_factory or QASystem
    return factory(config=config, config_path=config_path)


def _build_case_context(extra_context: Optional[str]) -> str:
    extra = str(extra_context or "").strip()
    if not extra:
        return SHORT_ANSWER_INSTRUCTION
    return f"{SHORT_ANSWER_INSTRUCTION}\n\nAdditional context:\n{extra}"


def _load_runtime_config(config_path: str, *, shadow_config_path: Optional[str]) -> Dict[str, Any]:
    config = copy.deepcopy(load_config(config_path))
    if not shadow_config_path:
        return config
    shadow_config = copy.deepcopy(load_config(shadow_config_path))
    return _deep_merge_dicts(config, shadow_config)


def _deep_merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _case_matches_selector(case: ChembenchSmokeCase, selector: str) -> bool:
    current = str(selector or "").strip()
    if not current:
        return False
    return current in {case.name, case.uuid, f"{case.subset}/{case.name}"}


def _resolve_suite_root(artifact_root: Optional[str]) -> Path:
    if artifact_root:
        root = Path(artifact_root)
    else:
        root = Path(DEFAULT_OUTPUT_ROOT) / generate_timestamp()
    ensure_dir(root)
    return root.resolve()


def _default_case_artifact_dir(case: ChembenchSmokeCase) -> Path:
    return Path(DEFAULT_OUTPUT_ROOT) / generate_timestamp() / _slugify(case.name)


def _resolve_artifact_root(*, result: QAResult, requested_artifact_dir: str) -> Path:
    qa_result_path = str(result.artifact_paths.get("qa_result") or "").strip()
    if qa_result_path:
        return Path(qa_result_path).resolve().parent
    return Path(requested_artifact_dir).resolve()


def _load_reasoning_reviewer_status(artifact_root: Path) -> Optional[str]:
    review_statuses_path = artifact_root / "review_statuses.json"
    if not review_statuses_path.exists():
        return None
    payload = load_config(str(review_statuses_path))
    if not isinstance(payload, list):
        return None
    for item in payload:
        if str((item or {}).get("reviewer_role") or "").strip() == "reasoning_consistency":
            status = str((item or {}).get("status") or "").strip()
            return status or None
    return None


def _write_case_report(report: ChembenchSmokeCaseReport) -> None:
    save_json(report.model_dump(exclude_none=True), Path(report.artifact_dir) / "chembench_smoke_case_report.json")


def _print_case_report(report: ChembenchSmokeCaseReport) -> None:
    status = "PASS" if report.pipeline_ok and report.correct else "FAIL"
    print(
        f"[{status}] {report.name} "
        f"pipeline_ok={report.pipeline_ok} extracted_ok={report.extracted_ok} "
        f"correct={report.correct} prediction={report.predicted_answer!r}"
    )


def _summarize_reports(reports: Sequence[ChembenchSmokeCaseReport]) -> Dict[str, int]:
    total = len(reports)
    pipeline_failed = sum(1 for report in reports if not report.pipeline_ok)
    correct = sum(1 for report in reports if report.pipeline_ok and report.correct)
    extracted_missing = sum(1 for report in reports if not report.extracted_ok)
    incorrect_or_failed = sum(1 for report in reports if (not report.pipeline_ok) or (not report.correct))
    return {
        "total": total,
        "pipeline_failed": pipeline_failed,
        "correct": correct,
        "extracted_missing_marker": extracted_missing,
        "incorrect_or_failed": incorrect_or_failed,
    }


def _normalize_text(text: str) -> str:
    value = str(text or "").strip().lower()
    value = value.replace("–", "-").replace("—", "-")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _extract_last_integer(text: str) -> Optional[int]:
    matches = re.findall(r"-?\d+", str(text or ""))
    if not matches:
        return None
    return int(matches[-1])


def _last_non_empty_line(text: str) -> str:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    if not lines:
        return ""
    return lines[-1]


def _slugify(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", str(value or "").strip().lower())
    normalized = normalized.strip("_")
    return normalized or "case"


if __name__ == "__main__":
    raise SystemExit(main())
