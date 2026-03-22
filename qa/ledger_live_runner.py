from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

from utils import ensure_dir, generate_timestamp, load_config, save_json


DEFAULT_QUESTION = "How does Pt/C affect HER activity in 1 M KOH?"
DEFAULT_SUITE_FILE = "./qa/resources/live_validation_ledger_suite.yaml"
DEFAULT_SHADOW_CONFIG = "./.cache/ledger_live_validation_shadow.yaml"
DEFAULT_REACT_CONTROL = "./outputs/qa_result_20260319_184954.json"
GENERIC_ANSWER_MARKERS = (
    "insufficient evidence",
    "limited evidence",
    "does not support a firm conclusion",
    "accepted evidence is too sparse",
    "output remains partial and conservative",
)
OFF_TOPIC_TERMS = (
    "oer",
    "oxygen evolution",
    "orr",
    "oxygen reduction",
    "ethanol",
    "alcohol",
    "fuel cell",
    "battery",
    "solar cell",
    "co2rr",
)


def _slugify(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", str(text or "").strip()).strip("_").lower()
    return cleaned[:64] or "ledger_question"


def _read_json(path: Path, *, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return default


def _write_shadow_config(path: Path) -> Path:
    ensure_dir(path.parent)
    payload = {
        "qa": {
            "workflow_mode": "ledger",
            "save_output": False,
        }
    }
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return path


def _load_suite_questions(suite_file: str, *, tier: str) -> List[Dict[str, Any]]:
    payload = load_config(suite_file)
    raw_cases = payload.get("cases") if isinstance(payload, dict) else payload
    if not isinstance(raw_cases, list):
        raise ValueError(f"Ledger suite must expose a top-level cases list: {suite_file}")

    normalized: List[Dict[str, Any]] = []
    for item in raw_cases:
        if not isinstance(item, dict):
            continue
        case_tier = str(item.get("tier") or "core").strip().lower()
        if tier != "all" and case_tier != tier:
            continue
        normalized.append(
            {
                "case_id": str(item.get("case_id") or "").strip() or _slugify(str(item.get("question") or "")),
                "question": str(item.get("question") or "").strip(),
                "expected_category": item.get("expected_category"),
                "tier": case_tier,
                "question_type": str(item.get("question_type") or "").strip(),
            }
        )
    return [item for item in normalized if item["question"]]


def _resolved_anchor_terms(*, question: str, entity_pack: Dict[str, Any]) -> List[str]:
    anchors: List[str] = []
    for entity in list(entity_pack.get("entities") or []):
        for value in [entity.get("mention"), entity.get("canonical_name"), *(entity.get("query_anchors") or [])]:
            text = str(value or "").strip().lower()
            if text:
                anchors.append(text)
    for condition in list(entity_pack.get("condition_mentions") or []):
        for value in [condition.get("raw_value"), condition.get("normalized_value")]:
            text = str(value or "").strip().lower()
            if text:
                anchors.append(text)
    if not anchors:
        anchors.extend(
            text.lower()
            for text in re.findall(r"[A-Za-z0-9/+.-]+(?:\s+[A-Za-z0-9/+.-]+){0,3}", str(question or ""))
            if len(text.strip()) >= 3
        )
    seen = set()
    unique: List[str] = []
    for item in anchors:
        if item in seen:
            continue
        seen.add(item)
        unique.append(item)
    return unique


def _candidate_payloads(artifact_dir: Path) -> List[Dict[str, Any]]:
    paper_candidates = _read_json(artifact_dir / "paper_candidates.json", default=None)
    if isinstance(paper_candidates, list):
        return [dict(item or {}) for item in paper_candidates]

    fallback: List[Dict[str, Any]] = []
    provider_dir = artifact_dir / "provider_raw" / "openalex"
    for path in sorted(provider_dir.glob("*.json")):
        if path.name.startswith("search_"):
            continue
        raw = _read_json(path, default=None)
        if not isinstance(raw, dict):
            continue
        fallback.append(
            {
                "paper_id": path.stem,
                "title": raw.get("display_name") or raw.get("title") or "",
                "abstract": _openalex_abstract(raw),
                "lane_sources": [],
                "ranking_features": {},
                "retrieval_score": None,
            }
        )
    return fallback


def _openalex_abstract(raw: Dict[str, Any]) -> str:
    direct = str(raw.get("abstract") or "").strip()
    if direct:
        return direct
    inverted = raw.get("abstract_inverted_index") or {}
    pairs: List[Tuple[int, str]] = []
    for token, positions in dict(inverted).items():
        for position in list(positions or []):
            try:
                pairs.append((int(position), str(token)))
            except (TypeError, ValueError):
                continue
    pairs.sort(key=lambda item: item[0])
    return " ".join(token for _, token in pairs).strip()


def _candidate_match_summary(*, question: str, entity_pack: Dict[str, Any], candidates: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    anchors = _resolved_anchor_terms(question=question, entity_pack=entity_pack)
    summaries: List[Dict[str, Any]] = []
    off_topic_titles: List[str] = []
    for candidate in candidates:
        title = str(candidate.get("title") or "").strip()
        abstract = str(candidate.get("abstract") or "").strip()
        corpus = f"{title} {abstract}".lower()
        matched = [anchor for anchor in anchors if anchor and anchor in corpus]
        off_topic = [term for term in OFF_TOPIC_TERMS if term in corpus]
        missing = [anchor for anchor in anchors if anchor and anchor not in corpus]
        summary = {
            "paper_id": str(candidate.get("paper_id") or "").strip(),
            "title": title,
            "matched_anchor_count": len(matched),
            "matched_anchors": matched[:6],
            "missing_anchors": missing[:6],
            "off_topic_terms": off_topic,
            "retrieval_score": candidate.get("retrieval_score"),
            "lane_sources": list(candidate.get("lane_sources") or []),
        }
        summaries.append(summary)
        if off_topic:
            off_topic_titles.append(title)
    summaries.sort(key=lambda item: (-int(item["matched_anchor_count"]), item["title"]))
    return {
        "anchor_terms": anchors,
        "candidate_count": len(candidates),
        "top_candidates": summaries[:8],
        "off_topic_titles": off_topic_titles[:8],
    }


def _grounding_diagnosis(*, question: str, artifact_dir: Path) -> Dict[str, Any]:
    entity_pack = _read_json(artifact_dir / "entity_resolver" / "entity_pack.json", default={}) or {}
    unresolved_mentions = [dict(item or {}) for item in list(entity_pack.get("unresolved_mentions") or [])]
    notes: List[str] = []
    if not (artifact_dir / "router" / "task_spec.json").exists():
        notes.append("Ledger runs do not currently emit router/task_spec artifacts; grounding diagnosis is limited to entity resolution outputs.")
    if unresolved_mentions:
        notes.append(f"{len(unresolved_mentions)} mentions remained unresolved or weakly normalized.")
    return {
        "resolved_entity_count": len(list(entity_pack.get("entities") or [])),
        "unresolved_mention_count": len(unresolved_mentions),
        "condition_count": len(list(entity_pack.get("condition_mentions") or [])),
        "unresolved_mentions": unresolved_mentions[:6],
        "notes": notes,
        "entity_pack": entity_pack,
        "question": question,
    }


def _synthesis_diagnosis(artifact_dir: Path) -> Dict[str, Any]:
    evidence_ledger = _read_json(artifact_dir / "evidence_ledger.json", default=None)
    reviewed_ledger = _read_json(artifact_dir / "evidence_ledger_reviewed.json", default=None)
    synthesis_pack = _read_json(artifact_dir / "synthesis_input_pack.json", default=None)
    qa_result = _read_json(artifact_dir / "qa_result.json", default=None)
    notes: List[str] = []
    if reviewed_ledger is None and evidence_ledger is None and (artifact_dir / "fulltext").exists():
        notes.append("Run reached document acquisition but did not materialize an evidence ledger.")
    if reviewed_ledger is not None and synthesis_pack is None:
        notes.append("Reviewed ledger exists, but synthesis input pack was never written.")
    if synthesis_pack is not None and qa_result is None:
        notes.append("Synthesis input pack exists, but final qa_result.json was not written.")

    qa_answer = str((qa_result or {}).get("final_answer") or "").strip().lower()
    if qa_answer and any(marker in qa_answer for marker in GENERIC_ANSWER_MARKERS):
        notes.append("Final answer fell back to a generic insufficiency template.")
    return {
        "evidence_ledger_present": evidence_ledger is not None,
        "reviewed_evidence_ledger_present": reviewed_ledger is not None,
        "synthesis_input_pack_present": synthesis_pack is not None,
        "qa_result_present": qa_result is not None,
        "accepted_claim_count": sum(
            1
            for claim in list((reviewed_ledger or evidence_ledger or {}).get("claims") or [])
            if str((claim or {}).get("status") or "").strip() == "accepted"
        ),
        "citation_count": len(list((qa_result or {}).get("citations") or [])),
        "notes": notes,
    }


def _run_state(*, artifact_dir: Path) -> str:
    if (artifact_dir / "qa_result.json").exists():
        return "completed"
    if (artifact_dir / "evidence_ledger_reviewed.json").exists() or (artifact_dir / "evidence_ledger.json").exists():
        return "stalled_after_ledger"
    if (artifact_dir / "fulltext").exists() or (artifact_dir / "indices").exists():
        return "stalled_after_document_acquisition"
    if (artifact_dir / "entity_resolver" / "entity_pack.json").exists():
        return "stalled_after_grounding"
    if (artifact_dir / "runtime_manifest.json").exists():
        return "stalled_after_startup"
    return "no_artifacts"


def analyze_artifact_dir(*, question: str, artifact_dir: Path) -> Dict[str, Any]:
    grounding = _grounding_diagnosis(question=question, artifact_dir=artifact_dir)
    candidates = _candidate_payloads(artifact_dir)
    retrieval = _candidate_match_summary(
        question=question,
        entity_pack=grounding["entity_pack"],
        candidates=candidates,
    )
    synthesis = _synthesis_diagnosis(artifact_dir)
    live_report = _read_json(artifact_dir / "live_validation_report.json", default={}) or {}
    runtime_manifest = _read_json(artifact_dir / "runtime_manifest.json", default={}) or {}
    workflow_mode = (
        str((live_report or {}).get("workflow_mode") or "").strip()
        or str(((runtime_manifest.get("qa") or {}).get("workflow_mode")) or "").strip()
        or None
    )
    provider_health = _read_json(artifact_dir / "provider_health.json", default={}) or {}
    retrieval_diagnostics = _read_json(artifact_dir / "retrieval_diagnostics.json", default=[]) or []
    notes: List[str] = []
    if workflow_mode != "ledger":
        notes.append(f"workflow_mode should be ledger but resolved to {workflow_mode!r}.")
    if retrieval["off_topic_titles"]:
        notes.append("Candidate set already shows off-topic literature before synthesis.")
    if synthesis["notes"]:
        notes.extend(synthesis["notes"])
    return {
        "question": question,
        "artifact_dir": str(artifact_dir),
        "run_state": _run_state(artifact_dir=artifact_dir),
        "workflow_mode": workflow_mode,
        "provider_health": provider_health,
        "retrieval_diagnostics": retrieval_diagnostics,
        "grounding": {
            key: value for key, value in grounding.items() if key != "entity_pack"
        },
        "retrieval": retrieval,
        "synthesis": synthesis,
        "diagnostic_notes": notes,
    }


def analyze_react_control(path: Path) -> Dict[str, Any]:
    payload = _read_json(path, default={}) or {}
    final_answer = str(payload.get("final_answer") or "").strip()
    normalized = final_answer.lower()
    flags: List[str] = []
    if str(payload.get("workflow_mode") or "").strip() != "ledger":
        flags.append("control sample is not a ledger run.")
    if "iridium" in normalized or "oxygen evolution" in normalized or "oer" in normalized:
        flags.append("final answer drifted to Ir/OER content unrelated to the Pt/C vs NiMo HER question.")
    if str(payload.get("review_completion_status") or "").strip() == "incomplete":
        flags.append("react_reviewed protocol closed with incomplete reviewer execution.")
    invalid_json_warnings = [
        item
        for item in list(payload.get("execution_warnings") or [])
        if "invalid_json" in str(item).lower()
    ]
    return {
        "path": str(path),
        "workflow_mode": payload.get("workflow_mode"),
        "review_completion_status": payload.get("review_completion_status"),
        "execution_warning_count": len(list(payload.get("execution_warnings") or [])),
        "flags": flags,
        "invalid_json_warnings": invalid_json_warnings[:4],
    }


def _decode_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def run_live_validation_question(
    *,
    question: str,
    artifact_dir: Path,
    config_path: str,
    shadow_config_path: Path,
    timeout_seconds: int,
) -> Dict[str, Any]:
    ensure_dir(artifact_dir)
    command = [
        sys.executable,
        "-m",
        "qa.live_validation",
        "--question",
        question,
        "--artifact-dir",
        str(artifact_dir),
        "--config",
        config_path,
        "--shadow-config",
        str(shadow_config_path),
    ]
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        completed = subprocess.run(
            command,
            cwd=Path.cwd(),
            env=env,
            capture_output=True,
            text=True,
            timeout=max(1, int(timeout_seconds)),
            check=False,
        )
        return {
            "question": question,
            "artifact_dir": str(artifact_dir),
            "returncode": completed.returncode,
            "timed_out": False,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "question": question,
            "artifact_dir": str(artifact_dir),
            "returncode": None,
            "timed_out": True,
            "stdout": _decode_text(exc.stdout),
            "stderr": _decode_text(exc.stderr),
        }


def _markdown_report(
    *,
    executions: Sequence[Dict[str, Any]],
    diagnoses: Sequence[Dict[str, Any]],
    react_control: Optional[Dict[str, Any]],
) -> str:
    lines: List[str] = ["# Ledger Live Validation Report", ""]
    for execution, diagnosis in zip(executions, diagnoses):
        lines.append(f"## {execution['question']}")
        lines.append(f"- Artifact dir: `{diagnosis['artifact_dir']}`")
        lines.append(f"- Workflow mode: `{diagnosis.get('workflow_mode')}`")
        lines.append(f"- Run state: `{diagnosis.get('run_state')}`")
        lines.append(f"- Timed out: `{execution.get('timed_out')}`")
        lines.append(f"- Return code: `{execution.get('returncode')}`")
        grounding = diagnosis.get("grounding") or {}
        retrieval = diagnosis.get("retrieval") or {}
        synthesis = diagnosis.get("synthesis") or {}
        lines.append(
            "- Grounding: "
            f"resolved={grounding.get('resolved_entity_count', 0)} "
            f"unresolved={grounding.get('unresolved_mention_count', 0)} "
            f"conditions={grounding.get('condition_count', 0)}"
        )
        lines.append(
            "- Retrieval: "
            f"candidates={retrieval.get('candidate_count', 0)} "
            f"off_topic_titles={len(retrieval.get('off_topic_titles') or [])}"
        )
        lines.append(
            "- Synthesis: "
            f"ledger={synthesis.get('evidence_ledger_present')} "
            f"reviewed_ledger={synthesis.get('reviewed_evidence_ledger_present')} "
            f"synthesis_pack={synthesis.get('synthesis_input_pack_present')} "
            f"qa_result={synthesis.get('qa_result_present')}"
        )
        if diagnosis.get("diagnostic_notes"):
            lines.append("- Notes:")
            for item in diagnosis["diagnostic_notes"]:
                lines.append(f"  - {item}")
        top_candidates = retrieval.get("top_candidates") or []
        if top_candidates:
            lines.append("- Top candidates:")
            for item in top_candidates[:5]:
                lines.append(
                    "  - "
                    f"{item.get('title')} "
                    f"(anchor_hits={item.get('matched_anchor_count')}, "
                    f"off_topic={','.join(item.get('off_topic_terms') or []) or 'none'})"
                )
        lines.append("")

    if react_control is not None:
        lines.append("## React Control")
        lines.append(f"- Path: `{react_control['path']}`")
        lines.append(f"- Workflow mode: `{react_control.get('workflow_mode')}`")
        lines.append(f"- Review completion: `{react_control.get('review_completion_status')}`")
        lines.append(f"- Warning count: `{react_control.get('execution_warning_count')}`")
        for item in react_control.get("flags") or []:
            lines.append(f"- {item}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ledger live validation with partial-artifact diagnosis.")
    parser.add_argument("--config", default="./config/config.yaml", help="Base config path.")
    parser.add_argument("--shadow-config", default=DEFAULT_SHADOW_CONFIG, help="Ledger shadow config path.")
    parser.add_argument("--artifact-root", default=None, help="Root directory for all outputs.")
    parser.add_argument("--question", action="append", help="Question to run. Repeat to add more.")
    parser.add_argument("--run-suite", action="store_true", help="Also execute questions from the ledger suite file.")
    parser.add_argument("--suite-file", default=DEFAULT_SUITE_FILE, help="Ledger suite YAML/JSON file.")
    parser.add_argument(
        "--suite-tier",
        choices=("core", "extended", "all"),
        default="all",
        help="Filter suite questions by tier.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=1200, help="Per-question timeout.")
    parser.add_argument("--compare-react-path", default=DEFAULT_REACT_CONTROL, help="Existing react control artifact.")
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Skip execution and only analyze whatever is already present under artifact-root question directories.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    artifact_root = ensure_dir(
        args.artifact_root or Path("./logs/runs") / generate_timestamp() / "ledger_live_runner"
    )
    shadow_config_path = _write_shadow_config(Path(args.shadow_config))

    questions: List[Dict[str, Any]] = []
    for question in list(args.question or []):
        questions.append({"case_id": _slugify(question), "question": str(question), "source": "manual"})
    if args.run_suite:
        for item in _load_suite_questions(args.suite_file, tier=args.suite_tier):
            questions.append({**item, "source": "suite"})
    if not questions:
        questions.append({"case_id": "default_question", "question": DEFAULT_QUESTION, "source": "default"})

    executions: List[Dict[str, Any]] = []
    diagnoses: List[Dict[str, Any]] = []
    for index, item in enumerate(questions, start=1):
        question = str(item["question"]).strip()
        question_dir = artifact_root / f"{index:02d}_{_slugify(question)}"
        if args.skip_run:
            execution = {
                "question": question,
                "artifact_dir": str(question_dir),
                "returncode": None,
                "timed_out": False,
                "stdout": "",
                "stderr": "",
            }
        else:
            execution = run_live_validation_question(
                question=question,
                artifact_dir=question_dir,
                config_path=args.config,
                shadow_config_path=shadow_config_path,
                timeout_seconds=args.timeout_seconds,
            )
        diagnosis = analyze_artifact_dir(question=question, artifact_dir=question_dir)
        execution.update(
            {
                "case_id": item.get("case_id"),
                "source": item.get("source"),
                "expected_category": item.get("expected_category"),
                "tier": item.get("tier"),
            }
        )
        executions.append(execution)
        diagnoses.append(diagnosis)

    react_control = None
    react_path = Path(args.compare_react_path)
    if react_path.exists():
        react_control = analyze_react_control(react_path)

    report = {
        "generated_at": generate_timestamp(),
        "artifact_root": str(artifact_root),
        "config_path": args.config,
        "shadow_config_path": str(shadow_config_path),
        "questions": executions,
        "diagnoses": diagnoses,
        "react_control": react_control,
    }
    report_path = artifact_root / "ledger_live_runner_report.json"
    save_json(report, report_path)
    markdown_path = artifact_root / "ledger_live_runner_report.md"
    markdown_path.write_text(
        _markdown_report(executions=executions, diagnoses=diagnoses, react_control=react_control),
        encoding="utf-8",
    )

    print(f"Artifact root: {artifact_root}")
    print(f"JSON report: {report_path}")
    print(f"Markdown report: {markdown_path}")
    for execution, diagnosis in zip(executions, diagnoses):
        print(
            f"[{diagnosis.get('run_state')}] {execution.get('question')} "
            f"(timed_out={execution.get('timed_out')}, workflow={diagnosis.get('workflow_mode')})"
        )

    if any(str(diagnosis.get("workflow_mode") or "").strip() != "ledger" for diagnosis in diagnoses):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
