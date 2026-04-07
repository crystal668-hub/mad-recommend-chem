#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bundle_common import dump_json, resolve_skill_root, write_text


def find_protocol_file(source_dir: Path) -> Path:
    candidates = (
        source_dir / "chemqa_review_protocol.json",
        source_dir / "debate-coordinator" / "chemqa_review_protocol.json",
        source_dir / "coordinator" / "chemqa_review_protocol.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise SystemExit(f"Could not find chemqa_review_protocol.json under {source_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild react_reviewed-style artifacts from chemqa-review outputs.")
    parser.add_argument("--skill-root", help="chemqa-review skill root; defaults to this script's parent bundle")
    parser.add_argument("--source-dir", required=True, help="Directory containing chemqa_review_protocol.json")
    parser.add_argument("--output-dir", required=True, help="Output directory for rebuilt artifacts")
    parser.add_argument("--protocol-file", help="Optional explicit protocol JSON path")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _skill_root = resolve_skill_root(args.skill_root, file_hint=__file__)
    source_dir = Path(args.source_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = Path(args.protocol_file).expanduser().resolve() if args.protocol_file else find_protocol_file(source_dir)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))

    artifact_paths = {
        "candidate_submission": str(dump_json(output_dir / "candidate_submission.json", dict(protocol.get("candidate_submission") or {}))),
        "acceptance_decision": str(dump_json(output_dir / "acceptance_decision.json", dict(protocol.get("acceptance_decision") or {}))),
        "submission_trace": str(dump_json(output_dir / "submission_trace.json", list(protocol.get("submission_trace") or []))),
        "submission_cycles": str(dump_json(output_dir / "submission_cycles.json", list(protocol.get("submission_cycles") or []))),
        "proposer_trajectory": str(dump_json(output_dir / "proposer_trajectory.json", dict(protocol.get("proposer_trajectory") or {}))),
        "reviewer_trajectories": str(dump_json(output_dir / "reviewer_trajectories.json", dict(protocol.get("reviewer_trajectories") or {}))),
        "review_statuses": str(dump_json(output_dir / "review_statuses.json", list(protocol.get("review_statuses") or []))),
        "final_review_items": str(dump_json(output_dir / "final_review_items.json", list(protocol.get("final_review_items") or []))),
        "final_answer": str(write_text(output_dir / "final_answer.md", str(protocol.get("final_answer") or "").strip() + "\n")),
    }
    if protocol.get("acceptance_status") == "accepted":
        artifact_paths["final_submission"] = str(
            dump_json(output_dir / "final_submission.json", dict(protocol.get("candidate_submission") or {}))
        )

    qa_result = {
        "question": str(protocol.get("question") or ""),
        "language": "en",
        "workflow_mode": "react_reviewed",
        "acceptance_status": str(protocol.get("acceptance_status") or "rejected"),
        "final_answer": str(protocol.get("final_answer") or ""),
        "sections": list(protocol.get("sections") or []),
        "citations": list(protocol.get("citations") or []),
        "claim_trace": list(protocol.get("claim_trace") or []),
        "submission_trace": list(protocol.get("submission_trace") or []),
        "review_completion_status": str(protocol.get("review_completion_status") or "incomplete"),
        "overall_confidence": dict(
            protocol.get("overall_confidence")
            or {"level": "low", "score": 0.0, "rationale": "Protocol did not provide confidence."}
        ),
        "section_confidence": list(protocol.get("section_confidence") or []),
        "insufficient_evidence": bool(protocol.get("insufficient_evidence", False)),
        "limitations_summary": str(protocol.get("limitations_summary") or ""),
        "retrieval_diagnostics_summary": str(protocol.get("retrieval_diagnostics_summary") or ""),
        "execution_warnings": list(protocol.get("execution_warnings") or []),
        "artifact_paths": {},
        "time_elapsed": float(protocol.get("time_elapsed") or 0.0),
    }
    artifact_paths["qa_result"] = str(output_dir / "qa_result.json")
    qa_result["artifact_paths"] = dict(artifact_paths)
    dump_json(output_dir / "qa_result.json", qa_result)

    payload = {
        "protocol_file": str(protocol_path),
        "output_dir": str(output_dir),
        "artifact_paths": artifact_paths,
    }
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
