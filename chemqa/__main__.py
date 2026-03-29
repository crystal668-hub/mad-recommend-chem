from __future__ import annotations

import argparse
import copy
from typing import Callable, Optional, Sequence

from dotenv import load_dotenv

from qa.facade import QASystem
from utils import load_config, setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ChemQA CLI")
    parser.add_argument("--question", required=True, help="Research question to answer.")
    parser.add_argument("--context", default=None, help="Optional user context or constraints.")
    parser.add_argument("--artifact-dir", default=None, help="Optional directory for run artifacts.")
    parser.add_argument(
        "--workflow-mode",
        choices=("react_reviewed",),
        default="react_reviewed",
        help="Workflow mode. Only `react_reviewed` is supported.",
    )
    parser.add_argument(
        "--save-output",
        action="store_true",
        help="Also export a user-facing `outputs/qa_result_<timestamp>.json` file.",
    )
    parser.add_argument(
        "--config",
        default="./config/config.yaml",
        help="Path to configuration file.",
    )
    return parser


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    system_factory: Optional[Callable[..., QASystem]] = None,
    configure_logging: bool = True,
) -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)

    config = copy.deepcopy(load_config(args.config))
    qa_config = dict(config.get("qa", {}) or {})
    qa_config["workflow_mode"] = str(args.workflow_mode or "react_reviewed")
    if args.save_output:
        qa_config["save_output"] = True
    config["qa"] = qa_config

    if configure_logging:
        setup_logging(config)

    factory = system_factory or QASystem
    system = factory(config=config, config_path=args.config)
    result = system.run_qa(
        question=args.question,
        context=args.context,
        artifact_dir=args.artifact_dir,
    )

    print(result.final_answer)
    warning_summary = str(result.retrieval_diagnostics_summary or "").strip()
    if warning_summary:
        print(f"\nWarnings: {warning_summary}")
    elif result.execution_warnings:
        preview = "; ".join(result.execution_warnings[:2])
        print(f"\nWarnings: {preview}")
    public_result_path = result.artifact_paths.get("public_result")
    if public_result_path:
        print(f"\nSaved QA result: {public_result_path}")
    else:
        qa_result_path = result.artifact_paths.get("qa_result")
        if qa_result_path:
            print(f"\nSaved QA artifacts: {qa_result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
