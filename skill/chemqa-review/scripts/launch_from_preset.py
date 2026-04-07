#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from bundle_common import default_template_dir, resolve_skill_root, run_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified chemqa-review entrypoint from preset to launch-ready assets.")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]), help="chemqa-review skill root")
    parser.add_argument("--preset", required=True, help="Preset ref, e.g. chemqa-review@1")
    parser.add_argument("--goal", required=True, help="Run goal or motion")
    parser.add_argument("--run-id", help="Optional explicit run id")
    parser.add_argument("--additional-file-workspace", help="Optional run-scoped opaque string for extra file context")
    parser.add_argument("--model-profile", help="Override model profile")
    parser.add_argument("--review-rounds", type=int)
    parser.add_argument("--rebuttal-rounds", type=int)
    parser.add_argument("--evidence-mode", help="Override evidence mode")
    parser.add_argument("--priority", default="normal")
    parser.add_argument("--reset-state", action="store_true")
    parser.add_argument("--launch-mode", choices=("none", "print", "run"), default="print")
    parser.add_argument("--template-dir", help="Optional template output directory")
    parser.add_argument("--command-map-dir", help="Optional command-map output directory")
    parser.add_argument("--runtime-dir", help="Optional deployed runtime helper directory")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def effective_template_dir(*, explicit: str | None, launch_mode: str) -> str | None:
    if explicit:
        return explicit
    if launch_mode in {"print", "run"}:
        return str(default_template_dir())
    return None


def main() -> int:
    args = parse_args()
    root = resolve_skill_root(args.root)
    scripts_dir = root / "scripts"

    compile_cmd = [
        "python3",
        str(scripts_dir / "compile_runplan.py"),
        "--root",
        str(root),
        "--preset",
        args.preset,
        "--goal",
        args.goal,
        "--priority",
        args.priority,
    ]
    if args.run_id:
        compile_cmd.extend(["--run-id", args.run_id])
    if args.additional_file_workspace:
        compile_cmd.extend(["--additional-file-workspace", args.additional_file_workspace])
    if args.model_profile:
        compile_cmd.extend(["--model-profile", args.model_profile])
    if args.review_rounds is not None:
        compile_cmd.extend(["--review-rounds", str(args.review_rounds)])
    if args.rebuttal_rounds is not None:
        compile_cmd.extend(["--rebuttal-rounds", str(args.rebuttal_rounds)])
    if args.evidence_mode:
        compile_cmd.extend(["--evidence-mode", args.evidence_mode])

    compiled = run_json(compile_cmd, cwd=root)

    materialize_cmd = [
        "python3",
        str(scripts_dir / "materialize_runplan.py"),
        "--root",
        str(root),
        "--run-id",
        compiled["run_id"],
    ]
    resolved_template_dir = effective_template_dir(explicit=args.template_dir, launch_mode=args.launch_mode)
    if resolved_template_dir:
        materialize_cmd.extend(["--template-dir", resolved_template_dir])
    if args.command_map_dir:
        materialize_cmd.extend(["--command-map-dir", args.command_map_dir])
    if args.runtime_dir:
        materialize_cmd.extend(["--runtime-dir", args.runtime_dir])
    if args.reset_state:
        materialize_cmd.append("--reset-state")
    if args.launch_mode == "none":
        materialize_cmd.append("--dry-run")

    materialized = run_json(materialize_cmd, cwd=root)
    launch_command = materialized.get("launch_command")
    launched = None
    if args.launch_mode == "run":
        if not launch_command:
            raise SystemExit("Materialization did not return a clawteam launch command.")
        env = dict(os.environ)
        if materialized.get("clawteam_data_dir"):
            env["CLAWTEAM_DATA_DIR"] = materialized["clawteam_data_dir"]
        result = subprocess.run(launch_command, cwd=str(root), env=env, check=False, capture_output=True, text=True)
        launched = {
            "command": launch_command,
            "clawteam_data_dir": env.get("CLAWTEAM_DATA_DIR"),
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        if result.returncode != 0:
            raise SystemExit(
                f"Launch failed ({result.returncode}): {' '.join(launch_command)}\n\n"
                f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
            )
    payload = {
        "preset": args.preset,
        "goal": args.goal,
        "run_id": compiled["run_id"],
        "compile": compiled,
        "materialize": materialized,
        "launch_mode": args.launch_mode,
        "launch_command": launch_command,
        "launched": launched,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
