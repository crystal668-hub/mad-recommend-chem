#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path

from bundle_common import (
    default_runtime_dir,
    dump_json,
    engine_script_path,
    openclaw_env_file,
    read_text,
    resolve_skill_root,
)
from control_store import FileControlStore


ROLE_TO_SLOT = {
    "debate-coordinator": "debate-coordinator",
    "proposer-1": "debate-1",
    "proposer-2": "debate-2",
    "proposer-3": "debate-3",
    "proposer-4": "debate-4",
    "proposer-5": "debate-5",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize a chemqa-review run plan.")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]), help="chemqa-review skill root")
    parser.add_argument("--run-id", required=True, help="Persisted run plan id")
    parser.add_argument("--template-dir", help="Output template directory")
    parser.add_argument("--command-map-dir", help="Output command-map directory")
    parser.add_argument("--runtime-dir", help="Path to deployed DebateClaw runtime helpers")
    parser.add_argument("--reset-state", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_prepare_output(stdout: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        parsed[key] = value
    return parsed


def build_command_map(run_plan: dict, *, wrapper_path: Path, env_file: str) -> dict[str, list[str]]:
    command_map: dict[str, list[str]] = {}
    slot_assignments = dict(run_plan.get("slot_assignments") or {})
    for role_name, slot_id in ROLE_TO_SLOT.items():
        session_id = run_plan["session_assignments"][slot_id]
        command = [
            str(wrapper_path),
            "--slot",
            slot_id,
            "--env-file",
            env_file,
            "--session-id",
            session_id,
        ]
        thinking = dict(slot_assignments.get(slot_id) or {}).get("thinking")
        if thinking:
            command.extend(["--thinking", str(thinking)])
        command_map[role_name] = command
    return command_map


def render_role_prompt(
    root: Path,
    run_plan: dict,
    *,
    role_name: str,
    runtime_root: str,
) -> str:
    assembly = dict(run_plan["prompt_assembly"][role_name])
    semantic_role = str(assembly["semantic_role"])
    additional_workspace = run_plan.get("runtime_context", {}).get("additional_file_workspace") or "none"
    role_intro = [
        f"You are chemqa-review role `{role_name}`.",
        f"Semantic role: `{semantic_role}`.",
        "This run uses DebateClaw review-loop transport, but only `proposer-1` is the candidate owner.",
        "The other proposer slots are fixed reviewer lanes and must not invent alternate final answers.",
        "",
        "Runtime commands:",
        f"- `{runtime_root}/debate_state.py status --team {{team_name}} --agent {{agent_name}} --json`",
        f"- `{runtime_root}/debate_state.py next-action --team {{team_name}} --agent {{agent_name}} --json`",
        f"- `{runtime_root}/debate_state.py advance --team {{team_name}} --agent {{agent_name}}`",
        "",
        "Sibling skill roots are available under the same skills directory as this bundle.",
        f"Additional file workspace: {additional_workspace}",
    ]
    parts = ["\n".join(role_intro).strip()]
    for rel_path in assembly.get("contracts", []):
        parts.append(read_text(root / rel_path).strip())
    for rel_path in assembly.get("modules", []):
        parts.append(read_text(root / rel_path).strip())
    return "\n\n---\n\n".join(part for part in parts if part.strip())


def main() -> int:
    args = parse_args()
    root = resolve_skill_root(args.root)
    store = FileControlStore(root)
    run_plan = store.get_run_plan(args.run_id)

    runtime_root = Path(args.runtime_dir).expanduser().resolve() if args.runtime_dir else default_runtime_dir()
    wrapper_path = runtime_root / "openclaw_debate_agent.py"
    debate_state_path = runtime_root / "debate_state.py"
    if not wrapper_path.is_file() or not debate_state_path.is_file():
        raise SystemExit(
            f"Deployed runtime helpers are missing under {runtime_root}. Expected openclaw_debate_agent.py and debate_state.py."
        )

    command_map_dir = Path(args.command_map_dir).resolve() if args.command_map_dir else (root / "generated" / "command-maps")
    template_dir = Path(args.template_dir).resolve() if args.template_dir else (root / "generated" / "templates")
    prompt_bundle_dir = root / "generated" / "prompt-bundles"
    runtime_context_dir = root / "generated" / "runtime-context"
    command_map_dir.mkdir(parents=True, exist_ok=True)
    template_dir.mkdir(parents=True, exist_ok=True)
    prompt_bundle_dir.mkdir(parents=True, exist_ok=True)
    runtime_context_dir.mkdir(parents=True, exist_ok=True)

    command_map = build_command_map(run_plan, wrapper_path=wrapper_path, env_file=openclaw_env_file())
    command_map_path = dump_json(command_map_dir / f"{args.run_id}-command-map.json", command_map)

    prompt_bundle = {
        role_name: render_role_prompt(root, run_plan, role_name=role_name, runtime_root=str(runtime_root))
        for role_name in ROLE_TO_SLOT
    }
    prompt_bundle_path = dump_json(prompt_bundle_dir / f"{args.run_id}-prompts.json", prompt_bundle)

    runtime_context_payload = {
        "run_id": run_plan["run_id"],
        "workflow_ref": run_plan["workflow_ref"],
        "engine_workflow_ref": run_plan["engine_workflow_ref"],
        "preset_ref": run_plan["preset_ref"],
        "runtime_context": run_plan.get("runtime_context", {}),
        "resolved_model_profile": run_plan.get("resolved_model_profile", {}),
    }
    runtime_context_path = dump_json(runtime_context_dir / f"{args.run_id}-context.json", runtime_context_payload)
    clawteam_data_dir = str(root / "generated" / "clawteam-data")

    prepare_script = engine_script_path(root, "prepare_debate.py")
    prepare_cmd = [
        "python3",
        str(prepare_script),
        "--workflow",
        "review-loop",
        "--team",
        run_plan["run_id"],
        "--goal",
        run_plan["request_snapshot"]["goal"],
        "--proposer-count",
        str(run_plan["protocol_defaults"]["proposer_count"]),
        "--backend",
        run_plan["launch_spec"]["backend"],
        "--command",
        "openclaw",
        "--runtime-root",
        str(runtime_root),
        "--template-dir",
        str(template_dir),
        "--agent-command-map-file",
        str(command_map_path),
        "--prompt-bundle-file",
        str(prompt_bundle_path),
    ]
    if run_plan["protocol_defaults"].get("review_rounds") is not None:
        prepare_cmd.extend(["--max-review-rounds", str(run_plan["protocol_defaults"]["review_rounds"])])
    if run_plan["protocol_defaults"].get("rebuttal_rounds") is not None:
        prepare_cmd.extend(["--max-rebuttal-rounds", str(run_plan["protocol_defaults"]["rebuttal_rounds"])])
    if args.reset_state:
        prepare_cmd.append("--reset-state")

    payload = {
        "run_id": run_plan["run_id"],
        "workflow_ref": run_plan["engine_workflow_ref"],
        "workflow_name": "review-loop",
        "command_map_path": str(command_map_path),
        "prompt_bundle_path": str(prompt_bundle_path),
        "runtime_context_path": str(runtime_context_path),
        "clawteam_data_dir": clawteam_data_dir,
        "template_dir": str(template_dir),
        "prepare_command": prepare_cmd,
    }
    if args.dry_run:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    env = os.environ.copy()
    env["PYTHONPATH"] = str(prepare_script.parent)
    env.setdefault("CLAWTEAM_DATA_DIR", clawteam_data_dir)
    result = subprocess.run(prepare_cmd, env=env, check=False, capture_output=True, text=True)
    payload["prepare_stdout"] = result.stdout
    payload["prepare_stderr"] = result.stderr
    payload["returncode"] = result.returncode
    payload.update(parse_prepare_output(result.stdout))
    if payload.get("launch_command"):
        payload["launch_command"] = shlex.split(payload["launch_command"])
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
