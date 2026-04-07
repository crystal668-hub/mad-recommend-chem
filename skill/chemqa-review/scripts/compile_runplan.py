#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bundle_common import REQUIRED_SKILLS, dependency_report, missing_skills_from_report, resolve_skill_root, safe_session_id
from control_store import FileControlStore


def iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def default_run_id(preset_ref: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{preset_ref.split('@', 1)[0]}-{stamp}"


def apply_override(*, resolved: dict[str, Any], preset: dict[str, Any], key: str, value: Any) -> None:
    if value is None:
        return
    rule = preset.get("overrides", {}).get(key)
    if not rule or rule.get("exposure") != "tunable":
        raise SystemExit(f"Preset `{preset['id']}@{preset['version']}` does not allow override for `{key}`.")
    if "min" in rule and value < rule["min"]:
        raise SystemExit(f"Override `{key}`={value} is below minimum {rule['min']}.")
    if "max" in rule and value > rule["max"]:
        raise SystemExit(f"Override `{key}`={value} is above maximum {rule['max']}.")
    resolved[key] = value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile a chemqa-review run plan.")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]), help="chemqa-review skill root")
    parser.add_argument("--preset", required=True, help="Preset ref, e.g. chemqa-review@1")
    parser.add_argument("--goal", required=True, help="Debate goal or task prompt")
    parser.add_argument("--run-id", help="Optional explicit run id")
    parser.add_argument("--additional-file-workspace", help="Optional extra file workspace locator")
    parser.add_argument("--model-profile", help="Override model profile")
    parser.add_argument("--proposer-count", type=int)
    parser.add_argument("--review-rounds", type=int)
    parser.add_argument("--rebuttal-rounds", type=int)
    parser.add_argument("--evidence-mode", help="Override evidence mode")
    parser.add_argument("--priority", default="normal")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = resolve_skill_root(args.root)
    dependency_payload = dependency_report(root)
    missing_skills = missing_skills_from_report(dependency_payload)
    if missing_skills:
        raise SystemExit("Missing required sibling skills: " + ", ".join(missing_skills))

    store = FileControlStore(root)
    preset = store.get_preset(args.preset)
    workflow = store.get_workflow(preset["workflow_ref"])
    config_snapshot = store.get_config_snapshot("react-reviewed-default")

    resolved = dict(preset.get("defaults", {}))
    apply_override(resolved=resolved, preset=preset, key="model_profile", value=args.model_profile)
    apply_override(resolved=resolved, preset=preset, key="review_rounds", value=args.review_rounds)
    apply_override(resolved=resolved, preset=preset, key="rebuttal_rounds", value=args.rebuttal_rounds)
    apply_override(resolved=resolved, preset=preset, key="evidence_mode", value=args.evidence_mode)

    if resolved["model_profile"] not in preset.get("allowed_model_profiles", []):
        raise SystemExit(
            f"Model profile `{resolved['model_profile']}` is not allowed by preset `{args.preset}`."
        )

    if args.proposer_count is not None and args.proposer_count != resolved["proposer_count"]:
        raise SystemExit("chemqa-review fixes proposer_count to the preset default and does not allow overrides.")

    model_profile = store.get_model_profile(resolved["model_profile"])
    slot_models = dict(model_profile.get("slot_models") or {})
    required_slots = ("debate-coordinator", "debate-1", "debate-2", "debate-3", "debate-4", "debate-5")
    missing_slots = [slot_id for slot_id in required_slots if slot_id not in slot_models]
    if missing_slots:
        raise SystemExit("Model profile is missing required slots: " + ", ".join(missing_slots))

    role_map = dict(workflow.get("role_map") or {})
    role_contracts = dict((preset.get("prompt_pack") or {}).get("role_contracts") or {})
    shared_modules = list((preset.get("prompt_pack") or {}).get("shared_modules") or [])

    run_id = args.run_id or default_run_id(args.preset)
    session_assignments = {
        "debate-coordinator": safe_session_id("chemqa-review", run_id, "coordinator"),
        "debate-1": safe_session_id("chemqa-review", run_id, "proposer-1"),
        "debate-2": safe_session_id("chemqa-review", run_id, "proposer-2"),
        "debate-3": safe_session_id("chemqa-review", run_id, "proposer-3"),
        "debate-4": safe_session_id("chemqa-review", run_id, "proposer-4"),
        "debate-5": safe_session_id("chemqa-review", run_id, "proposer-5"),
    }
    additional_file_workspace = (args.additional_file_workspace or "").strip() or None
    prompt_assembly = {
        role_name: {
            "contracts": list(role_contracts.get(role_name, [])),
            "modules": list(shared_modules),
            "semantic_role": role_map[role_name],
        }
        for role_name in role_map
    }
    run_plan = {
        "run_id": run_id,
        "created_at": iso_now(),
        "request_snapshot": {
            "preset_ref": args.preset,
            "goal": args.goal,
            "inputs": {"additional_file_workspace": additional_file_workspace},
            "metadata": {"priority": args.priority},
            "overrides": resolved,
        },
        "workflow_ref": f"{workflow['id']}@{workflow['version']}",
        "preset_ref": f"{preset['id']}@{preset['version']}",
        "engine_workflow_ref": str(workflow.get("engine_workflow_ref") or "review-loop@1"),
        "engine_preset_ref": str(preset.get("engine_preset_ref") or "review-loop@1"),
        "resolved_model_profile": model_profile,
        "slot_assignments": slot_models,
        "session_assignments": session_assignments,
        "prompt_assembly": prompt_assembly,
        "launch_spec": {
            "backend": resolved.get("backend", "subprocess"),
            "final_decider": resolved.get("final_decider", "outer-entry-agent"),
            "proposer_slots": ["debate-1", "debate-2", "debate-3", "debate-4", "debate-5"],
            "coordinator_slot": "debate-coordinator",
            "engine_workflow_name": "review-loop",
        },
        "protocol_defaults": {
            "proposer_count": resolved["proposer_count"],
            "review_rounds": resolved.get("review_rounds"),
            "rebuttal_rounds": resolved.get("rebuttal_rounds"),
            "evidence_mode": resolved.get("evidence_mode"),
        },
        "runtime_context": {
            "additional_file_workspace": additional_file_workspace,
            "final_decider": resolved.get("final_decider"),
            "backend": resolved.get("backend", "subprocess"),
            "evidence_mode": resolved.get("evidence_mode"),
            "chemqa_review": {
                "engine_skill_root": str(root.parent / "debateclaw-v1"),
                "required_skills": list(REQUIRED_SKILLS),
                "role_map": role_map,
                "artifact_contract_version": "react-reviewed-v1",
                "react_reviewed_config_snapshot": config_snapshot,
                "acceptance_policy": {
                    "require_all_reviewers": True,
                    "review_failure_blocks_acceptance": True,
                    "stop_on_no_blocking_items": True,
                },
            },
        },
        "artifacts_root": f"<debate-run-root>/{run_id}/artifacts",
        "protocol_state_path": f"<debate-run-root>/{run_id}/state.db",
        "status": "planned",
    }
    if not args.dry_run:
        store.save_run_plan(run_plan)
        store.update_run_status(run_id, {"run_id": run_id, "status": "planned", "updated_at": iso_now()})
    print(json.dumps(run_plan, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
