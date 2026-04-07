#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bundle_common import (
    DEFAULT_TEMPLATES,
    collect_local_runtime_helpers,
    collect_templates,
    default_runtime_dir,
    default_template_dir,
    dependency_report,
    missing_skills_from_report,
    probe_binary,
    resolve_skill_root,
)


SUPPORTED_AGENTS = ("codex", "claude", "openclaw", "none", "auto")


def detect_agent(choice: str) -> str:
    if choice != "auto":
        return choice
    for candidate in ("openclaw", "codex", "claude"):
        if probe_binary(candidate, [candidate, "--version"]).get("found"):
            return candidate
    return "none"


def resolve_backend(agent_choice: str, backend_choice: str) -> str:
    if backend_choice != "auto":
        return backend_choice
    return "subprocess" if detect_agent(agent_choice) == "openclaw" else "tmux"


def build_report(
    *,
    skill_root: str | Path,
    agent_choice: str = "auto",
    backend_choice: str = "auto",
    require_clawteam: bool = False,
    require_runtime_assets: bool = False,
) -> dict[str, object]:
    root = resolve_skill_root(skill_root)
    selected_agent = detect_agent(agent_choice)
    selected_backend = resolve_backend(agent_choice, backend_choice)
    dependency_payload = dependency_report(root)
    missing_skills = missing_skills_from_report(dependency_payload)

    template_dir = default_template_dir()
    runtime_dir = default_runtime_dir()
    deployed_templates = collect_templates(template_dir)
    runtime_helpers = collect_local_runtime_helpers(runtime_dir)
    checks = {
        "uv": probe_binary("uv", ["uv", "--version"]),
        "tmux": probe_binary("tmux", ["tmux", "-V"]),
        "clawteam": probe_binary("clawteam", ["clawteam", "--version"]),
        "codex": probe_binary("codex", ["codex", "--version"]),
        "claude": probe_binary("claude", ["claude", "--version"]),
        "openclaw": probe_binary("openclaw", ["openclaw", "--version"]),
    }

    required_components = ["uv"]
    if require_clawteam:
        required_components.append("clawteam")
    if selected_backend == "tmux":
        required_components.append("tmux")
    if selected_agent != "none":
        required_components.append(selected_agent)
    missing_required_components = [name for name in required_components if not checks[name]["found"]]
    if require_runtime_assets:
        if any(name not in deployed_templates for name in DEFAULT_TEMPLATES):
            missing_required_components.append("debateclaw-templates")
        if not all(runtime_helpers.values()):
            missing_required_components.append("debateclaw-runtime")

    ready = not missing_skills and not missing_required_components
    return {
        "skill_root": str(root),
        "skills_root": str(root.parent),
        "selected_agent": selected_agent,
        "selected_backend": selected_backend,
        "required_skills": list(dependency_payload.keys()),
        "skill_dependencies": dependency_payload,
        "missing_skills": missing_skills,
        "checks": checks,
        "required_components": required_components,
        "missing_required_components": missing_required_components,
        "templates_dir": str(template_dir),
        "runtime_dir": str(runtime_dir),
        "deployed_debate_templates": deployed_templates,
        "runtime_helpers": runtime_helpers,
        "ready": ready,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check whether chemqa-review dependencies are ready.")
    parser.add_argument("--skill-root", help="chemqa-review skill root; defaults to this script's parent bundle")
    parser.add_argument("--agent", choices=SUPPORTED_AGENTS, default="auto")
    parser.add_argument("--backend", choices=("auto", "tmux", "subprocess"), default="auto")
    parser.add_argument("--require-clawteam", action="store_true")
    parser.add_argument("--require-runtime-assets", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(
        skill_root=args.skill_root or Path(__file__).resolve().parents[1],
        agent_choice=args.agent,
        backend_choice=args.backend,
        require_clawteam=args.require_clawteam,
        require_runtime_assets=args.require_runtime_assets,
    )
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
