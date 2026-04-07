#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


REQUIRED_SKILLS = (
    "debateclaw-v1",
    "paper-retrieval",
    "paper-access",
    "paper-parse",
    "paper-rerank",
)

ENGINE_REQUIRED_FILES = (
    "SKILL.md",
    "scripts/prepare_debate.py",
    "scripts/debate_state.py",
    "scripts/openclaw_debate_agent.py",
    "scripts/debate_templates.py",
)

LOCAL_RUNTIME_HELPERS = (
    "openclaw_debate_agent.py",
    "debate_state.py",
)

DEFAULT_TEMPLATES = (
    "debate-parallel-judge",
    "debate-review-loop",
)


def resolve_skill_root(value: str | Path | None, *, file_hint: str | Path | None = None) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    if file_hint is None:
        raise ValueError("skill root could not be resolved without a value or file hint")
    return Path(file_hint).resolve().parents[1]


def resolve_skills_root(skill_root: str | Path) -> Path:
    return Path(skill_root).expanduser().resolve().parent


def sibling_skill_root(skill_root: str | Path, skill_name: str) -> Path:
    return resolve_skills_root(skill_root) / skill_name


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump_json(path: str | Path, payload: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return target


def safe_session_id(*parts: str) -> str:
    raw = "-".join(part for part in parts if part)
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", raw)
    normalized = re.sub(r"-{2,}", "-", normalized).strip("-")
    if not normalized:
        raise SystemExit("Could not build a valid session id.")
    return normalized


def probe_binary(name: str, version_command: list[str]) -> dict[str, object]:
    resolved = shutil.which(name)
    if not resolved:
        return {"name": name, "found": False, "path": "", "version": ""}
    try:
        result = subprocess.run(version_command, check=False, capture_output=True, text=True)
    except OSError as exc:
        return {"name": name, "found": False, "path": resolved, "version": str(exc)}
    output = (result.stdout or result.stderr or "").strip()
    return {"name": name, "found": result.returncode == 0, "path": resolved, "version": output}


def collect_templates(template_dir: Path) -> list[str]:
    if not template_dir.is_dir():
        return []
    return sorted(path.stem for path in template_dir.glob("debate-*.toml"))


def collect_local_runtime_helpers(runtime_dir: Path) -> dict[str, bool]:
    return {name: (runtime_dir / name).is_file() for name in LOCAL_RUNTIME_HELPERS}


def dependency_report(skill_root: str | Path) -> dict[str, dict[str, Any]]:
    root = Path(skill_root).expanduser().resolve()
    report: dict[str, dict[str, Any]] = {}
    for skill_name in REQUIRED_SKILLS:
        sibling_root = sibling_skill_root(root, skill_name)
        payload: dict[str, Any] = {
            "name": skill_name,
            "present": sibling_root.is_dir(),
            "path": str(sibling_root),
        }
        if skill_name == "debateclaw-v1":
            payload["required_files"] = {
                rel: (sibling_root / rel).is_file()
                for rel in ENGINE_REQUIRED_FILES
            }
        elif sibling_root.is_dir():
            script_dir = sibling_root / "scripts"
            payload["has_skill_doc"] = (sibling_root / "SKILL.md").is_file()
            payload["scripts_present"] = script_dir.is_dir() and any(script_dir.glob("*.py"))
        report[skill_name] = payload
    return report


def missing_skills_from_report(report: dict[str, dict[str, Any]]) -> list[str]:
    missing: list[str] = []
    for skill_name, payload in report.items():
        if not payload.get("present"):
            missing.append(skill_name)
            continue
        required_files = payload.get("required_files") or {}
        if required_files and not all(required_files.values()):
            missing.append(skill_name)
    return missing


def load_module_from_path(module_name: str, path: str | Path):
    location = Path(path).expanduser().resolve()
    spec = importlib.util.spec_from_file_location(module_name, location)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {location}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def engine_skill_root(skill_root: str | Path) -> Path:
    root = sibling_skill_root(skill_root, "debateclaw-v1")
    if not root.is_dir():
        raise SystemExit(f"Required sibling skill is missing: {root}")
    return root


def engine_script_path(skill_root: str | Path, script_name: str) -> Path:
    path = engine_skill_root(skill_root) / "scripts" / script_name
    if not path.is_file():
        raise SystemExit(f"Required DebateClaw engine script is missing: {path}")
    return path


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def write_text(path: str | Path, text: str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


def run_json(command: list[str], *, cwd: Path) -> dict[str, Any]:
    result = subprocess.run(command, cwd=str(cwd), check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(
            f"Command failed ({result.returncode}): {' '.join(command)}\n\n"
            f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"Command did not return JSON: {' '.join(command)}\n\n"
            f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        ) from exc


def default_runtime_dir() -> Path:
    return (Path.home() / ".clawteam" / "debateclaw" / "bin").resolve()


def default_template_dir() -> Path:
    return (Path.home() / ".clawteam" / "templates").resolve()


def openclaw_env_file() -> str:
    return os.environ.get("OPENCLAW_ENV_FILE", str(Path.home() / ".openclaw" / ".env"))
