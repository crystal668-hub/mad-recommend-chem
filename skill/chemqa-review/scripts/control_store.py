#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class FileControlStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.control = self.root / "control"
        self.workflows = self.root / "workflows"
        self.presets = self.root / "presets"
        self.generated = self.root / "generated"

    def _dir(self, *parts: str) -> Path:
        path = self.control.joinpath(*parts)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _load_json(self, path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    def _dump_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def get_workflow(self, workflow_ref: str) -> dict[str, Any]:
        return self._load_json(self.workflows / f"{workflow_ref}.json")

    def get_preset(self, preset_ref: str) -> dict[str, Any]:
        return self._load_json(self.presets / f"{preset_ref}.json")

    def get_model_profile(self, profile_id: str) -> dict[str, Any]:
        return self._load_json(self._dir("model-profiles") / f"{profile_id}.json")

    def get_config_snapshot(self, snapshot_id: str) -> dict[str, Any]:
        return self._load_json(self._dir("config-snapshots") / f"{snapshot_id}.json")

    def save_run_plan(self, payload: dict[str, Any]) -> None:
        self._dump_json(self._dir("runplans") / f"{payload['run_id']}.json", payload)

    def get_run_plan(self, run_id: str) -> dict[str, Any]:
        return self._load_json(self._dir("runplans") / f"{run_id}.json")

    def update_run_status(self, run_id: str, payload: dict[str, Any]) -> None:
        self._dump_json(self._dir("run-status") / f"{run_id}.json", payload)

    def generated_paths_for_run(self, run_id: str) -> dict[str, Path]:
        return {
            "command_map": self.generated / "command-maps" / f"{run_id}-command-map.json",
            "prompt_bundle": self.generated / "prompt-bundles" / f"{run_id}-prompts.json",
            "runtime_context": self.generated / "runtime-context" / f"{run_id}-context.json",
        }
