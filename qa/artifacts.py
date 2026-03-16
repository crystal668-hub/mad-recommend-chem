from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from utils import ensure_dir, generate_timestamp, get_run_dir


class QAArtifactStore:
    def __init__(self, base_dir: Optional[str | Path] = None, run_id: Optional[str] = None) -> None:
        if base_dir is not None:
            root_dir = Path(base_dir)
        else:
            current_run_dir = get_run_dir()
            if current_run_dir is not None:
                root_dir = current_run_dir / "qa_artifacts"
            else:
                root_dir = Path("./logs/runs") / (run_id or generate_timestamp()) / "qa_artifacts"
        self.root_dir = ensure_dir(root_dir)

    def write_json(self, relative_path: str | Path, payload: Any) -> str:
        destination = self.root_dir / relative_path
        ensure_dir(destination.parent)
        with destination.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
        return str(destination)

    def write_text(self, relative_path: str | Path, content: str) -> str:
        destination = self.root_dir / relative_path
        ensure_dir(destination.parent)
        destination.write_text(content, encoding="utf-8")
        return str(destination)

    def write_bytes(self, relative_path: str | Path, content: bytes) -> str:
        destination = self.root_dir / relative_path
        ensure_dir(destination.parent)
        destination.write_bytes(content)
        return str(destination)

    def path(self, relative_path: str | Path) -> Path:
        return self.root_dir / relative_path
