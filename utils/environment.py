"""Project environment loading helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import dotenv_values, load_dotenv


def load_project_environment(env_path: Optional[Path | str] = None) -> Optional[Path]:
    """Replace inherited values for keys declared by the project .env file."""
    target = (
        Path(env_path)
        if env_path is not None
        else Path(__file__).resolve().parents[1] / ".env"
    )
    if not target.is_file():
        return None

    declared_values = dotenv_values(target)
    for key in declared_values:
        os.environ.pop(key, None)
    load_dotenv(dotenv_path=target, override=True)
    return target
