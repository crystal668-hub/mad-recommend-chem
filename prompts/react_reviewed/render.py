from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from string import Template
from typing import Any

import yaml


_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


@lru_cache(maxsize=None)
def load_template(name: str) -> str:
    path = _TEMPLATE_DIR / name
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"React-reviewed prompt template {name!r} is not valid YAML.") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"React-reviewed prompt template {name!r} must be a YAML mapping with a 'prompt' field.")

    prompt = payload.get("prompt")
    if not isinstance(prompt, str):
        raise ValueError(f"React-reviewed prompt template {name!r} must define 'prompt' as a string.")
    if not prompt.strip():
        raise ValueError(f"React-reviewed prompt template {name!r} must define a non-empty 'prompt' string.")
    return prompt


def render_template(name: str, **values: str) -> str:
    return Template(load_template(name)).substitute(**values).strip()


def json_block(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def json_preview(payload: Any, *, limit: int = 12000) -> str:
    text = json_block(payload)
    if len(text) > limit:
        return text[:limit] + "\n...(truncated)"
    return text
