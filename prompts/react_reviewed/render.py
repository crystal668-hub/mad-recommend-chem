from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from string import Template
from typing import Any


_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


@lru_cache(maxsize=None)
def load_template(name: str) -> str:
    return (_TEMPLATE_DIR / name).read_text(encoding="utf-8")


def render_template(name: str, **values: str) -> str:
    return Template(load_template(name)).substitute(**values).strip()


def json_block(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def json_preview(payload: Any, *, limit: int = 12000) -> str:
    text = json_block(payload)
    if len(text) > limit:
        return text[:limit] + "\n...(truncated)"
    return text
