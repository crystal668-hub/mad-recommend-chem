"""
Prompt "building blocks" to keep long prompts maintainable.

This module intentionally stays dependency-free so it can be imported in tests and
runtime without requiring any LLM/provider packages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class PromptBlock:
    """
    A named chunk of prompt text.

    - `name` is used for deduplication in compose(...).
    - `priority` is informational ("MUST"/"SHOULD") for humans; it has no runtime effect.
    """

    name: str
    text: str
    priority: Optional[str] = None


def compose(*blocks: PromptBlock, sep: str = "\n\n") -> str:
    """
    Compose a prompt from blocks, deduplicating by block.name while preserving order.
    """

    seen: set[str] = set()
    parts: List[str] = []

    for b in blocks:
        if b is None:
            continue
        name = str(getattr(b, "name", "") or "").strip()
        if not name:
            # Do not try to guess; unnamed blocks are likely a bug.
            raise ValueError("PromptBlock.name must be a non-empty string")
        if name in seen:
            continue
        seen.add(name)

        text = str(getattr(b, "text", "") or "").strip()
        if text:
            parts.append(text)

    return sep.join(parts).strip()


def iter_nonempty_lines(text: str) -> Iterable[str]:
    """
    Utility for prompt linting: yields non-empty, stripped lines.
    """

    for line in str(text or "").splitlines():
        s = line.strip()
        if s:
            yield s

