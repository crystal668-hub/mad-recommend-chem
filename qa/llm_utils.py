from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional


def invoke_llm(llm: Any, messages: List[Dict[str, str]]) -> Any:
    if hasattr(llm, "invoke"):
        response = llm.invoke(messages)
    elif callable(llm):
        response = llm(messages)
    else:
        raise TypeError("LLM must be callable or expose invoke(messages)")
    return getattr(response, "content", response)


def coerce_llm_text(payload: Any) -> str:
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload
    if isinstance(payload, list):
        parts: List[str] = []
        for item in payload:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)
    return str(payload)


def parse_json_object(payload: Any) -> Optional[Dict[str, Any]]:
    parsed = parse_json_payload(payload)
    return parsed if isinstance(parsed, dict) else None


def parse_json_array(payload: Any) -> Optional[List[Any]]:
    parsed = parse_json_payload(payload)
    return parsed if isinstance(parsed, list) else None


def parse_json_payload(payload: Any) -> Optional[Any]:
    if isinstance(payload, (dict, list)):
        return payload
    text = coerce_llm_text(payload).strip()
    if not text:
        return None
    candidate = _strip_code_fences(text)
    parsed = _try_json_load(candidate)
    if parsed is not None:
        return parsed
    for opener, closer in (("{", "}"), ("[", "]")):
        fragment = _extract_balanced_json(candidate, opener=opener, closer=closer)
        if fragment is None:
            continue
        parsed = _try_json_load(fragment)
        if parsed is not None:
            return parsed
    return None


def _strip_code_fences(text: str) -> str:
    if text.startswith("```"):
        return re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.S).strip()
    return text


def _try_json_load(text: str) -> Optional[Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _extract_balanced_json(text: str, *, opener: str, closer: str) -> Optional[str]:
    start = text.find(opener)
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None
