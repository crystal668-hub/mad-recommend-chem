from __future__ import annotations

import inspect
import os
from typing import Any, Dict, Optional, Tuple, Type


def resolve_env_var(value: Optional[str]) -> Optional[str]:
    if not value:
        return value
    if value.startswith("${") and value.endswith("}"):
        return os.getenv(value[2:-1])
    return value


def lazy_chat_model_import() -> Type[Any]:
    try:
        from langchain_openai import ChatOpenAI
    except Exception as exc:
        raise ImportError(
            "LangChain dependencies not found. Install: langchain-core langchain-openai."
        ) from exc
    return ChatOpenAI


def describe_chat_model_config(model_config: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], bool]:
    config = dict(model_config or {})
    provider = str(config.get("provider") or "").strip() or None
    model = str(config.get("model") or "").strip() or None
    has_api_key = bool(resolve_env_var(config.get("api_key")))
    return provider, model, has_api_key


def build_chat_model_from_config(model_config: Dict[str, Any]) -> Any:
    ChatOpenAI = lazy_chat_model_import()

    api_key = resolve_env_var(model_config.get("api_key"))
    if not api_key:
        raise ValueError("API key not provided (or env var not set) in model_config")

    base_url = model_config.get("base_url")
    model = model_config.get("model")
    temperature = float(model_config.get("temperature", 0.9))
    max_tokens = int(model_config.get("max_tokens", 2000))

    sig = inspect.signature(ChatOpenAI)
    kwargs: Dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if "api_key" in sig.parameters:
        kwargs["api_key"] = api_key
    elif "openai_api_key" in sig.parameters:
        kwargs["openai_api_key"] = api_key

    if "base_url" in sig.parameters:
        kwargs["base_url"] = base_url
    elif "openai_api_base" in sig.parameters:
        kwargs["openai_api_base"] = base_url

    timeout_s = model_config.get("timeout") or model_config.get("request_timeout")
    if timeout_s is not None:
        try:
            timeout_s = float(timeout_s)
        except Exception:
            timeout_s = None
    if timeout_s is not None:
        if "timeout" in sig.parameters:
            kwargs["timeout"] = timeout_s
        elif "request_timeout" in sig.parameters:
            kwargs["request_timeout"] = timeout_s

    return ChatOpenAI(**kwargs)
