"""
Agent factory (LangChain-based).

This replaces the previous OpenAI SDK based implementation. The agent is created via
LangChain ChatOpenAI (OpenAI-compatible endpoints supported) and uses tool-calling to
avoid manual tool message formatting.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from agents.react_agent import ReActAgent
from prompts.system_prompts import UNIFIED_SYSTEM_PROMPT


def create_agent(
    agent_type: str,
    agent_id: str,
    name: str,
    model_config: Dict[str, Any],
    rag_system=None,
    experience_store=None,
) -> ReActAgent:
    """
    Create an agent using the 'config.yaml' model settings.

    The project uses OpenAI-compatible endpoints:
    - OpenRouter for openai/deepseek/google/qwen(base model)
    - DashScope compatible-mode for 'Qwen Embedding' (when provider is `qwen` and base_url is not overridden)
    """

    provider_defaults = {
        "openai": {"base_url": "https://openrouter.ai/api/v1"},
        "deepseek": {"base_url": "https://openrouter.ai/api/v1"},
        "google": {"base_url": "https://openrouter.ai/api/v1"},
        "gemini": {"base_url": "https://openrouter.ai/api/v1"},
        "qwen": {"base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"},
    }

    provider_key = (agent_type or "").lower().strip()
    if provider_key not in provider_defaults:
        raise ValueError(f"Unsupported agent type: {agent_type}")

    merged_config = dict(model_config or {})
    if "base_url" not in merged_config:
        merged_config.update(provider_defaults[provider_key])

    return ReActAgent(
        agent_id=agent_id,
        name=name,
        model_config=merged_config,
        rag_system=rag_system,
        experience_store=experience_store,
        system_prompt=UNIFIED_SYSTEM_PROMPT,
    )

