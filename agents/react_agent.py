"""
LangChain-based ReAct Agent.

Goal:
- Keep explicit "Thought -> Action -> Observation" trajectories.
- Avoid manual OpenAI tool message formatting by using LangChain message classes.
- Keep the original 4 tools: search_rag, search_experience, analyze, conclude.

Design:
Each ReAct iteration runs TWO LLM calls:
1) THOUGHT call (no tools): force explicit reasoning text.
2) ACTION call (with tools): model emits tool calls; we execute them and record observations.

Notes:
- The project currently uses OpenAI-compatible chat APIs (OpenRouter/DashScope). This agent
  builds a ChatOpenAI instance from config.yaml fields: model/api_key/base_url/temperature/max_tokens.
- LangChain is imported lazily so the codebase stays importable in minimal environments.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

from agents.react_reasoning import ReActTrajectory, ReActStep, ActionType, ToolCallRecord
from utils.source_id import build_chroma_source_id


@dataclass
class AgentResponse:
    """Agent response payload."""

    content: str
    reasoning: Optional[str] = None
    confidence: Optional[float] = None
    sources: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ToolResult:
    """Return type for tools that keeps raw data while displaying a compact observation."""

    observation: str
    data: Any = None

    def __str__(self) -> str:  # ToolMessage content
        return self.observation


def _resolve_env_var(value: Optional[str]) -> Optional[str]:
    if value and value.startswith("${") and value.endswith("}"):
        return os.getenv(value[2:-1])
    return value


def _lazy_langchain_imports():
    try:
        from langchain_openai import ChatOpenAI  
        from langchain_core.messages import (  
            SystemMessage,
            HumanMessage,
            AIMessage,
            ToolMessage,
        )
        from langchain_core.tools import StructuredTool  
    except Exception as e:  
        raise ImportError(
            "LangChain dependencies not found. Install: langchain-core langchain-openai."
        ) from e

    return ChatOpenAI, SystemMessage, HumanMessage, AIMessage, ToolMessage, StructuredTool


def _build_chat_model_from_config(model_config: Dict[str, Any]):
    ChatOpenAI, *_rest = _lazy_langchain_imports()

    api_key = _resolve_env_var(model_config.get("api_key"))
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

    # Support both old/new param names across langchain_openai versions.
    if "api_key" in sig.parameters:
        kwargs["api_key"] = api_key
    elif "openai_api_key" in sig.parameters:
        kwargs["openai_api_key"] = api_key

    if "base_url" in sig.parameters:
        kwargs["base_url"] = base_url
    elif "openai_api_base" in sig.parameters:
        kwargs["openai_api_base"] = base_url

    # Best-effort request timeout support (varies across langchain_openai versions).
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


class ReActAgent:
    """
    Memoryless-per-call agent that produces explicit ReAct trajectories.

    Compatibility:
    - Keeps generate_response_with_react(...) signature used by debate coordinators.
    - Keeps generate_response(...) for legacy callers.
    """

    def __init__(
        self,
        agent_id: str,
        name: str,
        model_config: Dict[str, Any],
        rag_system: Optional[Any] = None,
        experience_store: Optional[Any] = None,
        system_prompt: Optional[str] = None,
        max_react_steps: int = 10,
        verbose: bool = True,
    ) -> None:
        self.agent_id = agent_id
        self.name = name
        self.model_config = dict(model_config or {})
        self.rag_system = rag_system
        self.experience_store = experience_store
        self.system_prompt = system_prompt or ""
        self.max_react_steps = int(max_react_steps)
        self.verbose = bool(verbose)

        self.logger = logging.getLogger(f"MAD.agent.{self.agent_id}")

        self.current_trajectory: Optional[ReActTrajectory] = None

        # Lazy init
        self._llm = None

    # -------------------------
    # Tools (raw retrieval)
    # -------------------------

    def retrieve_knowledge(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.rag_system is None:
            raise RuntimeError("RAG System is not configured.")
        try:
            # Some adapters accept a `top_k` arg; others don't. Try both.
            try:
                results = self.rag_system.retrieve(query, top_k=int(top_k))
            except TypeError:
                results = self.rag_system.retrieve(query)
            return (results or [])[: int(top_k)]
        except Exception as e:
            return [{"text": "", "score": 0.0, "metadata": {}, "error": str(e)}]

    def retrieve_experience(self, components: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        if self.experience_store is None:
            return []
        try:
            return self.experience_store.query_experiences(components=components, top_k=int(top_k)) or []
        except Exception as e:
            return [{"error": str(e), "components": components}]

    # -------------------------
    # LangChain tool wrappers
    # -------------------------

    def _tool_search_rag(self, query: str, top_k: int = 5) -> ToolResult:
        # Overfetch then rerank by task relevance (elements + reaction type) to reduce "HEA keyword drift"
        # (e.g., being pulled toward popular but off-target systems like CoCrFeMnNi for HEA queries).
        try:
            overfetch = int(self.model_config.get("rag_overfetch", 3))
        except Exception:
            overfetch = 3
        overfetch = max(1, min(10, overfetch))
        fetch_k = max(int(top_k), int(top_k) * overfetch)

        results = self.retrieve_knowledge(query=query, top_k=fetch_k)

        # Enrich with stable source_id for later verification.
        collection = getattr(self.rag_system, "collection_name", "unknown")
        for item in results or []:
            meta = item.get("metadata") or {}
            doc_id = meta.get("doc_id") or meta.get("doi") or meta.get("id")

            # New schema (2026-02): chunk_id is a string uid used as Chroma id,
            # and chunk_index holds the numeric chunk index used for citations/source_id.
            chunk_index = meta.get("chunk_index")
            if chunk_index is None:
                chunk_index = meta.get("chunk_id")
            if doc_id is None or chunk_index is None:
                continue
            try:
                item["source_id"] = build_chroma_source_id(str(collection), str(doc_id), int(chunk_index))
            except Exception:
                continue

        # Annotate + rerank results using task constraints extracted from the call context.
        ctx_text = ""
        try:
            ctx_text = str(getattr(getattr(self, "current_trajectory", None), "query", "") or "")
        except Exception:
            ctx_text = ""

        task_components, task_reaction = _infer_task_constraints(ctx_text, fallback_components=None, fallback_reaction=None)
        required = [c for c in (task_components or []) if str(c).strip()]
        required_set = {c for c in required}

        for item in results or []:
            text = str(item.get("text") or "")
            detected = _extract_element_symbols(text)
            item["detected_elements"] = sorted(list(detected))
            item["required_elements"] = required

            match = detected.intersection(required_set) if required_set else set()
            missing = required_set.difference(detected) if required_set else set()
            forbidden = detected.difference(required_set.union(_ALLOWED_NON_CATALYST_ELEMENTS)) if required_set else set()

            item["element_match_count"] = len(match)
            item["element_match"] = sorted(list(match))
            item["element_missing"] = sorted(list(missing))
            item["forbidden_elements"] = sorted(list(forbidden))

            if task_reaction:
                item["reaction_match"] = _text_mentions_reaction(text, task_reaction)

        # Rerank: prefer correct reaction, more required elements, fewer forbidden metals, then original score.
        def _score_key(r: Dict[str, Any]) -> tuple:
            rm = bool(r.get("reaction_match")) if task_reaction else True
            match_n = int(r.get("element_match_count") or 0)
            forbid_n = len(r.get("forbidden_elements") or [])
            score = r.get("score")
            try:
                score_f = float(score) if score is not None else 0.0
            except Exception:
                score_f = 0.0
            return (int(rm), match_n, -forbid_n, score_f)

        try:
            results = sorted(list(results or []), key=_score_key, reverse=True)
        except Exception:
            results = results or []

        results = (results or [])[: int(top_k)]
        observation = _format_rag_observation(results)
        return ToolResult(observation=observation, data=results)

    def _tool_search_experience(self, components: List[str], top_k: int = 3) -> ToolResult:
        experiences = self.retrieve_experience(components=components, top_k=top_k)
        observation = _format_experience_observation(experiences)
        return ToolResult(observation=observation, data=experiences)

    def _tool_analyze(self, gap_analysis: str, next_step_plan: str) -> ToolResult:
        payload = {"gap_analysis": gap_analysis, "next_step_plan": next_step_plan}
        observation = (
            "Analysis Recorded:\n"
            f"- Gaps/Findings: {gap_analysis}\n"
            f"- Next Strategic Move: {next_step_plan}"
        )
        return ToolResult(observation=observation, data=payload)

    def _tool_conclude(self, conclusion: str) -> ToolResult:
        return ToolResult(observation=conclusion, data=conclusion)

    def _build_tools(self):
        _ChatOpenAI, _SystemMessage, _HumanMessage, _AIMessage, _ToolMessage, StructuredTool = _lazy_langchain_imports()

        tools = [
            StructuredTool.from_function(
                func=self._tool_search_rag,
                name=ActionType.SEARCH_RAG.value,
                description=(
                    "Search the local literature database (Chroma-backed RAG) and return relevant chunks. "
                    "Use AFTER `search_experience` when you need verifiable citations; cite source_id from results."
                ),
            ),
            StructuredTool.from_function(
                func=self._tool_search_experience,
                name=ActionType.SEARCH_EXPERIENCE.value,
                description="Search the experience database for similar past cases / guidelines (preferred before `search_rag`).",
            ),
            StructuredTool.from_function(
                func=self._tool_analyze,
                name=ActionType.ANALYZE.value,
                description="Stop searching and synthesize what is known/unknown; plan next step.",
            ),
            StructuredTool.from_function(
                func=self._tool_conclude,
                name=ActionType.CONCLUDE.value,
                description="Submit the final answer and stop.",
            ),
        ]
        tools_by_name = {t.name: t for t in tools}
        return tools, tools_by_name

    # -------------------------
    # LLM helpers
    # -------------------------

    def _get_llm(self):
        if self._llm is None:
            self._llm = _build_chat_model_from_config(self.model_config)
        return self._llm

    @staticmethod
    def _get_thought_phase_instruction() -> str:
        return (
            "CURRENT PHASE: THOUGHT\n"
            "Output ONLY a brief reasoning in plain text (1-3 short sentences).\n"
            "- Do NOT call tools.\n"
            "- Do NOT include any tool-call markup/tags (e.g. <invoke ...>, <tool_call ...>, <|...|>, DSML blocks).\n"
            "- Do NOT output JSON or markdown.\n"
            "- Avoid filler/self-talk (e.g. 'ok', 'sorry', repeating 'I will call tool').\n"
        )

    @staticmethod
    def _get_action_phase_instruction() -> str:
        return (
            "CURRENT PHASE: ACTION\n"
            "You MUST call one or more tools.\n"
            "Do NOT output normal text answers in this phase; use tool calls.\n"
            "Critical constraint:\n"
            "- In a single ACTION step, do NOT mix search tools (`search_rag`, `search_experience`) with "
            "analysis tools (`analyze`, `conclude`).\n"
            "  - If you need evidence: call one or more search tools first.\n"
            "  - After you receive observations, in the NEXT step you may call `analyze` or `conclude`.\n"
            "Rules:\n"
            "- Tool priority: prefer `search_experience` FIRST; then use `search_rag` for verifiable citations.\n"
            "- When calling `search_rag`, include the target reaction type (e.g., OER/HER/ORR/HOR) AND ALL provided metal element symbols in the query.\n"
            "- Do NOT drift to evidence about different catalyst metals; ignore off-target compositions.\n"
            "- Using tools instead of guessing.\n"
            "- If you have enough evidence, call `conclude`.\n"
        )

    # -------------------------
    # Public API
    # -------------------------

    def generate_response(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        components = None
        if context:
            components = context.get("components")
        response, _trajectory = self.generate_response_with_react(query=prompt, components=components, context=context)
        return response

    def generate_response_with_react(
        self,
        query: str,
        components: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
        system_prompt_override: Optional[str] = None,
        max_steps_override: Optional[int] = None,
        llm_timeout_seconds: Optional[float] = None,
    ) -> Tuple[AgentResponse, ReActTrajectory]:
        ChatOpenAI, SystemMessage, HumanMessage, AIMessage, ToolMessage, _StructuredTool = _lazy_langchain_imports()

        def _preview(obj: Any, limit: int = 1500) -> str:
            """Best-effort safe preview for logs (avoid huge payloads / non-serializable objects)."""
            try:
                s = json.dumps(obj, ensure_ascii=True, default=str)
            except Exception:
                s = str(obj)
            if len(s) > limit:
                return s[:limit] + "...(truncated)"
            return s

        full_query = query
        if components:
            full_query = f"{query}\n\nComponents: {', '.join(components)}"

        trajectory = ReActTrajectory(query=full_query)
        self.current_trajectory = trajectory

        effective_max_steps = self.max_react_steps
        if max_steps_override is not None:
            try:
                effective_max_steps = max(1, int(max_steps_override))
            except Exception:
                effective_max_steps = self.max_react_steps

        self.logger.info(
            "react_call_start",
            extra={
                "event": "agent.react.start",
                "agent_id": self.agent_id,
                "agent_name": self.name,
                "max_react_steps": effective_max_steps,
                "has_components": bool(components),
                "query_preview": (full_query or "")[:500],
            },
        )

        # If the caller supplies an explicit timeout, build an ephemeral model instance
        # with that timeout to avoid long-hanging network calls during debates.
        if llm_timeout_seconds is not None:
            llm_cfg = dict(self.model_config or {})
            llm_cfg["timeout"] = float(llm_timeout_seconds)
            llm = _build_chat_model_from_config(llm_cfg)
        else:
            llm = self._get_llm()
        tools, tools_by_name = self._build_tools()

        # Bind tools to enable tool-calling.
        if hasattr(llm, "bind_tools"):
            llm_with_tools = llm.bind_tools(tools)
        else:  # pragma: no cover
            llm_with_tools = llm.bind(tools=tools)

        system_prompt = system_prompt_override if system_prompt_override is not None else self.system_prompt
        messages: List[Any] = [SystemMessage(content=system_prompt), HumanMessage(content=full_query)]

        step_number = 0
        final_answer: Optional[str] = None
        no_tool_call_streak = 0

        # Many debate phases (REVIEW/REBUTTAL) require STRICT JSON; PROPOSE does not.
        requires_strict_json = "STRICT JSON" in (system_prompt or "").upper()
        task_components, task_reaction = _infer_task_constraints(
            full_query, fallback_components=components, fallback_reaction=None
        )

        # Local guards to reduce "self-looping" thought outputs.
        thought_max_chars = int(self.model_config.get("thought_max_chars", 300))
        thought_max_chars = max(50, thought_max_chars)
        no_tool_call_threshold = int(self.model_config.get("no_tool_call_threshold", 1))
        no_tool_call_threshold = max(1, no_tool_call_threshold)

        # Prepare a best-effort "force tool call" runnable for when models keep failing to emit tool_calls.
        llm_with_tools_forced = llm_with_tools
        try:
            if hasattr(llm, "bind_tools"):
                llm_with_tools_forced = llm.bind_tools(tools, tool_choice="required")
            else:  # pragma: no cover
                llm_with_tools_forced = llm.bind(tools=tools, tool_choice="required")
        except Exception:
            llm_with_tools_forced = llm_with_tools

        provider_hint = str(self.model_config.get("provider") or "").strip().lower()
        model_name_hint = str(self.model_config.get("model") or "").strip().lower()
        use_user_role_for_thought = provider_hint in {"google", "gemini"} or ("gemini" in model_name_hint)

        while step_number < effective_max_steps:

            # ----- THOUGHT -----

            thought_instruction = self._get_thought_phase_instruction()
            thought_prompt_msg = SystemMessage(content=thought_instruction)
            if use_user_role_for_thought:
                # Some Gemini/OpenAI-compatible routes may return empty content for "thought" when sent as a SystemMessage.
                # Using a HumanMessage tends to be more reliable while still keeping the THOUGHT out of the chat history.
                thought_prompt_msg = HumanMessage(content=thought_instruction)
            
            # Get thinking content
            thought_msg = llm.invoke(messages + [thought_prompt_msg])
            if self.verbose:
                self.logger.debug(
                    "react_thought_raw",
                    extra={
                        "event": "agent.react.thought.raw",
                        "agent_id": self.agent_id,
                        "step": step_number + 1,
                        "thought_additional_kwargs": _preview(getattr(thought_msg, "additional_kwargs", None)),
                        "thought_tool_calls": _preview(getattr(thought_msg, "tool_calls", None)),
                        "raw_thought_text": (getattr(thought_msg, "content", "") or "")[:500],
                    },
                )
            thought_content = getattr(thought_msg, "content", "") or ""
            thought_content = thought_content.strip()
            if not thought_content:
                thought_content = _fallback_thought(
                    step_number=step_number,
                    task_reaction=task_reaction,
                    task_components=task_components,
                    trajectory=trajectory,
                )
            thought_content = _sanitize_thought(thought_content)
            if len(thought_content) > thought_max_chars:
                thought_content = thought_content[:thought_max_chars].rstrip()

            # ----- ACTION (tool calls) -----
            tool_calls: List[Dict[str, Any]] = []
            action_msg = None
            raw_action_text = ""
            for attempt in range(2):
                retry_hint = ""
                if attempt > 0:
                    retry_hint = (
                        "\nERROR: You did not call any tools in the previous ACTION attempt.\n"
                        "You MUST call at least one tool now (no free-form answers).\n"
                    )

                action_llm = llm_with_tools
                if no_tool_call_streak >= no_tool_call_threshold:
                    action_llm = llm_with_tools_forced

                # Get action content         
                action_msg = action_llm.invoke(
                    messages
                    + [
                        SystemMessage(content=self._get_action_phase_instruction() + retry_hint),
                        SystemMessage(content=f"THOUGHT (plan; do not repeat):\n{thought_content}"),
                    ]
                )
                tool_calls = _extract_tool_calls(action_msg)
                raw_action_text = (getattr(action_msg, "content", "") or "").strip()
                if self.verbose:
                    self.logger.debug(
                        "react_action_raw",
                        extra={
                            "event": "agent.react.action.raw",
                            "agent_id": self.agent_id,
                            "step": step_number + 1,
                            "attempt": attempt,
                            "forced_tool_choice": bool(action_llm is llm_with_tools_forced),
                            "action_additional_kwargs": _preview(getattr(action_msg, "additional_kwargs", None)),
                            "raw_action_text": raw_action_text[:1500] + ("...(truncated)" if len(raw_action_text) > 1500 else ""),
                        },
                    )
                if tool_calls:
                    break

            if not tool_calls:
                no_tool_call_streak += 1

                # Feed back a short, explicit failure note to the model to break "I will call tool..." loops.
                # Keep it concise to avoid polluting context.
                failure_note = (
                    "ACTION FAILURE: You did not emit any tool calls.\n"
                    "Next ACTION MUST emit at least one tool call via the tool-calling mechanism (no plain text).\n"
                    "If you need evidence, call `search_experience` and/or `search_rag`.\n"
                )
                # Avoid spamming the same failure note repeatedly.
                try:
                    last_content = getattr(messages[-1], "content", "") if messages else ""
                except Exception:
                    last_content = ""
                if not (isinstance(last_content, str) and last_content.startswith("ACTION FAILURE:")):
                    messages.append(SystemMessage(content=failure_note))

                # Record an explicit failure step, but DO NOT accept this as the final answer.
                # Also do NOT append the raw assistant content to the chat history, as it can be off-topic/noisy.
                step_number += 1
                observation = (
                    "ACTION FAILED: model did not emit any tool calls. "
                    "Retry in next step.\n"
                    f"Raw content (truncated): {(raw_action_text[:500] + '...') if len(raw_action_text) > 500 else raw_action_text}"
                )
                trajectory.add_step(
                    ReActStep(
                        step_number=step_number,
                        thought=thought_content,
                        action="no_tool_call",
                        action_input={},
                        observation=observation,
                        tool_calls=[],
                    )
                )
                continue

            no_tool_call_streak = 0

            # Only keep the ACTION assistant message in history if it actually issued tool calls.
            # Some providers may return legacy `function_call` instead of `tool_calls` (content can be empty).
            # When that happens, synthesize an assistant message with `tool_calls` so the following ToolMessage(s)
            # are consistent and future turns remain OpenAI-tool-call compatible.
            from_fc = any(isinstance(c, dict) and c.get("__from_function_call") for c in tool_calls)
            if from_fc:
                sanitized: List[Dict[str, Any]] = []
                for c in tool_calls:
                    if not isinstance(c, dict):
                        continue
                    cc = dict(c)
                    cc.pop("__from_function_call", None)
                    sanitized.append(cc)
                messages.append(AIMessage(content=raw_action_text or "", additional_kwargs={"tool_calls": sanitized}))
                tool_calls = sanitized
            else:
                messages.append(action_msg)

            normalized_calls: List[Tuple[str, Dict[str, Any], str]] = [
                _normalize_tool_call(call) for call in tool_calls
            ]

            if self.verbose:
                self.logger.debug(
                    "react_action_tools",
                    extra={
                        "event": "agent.react.action",
                        "agent_id": self.agent_id,
                        "step": step_number + 1,
                        "tools": [name for name, _args, _id in normalized_calls],
                    },
                )
            search_tools = {ActionType.SEARCH_RAG.value, ActionType.SEARCH_EXPERIENCE.value}
            analysis_tools = {ActionType.ANALYZE.value, ActionType.CONCLUDE.value}
            has_search = any(name in search_tools for name, _args, _id in normalized_calls)
            has_analysis = any(name in analysis_tools for name, _args, _id in normalized_calls)
            mixed_search_and_analysis = has_search and has_analysis
            mixed_error = (
                "Policy violation: mixed search and analysis in one ACTION step. "
                "Call only search tools first; after receiving observations, call `analyze`/`conclude` in the next step."
            )

            tool_call_records: List[ToolCallRecord] = []
            observation_sections: List[str] = []

            for tool_name, tool_args, tool_call_id in normalized_calls:

                # If the model tried to both search and analyze/conclude in the same ACTION step,
                # refuse the analysis/conclude calls (they wouldn't be grounded in the fresh observations).
                blocked_mixed = mixed_search_and_analysis and tool_name in analysis_tools

                # Guard: in PROPOSE (non-strict JSON) the final conclusion must stay on the provided metal elements.
                blocked_guard = False
                guard_msg = ""
                if (
                    tool_name == ActionType.CONCLUDE.value
                    and (not requires_strict_json)
                    and task_components
                ):
                    conclusion_text = ""
                    if isinstance(tool_args, dict):
                        conclusion_text = str(tool_args.get("conclusion") or "")
                    ok, reason = _validate_conclusion_against_task_with_evidence(conclusion_text, task_components, trajectory)
                    if not ok:
                        blocked_guard = True
                        try:
                            self.logger.warning(
                                "react_conclude_guard_blocked",
                                extra={
                                    "event": "agent.react.conclude.guard_blocked",
                                    "agent_id": self.agent_id,
                                    "step": step_number + 1,
                                    "reason": reason,
                                    "required_components": task_components,
                                    "cited_source_ids": _extract_source_ids_from_text(conclusion_text)[:10],
                                },
                            )
                        except Exception:
                            pass
                        guard_msg = (
                            "Conclusion out of scope: " + reason + "\n"
                            "You MUST revise and conclude for the exact metal elements: "
                            + ", ".join([str(c) for c in (task_components or [])])
                            + ".\n"
                            "Do NOT introduce other catalyst metals (e.g., Cr, Mn)."
                        )

                blocked = blocked_mixed or blocked_guard
                if blocked:
                    if blocked_mixed:
                        result = ToolResult(observation=mixed_error, data={"error": "mixed_search_and_analysis"})
                    else:
                        result = ToolResult(observation=guard_msg or "Conclusion rejected by task-alignment guard.", data={"error": "conclusion_out_of_scope"})
                else:
                    tool = tools_by_name.get(tool_name)
                    if tool is None:
                        result = ToolResult(observation=f"Error: Unknown tool '{tool_name}'", data=None)
                    else:
                        try:
                            result = tool.invoke(tool_args)
                            if not isinstance(result, ToolResult):
                                result = ToolResult(observation=str(result), data=result)
                        except Exception as e:
                            result = ToolResult(observation=f"Tool error: {str(e)}", data=None)

                observation_text = str(result)
                messages.append(ToolMessage(content=observation_text, tool_call_id=tool_call_id))

                tool_call_records.append(
                    ToolCallRecord(
                        tool_name=tool_name,
                        tool_call_id=tool_call_id,
                        tool_args=tool_args,
                        observation=observation_text,
                        observation_data=result.data,
                    )
                )
                observation_sections.append(f"--- Observation ({tool_name}) ---\n{observation_text}")

                if self.verbose and tool_name in {ActionType.SEARCH_RAG.value, ActionType.SEARCH_EXPERIENCE.value}:
                    n_items = len(result.data) if isinstance(result.data, list) else None
                    sid_preview: List[str] = []
                    if tool_name == ActionType.SEARCH_RAG.value and isinstance(result.data, list):
                        for item in result.data[:3]:
                            if isinstance(item, dict) and item.get("source_id"):
                                sid_preview.append(str(item["source_id"]))
                    self.logger.debug(
                        "react_tool_observation",
                        extra={
                            "event": "agent.react.observation",
                            "agent_id": self.agent_id,
                            "step": step_number + 1,
                            "tool": tool_name,
                            "items": n_items,
                            "source_id_preview": sid_preview,
                        },
                    )

                if tool_name == ActionType.CONCLUDE.value and not blocked:
                    final_answer = observation_text
                    break

            step_number += 1
            aggregated_observation = "\n\n".join(observation_sections).strip()
            if not aggregated_observation:
                aggregated_observation = "(no observation)"

            # Keep legacy single-action fields populated for compatibility/debugging,
            # but the authoritative per-call details live in `tool_calls`.
            if len(tool_call_records) == 1:
                action = tool_call_records[0].tool_name
                action_input = tool_call_records[0].tool_args
                tool_call_id = tool_call_records[0].tool_call_id
                observation_data = tool_call_records[0].observation_data
            else:
                action = "multi_tool"
                action_input = {"tool_calls": [{"tool_name": c.tool_name, "tool_args": c.tool_args} for c in tool_call_records]}
                tool_call_id = None
                observation_data = None

            trajectory.add_step(
                ReActStep(
                    step_number=step_number,
                    thought=thought_content,
                    action=action,
                    action_input=action_input,
                    observation=aggregated_observation,
                    tool_call_id=tool_call_id,
                    observation_data=observation_data,
                    tool_calls=tool_call_records,
                )
            )

            if final_answer is not None:
                break

        if final_answer is None:
            # Hard fallback: ask for a final answer text, then record it via the `conclude` tool
            # so downstream callers (tests / debate protocol) see a proper CONCLUDE action.
            self.logger.warning(
                "react_forced_conclude",
                extra={
                    "event": "agent.react.forced_conclude",
                    "agent_id": self.agent_id,
                    "steps_so_far": step_number,
                },
            )
            retrieved = _collect_retrieved_source_ids_from_trajectory(trajectory)
            sid_hint = ""
            if retrieved:
                sid_hint = (
                    "\nYou MUST cite at least one of these source_id values verbatim in your final answer:\n"
                    + "\n".join(f"- {sid}" for sid in sorted(list(retrieved))[:10])
                )
            elements_hint = ""
            if task_components:
                elements_hint = (
                    "\nThe metal catalyst elements for this task are EXACTLY:\n- "
                    + ", ".join([str(c) for c in (task_components or [])])
                    + "\nDo NOT introduce other catalyst metals."
                )

            # Some callers (e.g. debate REVIEW/REBUTTAL) require strict JSON only; don't inject free-form text
            # or extra evidence lines that would break parsing.
            requires_strict_json = "STRICT JSON" in (system_prompt or "").upper()

            # Prefer forcing a `conclude` tool call. Some providers (notably Gemini via OpenAI-compatible routes)
            # can return empty `content` for a free-form completion here; tool-calling is more reliable.
            draft = ""
            forced_conclude_tool_call_id = "forced_conclude"

            # Best-effort: force a specific tool if the backend supports it; otherwise fall back to "required".
            force_conclude_llm = llm_with_tools_forced
            try:
                if hasattr(llm, "bind_tools"):
                    force_conclude_llm = llm.bind_tools(tools, tool_choice=ActionType.CONCLUDE.value)
                else:  # pragma: no cover
                    force_conclude_llm = llm.bind(tools=tools, tool_choice=ActionType.CONCLUDE.value)
            except Exception:
                force_conclude_llm = llm_with_tools_forced

            forced_action_msg = None
            try:
                forced_action_msg = force_conclude_llm.invoke(
                    messages
                    + [
                        SystemMessage(
                            content=(
                                (
                                    "FINAL ACTION: You MUST call ONLY the `conclude` tool now.\n"
                                    "Set the `conclusion` argument to STRICT JSON ONLY that follows the schema in the system prompt EXACTLY.\n"
                                    "- No markdown, no extra text.\n"
                                    "- If evidence is required, cite at least one verifiable source_id.\n"
                                    + sid_hint
                                    + elements_hint
                                )
                                if requires_strict_json
                                else (
                                    "FINAL ACTION: You MUST call ONLY the `conclude` tool now.\n"
                                    "Set the `conclusion` argument to the best possible final answer.\n"
                                    "- Include the reaction type explicitly.\n"
                                    "- Explicitly restate the catalyst metal elements exactly as provided.\n"
                                    "- Summarize key performance metrics.\n"
                                    "- If you used literature evidence, cite source_id exactly as provided.\n"
                                    + sid_hint
                                    + elements_hint
                                )
                            )
                        )
                    ]
                )
            except Exception:
                forced_action_msg = None

            if self.verbose and forced_action_msg is not None:
                self.logger.debug(
                    "react_forced_conclude_action_raw",
                    extra={
                        "event": "agent.react.forced_conclude.action.raw",
                        "agent_id": self.agent_id,
                        "forced_action_additional_kwargs": _preview(getattr(forced_action_msg, "additional_kwargs", None)),
                        "forced_action_tool_calls": _preview(getattr(forced_action_msg, "tool_calls", None)),
                        "forced_action_text": (getattr(forced_action_msg, "content", "") or "")[:1500],
                    },
                )

            if forced_action_msg is not None:
                forced_calls = _extract_tool_calls(forced_action_msg)
                for name, args, call_id in (_normalize_tool_call(c) for c in forced_calls):
                    if name != ActionType.CONCLUDE.value:
                        continue
                    if call_id:
                        forced_conclude_tool_call_id = call_id
                    conclusion = None
                    if isinstance(args, dict):
                        conclusion = args.get("conclusion") or args.get("final_answer")
                    if isinstance(conclusion, (dict, list)):
                        try:
                            conclusion = json.dumps(conclusion, ensure_ascii=False)
                        except Exception:
                            conclusion = str(conclusion)
                    if conclusion is not None:
                        draft = str(conclusion).strip()
                        break

            if not draft:
                # Fallback: ask for a final answer in free-form text (no tools).
                forced = llm.invoke(
                    messages
                    + [
                        SystemMessage(
                            content=(
                                (
                                    "FINAL PHASE: Output STRICT JSON ONLY.\n"
                                    "- Follow the output schema in the system prompt EXACTLY.\n"
                                "- No markdown, no extra text.\n"
                                "- If evidence is required, cite at least one verifiable source_id.\n"
                                + sid_hint
                                + elements_hint
                            )
                            if requires_strict_json
                            else (
                                "FINAL PHASE: Write the best possible final answer now.\n"
                                "- Include the reaction type explicitly.\n"
                                "- Explicitly restate the catalyst metal elements exactly as provided.\n"
                                "- Summarize key performance metrics.\n"
                                "- If you used literature evidence, cite source_id exactly as provided.\n"
                                + sid_hint
                                + elements_hint
                            )
                        )
                    )
                ]
                )
                if self.verbose:
                    self.logger.debug(
                        "react_forced_conclude_text_raw",
                        extra={
                            "event": "agent.react.forced_conclude.text.raw",
                            "agent_id": self.agent_id,
                            "forced_text_additional_kwargs": _preview(getattr(forced, "additional_kwargs", None)),
                            "forced_text_tool_calls": _preview(getattr(forced, "tool_calls", None)),
                            "forced_text": (getattr(forced, "content", "") or "")[:1500],
                        },
                    )
                draft = (getattr(forced, "content", "") or "").strip()
                if not draft:
                    # Some providers return a legacy function_call even when tools are not bound.
                    fc = (getattr(forced, "additional_kwargs", {}) or {}).get("function_call")
                    if isinstance(fc, dict):
                        args = fc.get("arguments")
                        parsed = None
                        if isinstance(args, str):
                            try:
                                parsed = json.loads(args)
                            except Exception:
                                parsed = None
                        elif isinstance(args, dict):
                            parsed = args
                        if isinstance(parsed, dict):
                            draft = str(parsed.get("conclusion") or parsed.get("final_answer") or "").strip()
            if not draft:
                draft = "No conclusion generated."

            # If strict JSON is not required and the model forgot to include a verifiable source_id,
            # attach a minimal evidence line.
            if (not requires_strict_json) and retrieved and not any(sid in draft for sid in retrieved):
                draft = draft.rstrip() + "\n\nEvidence (retrieved source_id): " + ", ".join(sorted(list(retrieved))[:3])

            final_answer = draft

            # Record a final CONCLUDE step.
            step_number += 1
            tool_call = ToolCallRecord(
                tool_name=ActionType.CONCLUDE.value,
                tool_call_id=forced_conclude_tool_call_id,
                tool_args={"conclusion": final_answer},
                observation=final_answer,
                observation_data=final_answer,
            )
            trajectory.add_step(
                ReActStep(
                    step_number=step_number,
                    thought="Forced conclusion (model failed to call conclude tool).",
                    action=ActionType.CONCLUDE.value,
                    action_input={"conclusion": final_answer},
                    observation=final_answer,
                    tool_call_id=forced_conclude_tool_call_id,
                    observation_data=final_answer,
                    tool_calls=[tool_call],
                )
            )

        trajectory.finalize(final_answer)
        retrieved_ids = _collect_retrieved_source_ids_from_trajectory(trajectory)
        self.logger.info(
            "react_call_end",
            extra={
                "event": "agent.react.end",
                "agent_id": self.agent_id,
                "steps": getattr(trajectory, "total_steps", 0),
                "retrieved_source_id_count": len(retrieved_ids),
                "final_answer_preview": (final_answer or "")[:500],
            },
        )
        response = AgentResponse(
            content=final_answer,
            reasoning=trajectory.get_trajectory_summary(),
            sources=_extract_sources(trajectory),
        )
        return response, trajectory

    def save_trajectory(self, output_path: str) -> None:
        if not self.current_trajectory:
            return
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(self.current_trajectory.to_json())


def _extract_tool_calls(action_msg: Any) -> List[Dict[str, Any]]:
    """
    Extract tool calls from an LLM message across provider/model variants.

    Supported shapes (best-effort):
    - LangChain tool-calling: message.tool_calls (list)
    - OpenAI-compatible: message.additional_kwargs["tool_calls"] (list)
    - Legacy function calling (common on some OpenAI-compatible routes):
      message.additional_kwargs["function_call"] (dict) or ["function_calls"] (list)
    """
    # LangChain AIMessage typically exposes .tool_calls
    tool_calls = getattr(action_msg, "tool_calls", None)
    if isinstance(tool_calls, list) and tool_calls:
        return tool_calls

    # Fallback: additional_kwargs
    add = getattr(action_msg, "additional_kwargs", {}) or {}
    tool_calls = add.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        return tool_calls
    if isinstance(tool_calls, dict):
        return [tool_calls]

    # Legacy: single function_call dict
    fc = add.get("function_call") or getattr(action_msg, "function_call", None)
    if isinstance(fc, dict) and fc.get("name"):
        call_id = fc.get("id") or f"fc_{id(action_msg)}_0"
        arguments = fc.get("arguments")
        if arguments is None:
            arguments = fc.get("args") or {}
        return [
            {
                "id": str(call_id),
                "function": {"name": str(fc.get("name")), "arguments": arguments},
                "__from_function_call": True,
            }
        ]

    # Legacy: list of function calls
    fcs = add.get("function_calls")
    if isinstance(fcs, list) and fcs:
        out: List[Dict[str, Any]] = []
        for i, item in enumerate(fcs):
            if not isinstance(item, dict):
                continue
            name = item.get("name") or (item.get("function") or {}).get("name")
            if not name:
                continue
            arguments = item.get("arguments") or (item.get("function") or {}).get("arguments") or item.get("args") or {}
            out.append(
                {
                    "id": str(item.get("id") or f"fc_{id(action_msg)}_{i}"),
                    "function": {"name": str(name), "arguments": arguments},
                    "__from_function_call": True,
                }
            )
        if out:
            return out

    return []



def _normalize_tool_call(call: Any) -> Tuple[str, Dict[str, Any], str]:
    """
    Normalize tool call formats across LangChain/openai variants.
    Returns (name, args, id).
    """
    if isinstance(call, dict):
        call_id = call.get("id") or call.get("tool_call_id") or f"tool_{id(call)}"

        # New style: {"name": "...", "args": {...}}
        name = call.get("name")
        args = call.get("args")
        if name and isinstance(args, dict):
            return str(name), args, str(call_id)

        # OpenAI style: {"function": {"name": "...", "arguments": "..."}}
        fn = call.get("function") or {}
        fn_name = fn.get("name")
        fn_args = fn.get("arguments")
        if fn_name:
            if isinstance(fn_args, dict):
                return str(fn_name), fn_args, str(call_id)
            if isinstance(fn_args, str):
                try:
                    import json

                    return str(fn_name), json.loads(fn_args), str(call_id)
                except Exception:
                    return str(fn_name), {}, str(call_id)

    # Unknown shape
    return "unknown_tool", {}, f"tool_{id(call)}"


def _sanitize_thought(text: str) -> str:
    """
    Some models emit tool-call markup in plain text (e.g. DSML/XML blocks) even when tools are disabled.
    That text pollutes later calls if we feed it back. Keep only a readable thought.
    """
    if not text:
        return ""

    # Remove common tool-call block patterns (best-effort).
    cleaned = re.sub(r"<[^>]*function_calls[^>]*>.*?</[^>]*function_calls[^>]*>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<[^>]*(invoke|tool_call)[^>]*>.*?</[^>]*(invoke|tool_call)[^>]*>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)

    # Drop standalone markup-like lines that usually appear with DSML (<|...|> or <｜...｜>).
    kept: List[str] = []
    for line in cleaned.splitlines():
        stripped = line.strip()
        if not stripped:
            kept.append(line)
            continue
        if stripped.startswith("<") and ("DSML" in stripped or "function_calls" in stripped or "invoke" in stripped):
            continue
        kept.append(line)
    cleaned = "\n".join(kept).strip()
    return cleaned


_VALID_ELEMENT_SYMBOLS: set[str] = {
    # Periodic table symbols (1-118). Used for lightweight element extraction from text.
    "H","He","Li","Be","B","C","N","O","F","Ne",
    "Na","Mg","Al","Si","P","S","Cl","Ar",
    "K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
    "Ga","Ge","As","Se","Br","Kr",
    "Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd",
    "In","Sn","Sb","Te","I","Xe",
    "Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu",
    "Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg",
    "Tl","Pb","Bi","Po","At","Rn",
    "Fr","Ra","Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr",
    "Rf","Db","Sg","Bh","Hs","Mt","Ds","Rg","Cn",
    "Nh","Fl","Mc","Lv","Ts","Og",
}

# Elements commonly present in electrolytes/intermediates that should not be treated as "catalyst drift".
_ALLOWED_NON_CATALYST_ELEMENTS: set[str] = {"H", "O", "C", "N", "S", "P", "F", "Cl", "Br", "I", "K", "Na", "Li"}


def _infer_task_constraints(
    text: str,
    fallback_components: Optional[List[str]] = None,
    fallback_reaction: Optional[str] = None,
) -> Tuple[List[str], Optional[str]]:
    """
    Infer (components, reaction_type) from the current call context.

    We parse common coordinator/user prompt patterns like:
      - "Target reaction: OER"
      - "Metal catalyst elements: Pt, Cu, Ni, Fe, Co"
      - "Components: Pt, Cu, Ni, Fe, Co"
    """
    raw = str(text or "")

    reaction = (fallback_reaction or "").strip() or None
    for pat in [r"Target reaction:\s*([A-Za-z0-9_+-]+)", r"Reaction Type:\s*([A-Za-z0-9_+-]+)", r"Reaction:\s*([A-Za-z0-9_+-]+)"]:
        m = re.search(pat, raw, flags=re.IGNORECASE)
        if m:
            reaction = str(m.group(1) or "").strip()
            break
    if reaction:
        reaction = reaction.strip()

    comps: List[str] = []
    # Prefer explicit fallback components (passed by the coordinator).
    if fallback_components:
        for c in fallback_components:
            s = str(c).strip()
            if s:
                comps.append(s)

    if not comps:
        for pat in [r"Metal catalyst elements:\s*([^\n\r]+)", r"Components:\s*([^\n\r]+)"]:
            m = re.search(pat, raw, flags=re.IGNORECASE)
            if not m:
                continue
            chunk = str(m.group(1) or "")
            # Split on commas/whitespace; keep plausible element-like tokens.
            tokens = [t.strip() for t in re.split(r"[,\s]+", chunk) if t and t.strip()]
            for t in tokens:
                if re.fullmatch(r"[A-Z][a-z]?", t):
                    comps.append(t)
            if comps:
                break

    # Deduplicate while preserving order.
    seen: set[str] = set()
    out: List[str] = []
    for c in comps:
        if c not in seen:
            seen.add(c)
            out.append(c)

    return out, reaction


def _extract_element_symbols(text: str) -> set[str]:
    """
    Best-effort extraction of element symbols from chemistry-like strings.

    We intentionally avoid single-token extraction to reduce false positives (e.g., English "In").
    """
    s = str(text or "")
    out: set[str] = set()

    # 1) Compact formula blocks: CoCrFeMnNi, NiFeOOH, (CoCrFeMnNi)3O4, etc.
    for m in re.finditer(r"(?<![a-z])(?:[A-Z][a-z]?\d*){2,}", s):
        block = m.group(0)
        syms = [x for x in re.findall(r"[A-Z][a-z]?", block) if x in _VALID_ELEMENT_SYMBOLS]
        if len(syms) >= 2:
            out.update(syms)

    # 2) Hyphen/slash separated lists: Pt-Cu-Ni-Fe-Co, NiFe/CoFe, etc.
    for m in re.finditer(r"(?:[A-Z][a-z]?\s*[-/]\s*)+[A-Z][a-z]?", s):
        block = m.group(0)
        syms = [x for x in re.findall(r"[A-Z][a-z]?", block) if x in _VALID_ELEMENT_SYMBOLS]
        if len(syms) >= 2:
            out.update(syms)

    # 3) Comma-separated lists: Pt, Cu, Ni, Fe, Co
    for m in re.finditer(r"(?:[A-Z][a-z]?\s*,\s*)+[A-Z][a-z]?", s):
        block = m.group(0)
        syms = [x for x in re.findall(r"[A-Z][a-z]?", block) if x in _VALID_ELEMENT_SYMBOLS]
        if len(syms) >= 2:
            out.update(syms)

    return out


def _text_mentions_reaction(text: str, reaction_type: str) -> bool:
    rt = str(reaction_type or "").strip().upper()
    if not rt:
        return False
    t = str(text or "").lower()

    kw_map = {
        "OER": ["oer", "oxygen evolution", "oxygen-evolution"],
        "HER": ["her", "hydrogen evolution"],
        "HOR": ["hor", "hydrogen oxidation"],
        "ORR": ["orr", "oxygen reduction"],
        "UOR": ["uor", "urea oxidation"],
        "EOR": ["eor", "ethanol oxidation"],
        "HZOR": ["hzor", "hydrazine oxidation"],
        "CO2RR": ["co2rr", "co2 reduction", "carbon dioxide reduction"],
    }
    for kw in kw_map.get(rt, [rt.lower()]):
        if kw and kw in t:
            return True
    return False


def _validate_conclusion_against_task(conclusion: str, required_components: List[str]) -> Tuple[bool, str]:
    """
    Guardrail for PROPOSE phase: keep the conclusion on the requested metal elements.
    """
    c = (conclusion or "").strip()
    if not c:
        return False, "empty conclusion"

    required = [str(x).strip() for x in (required_components or []) if str(x).strip()]
    required_set = {x for x in required}
    if not required_set:
        return True, ""

    detected = _extract_element_symbols(c)
    mentioned = set(detected)
    # Also count standalone mentions (e.g., "Pt-based") as coverage.
    for sym in required:
        try:
            if re.search(rf"\\b{re.escape(sym)}\\b", c):
                mentioned.add(sym)
        except re.error:
            continue

    missing = sorted(list(required_set.difference(mentioned)))
    if missing:
        return False, "missing required catalyst metal(s): " + ", ".join(missing)

    forbidden = sorted(list(detected.difference(required_set.union(_ALLOWED_NON_CATALYST_ELEMENTS))))
    if forbidden:
        return False, "mentions forbidden element(s) not in task: " + ", ".join(forbidden)

    return True, ""


def _extract_source_ids_from_text(text: str) -> List[str]:
    """
    Extract canonical rag:chroma/... source_id strings from arbitrary text.
    """
    s = str(text or "")
    if not s:
        return []
    # Keep it permissive; downstream validation checks canonical formatting separately.
    found = re.findall(r"rag:chroma/[^\s\]\)\},;]+", s)
    out: List[str] = []
    seen: set[str] = set()
    for sid in found:
        sid = str(sid).strip()
        if sid and sid not in seen:
            seen.add(sid)
            out.append(sid)
    return out


def _validate_conclusion_against_task_with_evidence(
    conclusion: str,
    required_components: List[str],
    trajectory: Optional["ReActTrajectory"],
) -> Tuple[bool, str]:
    """
    Stronger PROPOSE guard: in addition to element checks on the conclusion text,
    reject conclusions that cite source_id chunks whose retrieved text clearly contains
    forbidden (off-task) catalyst metals.
    """
    ok, reason = _validate_conclusion_against_task(conclusion, required_components)
    if not ok:
        return ok, reason

    cited = _extract_source_ids_from_text(conclusion)
    if not cited or trajectory is None:
        return True, ""

    by_sid: Dict[str, Dict[str, Any]] = {}
    try:
        for step in getattr(trajectory, "steps", []) or []:
            for call in getattr(step, "tool_calls", []) or []:
                if getattr(call, "tool_name", "") != ActionType.SEARCH_RAG.value:
                    continue
                data = getattr(call, "observation_data", None) or []
                if not isinstance(data, list):
                    continue
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    sid = item.get("source_id")
                    if sid:
                        by_sid[str(sid)] = item
    except Exception:
        by_sid = {}

    offenders: List[str] = []
    for sid in cited:
        meta = by_sid.get(sid)
        if not isinstance(meta, dict):
            continue
        forbidden = meta.get("forbidden_elements") or []
        if forbidden:
            offenders.append(
                f"{sid} (forbidden: {', '.join([str(x) for x in forbidden[:6]])}{'...' if len(forbidden) > 6 else ''})"
            )

    if offenders:
        return (
            False,
            "cited evidence appears off-task (forbidden catalyst metals in retrieved chunk): "
            + "; ".join(offenders[:3]),
        )

    return True, ""


def _fallback_thought(
    step_number: int,
    task_reaction: Optional[str],
    task_components: List[str],
    trajectory: Optional["ReActTrajectory"],
) -> str:
    """
    When a model returns empty THOUGHT content (common on some Gemini routes), generate a short,
    task-grounded plan so logs/trajectories are still interpretable.
    """
    rt = (task_reaction or "").strip() or "the target reaction"
    elems = ", ".join([str(c) for c in (task_components or [])]) if task_components else "the provided elements"

    last_action = None
    try:
        if trajectory and getattr(trajectory, "steps", None):
            last = (trajectory.steps or [])[-1]
            last_action = getattr(last, "action_name", None) or getattr(last, "action", None)
    except Exception:
        last_action = None

    if step_number <= 0:
        return f"Plan: gather evidence for {rt} on {elems}, then conclude with grounded metrics and source_id."

    if last_action in {"search_rag", "search_experience"}:
        return f"Plan: pick the most on-target evidence (must match {elems}) and conclude."
    if last_action == "analyze":
        return f"Plan: conclude concisely for {rt} on {elems} with cited source_id."
    return f"Plan: continue evidence gathering for {rt} on {elems}, then conclude."


def _collect_retrieved_source_ids_from_trajectory(trajectory: Optional[ReActTrajectory]) -> set[str]:
    if trajectory is None:
        return set()
    sids: set[str] = set()
    for step in getattr(trajectory, "steps", []) or []:
        for call in getattr(step, "tool_calls", []) or []:
            if getattr(call, "tool_name", "") != ActionType.SEARCH_RAG.value:
                continue
            data = getattr(call, "observation_data", None) or []
            for item in data:
                sid = item.get("source_id")
                if sid:
                    sids.add(sid)
    # Backward-compatible fallback (legacy single-tool steps).
    for step in getattr(trajectory, "steps", []) or []:
        if getattr(step, "action_name", "") != ActionType.SEARCH_RAG.value:
            continue
        data = getattr(step, "observation_data", None) or []
        for item in data:
            sid = item.get("source_id")
            if sid:
                sids.add(sid)
    return sids


def _format_rag_observation(results: List[Dict[str, Any]]) -> str:
    if not results:
        return "No relevant literature knowledge found."

    lines: List[str] = [f"Found {len(results)} relevant documents:"]
    for i, r in enumerate(results[:5], 1):
        score = r.get("score")
        source_id = r.get("source_id")
        meta = r.get("metadata") or {}
        if not source_id:
            doc_id = meta.get("doc_id")
            chunk_index = meta.get("chunk_index")
            if chunk_index is None:
                cid = meta.get("chunk_id")
                try:
                    chunk_index = int(cid)
                except Exception:
                    chunk_index = None
            if doc_id is not None and chunk_index is not None:
                source_id = f"doi:{doc_id}#chunk:{chunk_index}"
        text = (r.get("text") or "").strip()
        if len(text) > 800:
            text = text[:800] + "...(truncated)"
        head = f"{i}."
        if score is not None:
            try:
                head += f" [Relevance: {float(score):.3f}]"
            except Exception:
                pass
        if source_id:
            head += f" [Source: {source_id}]"
        # Optional task-alignment annotations (added by _tool_search_rag).
        required = r.get("required_elements") or []
        match_n = r.get("element_match_count")
        if required and match_n is not None:
            try:
                head += f" [ElemMatch: {int(match_n)}/{len(required)}]"
            except Exception:
                pass
        forbidden = r.get("forbidden_elements") or []
        if forbidden:
            shown = ", ".join([str(x) for x in forbidden[:6]])
            head += f" [Forbidden: {shown}{'...' if len(forbidden) > 6 else ''}]"
        detected = r.get("detected_elements") or []
        if detected:
            shown = ", ".join([str(x) for x in detected[:10]])
            head += f" [Elements: {shown}{'...' if len(detected) > 10 else ''}]"
        lines.append(head)
        lines.append(text)
    return "\n".join(lines)


def _format_experience_observation(experiences: List[Dict[str, Any]]) -> str:
    if not experiences:
        return "No relevant historical experiences found."
    lines: List[str] = [f"Found {len(experiences)} relevant experiences:"]
    for i, exp in enumerate(experiences[:5], 1):
        kind = (exp.get("kind") or "experience").strip()
        gid = (exp.get("guideline_id") or "").strip()
        comps = exp.get("components") or []
        reaction = exp.get("reaction_type") or "unknown"
        pack = exp.get("source_pack") or ""
        sim = exp.get("similarity")

        header_parts = [f"{i}.", f"[{kind}]"]
        if gid:
            header_parts.append(f"[{gid}]")
        if sim is not None:
            try:
                header_parts.append(f"[Similarity: {float(sim):.3f}]")
            except Exception:
                pass
        if comps:
            header_parts.append(f"Components: {', '.join(comps)}")
        header_parts.append(f"Reaction: {reaction}")
        if pack:
            header_parts.append(f"Pack: {pack}")
        lines.append(" ".join(header_parts))

        title = (exp.get("title") or exp.get("products") or exp.get("id") or "").strip()
        if title:
            lines.append(f"Title: {title}")

        perf = exp.get("performance")
        if perf:
            lines.append(f"Performance: {str(perf).strip()}")

        notes = exp.get("content") or exp.get("reasoning") or exp.get("key_arguments")
        if notes and notes != perf:
            snippet = str(notes).strip()
            if len(snippet) > 600:
                snippet = snippet[:600] + "...(truncated)"
            lines.append(f"Notes: {snippet}")
    return "\n".join(lines)


def _extract_sources(trajectory: ReActTrajectory) -> List[Dict[str, Any]]:
    sources: List[Dict[str, Any]] = []
    for step in trajectory.steps:
        # New: multi-tool step support.
        for call in getattr(step, "tool_calls", []) or []:
            if call.tool_name == ActionType.SEARCH_RAG.value and isinstance(call.observation_data, list):
                sources.extend(call.observation_data[:3])
            if call.tool_name == ActionType.SEARCH_EXPERIENCE.value and isinstance(call.observation_data, list):
                sources.extend(call.observation_data[:2])

        # Backward-compatible fallback (legacy single-tool steps).
        action_name = getattr(step, "action_name", "")
        if action_name == ActionType.SEARCH_RAG.value and isinstance(getattr(step, "observation_data", None), list):
            sources.extend(getattr(step, "observation_data")[:3])
        if action_name == ActionType.SEARCH_EXPERIENCE.value and isinstance(getattr(step, "observation_data", None), list):
            sources.extend(getattr(step, "observation_data")[:2])
    return sources
