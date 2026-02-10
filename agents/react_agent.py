"""
LangChain-based ReAct Agent.

Goal:
- Keep explicit "Thought -> Action -> Observation" trajectories.
- Avoid manual OpenAI tool message formatting by using LangChain message classes.
- Keep the original 4 tools: search_literature, search_experience, analyze, conclude.

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
from utils.source_id import build_chroma_source_id, normalize_doc_id, parse_chroma_source_id
from utils.request_limiter import get_global_limiter


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

    def retrieve_knowledge(
        self, query: str, top_k: int = 5, where: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        if self.rag_system is None:
            raise RuntimeError("RAG System is not configured.")
        try:
            # Some adapters accept `top_k` and/or `where`; others don't. Try best-effort combos.
            try:
                results = self.rag_system.retrieve(query, top_k=int(top_k), where=where)
            except TypeError:
                try:
                    results = self.rag_system.retrieve(query, top_k=int(top_k))
                except TypeError:
                    try:
                        results = self.rag_system.retrieve(query, where=where)
                    except TypeError:
                        results = self.rag_system.retrieve(query)
            return (results or [])[: int(top_k)]
        except Exception as e:
            return [{"text": "", "score": 0.0, "metadata": {}, "error": str(e)}]

    def retrieve_experience(
        self, components: List[str], top_k: int = 5, query_text: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        if self.experience_store is None:
            return []
        try:
            # Backward-compatible: some stores/mocks may not accept `query_text`.
            try:
                return (
                    self.experience_store.query_experiences(
                        components=components, top_k=int(top_k), query_text=query_text
                    )
                    or []
                )
            except TypeError:
                return self.experience_store.query_experiences(components=components, top_k=int(top_k)) or []
        except Exception as e:
            return [{"error": str(e), "components": components}]

    # -------------------------
    # LangChain tool wrappers
    # -------------------------

    def _tool_search_literature(self, query: str, top_k: int = 5) -> ToolResult:
        # Clamp `top_k` in STRICT JSON phases (REVIEW/REBUTTAL) to reduce noisy retrieval.
        try:
            top_k_int = int(top_k)
        except Exception:
            top_k_int = 5
        top_k_int = max(1, top_k_int)

        # NOTE: PROPOSE may also be STRICT JSON, but we keep full retrieval capacity in PROPOSE.
        if bool(getattr(self, "_current_requires_strict_json", False)) and not bool(
            getattr(self, "_current_is_propose_phase", False)
        ):
            try:
                strict_cap = int(self.model_config.get("rag_top_k_strict_json", 3))
            except Exception:
                strict_cap = 3
            strict_cap = max(1, strict_cap)
            top_k_int = min(top_k_int, strict_cap)

        top_k = top_k_int

        # Overfetch then rerank by task relevance (elements + reaction type) to reduce "HEA keyword drift"
        # (e.g., being pulled toward popular but off-target systems like CoCrFeMnNi for HEA queries).
        try:
            overfetch = int(self.model_config.get("rag_overfetch", 3))
        except Exception:
            overfetch = 3
        overfetch = max(1, min(10, overfetch))
        fetch_k = max(int(top_k), int(top_k) * overfetch)

        # Infer task constraints from call context (current trajectory query).
        ctx_text = ""
        try:
            ctx_text = str(getattr(getattr(self, "current_trajectory", None), "query", "") or "")
        except Exception:
            ctx_text = ""

        task_components, task_reaction = _infer_task_constraints(
            ctx_text, fallback_components=None, fallback_reaction=None
        )
        
        # Only recall task_reaction relevant chunks if specified.
        rag_filter_by_reaction_type = bool(self.model_config.get("rag_filter_by_reaction_type", True))
        where = None
        if rag_filter_by_reaction_type and task_reaction:
            where = {"reaction_type": str(task_reaction).strip().upper()}

        results = self.retrieve_knowledge(query=query, top_k=fetch_k, where=where)

        # Filter junk chunks (e.g., pure headings) to reduce high-similarity / low-information hits.
        rag_filter_junk_chunks = bool(self.model_config.get("rag_filter_junk_chunks", True))
        try:
            rag_min_chunk_chars = int(self.model_config.get("rag_min_chunk_chars", 80))
        except Exception:
            rag_min_chunk_chars = 80
        rag_min_chunk_chars = max(1, rag_min_chunk_chars)
        rag_keep_if_has_number = bool(self.model_config.get("rag_keep_if_has_number", True))

        raw_results = list(results or [])

        # Hard guarantee: if a target reaction type is known, only return chunks whose metadata.reaction_type matches it.
        # Missing/unknown reaction_type is treated as mismatch and dropped (no cross-type fallback).
        if rag_filter_by_reaction_type and task_reaction:
            target_rt = str(task_reaction).strip().upper()
            filtered: List[Dict[str, Any]] = []
            for r in raw_results:
                meta = r.get("metadata") or {}
                rt_val = meta.get("reaction_type")
                rt = str(rt_val or "").strip().upper()
                if not rt or rt == "UNKNOWN":
                    continue
                if rt != target_rt:
                    continue
                filtered.append(r)
            raw_results = filtered

        if rag_filter_junk_chunks and raw_results:
            kept: List[Dict[str, Any]] = []
            soft_junk: List[Dict[str, Any]] = []
            for r in raw_results:
                text = str(r.get("text") or "")
                kind = _classify_junk_chunk(
                    text, min_chars=rag_min_chunk_chars, keep_if_has_number=rag_keep_if_has_number
                )
                if kind == "hard":
                    # Never backfill hard junk (headings-only / keywords-only / empty).
                    continue
                if kind == "soft":
                    soft_junk.append(r)
                    continue
                kept.append(r)

            # Backfill only from non-hard junk, and allow returning fewer than top_k (prefer precision over volume).
            if len(kept) < int(top_k):
                for r in soft_junk:
                    kept.append(r)
                    if len(kept) >= int(top_k):
                        break
            raw_results = kept

        results = raw_results

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
                # Normalize doc_id to avoid Markdown wrappers like trailing "**" from bold DOIs.
                item["source_id"] = build_chroma_source_id(
                    str(collection),
                    normalize_doc_id(str(doc_id)),
                    int(chunk_index),
                )
            except Exception:
                continue

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

    def _tool_fetch_literature_chunk(self, source_id: str) -> ToolResult:
        """
        Deterministically fetch a specific literature chunk by canonical source_id.

        This bypasses semantic similarity search, which can differ across agents due to
        per-agent collections and embedding models.
        """
        sid = str(source_id or "").strip()
        ref = parse_chroma_source_id(sid)
        if ref is None:
            return ToolResult(observation=f"Invalid source_id (expected rag:chroma/...): {sid}", data=[])

        if self.rag_system is None:
            return ToolResult(observation="RAG System is not configured.", data=[])

        collection_name = str(ref.collection or "").strip()
        doc_id = normalize_doc_id(str(ref.doc_id or ""))
        try:
            chunk_idx = int(ref.chunk_id)
        except Exception:
            chunk_idx = None
        if not collection_name or not doc_id or chunk_idx is None:
            return ToolResult(observation=f"Invalid source_id fields: {sid}", data=[])

        vs = getattr(self.rag_system, "vector_store", None)
        client = getattr(vs, "client", None)
        if client is None:
            return ToolResult(observation="RAG not configured (missing vector_store client).", data=[])

        # Resolve the referenced collection without creating new state.
        col = None
        try:
            if hasattr(client, "get_collection"):
                col = client.get_collection(name=collection_name)
            else:
                # Defensive: avoid get_or_create_collection unless we can confirm existence.
                if hasattr(client, "list_collections"):
                    names: List[str] = []
                    for c in client.list_collections() or []:
                        n = getattr(c, "name", None)
                        if not n and isinstance(c, dict):
                            n = c.get("name")
                        if n:
                            names.append(str(n))
                    if collection_name not in set(names):
                        return ToolResult(observation=f"Collection not found: {collection_name}", data=[])
                if hasattr(client, "get_or_create_collection"):
                    col = client.get_or_create_collection(name=collection_name)
                else:
                    return ToolResult(
                        observation="Chroma client missing get_collection/get_or_create_collection.",
                        data=[],
                    )
        except Exception as e:
            return ToolResult(observation=f"Failed to open collection {collection_name}: {str(e)}", data=[])

        fetched = None
        include = ["documents", "metadatas"]

        # 1) Preferred: metadata filter by (doc_id, chunk_index) — works for DOI and no-doi docs.
        try:
            fetched = col.get(where={"doc_id": doc_id, "chunk_index": int(chunk_idx)}, include=include)
        except TypeError:
            try:
                fetched = col.get(where={"doc_id": doc_id, "chunk_index": int(chunk_idx)}, include=include, limit=1)
            except Exception:
                fetched = None
        except Exception:
            fetched = None

        # 2) Fallback: some stores only have legacy numeric chunk_id metadata.
        if not fetched or not (fetched.get("ids") or []):
            try:
                fetched = col.get(where={"doc_id": doc_id, "chunk_id": int(chunk_idx)}, include=include)
            except TypeError:
                try:
                    fetched = col.get(where={"doc_id": doc_id, "chunk_id": int(chunk_idx)}, include=include, limit=1)
                except Exception:
                    fetched = None
            except Exception:
                fetched = None

        # 3) DOI fast-path: stable chunk UID used as the Chroma id.
        if (not fetched or not (fetched.get("ids") or [])) and re.match(r"(?i)^10\.\d{4,9}/", doc_id):
            try:
                uid = f"{doc_id}#chunk:{int(chunk_idx)}"
                fetched = col.get(ids=[uid], include=include)
            except Exception:
                fetched = fetched

        docs = (fetched or {}).get("documents") or []
        metas = (fetched or {}).get("metadatas") or []
        # Defensive: avoid nested list shapes.
        if docs and isinstance(docs, list) and len(docs) == 1 and isinstance(docs[0], list):
            docs = docs[0]
        if metas and isinstance(metas, list) and len(metas) == 1 and isinstance(metas[0], list):
            metas = metas[0]

        results: List[Dict[str, Any]] = []
        if isinstance(docs, list) and docs:
            meta0 = metas[0] if isinstance(metas, list) and metas else {}
            results.append(
                {
                    "text": str(docs[0] or ""),
                    "score": None,
                    "metadata": meta0 or {},
                    "source_id": build_chroma_source_id(collection_name, doc_id, int(chunk_idx)),
                }
            )

        if not results:
            return ToolResult(observation=f"No chunk found for {sid}", data=[])

        observation = _format_rag_observation(results)
        return ToolResult(observation=observation, data=results)

    def _tool_search_experience(self, components: List[str], top_k: int = 5) -> ToolResult:
        # Provide a query hint so the store can rank global [G*] guidelines by relevance.
        ctx_text = ""
        try:
            ctx_text = str(getattr(getattr(self, "current_trajectory", None), "query", "") or "")
        except Exception:
            ctx_text = ""

        _task_components, task_reaction = _infer_task_constraints(
            ctx_text, fallback_components=components, fallback_reaction=None
        )

        hints: List[str] = ["search evidence synthesize broadening validate unit conversion"]
        if bool(getattr(self, "_current_requires_strict_json", False)):
            hints.append("strict json format key matching cross-validation conclude")

        rt = str(task_reaction or "").strip().upper()
        if rt == "CO2RR":
            hints.append("co2rr faradaic efficiency fe partial current density reduction current sign")
        elif rt in {"OER", "HER", "UOR", "HZOR"}:
            hints.append("overpotential 10 mA/cm^2 potential")
        elif rt == "ORR":
            hints.append("half-wave potential e1/2")
        elif rt == "HOR":
            hints.append("exchange current density j0")
        elif rt == "EOR":
            hints.append("mass activity")
        elif rt == "O5H":
            hints.append("faradaic efficiency fe")

        query_text = ctx_text
        if hints:
            query_text = (ctx_text or "") + "\n\n[experience_search_hints]\n" + " ".join(hints)

        experiences = self.retrieve_experience(components=components, top_k=top_k, query_text=query_text)
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

    def _tool_conclude(self, conclusion: Any) -> ToolResult:
        """
        Submit the final answer.

        Note: In STRICT JSON phases, providers sometimes send the JSON payload as a structured
        object (dict/list) rather than a pre-serialized string. Accept both to avoid
        brittle manual string construction (e.g., unescaped newlines inside JSON strings).
        """
        if isinstance(conclusion, (dict, list)):
            try:
                conclusion = json.dumps(conclusion, ensure_ascii=False)
            except Exception:
                conclusion = str(conclusion)
        else:
            conclusion = str(conclusion)
        return ToolResult(observation=conclusion, data=conclusion)

    def _build_tools(self):
        _ChatOpenAI, _SystemMessage, _HumanMessage, _AIMessage, _ToolMessage, StructuredTool = _lazy_langchain_imports()

        tools = [
            StructuredTool.from_function(
                func=self._tool_search_literature,
                name=ActionType.SEARCH_LITERATURE.value,
                description=(
                    "Search the local literature database (Chroma-backed RAG) and return relevant chunks. "
                    "Use AFTER `search_experience` when you need verifiable citations; cite source_id from results."
                ),
            ),
            StructuredTool.from_function(
                func=self._tool_fetch_literature_chunk,
                name=ActionType.FETCH_LITERATURE_CHUNK.value,
                description=(
                    "Fetch a specific literature chunk deterministically by canonical source_id "
                    "(rag:chroma/<collection>/doi:<doc_id>#chunk:<idx>). Use to verify another agent's cited evidence."
                ),
            ),
            StructuredTool.from_function(
                func=self._tool_search_experience,
                name=ActionType.SEARCH_EXPERIENCE.value,
                description="Search the experience database for similar past cases / guidelines (preferred before `search_literature`).",
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
            "- In a single ACTION step, do NOT mix search tools (`search_literature`, `search_experience`, `fetch_literature_chunk`) with "
            "analysis tools (`analyze`, `conclude`).\n"
            "  - If you need evidence: call one or more search tools first.\n"
            "  - After you receive observations, in the NEXT step you may call `analyze` or `conclude`.\n"
            "Rules:\n"
            "- Tool priority: prefer `search_experience` FIRST; then use `search_literature` for verifiable citations.\n"
            "- To verify another agent's cited evidence, use `fetch_literature_chunk(source_id)`.\n"
            "- When calling `search_literature`, include the target reaction type (e.g., OER/HER/ORR/HOR) AND ALL provided metal element symbols in the query.\n"
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

        def _invoke_llm(llm_obj: Any, msgs: List[Any], kind: str = "llm") -> Any:
            """
            Invoke an LLM call under the global in-process request limiter.

            This caps total in-flight outbound requests across all reactions/agents/threads,
            reducing burst-induced upstream flakiness.
            """

            limiter = get_global_limiter()
            with limiter.slot(kind=str(kind or "llm")):
                return llm_obj.invoke(msgs)

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

        # Many debate phases (REVIEW/REBUTTAL/PROPOSE) may require STRICT JSON.
        requires_strict_json = "STRICT JSON" in (system_prompt or "").upper()
        # Expose per-call STRICT JSON requirement to tools (e.g., top_k clamp in search_literature).
        self._current_requires_strict_json = bool(requires_strict_json)
        # PROPOSE can now be STRICT JSON too; we still want full-fidelity retrieval in PROPOSE.
        self._current_is_propose_phase = bool(_is_propose_phase(system_prompt or ""))
        task_components, task_reaction = _infer_task_constraints(
            full_query, fallback_components=components, fallback_reaction=None
        )
        propose_phase = bool(self._current_is_propose_phase)
        retrieval_budget = _parse_retrieval_budget_from_system_prompt(system_prompt or "")
        retrieval_action_steps_used = 0

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
        use_user_role_for_action = provider_hint in {"google", "gemini"} or ("gemini" in model_name_hint)

        deadline_mode = bool(self.model_config.get("deadline_mode", True))

        propose_metric_line_enforce_point_estimate = bool(
            self.model_config.get("propose_metric_line_enforce_point_estimate", True)
        )
        propose_metric_line_require_confidence = bool(
            self.model_config.get("propose_metric_line_require_confidence", True)
        )

        def _log_strict_json_fallback(schema_detected: str, reason: str, original_preview: str) -> None:
            try:
                self.logger.warning(
                    "react_strict_json_fallback_used",
                    extra={
                        "event": "agent.react.strict_json_fallback_used",
                        "agent_id": self.agent_id,
                        "schema_detected": schema_detected,
                        "phase_hint": schema_detected if schema_detected != "unknown" else "unknown",
                        "reason": reason,
                        "original_preview": (original_preview or "")[:500],
                    },
                )
            except Exception:
                pass

        def _log_strict_json_repair(
            schema_detected: str,
            repair_reason: str,
            original_preview: str,
            repaired_preview: str,
            repaired_ok: bool,
        ) -> None:
            try:
                self.logger.warning(
                    "react_strict_json_repair",
                    extra={
                        "event": "agent.react.strict_json_repair",
                        "agent_id": self.agent_id,
                        "schema_detected": schema_detected,
                        "repair_reason": repair_reason,
                        "repaired_ok": bool(repaired_ok),
                        "original_preview": (original_preview or "")[:500],
                        "repaired_preview": (repaired_preview or "")[:500],
                    },
                )
            except Exception:
                pass

        def _log_strict_json_salvage(
            schema_detected: str,
            salvage_reason: str,
            original_preview: str,
            salvaged_preview: str,
            salvage_used: bool,
        ) -> None:
            try:
                self.logger.warning(
                    "react_strict_json_salvage",
                    extra={
                        "event": "agent.react.strict_json_salvage",
                        "agent_id": self.agent_id,
                        "schema_detected": schema_detected,
                        "salvage_used": bool(salvage_used),
                        "salvage_reason": salvage_reason,
                        "original_preview": (original_preview or "")[:500],
                        "salvaged_preview": (salvaged_preview or "")[:500],
                    },
                )
            except Exception:
                pass

        def _run_forced_conclude(messages_: List[Any]) -> Tuple[str, str, bool, str]:
            """Generate a final answer draft and a tool_call_id via a forced conclude attempt (best-effort)."""
            retrieved_ = _collect_retrieved_source_ids_from_trajectory(trajectory)
            sid_hint_ = ""
            if retrieved_:
                sid_hint_ = (
                    "\nYou MUST cite at least one of these source_id values verbatim in your final answer:\n"
                    + "\n".join(f"- {sid}" for sid in sorted(list(retrieved_))[:10])
                )
            elements_hint_ = ""
            if task_components:
                elements_hint_ = (
                    "\nThe metal catalyst elements for this task are EXACTLY:\n- "
                    + ", ".join([str(c) for c in (task_components or [])])
                    + "\nYou MUST explicitly include ALL of these elements in your final conclusion."
                )

            draft_ = ""
            forced_conclude_tool_call_id_ = "forced_conclude"
            strict_json_fallback_used_ = False
            schema_detected_ = "unknown"

            # Best-effort: force a specific tool if the backend supports it; otherwise fall back to "required".
            force_conclude_llm_ = llm_with_tools_forced
            try:
                if hasattr(llm, "bind_tools"):
                    force_conclude_llm_ = llm.bind_tools(tools, tool_choice=ActionType.CONCLUDE.value)
                else:  # pragma: no cover
                    force_conclude_llm_ = llm.bind(tools=tools, tool_choice=ActionType.CONCLUDE.value)
            except Exception:
                force_conclude_llm_ = llm_with_tools_forced

            forced_action_msg_ = None
            try:
                forced_msg_cls_ = HumanMessage if use_user_role_for_action else SystemMessage
                forced_action_msg_ = _invoke_llm(
                    force_conclude_llm_,
                    messages_
                    + [
                        forced_msg_cls_(
                            content=(
                                (
                                    "FINAL ACTION: You MUST call ONLY the `conclude` tool now.\n"
                                    "Set the `conclusion` argument to STRICT JSON ONLY that follows the schema in the system prompt EXACTLY.\n"
                                    "- No markdown, no extra text.\n"
                                    "- If evidence is required, cite at least one verifiable source_id.\n"
                                    + sid_hint_
                                    + elements_hint_
                                )
                                if requires_strict_json
                                else (
                                    "FINAL ACTION: You MUST call ONLY the `conclude` tool now.\n"
                                    "Set the `conclusion` argument to the best possible final answer.\n"
                                    "- Include the reaction type explicitly.\n"
                                    "- Explicitly restate the catalyst metal elements exactly as provided.\n"
                                    "- Provide a single speculated value + confidence for key performance metric(s) (no numeric range).\n"
                                    "- If you used literature evidence, cite source_id exactly as provided.\n"
                                    + sid_hint_
                                    + elements_hint_
                                )
                            )
                        )
                    ],
                )
            except Exception:
                forced_action_msg_ = None

            if self.verbose and forced_action_msg_ is not None:
                self.logger.debug(
                    "react_forced_conclude_action_raw",
                    extra={
                        "event": "agent.react.forced_conclude.action.raw",
                        "agent_id": self.agent_id,
                        "forced_action_additional_kwargs": _preview(getattr(forced_action_msg_, "additional_kwargs", None)),
                        "forced_action_tool_calls": _preview(getattr(forced_action_msg_, "tool_calls", None)),
                        "forced_action_text": (getattr(forced_action_msg_, "content", "") or "")[:1500],
                    },
                )

            if forced_action_msg_ is not None:
                forced_calls_ = _extract_tool_calls(forced_action_msg_)
                for name, args, call_id in (_normalize_tool_call(c) for c in forced_calls_):
                    if name != ActionType.CONCLUDE.value:
                        continue
                    if call_id:
                        forced_conclude_tool_call_id_ = call_id
                    conclusion_ = None
                    if isinstance(args, dict):
                        conclusion_ = args.get("conclusion") or args.get("final_answer")
                    if isinstance(conclusion_, (dict, list)):
                        try:
                            conclusion_ = json.dumps(conclusion_, ensure_ascii=False)
                        except Exception:
                            conclusion_ = str(conclusion_)
                    if conclusion_ is not None:
                        draft_ = str(conclusion_).strip()
                        break

            if not draft_:
                # Fallback: ask for a final answer in free-form text (no tools).
                forced_ = _invoke_llm(
                    llm,
                    messages_
                    + [
                        (HumanMessage if use_user_role_for_action else SystemMessage)(
                            content=(
                                (
                                    "FINAL PHASE: Output STRICT JSON ONLY.\n"
                                    "- Follow the output schema in the system prompt EXACTLY.\n"
                                    "- No markdown, no extra text.\n"
                                    "- If evidence is required, cite at least one verifiable source_id.\n"
                                    + sid_hint_
                                    + elements_hint_
                                )
                                if requires_strict_json
                                else (
                                    "FINAL PHASE: Write the best possible final answer now.\n"
                                    "- Include the reaction type explicitly.\n"
                                    "- Explicitly restate the catalyst metal elements exactly as provided.\n"
                                    "- Provide a single speculated value + confidence for key performance metric(s) (no numeric range).\n"
                                    "- If you used literature evidence, cite source_id exactly as provided.\n"
                                    + sid_hint_
                                    + elements_hint_
                                )
                            )
                        )
                    ],
                )
                if self.verbose:
                    self.logger.debug(
                        "react_forced_conclude_text_raw",
                        extra={
                            "event": "agent.react.forced_conclude.text.raw",
                            "agent_id": self.agent_id,
                            "forced_text_additional_kwargs": _preview(getattr(forced_, "additional_kwargs", None)),
                            "forced_text_tool_calls": _preview(getattr(forced_, "tool_calls", None)),
                            "forced_text": (getattr(forced_, "content", "") or "")[:1500],
                        },
                    )
                draft_ = (getattr(forced_, "content", "") or "").strip()
                if not draft_:
                    # Some providers return a legacy function_call even when tools are not bound.
                    fc_ = (getattr(forced_, "additional_kwargs", {}) or {}).get("function_call")
                    if isinstance(fc_, dict):
                        args_ = fc_.get("arguments")
                        parsed_ = None
                        if isinstance(args_, str):
                            try:
                                parsed_ = json.loads(args_)
                            except Exception:
                                parsed_ = None
                        elif isinstance(args_, dict):
                            parsed_ = args_
                        if isinstance(parsed_, dict):
                            draft_ = str(parsed_.get("conclusion") or parsed_.get("final_answer") or "").strip()

            if requires_strict_json:
                schema_detected_ = _detect_strict_json_schema(system_prompt)
                if not _is_valid_strict_json_payload(draft_, system_prompt):
                    _orig_ = draft_
                    repaired_, repair_reason_ = _repair_strict_json_text(draft_, system_prompt)
                    if repaired_ and _is_valid_strict_json_payload(repaired_, system_prompt):
                        draft_ = repaired_
                        _log_strict_json_repair(
                            schema_detected=schema_detected_,
                            repair_reason=repair_reason_,
                            original_preview=str(_orig_),
                            repaired_preview=str(repaired_),
                            repaired_ok=True,
                        )
                    else:
                        _log_strict_json_repair(
                            schema_detected=schema_detected_,
                            repair_reason=repair_reason_,
                            original_preview=str(_orig_),
                            repaired_preview=str(repaired_ or ""),
                            repaired_ok=False,
                        )
                        salvaged_, salvage_reason_ = _salvage_invalid_strict_json_payload(
                            _orig_,
                            system_prompt=system_prompt,
                            full_query=full_query,
                            task_reaction=task_reaction,
                            task_components=task_components,
                        )
                        if salvaged_ and _is_valid_strict_json_payload(salvaged_, system_prompt):
                            draft_ = salvaged_
                            _log_strict_json_salvage(
                                schema_detected=schema_detected_,
                                salvage_reason=salvage_reason_,
                                original_preview=str(_orig_),
                                salvaged_preview=str(salvaged_),
                                salvage_used=True,
                            )
                        else:
                            strict_json_fallback_used_ = True
                            draft_ = _minimal_strict_json_payload(system_prompt)
                            _log_strict_json_fallback(
                                schema_detected=schema_detected_,
                                reason="forced_conclude_invalid_or_empty",
                                original_preview=str(_orig_),
                            )

            if not draft_:
                draft_ = "No conclusion generated."

            # If strict JSON is not required and the model forgot to include a verifiable source_id,
            # attach a minimal evidence line.
            if (not requires_strict_json) and retrieved_ and not any(sid in draft_ for sid in retrieved_):
                draft_ = draft_.rstrip() + "\n\nEvidence (retrieved source_id): " + ", ".join(sorted(list(retrieved_))[:3])

            # If a PROPOSE conclusion is missing required task elements, patch a minimal explicit line.
            # This prevents the agent from getting stuck on the final step due to guard enforcement.
            if (not requires_strict_json) and task_components:
                ok_, reason_ = _validate_conclusion_against_task(draft_, task_components)
                if not ok_ and "missing required catalyst" in str(reason_).lower():
                    draft_ = (
                        draft_.rstrip()
                        + "\n\nCatalyst metal elements (exactly as provided): "
                        + ", ".join([str(c) for c in (task_components or [])])
                    )

            return draft_, forced_conclude_tool_call_id_, strict_json_fallback_used_, schema_detected_

        while step_number < effective_max_steps:
            remaining_steps = effective_max_steps - step_number
            if deadline_mode and remaining_steps == 1:
                # Deadline mode: force an in-loop conclude so we don't fall into the post-loop forced_conclude path.
                self.logger.warning(
                    "react_deadline_force_conclude",
                    extra={
                        "event": "agent.react.deadline_force_conclude",
                        "agent_id": self.agent_id,
                        "steps_so_far": step_number,
                        "max_react_steps": effective_max_steps,
                    },
                )
                draft, forced_tool_call_id, strict_json_fallback_used, schema_detected = _run_forced_conclude(messages)
                final_answer = draft

                if requires_strict_json and not _is_valid_strict_json_payload(final_answer, system_prompt):
                    schema_detected = _detect_strict_json_schema(system_prompt)
                    _orig = final_answer
                    repaired, repair_reason = _repair_strict_json_text(final_answer, system_prompt)
                    if repaired and _is_valid_strict_json_payload(repaired, system_prompt):
                        final_answer = repaired
                        _log_strict_json_repair(
                            schema_detected=schema_detected,
                            repair_reason=repair_reason,
                            original_preview=str(_orig),
                            repaired_preview=str(repaired),
                            repaired_ok=True,
                        )
                    else:
                        _log_strict_json_repair(
                            schema_detected=schema_detected,
                            repair_reason=repair_reason,
                            original_preview=str(_orig),
                            repaired_preview=str(repaired or ""),
                            repaired_ok=False,
                        )
                        salvaged, salvage_reason = _salvage_invalid_strict_json_payload(
                            _orig,
                            system_prompt=system_prompt,
                            full_query=full_query,
                            task_reaction=task_reaction,
                            task_components=task_components,
                        )
                        if salvaged and _is_valid_strict_json_payload(salvaged, system_prompt):
                            final_answer = salvaged
                            _log_strict_json_salvage(
                                schema_detected=schema_detected,
                                salvage_reason=salvage_reason,
                                original_preview=str(_orig),
                                salvaged_preview=str(salvaged),
                                salvage_used=True,
                            )
                        else:
                            strict_json_fallback_used = True
                            final_answer = _minimal_strict_json_payload(system_prompt)
                            _log_strict_json_fallback(
                                schema_detected=schema_detected,
                                reason="deadline_force_conclude_invalid_or_empty",
                                original_preview=str(_orig),
                            )

                # PROPOSE-only: ensure the final answer respects the unified output contract even in deadline mode.
                if propose_phase and (not requires_strict_json):
                    before = final_answer
                    missing = _diagnose_propose_contract_missing_lines(before)
                    final_answer = _coerce_propose_conclusion_to_contract(
                        draft=before,
                        full_query=full_query,
                        task_reaction=task_reaction,
                        task_components=task_components,
                        trajectory=trajectory,
                    )
                    if final_answer != before:
                        try:
                            self.logger.warning(
                                "react_propose_contract_rewrite",
                                extra={
                                    "event": "agent.react.propose.contract_rewrite",
                                    "agent_id": self.agent_id,
                                    "step": step_number + 1,
                                    "missing_lines": missing,
                                    "original_preview": (before or "")[:500],
                                    "rewritten_preview": (final_answer or "")[:500],
                                },
                            )
                        except Exception:
                            pass

                step_number += 1
                tool_call = ToolCallRecord(
                    tool_name=ActionType.CONCLUDE.value,
                    tool_call_id=forced_tool_call_id or "deadline_conclude",
                    tool_args={"conclusion": final_answer},
                    observation=final_answer,
                    observation_data=final_answer,
                )
                thought_text = "Deadline mode: forced conclude on the final step to avoid budget overrun."
                if strict_json_fallback_used:
                    thought_text += (
                        " Strict JSON fallback: emitted minimal schema JSON due to empty/invalid conclude."
                    )
                trajectory.add_step(
                    ReActStep(
                        step_number=step_number,
                        thought=thought_text,
                        action=ActionType.CONCLUDE.value,
                        action_input={"conclusion": final_answer},
                        observation=final_answer,
                        tool_call_id=forced_tool_call_id or "deadline_conclude",
                        observation_data=final_answer,
                        tool_calls=[tool_call],
                    )
                )
                break

            # ----- THOUGHT -----

            thought_instruction = self._get_thought_phase_instruction()
            thought_prompt_msg = SystemMessage(content=thought_instruction)
            if use_user_role_for_thought:
                # Some Gemini/OpenAI-compatible routes may return empty content for "thought" when sent as a SystemMessage.
                # Using a HumanMessage tends to be more reliable while still keeping the THOUGHT out of the chat history.
                thought_prompt_msg = HumanMessage(content=thought_instruction)
            
            # Get thinking content
            thought_msg = _invoke_llm(llm, messages + [thought_prompt_msg])
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
                # Some Gemini/OpenAI-compatible routes may (incorrectly) attach tool_calls even when tools are disabled.
                # If that happens, prefer a short, informative fallback thought instead of an empty string.
                tc = getattr(thought_msg, "tool_calls", None)
                if isinstance(tc, list) and tc:
                    names: List[str] = []
                    for c in tc:
                        name = None
                        if isinstance(c, dict):
                            name = c.get("name") or (c.get("function") or {}).get("name")
                        else:
                            name = getattr(c, "name", None)
                            if name is None:
                                fn = getattr(c, "function", None)
                                name = getattr(fn, "name", None) if fn is not None else None
                        if name:
                            s = str(name).strip()
                            if s and s not in names:
                                names.append(s)
                    if names:
                        thought_content = (
                            "Plan: call tools ("
                            + ", ".join(names[:3])
                            + (", ..." if len(names) > 3 else "")
                            + "), then conclude."
                        )
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
            deadline_hint = ""
            if deadline_mode and remaining_steps == 2:
                deadline_hint = (
                    "\nDEADLINE MODE: You have only 2 steps left.\n"
                    "- Do NOT call `search_literature` or `search_experience`.\n"
                    "- You may ONLY call `analyze` or `conclude`.\n"
                )
            for attempt in range(2):
                retry_hint = ""
                if attempt > 0:
                    retry_hint = (
                        "\nERROR: You did not call any tools in the previous ACTION attempt.\n"
                        "You MUST call at least one tool now (no free-form answers).\n"
                    )

                action_llm = llm_with_tools
                # Second attempt should always force a tool call (helps flaky tool-calling providers).
                if attempt == 1:
                    action_llm = llm_with_tools_forced
                elif no_tool_call_streak >= no_tool_call_threshold:
                    action_llm = llm_with_tools_forced

                # Get action content         
                action_msg = _invoke_llm(
                    action_llm,
                    messages
                    + [
                        (
                            HumanMessage(
                                content=self._get_action_phase_instruction()
                                + retry_hint
                                + deadline_hint
                            )
                            if use_user_role_for_action
                            else SystemMessage(
                                content=self._get_action_phase_instruction()
                                + retry_hint
                                + deadline_hint
                            )
                        ),
                        (
                            HumanMessage(content=f"THOUGHT (plan; do not repeat):\n{thought_content}")
                            if use_user_role_for_action
                            else SystemMessage(content=f"THOUGHT (plan; do not repeat):\n{thought_content}")
                        ),
                    ],
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
                            "action_tool_calls": _preview(getattr(action_msg, "tool_calls", None)),
                            "raw_action_text": raw_action_text[:1500] + ("...(truncated)" if len(raw_action_text) > 1500 else ""),
                        },
                    )
                if tool_calls:
                    break

            if not tool_calls:
                # Some providers (notably Gemini via OpenAI-compatible routes) may output STRICT JSON directly as
                # normal assistant content while failing to emit tool_calls. If we can repair/validate it, accept
                # it as a synthetic conclude to avoid repeated "no_tool_call" loops.
                if requires_strict_json and raw_action_text and _extract_first_json_object(raw_action_text) is not None:
                    schema_detected = _detect_strict_json_schema(system_prompt)
                    candidate, repair_reason = _repair_strict_json_text(raw_action_text, system_prompt)
                    candidate_reason = f"repair:{repair_reason}"
                    if not (candidate and _is_valid_strict_json_payload(candidate, system_prompt)):
                        salvaged, salvage_reason = _salvage_invalid_strict_json_payload(
                            raw_action_text,
                            system_prompt=system_prompt,
                            full_query=full_query,
                            task_reaction=task_reaction,
                            task_components=task_components,
                        )
                        if salvaged and _is_valid_strict_json_payload(salvaged, system_prompt):
                            candidate = salvaged
                            candidate_reason = f"salvage:{salvage_reason}"

                    if candidate and _is_valid_strict_json_payload(candidate, system_prompt):
                        try:
                            self.logger.warning(
                                "react_action_json_content_accepted",
                                extra={
                                    "event": "agent.react.action.json_content_accepted",
                                    "agent_id": self.agent_id,
                                    "schema_detected": schema_detected,
                                    "reason": candidate_reason,
                                    "raw_preview": (raw_action_text or "")[:500],
                                    "accepted_preview": (candidate or "")[:500],
                                },
                            )
                        except Exception:
                            pass

                        final_answer = candidate
                        step_number += 1
                        tool_call = ToolCallRecord(
                            tool_name=ActionType.CONCLUDE.value,
                            tool_call_id="direct_json_content",
                            tool_args={"conclusion": candidate},
                            observation=candidate,
                            observation_data=candidate,
                        )
                        trajectory.add_step(
                            ReActStep(
                                step_number=step_number,
                                thought=(
                                    "ACTION produced STRICT JSON in content without tool calls; "
                                    "accepted as final answer."
                                ),
                                action=ActionType.CONCLUDE.value,
                                action_input={"conclusion": candidate},
                                observation=candidate,
                                tool_call_id="direct_json_content",
                                observation_data=candidate,
                                tool_calls=[tool_call],
                            )
                        )
                        break

                no_tool_call_streak += 1

                # Feed back a short, explicit failure note to the model to break "I will call tool..." loops.
                # Keep it concise to avoid polluting context.
                failure_note = (
                    "ACTION FAILURE: You did not emit any tool calls.\n"
                    "Next ACTION MUST emit at least one tool call via the tool-calling mechanism (no plain text).\n"
                    "If you need evidence, call `search_experience` and/or `search_literature`.\n"
                )
                # Avoid spamming the same failure note repeatedly.
                try:
                    last_content = getattr(messages[-1], "content", "") if messages else ""
                except Exception:
                    last_content = ""
                if not (isinstance(last_content, str) and last_content.startswith("ACTION FAILURE:")):
                    if use_user_role_for_action:
                        messages.append(HumanMessage(content=failure_note))
                    else:
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
            search_tools = {
                ActionType.SEARCH_LITERATURE.value,
                ActionType.SEARCH_EXPERIENCE.value,
                ActionType.FETCH_LITERATURE_CHUNK.value,
            }
            analysis_tools = {ActionType.ANALYZE.value, ActionType.CONCLUDE.value}

            retrieval_budget_exhausted = (
                retrieval_budget is not None and retrieval_action_steps_used >= int(retrieval_budget)
            )
            deadline_no_retrieval = bool(deadline_mode and remaining_steps == 2)
            block_search_this_step = bool(retrieval_budget_exhausted or deadline_no_retrieval)

            has_search = any(name in search_tools for name, _args, _id in normalized_calls)
            has_search_unblocked = bool(has_search and (not block_search_this_step))
            has_analysis = any(name in analysis_tools for name, _args, _id in normalized_calls)
            # If retrieval is blocked by policy, do NOT treat this step as mixed (allow analyze/conclude to run).
            mixed_search_and_analysis = has_search_unblocked and has_analysis
            mixed_error = (
                "Policy violation: mixed search and analysis in one ACTION step. "
                "Call only search tools first; after receiving observations, call `analyze`/`conclude` in the next step."
            )

            tool_call_records: List[ToolCallRecord] = []
            observation_sections: List[str] = []
            retrieval_executed_this_step = False

            for tool_name, tool_args, tool_call_id in normalized_calls:

                # If the model tried to both search and analyze/conclude in the same ACTION step,
                # refuse the analysis/conclude calls (they wouldn't be grounded in the fresh observations).
                blocked_mixed = mixed_search_and_analysis and tool_name in analysis_tools
                blocked_retrieval = tool_name in search_tools and block_search_this_step

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
                    ok, reason = _validate_conclusion_against_task_with_evidence(
                        conclusion_text,
                        task_components,
                        trajectory,
                        enforce_point_estimate=propose_metric_line_enforce_point_estimate,
                        require_confidence=propose_metric_line_require_confidence,
                    )
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
                            "You MUST revise your conclusion to explicitly include ALL required catalyst metal elements "
                            "(exactly as provided): "
                            + ", ".join([str(c) for c in (task_components or [])])
                            + "."
                        )
                    elif reason:
                        # Evidence-level warnings should not block a PROPOSE conclusion.
                        # We still log them for debugging/analytics.
                        try:
                            self.logger.warning(
                                "react_conclude_guard_warning",
                                extra={
                                    "event": "agent.react.conclude.guard_warning",
                                    "agent_id": self.agent_id,
                                    "step": step_number + 1,
                                    "warning": reason,
                                    "required_components": task_components,
                                    "cited_source_ids": _extract_source_ids_from_text(conclusion_text)[:10],
                                },
                            )
                        except Exception:
                            pass

                blocked = blocked_mixed or blocked_guard or blocked_retrieval
                if blocked:
                    if blocked_mixed:
                        result = ToolResult(observation=mixed_error, data={"error": "mixed_search_and_analysis"})
                    elif blocked_retrieval:
                        if deadline_no_retrieval:
                            obs = (
                                "Policy: retrieval is disabled when only 2 steps remain in this call.\n"
                                "Do NOT call search tools now; call `analyze` or `conclude`."
                            )
                            result = ToolResult(observation=obs, data={"error": "deadline_no_retrieval"})
                        else:
                            obs = (
                                "Policy: retrieval budget exceeded for this phase.\n"
                                "Do NOT call search tools now; call `analyze` or `conclude`."
                            )
                            result = ToolResult(observation=obs, data={"error": "retrieval_budget_exceeded"})
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
                if tool_name == ActionType.CONCLUDE.value and requires_strict_json and not blocked:
                    if not _is_valid_strict_json_payload(observation_text, system_prompt):
                        schema_detected = _detect_strict_json_schema(system_prompt)
                        repaired, repair_reason = _repair_strict_json_text(observation_text, system_prompt)
                        if repaired and _is_valid_strict_json_payload(repaired, system_prompt):
                            _log_strict_json_repair(
                                schema_detected=schema_detected,
                                repair_reason=repair_reason,
                                original_preview=str(observation_text),
                                repaired_preview=str(repaired),
                                repaired_ok=True,
                            )
                            observation_text = repaired
                            result = ToolResult(observation=observation_text, data=observation_text)
                        else:
                            _log_strict_json_repair(
                                schema_detected=schema_detected,
                                repair_reason=repair_reason,
                                original_preview=str(observation_text),
                                repaired_preview=str(repaired or ""),
                                repaired_ok=False,
                            )
                            salvaged, salvage_reason = _salvage_invalid_strict_json_payload(
                                observation_text,
                                system_prompt=system_prompt,
                                full_query=full_query,
                                task_reaction=task_reaction,
                                task_components=task_components,
                            )
                            if salvaged and _is_valid_strict_json_payload(salvaged, system_prompt):
                                _log_strict_json_salvage(
                                    schema_detected=schema_detected,
                                    salvage_reason=salvage_reason,
                                    original_preview=str(observation_text),
                                    salvaged_preview=str(salvaged),
                                    salvage_used=True,
                                )
                                observation_text = salvaged
                                result = ToolResult(observation=observation_text, data=observation_text)
                            else:
                                _log_strict_json_fallback(
                                    schema_detected=schema_detected,
                                    reason="conclude_tool_output_invalid_or_empty",
                                    original_preview=str(observation_text),
                                )
                                observation_text = _minimal_strict_json_payload(system_prompt)
                                result = ToolResult(observation=observation_text, data=observation_text)

                # PROPOSE-only: coerce conclude output into the unified output contract (no extra LLM calls).
                if tool_name == ActionType.CONCLUDE.value and (not requires_strict_json) and propose_phase and not blocked:
                    before = observation_text
                    missing = _diagnose_propose_contract_missing_lines(before)
                    observation_text = _coerce_propose_conclusion_to_contract(
                        draft=before,
                        full_query=full_query,
                        task_reaction=task_reaction,
                        task_components=task_components,
                        trajectory=trajectory,
                    )
                    if observation_text != before:
                        try:
                            self.logger.warning(
                                "react_propose_contract_rewrite",
                                extra={
                                    "event": "agent.react.propose.contract_rewrite",
                                    "agent_id": self.agent_id,
                                    "step": step_number + 1,
                                    "missing_lines": missing,
                                    "original_preview": (before or "")[:500],
                                    "rewritten_preview": (observation_text or "")[:500],
                                },
                            )
                        except Exception:
                            pass
                        result = ToolResult(observation=observation_text, data=observation_text)
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

                if self.verbose and tool_name in {ActionType.SEARCH_LITERATURE.value, ActionType.SEARCH_EXPERIENCE.value}:
                    n_items = len(result.data) if isinstance(result.data, list) else None
                    sid_preview: List[str] = []
                    if tool_name == ActionType.SEARCH_LITERATURE.value and isinstance(result.data, list):
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

                if tool_name in search_tools and (not blocked_retrieval):
                    retrieval_executed_this_step = True

                if tool_name == ActionType.CONCLUDE.value and not blocked:
                    final_answer = observation_text
                    break

            if retrieval_executed_this_step:
                retrieval_action_steps_used += 1

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
                    + "\nYou MUST explicitly include ALL of these elements in your final conclusion."
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
                forced_action_msg = _invoke_llm(
                    force_conclude_llm,
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
                                    "- Provide a single speculated value + confidence for key performance metric(s) (no numeric range).\n"
                                    "- If you used literature evidence, cite source_id exactly as provided.\n"
                                    + sid_hint
                                    + elements_hint
                                )
                            )
                        )
                    ],
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
                forced = _invoke_llm(
                    llm,
                    messages
                    + [
                        (HumanMessage if use_user_role_for_action else SystemMessage)(
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
                                    "- Provide a single speculated value + confidence for key performance metric(s) (no numeric range).\n"
                                    "- If you used literature evidence, cite source_id exactly as provided.\n"
                                    + sid_hint
                                    + elements_hint
                                )
                            )
                        )
                    ],
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
            strict_json_fallback_used = False
            if requires_strict_json and not _is_valid_strict_json_payload(draft, system_prompt):
                schema_detected = _detect_strict_json_schema(system_prompt)
                _orig = draft
                repaired, repair_reason = _repair_strict_json_text(draft, system_prompt)
                if repaired and _is_valid_strict_json_payload(repaired, system_prompt):
                    draft = repaired
                    _log_strict_json_repair(
                        schema_detected=schema_detected,
                        repair_reason=repair_reason,
                        original_preview=str(_orig),
                        repaired_preview=str(repaired),
                        repaired_ok=True,
                    )
                else:
                    _log_strict_json_repair(
                        schema_detected=schema_detected,
                        repair_reason=repair_reason,
                        original_preview=str(_orig),
                        repaired_preview=str(repaired or ""),
                        repaired_ok=False,
                    )
                    salvaged, salvage_reason = _salvage_invalid_strict_json_payload(
                        _orig,
                        system_prompt=system_prompt,
                        full_query=full_query,
                        task_reaction=task_reaction,
                        task_components=task_components,
                    )
                    if salvaged and _is_valid_strict_json_payload(salvaged, system_prompt):
                        draft = salvaged
                        _log_strict_json_salvage(
                            schema_detected=schema_detected,
                            salvage_reason=salvage_reason,
                            original_preview=str(_orig),
                            salvaged_preview=str(salvaged),
                            salvage_used=True,
                        )
                    else:
                        strict_json_fallback_used = True
                        draft = _minimal_strict_json_payload(system_prompt)
                        _log_strict_json_fallback(
                            schema_detected=schema_detected,
                            reason="post_loop_forced_conclude_invalid_or_empty",
                            original_preview=str(_orig),
                        )
            elif not draft:
                draft = "No conclusion generated."

            # If strict JSON is not required and the model forgot to include a verifiable source_id,
            # attach a minimal evidence line.
            if (not requires_strict_json) and retrieved and not any(sid in draft for sid in retrieved):
                draft = draft.rstrip() + "\n\nEvidence (retrieved source_id): " + ", ".join(sorted(list(retrieved))[:3])

            # If a PROPOSE conclusion is missing required task elements, patch a minimal explicit line.
            # This keeps forced-conclude fallbacks consistent with the normal conclude guard behavior.
            if (not requires_strict_json) and task_components:
                ok_, reason_ = _validate_conclusion_against_task(draft, task_components)
                if not ok_ and "missing required catalyst" in str(reason_).lower():
                    draft = (
                        draft.rstrip()
                        + "\n\nCatalyst metal elements (exactly as provided): "
                        + ", ".join([str(c) for c in (task_components or [])])
                    )

            final_answer = draft

            # PROPOSE-only: ensure the final answer respects the unified output contract even in forced-conclude paths.
            if propose_phase and (not requires_strict_json):
                before = final_answer
                missing = _diagnose_propose_contract_missing_lines(before)
                final_answer = _coerce_propose_conclusion_to_contract(
                    draft=before,
                    full_query=full_query,
                    task_reaction=task_reaction,
                    task_components=task_components,
                    trajectory=trajectory,
                )
                if final_answer != before:
                    try:
                        self.logger.warning(
                            "react_propose_contract_rewrite",
                            extra={
                                "event": "agent.react.propose.contract_rewrite",
                                "agent_id": self.agent_id,
                                "step": step_number + 1,
                                "missing_lines": missing,
                                "original_preview": (before or "")[:500],
                                "rewritten_preview": (final_answer or "")[:500],
                            },
                        )
                    except Exception:
                        pass

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
                    thought=(
                        "Forced conclusion (model failed to call conclude tool)."
                        + (
                            " Strict JSON fallback: emitted minimal schema JSON due to empty/invalid conclude."
                            if strict_json_fallback_used
                            else ""
                        )
                    ),
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
        # Reset per-call state to avoid leaking into the next call on the same agent instance.
        self._current_requires_strict_json = False
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


def _has_metric_number(text: str) -> bool:
    """
    Return True if `text` contains a *quantitative* number/metric.

    We intentionally do NOT treat chemical formulas like O2/CO2/H2 as "has number".
    """
    s = str(text or "")
    if not s:
        return False

    # Decimal numbers (e.g., 0.83).
    if re.search(r"\b\d+\.\d+\b", s):
        return True

    # E1/2-like electrochemical metric tokens: E1/2, E_{1/2}, E 1/2, etc.
    if re.search(r"\bE\s*(?:_\{)?\s*1\s*/\s*2\s*(?:\})?", s, flags=re.IGNORECASE):
        return True

    # Numbers + common units/contexts (domain-light but useful).
    if re.search(
        r"\b\d+(?:\.\d+)?\s*(?:V|mV|mA|A|%|rpm|cm\s*(?:-2|\^-2)|A\s*g\s*(?:-1|\^-1)|mA\s*cm\s*(?:-2|\^-2))\b",
        s,
        flags=re.IGNORECASE,
    ):
        return True

    return False


def _classify_junk_chunk(text: str, min_chars: int = 80, keep_if_has_number: bool = True) -> str:
    """
    Classify low-information chunks.

    Returns:
        ""      -> keep
        "soft"  -> low-information fallback (may be backfilled if needed)
        "hard"  -> hard junk (never backfilled)
    """
    s = str(text or "").strip()
    if not s:
        return "hard"

    has_metric_number = _has_metric_number(s)

    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    first = lines[0] if lines else ""

    # Keywords-only chunks are hard junk.
    if re.match(r"^(?:#{1,6}\s*)?keywords?\b", first, flags=re.IGNORECASE):
        return "hard"

    # Heading-only chunks are hard junk unless they contain a metric number.
    if len(lines) == 1:
        ln = lines[0]
        if re.fullmatch(r"#{1,6}\s+\S.*", ln) and not has_metric_number:
            return "hard"

    # Too short and no quantitative anchor.
    if len(s) < int(min_chars):
        if keep_if_has_number and has_metric_number:
            return ""
        return "soft"

    return ""


def _is_junk_chunk(text: str, min_chars: int = 80, keep_if_has_number: bool = True) -> bool:
    """
    Backward-compatible boolean wrapper around `_classify_junk_chunk`.
    """
    return _classify_junk_chunk(text, min_chars=min_chars, keep_if_has_number=keep_if_has_number) in {"hard", "soft"}


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
            # Accept tokens like "Ni(69.00%)" but normalize to element symbols for guards.
            s = str(c).strip()
            m = re.match(r"^\s*([A-Z][a-z]?)", s)
            if m:
                s = m.group(1)
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

    required = []
    for x in (required_components or []):
        s = str(x).strip()
        if not s:
            continue
        # Normalize "Ni(69.00%)" -> "Ni" so the guard remains stable even if callers
        # pass percentage-decorated component tokens.
        m = re.match(r"^\s*([A-Z][a-z]?)", s)
        if m:
            s = m.group(1)
        required.append(s)
    required_set = {x for x in required}
    if not required_set:
        return True, ""

    detected = _extract_element_symbols(c)
    mentioned = set(detected)
    # Also count standalone mentions (e.g., "Pt-based") as coverage.
    for sym in required:
        try:
            if re.search(rf"\b{re.escape(sym)}\b", c):
                mentioned.add(sym)
        except re.error:
            continue

    missing = sorted(list(required_set.difference(mentioned)))
    if missing:
        return False, "missing required catalyst metal(s): " + ", ".join(missing)

    return True, ""


def _extract_first_json_object(text: str) -> Optional[str]:
    """
    Best-effort extraction of the first top-level JSON object from a string.

    This mirrors the debate coordinator's robustness so STRICT JSON fallbacks do not
    accidentally override outputs that contain a valid JSON object plus extra text.
    """
    s = str(text or "")
    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(s)):
        ch = s[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None


def _try_parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    s = str(text or "").strip()
    if not s:
        return None
    try:
        parsed = json.loads(s)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    obj = _extract_first_json_object(s)
    if obj:
        try:
            parsed = json.loads(obj)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return None
    return None


def _escape_unescaped_control_chars_in_json_strings(text: str) -> str:
    """
    Escape unescaped control characters inside JSON strings.

    This repairs a common model failure mode where it emits literal newlines inside a quoted
    JSON string (invalid JSON: "Invalid control character"), while keeping pretty-printed
    JSON newlines outside strings untouched.
    """
    s = str(text or "")
    if not s:
        return s

    out: List[str] = []
    in_str = False
    escaped = False

    for ch in s:
        if not in_str:
            if ch == '"':
                in_str = True
            out.append(ch)
            escaped = False
            continue

        # Inside a JSON string
        if escaped:
            out.append(ch)
            escaped = False
            continue
        if ch == "\\":
            out.append(ch)
            escaped = True
            continue
        if ch == '"':
            out.append(ch)
            in_str = False
            continue

        o = ord(ch)
        if ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif o < 0x20:
            out.append(f"\\u{o:04x}")
        else:
            out.append(ch)

    return "".join(out)


def _repair_strict_json_text(text: str, system_prompt: str) -> Tuple[Optional[str], str]:
    """
    Best-effort strict-JSON repair for provider/model quirks.

    Returns:
      (repaired_json_text_or_none, repair_reason)
    """
    raw = str(text or "")
    if not raw.strip():
        return None, "empty"

    parsed = _try_parse_json_object(raw)
    if isinstance(parsed, dict):
        return json.dumps(parsed, ensure_ascii=False), "parsed_ok"

    obj = _extract_first_json_object(raw)
    if obj is None:
        return None, "no_json_object"

    escaped = _escape_unescaped_control_chars_in_json_strings(obj)
    try:
        parsed2 = json.loads(escaped)
    except Exception as e:
        return None, f"escape_failed:{type(e).__name__}"

    if isinstance(parsed2, dict):
        return json.dumps(parsed2, ensure_ascii=False), "escaped_control_chars"
    return None, "escape_failed:non_dict_root"


def _extract_target_proposal_id_from_review_query(full_query: str) -> Optional[str]:
    s = str(full_query or "")
    m = re.search(r"target_proposal_id\s*:\s*(\S+)", s, flags=re.IGNORECASE)
    if not m:
        return None
    return str(m.group(1) or "").strip() or None


def _extract_step_numbers_from_review_query(full_query: str) -> List[int]:
    s = str(full_query or "")
    nums: List[int] = []
    for m in re.finditer(r"step_number\s*=\s*(\d+)", s, flags=re.IGNORECASE):
        try:
            nums.append(int(m.group(1)))
        except Exception:
            continue
    # De-dupe while preserving order.
    out: List[int] = []
    seen: set[int] = set()
    for n in nums:
        if n in seen:
            continue
        seen.add(n)
        out.append(n)
    return out


def _extract_review_ids_from_rebuttal_query(full_query: str) -> List[str]:
    s = str(full_query or "")
    ids: List[str] = []
    seen: set[str] = set()
    for m in re.finditer(r"review_id\s*=\s*([^\s]+)", s, flags=re.IGNORECASE):
        rid = str(m.group(1) or "").strip()
        if not rid or rid in seen:
            continue
        seen.add(rid)
        ids.append(rid)
    return ids


def _salvage_invalid_strict_json_payload(
    text: str,
    system_prompt: str,
    full_query: str,
    task_reaction: Optional[str],
    task_components: List[str],
) -> Tuple[Optional[str], str]:
    """
    Deterministically wrap invalid STRICT JSON outputs into a schema-compatible payload.

    Design goal:
    - Preserve useful raw content (metrics/critique/response) instead of falling back to a minimal empty JSON.
    - Do NOT add any additional LLM calls; this is a pure transformation.
    """
    raw = str(text or "").strip()
    if not raw:
        return None, "empty"

    schema = _detect_strict_json_schema(system_prompt)

    if schema == "proposal":
        metric_line = (raw.splitlines()[0] if raw.splitlines() else raw).strip()
        if len(metric_line) > 200:
            metric_line = metric_line[:200].rstrip()

        rt = str(task_reaction or "").strip() or "UNKNOWN"
        rt = rt.upper()
        electrode = _extract_electrode_composition_from_query(full_query)
        elems = [str(x).strip() for x in (task_components or []) if str(x).strip()]
        payload: Dict[str, Any] = {
            "reaction_type": rt,
            "electrode_composition": electrode,
            "catalyst_metal_elements": elems,
            "products": "N/A",
            "performance_metrics": metric_line,
            "confidence": "low",
            "evidence": [{"source_id": "llm"}],
            "rationale": "AUTO-SALVAGE: invalid STRICT JSON; preserved raw text in performance_metrics.",
        }
        return json.dumps(payload, ensure_ascii=False), "salvaged_proposal_text"

    if schema == "reviews":
        target_id = _extract_target_proposal_id_from_review_query(full_query)
        step_numbers = _extract_step_numbers_from_review_query(full_query)
        if not target_id or not step_numbers:
            return None, "reviews_missing_target_or_steps"

        critique = raw
        if len(critique) > 1200:
            critique = critique[:1200].rstrip() + "...(truncated)"
        payload = {
            "reviews": [
                {
                    "target_proposal_id": target_id,
                    "target_step_number": max(step_numbers),
                    "flaw_type": "other",
                    "critique": "AUTO-SALVAGE: invalid STRICT JSON/tool_calls; wrapped raw text as critique.\nRaw:\n"
                    + critique,
                    "evidence": [{"source_id": "llm"}],
                }
            ]
        }
        return json.dumps(payload, ensure_ascii=False), "salvaged_reviews_text"

    if schema == "rebuttals":
        review_ids = _extract_review_ids_from_rebuttal_query(full_query)
        if not review_ids:
            return None, "rebuttals_missing_review_ids"

        response = raw
        if len(response) > 1200:
            response = response[:1200].rstrip() + "...(truncated)"
        mode = "defend" if len(raw) >= 20 else "no_response"
        payload = {
            "rebuttals": [
                {
                    "target_review_id": rid,
                    "response_mode": mode,
                    "response": "AUTO-SALVAGE: invalid STRICT JSON/tool_calls; wrapped raw text as response.\nRaw:\n"
                    + response,
                    "evidence": [{"source_id": "llm"}],
                }
                for rid in review_ids[:20]
            ],
            "revised_claim": None,
        }
        return json.dumps(payload, ensure_ascii=False), "salvaged_rebuttals_text"

    return None, f"unknown_schema:{schema}"


def _detect_strict_json_schema(system_prompt: str) -> str:
    """
    Infer which STRICT JSON schema the current debate phase expects.
    """
    sp = str(system_prompt or "")
    if re.search(r"\"rebuttals\"\\s*:", sp):
        return "rebuttals"
    if re.search(r"\"reviews\"\\s*:", sp):
        return "reviews"
    # PROPOSE schema (top-level proposal fields).
    if re.search(r"\"reaction_type\"\\s*:", sp) and re.search(r"\"electrode_composition\"\\s*:", sp):
        return "proposal"
    low = sp.lower()
    if "rebuttals" in low:
        return "rebuttals"
    if "reviews" in low:
        return "reviews"
    if "electrode_composition" in low and "reaction_type" in low:
        return "proposal"
    return "unknown"


def _minimal_strict_json_payload(system_prompt: str) -> str:
    """
    Emit a minimal schema-compatible STRICT JSON payload.
    """
    schema = _detect_strict_json_schema(system_prompt)
    if schema == "proposal":
        payload: Dict[str, Any] = {
            "reaction_type": "UNKNOWN",
            "electrode_composition": "",
            "catalyst_metal_elements": [],
            "products": "N/A",
            "performance_metrics": "",
            "confidence": "low",
            "evidence": [],
            "rationale": "",
        }
    elif schema == "rebuttals":
        payload: Dict[str, Any] = {"rebuttals": [], "revised_claim": None}
    elif schema == "reviews":
        payload = {"reviews": []}
    else:
        # Default to REVIEW schema on unknown to keep the output parseable.
        payload = {"reviews": []}
    return json.dumps(payload, ensure_ascii=False)


def _is_valid_json_object(text: str) -> bool:
    return _try_parse_json_object(text) is not None


def _is_valid_strict_json_payload(text: str, system_prompt: str) -> bool:
    """
    Validate that text contains a parseable JSON object matching the expected top-level key.
    """
    parsed = _try_parse_json_object(text)
    if not isinstance(parsed, dict):
        return False
    schema = _detect_strict_json_schema(system_prompt)
    if schema == "proposal":
        # PROPOSE outputs are normalized by the coordinator; accept any parseable dict
        # to avoid format-related forced fallbacks that would discard useful content.
        return True
    if schema == "rebuttals":
        return isinstance(parsed.get("rebuttals"), list)
    # Default to reviews.
    return isinstance(parsed.get("reviews"), list)


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


def _is_propose_phase(system_prompt: str) -> bool:
    """
    Return True if the current call is in the debate PROPOSE phase.

    We only apply PROPOSE-specific output coercions (contract rewrite) in this phase.
    """
    s = str(system_prompt or "")
    return re.search(r"###\s*Debate Phase\s*:\s*PROPOSE\b", s, flags=re.IGNORECASE) is not None


def _parse_retrieval_budget_from_system_prompt(system_prompt: str) -> Optional[int]:
    """
    Parse per-phase retrieval budget from a system prompt.

    Expected formats (case-insensitive):
      - "Retrieval budget: at most ONE ACTION step ..."
      - "Retrieval budget: at most TWO ACTION steps ..."
    """
    s = str(system_prompt or "")
    if not s:
        return None

    m = re.search(
        r"Retrieval budget\s*:\s*at most\s*(ONE|TWO|THREE|\d+)\s+ACTION\s+step(?:s)?\b",
        s,
        flags=re.IGNORECASE,
    )
    if not m:
        return None

    token = str(m.group(1) or "").strip().lower()
    word_map = {"one": 1, "two": 2, "three": 3}
    if token in word_map:
        return int(word_map[token])
    try:
        return int(token)
    except Exception:
        return None


def _diagnose_propose_contract_missing_lines(text: str) -> List[str]:
    """
    Return a list of missing required output-contract lines for PROPOSE conclusions.
    """
    s = str(text or "")
    missing: List[str] = []

    required = [
        ("Reaction Type", r"^\s*Reaction Type\s*:"),
        ("Electrode composition (exactly as provided)", r"^\s*Electrode composition\s*\(exactly as provided\)\s*:"),
        ("Products", r"^\s*Products\s*:"),
        ("Performance Metrics", r"^\s*Performance Metrics\s*:"),
        ("Evidence", r"^\s*Evidence\s*:"),
    ]
    for label, pat in required:
        if re.search(pat, s, flags=re.IGNORECASE | re.MULTILINE) is None:
            missing.append(label)
    return missing


def _extract_electrode_composition_from_query(full_query: str) -> str:
    """
    Extract the electrode composition line from the initial query (if present).
    """
    q = str(full_query or "")
    if not q:
        return ""

    for pat in [
        r"Electrode composition\s*\(relative %\)\s*:\s*([^\n\r]+)",
        r"Electrode composition\s*\(exactly as provided\)\s*:\s*([^\n\r]+)",
    ]:
        m = re.search(pat, q, flags=re.IGNORECASE)
        if m:
            return str(m.group(1) or "").strip()
    return ""


def _extract_line_starting_with(prefix: str, text: str) -> Optional[str]:
    """
    Extract the first line whose prefix matches `<prefix>:` (case-insensitive).
    """
    p = str(prefix or "").strip()
    if not p:
        return None
    for raw in (text or "").splitlines():
        if re.match(rf"^\s*{re.escape(p)}\s*:", raw, flags=re.IGNORECASE):
            return raw.strip()
    return None


def _extract_overpotential_point_estimate(text: str) -> Optional[str]:
    """
    Best-effort extraction of a single overpotential point estimate from text.
    """
    s = str(text or "")
    if not s:
        return None

    # Prefer patterns anchored to 10 mA.
    pats = [
        r"\b(\d+(?:\.\d+)?)\s*mV\b[^\n]{0,120}\b10\s*mA\b",
        r"\b10\s*mA\b[^\n]{0,120}\b(\d+(?:\.\d+)?)\s*mV\b",
        r"\b(\d+(?:\.\d+)?)\s*V\b[^\n]{0,120}\b10\s*mA\b",
        r"\b10\s*mA\b[^\n]{0,120}\b(\d+(?:\.\d+)?)\s*V\b",
    ]
    for pat in pats:
        m = re.search(pat, s, flags=re.IGNORECASE)
        if m:
            val = str(m.group(1) or "").strip()
            if not val:
                continue
            unit = "mV" if "mv" in pat.lower() else "V"
            return f"{val} {unit}"

    # Fallback: any mV/V value.
    m = re.search(r"\b(\d+(?:\.\d+)?)\s*mV\b", s, flags=re.IGNORECASE)
    if m:
        return f"{str(m.group(1)).strip()} mV"
    m = re.search(r"\b(\d+(?:\.\d+)?)\s*V\b", s, flags=re.IGNORECASE)
    if m:
        return f"{str(m.group(1)).strip()} V"
    return None


def _iter_search_literature_items(trajectory: Optional["ReActTrajectory"]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if trajectory is None:
        return items

    for step in getattr(trajectory, "steps", []) or []:
        for call in getattr(step, "tool_calls", []) or []:
            if getattr(call, "tool_name", "") != ActionType.SEARCH_LITERATURE.value:
                continue
            data = getattr(call, "observation_data", None) or []
            if not isinstance(data, list):
                continue
            for it in data:
                if isinstance(it, dict):
                    items.append(it)
    return items


def _select_evidence_source_ids_from_trajectory(
    trajectory: Optional["ReActTrajectory"], limit: int = 3
) -> List[str]:
    """
    Pick up to `limit` rag:chroma/... ids from retrieved search_literature results,
    preferring on-reaction and non-forbidden chunks.
    """
    items = _iter_search_literature_items(trajectory)

    preferred: List[Dict[str, Any]] = []
    fallback: List[Dict[str, Any]] = []
    for it in items:
        sid = it.get("source_id")
        if not sid:
            continue
        rm = bool(it.get("reaction_match"))
        forbid = it.get("forbidden_elements") or []
        if rm and not forbid:
            preferred.append(it)
        else:
            fallback.append(it)

    out: List[str] = []
    seen: set[str] = set()
    for it in (preferred + fallback):
        sid = str(it.get("source_id") or "").strip()
        if not sid or sid in seen:
            continue
        seen.add(sid)
        out.append(sid)
        if len(out) >= int(limit):
            break
    return out


def _extract_overpotential_from_trajectory(trajectory: Optional["ReActTrajectory"]) -> Optional[str]:
    items = _iter_search_literature_items(trajectory)
    preferred: List[Dict[str, Any]] = []
    fallback: List[Dict[str, Any]] = []
    for it in items:
        rm = bool(it.get("reaction_match"))
        forbid = it.get("forbidden_elements") or []
        if rm and not forbid:
            preferred.append(it)
        else:
            fallback.append(it)

    for it in (preferred + fallback):
        val = _extract_overpotential_point_estimate(it.get("text") or "")
        if val:
            return val
    return None


def _coerce_propose_conclusion_to_contract(
    draft: str,
    full_query: str,
    task_reaction: Optional[str],
    task_components: List[str],
    trajectory: Optional["ReActTrajectory"],
) -> str:
    """
    PROPOSE-only: rewrite a draft conclusion into the unified output contract.

    This is a deterministic, local transformation: no extra model calls.
    """
    s = str(draft or "").strip()
    missing = _diagnose_propose_contract_missing_lines(s)
    if not missing:
        return s

    rt = str(task_reaction or "").strip().upper() or "UNKNOWN"
    electrode = _extract_electrode_composition_from_query(full_query)
    if not electrode and task_components:
        electrode = ", ".join([str(c) for c in (task_components or []) if str(c).strip()])

    # Products are only applicable to CO2RR.
    products_line = None
    if rt == "CO2RR":
        products_line = _extract_line_starting_with("Products", s)
        if not products_line:
            products_line = "Products: (missing)"
    else:
        products_line = "Products: N/A"

    metrics_line = _extract_line_starting_with("Performance Metrics", s)
    if not metrics_line:
        op = _extract_overpotential_point_estimate(s)
        if not op:
            op = _extract_overpotential_from_trajectory(trajectory)
        if op and rt in {"OER", "HER", "UOR", "EOR", "HZOR"}:
            metrics_line = f"Performance Metrics: {op} overpotential at 10 mA/cm^2 (Confidence: Low)"
        elif op:
            metrics_line = f"Performance Metrics: {op} (Confidence: Low)"
        else:
            metrics_line = "Performance Metrics: (missing) (Confidence: Low)"
    else:
        # Ensure confidence marker exists.
        if re.search(r"\bconf(?:idence)?\b", metrics_line, flags=re.IGNORECASE) is None:
            metrics_line = metrics_line.rstrip() + " (Confidence: Low)"

    evidence_ids = _extract_source_ids_from_text(s)
    if not evidence_ids:
        evidence_ids = _select_evidence_source_ids_from_trajectory(trajectory, limit=3)
    if evidence_ids:
        evidence_line = "Evidence: " + "; ".join(evidence_ids[:3])
    else:
        evidence_line = "Evidence: llm"

    lines: List[str] = [
        f"Reaction Type: {rt}",
        f"Electrode composition (exactly as provided): {electrode}".rstrip(),
        products_line,
        metrics_line,
        evidence_line,
        "",
        "Rationale:",
        s if s else "(empty draft)",
    ]
    return "\n".join([ln for ln in lines if ln is not None]).strip()


def _validate_conclusion_against_task_with_evidence(
    conclusion: str,
    required_components: List[str],
    trajectory: Optional["ReActTrajectory"],
    enforce_point_estimate: bool = True,
    require_confidence: bool = True,
) -> Tuple[bool, str]:
    """
    PROPOSE guard: enforce element checks on the conclusion text, and (best-effort)
    flag when cited source_id chunks contain forbidden (off-task) catalyst metals.

    IMPORTANT:
    - We still BLOCK if the conclusion is missing any required task metal elements.
    - We DO NOT block solely because the conclusion mentions additional elements or because a cited
      chunk contains forbidden elements; we return a warning so the agent can still conclude without
      getting stuck in retrieval loops.
    """
    ok, reason = _validate_conclusion_against_task(conclusion, required_components)
    if not ok:
        return ok, reason

    warnings: List[str] = []
    # Warn (but do not block) if the conclusion mentions additional catalyst metals beyond the task.
    required = []
    for x in (required_components or []):
        s = str(x).strip()
        if not s:
            continue
        m = re.match(r"^\s*([A-Z][a-z]?)", s)
        if m:
            s = m.group(1)
        required.append(s)
    required_set = {x for x in required}
    detected = _extract_element_symbols(conclusion)
    extra_elements = sorted(list(detected.difference(required_set.union(_ALLOWED_NON_CATALYST_ELEMENTS))))
    if extra_elements:
        warnings.append(
            "conclusion mentions extra catalyst metal(s) not in task: "
            + ", ".join([str(x) for x in extra_elements[:10]])
        )

    # PROPOSE metric-line rules: only enforce/warn on the `Performance Metrics:` line.
    # We allow literature ranges elsewhere (Rationale/Evidence), but prefer a single point estimate + confidence
    # on the designated metric line for downstream parsing/analytics.
    if bool(enforce_point_estimate) or bool(require_confidence):
        metrics_line: Optional[str] = None
        for raw in (conclusion or "").splitlines():
            if re.match(r"^\s*Performance Metrics\s*:", raw, flags=re.IGNORECASE):
                metrics_line = raw.strip()
                break

        if not metrics_line:
            warnings.append("missing_metrics_line (expected: 'Performance Metrics: <point estimate> (Confidence: ...)')")
        else:
            if bool(enforce_point_estimate):
                has_pm = ("±" in metrics_line) or (re.search(r"\+/\-", metrics_line) is not None)
                has_range = re.search(
                    r"\b\d+(?:\.\d+)?\s*[-\u2013\u2014]\s*\d+(?:\.\d+)?\b", metrics_line
                ) is not None
                has_to = re.search(
                    r"\b\d+(?:\.\d+)?\s+to\s+\d+(?:\.\d+)?\b", metrics_line, flags=re.IGNORECASE
                ) is not None
                if has_pm or has_range or has_to:
                    warnings.append(
                        "performance_metrics_should_be_point_estimate_plus_confidence (avoid ±/ranges on the Performance Metrics line)"
                    )

            if bool(require_confidence) and (re.search(r"\bconf(?:idence)?\.?\b", metrics_line, flags=re.IGNORECASE) is None):
                warnings.append(
                    "missing_confidence_on_performance_metrics_line (include 'Confidence:' or 'Conf.')"
                )

    cited = _extract_source_ids_from_text(conclusion)
    if not cited or trajectory is None:
        if warnings:
            return True, "warning: " + " | ".join(warnings)
        return True, ""

    by_sid: Dict[str, Dict[str, Any]] = {}
    try:
        for step in getattr(trajectory, "steps", []) or []:
            for call in getattr(step, "tool_calls", []) or []:
                if getattr(call, "tool_name", "") != ActionType.SEARCH_LITERATURE.value:
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
        warnings.append(
            "cited evidence appears off-task (forbidden catalyst metals in retrieved chunk): "
            + "; ".join(offenders[:3])
        )

    if warnings:
        return True, "warning: " + " | ".join(warnings)
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

    if last_action in {"search_literature", "search_experience"}:
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
            if getattr(call, "tool_name", "") not in {
                ActionType.SEARCH_LITERATURE.value,
                ActionType.FETCH_LITERATURE_CHUNK.value,
            }:
                continue
            data = getattr(call, "observation_data", None)
            if not isinstance(data, list):
                continue
            for item in data:
                if not isinstance(item, dict):
                    continue
                sid = item.get("source_id")
                if sid:
                    sids.add(sid)
    # Backward-compatible fallback (legacy single-tool steps).
    for step in getattr(trajectory, "steps", []) or []:
        if getattr(step, "action_name", "") not in {
            ActionType.SEARCH_LITERATURE.value,
            ActionType.FETCH_LITERATURE_CHUNK.value,
        }:
            continue
        data = getattr(step, "observation_data", None)
        if not isinstance(data, list):
            continue
        for item in data:
            if not isinstance(item, dict):
                continue
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
        # Optional task-alignment annotations (added by _tool_search_literature).
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
            if call.tool_name in {ActionType.SEARCH_LITERATURE.value, ActionType.FETCH_LITERATURE_CHUNK.value} and isinstance(call.observation_data, list):
                sources.extend(call.observation_data[:3])
            if call.tool_name == ActionType.SEARCH_EXPERIENCE.value and isinstance(call.observation_data, list):
                sources.extend(call.observation_data[:2])

        # Backward-compatible fallback (legacy single-tool steps).
        action_name = getattr(step, "action_name", "")
        if action_name in {ActionType.SEARCH_LITERATURE.value, ActionType.FETCH_LITERATURE_CHUNK.value} and isinstance(getattr(step, "observation_data", None), list):
            sources.extend(getattr(step, "observation_data")[:3])
        if action_name == ActionType.SEARCH_EXPERIENCE.value and isinstance(getattr(step, "observation_data", None), list):
            sources.extend(getattr(step, "observation_data")[:2])
    return sources
