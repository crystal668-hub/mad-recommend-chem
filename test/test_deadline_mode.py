from __future__ import annotations

import unittest
from unittest.mock import patch


class _Msg:
    def __init__(self, content: str = "", additional_kwargs=None, tool_calls=None, tool_call_id: str | None = None):
        self.content = content
        self.additional_kwargs = additional_kwargs or {}
        self.tool_calls = tool_calls
        self.tool_call_id = tool_call_id


class _SystemMessage(_Msg):
    pass


class _HumanMessage(_Msg):
    pass


class _AIMessage(_Msg):
    pass


class _ToolMessage(_Msg):
    def __init__(self, content: str = "", tool_call_id: str | None = None, additional_kwargs=None, tool_calls=None):
        super().__init__(content=content, additional_kwargs=additional_kwargs, tool_calls=tool_calls, tool_call_id=tool_call_id)


class _DummyLLM:
    def __init__(self, tool_choice: str | None = None):
        self._tool_choice = tool_choice

    def bind_tools(self, tools, tool_choice: str | None = None):
        # Return a new instance to mimic LangChain RunnableBinding behavior.
        return _DummyLLM(tool_choice=tool_choice)

    def bind(self, tools=None, tool_choice: str | None = None):  # pragma: no cover (compat path)
        return self.bind_tools(tools, tool_choice=tool_choice)

    def invoke(self, messages):
        # In deadline/forced conclude paths we should be invoked with tool_choice="conclude".
        if self._tool_choice == "conclude":
            return _AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_conclude_1",
                        "name": "conclude",
                        "args": {"conclusion": "Final answer."},
                    }
                ],
            )
        # THOUGHT / fallback free-form.
        return _AIMessage(content="Thought.")


class _ReactiveLLMBackend:
    def __init__(self) -> None:
        self.call_count = 0

    def invoke(self, messages, *, tool_choice: str | None = None):
        self.call_count += 1
        last_content = str(getattr(messages[-1], "content", "") or "")
        if "CURRENT PHASE: THOUGHT" in last_content:
            return _AIMessage(content="Use conclude.")
        if "CURRENT PHASE: ACTION" in last_content:
            return _AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_conclude_1",
                        "name": "conclude",
                        "args": {"conclusion": "Final answer."},
                    }
                ],
            )
        if tool_choice == "conclude":
            return _AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_conclude_1",
                        "name": "conclude",
                        "args": {"conclusion": "Final answer."},
                    }
                ],
            )
        if self.call_count >= 2:
            return _AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_conclude_1",
                        "name": "conclude",
                        "args": {"conclusion": "Final answer."},
                    }
                ],
            )
        return _AIMessage(content="Thought.")


class _ReactiveLLM:
    def __init__(self, backend: _ReactiveLLMBackend, tool_choice: str | None = None):
        self._backend = backend
        self._tool_choice = tool_choice

    def bind_tools(self, tools, tool_choice: str | None = None):
        return _ReactiveLLM(self._backend, tool_choice=tool_choice)

    def bind(self, tools=None, tool_choice: str | None = None):  # pragma: no cover
        return self.bind_tools(tools, tool_choice=tool_choice)

    def invoke(self, messages):
        return self._backend.invoke(messages, tool_choice=self._tool_choice)


class _StructuredDummyLLM:
    def __init__(self, tool_choice: str | None = None, *, emit_tool_call: bool = True):
        self._tool_choice = tool_choice
        self._emit_tool_call = emit_tool_call

    def bind_tools(self, tools, tool_choice: str | None = None):
        return _StructuredDummyLLM(tool_choice=tool_choice, emit_tool_call=self._emit_tool_call)

    def bind(self, tools=None, tool_choice: str | None = None):  # pragma: no cover
        return self.bind_tools(tools, tool_choice=tool_choice)

    def invoke(self, messages):
        if self._tool_choice == "conclude" and self._emit_tool_call:
            return _AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_conclude_structured",
                        "name": "conclude",
                        "args": {"submission": {"submission_id": "submission_cycle_1"}},
                    }
                ],
            )
        return _AIMessage(content="Thought.")


class _DeadlineStructuredBackend:
    def __init__(self) -> None:
        self.action_calls = 0

    def invoke(self, messages, *, tool_choice: str | None = None):
        history_text = "\n".join(str(getattr(message, "content", "") or "") for message in list(messages or []))
        last_content = str(getattr(messages[-1], "content", "") or "")
        if "CURRENT PHASE: THOUGHT" in last_content:
            return _AIMessage(content="Search first, then conclude.")
        if tool_choice == "conclude":
            return _AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_conclude_structured",
                        "name": "conclude",
                        "args": {"submission": {"submission_id": "submission_cycle_1"}},
                    }
                ],
            )
        if "CURRENT PHASE: ACTION" in history_text:
            self.action_calls += 1
            return _AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_search_1",
                        "name": "search_papers",
                        "args": {"query": "pt/c her"},
                    }
                ],
            )
        return _AIMessage(content="Thought.")


class _DeadlineBlockedSearchBackend:
    def __init__(self) -> None:
        self.action_calls = 0

    def invoke(self, messages, *, tool_choice: str | None = None):
        history_text = "\n".join(str(getattr(message, "content", "") or "") for message in list(messages or []))
        last_content = str(getattr(messages[-1], "content", "") or "")
        if "CURRENT PHASE: THOUGHT" in last_content:
            return _AIMessage(content="Need more search.")
        if tool_choice == "conclude":
            saw_policy_tool_message = any(
                "broad discovery is disabled when only 2 steps remain" in str(getattr(message, "content", "") or "")
                for message in list(messages or [])
            )
            if saw_policy_tool_message:
                return _AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_conclude_structured",
                            "name": "conclude",
                            "args": {"submission": {"submission_id": "submission_cycle_1"}},
                        }
                    ],
                )
            return _AIMessage(content="", tool_calls=[])
        if "CURRENT PHASE: ACTION" in history_text:
            self.action_calls += 1
            return _AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_search_1",
                        "name": "search_papers",
                        "args": {"query": "pt/c her"},
                    }
                ],
            )
        return _AIMessage(content="Thought.")


class _DeadlineFollowupRetrievalBackend:
    def __init__(self) -> None:
        self.action_calls = 0

    def invoke(self, messages, *, tool_choice: str | None = None):
        history_text = "\n".join(str(getattr(message, "content", "") or "") for message in list(messages or []))
        last_content = str(getattr(messages[-1], "content", "") or "")
        if "CURRENT PHASE: THOUGHT" in last_content:
            return _AIMessage(content="Acquire the best paper and extract evidence before conclude.")
        if tool_choice == "conclude":
            return _AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_conclude_structured",
                        "name": "conclude",
                        "args": {"submission": {"submission_id": "submission_cycle_1"}},
                    }
                ],
            )
        if "CURRENT PHASE: ACTION" in history_text:
            self.action_calls += 1
            return _AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_search_1",
                        "name": "search_papers",
                        "args": {"query": "pt/c her"},
                    },
                    {
                        "id": "call_acquire_1",
                        "name": "download_document",
                        "args": {"paper_id": "paper-1"},
                    },
                    {
                        "id": "call_extract_1",
                        "name": "extract_evidence",
                        "args": {"paper_id": "paper-1"},
                    },
                ],
            )
        return _AIMessage(content="Thought.")


class _DeadlineLastChanceExtractBackend:
    def __init__(self) -> None:
        self.action_calls = 0
        self.conclude_messages = []

    def invoke(self, messages, *, tool_choice: str | None = None):
        history_text = "\n".join(str(getattr(message, "content", "") or "") for message in list(messages or []))
        last_content = str(getattr(messages[-1], "content", "") or "")
        if "CURRENT PHASE: THOUGHT" in last_content:
            return _AIMessage(content="Acquire a relevant paper before concluding.")
        if tool_choice == "conclude":
            self.conclude_messages = list(messages or [])
            return _AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_conclude_structured",
                        "name": "conclude",
                        "args": {"submission": {"submission_id": "submission_cycle_1"}},
                    }
                ],
            )
        if "CURRENT PHASE: ACTION" in history_text:
            self.action_calls += 1
            return _AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_acquire_1",
                        "name": "download_document",
                        "args": {"paper_id": "paper-1"},
                    }
                ],
            )
        return _AIMessage(content="Thought.")


class _DeadlineSearchOnlyBackend:
    def __init__(self) -> None:
        self.action_calls = 0

    def invoke(self, messages, *, tool_choice: str | None = None):
        history_text = "\n".join(str(getattr(message, "content", "") or "") for message in list(messages or []))
        last_content = str(getattr(messages[-1], "content", "") or "")
        if "CURRENT PHASE: THOUGHT" in last_content:
            return _AIMessage(content="Search first, then keep searching if needed.")
        if tool_choice == "conclude":
            return _AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_conclude_structured",
                        "name": "conclude",
                        "args": {"submission": {"submission_id": "submission_cycle_1"}},
                    }
                ],
            )
        if "CURRENT PHASE: ACTION" in history_text:
            self.action_calls += 1
            return _AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_search_1",
                        "name": "search_papers",
                        "args": {"query": "pt/c her"},
                    }
                ],
            )
        return _AIMessage(content="Thought.")


class _DeadlineParsedPaperBackend:
    def __init__(self) -> None:
        self.action_calls = 0

    def invoke(self, messages, *, tool_choice: str | None = None):
        history_text = "\n".join(str(getattr(message, "content", "") or "") for message in list(messages or []))
        last_content = str(getattr(messages[-1], "content", "") or "")
        if "CURRENT PHASE: THOUGHT" in last_content:
            return _AIMessage(content="Parse the chosen paper, then conclude.")
        if tool_choice == "conclude":
            return _AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_conclude_structured",
                        "name": "conclude",
                        "args": {"submission": {"submission_id": "submission_cycle_1"}},
                    }
                ],
            )
        if "CURRENT PHASE: ACTION" in history_text:
            self.action_calls += 1
            return _AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_parse_1",
                        "name": "parse_document",
                        "args": {"paper_id": "paper-1"},
                    }
                ],
            )
        return _AIMessage(content="Thought.")


class _InvokeTool:
    def __init__(self, fn):
        self._fn = fn

    def invoke(self, payload):
        return self._fn(payload)


class DeadlineModeTests(unittest.TestCase):
    def test_deadline_mode_forces_conclude_on_last_step(self):
        from agents.react_agent import ReActAgent

        dummy_llm = _DummyLLM()

        with patch(
            "agents.react_agent._lazy_langchain_imports",
            return_value=(object, _SystemMessage, _HumanMessage, _AIMessage, _ToolMessage, object),
        ):
            agent = ReActAgent(
                agent_id="t1",
                name="test",
                model_config={"deadline_mode": True},
                rag_system=None,
                experience_store=None,
                system_prompt="",
                max_react_steps=10,
                verbose=False,
            )

            with patch.object(agent, "_get_llm", return_value=dummy_llm), patch.object(
                agent, "_build_tools", return_value=([], {})
            ):
                response, trajectory = agent.generate_response_with_react(
                    query="Deadline mode unit test.",
                    components=["Ni", "Fe", "Co"],
                    max_steps_override=1,
                )

        self.assertEqual(len(getattr(trajectory, "steps", []) or []), 1)
        step = (trajectory.steps or [None])[0]
        self.assertEqual(getattr(step, "action", None), "conclude")
        self.assertTrue((response.content or "").strip())
        # The forced conclude helper should patch missing required components if absent.
        self.assertIn("Ni, Fe, Co", response.content)

    def test_real_conclude_tool_sets_structured_output(self):
        from agents.react_agent import ReActAgent, ToolResult

        backend = _ReactiveLLMBackend()
        llm = _ReactiveLLM(backend)

        def _conclude(_payload):
            return ToolResult(
                observation='{"submission_id":"submission_cycle_1"}',
                data={"__conclude_valid__": True, "submission": {"submission_id": "submission_cycle_1"}},
            )

        conclude_tool = _InvokeTool(_conclude)

        with patch(
            "agents.react_agent._lazy_langchain_imports",
            return_value=(object, _SystemMessage, _HumanMessage, _AIMessage, _ToolMessage, object),
        ):
            agent = ReActAgent(
                agent_id="t2",
                name="test",
                model_config={"deadline_mode": False},
                rag_system=None,
                experience_store=None,
                system_prompt="",
                max_react_steps=2,
                verbose=False,
            )

            with patch.object(agent, "_get_llm", return_value=llm), patch.object(
                agent,
                "_build_tools",
                return_value=([conclude_tool], {"conclude": conclude_tool}),
            ):
                response, _trajectory = agent.generate_response_with_react(
                    query="Return a structured submission.",
                )

        self.assertEqual(
            {"kind": "submission", "payload": {"submission_id": "submission_cycle_1"}},
            response.structured_output,
        )

    def test_forced_conclude_free_text_keeps_structured_output_empty(self):
        from agents.react_agent import ReActAgent

        dummy_llm = _DummyLLM()

        with patch(
            "agents.react_agent._lazy_langchain_imports",
            return_value=(object, _SystemMessage, _HumanMessage, _AIMessage, _ToolMessage, object),
        ):
            agent = ReActAgent(
                agent_id="t3",
                name="test",
                model_config={"deadline_mode": True},
                rag_system=None,
                experience_store=None,
                system_prompt="",
                max_react_steps=10,
                verbose=False,
            )

            with patch.object(agent, "_get_llm", return_value=dummy_llm), patch.object(
                agent, "_build_tools", return_value=([], {})
            ):
                response, _trajectory = agent.generate_response_with_react(
                    query="Forced conclude unit test.",
                    components=["Ni", "Fe", "Co"],
                    max_steps_override=1,
                )

        self.assertIsNone(response.structured_output)

    def test_structured_deadline_force_conclude_invokes_real_tool(self):
        from agents.react_agent import ReActAgent, ToolResult

        llm = _StructuredDummyLLM()

        def _conclude(_payload):
            return ToolResult(
                observation='{"submission_id":"submission_cycle_1"}',
                data={"__conclude_valid__": True, "submission": {"submission_id": "submission_cycle_1"}},
            )

        conclude_tool = _InvokeTool(_conclude)

        with patch(
            "agents.react_agent._lazy_langchain_imports",
            return_value=(object, _SystemMessage, _HumanMessage, _AIMessage, _ToolMessage, object),
        ):
            agent = ReActAgent(
                agent_id="t4",
                name="test",
                model_config={"deadline_mode": True},
                rag_system=None,
                experience_store=None,
                system_prompt="",
                max_react_steps=10,
                verbose=False,
                conclude_argument_name="submission",
                conclude_output_kind="submission",
            )

            with patch.object(agent, "_get_llm", return_value=llm), patch.object(
                agent,
                "_build_tools",
                return_value=([conclude_tool], {"conclude": conclude_tool}),
            ):
                response, trajectory = agent.generate_response_with_react(
                    query="Return a structured submission.",
                    max_steps_override=1,
                )

        self.assertEqual({"kind": "submission", "payload": {"submission_id": "submission_cycle_1"}}, response.structured_output)
        self.assertEqual("conclude", trajectory.steps[0].action)

    def test_structured_deadline_retrieval_attempt_is_forced_to_conclude_same_step(self):
        from agents.react_agent import ReActAgent, ToolResult

        backend = _DeadlineStructuredBackend()
        llm = _ReactiveLLM(backend)

        def _search(_payload):
            return ToolResult(observation="[]", data=[])

        def _conclude(_payload):
            return ToolResult(
                observation='{"submission_id":"submission_cycle_1"}',
                data={"__conclude_valid__": True, "submission": {"submission_id": "submission_cycle_1"}},
            )

        search_tool = _InvokeTool(_search)
        conclude_tool = _InvokeTool(_conclude)

        with patch(
            "agents.react_agent._lazy_langchain_imports",
            return_value=(object, _SystemMessage, _HumanMessage, _AIMessage, _ToolMessage, object),
        ):
            agent = ReActAgent(
                agent_id="t5",
                name="test",
                model_config={"deadline_mode": True},
                rag_system=None,
                experience_store=None,
                system_prompt="",
                max_react_steps=10,
                verbose=False,
                tools=[search_tool, conclude_tool],
                search_tool_names=["search_papers"],
                analysis_tool_names=["conclude"],
                conclude_argument_name="submission",
                conclude_output_kind="submission",
            )

            with patch.object(agent, "_get_llm", return_value=llm), patch.object(
                agent,
                "_build_tools",
                return_value=([search_tool, conclude_tool], {"search_papers": search_tool, "conclude": conclude_tool}),
            ):
                response, trajectory = agent.generate_response_with_react(
                    query="Return a structured submission.",
                    max_steps_override=2,
                )

        self.assertEqual("conclude", trajectory.steps[-1].action)
        self.assertFalse(
            any(call.tool_name == "search_papers" for step in trajectory.steps for call in step.tool_calls)
        )
        self.assertEqual({"kind": "submission", "payload": {"submission_id": "submission_cycle_1"}}, response.structured_output)

    def test_structured_force_conclude_without_tool_call_raises(self):
        from agents.react_agent import ReActAgent

        llm = _StructuredDummyLLM(emit_tool_call=False)

        with patch(
            "agents.react_agent._lazy_langchain_imports",
            return_value=(object, _SystemMessage, _HumanMessage, _AIMessage, _ToolMessage, object),
        ):
            agent = ReActAgent(
                agent_id="t6",
                name="test",
                model_config={"deadline_mode": True},
                rag_system=None,
                experience_store=None,
                system_prompt="",
                max_react_steps=10,
                verbose=False,
                conclude_argument_name="submission",
                conclude_output_kind="submission",
            )

            with patch.object(agent, "_get_llm", return_value=llm), patch.object(
                agent, "_build_tools", return_value=([], {})
            ):
                with self.assertRaises(RuntimeError):
                    agent.generate_response_with_react(
                        query="Forced conclude must fail without a structured tool call.",
                        max_steps_override=1,
                    )

    def test_structured_deadline_blocked_search_history_is_passed_to_forced_conclude(self):
        from agents.react_agent import ReActAgent, ToolResult

        backend = _DeadlineBlockedSearchBackend()
        llm = _ReactiveLLM(backend)

        def _search(_payload):
            return ToolResult(observation="[]", data=[])

        def _conclude(_payload):
            return ToolResult(
                observation='{"submission_id":"submission_cycle_1"}',
                data={"__conclude_valid__": True, "submission": {"submission_id": "submission_cycle_1"}},
            )

        search_tool = _InvokeTool(_search)
        conclude_tool = _InvokeTool(_conclude)

        with patch(
            "agents.react_agent._lazy_langchain_imports",
            return_value=(object, _SystemMessage, _HumanMessage, _AIMessage, _ToolMessage, object),
        ):
            agent = ReActAgent(
                agent_id="t7",
                name="test",
                model_config={"deadline_mode": True},
                rag_system=None,
                experience_store=None,
                system_prompt="",
                max_react_steps=10,
                verbose=False,
                tools=[search_tool, conclude_tool],
                search_tool_names=["search_papers"],
                analysis_tool_names=["conclude"],
                conclude_argument_name="submission",
                conclude_output_kind="submission",
            )

            with patch.object(agent, "_get_llm", return_value=llm), patch.object(
                agent,
                "_build_tools",
                return_value=([search_tool, conclude_tool], {"search_papers": search_tool, "conclude": conclude_tool}),
            ):
                response, trajectory = agent.generate_response_with_react(
                    query="Return a structured submission.",
                    max_steps_override=2,
                )

        self.assertEqual({"kind": "submission", "payload": {"submission_id": "submission_cycle_1"}}, response.structured_output)
        self.assertEqual("conclude", trajectory.steps[-1].action)

    def test_structured_deadline_allows_followup_retrieval_tools_before_forced_conclude(self):
        from agents.react_agent import ReActAgent, ToolResult

        backend = _DeadlineFollowupRetrievalBackend()
        llm = _ReactiveLLM(backend)
        executed: list[str] = []

        def _search(_payload):
            executed.append("search_papers")
            return ToolResult(observation="[]", data=[])

        def _download(_payload):
            executed.append("download_document")
            return ToolResult(observation='{"paper_id":"paper-1"}', data={"paper_id": "paper-1"})

        def _parse(_payload):
            executed.append("parse_document")
            return ToolResult(
                observation='{"paper_id":"paper-1","fulltext_status":"fulltext_indexed"}',
                data={"paper_id": "paper-1", "fulltext_status": "fulltext_indexed"},
            )

        def _extract(_payload):
            executed.append("extract_evidence")
            return ToolResult(observation='{"evidence":["ev-1"]}', data={"evidence": ["ev-1"]})

        def _conclude(_payload):
            return ToolResult(
                observation='{"submission_id":"submission_cycle_1"}',
                data={"__conclude_valid__": True, "submission": {"submission_id": "submission_cycle_1"}},
            )

        search_tool = _InvokeTool(_search)
        download_tool = _InvokeTool(_download)
        parse_tool = _InvokeTool(_parse)
        extract_tool = _InvokeTool(_extract)
        conclude_tool = _InvokeTool(_conclude)

        with patch(
            "agents.react_agent._lazy_langchain_imports",
            return_value=(object, _SystemMessage, _HumanMessage, _AIMessage, _ToolMessage, object),
        ):
            agent = ReActAgent(
                agent_id="t8",
                name="test",
                model_config={"deadline_mode": True},
                rag_system=None,
                experience_store=None,
                system_prompt="",
                max_react_steps=10,
                verbose=False,
                tools=[search_tool, download_tool, parse_tool, extract_tool, conclude_tool],
                search_tool_names=["search_papers", "download_document", "parse_document", "extract_evidence"],
                analysis_tool_names=["conclude"],
                conclude_argument_name="submission",
                conclude_output_kind="submission",
            )

            with patch.object(agent, "_get_llm", return_value=llm), patch.object(
                agent,
                "_build_tools",
                return_value=(
                    [search_tool, download_tool, parse_tool, extract_tool, conclude_tool],
                    {
                        "search_papers": search_tool,
                        "download_document": download_tool,
                        "parse_document": parse_tool,
                        "extract_evidence": extract_tool,
                        "conclude": conclude_tool,
                    },
                ),
            ):
                response, trajectory = agent.generate_response_with_react(
                    query="Return a structured submission.",
                    max_steps_override=2,
                )

        self.assertEqual(["download_document", "extract_evidence"], executed)
        self.assertEqual({"kind": "submission", "payload": {"submission_id": "submission_cycle_1"}}, response.structured_output)
        self.assertEqual("conclude", trajectory.steps[-1].action)

    def test_structured_deadline_last_chance_extract_runs_before_forced_conclude(self):
        from agents.react_agent import ReActAgent, ToolResult

        backend = _DeadlineLastChanceExtractBackend()
        llm = _ReactiveLLM(backend)
        executed: list[str] = []

        def _download(_payload):
            executed.append("download_document")
            return ToolResult(observation='{"paper_id":"paper-1"}', data={"paper_id": "paper-1"})

        def _parse(_payload):
            executed.append("parse_document")
            return ToolResult(
                observation='{"paper_id":"paper-1","fulltext_status":"fulltext_indexed"}',
                data={"paper_id": "paper-1", "fulltext_status": "fulltext_indexed"},
            )

        def _extract(_payload):
            executed.append("extract_evidence")
            return ToolResult(observation='{"evidence":["ev-1"]}', data={"evidence": [{"evidence_id": "ev-1"}]})

        def _conclude(_payload):
            return ToolResult(
                observation='{"submission_id":"submission_cycle_1"}',
                data={"__conclude_valid__": True, "submission": {"submission_id": "submission_cycle_1"}},
            )

        download_tool = _InvokeTool(_download)
        parse_tool = _InvokeTool(_parse)
        extract_tool = _InvokeTool(_extract)
        conclude_tool = _InvokeTool(_conclude)

        with patch(
            "agents.react_agent._lazy_langchain_imports",
            return_value=(object, _SystemMessage, _HumanMessage, _AIMessage, _ToolMessage, object),
        ):
            agent = ReActAgent(
                agent_id="t9",
                name="test",
                model_config={"deadline_mode": True},
                rag_system=None,
                experience_store=None,
                system_prompt="",
                max_react_steps=10,
                verbose=False,
                tools=[download_tool, parse_tool, extract_tool, conclude_tool],
                search_tool_names=["download_document", "parse_document", "extract_evidence"],
                analysis_tool_names=["conclude"],
                conclude_argument_name="submission",
                conclude_output_kind="submission",
            )

            with patch.object(agent, "_get_llm", return_value=llm), patch.object(
                agent,
                "_build_tools",
                return_value=(
                    [download_tool, parse_tool, extract_tool, conclude_tool],
                    {
                        "download_document": download_tool,
                        "parse_document": parse_tool,
                        "extract_evidence": extract_tool,
                        "conclude": conclude_tool,
                    },
                ),
            ):
                response, trajectory = agent.generate_response_with_react(
                    query="Return a structured submission.",
                    max_steps_override=2,
                )

        self.assertEqual(["download_document", "parse_document", "extract_evidence"], executed)
        self.assertEqual({"kind": "submission", "payload": {"submission_id": "submission_cycle_1"}}, response.structured_output)
        self.assertEqual("conclude", trajectory.steps[-1].action)

    def test_structured_deadline_last_chance_downloads_after_search_before_forced_conclude(self):
        from agents.react_agent import ReActAgent, ToolResult

        backend = _DeadlineSearchOnlyBackend()
        llm = _ReactiveLLM(backend)
        executed: list[str] = []

        def _search(_payload):
            executed.append("search_papers")
            return ToolResult(
                observation='{"paper_ids":["paper-1"]}',
                data={"paper_ids": ["paper-1"], "papers": [{"paper_id": "paper-1"}]},
            )

        def _download(_payload):
            executed.append("download_document")
            return ToolResult(observation='{"paper_id":"paper-1"}', data={"paper_id": "paper-1"})

        def _parse(_payload):
            executed.append("parse_document")
            return ToolResult(
                observation='{"paper_id":"paper-1","fulltext_status":"fulltext_indexed"}',
                data={"paper_id": "paper-1", "fulltext_status": "fulltext_indexed"},
            )

        def _extract(_payload):
            executed.append("extract_evidence")
            return ToolResult(observation='{"evidence":["ev-1"]}', data={"evidence": [{"evidence_id": "ev-1"}]})

        def _conclude(_payload):
            return ToolResult(
                observation='{"submission_id":"submission_cycle_1"}',
                data={"__conclude_valid__": True, "submission": {"submission_id": "submission_cycle_1"}},
            )

        search_tool = _InvokeTool(_search)
        download_tool = _InvokeTool(_download)
        parse_tool = _InvokeTool(_parse)
        extract_tool = _InvokeTool(_extract)
        conclude_tool = _InvokeTool(_conclude)

        with patch(
            "agents.react_agent._lazy_langchain_imports",
            return_value=(object, _SystemMessage, _HumanMessage, _AIMessage, _ToolMessage, object),
        ):
            agent = ReActAgent(
                agent_id="t10",
                name="test",
                model_config={"deadline_mode": True},
                rag_system=None,
                experience_store=None,
                system_prompt="",
                max_react_steps=10,
                verbose=False,
                tools=[search_tool, download_tool, parse_tool, extract_tool, conclude_tool],
                search_tool_names=["search_papers", "download_document", "parse_document", "extract_evidence"],
                analysis_tool_names=["conclude"],
                conclude_argument_name="submission",
                conclude_output_kind="submission",
            )

            with patch.object(agent, "_get_llm", return_value=llm), patch.object(
                agent,
                "_build_tools",
                return_value=(
                    [search_tool, download_tool, parse_tool, extract_tool, conclude_tool],
                    {
                        "search_papers": search_tool,
                        "download_document": download_tool,
                        "parse_document": parse_tool,
                        "extract_evidence": extract_tool,
                        "conclude": conclude_tool,
                    },
                ),
            ):
                response, trajectory = agent.generate_response_with_react(
                    query="Return a structured submission.",
                    max_steps_override=3,
                )

        self.assertEqual(
            ["search_papers", "download_document", "parse_document", "extract_evidence"],
            executed,
        )
        self.assertEqual({"kind": "submission", "payload": {"submission_id": "submission_cycle_1"}}, response.structured_output)
        self.assertEqual("conclude", trajectory.steps[-1].action)

    def test_structured_deadline_allows_repeat_search_until_candidate_target_is_reached(self):
        from agents.react_agent import ReActAgent, ToolResult

        backend = _DeadlineSearchOnlyBackend()
        llm = _ReactiveLLM(backend)
        executed: list[str] = []
        search_call_count = 0

        def _search(_payload):
            nonlocal search_call_count
            search_call_count += 1
            executed.append("search_papers")
            return ToolResult(
                observation=f'{{"paper_ids":["paper-{search_call_count}"]}}',
                data={
                    "paper_ids": [f"paper-{search_call_count}"],
                    "papers": [
                        {
                            "paper_id": f"paper-{search_call_count}",
                            "doi": f"10.1000/test-{search_call_count}",
                            "oa_eligible": True,
                            "open_access_pdf_url": f"https://example.org/paper-{search_call_count}.pdf",
                            "oa_url": f"https://example.org/paper-{search_call_count}.pdf",
                        }
                    ],
                },
            )

        def _download(_payload):
            executed.append("download_document")
            return ToolResult(observation='{"paper_id":"paper-1"}', data={"paper_id": "paper-1"})

        def _parse(_payload):
            executed.append("parse_document")
            return ToolResult(
                observation='{"paper_id":"paper-1","fulltext_status":"fulltext_indexed"}',
                data={"paper_id": "paper-1", "fulltext_status": "fulltext_indexed"},
            )

        def _extract(_payload):
            executed.append("extract_evidence")
            return ToolResult(observation='{"evidence":["ev-1"]}', data={"evidence": [{"evidence_id": "ev-1"}]})

        def _conclude(_payload):
            return ToolResult(
                observation='{"submission_id":"submission_cycle_1"}',
                data={"__conclude_valid__": True, "submission": {"submission_id": "submission_cycle_1"}},
            )

        search_tool = _InvokeTool(_search)
        download_tool = _InvokeTool(_download)
        parse_tool = _InvokeTool(_parse)
        extract_tool = _InvokeTool(_extract)
        conclude_tool = _InvokeTool(_conclude)

        with patch(
            "agents.react_agent._lazy_langchain_imports",
            return_value=(object, _SystemMessage, _HumanMessage, _AIMessage, _ToolMessage, object),
        ):
            agent = ReActAgent(
                agent_id="t10a",
                name="test",
                model_config={"deadline_mode": True},
                rag_system=None,
                experience_store=None,
                system_prompt="Proposer candidate target: 2 cumulative strict-PDF candidates within the current cycle.",
                max_react_steps=10,
                verbose=False,
                tools=[search_tool, download_tool, parse_tool, extract_tool, conclude_tool],
                search_tool_names=["search_papers", "download_document", "parse_document", "extract_evidence"],
                analysis_tool_names=["conclude"],
                conclude_argument_name="submission",
                conclude_output_kind="submission",
            )

            with patch.object(agent, "_get_llm", return_value=llm), patch.object(
                agent,
                "_build_tools",
                return_value=(
                    [search_tool, download_tool, parse_tool, extract_tool, conclude_tool],
                    {
                        "search_papers": search_tool,
                        "download_document": download_tool,
                        "parse_document": parse_tool,
                        "extract_evidence": extract_tool,
                        "conclude": conclude_tool,
                    },
                ),
            ):
                response, trajectory = agent.generate_response_with_react(
                    query="Return a structured submission.",
                    max_steps_override=4,
                )

        self.assertEqual(["search_papers", "search_papers"], executed[:2])
        self.assertIn("download_document", executed)
        self.assertIn("parse_document", executed)
        self.assertIn("extract_evidence", executed)
        self.assertEqual(2, search_call_count)
        self.assertEqual({"kind": "submission", "payload": {"submission_id": "submission_cycle_1"}}, response.structured_output)
        self.assertEqual("conclude", trajectory.steps[-1].action)

    def test_structured_deadline_blocks_repeat_search_after_candidate_target_is_reached(self):
        from agents.react_agent import ReActAgent, ToolResult

        backend = _DeadlineSearchOnlyBackend()
        llm = _ReactiveLLM(backend)
        executed: list[str] = []

        def _search(_payload):
            executed.append("search_papers")
            return ToolResult(
                observation='{"paper_ids":["paper-1"]}',
                data={
                    "paper_ids": ["paper-1"],
                    "papers": [
                        {
                            "paper_id": "paper-1",
                            "doi": "10.1000/test-1",
                            "open_access_pdf_url": "https://example.org/paper-1.pdf",
                            "oa_url": "https://example.org/paper-1.pdf",
                        }
                    ],
                },
            )

        def _download(_payload):
            executed.append("download_document")
            return ToolResult(observation='{"paper_id":"paper-1"}', data={"paper_id": "paper-1"})

        def _parse(_payload):
            executed.append("parse_document")
            return ToolResult(
                observation='{"paper_id":"paper-1","fulltext_status":"fulltext_indexed"}',
                data={"paper_id": "paper-1", "fulltext_status": "fulltext_indexed"},
            )

        def _extract(_payload):
            executed.append("extract_evidence")
            return ToolResult(observation='{"evidence":["ev-1"]}', data={"evidence": [{"evidence_id": "ev-1"}]})

        def _conclude(_payload):
            return ToolResult(
                observation='{"submission_id":"submission_cycle_1"}',
                data={"__conclude_valid__": True, "submission": {"submission_id": "submission_cycle_1"}},
            )

        search_tool = _InvokeTool(_search)
        download_tool = _InvokeTool(_download)
        parse_tool = _InvokeTool(_parse)
        extract_tool = _InvokeTool(_extract)
        conclude_tool = _InvokeTool(_conclude)

        with patch(
            "agents.react_agent._lazy_langchain_imports",
            return_value=(object, _SystemMessage, _HumanMessage, _AIMessage, _ToolMessage, object),
        ):
            agent = ReActAgent(
                agent_id="t10b",
                name="test",
                model_config={"deadline_mode": True},
                rag_system=None,
                experience_store=None,
                system_prompt="Proposer candidate target: 1 cumulative strict-PDF candidates within the current cycle.",
                max_react_steps=10,
                verbose=False,
                tools=[search_tool, download_tool, parse_tool, extract_tool, conclude_tool],
                search_tool_names=["search_papers", "download_document", "parse_document", "extract_evidence"],
                analysis_tool_names=["conclude"],
                conclude_argument_name="submission",
                conclude_output_kind="submission",
            )

            with patch.object(agent, "_get_llm", return_value=llm), patch.object(
                agent,
                "_build_tools",
                return_value=(
                    [search_tool, download_tool, parse_tool, extract_tool, conclude_tool],
                    {
                        "search_papers": search_tool,
                        "download_document": download_tool,
                        "parse_document": parse_tool,
                        "extract_evidence": extract_tool,
                        "conclude": conclude_tool,
                    },
                ),
            ):
                response, trajectory = agent.generate_response_with_react(
                    query="Return a structured submission.",
                    max_steps_override=4,
                )

        self.assertEqual(
            ["search_papers", "download_document", "parse_document", "extract_evidence"],
            executed,
        )
        self.assertTrue(
            any(
                any(
                    getattr(call, "observation_data", None) == {"error": "phase_budget_requires_followup"}
                    for call in list(getattr(step, "tool_calls", []) or [])
                )
                for step in list(getattr(trajectory, "steps", []) or [])
            )
        )
        self.assertEqual({"kind": "submission", "payload": {"submission_id": "submission_cycle_1"}}, response.structured_output)
        self.assertEqual("conclude", trajectory.steps[-1].action)

    def test_phase_budget_followup_block_does_not_trigger_after_download_phase_has_started(self):
        from agents.react_agent import ReActAgent, ToolResult

        class _SearchDownloadSearchThenConcludeBackend:
            def __init__(self) -> None:
                self.action_calls = 0

            def invoke(self, messages, *, tool_choice: str | None = None):
                last_content = str(getattr(messages[-1], "content", "") or "")
                if "CURRENT PHASE: THOUGHT" in last_content:
                    return _AIMessage(content="Keep moving through retrieval steps.")
                if tool_choice == "conclude":
                    return _AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "id": "call_conclude_structured",
                                "name": "conclude",
                                "args": {"submission": {"submission_id": "submission_cycle_1"}},
                            }
                        ],
                    )
                self.action_calls += 1
                if self.action_calls == 1:
                    return _AIMessage(
                        content="",
                        tool_calls=[{"id": "call_search_1", "name": "search_papers", "args": {"query": "pt/c her"}}],
                    )
                if self.action_calls == 2:
                    return _AIMessage(
                        content="",
                        tool_calls=[{"id": "call_download_1", "name": "download_document", "args": {"paper_id": "paper-1"}}],
                    )
                if self.action_calls == 3:
                    return _AIMessage(
                        content="",
                        tool_calls=[{"id": "call_search_2", "name": "search_papers", "args": {"query": "pt/c her retry"}}],
                    )
                return _AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_conclude_structured",
                            "name": "conclude",
                            "args": {"submission": {"submission_id": "submission_cycle_1"}},
                        }
                    ],
                )

        backend = _SearchDownloadSearchThenConcludeBackend()
        llm = _ReactiveLLM(backend)
        executed: list[str] = []

        def _search(_payload):
            executed.append("search_papers")
            return ToolResult(
                observation='{"paper_ids":["paper-1"]}',
                data={
                    "paper_ids": ["paper-1"],
                    "papers": [
                        {
                            "paper_id": "paper-1",
                            "doi": "10.1000/test-1",
                            "open_access_pdf_url": "https://example.org/paper-1.pdf",
                            "oa_url": "https://example.org/paper-1.pdf",
                        }
                    ],
                },
            )

        def _download(_payload):
            executed.append("download_document")
            return ToolResult(observation='{"paper_id":"paper-1"}', data={"paper_id": "paper-1"})

        def _conclude(_payload):
            executed.append("conclude")
            return ToolResult(
                observation='{"submission_id":"submission_cycle_1"}',
                data={"__conclude_valid__": True, "submission": {"submission_id": "submission_cycle_1"}},
            )

        search_tool = _InvokeTool(_search)
        download_tool = _InvokeTool(_download)
        conclude_tool = _InvokeTool(_conclude)

        with patch(
            "agents.react_agent._lazy_langchain_imports",
            return_value=(object, _SystemMessage, _HumanMessage, _AIMessage, _ToolMessage, object),
        ):
            agent = ReActAgent(
                agent_id="t10c",
                name="test",
                model_config={"deadline_mode": False},
                rag_system=None,
                experience_store=None,
                system_prompt="Proposer candidate target: 1 cumulative strict-PDF candidates within the current cycle.",
                max_react_steps=10,
                verbose=False,
                tools=[search_tool, download_tool, conclude_tool],
                search_tool_names=["search_papers", "download_document"],
                analysis_tool_names=["conclude"],
                conclude_argument_name="submission",
                conclude_output_kind="submission",
            )

            with patch.object(agent, "_get_llm", return_value=llm), patch.object(
                agent,
                "_build_tools",
                return_value=(
                    [search_tool, download_tool, conclude_tool],
                    {
                        "search_papers": search_tool,
                        "download_document": download_tool,
                        "conclude": conclude_tool,
                    },
                ),
            ):
                response, trajectory = agent.generate_response_with_react(
                    query="Return a structured submission.",
                    max_steps_override=4,
                )

        self.assertEqual(["search_papers", "download_document", "search_papers", "conclude"], executed)
        self.assertFalse(
            any(
                any(
                    getattr(call, "observation_data", None) == {"error": "phase_budget_requires_followup"}
                    for call in list(getattr(step, "tool_calls", []) or [])
                )
                for step in list(getattr(trajectory, "steps", []) or [])
            )
        )
        self.assertEqual({"kind": "submission", "payload": {"submission_id": "submission_cycle_1"}}, response.structured_output)

    def test_structured_deadline_reuses_already_parsed_paper_without_parse_result_error(self):
        from agents.react_agent import ReActAgent, ToolResult

        backend = _DeadlineParsedPaperBackend()
        llm = _ReactiveLLM(backend)
        executed: list[str] = []

        def _parse(_payload):
            executed.append("parse_document")
            return ToolResult(
                observation='{"paper_id":"paper-1","fulltext_status":"fulltext_indexed"}',
                data={"paper_id": "paper-1", "fulltext_status": "fulltext_indexed"},
            )

        def _extract(_payload):
            executed.append("extract_evidence")
            return ToolResult(observation='{"evidence":["ev-1"]}', data={"evidence": [{"evidence_id": "ev-1"}]})

        def _conclude(_payload):
            return ToolResult(
                observation='{"submission_id":"submission_cycle_1"}',
                data={"__conclude_valid__": True, "submission": {"submission_id": "submission_cycle_1"}},
            )

        parse_tool = _InvokeTool(_parse)
        extract_tool = _InvokeTool(_extract)
        conclude_tool = _InvokeTool(_conclude)

        with patch(
            "agents.react_agent._lazy_langchain_imports",
            return_value=(object, _SystemMessage, _HumanMessage, _AIMessage, _ToolMessage, object),
        ):
            agent = ReActAgent(
                agent_id="t10",
                name="test",
                model_config={"deadline_mode": True},
                rag_system=None,
                experience_store=None,
                system_prompt="",
                max_react_steps=10,
                verbose=False,
                tools=[parse_tool, extract_tool, conclude_tool],
                search_tool_names=["parse_document", "extract_evidence"],
                analysis_tool_names=["conclude"],
                conclude_argument_name="submission",
                conclude_output_kind="submission",
            )

            with patch.object(agent, "_get_llm", return_value=llm), patch.object(
                agent,
                "_build_tools",
                return_value=(
                    [parse_tool, extract_tool, conclude_tool],
                    {
                        "parse_document": parse_tool,
                        "extract_evidence": extract_tool,
                        "conclude": conclude_tool,
                    },
                ),
            ):
                response, trajectory = agent.generate_response_with_react(
                    query="Return a structured submission.",
                    max_steps_override=2,
                )

        self.assertEqual(["parse_document", "extract_evidence"], executed)
        self.assertEqual({"kind": "submission", "payload": {"submission_id": "submission_cycle_1"}}, response.structured_output)
        self.assertEqual("conclude", trajectory.steps[-1].action)

    def test_structured_deadline_last_chance_retries_extract_with_explicit_section_ids(self):
        from agents.react_agent import ReActAgent, ToolResult

        backend = _DeadlineLastChanceExtractBackend()
        llm = _ReactiveLLM(backend)
        executed: list[str] = []

        def _download(_payload):
            executed.append("download_document")
            return ToolResult(observation='{"paper_id":"paper-1"}', data={"paper_id": "paper-1"})

        def _parse(_payload):
            executed.append("parse_document")
            return ToolResult(
                observation='{"paper_id":"paper-1","fulltext_status":"fulltext_indexed"}',
                data={"paper_id": "paper-1", "fulltext_status": "fulltext_indexed"},
            )

        def _read_sections(_payload):
            executed.append("read_sections")
            return ToolResult(
                observation='{"sections":[{"section_id":"sec-results"}]}',
                data={"sections": [{"section_id": "sec-results"}]},
            )

        def _extract(payload):
            if payload.get("preferred_sections"):
                executed.append("extract_evidence_preferred")
                return ToolResult(observation='{"evidence":[]}', data={"evidence": []})
            executed.append("extract_evidence_explicit")
            return ToolResult(
                observation='{"evidence":["ev-1"]}',
                data={"evidence": [{"evidence_id": "ev-1", "section_id": "sec-results"}]},
            )

        def _conclude(_payload):
            return ToolResult(
                observation='{"submission_id":"submission_cycle_1"}',
                data={"__conclude_valid__": True, "submission": {"submission_id": "submission_cycle_1"}},
            )

        download_tool = _InvokeTool(_download)
        parse_tool = _InvokeTool(_parse)
        read_sections_tool = _InvokeTool(_read_sections)
        extract_tool = _InvokeTool(_extract)
        conclude_tool = _InvokeTool(_conclude)

        with patch(
            "agents.react_agent._lazy_langchain_imports",
            return_value=(object, _SystemMessage, _HumanMessage, _AIMessage, _ToolMessage, object),
        ):
            agent = ReActAgent(
                agent_id="t11",
                name="test",
                model_config={"deadline_mode": True},
                rag_system=None,
                experience_store=None,
                system_prompt="",
                max_react_steps=10,
                verbose=False,
                tools=[download_tool, parse_tool, read_sections_tool, extract_tool, conclude_tool],
                search_tool_names=["download_document", "parse_document", "read_sections", "extract_evidence"],
                analysis_tool_names=["conclude"],
                conclude_argument_name="submission",
                conclude_output_kind="submission",
            )

            with patch.object(agent, "_get_llm", return_value=llm), patch.object(
                agent,
                "_build_tools",
                return_value=(
                    [download_tool, parse_tool, read_sections_tool, extract_tool, conclude_tool],
                    {
                        "download_document": download_tool,
                        "parse_document": parse_tool,
                        "read_sections": read_sections_tool,
                        "extract_evidence": extract_tool,
                        "conclude": conclude_tool,
                    },
                ),
            ):
                response, trajectory = agent.generate_response_with_react(
                    query="Return a structured submission.",
                    max_steps_override=2,
                )

        self.assertEqual(
            [
                "download_document",
                "parse_document",
                "read_sections",
                "extract_evidence_preferred",
                "extract_evidence_explicit",
            ],
            executed,
        )
        self.assertEqual({"kind": "submission", "payload": {"submission_id": "submission_cycle_1"}}, response.structured_output)
        self.assertEqual("conclude", trajectory.steps[-1].action)


if __name__ == "__main__":
    unittest.main()
