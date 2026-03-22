from __future__ import annotations

import json
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
        super().__init__(
            content=content,
            additional_kwargs=additional_kwargs,
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
        )


class _DummyLLM:
    """
    Simulate a backend that fails to emit tool_calls and returns empty content even when forced to conclude.
    """

    def __init__(self, tool_choice: str | None = None):
        self._tool_choice = tool_choice

    def bind_tools(self, tools, tool_choice: str | None = None):
        return _DummyLLM(tool_choice=tool_choice)

    def bind(self, tools=None, tool_choice: str | None = None):  # pragma: no cover
        return self.bind_tools(tools, tool_choice=tool_choice)

    def invoke(self, messages):
        # Forced-conclude path tries tool_choice="conclude"; return empty tool_calls/content.
        if self._tool_choice == "conclude":
            return _AIMessage(content="", tool_calls=[])
        # Free-form fallback also returns empty content.
        return _AIMessage(content="")


class _ForcedConcludeLLM:
    def __init__(self, mode: str, tool_choice: str | None = None):
        self._mode = mode
        self._tool_choice = tool_choice

    def bind_tools(self, tools, tool_choice: str | None = None):
        return _ForcedConcludeLLM(mode=self._mode, tool_choice=tool_choice)

    def bind(self, tools=None, tool_choice: str | None = None):  # pragma: no cover
        return self.bind_tools(tools, tool_choice=tool_choice)

    def invoke(self, messages):
        if self._tool_choice == "conclude":
            if self._mode == "tool_calls":
                return _AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "tc_1",
                            "function": {
                                "name": "conclude",
                                "arguments": json.dumps({"submission": {"summary": "ok"}}),
                            },
                        }
                    ],
                )
            if self._mode == "function_call":
                return _AIMessage(
                    content="",
                    additional_kwargs={
                        "function_call": {
                            "name": "conclude",
                            "arguments": json.dumps({"submission": {"summary": "ok"}}),
                        }
                    },
                )
            if self._mode == "none":
                return _AIMessage(content="", tool_calls=[])
            if self._mode == "json_content_submission":
                return _AIMessage(content=json.dumps({"submission": {"summary": "ok"}}))
            if self._mode == "json_content_review_items":
                return _AIMessage(content=json.dumps({"review_items": [{"critique": "x"}]}))
            if self._mode == "no_tool_then_json_submission":
                return _AIMessage(content="", tool_calls=[])
        if self._mode == "no_tool_then_json_submission":
            return _AIMessage(content=json.dumps({"submission": {"summary": "ok"}}))
        if self._mode == "raise_then_json_submission":
            return _AIMessage(content=json.dumps({"submission": {"summary": "ok"}}))
        return _AIMessage(content="")


class _ForcedConcludeInvokeErrorLLM(_ForcedConcludeLLM):
    def bind_tools(self, tools, tool_choice: str | None = None):
        return _ForcedConcludeInvokeErrorLLM(mode=self._mode, tool_choice=tool_choice)

    def bind(self, tools=None, tool_choice: str | None = None):  # pragma: no cover
        return self.bind_tools(tools, tool_choice=tool_choice)

    def invoke(self, messages):
        if self._tool_choice == "conclude":
            raise RuntimeError("forced conclude provider error")
        return super().invoke(messages)


class _ConcludeTool:
    name = "conclude"

    def invoke(self, tool_input):
        from agents.react_agent import ToolResult

        submission = dict((tool_input or {}).get("submission") or {})
        return ToolResult(
            observation="validated conclude",
            data={"__conclude_valid__": True, "submission": submission},
        )


class _InvalidConcludeTool:
    name = "conclude"

    def invoke(self, tool_input):
        from agents.react_agent import ToolResult

        return ToolResult(
            observation="invalid conclude",
            data={"__conclude_valid__": False, "submission": dict((tool_input or {}).get("submission") or {})},
        )


class StrictJsonFallbackTests(unittest.TestCase):
    def test_tool_balanced_history_truncates_dangling_tool_call_turn(self):
        from agents.react_agent import _build_tool_balanced_history

        messages = [
            _SystemMessage(content="sys"),
            _HumanMessage(content="question"),
            _AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_dangling",
                        "function": {"name": "search_literature", "arguments": json.dumps({"query": "her"})},
                    }
                ],
            ),
            _HumanMessage(content="should be trimmed with dangling turn"),
        ]

        sanitized, audit = _build_tool_balanced_history(messages)

        self.assertEqual(2, len(sanitized))
        self.assertEqual(["call_dangling"], audit["dangling_tool_call_ids"])
        self.assertEqual(2, audit["truncated_from_index"])
        self.assertTrue(audit["history_rewrite_applied"])

    def test_tool_balanced_history_drops_orphan_tool_messages(self):
        from agents.react_agent import _build_tool_balanced_history

        messages = [
            _SystemMessage(content="sys"),
            _HumanMessage(content="question"),
            _ToolMessage(content="orphan", tool_call_id="ghost_call"),
            _AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_ok",
                        "function": {"name": "search_literature", "arguments": json.dumps({"query": "her"})},
                    }
                ],
            ),
            _ToolMessage(content="ok", tool_call_id="call_ok"),
        ]

        sanitized, audit = _build_tool_balanced_history(messages)

        self.assertEqual(4, len(sanitized))
        self.assertEqual(["ghost_call"], audit["orphan_tool_message_ids"])
        self.assertEqual([2], audit["orphan_tool_message_indexes"])
        self.assertEqual([], audit["dangling_tool_call_ids"])
        self.assertIsNone(audit["truncated_from_index"])
        self.assertTrue(audit["history_rewrite_applied"])

    def test_propose_strict_json_fallback_emits_minimal_schema_json(self):
        from agents.react_agent import ReActAgent
        from prompts.debate_phase_prompts import DEBATE_PROPOSE_SYSTEM_PROMPT

        dummy_llm = _DummyLLM()

        with patch(
            "agents.react_agent._lazy_langchain_imports",
            return_value=(object, _SystemMessage, _HumanMessage, _AIMessage, _ToolMessage, object),
        ):
            agent = ReActAgent(
                agent_id="t_prop",
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
                    query="STRICT JSON fallback test (propose).",
                    components=["Ni", "Fe", "Co"],
                    system_prompt_override=DEBATE_PROPOSE_SYSTEM_PROMPT,
                    max_steps_override=1,
                )

        parsed = json.loads(response.content)
        self.assertIsInstance(parsed, dict)
        self.assertIn("reaction_type", parsed)
        self.assertIn("electrode_composition", parsed)
        self.assertIn("catalyst_metal_elements", parsed)
        self.assertIn("performance_metrics", parsed)
        self.assertIn("confidence", parsed)
        self.assertIn("evidence", parsed)
        self.assertIsInstance(parsed["evidence"], list)
        self.assertEqual(len(getattr(trajectory, "steps", []) or []), 1)
        self.assertIn("Strict JSON fallback", getattr(trajectory.steps[0], "thought", "") or "")

    def test_review_strict_json_fallback_emits_minimal_schema_json(self):
        from agents.react_agent import ReActAgent
        from prompts.debate_phase_prompts import DEBATE_REVIEW_SYSTEM_PROMPT

        dummy_llm = _DummyLLM()

        with patch(
            "agents.react_agent._lazy_langchain_imports",
            return_value=(object, _SystemMessage, _HumanMessage, _AIMessage, _ToolMessage, object),
        ):
            agent = ReActAgent(
                agent_id="t_review",
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
                    query="STRICT JSON fallback test (review).",
                    components=["Ni", "Fe", "Co"],
                    system_prompt_override=DEBATE_REVIEW_SYSTEM_PROMPT,
                    max_steps_override=1,
                )

        parsed = json.loads(response.content)
        self.assertIsInstance(parsed, dict)
        self.assertIn("reviews", parsed)
        self.assertIsInstance(parsed["reviews"], list)
        self.assertEqual(len(getattr(trajectory, "steps", []) or []), 1)
        self.assertIn("Strict JSON fallback", getattr(trajectory.steps[0], "thought", "") or "")

    def test_rebuttal_strict_json_fallback_emits_minimal_schema_json(self):
        from agents.react_agent import ReActAgent
        from prompts.debate_phase_prompts import DEBATE_REBUTTAL_SYSTEM_PROMPT

        dummy_llm = _DummyLLM()

        with patch(
            "agents.react_agent._lazy_langchain_imports",
            return_value=(object, _SystemMessage, _HumanMessage, _AIMessage, _ToolMessage, object),
        ):
            agent = ReActAgent(
                agent_id="t_reb",
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
                    query="STRICT JSON fallback test (rebuttal).",
                    components=["Ni", "Fe", "Co"],
                    system_prompt_override=DEBATE_REBUTTAL_SYSTEM_PROMPT,
                    max_steps_override=1,
                )

        parsed = json.loads(response.content)
        self.assertIsInstance(parsed, dict)
        self.assertIn("rebuttals", parsed)
        self.assertIsInstance(parsed["rebuttals"], list)
        self.assertIn("revised_claim", parsed)
        self.assertEqual(parsed["revised_claim"], None)
        self.assertEqual(len(getattr(trajectory, "steps", []) or []), 1)
        self.assertIn("Strict JSON fallback", getattr(trajectory.steps[0], "thought", "") or "")

    def test_structured_forced_conclude_accepts_tool_calls_shape(self):
        from agents.react_agent import ReActAgent

        llm = _ForcedConcludeLLM(mode="tool_calls")

        with patch(
            "agents.react_agent._lazy_langchain_imports",
            return_value=(object, _SystemMessage, _HumanMessage, _AIMessage, _ToolMessage, object),
        ):
            agent = ReActAgent(
                agent_id="t_struct_tool_calls",
                name="test",
                model_config={"deadline_mode": True},
                system_prompt="",
                max_react_steps=1,
                verbose=False,
                tools=[_ConcludeTool()],
                conclude_argument_name="submission",
                conclude_output_kind="submission",
            )

            with patch.object(agent, "_get_llm", return_value=llm):
                response, trajectory = agent.generate_response_with_react(
                    query="structured forced conclude test",
                    system_prompt_override="Return the final answer via conclude.",
                    max_steps_override=1,
                )

        self.assertEqual(response.structured_output, {"kind": "submission", "payload": {"summary": "ok"}})
        self.assertEqual(len(getattr(trajectory, "steps", []) or []), 1)
        self.assertEqual(getattr(trajectory.steps[0], "action", None), "conclude")

    def test_structured_forced_conclude_accepts_function_call_shape(self):
        from agents.react_agent import ReActAgent

        llm = _ForcedConcludeLLM(mode="function_call")

        with patch(
            "agents.react_agent._lazy_langchain_imports",
            return_value=(object, _SystemMessage, _HumanMessage, _AIMessage, _ToolMessage, object),
        ):
            agent = ReActAgent(
                agent_id="t_struct_function_call",
                name="test",
                model_config={"deadline_mode": True},
                system_prompt="",
                max_react_steps=1,
                verbose=False,
                tools=[_ConcludeTool()],
                conclude_argument_name="submission",
                conclude_output_kind="submission",
            )

            with patch.object(agent, "_get_llm", return_value=llm):
                response, trajectory = agent.generate_response_with_react(
                    query="structured forced conclude test",
                    system_prompt_override="Return the final answer via conclude.",
                    max_steps_override=1,
                )

        self.assertEqual(response.structured_output, {"kind": "submission", "payload": {"summary": "ok"}})
        self.assertEqual(len(getattr(trajectory, "steps", []) or []), 1)
        self.assertEqual(getattr(trajectory.steps[0], "action", None), "conclude")

    def test_structured_forced_conclude_salvages_submission_json_content(self):
        from agents.react_agent import ReActAgent

        llm = _ForcedConcludeLLM(mode="json_content_submission")

        with patch(
            "agents.react_agent._lazy_langchain_imports",
            return_value=(object, _SystemMessage, _HumanMessage, _AIMessage, _ToolMessage, object),
        ):
            agent = ReActAgent(
                agent_id="t_struct_json_submission",
                name="test",
                model_config={"deadline_mode": True},
                system_prompt="",
                max_react_steps=1,
                verbose=False,
                tools=[_ConcludeTool()],
                conclude_argument_name="submission",
                conclude_output_kind="submission",
            )

            with patch.object(agent, "_get_llm", return_value=llm):
                response, trajectory = agent.generate_response_with_react(
                    query="structured forced conclude test",
                    system_prompt_override="Return the final answer via conclude.",
                    max_steps_override=1,
                )

        self.assertEqual(response.structured_output, {"kind": "submission", "payload": {"summary": "ok"}})
        self.assertEqual(len(getattr(trajectory, "steps", []) or []), 1)
        self.assertEqual(getattr(trajectory.steps[0], "action", None), "conclude")

    def test_structured_forced_conclude_uses_second_json_only_pass(self):
        from agents.react_agent import ReActAgent

        llm = _ForcedConcludeLLM(mode="no_tool_then_json_submission")

        with patch(
            "agents.react_agent._lazy_langchain_imports",
            return_value=(object, _SystemMessage, _HumanMessage, _AIMessage, _ToolMessage, object),
        ):
            agent = ReActAgent(
                agent_id="t_struct_second_pass",
                name="test",
                model_config={"deadline_mode": True},
                system_prompt="",
                max_react_steps=1,
                verbose=False,
                tools=[_ConcludeTool()],
                conclude_argument_name="submission",
                conclude_output_kind="submission",
            )

            with patch.object(agent, "_get_llm", return_value=llm):
                response, trajectory = agent.generate_response_with_react(
                    query="structured forced conclude test",
                    system_prompt_override="Return the final answer via conclude.",
                    max_steps_override=1,
                )

        self.assertEqual(response.structured_output, {"kind": "submission", "payload": {"summary": "ok"}})
        self.assertEqual(len(getattr(trajectory, "steps", []) or []), 1)
        self.assertEqual(getattr(trajectory.steps[0], "action", None), "conclude")

    def test_structured_forced_conclude_uses_second_json_only_pass_after_action_error(self):
        from agents.react_agent import ReActAgent

        llm = _ForcedConcludeInvokeErrorLLM(mode="raise_then_json_submission")

        with patch(
            "agents.react_agent._lazy_langchain_imports",
            return_value=(object, _SystemMessage, _HumanMessage, _AIMessage, _ToolMessage, object),
        ):
            agent = ReActAgent(
                agent_id="t_struct_second_pass_after_error",
                name="test",
                model_config={"deadline_mode": True},
                system_prompt="",
                max_react_steps=1,
                verbose=False,
                tools=[_ConcludeTool()],
                conclude_argument_name="submission",
                conclude_output_kind="submission",
            )

            with patch.object(agent, "_get_llm", return_value=llm):
                response, trajectory = agent.generate_response_with_react(
                    query="structured forced conclude test",
                    system_prompt_override="Return the final answer via conclude.",
                    max_steps_override=1,
                )

        self.assertEqual(response.structured_output, {"kind": "submission", "payload": {"summary": "ok"}})
        self.assertEqual(len(getattr(trajectory, "steps", []) or []), 1)
        self.assertEqual(getattr(trajectory.steps[0], "action", None), "conclude")
        raw_payload = getattr(response, "response_content", None)
        self.assertIsInstance(raw_payload, dict)
        self.assertIsNone(raw_payload["forced_conclude_action_response"])
        self.assertEqual(
            {"error_type": "RuntimeError", "message": "forced conclude provider error"},
            raw_payload["forced_conclude_action_error"],
        )
        self.assertEqual(
            json.dumps({"submission": {"summary": "ok"}}),
            raw_payload["forced_conclude_structured_json_response"]["content"],
        )

    def test_structured_forced_conclude_still_fails_without_recognized_conclude(self):
        from agents.react_agent import ReActAgent

        llm = _ForcedConcludeLLM(mode="none")

        with patch(
            "agents.react_agent._lazy_langchain_imports",
            return_value=(object, _SystemMessage, _HumanMessage, _AIMessage, _ToolMessage, object),
        ):
            agent = ReActAgent(
                agent_id="t_struct_none",
                name="test",
                model_config={"deadline_mode": True},
                system_prompt="",
                max_react_steps=1,
                verbose=False,
                tools=[_ConcludeTool()],
                conclude_argument_name="submission",
                conclude_output_kind="submission",
            )

            with patch.object(agent, "_get_llm", return_value=llm):
                with self.assertRaisesRegex(RuntimeError, "recognized `conclude` tool call"):
                    agent.generate_response_with_react(
                        query="structured forced conclude test",
                        system_prompt_override="Return the final answer via conclude.",
                        max_steps_override=1,
                    )

    def test_structured_forced_conclude_invalid_tool_result_returns_repairable_response(self):
        from agents.react_agent import ReActAgent

        llm = _ForcedConcludeLLM(mode="tool_calls")

        with patch(
            "agents.react_agent._lazy_langchain_imports",
            return_value=(object, _SystemMessage, _HumanMessage, _AIMessage, _ToolMessage, object),
        ):
            agent = ReActAgent(
                agent_id="t_struct_invalid_result",
                name="test",
                model_config={"deadline_mode": True},
                system_prompt="",
                max_react_steps=1,
                verbose=False,
                tools=[_InvalidConcludeTool()],
                conclude_argument_name="submission",
                conclude_output_kind="submission",
            )

            with patch.object(agent, "_get_llm", return_value=llm):
                response, trajectory = agent.generate_response_with_react(
                    query="structured forced conclude test",
                    system_prompt_override="Return the final answer via conclude.",
                    max_steps_override=1,
                )

        self.assertIsNone(response.structured_output)
        self.assertEqual("invalid conclude", response.content)
        self.assertEqual(1, len(getattr(trajectory, "steps", []) or []))
        raw_payload = getattr(response, "response_content", None)
        self.assertIsInstance(raw_payload, dict)
        self.assertEqual(
            "",
            raw_payload["forced_conclude_action_response"]["content"],
        )

    def test_structured_forced_conclude_invalid_tool_arguments_return_repairable_response(self):
        from langchain_core.tools import StructuredTool

        from agents import react_tool_schemas as tool_schemas
        from agents.react_agent import ReActAgent, ToolResult

        llm = _ForcedConcludeLLM(mode="tool_calls")

        def conclude(submission):
            """Validate proposer conclude payload."""
            if hasattr(submission, "model_dump"):
                submission = submission.model_dump(exclude_none=True)
            return ToolResult(
                observation="validated conclude",
                data={"__conclude_valid__": True, "submission": submission},
            )

        tool = StructuredTool.from_function(
            conclude,
            name="conclude",
            args_schema=tool_schemas.ProposerConcludeToolInput,
        )

        with patch(
            "agents.react_agent._lazy_langchain_imports",
            return_value=(object, _SystemMessage, _HumanMessage, _AIMessage, _ToolMessage, object),
        ):
            agent = ReActAgent(
                agent_id="t_struct_invalid_args",
                name="test",
                model_config={"deadline_mode": True},
                system_prompt="",
                max_react_steps=1,
                verbose=False,
                tools=[tool],
                conclude_argument_name="submission",
                conclude_output_kind="submission",
            )

            with patch.object(agent, "_get_llm", return_value=llm):
                response, trajectory = agent.generate_response_with_react(
                    query="structured forced conclude test",
                    system_prompt_override="Return the final answer via conclude.",
                    max_steps_override=1,
                )

        self.assertIsNone(response.structured_output)
        self.assertIn("Invalid tool arguments", response.content)
        self.assertEqual("conclude", getattr(trajectory.steps[0], "action", None))
        raw_payload = getattr(response, "response_content", None)
        self.assertIsInstance(raw_payload, dict)
        self.assertEqual("", raw_payload["forced_conclude_action_response"]["content"])


if __name__ == "__main__":
    unittest.main()

