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


class _ReviewerConcludeTool:
    name = "conclude"

    def invoke(self, tool_input):
        from agents.react_agent import ToolResult

        review = dict((tool_input or {}).get("review") or {})
        return ToolResult(
            observation="validated reviewer conclude",
            data={"__conclude_valid__": True, "review_items": list(review.get("review_items") or [])},
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

    def test_structured_forced_conclude_salvages_review_items_json_content(self):
        from agents.react_agent import ReActAgent

        llm = _ForcedConcludeLLM(mode="json_content_review_items")

        with patch(
            "agents.react_agent._lazy_langchain_imports",
            return_value=(object, _SystemMessage, _HumanMessage, _AIMessage, _ToolMessage, object),
        ):
            agent = ReActAgent(
                agent_id="t_struct_json_review_items",
                name="test",
                model_config={"deadline_mode": True},
                system_prompt="",
                max_react_steps=1,
                verbose=False,
                tools=[_ReviewerConcludeTool()],
                conclude_argument_name="review",
                conclude_output_kind="review_items",
            )

            with patch.object(agent, "_get_llm", return_value=llm):
                response, trajectory = agent.generate_response_with_react(
                    query="structured forced conclude test",
                    system_prompt_override="Return the final review payload via conclude.",
                    max_steps_override=1,
                )

        self.assertEqual(response.structured_output, {"kind": "review_items", "payload": [{"critique": "x"}]})
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
        self.assertEqual("", raw_payload["forced_conclude_action_response"]["content"])

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
