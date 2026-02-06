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


if __name__ == "__main__":
    unittest.main()

