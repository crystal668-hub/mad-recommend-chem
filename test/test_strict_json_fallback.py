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


class StrictJsonFallbackTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

