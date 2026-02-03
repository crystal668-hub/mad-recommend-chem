import unittest

from agents.react_agent import _normalize_tool_call


class ToolCallParsingTests(unittest.TestCase):
    def test_normalize_langchain_style(self):
        name, args, call_id = _normalize_tool_call(
            {"id": "call_1", "name": "search_rag", "args": {"query": "abc", "top_k": 3}}
        )
        self.assertEqual(name, "search_rag")
        self.assertEqual(args["query"], "abc")
        self.assertEqual(args["top_k"], 3)
        self.assertEqual(call_id, "call_1")

    def test_normalize_openai_style_function_arguments(self):
        name, args, call_id = _normalize_tool_call(
            {
                "id": "call_2",
                "function": {"name": "conclude", "arguments": "{\"conclusion\": \"ok\"}"},
            }
        )
        self.assertEqual(name, "conclude")
        self.assertEqual(args["conclusion"], "ok")
        self.assertEqual(call_id, "call_2")

    def test_unknown_shape(self):
        name, args, call_id = _normalize_tool_call("weird")
        self.assertEqual(name, "unknown_tool")
        self.assertEqual(args, {})
        self.assertTrue(call_id.startswith("tool_"))


if __name__ == "__main__":
    unittest.main()

