from __future__ import annotations

import unittest


class _DummyRAG:
    collection_name = "test_collection"

    def __init__(self):
        self.last_query = None
        self.last_top_k = None
        self.last_where = None

    def retrieve(self, query: str, top_k: int = 5, where=None):
        self.last_query = query
        self.last_top_k = top_k
        self.last_where = where
        return [
            {
                "text": (
                    "### Oxygen Evolution Reaction\n"
                    "This is a sufficiently long dummy chunk about oxygen evolution reaction (OER), "
                    "included to test reaction_type hard filtering."
                ),
                "score": 0.95,
                "metadata": {"doc_id": "10.1234/dummy", "chunk_id": 1, "reaction_type": "OER"},
            },
            {
                "text": (
                    "This is a sufficiently long dummy chunk about oxygen reduction reaction (ORR) "
                    "showing typical metrics and considerations for catalysts in alkaline media."
                ),
                "score": 0.9,
                "metadata": {"doc_id": "10.1234/dummy", "chunk_id": 2, "reaction_type": "ORR"},
            }
        ]


class RAGReactionFilterTests(unittest.TestCase):
    def test_reaction_type_where_filter_is_passed_to_rag_adapter(self):
        from agents.react_agent import ReActAgent
        from agents.react_reasoning import ReActTrajectory

        rag = _DummyRAG()
        agent = ReActAgent(
            agent_id="t2",
            name="test",
            model_config={
                "rag_filter_by_reaction_type": True,
                "rag_filter_junk_chunks": False,  # irrelevant for this test
            },
            rag_system=rag,
            experience_store=None,
            system_prompt="",
            verbose=False,
        )

        agent.current_trajectory = ReActTrajectory(
            query="Reaction Type: ORR\nMetal catalyst elements: Pt, Cu, Ni, Fe, Co"
        )

        out = agent._tool_search_literature(query="orr activity", top_k=1)
        self.assertEqual(rag.last_where, {"reaction_type": "ORR"})
        data = out.data or []
        self.assertEqual(len(data), 1)
        self.assertEqual(((data[0].get("metadata") or {}).get("reaction_type") or "").upper(), "ORR")


if __name__ == "__main__":
    unittest.main()
