from __future__ import annotations

import unittest


class RAGJunkFilterTests(unittest.TestCase):
    def test_heading_only_is_junk(self):
        from agents.react_agent import _is_junk_chunk

        self.assertTrue(_is_junk_chunk("### Oxygen Reduction Reaction", min_chars=80, keep_if_has_number=True))

    def test_o2_in_heading_is_not_treated_as_metric_number(self):
        from agents.react_agent import _is_junk_chunk

        # The digit in "O2" should not count as quantitative evidence.
        self.assertTrue(_is_junk_chunk("### *Electrocatalytic O2 Reduction*", min_chars=80, keep_if_has_number=True))

    def test_heading_with_metric_number_is_not_junk(self):
        from agents.react_agent import _is_junk_chunk

        self.assertFalse(_is_junk_chunk("### 0.83 V vs. RHE", min_chars=80, keep_if_has_number=True))

    def test_keywords_line_is_hard_junk(self):
        from agents.react_agent import _is_junk_chunk

        self.assertTrue(_is_junk_chunk("#### KEYWORDS: ORR; HEA; catalyst", min_chars=80, keep_if_has_number=True))

    def test_search_literature_never_backfills_hard_junk(self):
        from agents.react_agent import ReActAgent

        class _DummyRAG:
            collection_name = "test_collection"

            def retrieve(self, query: str, top_k: int = 5, where=None):
                return [
                    {
                        "text": "### Oxygen Reduction Reaction",
                        "score": 0.99,
                        "metadata": {"doc_id": "10.1/dummy", "chunk_id": 1},
                    },
                    {
                        "text": (
                            "This is a sufficiently long chunk that contains useful electrochemical context "
                            "and a metric like 0.83 V vs. RHE under alkaline conditions, so it should be kept."
                        ),
                        "score": 0.5,
                        "metadata": {"doc_id": "10.2/dummy", "chunk_id": 2},
                    },
                ]

        agent = ReActAgent(
            agent_id="t_junk",
            name="test",
            model_config={
                "rag_filter_by_reaction_type": False,
                "rag_filter_junk_chunks": True,
                "rag_min_chunk_chars": 80,
                "rag_keep_if_has_number": True,
            },
            rag_system=_DummyRAG(),
            experience_store=None,
            system_prompt="",
            verbose=False,
        )

        result = agent._tool_search_literature(query="dummy query", top_k=2)
        data = result.data or []
        self.assertEqual(len(data), 1)
        self.assertFalse(str(data[0].get("text") or "").lstrip().startswith("#"))


if __name__ == "__main__":
    unittest.main()
