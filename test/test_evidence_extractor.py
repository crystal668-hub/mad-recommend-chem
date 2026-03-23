import unittest

from qa.evidence import (
    EvidenceExtractor,
    MAX_LLM_SNIPPETS_PER_FULLTEXT_SECTION,
    MAX_SNIPPETS_PER_FULLTEXT_SECTION,
)
from qa.retrieval_state import PaperRecord, SectionTextView
from qa.state import EntityPack, TaskSpec


def _task_spec() -> TaskSpec:
    return TaskSpec.model_validate(
        {
            "question": "Does Pt/C improve HER activity in alkaline media?",
            "normalized_question": "does pt/c improve her activity in alkaline media?",
            "question_type": "fact",
            "recency_policy": "none",
            "answer_sections": [
                {
                    "section_id": "direct_answer",
                    "title": "Direct Answer",
                    "required": True,
                    "instruction": "Answer directly.",
                }
            ],
            "router_confidence": 0.9,
        }
    )


class EvidenceExtractorPerformanceTests(unittest.TestCase):
    def test_fulltext_section_caps_llm_classification_and_snippet_count(self):
        extractor = EvidenceExtractor(llm=object())
        call_counter = {"count": 0}

        def _fake_classify_with_llm(*, snippet, task_spec, section_type):
            del snippet, task_spec, section_type
            call_counter["count"] += 1
            return None

        extractor._classify_with_llm = _fake_classify_with_llm  # type: ignore[method-assign]
        paper_record = PaperRecord.model_validate(
            {
                "paper_id": "paper-1",
                "title": "Pt/C HER fulltext",
                "fulltext_available": True,
                "fulltext_status": "fulltext_indexed",
            }
        )
        section_view = SectionTextView.model_validate(
            {
                "paper_id": "paper-1",
                "section_id": "sec_results",
                "section_type": "results",
                "heading": "Results",
                "text": " ".join(
                    f"Sentence {index} improved HER activity by {index + 10} mV in 1 M KOH."
                    for index in range(40)
                ),
                "fulltext_char_start": 0,
                "fulltext_char_end": 4000,
            }
        )

        evidence_items = extractor._extract_from_section(
            task_spec=_task_spec(),
            entity_pack=EntityPack.model_validate({}),
            paper_record=paper_record,
            section_view=section_view,
        )

        self.assertEqual(MAX_LLM_SNIPPETS_PER_FULLTEXT_SECTION, call_counter["count"])
        observation_count = len([item for item in evidence_items if item.role == "observation"])
        self.assertLessEqual(observation_count, MAX_SNIPPETS_PER_FULLTEXT_SECTION)

    def test_extract_from_section_handles_structured_llm_payload_lists(self):
        extractor = EvidenceExtractor(llm=object())

        def _fake_classify_with_llm(*, snippet, task_spec, section_type):
            del snippet, task_spec, section_type
            return {
                "roles": [{"role": "observation"}, {"role": "mechanism"}, {"role": {"bad": "shape"}}],
                "claim_polarity": "Support",
                "entity_mentions": [
                    {"entity": "Pt/C", "type": "catalyst"},
                    {"mention": "alkaline HER"},
                ],
                "metric_mentions": [
                    {"metric": "overpotential"},
                    {"family": "activity"},
                ],
                "notes": "Structured payload",
            }

        extractor._classify_with_llm = _fake_classify_with_llm  # type: ignore[method-assign]
        paper_record = PaperRecord.model_validate(
            {
                "paper_id": "paper-2",
                "title": "Pt/C HER structured payload",
                "abstract": "Pt/C improved HER activity by 35 mV in 1 M KOH.",
                "fulltext_available": True,
                "fulltext_status": "fulltext_indexed",
            }
        )
        section_view = SectionTextView.model_validate(
            {
                "paper_id": "paper-2",
                "section_id": "sec_results",
                "section_type": "results",
                "heading": "Results",
                "text": "Pt/C improved HER activity by 35 mV in 1 M KOH because defects altered water adsorption.",
                "fulltext_char_start": 0,
                "fulltext_char_end": 200,
            }
        )

        evidence_items = extractor._extract_from_section(
            task_spec=_task_spec(),
            entity_pack=EntityPack.model_validate({}),
            paper_record=paper_record,
            section_view=section_view,
        )

        observation_items = [item for item in evidence_items if item.role == "observation"]
        mechanism_items = [item for item in evidence_items if item.role == "mechanism"]
        self.assertTrue(observation_items)
        self.assertTrue(mechanism_items)
        self.assertEqual("support", observation_items[0].claim_polarity)
        self.assertIn("Pt/C", observation_items[0].entity_mentions)
        self.assertIn("alkaline HER", observation_items[0].entity_mentions)
        self.assertIn("overpotential", observation_items[0].metric_mentions)
        self.assertIn("activity", observation_items[0].metric_mentions)


if __name__ == "__main__":
    unittest.main()
