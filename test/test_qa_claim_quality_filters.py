from __future__ import annotations

import unittest

from qa.evidence import ClaimMiner
from qa.retrieval_state import EvidenceItem


def _evidence_item(*, evidence_id: str, snippet: str, entity_mentions: list[str]) -> EvidenceItem:
    return EvidenceItem.model_validate(
        {
            "evidence_id": evidence_id,
            "paper_id": "paper-1",
            "doi": "10.1000/test",
            "section_id": "sec_abstract",
            "section_type": "abstract",
            "role": "observation",
            "snippet": snippet,
            "source_span": {"start": 0, "end": max(1, len(snippet))},
            "source_layer": "abstract",
            "claim_polarity": "support",
            "conditions": {},
            "condition_source_refs": [],
            "metric_mentions": ["activity"],
            "entity_mentions": entity_mentions,
            "extraction_confidence": 0.82,
            "extraction_notes": "fixture",
        }
    )


class ClaimQualityFilterTests(unittest.TestCase):
    def test_claim_miner_drops_generic_main_entities(self):
        miner = ClaimMiner(llm=None)
        bad_item = _evidence_item(
            evidence_id="ev-bad",
            snippet="Abstract shows improved HER activity in 1 M KOH.",
            entity_mentions=["Abstract"],
        )

        claims = miner.run([bad_item])

        self.assertEqual([], claims)

    def test_claim_miner_drops_garbled_snippets(self):
        miner = ClaimMiner(llm=None)
        garbage_item = _evidence_item(
            evidence_id="ev-garbage",
            snippet="JFIF ICC_PROFILE 8BIM " * 20,
            entity_mentions=["Pt/C"],
        )

        claims = miner.run([garbage_item])

        self.assertEqual([], claims)

    def test_claim_miner_keeps_valid_supported_entities(self):
        miner = ClaimMiner(llm=None)
        good_item = _evidence_item(
            evidence_id="ev-good",
            snippet="Pt/C improves HER activity in 1 M KOH relative to untreated carbon.",
            entity_mentions=["Pt/C"],
        )

        claims = miner.run([good_item])

        self.assertEqual(1, len(claims))
        self.assertEqual("Pt/C", claims[0].main_entity)


if __name__ == "__main__":
    unittest.main()
