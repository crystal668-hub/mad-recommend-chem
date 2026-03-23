from __future__ import annotations

import unittest
from unittest.mock import patch

from qa.nodes.citation_reviewer import CitationReviewer
from qa.nodes.claim_revision import ClaimRevisionNode
from qa.nodes.contradiction_reviewer import ContradictionReviewer
from qa.nodes.methodology_reviewer import MethodologyReviewer
from qa.nodes.review_merge import ReviewMergeNode
from qa.peer_review_errors import PeerReviewExecutionError
from qa.review_pipeline import StructuredPeerReviewPipeline
from qa.retrieval_state import ClaimRecord, EvidenceItem, EvidenceLedger
from qa.runtime import build_qa_runtime, resolve_qa_runtime_config


class _StaticLLM:
    def __init__(self, payload):
        self.payload = payload

    def invoke(self, messages):
        return self.payload


def _evidence_item(evidence_id: str, *, source_layer: str = "abstract") -> EvidenceItem:
    return EvidenceItem.model_validate(
        {
            "evidence_id": evidence_id,
            "paper_id": "paper-1",
            "doi": "10.1000/test",
            "section_id": "sec_abstract",
            "section_type": "abstract",
            "role": "mechanism",
            "snippet": "Pt/C may improve HER activity in 1 M KOH by facilitating hydrogen adsorption.",
            "source_span": {"start": 0, "end": 84},
            "source_layer": source_layer,
            "claim_polarity": "support",
            "conditions": {},
            "condition_source_refs": [],
            "metric_mentions": ["activity"],
            "entity_mentions": ["Pt/C"],
            "extraction_confidence": 0.83,
            "extraction_notes": "fixture",
        }
    )


def _claim_record(claim_id: str, evidence_id: str) -> ClaimRecord:
    return ClaimRecord.model_validate(
        {
            "claim_id": claim_id,
            "claim_type": "mechanism",
            "section_id": "sec_abstract",
            "claim_text": "Pt/C may improve HER activity.",
            "main_entity": "Pt/C",
            "relation_type": "mechanistic_explanation",
            "metric_family": "activity",
            "condition_scope": {},
            "condition_signature": "{}",
            "supporting_evidence_ids": [evidence_id],
            "opposing_evidence_ids": [],
            "status": "draft",
            "claim_confidence": 0.79,
            "cluster_size": 1,
            "provenance_notes": "fixture",
        }
    )


class QARuntimeConfigTests(unittest.TestCase):
    def test_resolve_runtime_config_includes_stability_defaults(self):
        resolved = resolve_qa_runtime_config({"qa": {}})

        self.assertEqual(45.0, resolved["model_timeout_seconds"])
        self.assertEqual(10, resolved["progress_log_every_claims"])
        self.assertEqual(40, resolved["peer_review"]["max_claims_for_llm_review"])
        self.assertEqual(15, resolved["peer_review"]["max_second_round_claims"])
        self.assertTrue(resolved["peer_review"]["disable_llm_review_when_abstract_only"])
        self.assertEqual("fail_fast_only", resolved["peer_review"]["fallback_mode"])
        self.assertEqual("./qa/resources/entity_seeds.yaml", resolved["entity_resolution"]["seed_file"])
        self.assertTrue(resolved["entity_resolution"]["emit_seed_suggestions"])
        self.assertTrue(resolved["entity_resolution"]["pubchem_enabled"])
        self.assertEqual(
            ["molecule", "solvent", "reagent", "ligand", "substrate"],
            resolved["entity_resolution"]["pubchem_entity_types"],
        )
        self.assertEqual(5, resolved["entity_resolution"]["max_pubchem_candidates"])
        self.assertEqual(0.7, resolved["entity_resolution"]["mention_extraction_min_confidence"])
        self.assertTrue(resolved["entity_resolution"]["llm_disambiguation_enabled"])
        self.assertEqual(0.7, resolved["entity_resolution"]["disambiguation_min_confidence"])
        self.assertTrue(resolved["entity_resolution"]["fail_open_on_provider_error"])
        self.assertEqual(120.0, resolved["react_reviewed"]["stage_watchdog_seconds"])
        self.assertEqual(45.0, resolved["providers"]["document_fetch_timeout_seconds"])
        self.assertEqual(300.0, resolved["providers"]["document_fetch_total_timeout_seconds"])
        self.assertEqual(8, resolved["providers"]["provider_redirect_limit"])
        self.assertEqual("pymupdf", resolved["pdf_extraction"]["primary_backend"])
        self.assertEqual("none", resolved["pdf_extraction"]["secondary_backend"])

    @patch("qa.runtime.build_chat_model_from_config")
    def test_build_runtime_injects_default_timeout_when_alias_has_none(self, mock_build_model):
        mock_build_model.return_value = object()
        runtime = build_qa_runtime(
            config={
                "llm": {
                    "agent1": {
                        "provider": "openai",
                        "model": "openai/gpt-5.2",
                        "api_key": "test-key",
                    }
                },
                "qa": {},
            }
        )

        self.assertTrue(mock_build_model.called)
        first_config = mock_build_model.call_args_list[0].args[0]
        self.assertEqual(45.0, first_config["timeout"])
        self.assertEqual(45.0, runtime.runtime_manifest["models"]["router"]["timeout_seconds"])
        self.assertEqual("fail_fast", runtime.runtime_manifest["models"]["router"]["fallback"])
        self.assertEqual("fail_fast", runtime.runtime_manifest["models"]["query_planner"]["fallback"])
        self.assertEqual("fail_fast", runtime.runtime_manifest["models"]["synthesizer"]["fallback"])
        self.assertEqual("fail_fast", runtime.runtime_manifest["models"]["methodology_reviewer"]["fallback"])
        self.assertEqual("fail_fast", runtime.runtime_manifest["models"]["citation_reviewer"]["fallback"])
        self.assertEqual("fail_fast", runtime.runtime_manifest["models"]["contradiction_reviewer"]["fallback"])
        self.assertEqual("fail_fast", runtime.runtime_manifest["models"]["claim_revision"]["fallback"])
        self.assertEqual("fail_fast", runtime.runtime_manifest["models"]["review_merge"]["fallback"])
        self.assertEqual("fail_fast", runtime.runtime_manifest["models"]["react_proposer"]["fallback"])
        self.assertEqual(
            "fail_fast",
            runtime.runtime_manifest["models"]["react_reviewer_search_coverage"]["fallback"],
        )
        self.assertEqual(
            "./qa/resources/entity_seeds.yaml",
            runtime.runtime_manifest["qa"]["entity_resolution"]["seed_file"],
        )
        self.assertTrue(runtime.runtime_manifest["providers"]["pubchem"]["enabled"])
        self.assertEqual(
            ["molecule", "solvent", "reagent", "ligand", "substrate"],
            runtime.runtime_manifest["providers"]["pubchem"]["entity_types"],
        )


class PeerReviewBudgetTests(unittest.TestCase):
    def test_peer_review_fails_fast_when_llm_review_budget_is_exceeded(self):
        evidence_items = [_evidence_item(f"ev-{index}") for index in range(3)]
        claims = [_claim_record(f"claim-{index}", f"ev-{index}") for index in range(3)]
        ledger = EvidenceLedger(claims=claims, evidence_items=evidence_items)

        pipeline = StructuredPeerReviewPipeline(
            max_claims_for_llm_review=2,
            max_second_round_claims=1,
            disable_llm_review_when_abstract_only=True,
            fallback_mode="fail_fast_only",
            progress_log_every_claims=1,
        )

        with self.assertRaises(PeerReviewExecutionError) as ctx:
            pipeline.run(ledger)

        error = ctx.exception
        self.assertEqual("peer_review_policy", error.stage)
        self.assertTrue(
            any("claim volume exceeded budget" in reason for reason in error.details["disable_reasons"])
        )

    def test_peer_review_caps_second_round_without_degrading(self):
        evidence_items = [_evidence_item(f"ev-{index}", source_layer="fulltext") for index in range(3)]
        claims = [_claim_record(f"claim-{index}", f"ev-{index}") for index in range(3)]
        ledger = EvidenceLedger(claims=claims, evidence_items=evidence_items)

        pipeline = StructuredPeerReviewPipeline(
            methodology_reviewer=MethodologyReviewer(llm=_StaticLLM({"flags": []})),
            citation_reviewer=CitationReviewer(llm=_StaticLLM({"flags": []})),
            contradiction_reviewer=ContradictionReviewer(llm=_StaticLLM({"conflict_type": "no_conflict"})),
            claim_revision_node=ClaimRevisionNode(
                llm=_StaticLLM({"claim_text": "Pt/C may improve HER activity.", "condition_scope": {}})
            ),
            review_merge_node=ReviewMergeNode(
                llm=_StaticLLM({"status": "contested", "rationale": "Warning-level issues remain after revision."})
            ),
            max_claims_for_llm_review=10,
            max_second_round_claims=1,
            disable_llm_review_when_abstract_only=True,
            fallback_mode="fail_fast_only",
            progress_log_every_claims=1,
        )

        reviewed = pipeline.run(ledger)

        self.assertEqual(3, len(reviewed.review_summaries))
        self.assertTrue(
            any("limited second-round review to 1 highest-risk claims" in warning for warning in pipeline.last_execution_warnings)
        )
        self.assertEqual(1, reviewed.cluster_stats["second_round_claim_count"])


if __name__ == "__main__":
    unittest.main()
