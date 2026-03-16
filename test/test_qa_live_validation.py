from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

from qa.live_validation import main as live_validation_main
from qa.live_validation import validate_live_qa
from qa.synthesis_state import QAResult
from utils import ensure_dir, save_json


def _confidence(score: float = 0.82) -> dict:
    return {
        "level": "high" if score >= 0.75 else "low",
        "score": score,
        "rationale": "validation fixture",
    }


class _BaseFakeValidationSystem:
    def __init__(self) -> None:
        self.qa_config = {
            "providers": {
                "openalex_mailto": "chemqa@example.com",
                "crossref_mailto": "chemqa@example.com",
                "semantic_scholar_api_key": None,
                "unpaywall_email": "chemqa@example.com",
                "http_timeout": 5.0,
            }
        }

    def _runtime_manifest(self) -> dict:
        return {
            "providers": {
                "openalex": {"enabled": True},
                "crossref": {"enabled": True},
                "semantic_scholar": {"enabled": True},
                "unpaywall": {"enabled": True},
            }
        }


class _EvidenceBackedSystem(_BaseFakeValidationSystem):
    def run_qa(self, *, question: str, context=None, artifact_dir=None):
        artifact_root = ensure_dir(Path(artifact_dir))
        paper_id = "paper-1"
        citation_id = "CIT-1"

        save_json(self._runtime_manifest(), artifact_root / "runtime_manifest.json")
        save_json(
            [{"paper_id": paper_id, "title": "Pt/C HER in alkaline media"}],
            artifact_root / "paper_candidates.json",
        )
        save_json(
            [{"paper_id": paper_id, "title": "Pt/C HER in alkaline media"}],
            artifact_root / "paper_records.json",
        )
        save_json(
            [{"provider": "openalex", "stage": "search", "lane": "review", "hit_count": 1}],
            artifact_root / "retrieval_diagnostics.json",
        )
        save_json(
            {
                "openalex": {
                    "status": "healthy",
                    "calls": 1,
                    "successes": 1,
                    "retry_exhausted_failures": 0,
                    "skipped_calls": 0,
                    "last_error": None,
                },
                "crossref": {
                    "status": "healthy",
                    "calls": 1,
                    "successes": 1,
                    "retry_exhausted_failures": 0,
                    "skipped_calls": 0,
                    "last_error": None,
                },
            },
            artifact_root / "provider_health.json",
        )
        save_json(
            {
                "section_claims": [
                    {
                        "section_id": "direct_answer",
                        "title": "Direct Answer",
                        "accepted_claim_ids": ["claim-1"],
                        "claim_summaries": ["Pt/C lowers HER overpotential in 1 M KOH."],
                        "core_citation_ids": [citation_id],
                        "section_confidence": _confidence(),
                    }
                ],
                "citation_catalog": [
                    {
                        "citation_id": citation_id,
                        "paper_id": paper_id,
                        "title": "Pt/C HER in alkaline media",
                        "year": 2024,
                    }
                ],
                "overall_confidence": _confidence(),
                "section_confidence": [],
                "insufficient_evidence": False,
                "claim_trace": [
                    {
                        "section_id": "direct_answer",
                        "claim_id": "claim-1",
                        "status": "accepted",
                        "citation_ids": [citation_id],
                        "confidence": 0.85,
                    }
                ],
                "retrieval_diagnostics_summary": "",
                "execution_warnings": [],
                "question": question,
                "task_spec": {
                    "version": "1.0",
                    "question": question,
                    "normalized_question": question,
                    "question_type": "mechanism",
                    "recency_policy": "none",
                    "answer_sections": [],
                    "required_condition_axes": [],
                    "query_constraints": {
                        "must_include_terms": [],
                        "should_include_terms": [],
                        "exclude_terms": [],
                        "preferred_entity_types": [],
                        "allow_broad_expansion": False,
                    },
                    "ambiguity_flags": [],
                    "router_confidence": 0.9,
                },
            },
            artifact_root / "synthesis_input_pack.json",
        )
        save_json(
            {
                "claims": [
                    {
                        "claim_id": "claim-1",
                        "status": "accepted",
                    }
                ]
            },
            artifact_root / "evidence_ledger_reviewed.json",
        )

        result = QAResult.model_validate(
            {
                "question": question,
                "language": "en",
                "final_answer": "Pt/C improves HER activity in 1 M KOH by lowering overpotential and accelerating interfacial hydrogen evolution. [CIT-1]",
                "sections": [],
                "citations": [
                    {
                        "citation_id": citation_id,
                        "paper_id": paper_id,
                        "title": "Pt/C HER in alkaline media",
                        "year": 2024,
                        "supporting_claim_ids": ["claim-1"],
                    }
                ],
                "claim_trace": [
                    {
                        "section_id": "direct_answer",
                        "claim_id": "claim-1",
                        "status": "accepted",
                        "citation_ids": [citation_id],
                        "confidence": 0.85,
                    }
                ],
                "overall_confidence": _confidence(),
                "section_confidence": [],
                "insufficient_evidence": False,
                "limitations_summary": "",
                "retrieval_diagnostics_summary": "",
                "execution_warnings": [],
                "artifact_paths": {
                    "qa_result": str(artifact_root / "qa_result.json"),
                    "runtime_manifest": str(artifact_root / "runtime_manifest.json"),
                    "synthesis_input_pack": str(artifact_root / "synthesis_input_pack.json"),
                    "final_answer": str(artifact_root / "final_answer.md"),
                },
                "time_elapsed": 0.21,
            }
        )
        save_json(result.model_dump(exclude_none=True), artifact_root / "qa_result.json")
        (artifact_root / "final_answer.md").write_text(result.final_answer, encoding="utf-8")
        return result


class _DegradedSystem(_BaseFakeValidationSystem):
    def run_qa(self, *, question: str, context=None, artifact_dir=None):
        artifact_root = ensure_dir(Path(artifact_dir))

        save_json(self._runtime_manifest(), artifact_root / "runtime_manifest.json")
        save_json([], artifact_root / "paper_candidates.json")
        save_json([], artifact_root / "paper_records.json")
        save_json(
            [
                {
                    "provider": "openalex",
                    "stage": "search",
                    "lane": "review",
                    "failure_count": 1,
                    "skipped_count": 3,
                    "sample_messages": ["retry exhausted; provider unavailable"],
                }
            ],
            artifact_root / "retrieval_diagnostics.json",
        )
        save_json(
            {
                "openalex": {
                    "status": "unavailable",
                    "calls": 1,
                    "successes": 0,
                    "retry_exhausted_failures": 1,
                    "skipped_calls": 3,
                    "last_error": "retry exhausted",
                },
                "crossref": {
                    "status": "idle",
                    "calls": 0,
                    "successes": 0,
                    "retry_exhausted_failures": 0,
                    "skipped_calls": 0,
                    "last_error": None,
                },
            },
            artifact_root / "provider_health.json",
        )
        save_json(
            {
                "section_claims": [],
                "citation_catalog": [],
                "overall_confidence": _confidence(0.2),
                "section_confidence": [],
                "insufficient_evidence": True,
                "claim_trace": [],
                "retrieval_diagnostics_summary": "External literature retrieval encountered issues: OpenAlex review search had 1 failure (retry exhausted).",
                "execution_warnings": [],
                "question": question,
                "task_spec": {
                    "version": "1.0",
                    "question": question,
                    "normalized_question": question,
                    "question_type": "mechanism",
                    "recency_policy": "none",
                    "answer_sections": [],
                    "required_condition_axes": [],
                    "query_constraints": {
                        "must_include_terms": [],
                        "should_include_terms": [],
                        "exclude_terms": [],
                        "preferred_entity_types": [],
                        "allow_broad_expansion": False,
                    },
                    "ambiguity_flags": [],
                    "router_confidence": 0.9,
                },
            },
            artifact_root / "synthesis_input_pack.json",
        )
        save_json({"claims": []}, artifact_root / "evidence_ledger_reviewed.json")

        result = QAResult.model_validate(
            {
                "question": question,
                "language": "en",
                "final_answer": "Available accepted evidence is limited and does not support a firm conclusion for this section.",
                "sections": [],
                "citations": [],
                "claim_trace": [],
                "overall_confidence": _confidence(0.2),
                "section_confidence": [],
                "insufficient_evidence": True,
                "limitations_summary": "External literature retrieval encountered issues.",
                "retrieval_diagnostics_summary": "External literature retrieval encountered issues: OpenAlex review search had 1 failure (retry exhausted).",
                "execution_warnings": [],
                "artifact_paths": {
                    "qa_result": str(artifact_root / "qa_result.json"),
                    "runtime_manifest": str(artifact_root / "runtime_manifest.json"),
                    "synthesis_input_pack": str(artifact_root / "synthesis_input_pack.json"),
                    "final_answer": str(artifact_root / "final_answer.md"),
                },
                "time_elapsed": 0.19,
            }
        )
        save_json(result.model_dump(exclude_none=True), artifact_root / "qa_result.json")
        (artifact_root / "final_answer.md").write_text(result.final_answer, encoding="utf-8")
        return result


class _BrokenSystem(_BaseFakeValidationSystem):
    def run_qa(self, *, question: str, context=None, artifact_dir=None):
        raise RuntimeError("synthetic pipeline failure")


class QALiveValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(".cache") / f"qa_live_validation_{uuid.uuid4().hex[:8]}"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_validate_live_qa_reports_pass_real_evidence(self):
        artifact_dir = self.temp_dir / "evidence"
        report = validate_live_qa(
            "How does Pt/C affect HER activity in 1 M KOH?",
            system=_EvidenceBackedSystem(),
            artifact_dir=str(artifact_dir),
            perform_network_probe=False,
        )

        self.assertEqual(report.category, "PASS_REAL_EVIDENCE")
        self.assertGreaterEqual(report.citation_count, 1)
        self.assertGreaterEqual(report.accepted_claim_count, 1)
        self.assertTrue(report.citation_paper_record_matches)
        self.assertTrue(Path(report.report_path).exists())

    def test_validate_live_qa_reports_pass_degraded_when_provider_is_blocked(self):
        artifact_dir = self.temp_dir / "degraded"
        report = validate_live_qa(
            "How does Pt/C affect HER activity in 1 M KOH?",
            system=_DegradedSystem(),
            artifact_dir=str(artifact_dir),
            perform_network_probe=False,
        )

        self.assertEqual(report.category, "PASS_DEGRADED")
        self.assertTrue(report.provider_failure_detected)
        self.assertTrue(report.insufficient_evidence)
        self.assertEqual(report.citation_count, 0)

    def test_live_validation_cli_returns_nonzero_for_pipeline_failure(self):
        config_path = self.temp_dir / "config.yaml"
        config_path.write_text(
            "\n".join(
                [
                    "paths:",
                    f"  outputs: \"{self.temp_dir.as_posix()}\"",
                    "qa:",
                    "  save_output: false",
                    "  outputs_dir: \"\"",
                    "  artifact_subdir: \"qa_artifacts\"",
                    "  enable_peer_review: true",
                ]
            ),
            encoding="utf-8",
        )
        artifact_root = self.temp_dir / "suite"

        def _factory(**_kwargs):
            return _BrokenSystem()

        exit_code = live_validation_main(
            [
                "--question",
                "How does Pt/C affect HER activity in 1 M KOH?",
                "--artifact-root",
                str(artifact_root),
                "--config",
                str(config_path),
            ],
            system_factory=_factory,
            configure_logging=False,
            perform_network_probe=False,
        )

        self.assertEqual(exit_code, 1)
        suite_report = artifact_root / "live_validation_suite.json"
        self.assertTrue(suite_report.exists())


if __name__ == "__main__":
    unittest.main()
