from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

from qa.nodes.entity_resolver import EntityResolverNode
from qa.state import TaskSpec


def _task_spec(question: str) -> TaskSpec:
    return TaskSpec.model_validate(
        {
            "question": question,
            "normalized_question": question.lower(),
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


def _mention(
    surface_form: str,
    *,
    candidate_entity_types: list[str],
    selected_entity_type: str | None,
    confidence: float = 0.92,
    rationale: str = "fixture",
) -> dict:
    return {
        "surface_form": surface_form,
        "candidate_entity_types": list(candidate_entity_types),
        "selected_entity_type": selected_entity_type,
        "confidence": confidence,
        "rationale": rationale,
    }


def _extraction_response(*mentions: dict) -> dict:
    return {"mentions": list(mentions)}


class _FakePubChemClient:
    def __init__(self, responses: dict[str, list[dict]] | None = None, *, error: Exception | None = None) -> None:
        self.responses = dict(responses or {})
        self.error = error
        self.calls: list[dict[str, object]] = []

    def search_candidates(self, query: str, max_candidates: int = 5):
        self.calls.append({"query": query, "max_candidates": max_candidates})
        if self.error is not None:
            raise self.error
        return [dict(item) for item in list(self.responses.get(query, []) or [])]


class _FakeLLM:
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        if not self.responses:
            raise AssertionError("LLM invoked more times than expected in test fixture.")
        return self.responses.pop(0)


class EntityResolverNodeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(".cache") / f"entity_resolver_{uuid.uuid4().hex[:8]}"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.empty_seed_path = self.temp_dir / "empty_seeds.yaml"
        self.empty_seed_path.write_text("{}\n", encoding="utf-8")

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_seed_hit_does_not_call_pubchem(self):
        question = "How does Pt/C affect HER activity?"
        llm = _FakeLLM(
            [
                _extraction_response(
                    _mention("Pt/C", candidate_entity_types=["catalyst"], selected_entity_type="catalyst"),
                    _mention("HER", candidate_entity_types=["reaction"], selected_entity_type="reaction"),
                )
            ]
        )
        pubchem = _FakePubChemClient()
        resolver = EntityResolverNode(llm=llm, pubchem_client=pubchem)

        result = resolver.resolve_detailed(question=question, task_spec=_task_spec(question))

        catalyst = next(entity for entity in result.entity_pack.entities if entity.mention == "Pt/C")
        reaction = next(entity for entity in result.entity_pack.entities if entity.mention == "HER")
        self.assertEqual("platinum on carbon", catalyst.canonical_name)
        self.assertEqual("hydrogen evolution reaction", reaction.canonical_name)
        self.assertEqual("seed", catalyst.resolver_source)
        self.assertEqual([], pubchem.calls)
        self.assertEqual([], result.provider_calls)

    def test_seed_hit_for_pubchem_eligible_type_queries_pubchem_for_enrichment(self):
        question = "What is the molecular formula of ethanol?"
        llm = _FakeLLM(
            [
                _extraction_response(
                    _mention("ethanol", candidate_entity_types=["solvent"], selected_entity_type="solvent")
                )
            ]
        )
        pubchem = _FakePubChemClient(
            responses={
                "ethanol": [
                    {
                        "canonical_name": "ethanol",
                        "formula": "C2H6O",
                        "smiles": "CCO",
                        "inchikey": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
                        "pubchem_cid": 702,
                        "aliases": ["ethanol", "EtOH"],
                    }
                ]
            }
        )
        resolver = EntityResolverNode(llm=llm, pubchem_client=pubchem)

        result = resolver.resolve_detailed(question=question, task_spec=_task_spec(question))

        entity = next(item for item in result.entity_pack.entities if item.mention == "ethanol")
        self.assertEqual("ethanol", entity.canonical_name)
        self.assertEqual("C2H6O", entity.formula)
        self.assertEqual(702, entity.pubchem_cid)
        self.assertEqual("seed", entity.resolver_source)
        self.assertEqual([{"query": "ethanol", "max_candidates": 5}], pubchem.calls)
        self.assertEqual("pubchem", result.provider_calls[0]["provider"])
        self.assertEqual("hit", result.provider_calls[0]["status"])

    def test_unique_pubchem_hit_returns_normalized_entity(self):
        question = "Does pyridine ligand improve selectivity?"
        llm = _FakeLLM(
            [
                _extraction_response(
                    _mention("pyridine", candidate_entity_types=["ligand"], selected_entity_type="ligand")
                )
            ]
        )
        pubchem = _FakePubChemClient(
            responses={
                "pyridine": [
                    {
                        "canonical_name": "pyridine",
                        "formula": "C5H5N",
                        "smiles": "C1=CC=NC=C1",
                        "inchikey": "JUJWROOIHBZHMG-UHFFFAOYSA-N",
                        "pubchem_cid": 1049,
                        "aliases": ["pyridine"],
                    }
                ]
            }
        )
        resolver = EntityResolverNode(
            llm=llm,
            pubchem_client=pubchem,
            seed_path=str(self.empty_seed_path),
        )

        result = resolver.resolve_detailed(question=question, task_spec=_task_spec(question))

        ligand = next(entity for entity in result.entity_pack.entities if entity.mention == "pyridine")
        self.assertEqual("ligand", ligand.entity_type)
        self.assertEqual("pyridine", ligand.canonical_name)
        self.assertEqual("C5H5N", ligand.formula)
        self.assertEqual("JUJWROOIHBZHMG-UHFFFAOYSA-N", ligand.inchikey)
        self.assertEqual(1049, ligand.pubchem_cid)
        self.assertEqual("pubchem", ligand.resolver_source)
        self.assertEqual([{"query": "pyridine", "max_candidates": 5}], pubchem.calls)
        self.assertEqual("hit", result.provider_calls[0]["status"])
        self.assertEqual(1, result.provider_calls[0]["candidate_count"])

    def test_pubchem_multi_candidate_with_low_confidence_stays_unresolved(self):
        question = "Does PCA ligand improve selectivity?"
        llm = _FakeLLM(
            [
                _extraction_response(
                    _mention("PCA", candidate_entity_types=["ligand"], selected_entity_type="ligand")
                ),
                {
                    "selected_index": 0,
                    "entity_type": "ligand",
                    "confidence": 0.4,
                    "rationale": "insufficient evidence",
                },
            ]
        )
        pubchem = _FakePubChemClient(
            responses={
                "PCA": [
                    {
                        "canonical_name": "2-pyridinecarboxylic acid",
                        "formula": "C6H5NO2",
                        "smiles": "O=C(O)C1=CC=CC=N1",
                        "inchikey": "HVMCQZPBFUCRLZ-UHFFFAOYSA-N",
                        "pubchem_cid": 1038,
                        "aliases": ["PCA"],
                    },
                    {
                        "canonical_name": "1-pyrenecarboxaldehyde",
                        "formula": "C17H10O",
                        "smiles": "O=CC1=CC2=CC=CC3=CC=CC=C3C2=C1",
                        "inchikey": "JYJTVFIEFKXWQR-UHFFFAOYSA-N",
                        "pubchem_cid": 12345,
                        "aliases": ["PCA"],
                    },
                ]
            }
        )
        resolver = EntityResolverNode(
            llm=llm,
            pubchem_client=pubchem,
            seed_path=str(self.empty_seed_path),
            llm_disambiguation_enabled=True,
            disambiguation_min_confidence=0.7,
        )

        result = resolver.resolve_detailed(question=question, task_spec=_task_spec(question))

        self.assertFalse(any(entity.mention == "PCA" for entity in result.entity_pack.entities))
        unresolved = next(item for item in result.entity_pack.unresolved_mentions if item.mention == "PCA")
        self.assertIn("PubChem returned multiple candidates", unresolved.reason)
        self.assertTrue(
            any(flag.target == "PCA" and "PubChem returned multiple candidates" in flag.note for flag in result.entity_pack.entity_ambiguity_flags)
        )
        self.assertEqual(2, len(llm.calls))
        self.assertEqual("hit", result.provider_calls[0]["status"])
        self.assertEqual(2, result.provider_calls[0]["candidate_count"])

    def test_same_pubchem_identity_merges_aliases_within_single_run(self):
        question = "Compare ethanol solvent with EtOH solvent."
        candidate = {
            "canonical_name": "ethanol",
            "formula": "C2H6O",
            "smiles": "CCO",
            "inchikey": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
            "pubchem_cid": 702,
            "aliases": [],
        }
        llm = _FakeLLM(
            [
                _extraction_response(
                    _mention("ethanol", candidate_entity_types=["solvent"], selected_entity_type="solvent"),
                    _mention("EtOH", candidate_entity_types=["solvent"], selected_entity_type="solvent"),
                )
            ]
        )
        pubchem = _FakePubChemClient(
            responses={
                "ethanol": [candidate],
                "EtOH": [candidate],
            }
        )
        resolver = EntityResolverNode(
            llm=llm,
            pubchem_client=pubchem,
            seed_path=str(self.empty_seed_path),
        )

        result = resolver.resolve_detailed(question=question, task_spec=_task_spec(question))

        solvent_entities = [entity for entity in result.entity_pack.entities if entity.entity_type == "solvent"]
        self.assertEqual(1, len(solvent_entities))
        self.assertEqual("ethanol", solvent_entities[0].canonical_name)
        self.assertEqual(["ethanol", "EtOH"], solvent_entities[0].aliases)
        self.assertEqual(2, len(pubchem.calls))
        self.assertIn("merge", [event["event"] for event in result.resolution_index["cache_events"]])

    def test_provider_error_is_fail_open_and_returns_unresolved(self):
        question = "Does pyridine ligand improve selectivity?"
        llm = _FakeLLM(
            [
                _extraction_response(
                    _mention("pyridine", candidate_entity_types=["ligand"], selected_entity_type="ligand")
                )
            ]
        )
        pubchem = _FakePubChemClient(error=RuntimeError("timeout"))
        resolver = EntityResolverNode(
            llm=llm,
            pubchem_client=pubchem,
            seed_path=str(self.empty_seed_path),
            fail_open_on_provider_error=True,
        )

        result = resolver.resolve_detailed(question=question, task_spec=_task_spec(question))

        self.assertEqual(1, len(pubchem.calls))
        self.assertEqual("error", result.provider_calls[0]["status"])
        unresolved = next(item for item in result.entity_pack.unresolved_mentions if item.mention == "pyridine")
        self.assertIn("Normalization did not resolve", unresolved.reason)

    def test_run_local_cache_does_not_leak_across_runs(self):
        question = "Does pyridine ligand improve selectivity?"
        llm = _FakeLLM(
            [
                _extraction_response(
                    _mention("pyridine", candidate_entity_types=["ligand"], selected_entity_type="ligand")
                ),
                _extraction_response(
                    _mention("pyridine", candidate_entity_types=["ligand"], selected_entity_type="ligand")
                ),
            ]
        )
        pubchem = _FakePubChemClient(
            responses={
                "pyridine": [
                    {
                        "canonical_name": "pyridine",
                        "formula": "C5H5N",
                        "smiles": "C1=CC=NC=C1",
                        "inchikey": "JUJWROOIHBZHMG-UHFFFAOYSA-N",
                        "pubchem_cid": 1049,
                        "aliases": ["pyridine"],
                    }
                ]
            }
        )
        resolver = EntityResolverNode(
            llm=llm,
            pubchem_client=pubchem,
            seed_path=str(self.empty_seed_path),
        )
        task_spec = _task_spec(question)

        first = resolver.resolve_detailed(question=question, task_spec=task_spec)
        second = resolver.resolve_detailed(question=question, task_spec=task_spec)

        self.assertEqual(2, len(pubchem.calls))
        self.assertEqual(1, len([entry for entry in first.resolution_index["entries"] if entry["canonical_name"] == "pyridine"]))
        self.assertEqual(1, len([entry for entry in second.resolution_index["entries"] if entry["canonical_name"] == "pyridine"]))

    def test_low_confidence_extraction_becomes_unresolved_without_pubchem(self):
        question = "Does pyridine ligand improve selectivity?"
        llm = _FakeLLM(
            [
                _extraction_response(
                    _mention(
                        "pyridine",
                        candidate_entity_types=["ligand"],
                        selected_entity_type="ligand",
                        confidence=0.4,
                    )
                )
            ]
        )
        pubchem = _FakePubChemClient(
            responses={
                "pyridine": [
                    {
                        "canonical_name": "pyridine",
                        "pubchem_cid": 1049,
                    }
                ]
            }
        )
        resolver = EntityResolverNode(
            llm=llm,
            pubchem_client=pubchem,
            seed_path=str(self.empty_seed_path),
            mention_extraction_min_confidence=0.7,
        )

        result = resolver.resolve_detailed(question=question, task_spec=_task_spec(question))

        self.assertEqual([], pubchem.calls)
        unresolved = next(item for item in result.entity_pack.unresolved_mentions if item.mention == "pyridine")
        self.assertIn("below the configured threshold", unresolved.reason)

    def test_invalid_extraction_span_is_skipped_without_crashing(self):
        question = "Does pyridine ligand improve selectivity?"
        llm = _FakeLLM(
            [
                _extraction_response(
                    _mention("pyridine improve", candidate_entity_types=["ligand"], selected_entity_type="ligand")
                )
            ]
        )
        resolver = EntityResolverNode(
            llm=llm,
            pubchem_client=_FakePubChemClient(),
        )

        result = resolver.resolve_detailed(question=question, task_spec=_task_spec(question))

        self.assertEqual([], result.entity_pack.entities)
        self.assertEqual([], result.entity_pack.unresolved_mentions)

    def test_out_of_order_mentions_align_without_crashing(self):
        question = "How does Pt/C affect HER activity?"
        llm = _FakeLLM(
            [
                _extraction_response(
                    _mention("HER", candidate_entity_types=["reaction"], selected_entity_type="reaction"),
                    _mention("Pt/C", candidate_entity_types=["catalyst"], selected_entity_type="catalyst"),
                )
            ]
        )
        resolver = EntityResolverNode(
            llm=llm,
            pubchem_client=_FakePubChemClient(),
        )

        result = resolver.resolve_detailed(question=question, task_spec=_task_spec(question))

        mentions = sorted(entity.mention for entity in result.entity_pack.entities)
        self.assertEqual(["HER", "Pt/C"], mentions)

    def test_no_rule_based_fallback_when_llm_returns_no_mentions(self):
        question = "Compare ligand improve with solvent with."
        llm = _FakeLLM([_extraction_response()])
        pubchem = _FakePubChemClient()
        resolver = EntityResolverNode(
            llm=llm,
            pubchem_client=pubchem,
            seed_path=str(self.empty_seed_path),
        )

        result = resolver.resolve_detailed(question=question, task_spec=_task_spec(question))

        self.assertEqual([], result.entity_pack.entities)
        self.assertEqual([], result.entity_pack.unresolved_mentions)
        self.assertEqual([], pubchem.calls)

    def test_missing_llm_fails_fast(self):
        resolver = EntityResolverNode(
            llm=None,
            pubchem_client=_FakePubChemClient(),
            seed_path=str(self.empty_seed_path),
        )

        with self.assertRaisesRegex(ValueError, "configured LLM"):
            resolver.resolve_detailed(
                question="Does pyridine ligand improve selectivity?",
                task_spec=_task_spec("Does pyridine ligand improve selectivity?"),
            )


if __name__ == "__main__":
    unittest.main()
