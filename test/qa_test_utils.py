from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

from qa.artifacts import QAArtifactStore
from qa.evidence import EvidenceLedgerBuilder
from qa.facade import QASystem
from qa.nodes.entity_resolver import EntityResolutionRunResult
from qa.retrieval_pipeline import HeterogeneousRetrievalPipeline
from qa.retrieval_state import (
    ClaimRecord,
    EvidenceItem,
    PaperCandidate,
    PaperRecord,
    QueryPlan,
    RetrievalDiagnosticRecord,
    ReviewSummary,
    Section,
    SectionIndex,
)
from qa.runtime import DEFAULT_QA_MODEL_ALIASES
from qa.state import EntityPack, GroundingState, SourceSpan, TaskSpec
from qa.synthesis_pipeline import VerifiedSynthesisPipeline
from utils import Logger
import utils.logger as logger_mod


def confidence_payload(score: float = 0.82) -> dict[str, Any]:
    return {
        "level": "high" if score >= 0.75 else "medium" if score >= 0.45 else "low",
        "score": score,
        "rationale": "test fixture",
    }


def make_task_spec(
    question: str = "How does Pt/C affect HER activity in 1 M KOH?",
    *,
    question_type: str = "fact",
) -> TaskSpec:
    return TaskSpec.model_validate(
        {
            "question": question,
            "normalized_question": question.lower(),
            "question_type": question_type,
            "recency_policy": "none",
            "answer_sections": [
                {
                    "section_id": "direct_answer",
                    "title": "Direct Answer",
                    "required": True,
                    "instruction": "Answer directly with the accepted evidence.",
                }
            ],
            "router_confidence": 0.92,
        }
    )


def make_entity_pack() -> EntityPack:
    return EntityPack.model_validate(
        {
            "entities": [
                {
                    "entity_id": "ent-1",
                    "mention": "Pt/C",
                    "canonical_name": "Pt/C",
                    "entity_type": "catalyst",
                    "aliases": ["platinum on carbon"],
                    "query_anchors": ["Pt/C", "platinum on carbon"],
                    "resolver_source": "seed",
                    "resolution_confidence": 0.96,
                    "status": "resolved",
                    "source_text": "Pt/C",
                    "source_span": {"start": 0, "end": 4},
                }
            ]
        }
    )


def make_entity_resolution_result(question: str) -> EntityResolutionRunResult:
    entity_pack = make_entity_pack()
    return EntityResolutionRunResult(
        entity_pack=entity_pack,
        resolution_index={
            "entries": [
                {
                    "entry_id": "res-1",
                    "entity_type": "catalyst",
                    "canonical_name": "Pt/C",
                    "aliases": ["platinum on carbon"],
                    "query_anchors": ["Pt/C", "platinum on carbon"],
                    "resolver_source": "seed",
                    "resolution_confidence": 0.96,
                    "status": "resolved",
                    "lookup_keys": ["pt/c", "platinum on carbon"],
                }
            ],
            "cache_events": [{"event": "store", "mention": "Pt/C", "entry_id": "res-1"}],
        },
        provider_calls=[
            {
                "provider": "pubchem",
                "query": "Pt/C",
                "status": "skipped",
                "reason": "seed hit",
            }
        ],
        seed_suggestions=[
            {
                "mention": "Pt/C",
                "entity_type": "catalyst",
                "seed_key": question,
            }
        ],
    )


class StaticGroundingPipeline:
    def __init__(self, *, task_spec: TaskSpec | None = None, resolution_result: EntityResolutionRunResult | None = None) -> None:
        self.task_spec = task_spec or make_task_spec()
        self.resolution_result = resolution_result or make_entity_resolution_result(self.task_spec.question)

    def run_detailed(self, question: str, context: str | None = None):
        task_spec = self.task_spec.model_copy(update={"question": question, "normalized_question": question.lower()})
        resolution_result = EntityResolutionRunResult(
            entity_pack=self.resolution_result.entity_pack,
            resolution_index=copy.deepcopy(self.resolution_result.resolution_index),
            provider_calls=copy.deepcopy(self.resolution_result.provider_calls),
            seed_suggestions=copy.deepcopy(self.resolution_result.seed_suggestions),
        )
        grounding_state = GroundingState(
            question=question,
            context=context,
            task_spec=task_spec,
            entity_pack=resolution_result.entity_pack,
        )
        return grounding_state, resolution_result


class StaticQueryPlanner:
    def run(self, *, task_spec: TaskSpec, entity_pack: EntityPack):
        return [
            QueryPlan(
                lane="review",
                query_text=f"{task_spec.question} Pt/C HER 1 M KOH",
                must_terms=["Pt/C", "HER"],
                preferred_sources=["openalex"],
            )
        ]


class StaticRetriever:
    def __init__(self) -> None:
        self.last_diagnostics = [
            RetrievalDiagnosticRecord(
                provider="openalex",
                stage="search",
                lane="review",
                hit_count=1,
            )
        ]
        self.last_provider_health = {
            "openalex": {
                "status": "healthy",
                "calls": 1,
                "successes": 1,
                "retry_exhausted_failures": 0,
                "skipped_calls": 0,
                "last_error": None,
            }
        }

    def run(self, *, task_spec: TaskSpec, entity_pack: EntityPack, query_plans, artifact_store=None):
        return [
            PaperCandidate(
                paper_id="paper-1",
                title="Pt/C HER in alkaline media",
                abstract="Pt/C improves HER activity in 1 M KOH.",
                year=2024,
                provider_hits=["openalex"],
                lane_sources=["review"],
                retrieval_score=0.93,
            )
        ]


class StaticDocumentAcquirer:
    def __init__(self, *, warnings: Sequence[str] | None = None) -> None:
        self.last_diagnostics = [
            RetrievalDiagnosticRecord(
                provider="oa_fetch",
                stage="fetch",
                hit_count=1,
            )
        ]
        self.last_provider_health = {
            "oa_fetch": {
                "status": "healthy",
                "calls": 1,
                "successes": 1,
                "retry_exhausted_failures": 0,
                "skipped_calls": 0,
                "last_error": None,
            }
        }
        self.last_execution_warnings = list(warnings or [])

    def run(self, *, candidates, artifact_store=None):
        store = artifact_store or QAArtifactStore()
        paper = candidates[0]
        fulltext = "Pt/C improves HER activity in 1 M KOH by lowering overpotential."
        fulltext_path = store.write_text(f"fulltext/{paper.paper_id}.txt", fulltext)
        return (
            [
                PaperRecord(
                    paper_id=paper.paper_id,
                    title=paper.title,
                    abstract=paper.abstract,
                    year=paper.year,
                    provider_sources=list(paper.provider_hits),
                    fulltext_available=True,
                    fulltext_status="fulltext_indexed",
                    fulltext_artifact_path=fulltext_path,
                )
            ],
            [
                SectionIndex(
                    paper_id=paper.paper_id,
                    fulltext_status="fulltext_indexed",
                    sections=[
                        Section(
                            section_id="sec-results",
                            section_type="results",
                            heading="Results",
                            fulltext_char_start=0,
                            fulltext_char_end=len(fulltext),
                        )
                    ],
                )
            ],
        )


class StaticEvidenceExtractor:
    def run(self, *, task_spec: TaskSpec, entity_pack: EntityPack, paper_record: PaperRecord, section_index: SectionIndex):
        return [
            EvidenceItem(
                evidence_id="ev-1",
                paper_id=paper_record.paper_id,
                section_id="sec-results",
                section_type="results",
                role="observation",
                snippet="Pt/C improves HER activity in 1 M KOH.",
                source_span=SourceSpan(start=0, end=16),
                source_layer="fulltext",
                claim_polarity="support",
                conditions={"electrolyte": "1 M KOH"},
                extraction_confidence=0.91,
            )
        ]


class StaticClaimMiner:
    def run(self, *, evidence_items: Sequence[EvidenceItem], task_spec: TaskSpec):
        evidence = evidence_items[0]
        return [
            ClaimRecord(
                claim_id="claim-1",
                claim_type="fact",
                section_id="direct_answer",
                claim_text="Pt/C improves HER activity in 1 M KOH.",
                main_entity="Pt/C",
                relation_type="improves_activity",
                metric_family="overpotential",
                condition_scope={"electrolyte": "1 M KOH"},
                condition_signature="electrolyte=1 M KOH",
                supporting_evidence_ids=[evidence.evidence_id],
                status="draft",
                claim_confidence=0.87,
                cluster_size=1,
            )
        ]


class StaticPeerReviewPipeline:
    def __init__(self, *, warnings: Sequence[str] | None = None) -> None:
        self.last_execution_warnings = list(warnings or [])

    def run(self, evidence_ledger, task_spec=None):
        reviewed = evidence_ledger.model_copy(deep=True)
        reviewed.claims = [
            claim.model_copy(update={"status": "accepted"})
            for claim in reviewed.claims
        ]
        reviewed.claim_index = {claim.claim_id: index for index, claim in enumerate(reviewed.claims)}
        reviewed.review_summaries = [
            ReviewSummary(
                claim_id=claim.claim_id,
                review_rounds=1,
                review_flags=[],
                conflict_edge_ids=[],
                revision_records=[],
                final_status="accepted",
                merge_rationale="Synthetic peer review accepted the claim.",
            )
            for claim in reviewed.claims
        ]
        return reviewed


class NullLedgerRetrievalPipeline:
    def run_from_grounding(self, grounding_state: GroundingState, *, artifact_dir: str | None = None):
        from qa.retrieval_state import RetrievalState

        return RetrievalState(
            question=grounding_state.question,
            context=grounding_state.context,
            task_spec=grounding_state.task_spec,
            entity_pack=grounding_state.entity_pack,
            artifact_dir=str(Path(artifact_dir or ".").resolve()),
        )


def make_base_config(root: Path, *, workflow_mode: str, save_output: bool) -> Dict[str, Any]:
    outputs_dir = root / "outputs"
    return {
        "paths": {"outputs": str(outputs_dir)},
        "logging": {
            "level": "INFO",
            "log_file": str(root / "logs" / "system.log"),
            "run_dir": str(root / "logs" / "runs"),
        },
        "qa": {
            "workflow_mode": workflow_mode,
            "save_output": save_output,
            "outputs_dir": str(outputs_dir),
            "artifact_subdir": "qa_artifacts",
            "enable_peer_review": True,
            "models": copy.deepcopy(DEFAULT_QA_MODEL_ALIASES),
        },
        "llm": {},
    }


def build_ledger_system(
    root: Path,
    *,
    save_output: bool = True,
    document_warnings: Sequence[str] | None = None,
    review_warnings: Sequence[str] | None = None,
) -> QASystem:
    config = make_base_config(root, workflow_mode="ledger", save_output=save_output)
    task_spec = make_task_spec()
    grounding_pipeline = StaticGroundingPipeline(task_spec=task_spec)
    retrieval_pipeline = HeterogeneousRetrievalPipeline(
        query_planner=StaticQueryPlanner(),
        retriever=StaticRetriever(),
        document_acquirer=StaticDocumentAcquirer(warnings=document_warnings),
        evidence_extractor=StaticEvidenceExtractor(),
        claim_miner=StaticClaimMiner(),
        ledger_builder=EvidenceLedgerBuilder(),
        peer_review_pipeline=None,
    )
    synthesis_pipeline = VerifiedSynthesisPipeline()
    return QASystem(
        config=config,
        grounding_pipeline=grounding_pipeline,
        retrieval_pipeline=retrieval_pipeline,
        peer_review_pipeline=StaticPeerReviewPipeline(warnings=review_warnings),
        synthesis_pipeline=synthesis_pipeline,
    )


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def assert_paths_exist(paths: Iterable[str | Path]) -> None:
    for path in paths:
        assert Path(path).exists(), str(path)


def reset_logging_state() -> None:
    logging.shutdown()
    root = logging.getLogger()
    for handler in list(root.handlers):
        try:
            handler.flush()
        except Exception:
            pass
        try:
            handler.close()
        except Exception:
            pass
        root.removeHandler(handler)
    logger_mod._CONFIGURED = False
    logger_mod._RUN_DIR = None
    logger_mod._RUN_ID.set("")
    Logger._loggers.clear()


def flush_logging_handlers() -> None:
    root = logging.getLogger()
    for handler in list(root.handlers):
        try:
            handler.flush()
        except Exception:
            pass
