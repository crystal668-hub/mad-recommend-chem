from __future__ import annotations

from typing import Optional

from qa.artifacts import QAArtifactStore
from qa.evidence import ClaimMiner, EvidenceExtractor, EvidenceLedgerBuilder
from qa.handoff import EvidenceExtractorHandoff
from qa.nodes.document_acquirer import DocumentAcquirerNode
from qa.nodes.query_planner import QueryPlannerNode
from qa.nodes.retriever import RetrieverNode
from qa.review_pipeline import StructuredPeerReviewPipeline
from qa.retrieval_state import RetrievalDiagnosticRecord, RetrievalState
from qa.state import EntityPack, GroundingState, TaskSpec


class HeterogeneousRetrievalPipeline:
    def __init__(
        self,
        query_planner: Optional[QueryPlannerNode] = None,
        retriever: Optional[RetrieverNode] = None,
        document_acquirer: Optional[DocumentAcquirerNode] = None,
        handoff: Optional[EvidenceExtractorHandoff] = None,
        evidence_extractor: Optional[EvidenceExtractor] = None,
        claim_miner: Optional[ClaimMiner] = None,
        ledger_builder: Optional[EvidenceLedgerBuilder] = None,
        peer_review_pipeline: Optional[StructuredPeerReviewPipeline] = None,
    ) -> None:
        self.query_planner = query_planner or QueryPlannerNode()
        self.retriever = retriever or RetrieverNode()
        self.document_acquirer = document_acquirer or DocumentAcquirerNode()
        self.handoff = handoff or EvidenceExtractorHandoff()
        self.evidence_extractor = evidence_extractor or EvidenceExtractor(handoff=self.handoff)
        self.claim_miner = claim_miner or ClaimMiner()
        self.ledger_builder = ledger_builder or EvidenceLedgerBuilder()
        self.peer_review_pipeline = peer_review_pipeline

    def run(
        self,
        question: str,
        task_spec: TaskSpec,
        entity_pack: EntityPack,
        *,
        context: Optional[str] = None,
        artifact_dir: Optional[str] = None,
    ) -> RetrievalState:
        store = QAArtifactStore(base_dir=artifact_dir)
        query_plans = self.query_planner.run(task_spec=task_spec, entity_pack=entity_pack)
        paper_candidates = self.retriever.run(
            task_spec=task_spec,
            entity_pack=entity_pack,
            query_plans=query_plans,
            artifact_store=store,
        )
        provider_health: dict[str, dict] = dict(getattr(self.retriever, "last_provider_health", {}) or {})
        retrieval_diagnostics: list[RetrievalDiagnosticRecord] = list(getattr(self.retriever, "last_diagnostics", []) or [])
        paper_records, section_indices = self.document_acquirer.run(
            candidates=paper_candidates,
            artifact_store=store,
        )
        provider_health.update(dict(getattr(self.document_acquirer, "last_provider_health", {}) or {}))
        retrieval_diagnostics.extend(list(getattr(self.document_acquirer, "last_diagnostics", []) or []))
        evidence_items = []
        for paper_record, section_index in zip(paper_records, section_indices):
            evidence_items.extend(
                self.evidence_extractor.run(
                    task_spec=task_spec,
                    entity_pack=entity_pack,
                    paper_record=paper_record,
                    section_index=section_index,
                )
            )
        claims = self.claim_miner.run(evidence_items=evidence_items, task_spec=task_spec)
        evidence_ledger = self.ledger_builder.run(claims=claims, evidence_items=evidence_items)
        if self.peer_review_pipeline is not None:
            evidence_ledger = self.peer_review_pipeline.run(evidence_ledger, task_spec=task_spec)

        store.write_json("query_plans.json", [item.model_dump(exclude_none=True) for item in query_plans])
        store.write_json("paper_candidates.json", [item.model_dump(exclude_none=True) for item in paper_candidates])
        store.write_json("paper_records.json", [item.model_dump(exclude_none=True) for item in paper_records])
        store.write_json("section_indices.json", [item.model_dump(exclude_none=True) for item in section_indices])
        store.write_json("retrieval_diagnostics.json", [item.model_dump(exclude_none=True) for item in retrieval_diagnostics])
        store.write_json("provider_health.json", provider_health)
        store.write_json("evidence_items.json", [item.model_dump(exclude_none=True) for item in evidence_items])
        store.write_json("evidence_ledger.json", evidence_ledger.model_dump(exclude_none=True))

        return RetrievalState(
            question=question,
            context=context,
            task_spec=task_spec,
            entity_pack=entity_pack,
            query_plans=query_plans,
            paper_candidates=paper_candidates,
            paper_records=paper_records,
            section_indices=section_indices,
            retrieval_diagnostics=retrieval_diagnostics,
            evidence_items=evidence_items,
            evidence_ledger=evidence_ledger,
            artifact_dir=str(store.root_dir),
        )

    def run_from_grounding(
        self,
        grounding_state: GroundingState,
        *,
        artifact_dir: Optional[str] = None,
    ) -> RetrievalState:
        return self.run(
            question=grounding_state.question,
            context=grounding_state.context,
            task_spec=grounding_state.task_spec,
            entity_pack=grounding_state.entity_pack,
            artifact_dir=artifact_dir,
        )

    __call__ = run
