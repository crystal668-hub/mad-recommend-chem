from __future__ import annotations

from collections import defaultdict
import logging
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from qa.nodes.citation_reviewer import CitationReviewer
from qa.nodes.claim_revision import ClaimRevisionNode
from qa.nodes.contradiction_reviewer import ContradictionReviewer
from qa.nodes.methodology_reviewer import MethodologyReviewer
from qa.nodes.review_merge import ReviewMergeNode
from qa.peer_review_errors import PeerReviewExecutionError
from qa.review_utils import (
    build_evidence_lookup,
    condition_scope_is_ambiguous,
    conflict_pair_key,
    same_topic,
    traceable_evidence_items,
)
from qa.retrieval_state import (
    ClaimRecord,
    ClaimRevisionRecord,
    ConflictEdge,
    EvidenceLedger,
    ReviewFlag,
    ReviewSummary,
)
from qa.state import TaskSpec


logger = logging.getLogger("MAD.qa.peer_review")


SECOND_ROUND_RISK_FOCUS = {
    "causal": {"Missing_Condition", "Incomplete_Condition", "Overgeneralized", "Unsupported", "Weak_Evidence", "Metric_Mismatch"},
    "mechanism": {
        "Missing_Condition",
        "Incomplete_Condition",
        "Overgeneralized",
        "Unsupported",
        "Weak_Evidence",
        "Metric_Mismatch",
        "Mechanism_Speculative",
    },
}


class StructuredPeerReviewPipeline:
    def __init__(
        self,
        methodology_reviewer: Optional[MethodologyReviewer] = None,
        citation_reviewer: Optional[CitationReviewer] = None,
        contradiction_reviewer: Optional[ContradictionReviewer] = None,
        claim_revision_node: Optional[ClaimRevisionNode] = None,
        review_merge_node: Optional[ReviewMergeNode] = None,
        *,
        max_claims_for_llm_review: int = 40,
        max_second_round_claims: int = 15,
        disable_llm_review_when_abstract_only: bool = True,
        fallback_mode: str = "fail_fast_only",
        progress_log_every_claims: int = 10,
    ) -> None:
        self.methodology_reviewer = methodology_reviewer or MethodologyReviewer()
        self.citation_reviewer = citation_reviewer or CitationReviewer()
        self.contradiction_reviewer = contradiction_reviewer or ContradictionReviewer()
        self.claim_revision_node = claim_revision_node or ClaimRevisionNode()
        self.review_merge_node = review_merge_node or ReviewMergeNode()
        self.max_claims_for_llm_review = max(1, int(max_claims_for_llm_review))
        self.max_second_round_claims = max(1, int(max_second_round_claims))
        self.disable_llm_review_when_abstract_only = bool(disable_llm_review_when_abstract_only)
        self.fallback_mode = str(fallback_mode or "fail_fast_only").strip().lower() or "fail_fast_only"
        self.progress_log_every_claims = max(1, int(progress_log_every_claims))
        self.last_execution_warnings: List[str] = []
        self.last_run_stats: Dict[str, object] = {}

    def run(
        self,
        evidence_ledger: EvidenceLedger,
        *,
        task_spec: Optional[TaskSpec] = None,
    ) -> EvidenceLedger:
        ledger = evidence_ledger.model_copy(deep=True)
        self.last_execution_warnings = []
        self.last_run_stats = self._summarize_ledger(ledger)
        self._ensure_llm_review_ready(self.last_run_stats)
        use_llm_review = True
        logger.info(
            "qa_peer_review_start claims=%s evidence=%s abstract_only_ratio=%.2f fulltext_sections=%s use_llm=%s",
            self.last_run_stats["claim_count"],
            self.last_run_stats["evidence_count"],
            self.last_run_stats["abstract_only_ratio"],
            self.last_run_stats["fulltext_section_count"],
            use_llm_review,
        )
        round1_methodology_flags: List[ReviewFlag] = []
        round1_citation_flags: List[ReviewFlag] = []
        total_claims = len(ledger.claims)
        for index, claim in enumerate(ledger.claims, start=1):
            round1_methodology_flags.extend(
                self.methodology_reviewer.run(
                    claim,
                    ledger,
                    task_spec=task_spec,
                    review_round=1,
                    use_llm=use_llm_review,
                )
            )
            round1_citation_flags.extend(
                self.citation_reviewer.run(
                    claim,
                    ledger,
                    review_round=1,
                    use_llm=use_llm_review,
                )
            )
            self._log_claim_progress(
                stage="qa_peer_review_round1_progress",
                index=index,
                total=total_claims,
                extra=f"flags={len(round1_methodology_flags) + len(round1_citation_flags)}",
            )
        round1_contradiction_flags, round1_conflict_edges, _ = self.contradiction_reviewer.run(
            ledger.claims,
            ledger,
            review_round=1,
            use_llm=use_llm_review,
        )
        logger.info(
            "qa_peer_review_round1_complete methodology_flags=%s citation_flags=%s contradiction_flags=%s conflict_edges=%s",
            len(round1_methodology_flags),
            len(round1_citation_flags),
            len(round1_contradiction_flags),
            len(round1_conflict_edges),
        )

        round1_all_flags = [*round1_methodology_flags, *round1_citation_flags, *round1_contradiction_flags]
        round1_flags_by_claim = self._group_flags(round1_all_flags)
        round1_method_citation_by_claim = self._group_flags([*round1_methodology_flags, *round1_citation_flags])
        round1_edges_by_claim = self._group_edges(round1_conflict_edges)
        round1_edges_by_pair = {conflict_pair_key(edge.left_claim_id, edge.right_claim_id): edge for edge in round1_conflict_edges}

        revised_claims: List[ClaimRecord] = []
        revision_records_by_claim: Dict[str, List[ClaimRevisionRecord]] = defaultdict(list)
        substantive_change_by_claim: Dict[str, bool] = {}
        for index, claim in enumerate(ledger.claims, start=1):
            claim_flags = round1_flags_by_claim.get(claim.claim_id, [])
            conflict_edges = round1_edges_by_claim.get(claim.claim_id, [])
            if claim_flags or conflict_edges:
                revised_claim, revision_record, substantive_change = self.claim_revision_node.run(
                    claim,
                    ledger,
                    task_spec=task_spec,
                    review_flags=claim_flags,
                    conflict_edges=conflict_edges,
                    use_llm=use_llm_review,
                )
                revised_claims.append(revised_claim)
                revision_records_by_claim[claim.claim_id].append(revision_record)
                substantive_change_by_claim[claim.claim_id] = substantive_change
            else:
                revised_claims.append(claim)
                substantive_change_by_claim[claim.claim_id] = False
            self._log_claim_progress(
                stage="qa_peer_review_revision_progress",
                index=index,
                total=total_claims,
                extra=f"revised={sum(1 for changed in substantive_change_by_claim.values() if changed)}",
            )

        ledger.claims = revised_claims
        ledger.claim_index = {claim.claim_id: index for index, claim in enumerate(ledger.claims)}

        second_round_targets, second_round_focus = self._select_second_round_targets(
            claims=ledger.claims,
            ledger=ledger,
            task_spec=task_spec,
            round1_method_citation_by_claim=round1_method_citation_by_claim,
            round1_edges_by_claim=round1_edges_by_claim,
            substantive_change_by_claim=substantive_change_by_claim,
        )
        second_round_targets, second_round_focus = self._limit_second_round_targets(
            claims=ledger.claims,
            second_round_targets=second_round_targets,
            second_round_focus=second_round_focus,
            round1_method_citation_by_claim=round1_method_citation_by_claim,
            round1_edges_by_claim=round1_edges_by_claim,
            substantive_change_by_claim=substantive_change_by_claim,
        )

        round2_methodology_flags: List[ReviewFlag] = []
        round2_citation_flags: List[ReviewFlag] = []
        round2_total = len(second_round_targets)
        round2_index = 0
        for claim in ledger.claims:
            if claim.claim_id not in second_round_targets:
                continue
            round2_index += 1
            focus_types = sorted(second_round_focus.get(claim.claim_id, set()))
            round2_methodology_flags.extend(
                self.methodology_reviewer.run(
                    claim,
                    ledger,
                    task_spec=task_spec,
                    review_round=2,
                    focus_flag_types=focus_types,
                    use_llm=use_llm_review,
                )
            )
            round2_citation_flags.extend(
                self.citation_reviewer.run(
                    claim,
                    ledger,
                    review_round=2,
                    focus_flag_types=focus_types,
                    use_llm=use_llm_review,
                )
            )
            self._log_claim_progress(
                stage="qa_peer_review_round2_progress",
                index=round2_index,
                total=round2_total,
                extra=f"flags={len(round2_methodology_flags) + len(round2_citation_flags)}",
            )

        round2_contradiction_flags: List[ReviewFlag] = []
        round2_conflict_edges: List[ConflictEdge] = []
        round2_reviewed_pairs: Set[Tuple[str, str]] = set()
        if second_round_targets:
            round2_candidate_pairs = self._build_second_round_pair_keys(
                claims=ledger.claims,
                round1_edges=round1_conflict_edges,
                second_round_targets=second_round_targets,
            )
            round2_contradiction_flags, round2_conflict_edges, round2_reviewed_pairs = self.contradiction_reviewer.run(
                ledger.claims,
                ledger,
                review_round=2,
                focus_claim_ids=sorted(second_round_targets),
                candidate_pair_keys=round2_candidate_pairs if round2_candidate_pairs else None,
                use_llm=use_llm_review,
            )
        logger.info(
            "qa_peer_review_round2_complete targets=%s methodology_flags=%s citation_flags=%s contradiction_flags=%s conflict_edges=%s",
            len(second_round_targets),
            len(round2_methodology_flags),
            len(round2_citation_flags),
            len(round2_contradiction_flags),
            len(round2_conflict_edges),
        )

        round2_all_flags = [*round2_methodology_flags, *round2_citation_flags, *round2_contradiction_flags]
        round2_method_citation_by_claim = self._group_flags([*round2_methodology_flags, *round2_citation_flags])

        active_edges_by_pair = dict(round1_edges_by_pair)
        for pair_key in round2_reviewed_pairs:
            active_edges_by_pair.pop(pair_key, None)
        for edge in round2_conflict_edges:
            active_edges_by_pair[conflict_pair_key(edge.left_claim_id, edge.right_claim_id)] = edge
        active_edges_by_claim = self._group_edges(active_edges_by_pair.values())

        all_flags = [*round1_all_flags, *round2_all_flags]
        all_edges = [*round1_conflict_edges, *round2_conflict_edges]
        all_flags_by_claim = self._group_flags(all_flags)

        final_claims: List[ClaimRecord] = []
        review_summaries: List[ReviewSummary] = []
        for index, claim in enumerate(ledger.claims, start=1):
            claim_id = claim.claim_id
            active_flags = (
                round2_method_citation_by_claim.get(claim_id, [])
                if claim_id in second_round_targets
                else round1_method_citation_by_claim.get(claim_id, [])
            )
            active_edges = active_edges_by_claim.get(claim_id, [])
            final_status, merge_rationale = self.review_merge_node.run(
                claim,
                ledger,
                active_flags=active_flags,
                active_conflict_edges=active_edges,
                revision_records=revision_records_by_claim.get(claim_id, []),
                use_llm=use_llm_review,
            )
            final_claims.append(claim.model_copy(update={"status": final_status}))
            review_summaries.append(
                ReviewSummary(
                    claim_id=claim_id,
                    review_rounds=2 if claim_id in second_round_targets else 1,
                    review_flags=all_flags_by_claim.get(claim_id, []),
                    conflict_edge_ids=[
                        edge.conflict_id
                        for edge in all_edges
                        if claim_id in {edge.left_claim_id, edge.right_claim_id}
                    ],
                    revision_records=revision_records_by_claim.get(claim_id, []),
                    final_status=final_status,
                    merge_rationale=merge_rationale,
                )
            )
            self._log_claim_progress(
                stage="qa_peer_review_merge_progress",
                index=index,
                total=total_claims,
                extra=f"accepted={sum(1 for item in final_claims if item.status == 'accepted')}",
            )

        ledger.claims = final_claims
        ledger.claim_index = {claim.claim_id: index for index, claim in enumerate(ledger.claims)}
        ledger.review_flags = all_flags
        ledger.conflict_edges = all_edges
        ledger.review_summaries = review_summaries
        ledger.cluster_stats = {
            **dict(ledger.cluster_stats),
            "accepted_claim_count": sum(1 for claim in ledger.claims if claim.status == "accepted"),
            "contested_claim_count": sum(1 for claim in ledger.claims if claim.status == "contested"),
            "rejected_claim_count": sum(1 for claim in ledger.claims if claim.status == "rejected"),
            "second_round_claim_count": len(second_round_targets),
        }
        ledger.ledger_notes = self._merge_notes(
            ledger.ledger_notes,
            [
                "Module 4 applies structured peer review with validator-gated LLM adjudication.",
                "Claims may receive at most one conservative revision and at most one targeted second review.",
            ],
        )
        logger.info(
            "qa_peer_review_complete accepted=%s contested=%s rejected=%s warnings=%s",
            sum(1 for claim in ledger.claims if claim.status == "accepted"),
            sum(1 for claim in ledger.claims if claim.status == "contested"),
            sum(1 for claim in ledger.claims if claim.status == "rejected"),
            len(self.last_execution_warnings),
        )
        return ledger

    __call__ = run

    def _summarize_ledger(self, ledger: EvidenceLedger) -> Dict[str, object]:
        evidence_items = list(ledger.evidence_items or [])
        claim_count = len(ledger.claims)
        evidence_count = len(evidence_items)
        claim_like_items = [
            item for item in evidence_items if item.role in {"observation", "limitation", "mechanism"}
        ]
        abstract_like_items = [
            item for item in claim_like_items if str(item.source_layer or "").strip().lower() == "abstract"
        ]
        abstract_only_ratio = 1.0 if claim_like_items and len(abstract_like_items) == len(claim_like_items) else 0.0
        if claim_like_items and len(abstract_like_items) != len(claim_like_items):
            abstract_only_ratio = len(abstract_like_items) / float(len(claim_like_items))
        fulltext_section_count = len(
            {
                (item.paper_id, item.section_id)
                for item in evidence_items
                if str(item.source_layer or "").strip().lower() == "fulltext"
            }
        )
        return {
            "claim_count": claim_count,
            "evidence_count": evidence_count,
            "abstract_only_ratio": round(abstract_only_ratio, 3),
            "fulltext_section_count": fulltext_section_count,
        }

    def _ensure_llm_review_ready(self, run_stats: Dict[str, object]) -> None:
        if self.fallback_mode != "fail_fast_only":
            raise ValueError(f"Unsupported peer review fallback mode: {self.fallback_mode!r}.")
        disable_reasons = self._llm_disable_reasons(run_stats)
        if disable_reasons:
            raise PeerReviewExecutionError(
                stage="peer_review_policy",
                message="Peer review cannot proceed under fail-fast policy.",
                details={"disable_reasons": disable_reasons},
                run_stats=dict(run_stats),
            )
        missing_nodes = self._missing_llm_nodes()
        if missing_nodes:
            raise PeerReviewExecutionError(
                stage="peer_review_startup",
                message="Peer review cannot start because one or more reviewer LLMs are unavailable.",
                details={"missing_nodes": missing_nodes},
                run_stats=dict(run_stats),
            )

    def _missing_llm_nodes(self) -> List[str]:
        nodes = [
            self.methodology_reviewer,
            self.citation_reviewer,
            self.contradiction_reviewer,
            self.claim_revision_node,
            self.review_merge_node,
        ]
        return [
            getattr(node, "reviewer_type", node.__class__.__name__)
            for node in nodes
            if getattr(node, "llm", None) is None
        ]

    def _llm_disable_reasons(self, run_stats: Dict[str, object]) -> List[str]:
        reasons: List[str] = []
        claim_count = int(run_stats.get("claim_count") or 0)
        if claim_count > self.max_claims_for_llm_review:
            reasons.append(
                f"claim volume exceeded budget ({claim_count} > {self.max_claims_for_llm_review})"
            )
        if (
            self.disable_llm_review_when_abstract_only
            and float(run_stats.get("abstract_only_ratio") or 0.0) >= 0.95
            and int(run_stats.get("fulltext_section_count") or 0) == 0
        ):
            reasons.append("evidence was abstract-only")
        return reasons

    def _limit_second_round_targets(
        self,
        *,
        claims: Sequence[ClaimRecord],
        second_round_targets: Set[str],
        second_round_focus: Dict[str, Set[str]],
        round1_method_citation_by_claim: Dict[str, List[ReviewFlag]],
        round1_edges_by_claim: Dict[str, List[ConflictEdge]],
        substantive_change_by_claim: Dict[str, bool],
    ) -> Tuple[Set[str], Dict[str, Set[str]]]:
        if len(second_round_targets) <= self.max_second_round_claims:
            return second_round_targets, second_round_focus

        claim_lookup = {claim.claim_id: claim for claim in claims}
        ranked_targets = sorted(
            second_round_targets,
            key=lambda claim_id: (
                -self._second_round_priority_score(
                    claim=claim_lookup[claim_id],
                    claim_flags=round1_method_citation_by_claim.get(claim_id, []),
                    claim_edges=round1_edges_by_claim.get(claim_id, []),
                    focus_types=second_round_focus.get(claim_id, set()),
                    substantive_change=substantive_change_by_claim.get(claim_id, False),
                ),
                claim_id,
            ),
        )
        kept_targets = set(ranked_targets[: self.max_second_round_claims])
        self.last_execution_warnings.append(
            "Peer review limited second-round review to "
            f"{self.max_second_round_claims} highest-risk claims (from {len(second_round_targets)})."
        )
        logger.info(
            "qa_peer_review_round2_capped kept=%s original=%s",
            len(kept_targets),
            len(second_round_targets),
        )
        return kept_targets, {claim_id: second_round_focus[claim_id] for claim_id in kept_targets}

    def _second_round_priority_score(
        self,
        *,
        claim: ClaimRecord,
        claim_flags: Sequence[ReviewFlag],
        claim_edges: Sequence[ConflictEdge],
        focus_types: Set[str],
        substantive_change: bool,
    ) -> int:
        warning_count = sum(1 for flag in claim_flags if flag.severity == "warning")
        critical_count = sum(1 for flag in claim_flags if flag.severity == "critical")
        true_conflict_count = sum(1 for edge in claim_edges if edge.conflict_type == "true_conflict")
        divergence_count = sum(1 for edge in claim_edges if edge.conflict_type == "condition_divergence")
        score = warning_count * 10
        score += critical_count * 40
        score += true_conflict_count * 50
        score += divergence_count * 15
        score += len(focus_types)
        if substantive_change:
            score += 8
        if claim.claim_type in {"causal", "mechanism", "comparison"}:
            score += 12
        return score

    def _log_claim_progress(self, *, stage: str, index: int, total: int, extra: str = "") -> None:
        if total <= 0:
            return
        if index != total and index % self.progress_log_every_claims != 0:
            return
        suffix = f" {extra}" if extra else ""
        logger.info("%s completed=%s/%s%s", stage, index, total, suffix)

    def _select_second_round_targets(
        self,
        *,
        claims: Sequence[ClaimRecord],
        ledger: EvidenceLedger,
        task_spec: Optional[TaskSpec],
        round1_method_citation_by_claim: Dict[str, List[ReviewFlag]],
        round1_edges_by_claim: Dict[str, List[ConflictEdge]],
        substantive_change_by_claim: Dict[str, bool],
    ) -> Tuple[Set[str], Dict[str, Set[str]]]:
        evidence_lookup = build_evidence_lookup(ledger)
        second_round_targets: Set[str] = set()
        focus_by_claim: Dict[str, Set[str]] = defaultdict(set)
        required_axes = list(task_spec.required_condition_axes or []) if task_spec is not None else []

        for claim in claims:
            claim_flags = round1_method_citation_by_claim.get(claim.claim_id, [])
            claim_edges = round1_edges_by_claim.get(claim.claim_id, [])
            support_items = traceable_evidence_items(claim.supporting_evidence_ids, evidence_lookup=evidence_lookup)

            if self._meets_accepted_threshold(claim_flags=claim_flags, claim_edges=claim_edges, claim=claim, support_items=support_items, required_axes=required_axes):
                continue

            has_warning = any(flag.severity == "warning" for flag in claim_flags)
            has_true_conflict = any(edge.conflict_type == "true_conflict" for edge in claim_edges)
            has_condition_divergence = any(edge.conflict_type == "condition_divergence" for edge in claim_edges)
            condition_still_ambiguous = condition_scope_is_ambiguous(
                claim,
                required_axes=required_axes,
                supporting_items=support_items,
            )
            substantive_change = substantive_change_by_claim.get(claim.claim_id, False)

            should_review_again = (
                claim.claim_type in {"causal", "mechanism"}
                or has_warning
                or has_true_conflict
                or (has_condition_divergence and condition_still_ambiguous)
                or substantive_change
            )
            if not should_review_again:
                continue

            second_round_targets.add(claim.claim_id)
            focus_by_claim[claim.claim_id].update(flag.flag_type for flag in claim_flags)
            focus_by_claim[claim.claim_id].update(SECOND_ROUND_RISK_FOCUS.get(claim.claim_type, set()))
            if has_true_conflict or has_condition_divergence:
                focus_by_claim[claim.claim_id].update({"Missing_Condition", "Incomplete_Condition", "Overgeneralized"})
            if substantive_change and not focus_by_claim[claim.claim_id]:
                focus_by_claim[claim.claim_id].update({"Missing_Condition", "Overgeneralized", "Unsupported", "Weak_Evidence"})

        return second_round_targets, focus_by_claim

    def _meets_accepted_threshold(
        self,
        *,
        claim_flags: Sequence[ReviewFlag],
        claim_edges: Sequence[ConflictEdge],
        claim: ClaimRecord,
        support_items: Sequence[object],
        required_axes: Sequence[str],
    ) -> bool:
        if any(flag.severity in {"warning", "critical"} for flag in claim_flags):
            return False
        if any(edge.conflict_type == "true_conflict" for edge in claim_edges):
            return False
        if not support_items:
            return False
        if condition_scope_is_ambiguous(claim, required_axes=required_axes, supporting_items=support_items):
            return False
        return True

    def _build_second_round_pair_keys(
        self,
        *,
        claims: Sequence[ClaimRecord],
        round1_edges: Sequence[ConflictEdge],
        second_round_targets: Set[str],
    ) -> Set[Tuple[str, str]]:
        if not second_round_targets:
            return set()
        pair_keys = {
            conflict_pair_key(edge.left_claim_id, edge.right_claim_id)
            for edge in round1_edges
            if edge.left_claim_id in second_round_targets or edge.right_claim_id in second_round_targets
        }
        claim_list = list(claims)
        for index, left_claim in enumerate(claim_list):
            for right_claim in claim_list[index + 1 :]:
                if left_claim.claim_id not in second_round_targets and right_claim.claim_id not in second_round_targets:
                    continue
                if same_topic(left_claim, right_claim):
                    pair_keys.add(conflict_pair_key(left_claim.claim_id, right_claim.claim_id))
        return pair_keys

    def _group_flags(self, flags: Iterable[ReviewFlag]) -> Dict[str, List[ReviewFlag]]:
        grouped: Dict[str, List[ReviewFlag]] = defaultdict(list)
        for flag in flags:
            grouped[flag.claim_id].append(flag)
        return grouped

    def _group_edges(self, edges: Iterable[ConflictEdge]) -> Dict[str, List[ConflictEdge]]:
        grouped: Dict[str, List[ConflictEdge]] = defaultdict(list)
        for edge in edges:
            grouped[edge.left_claim_id].append(edge)
            grouped[edge.right_claim_id].append(edge)
        return grouped

    def _merge_notes(self, existing_notes: Sequence[str], extra_notes: Sequence[str]) -> List[str]:
        merged = list(existing_notes or [])
        for note in extra_notes:
            if note not in merged:
                merged.append(note)
        return merged
